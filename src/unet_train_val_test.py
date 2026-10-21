import argparse
import numpy as np
import tensorflow as tf
import os
import imageio
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Dropout, Conv2DTranspose, concatenate
)
from tensorflow.keras import Model

from pathlib import Path

from sklearn.model_selection import train_test_split

from utils.preprocessing import load_image, convert_to_8bit_v2, down_sample

import h5py

def load_weights_keras3_in_keras2(model, filepath):
    """Extrait tous les datasets d'un .weights.h5 (Keras 3) et les injecte dans Keras 2."""
    weights_dict = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            weights_dict[name] = obj[()]

    with h5py.File(filepath, "r") as f:
        f.visititems(visitor)

    print(f"Datasets trouvés dans le fichier H5 : {len(weights_dict)}")

    # Associer les poids aux couches du modèle par correspondance de nom
    assigned = 0
    for layer in model.layers:
        # On cherche tous les datasets qui contiennent le nom exact de la couche
        layer_weights = [
            val for key, val in weights_dict.items()
            if f"/{layer.name}/" in f"/{key}/" or key.startswith(f"{layer.name}/")
        ]
        
        if len(layer_weights) > 0 and len(layer.weights) > 0:
            # Trier pour respecter l'ordre usuel (kernel d'abord, puis bias)
            # En général le kernel a un rang plus élevé que le bias
            layer_weights.sort(key=lambda w: -w.ndim)
            try:
                layer.set_weights(layer_weights)
                assigned += 1
            except Exception as e:
                print(f"Impossible d'assigner {layer.name} : {e}")

    print(f"Poids injectés avec succès dans {assigned}/{len(model.layers)} couches.")
    if assigned == 0:
        raise RuntimeError("Aucune couche n'a pu recevoir ses poids !")

def get_args():
    p = argparse.ArgumentParser(description="Entraînement U-Net : segmentation des mitochondries.")
    p.add_argument("--image_dir", type=Path, default=Path("data/img_test"))
    p.add_argument("--mask_dir", type=Path, default=Path("data/mask_test"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/run"),
                   help="Dossier de sortie (poids, logs CSV, masques prédits).")
    p.add_argument("--weights_dir", type=Path, default=Path("outputs/run"),
                    help="Dossier de sortie des poids.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--repeat", type=int, default=40, help="Patchs par image et par epoch.")
    p.add_argument("--n_filters", type=int, default=8)
    p.add_argument("--mito_weight", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--validation_freq", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize(image):
    image = tf.expand_dims(image, -1)                    # (H, W) -> (H, W, 1)
    mean = tf.reduce_mean(image)
    std = tf.math.reduce_std(image)
    image = (image - mean) / (std + 1e-6)
    image.set_shape([None, None, 1])
    return image

def process_path(image_path, mask_path):
    def load_and_preprocess(image_path, mask_path):
        image_path = Path(image_path.numpy().decode("utf-8"))
        mask_path = Path(mask_path.numpy().decode("utf-8"))

        img = load_image(image_path)
        if image_path.suffix.lower() == ".dm3":
            img = convert_to_8bit_v2(img)

        mask = (load_image(mask_path) > 0).astype(np.float32)
        return img.astype(np.float32), mask

    img, mask = tf.py_function(
        load_and_preprocess, [image_path, mask_path], [tf.float32, tf.float32]
    )
    return img, mask
    
def preprocess(image, mask):
    mask.set_shape([None, None])
    return normalize(image), mask

def conv_block(inputs, n_filters, dropout_prob=0, max_pooling=True):
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    next_layer = MaxPooling2D((2, 2))(conv) if max_pooling else conv
    skip_connection = conv
    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters):
    up = Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)

    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(conv)
    return conv

def unet_model(input_size=(2048,2048,1), n_filters=4, n_classes=2):
    inputs = Input(input_size)

    c1 = conv_block(inputs, n_filters)
    c2 = conv_block(c1[0], n_filters*2)
    c3 = conv_block(c2[0], n_filters*4)
    c4 = conv_block(c3[0], n_filters*8, dropout_prob=0.3)
    c5 = conv_block(c4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    u6 = upsampling_block(c5[0], c4[1], n_filters*8)
    u7 = upsampling_block(u6, c3[1], n_filters*4)
    u8 = upsampling_block(u7, c2[1], n_filters*2)
    u9 = upsampling_block(u8, c1[1], n_filters)

    outputs = Conv2D(n_classes, 1, activation='softmax')(u9)
    return Model(inputs, outputs)

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    y_true : (B, H, W) ou (B, H, W, 1), indices de classes (0, 1, ...)
    y_pred : (B, H, W, C), probabilités (sortie softmax)
    """
    n_classes = y_pred.shape[-1]

    # Masque sparse -> one-hot : (B, H, W, C)
    y_true = tf.cast(y_true, tf.int32)
    if y_true.shape.rank == 4:
        y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.one_hot(y_true, depth=n_classes, dtype=y_pred.dtype)

    # Dice calculé par classe, en sommant sur le batch et les pixels
    axes = (0, 1, 2)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)

    dice_per_class = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class[1:])


def ce_dice_loss(y_true, y_pred, ce_weight=0.5, dice_weight=0.5, mito_weight=2.0):
    y_true_ce = tf.cast(y_true, tf.int32)
    if y_true_ce.shape.rank == 4:
        y_true_ce = tf.squeeze(y_true_ce, axis=-1)

    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_ce, y_pred, from_logits=False)
    w = 1.0 + (mito_weight - 1.0) * tf.cast(y_true_ce, tf.float32)    # 1 fond, 10 mito
    ce = tf.reduce_sum(ce * w) / tf.reduce_sum(w)

    return ce_weight * ce + dice_weight * dice_loss(y_true, y_pred)

def dice_mito(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)[:-1]), tf.float32)
    pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    inter = tf.reduce_sum(y_true * pred)
    return (2 * inter + 1e-6) / (tf.reduce_sum(y_true) + tf.reduce_sum(pred) + 1e-6)

PATCH = 256

def random_crop_aug(image, mask):
    """image : (H, W, 1), mask : (H, W) -> patch (256, 256, 1), (256, 256)"""
    stacked = tf.concat([image, tf.expand_dims(mask, -1)], axis=-1)   # (H, W, 2)

    init = tf.image.random_crop(stacked, size=(PATCH, PATCH, 2))

    # jusqu'à 10 essais pour tomber sur un patch avec au moins 1 % de mito
    def cond(i, c):
        return tf.logical_and(i < 10, tf.reduce_mean(c[..., 1]) <= 0.01)

    def body(i, c):
        return i + 1, tf.image.random_crop(stacked, size=(PATCH, PATCH, 2))

    _, crop = tf.while_loop(cond, body, [0, init])

    # augmentation : rotation de 90° et flip
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    crop = tf.image.rot90(crop, k)
    crop = tf.image.random_flip_left_right(crop)
    return crop[..., :1], crop[..., 1]


def make_dataset(image_list, mask_list, batch_size=8, train=False, repeat=40, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if train:
        ds = ds.cache()                    # images pleines en RAM (16 Mo + 16 Mo par image)
        ds = ds.repeat(repeat)             # `repeat` patchs par image et par epoch
        ds = ds.shuffle(100, seed=seed)
        ds = ds.map(random_crop_aug, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # validation / test : images complètes, une à la fois
    return ds.batch(1).prefetch(tf.data.AUTOTUNE)

def process_image_only(image_path):
    def _load(image_path):
        image_path = Path(image_path.numpy().decode("utf-8"))
        img = load_image(image_path)
        if image_path.suffix.lower() == ".dm3":
            img = convert_to_8bit_v2(img)
        return img.astype(np.float32)

    img = tf.py_function(_load, [image_path], tf.float32)
    return normalize(img)

def predict_and_save(model, image_list, output_dir, mask_list=None, threshold=0.5, save_proba=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(image_list):
        path_obj = Path(img_path)
        img = load_image(path_obj)
        if path_obj.suffix.lower() == ".dm3":
            img = convert_to_8bit_v2(img)
            
        img = img.astype(np.float32)
        mean = np.mean(img)
        std = np.std(img)
        img_norm = (img - mean) / (std + 1e-6)
        
        # Format d'entrée : (1, H, W, 1)
        inp = img_norm[np.newaxis, ..., np.newaxis]
        
        # Sortie brute : (1, H, W, 2)
        raw_pred = model(inp, training=False)
        proba = raw_pred.numpy()[0, ..., 1]  # canal 1 (mito)
        stem = path_obj.stem

        # Binarisation et sauvegarde
        mask = ((proba > threshold) * 255).astype(np.uint8)
        imageio.imwrite(output_dir / f"{stem}_pred.tif", mask)
        
        if save_proba:
            imageio.imwrite(output_dir / f"{stem}_proba.tif", (proba * 255).astype(np.uint8))

        # Calcul du Dice avec ta fonction
        dice_str = ""
        if mask_list is not None and idx < len(mask_list):
            true_mask = (load_image(Path(mask_list[idx])) > 0).astype(np.float32)
            # true_mask -> shape (1, H, W) pour matcher raw_pred (1, H, W, 2)
            true_tensor = tf.convert_to_tensor(true_mask[np.newaxis, ...], dtype=tf.float32)
            dice_val = float(dice_mito(true_tensor, raw_pred))
            dice_str = f" | Dice : {dice_val:.4f}"

        print(f"{stem} | proba max : {proba.max():.3f} | mito prédit : {(proba > threshold).mean()*100:.2f} %{dice_str}")

if __name__ == "__main__":
    args = get_args()

    image_path = "data/dm3_red"
    mask_path  = "data/mask_ok"

    image_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.tif')])
    mask_list  = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('.tif')])
    assert len(image_list) == len(mask_list) > 0

    trainval_images, test_images, trainval_masks, test_masks = train_test_split(
        image_list, mask_list, test_size=0.15, random_state=42, shuffle=True
    )

    train_images, val_images, train_masks, val_masks = train_test_split(
            trainval_images, trainval_masks, test_size=0.15, random_state=42, shuffle=True
        )
    
    print(f"Train : {len(train_images)} | Validation : {len(val_images)} | Test : {len(test_images)}")

    train_ds = make_dataset(train_images, train_masks, batch_size=8, train=True, repeat=40)
    val_ds  = make_dataset(val_images, val_masks, batch_size=1, train=False)
    test_ds  = make_dataset(test_images,  test_masks)

    # x, y = next(iter(train_ds))
    # print("image :", x.shape, "| mask :", y.shape)       # (8, 256, 256, 1) | (8, 256, 256)
    # print("fraction mito dans le batch :", float(tf.reduce_mean(y)))

    unet = unet_model(input_size=(None, None, 1), n_filters=8)
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=ce_dice_loss,
        metrics=[dice_mito]
    )

    Path(args.weights_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.weights_dir/"unet_mito_best_ok.weights.h5", monitor="val_dice_mito", mode="max",
            save_best_only=True, save_weights_only=True
        ),

        tf.keras.callbacks.EarlyStopping(monitor="val_dice_mito", mode="max", patience=5, restore_best_weights=True),
    ]
    unet.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # unet.load_weights(args.weights_dir/"unet_mito_best_debug_fulldata.weights.h5")         # meilleur modèle selon le Dice mito en val
    _ = unet(tf.zeros((1, 256, 256, 1), dtype=tf.float32))

    weights_path = args.weights_dir / "unet_mito_best_ok.weights.h5"
    try:
        unet.load_weights(weights_path)
    except ValueError:
        print("Format Keras 3 détecté sous Keras 2. Chargement manuel...")
        load_weights_keras3_in_keras2(unet, weights_path)
    
    predict_and_save(
            model=unet,
            image_list=test_images,
            output_dir=args.output_dir / "pred_masks_ok",
            mask_list=test_masks,
            threshold=0.5
        )

    predict_and_save(
                model=unet,
                image_list=trainval_images,
                output_dir=args.output_dir / "pred_masks_train_ok",
                mask_list=trainval_masks,
                threshold=0.5
            )

    
    

    # print("\n--- Évaluation Test ---")
    # test_res = unet.evaluate(test_ds)
    # print(f"Test Loss : {test_res[0]:.4f} | Test Dice Mito : {test_res[1]:.4f}")
    

    