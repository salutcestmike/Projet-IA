import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, 
    Conv2DTranspose, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, str(Path("src").resolve()))
from utils import preprocessing

# Modele U-Net 
def conv_block(inputs, n_filters, dropout_prob=0.0, max_pooling=True):
    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="he_normal")(inputs)
    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="he_normal")(conv)
    if dropout_prob > 0.0:
        conv = Dropout(dropout_prob)(conv)
    next_layer = MaxPooling2D((2, 2))(conv) if max_pooling else conv
    skip_connection = conv
    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters):
    up = Conv2DTranspose(n_filters, 3, strides=2, padding="same")(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="he_normal")(merge)
    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="he_normal")(conv)
    return conv


def build_unet(input_shape=(512, 512, 1), n_filters=8, num_classes=2):
    inputs = Input(input_shape)

    c1, s1 = conv_block(inputs, n_filters)
    c2, s2 = conv_block(c1, n_filters * 2)
    c3, s3 = conv_block(c2, n_filters * 4)
    c4, s4 = conv_block(c3, n_filters * 8, dropout_prob=0.3)
    c5, _  = conv_block(c4, n_filters * 16, dropout_prob=0.3, max_pooling=False)

    u6 = upsampling_block(c5, s4, n_filters * 8)
    u7 = upsampling_block(u6, s3, n_filters * 4)
    u8 = upsampling_block(u7, s2, n_filters * 2)
    u9 = upsampling_block(u8, s1, n_filters)

    outputs = Conv2D(num_classes, 1, activation="softmax")(u9)
    return Model(inputs=[inputs], outputs=[outputs], name="unet_imane_v3")

# Fonctions de perte et métriques

def dice_coef(y_true, y_pred, smooth=1.0):
    """Calcul continu du Dice sur le canal mitochondrie (canal 1)."""
    y_true_mito = tf.cast(y_true[..., 1], tf.float32)
    y_pred_mito = tf.cast(y_pred[..., 1], tf.float32)
    intersection = tf.reduce_sum(y_true_mito * y_pred_mito, axis=[1, 2])
    union = tf.reduce_sum(y_true_mito, axis=[1, 2]) + tf.reduce_sum(y_pred_mito, axis=[1, 2])
    return tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


def ce_dice_loss(y_true, y_pred):
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    ce = tf.reduce_mean(ce)
    dice = dice_coef(y_true, y_pred)
    return ce + (1.0 - dice)


# Augmentation de données

def augment(img, mask):
    """Flips et rotations à 90 degrés synchronisés."""
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    mask = tf.image.rot90(mask, k)
    return img, mask


def load_and_preprocess(img_path, mask_path, img_size=(512, 512), training=False):
    def _py_load(i_path, m_path):
        img = preprocessing.load_image(Path(i_path.numpy().decode("utf-8")))
        mask = preprocessing.load_image(Path(m_path.numpy().decode("utf-8")))

        img_norm = preprocessing.normalize_image(img).astype(np.float32)
        mask_bin = (np.asarray(mask) > 0).astype(np.uint8)
        return img_norm[..., None], mask_bin[..., None]

    img, mask = tf.py_function(_py_load, [img_path, mask_path], [tf.float32, tf.uint8])
    img.set_shape([None, None, 1])
    mask.set_shape([None, None, 1])

    img = tf.image.resize(img, img_size, antialias=True)
    mask = tf.image.resize(mask, img_size, method="nearest")

    img.set_shape([img_size[0], img_size[1], 1])
    mask.set_shape([img_size[0], img_size[1], 1])

    if training:
        img, mask = augment(img, mask)

    # Masque en one-hot : (H, W, 2)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.one_hot(tf.cast(mask, tf.int32), depth=2)
    return img, mask


def get_dataset(img_paths, mask_paths, batch_size=4, training=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if training:
        ds = ds.shuffle(buffer_size=len(img_paths), seed=seed)
    ds = ds.map(lambda i, m: load_and_preprocess(i, m, training=training), 
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Entraînement et évaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement U-Net v3 : segmentation des mitochondries.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/imane_run"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path("data/dm3")
    target_dir = Path("data/mod_tiff")

    imgs = {f.stem: str(f) for f in input_dir.glob("*.dm3")}
    masks = {f.stem: str(f) for f in target_dir.glob("*.tif")}
    stems = sorted(list(imgs.keys() & masks.keys()))

    # Split train / val / test (70 / 15 / 15)
    idx = np.arange(len(stems))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, random_state=42, shuffle=True)

    print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    np.savez(args.output_dir / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)

    train_imgs = [imgs[stems[i]] for i in train_idx]
    train_masks = [masks[stems[i]] for i in train_idx]

    val_imgs = [imgs[stems[i]] for i in val_idx]
    val_masks = [masks[stems[i]] for i in val_idx]

    test_imgs = [imgs[stems[i]] for i in test_idx]
    test_masks = [masks[stems[i]] for i in test_idx]

    train_ds = get_dataset(train_imgs, train_masks, batch_size=args.batch_size, training=True)
    val_ds   = get_dataset(val_imgs, val_masks, batch_size=args.batch_size, training=False)
    test_ds  = get_dataset(test_imgs, test_masks, batch_size=args.batch_size, training=False)

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        n_batches = tf.data.experimental.cardinality(ds).numpy()
        print(f"{name}: {n_batches} batchs")

    model = build_unet()
    

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer, 
        loss=ce_dice_loss, 
        metrics=["accuracy", dice_coef]
    )

    best_model_path = args.output_dir / "best_model.keras"
    callbacks = [
        ModelCheckpoint(
            str(best_model_path),
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_dice_coef",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    test_loss, test_acc, test_dice = model.evaluate(test_ds)
