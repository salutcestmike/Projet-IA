import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Dropout, Conv2DTranspose, concatenate
)
from tensorflow.keras import Model
import tifffile

from utils import preprocessing
import utils.stats as stats  # doit contenir DiceCoefficient (name="dice")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paramètres
# --------------------------------------------------------------------------- #
input_dir = "data/dm3"
target_dir = "data/mod_tiff"
output_dir = Path("outputs/xception_run6")
output_dir.mkdir(parents=True, exist_ok=True)

img_size = (512, 512)
num_classes = 2
batch_size = 4          # 32 est beaucoup trop pour du 1024² -> OOM
epochs = 100
SEED = 42


# --------------------------------------------------------------------------- #
# Liste et appariement des fichiers
# --------------------------------------------------------------------------- #
def list_files(folder):
    """{stem: chemin} pour les .tif/.tiff (insensible à la casse, fichiers cachés ignorés)."""
    return {
        Path(f).stem: os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".tif", ".tiff", ".dm3")) and not f.startswith(".")
    }


def pair_files(input_dir, target_dir):
    imgs, masks = list_files(input_dir), list_files(target_dir)
    stems = sorted(imgs.keys() & masks.keys())
    print(f"images={len(imgs)}  masques={len(masks)}  paires={len(stems)}")
    if len(stems) < max(len(imgs), len(masks)):
        print("Attention : certains fichiers n'ont pas de correspondant (ignorés).")
    return (np.asarray([imgs[s] for s in stems]),
            np.asarray([masks[s] for s in stems]))


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def _load_pair_numpy(image_path, mask_path):
    """Lecture NumPy (appelée via py_function)."""
    image_path = Path(image_path.numpy().decode("utf-8"))
    mask_path = Path(mask_path.numpy().decode("utf-8"))

    img = preprocessing.load_image(image_path)
    img = preprocessing.normalize_image(img)
    img = img.astype(np.float32)[..., None]                      # (H, W, 1)

    mask = preprocessing.load_image(mask_path)
    mask = (np.asarray(mask) > 0).astype(np.uint8)[..., None]    # (H, W, 1)
    return img, mask


def load_img_masks(image_path, mask_path):
    img, mask = tf.py_function(_load_pair_numpy, [image_path, mask_path],
                               Tout=[tf.float32, tf.uint8])
    img.set_shape([None, None, 1])
    mask.set_shape([None, None, 1])
    img = tf.image.resize(img, img_size, antialias=True)         # float32
    mask = tf.image.resize(mask, img_size, method="nearest")     # reste uint8
    mask = tf.cast(mask, tf.uint8)

    mask.set_shape([*img_size, 1])

    return img, mask


def augment(image, mask):
    """Mêmes flips / rotations sur l'image et le masque."""
    x = tf.concat([image, tf.cast(mask, tf.float32)], axis=-1)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    image = x[..., :1]
    mask = tf.cast(tf.round(x[..., 1:]), tf.uint8)
    image = image * tf.random.uniform([], 0.9, 1.1) + tf.random.uniform([], -0.05, 0.05)
    return tf.clip_by_value(image, 0.0, 1.0), mask


def get_dataset(batch_size, img_paths, mask_paths, training=False):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if training:
        ds = ds.shuffle(len(img_paths), seed=SEED)
    ds = ds.map(load_img_masks, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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

def unet_model(input_size, n_filters=8, n_classes=2):
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

@tf.keras.utils.register_keras_serializable(package="mito")
def ce_dice_loss(y_true, y_pred, smooth=1.0):
    y_pred = tf.cast(y_pred, tf.float32)
    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, -1)
    y_true = tf.cast(y_true, tf.int32)

    ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))

    t = tf.cast(y_true, tf.float32)
    p = y_pred[..., 1]
    inter = tf.reduce_sum(t * p)
    union = tf.reduce_sum(t) + tf.reduce_sum(p)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return ce + (1.0 - dice)


def plot_history(history, save_path):
    h = history.history if hasattr(history, "history") else history
    epochs = np.arange(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss (la meilleure epoch = minimum de val_loss)
    axes[0].plot(epochs, h["loss"], label="train")
    axes[0].plot(epochs, h["val_loss"], label="val")
    best = int(np.argmin(h["val_loss"]))
    axes[0].axvline(best + 1, color="gray", linestyle="--", linewidth=1,
                    label=f"meilleure val_loss (epoch {best + 1})")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Dice (la meilleure epoch = maximum de val_dice)
    axes[1].plot(epochs, h["dice"], label="train")
    axes[1].plot(epochs, h["val_dice"], label="val")
    best = int(np.argmax(h["val_dice"]))
    axes[1].axvline(best + 1, color="gray", linestyle="--", linewidth=1,
                    label=f"meilleur val_dice = {h['val_dice'][best]:.3f} (epoch {best + 1})")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("dice")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
        
    input_img_paths, target_img_paths = pair_files(input_dir, target_dir)
    assert len(input_img_paths) > 0, "Aucune paire trouvée"

    # --------------------------------------------------------------------------- #
    # Split train / val / test (70 / 15 / 15)
    # --------------------------------------------------------------------------- #
    idx = np.arange(len(input_img_paths))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.3, random_state=SEED)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=SEED)
    print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    np.savez(output_dir / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)

    all_dataset = get_dataset(batch_size, input_img_paths, target_img_paths)

    train_dataset = get_dataset(batch_size, input_img_paths[train_idx], target_img_paths[train_idx], training=True)
    valid_dataset = get_dataset(batch_size, input_img_paths[val_idx], target_img_paths[val_idx])
    test_dataset = get_dataset(batch_size, input_img_paths[test_idx], target_img_paths[test_idx])

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"{name}: {len(idx)} images, {len(input_img_paths[idx])} masques")

    for name, ds in [("train", train_dataset), ("val", valid_dataset), ("test", test_dataset)]:
        n_batches = tf.data.experimental.cardinality(ds).numpy()
        print(f"{name}: {n_batches} batchs")


    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,                  # nb d'epochs sans amélioration avant l'arrêt
            min_delta=1e-3,              # amélioration minimale pour compter comme telle
            restore_best_weights=True,   # remet les poids de la meilleure epoch à la fin
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_model.keras"),
            monitor="val_loss", mode="min", save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
    ]

    unet = unet_model(input_size=(*img_size, 1))
    unet.compile(
        optimizer='adam',
        loss=ce_dice_loss,
        metrics=['accuracy', stats.DiceCoefficient()]
    )
    unet.summary()

    history = unet.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=callbacks,)
    unet.save(output_dir / "unet_final.keras")
    test_loss, test_acc, test_dice = unet.evaluate(test_dataset)

    print(f"Test loss={test_loss:.4f}  accuracy={test_acc:.4f}  dice={test_dice:.4f}")


    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    test_paths = input_img_paths[test_idx]

    results, k = [], 0
    for imgs, masks in test_dataset:
        probs = unet.predict(imgs, verbose=0)[..., 1]            # (B, H, W) proba mitochondrie
        preds = (probs > 0.5).astype(np.uint8)                   # (B, H, W) masque binaire

        for j in range(len(imgs)):
            name = Path(test_paths[k]).stem
            k += 1
            d = stats.dice_coef(preds[j], masks[j, ..., 0].numpy())    # (H, W) contre (H, W)
            results.append((name, d, imgs[j, ..., 0].numpy(), masks[j, ..., 0].numpy(), preds[j]))
            tifffile.imwrite(pred_dir / f"{name}_pred.tif", preds[j] * 255)

    for name, d, *_ in results:
        print(f"{name}: dice={d:.3f}")
    print(f"Dice moyen par image : {np.mean([r[1] for r in results]):.3f}")



    pred_dir = output_dir / "all_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    all_paths = input_img_paths
    results, k = [], 0

    for imgs, masks in all_dataset:
        probs = unet.predict(imgs, verbose=0)[..., 1]            # (B, H, W) proba mitochondrie
        preds = (probs > 0.5).astype(np.uint8)                   # (B, H, W) masque binaire

        for j in range(len(imgs)):
            name = Path(all_paths[k]).stem
            k += 1
            d = stats.dice_coef(preds[j], masks[j, ..., 0].numpy())    # (H, W) contre (H, W)
            results.append((name, d, imgs[j, ..., 0].numpy(), masks[j, ..., 0].numpy(), preds[j]))
            tifffile.imwrite(pred_dir / f"{name}_pred.tif", preds[j] * 255)

    for name, d, *_ in results:
        print(f"{name}: dice={d:.3f}")
    print(f"Dice moyen par image : {np.mean([r[1] for r in results]):.3f}")

    plot_history(history, output_dir / "training_curves.png")

    