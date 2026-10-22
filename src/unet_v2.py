from pathlib import Path
import random
import numpy as np
import tensorflow as tf
import tifffile
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
)

# ----------------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------------
IMAGE_DIR = Path("./data/image_tiff")
MASK_DIR = Path("./data/mask_tiff")

TILE = 256            
TRAIN_STRIDE = 256    
BATCH_SIZE = 16
EPOCHS = 30
VAL_FRACTION = 0.2    
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------------------------------------------------------
# Lecture des données
# ----------------------------------------------------------------------------
def to_2d(a, reduce):
    """Ramène un tableau TIFF à (H, W). `reduce` gère un éventuel axe canal en dernier."""
    a = np.squeeze(a)
    if a.ndim == 3 and a.shape[-1] <= 4:
        a = reduce(a, axis=-1)
    if a.ndim != 2:
        raise ValueError(f"Forme non gérée : {a.shape} (piles 3D non supportées)")
    return a


def normalize_u8(img):
    """Normalisation par percentiles 1-99, stockée en uint8 pour limiter la mémoire."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return np.round(img * 255).astype(np.uint8)


def _key(path):
    """Nom de base commun image/masque (retire les suffixes _mod, _seg, _mask)."""
    stem = path.stem
    for suffix in ("_mod", "_seg", "_mask"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def list_pairs():
    """Apparie images et masques par nom (et non par ordre de tri)."""
    exts = {".tif", ".tiff"}
    images = {_key(p): p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in exts}
    masks = {_key(p): p for p in MASK_DIR.iterdir() if p.suffix.lower() in exts}

    for name in sorted(images.keys() - masks.keys()):
        print(f"[!] pas de masque pour l'image {name}")
    for name in sorted(masks.keys() - images.keys()):
        print(f"[!] pas d'image pour le masque {name}")

    names = sorted(images.keys() & masks.keys())
    if not names:
        raise FileNotFoundError("Aucune paire image/masque trouvée.")
    return [(images[n], masks[n]) for n in names]


def tile_starts(size, tile, stride):
    starts = list(range(0, max(size - tile, 0) + 1, stride))
    if starts[-1] != max(size - tile, 0):
        starts.append(max(size - tile, 0))
    return starts


def make_tiles(img, msk, tile=TILE, stride=TRAIN_STRIDE):
    h, w = img.shape
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="edge")
        msk = np.pad(msk, ((0, ph), (0, pw)), mode="constant")
    H, W = img.shape

    xs, ys = [], []
    for y in tile_starts(H, tile, stride):
        for x in tile_starts(W, tile, stride):
            xs.append(img[y:y + tile, x:x + tile])
            ys.append(msk[y:y + tile, x:x + tile])
    return xs, ys


def load_tiles(pairs):
    X, Y = [], []
    for img_path, msk_path in pairs:
        img = to_2d(tifffile.imread(img_path), np.mean)
        msk = to_2d(tifffile.imread(msk_path), np.max)
        if img.shape != msk.shape:
            raise ValueError(
                f"Tailles différentes : {img_path.name} {img.shape} vs {msk_path.name} {msk.shape}"
            )
        msk = (msk > 0).astype(np.uint8)       # 0/255 -> 0/1
        xs, ys = make_tiles(normalize_u8(img), msk)
        X += xs
        Y += ys
    X = np.stack(X)[..., None]          
    Y = np.stack(Y)[..., None]
    return X, Y


# ----------------------------------------------------------------------------
# Pipeline tf.data
# ----------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE


def to_float(img, msk):
    return tf.cast(img, tf.float32) / 255.0, tf.cast(msk, tf.float32)


def augment(img, msk):
    # Mêmes transformations géométriques sur l'image et le masque
    x = tf.concat([img, msk], axis=-1)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x, tf.random.uniform([], 0, 4, dtype=tf.int32))
    img, msk = x[..., :1], x[..., 1:]
    # Variations photométriques sur l'image uniquement
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 1.0), msk


def build_dataset(X, Y, training):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if training:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(to_float, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# ----------------------------------------------------------------------------
# Modèle U-Net
# ----------------------------------------------------------------------------
def conv_block(inputs, n_filters, dropout_prob=0.0, max_pooling=True):
    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="he_normal")(inputs)
    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="he_normal")(conv)
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    next_layer = MaxPooling2D((2, 2))(conv) if max_pooling else conv
    return next_layer, conv


def upsampling_block(expansive_input, contractive_input, n_filters):
    up = Conv2DTranspose(n_filters, 3, strides=2, padding="same")(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="he_normal")(merge)
    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="he_normal")(conv)
    return conv


def unet_model(input_size=(TILE, TILE, 1), n_filters=16):
    inputs = Input(input_size)

    c1 = conv_block(inputs, n_filters)
    c2 = conv_block(c1[0], n_filters * 2)
    c3 = conv_block(c2[0], n_filters * 4)
    c4 = conv_block(c3[0], n_filters * 8, dropout_prob=0.3)
    c5 = conv_block(c4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    u6 = upsampling_block(c5[0], c4[1], n_filters * 8)
    u7 = upsampling_block(u6, c3[1], n_filters * 4)
    u8 = upsampling_block(u7, c2[1], n_filters * 2)
    u9 = upsampling_block(u8, c1[1], n_filters)

    outputs = Conv2D(1, 1, activation="sigmoid")(u9)
    return Model(inputs, outputs)


# ----------------------------------------------------------------------------
# Loss et métriques (le fond domine : l'accuracy seule est trompeuse)
# ----------------------------------------------------------------------------
def dice_loss(y_true, y_pred, eps=1e-6):
    inter = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denom = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3))
    return 1.0 - tf.reduce_mean((2.0 * inter + eps) / (denom + eps))


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


def dice_coef(y_true, y_pred, eps=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + eps) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + eps)


def iou(y_true, y_pred, eps=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + eps) / (union + eps)


# ----------------------------------------------------------------------------
# Inférence sur une image complète (fenêtre glissante avec chevauchement)
# ----------------------------------------------------------------------------
def predict_image(model, image, threshold=0.5, stride=TILE // 2):
    img = to_2d(image, np.mean)
    h, w = img.shape
    img = normalize_u8(img).astype(np.float32) / 255.0

    ph, pw = max(0, TILE - h), max(0, TILE - w)
    img = np.pad(img, ((0, ph), (0, pw)), mode="edge")
    H, W = img.shape

    coords = [(y, x) for y in tile_starts(H, TILE, stride) for x in tile_starts(W, TILE, stride)]
    tiles = np.stack([img[y:y + TILE, x:x + TILE] for y, x in coords])[..., None]
    probs = model.predict(tiles, batch_size=BATCH_SIZE, verbose=0)[..., 0]

    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for (y, x), p in zip(coords, probs):
        acc[y:y + TILE, x:x + TILE] += p
        cnt[y:y + TILE, x:x + TILE] += 1
    prob = (acc / cnt)[:h, :w]
    return prob, (prob > threshold).astype(np.uint8)


def save_all_predictions(model, dataset, output_dir="resultats_predictions"):
    """
    Sauvegarde la comparaison (Image | Masque | Prédiction | Overlay) 
    pour chaque image du dataset dans un dossier spécifique.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    images_list, masks_list = [], []
    for img_batch, msk_batch in dataset:
        images_list.append(img_batch.numpy())
        masks_list.append(msk_batch.numpy())
    
    images = np.concatenate(images_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    preds = model.predict(images, verbose=0)
    
    total = images.shape[0]
    print(f"Sauvegarde de {total} images de prédiction dans '{output_dir}/'...")

    for i in range(total):
        img = images[i, ..., 0]
        msk = masks[i, ..., 0]
        pred_bin = preds[i, ..., 0] > 0.5

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Image originale")
        
        ax[1].imshow(msk, cmap="gray", vmin=0, vmax=1)
        ax[1].set_title("Masque réel")
        
        ax[2].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
        ax[2].set_title("Prédiction U-Net")
        
        overlay = np.stack([img, img, img], axis=-1)
        overlay[pred_bin] = [1.0, 0.2, 0.2]  # Rouge pour les mitochondries prédites
        ax[3].imshow(overlay)
        ax[3].set_title("Superposition")

        for j in range(4):
            ax[j].axis("off")
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"prediction_{i+1:03d}.png"), dpi=150)
        plt.close(fig)
        
    print("Sauvegarde terminée avec succès !")


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------
def show_predictions(model, dataset, num=4):
    images, masks = next(iter(dataset))
    preds = model.predict(images, verbose=0)
    n = min(num, images.shape[0])

    fig, ax = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    for i in range(n):
        ax[i, 0].imshow(images[i, ..., 0], cmap="gray")
        ax[i, 0].set_title("Image")
        ax[i, 1].imshow(masks[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        ax[i, 1].set_title("Masque réel")
        ax[i, 2].imshow(preds[i, ..., 0] > 0.5, cmap="gray", vmin=0, vmax=1)
        ax[i, 2].set_title("Prédiction U-Net")
        for j in range(3):
            ax[i, j].axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    pairs = list_pairs()
    print(f"{len(pairs)} paires image/masque")

    # Split par IMAGE avant le découpage en tuiles
    random.Random(SEED).shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRACTION)) if len(pairs) > 1 else 0
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    if not val_pairs:
        raise ValueError("Il faut au moins 2 images pour avoir un jeu de validation indépendant.")

    X_train, Y_train = load_tiles(train_pairs)
    X_val, Y_val = load_tiles(val_pairs)
    print(f"Tuiles : train {len(X_train)} | val {len(X_val)}")
    print(f"Proportion de pixels mitochondrie (train) : {Y_train.mean():.3%}")

    train_ds = build_dataset(X_train, Y_train, training=True)
    val_ds = build_dataset(X_val, Y_val, training=False)

    unet = unet_model()
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou],
    )
    unet.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_unet.weights.h5", monitor="val_dice_coef", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef", mode="max", patience=15, restore_best_weights=True),
    ]

    history = unet.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Sauvegarde toutes les images de validation
    save_all_predictions(unet, val_ds)

    # Courbes d'apprentissage
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history.history["dice_coef"], label="train")
    ax[1].plot(history.history["val_dice_coef"], label="val")
    ax[1].set_title("Dice")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # Prédictions sur la VALIDATION
    show_predictions(unet, val_ds, num=6)