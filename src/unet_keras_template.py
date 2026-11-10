#!/usr/bin/env python3
"""Entraînement et évaluation d'un U-Net pour la segmentation mitochondriale."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Conv2DTranspose,
    concatenate,
)

from utils import preprocessing
import utils.stats as stats  # DiceCoefficient doit avoir name="dice"


# Quatre opérations de pooling imposent des dimensions compatibles avec 2**4.
IMG_SIZE = (512, 512) 
N_SPLITS = 5
INITIAL_LEARNING_RATE = 1e-3
PATCH = 64

# ---------------------------------------------------------------------------
# Arguments et configuration
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande et valide la configuration.

    Returns:
        Arguments validés. Le dossier de sortie est créé s'il n'existe pas."""
    parser = argparse.ArgumentParser(
        description="Segmente des images TIFF ou DM3 avec un modèle U-Net."
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=Path,
        default=Path("./data/dm3_downsample_8"),
        help="Dossier contenant les images TIFF ou DM3.",
    )
    parser.add_argument(
        "-m", "--target_dir",
        type=Path,
        default=Path("./data/mod_tiff_downsample_8"),
        help="Dossier contenant les masques de référence.",
    )
    parser.add_argument(
        "-p", "--splits",
        type=Path,
        default=None,
        help="CSV contenant les colonnes 'stem' et 'split'. S'il est absent, un split est créé.",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=Path("./outputs/run_default"),
        help="Dossier de sortie.",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=4,
        help="Taille des batches d'entraînement et d'évaluation.",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=0,
        help="Graine aléatoire.",
    )
    parser.add_argument(
        "-e", "--max_epochs",
        type=int,
        default=100,
        help="Nombre maximal d'epochs pour chaque fold.",
    )
    parser.add_argument(
        "-t", "--test_size",
        type=float,
        default=0.3,
        help="Proportion du jeu réservée au test lors de la création du split.",
    )
    parser.add_argument(
        "-l", "--load_model",
        type=Path,
        default=None,
        help="Charger un modèle au lieu de l'entraîner.",
    )
    parser.add_argument(
        "-d", "--downsampling",
        type=int,
        default=8,
        help="Facteur de sous-échantillonnage pour l'inférence (1 = pas de sous-échantillonnage).",
    )
    

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Le dossier d'images n'existe pas : {args.input_dir}")
    if not args.target_dir.is_dir():
        parser.error(f"Le dossier de masques n'existe pas : {args.target_dir}")
    if args.batch_size < 1:
        parser.error("batch_size doit être supérieur ou égal à 1.")
    if args.max_epochs < 1:
        parser.error("max_epochs doit être supérieur ou égal à 1.")
    if not 0 < args.test_size < 1:
        parser.error("test_size doit être compris entre 0 et 1.")
    if args.load_model:
        if not args.load_model.is_file():
            parser.error(f"Le fichier du modèle n'existe pas : {args.load_model}")
    else:
        args.load_modele = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


# ---------------------------------------------------------------------------
# Appariement et séparation des fichiers
# ---------------------------------------------------------------------------
def list_files(folder : Union[str, Path]) -> Dict[str, Path]:
    """Liste les fichiers image pris en charge dans un dossier.

    Le filtrage des extensions est insensible à la casse et les fichiers cachés
    sont ignorés.

    Args:
        folder (Union[str, Path]): Dossier à parcourir.

    Returns:
        Dict[str, Path]: Dictionnaire associant chaque nom sans extension (stem)
            au chemin du fichier correspondant."""
    folder = Path(folder)
    extensions = (".tif", ".tiff", ".dm3")
    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in extensions and not path.name.startswith(".")
    }


def pair_files(input_dir: Union[str, Path], target_dir: Union[str, Path]) -> Tuple[List[str], List[str], List[str]]:
    """Associe les images et les masques à partir de leur nom sans extension.

    Seuls les stems présents dans les deux dossiers sont conservés. Les paires
    sont retournées dans l'ordre alphabétique des stems.

    Args:
        input_dir (Union[str, Path]): Dossier contenant les images.
        target_dir (Union[str, Path]): Dossier contenant les masques.

    Returns:
        Tuple[List[str], List[str], List[str]]: Chemins des images, chemins des
            masques et stems communs, dans le même ordre.

    Raises:
        ValueError: Si aucune paire image/masque n'est trouvée."""
    images = list_files(input_dir)
    masks = list_files(target_dir)
    common_stems = sorted(images.keys() & masks.keys())

    print(
        f"Images : {len(images)} | Masques : {len(masks)} | "
        f"Paires : {len(common_stems)}"
    )
    if len(common_stems) != len(images) or len(common_stems) != len(masks):
        print("Attention : certains fichiers n'ont pas de correspondant et seront ignorés.")
    if not common_stems:
        raise ValueError("Aucune paire image/masque n'a été trouvée.")

    image_paths = [str(images[stem]) for stem in common_stems]
    mask_paths = [str(masks[stem]) for stem in common_stems]
    return image_paths, mask_paths, common_stems


def create_train_test_split(input_dir: Union[str, Path], target_dir: Union[str, Path], output_path: Union[str, Path], test_size: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Crée et sauvegarde une séparation train/test reproductible.

    La séparation est effectuée sur les stems des paires image/masque, puis
    exportée dans un CSV contenant les colonnes ``stem`` et ``split``.

    Args:
        input_dir (Union[str, Path]): Dossier contenant les images.
        target_dir (Union[str, Path]): Dossier contenant les masques.
        output_path (Union[str, Path]): Chemin du CSV à créer.
        test_size (float): Proportion des paires affectée au jeu de test.
        seed (int): Graine utilisée pour rendre la séparation reproductible.

    Returns:
        pd.DataFrame: Tableau contenant le stem et l'affectation train/test."""
    _, _, stems = pair_files(input_dir, target_dir)
    indices = np.arange(len(stems))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
    )

    split_labels = np.full(len(stems), "test", dtype=object)
    split_labels[train_indices] = "train"

    split_df = pd.DataFrame({"stem": stems, "split": split_labels})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)

    print(f"Train : {len(train_indices)} | Test : {len(test_indices)}")
    print(f"Fichier de split sauvegardé : {output_path}")
    return split_df


def load_train_test_paths(input_dir: Union[str, Path], target_dir: Union[str, Path], split_path: Union[str, Path]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Recharge un split CSV et reconstruit les chemins des images et masques.

    La fonction accepte également l'ancien nom de colonne ``image`` à la place
    de ``stem``.

    Args:
        input_dir (Union[str, Path]): Dossier contenant les images.
        target_dir (Union[str, Path]): Dossier contenant les masques.
        split_path (Union[str, Path]): CSV décrivant l'affectation des stems.

    Returns:
        Tuple[List[str], List[str], List[str], List[str]]: Chemins des images
            train, masques train, images test et masques test.

    Raises:
        ValueError: Si les colonnes requises sont absentes, si des fichiers
            référencés sont manquants ou si un des deux jeux est vide."""
    image_files = list_files(input_dir)
    mask_files = list_files(target_dir)
    split_df = pd.read_csv(split_path)

    # Compatibilité avec l'ancien nom de colonne "image".
    if "stem" not in split_df.columns and "image" in split_df.columns:
        split_df = split_df.rename(columns={"image": "stem"})

    required_columns = {"stem", "split"}
    if not required_columns.issubset(split_df.columns):
        raise ValueError("Le CSV doit contenir les colonnes 'stem' et 'split'.")

    train_stems = split_df.loc[split_df["split"] == "train", "stem"].astype(str).tolist()
    test_stems = split_df.loc[split_df["split"] == "test", "stem"].astype(str).tolist()

    missing = [
        stem for stem in (train_stems + test_stems)
        if stem not in image_files or stem not in mask_files
    ]
    if missing:
        raise ValueError(f"Fichiers manquants pour les stems suivants : {missing[:10]}")
    if not train_stems or not test_stems:
        raise ValueError("Le split doit contenir au moins une image en train et une en test.")

    train_images = [str(image_files[stem]) for stem in train_stems]
    train_masks = [str(mask_files[stem]) for stem in train_stems]
    test_images = [str(image_files[stem]) for stem in test_stems]
    test_masks = [str(mask_files[stem]) for stem in test_stems]

    return train_images, train_masks, test_images, test_masks


# ---------------------------------------------------------------------------
# Chargement et préparation des données
# ---------------------------------------------------------------------------
def _load_pair_numpy(image_path: tf.Tensor, mask_path: tf.Tensor, downsampling: int) -> Tuple[np.ndarray, np.ndarray]:
    """Charge et prépare une paire image/masque avec NumPy.

    Cette fonction est appelée depuis ``tf.py_function`` : les chemins reçus
    sont donc des tenseurs TensorFlow contenant des chaînes d'octets.

    Args:
        image_path (tf.Tensor): Chemin encodé de l'image à charger.
        mask_path (tf.Tensor): Chemin encodé du masque de référence.
        downsampling (int): Facteur de sous-échantillonnage.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Image normalisée en float32 et masque
            binaire en uint8, tous deux au format (hauteur, largeur, 1).

    Raises:
        ValueError: Si l'image ou le masque n'est pas monocanal ou si leurs
            dimensions spatiales diffèrent."""
    image_path = Path(image_path.numpy().decode("utf-8"))
    mask_path = Path(mask_path.numpy().decode("utf-8"))

    image = preprocessing.down_sample(image_path, downsampling=downsampling)
    image = preprocessing.normalize_image(image)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[..., None]

    mask = preprocessing.down_sample(mask_path, downsampling=downsampling)
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.ndim == 2:
        mask = mask[..., None]

    if image.ndim != 3 or image.shape[-1] != 1:
        raise ValueError(f"L'image doit être en niveaux de gris : {image_path}")
    if mask.ndim != 3 or mask.shape[-1] != 1:
        raise ValueError(f"Le masque doit être monocanal : {mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Dimensions différentes pour {image_path.name} : "
            f"image={image.shape[:2]}, masque={mask.shape[:2]}"
        )

    return image, mask


def load_img_masks(image_path: tf.Tensor, mask_path: tf.Tensor, downsampling: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Charge une paire image/masque et la redimensionne à ``IMG_SIZE``.

    L'image est interpolée bilinéairement, tandis que le masque utilise le plus
    proche voisin afin de préserver ses valeurs de classes.

    Args:
        image_path (tf.Tensor): Chemin TensorFlow de l'image.
        mask_path (tf.Tensor): Chemin TensorFlow du masque.
        downsampling (int): Facteur de sous-échantillonnage.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Image float32 normalisée et masque uint8,
            de forme (IMG_SIZE[0], IMG_SIZE[1], 1)."""
    image, mask = tf.py_function(
        _load_pair_numpy,
        [image_path, mask_path, downsampling],
        Tout=[tf.float32, tf.uint8],
    )
    image.set_shape([None, None, 1])
    mask.set_shape([None, None, 1])

    # Toutes les images sont ramenées à la même taille pour pouvoir être batchées.
    image = tf.image.resize(image, IMG_SIZE, method="bilinear", antialias=True)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask, tf.uint8)
    image.set_shape([*IMG_SIZE, 1])
    mask.set_shape([*IMG_SIZE, 1])
    return image, mask


def _augment(image: tf.Tensor, mask: tf.Tensor, max_tries: int = 10, min_mito: float = 0.01) -> Tuple[tf.Tensor, tf.Tensor]:
    """Extrait un patch contenant si possible une proportion minimale de mitochondries.

    Le recadrage est retenté au maximum ``max_tries`` fois. Les transformations
    géométriques sont communes à l'image et au masque ; les transformations
    photométriques ne concernent que l'image.

    Args:
        image (tf.Tensor): Image float32 de forme (H, W, 1), dans [0, 1].
        mask (tf.Tensor): Masque de forme (H, W) ou (H, W, 1).
        max_tries (int): Nombre maximal de tentatives de recadrage.
        min_mito (float): Proportion minimale de pixels mitochondriaux recherchée.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Patch image float32 et masque uint8, tous
            deux de forme (PATCH, PATCH, 1)."""
    if mask.shape.rank == 2:                       # accepte les deux formats de masque
        mask = mask[..., tf.newaxis]
    stacked = tf.concat([image, tf.cast(mask, tf.float32)], axis=-1)   # (H, W, 2)

    # 1) Recadrage aléatoire : jusqu'à max_tries essais pour un patch avec > min_mito de mito
    init = tf.image.random_crop(stacked, size=(PATCH, PATCH, 2))

    def cond(i: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        return tf.logical_and(i < max_tries, tf.reduce_mean(c[..., 1]) <= min_mito)

    def body(i: tf.Tensor, c: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return i + 1, tf.image.random_crop(stacked, size=(PATCH, PATCH, 2))

    _, combined = tf.while_loop(cond, body, [0, init])

    # 2) Transformations géométriques identiques sur l'image et le masque
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    combined = tf.image.rot90(
        combined,
        k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32),
    )

    image = combined[..., :1]
    mask = tf.cast(tf.round(combined[..., 1:]), tf.uint8)

    # 3) Modifications photométriques appliquées uniquement à l'image
    image = image * tf.random.uniform([], 0.9, 1.1)
    image = image + tf.random.uniform([], -0.05, 0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask

def augment(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Augmente une image et son masque sans modifier leur correspondance spatiale.

    Des retournements et rotations aléatoires sont appliqués conjointement aux
    deux tenseurs. Une variation de luminosité et de contraste simplifiée est
    ensuite appliquée uniquement à l'image.

    Args:
        image (tf.Tensor): Image d'entrée de forme (H, W, 1).
        mask (tf.Tensor): Masque binaire de forme (H, W, 1).

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Image augmentée et masque uint8 augmenté."""
    combined = tf.concat([image, tf.cast(mask, tf.float32)], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    combined = tf.image.rot90(
        combined,
        k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32),
    )

    image = combined[..., :1]
    mask = tf.cast(tf.round(combined[..., 1:]), tf.uint8)

    # Modifications photométriques appliquées uniquement à l'image.
    image = image * tf.random.uniform([], 0.9, 1.1)
    image = image + tf.random.uniform([], -0.05, 0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask


def make_dataset(image_paths: List[str], mask_paths: List[str], downsampling: int, batch_size: int, training: bool = False, seed: int = 0) -> tf.data.Dataset:
    """Construit un pipeline ``tf.data`` à partir de chemins appariés.

    En mode entraînement, les paires sont mélangées et augmentées. Dans tous les
    cas, les données sont regroupées en batches et préchargées en parallèle.

    Args:
        image_paths (List[str]): Chemins des images.
        mask_paths (List[str]): Chemins des masques correspondants.
        downsampling (int): Facteur de sous-échantillonnage.
        batch_size (int): Nombre d'échantillons par batch.
        training (bool): Active le mélange et l'augmentation si vrai.
        seed (int): Graine du mélange des données.

    Returns:
        tf.data.Dataset: Dataset produisant des batches (images, masques).

    Raises:
        ValueError: Si les listes n'ont pas la même longueur ou sont vides."""
    if len(image_paths) != len(mask_paths):
        raise ValueError("Le nombre d'images et de masques doit être identique.")
    if not image_paths:
        raise ValueError("Impossible de créer un dataset vide.")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(lambda img, mask: load_img_masks(img, mask, downsampling), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Architecture U-Net et fonction de perte
# ---------------------------------------------------------------------------
def conv_block(inputs: tf.Tensor, n_filters: int, dropout_prob: float = 0.0, max_pooling: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
    """Construit un bloc encodeur composé de deux convolutions 2D.

    Un dropout peut être ajouté après les convolutions. Le max-pooling est
    optionnel afin de permettre au dernier bloc de l'encodeur de ne pas réduire
    la résolution.

    Args:
        inputs (tf.Tensor): Tenseur d'entrée du bloc.
        n_filters (int): Nombre de filtres de chaque convolution.
        dropout_prob (float): Probabilité de dropout ; 0 désactive cette couche.
        max_pooling (bool): Indique si un max-pooling 2D est appliqué.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Tenseur transmis au bloc suivant et tenseur
            de convolution conservé pour la skip connection."""
    conv = Conv2D(
        n_filters, 3, activation="relu", padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    conv = Conv2D(
        n_filters, 3, activation="relu", padding="same",
        kernel_initializer="he_normal",
    )(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    next_layer = MaxPooling2D((2, 2))(conv) if max_pooling else conv
    return next_layer, conv


def upsampling_block(expansive_input: tf.Tensor, contractive_input: tf.Tensor, n_filters: int) -> tf.Tensor:
    """Construit un bloc décodeur avec suréchantillonnage et skip connection.

    L'entrée du décodeur est agrandie par convolution transposée, concaténée
    avec la sortie correspondante de l'encodeur, puis traitée par deux
    convolutions 2D.

    Args:
        expansive_input (tf.Tensor): Tenseur provenant du chemin décodeur.
        contractive_input (tf.Tensor): Tenseur du chemin encodeur à concaténer.
        n_filters (int): Nombre de filtres des convolutions.

    Returns:
        tf.Tensor: Tenseur de caractéristiques après concaténation et convolutions."""
    up = Conv2DTranspose(
        n_filters, 3, strides=2, padding="same"
    )(expansive_input)
    merged = concatenate([up, contractive_input], axis=-1)

    conv = Conv2D(
        n_filters, 3, activation="relu", padding="same",
        kernel_initializer="he_normal",
    )(merged)
    conv = Conv2D(
        n_filters, 3, activation="relu", padding="same",
        kernel_initializer="he_normal",
    )(conv)
    return conv


def unet_model(input_size: Tuple[Optional[int], Optional[int], int] = (None, None, 1), n_filters: int = 8, n_classes: int = 2) -> Model:
    """Construit l'architecture U-Net utilisée pour la segmentation binaire.

    L'encodeur comporte quatre niveaux de pooling, suivis d'un niveau profond.
    Le décodeur réutilise les cartes de caractéristiques de l'encodeur via des
    skip connections. La couche finale produit une probabilité par classe.

    Args:
        input_size (Tuple[Optional[int], Optional[int], int]): Forme de l'entrée
            (hauteur, largeur, canaux). Les dimensions spatiales peuvent être None.
        n_filters (int): Nombre de filtres du premier niveau de l'encodeur.
        n_classes (int): Nombre de classes en sortie du modèle.

    Returns:
        Model: Modèle Keras U-Net non encore compilé."""
    inputs = Input(shape=input_size)

    c1 = conv_block(inputs, n_filters)
    c2 = conv_block(c1[0], n_filters * 2)
    c3 = conv_block(c2[0], n_filters * 4)
    c4 = conv_block(c3[0], n_filters * 8, dropout_prob=0.3)
    c5 = conv_block(c4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    u6 = upsampling_block(c5[0], c4[1], n_filters * 8)
    u7 = upsampling_block(u6, c3[1], n_filters * 4)
    u8 = upsampling_block(u7, c2[1], n_filters * 2)
    u9 = upsampling_block(u8, c1[1], n_filters)

    outputs = Conv2D(n_classes, 1, activation="softmax")(u9)
    return Model(inputs=inputs, outputs=outputs)


@tf.keras.utils.register_keras_serializable(package="mito")
def ce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Calcule la somme de l'entropie croisée sparse et de ``1 - Dice``.

    Le Dice est calculé à partir de la probabilité prédite pour la classe 1
    (mitochondrie) et du masque de vérité terrain. ``smooth`` stabilise le
    calcul lorsque le dénominateur est nul.

    Args:
        y_true (tf.Tensor): Masques de référence entiers, avec un canal final.
        y_pred (tf.Tensor): Probabilités prédites pour chaque classe.
        smooth (float): Constante de lissage du coefficient Dice.

    Returns:
        tf.Tensor: Valeur scalaire de la fonction de perte."""
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)

    cross_entropy = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    )

    true_positive_mask = tf.cast(y_true, tf.float32)
    predicted_probability = y_pred[..., 1]
    intersection = tf.reduce_sum(true_positive_mask * predicted_probability)
    denominator = (
        tf.reduce_sum(true_positive_mask)
        + tf.reduce_sum(predicted_probability)
    )
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return cross_entropy + (1.0 - dice)


# ---------------------------------------------------------------------------
# Entraînement et validation croisée
# ---------------------------------------------------------------------------
def compile_model() -> Model:
    """Construit le U-Net et configure son apprentissage.

    L'optimiseur Adam utilise le taux d'apprentissage global. La fonction de
    perte combine entropie croisée et Dice ; l'accuracy et le Dice sont suivis.

    Returns:
        Model: Modèle U-Net compilé, prêt pour ``fit`` ou ``evaluate``."""
    model = unet_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss=ce_dice_loss,
        metrics=["accuracy", stats.DiceCoefficient()],
    )
    return model


def make_callbacks(output_dir: Path, fold: int) -> List[tf.keras.callbacks.Callback]:
    """Prépare les callbacks utilisés pendant l'entraînement d'un fold.

    La réduction du taux d'apprentissage intervient lorsque la loss de validation
    stagne ; l'arrêt anticipé restaure les meilleurs poids et le checkpoint
    sauvegarde le meilleur modèle du fold.

    Args:
        output_dir (Path): Dossier de sauvegarde des modèles.
        fold (int): Numéro du fold courant.

    Returns:
        List[tf.keras.callbacks.Callback]: Callbacks Keras à transmettre à ``fit``."""
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir / f"unet_fold_{fold}.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


def run_cross_validation(train_images: List[str], train_masks: List[str], args: argparse.Namespace) -> Tuple[List[Dict[str, List[float]]], List[int]]:
    """Entraîne et valide un modèle U-Net sur chacun des folds.

    Les données d'entraînement sont réparties en ``N_SPLITS`` folds. Un nouveau
    modèle est compilé pour chaque fold ; les historiques et le numéro de la
    meilleure epoch selon ``val_loss`` sont conservés.

    Args:
        train_images (List[str]): Chemins des images du jeu d'entraînement.
        train_masks (List[str]): Chemins des masques correspondants.
        args (argparse.Namespace): Configuration, notamment seed, batch_size,
            max_epochs et output_dir.

    Returns:
        Tuple[List[Dict[str, List[float]]], List[int]]: Historiques Keras des
            folds et meilleures epochs associées."""
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    histories = []
    best_epochs = []

    image_paths = np.asarray(train_images)
    mask_paths = np.asarray(train_masks)

    for fold, (train_indices, val_indices) in enumerate(
        kfold.split(image_paths), start=1
    ):
        print(f"\n{'=' * 20} FOLD {fold}/{N_SPLITS} {'=' * 20}")

        fold_train_dataset = make_dataset(
            image_paths[train_indices].tolist(),
            mask_paths[train_indices].tolist(),
            batch_size=args.batch_size,
            training=True,
            seed=args.seed + fold,
            downsampling=args.downsampling
        )
        fold_val_dataset = make_dataset(
            image_paths[val_indices].tolist(),
            mask_paths[val_indices].tolist(),
            batch_size=args.batch_size,
            training=False,
            seed=args.seed,
            downsampling=args.downsampling
        )

        model = compile_model()
        history = model.fit(
            fold_train_dataset,
            validation_data=fold_val_dataset,
            epochs=args.max_epochs,
            callbacks=make_callbacks(args.output_dir, fold),
            verbose=1,
        )
        histories.append(history.history)

        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        best_epochs.append(best_epoch)
        print(f"Fold {fold} - meilleure epoch (val_loss) : {best_epoch}")

        tf.keras.backend.clear_session()

    return histories, best_epochs


# ---------------------------------------------------------------------------
# Visualisation des courbes de validation croisée
# ---------------------------------------------------------------------------
def plot_cv_history(histories: List[Dict[str, List[float]]], save_path: Union[str, Path]) -> None:
    """Trace et sauvegarde les courbes de loss et de Dice de la validation croisée.

    Pour chaque fold, les courbes d'entraînement et de validation sont affichées.
    Les meilleures loss et valeurs de Dice sont ensuite récapitulées dans la
    sortie standard.

    Args:
        histories (List[Dict[str, List[float]]]): Historiques renvoyés par Keras.
        save_path (Union[str, Path]): Chemin du graphique à enregistrer.

    Raises:
        KeyError: Si aucune métrique Dice exploitable n'est présente dans un
            historique."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    best_losses = []
    best_dices = []
    best_loss_epochs = []

    for fold_index, history in enumerate(histories, start=1):
        epochs = np.arange(1, len(history["loss"]) + 1)
        dice_key = "dice" if "dice" in history else next(
            (key for key in history if key.lower().endswith("dice")), None
        )
        if dice_key is None or f"val_{dice_key}" not in history:
            raise KeyError(
                "Métrique Dice introuvable dans l'historique. "
                f"Clés disponibles : {list(history)}"
            )

        val_losses = np.asarray(history["val_loss"])
        val_dices = np.asarray(history[f"val_{dice_key}"])
        best_loss_index = int(np.argmin(val_losses))
        best_dice_index = int(np.argmax(val_dices))

        best_losses.append(float(val_losses[best_loss_index]))
        best_dices.append(float(val_dices[best_dice_index]))
        best_loss_epochs.append(best_loss_index + 1)

        axes[0].plot(epochs, history["loss"], alpha=0.6, label=f"Fold {fold_index} train")
        axes[0].plot(
            epochs, val_losses, linestyle="--", alpha=0.8,
            label=f"Fold {fold_index} val",
        )
        axes[0].scatter(
            best_loss_index + 1, val_losses[best_loss_index], marker="o", s=35
        )

        axes[1].plot(
            epochs, history[dice_key], alpha=0.6,
            label=f"Fold {fold_index} train",
        )
        axes[1].plot(
            epochs, val_dices, linestyle="--", alpha=0.8,
            label=f"Fold {fold_index} val",
        )
        axes[1].scatter(
            best_dice_index + 1, val_dices[best_dice_index], marker="o", s=35
        )

    axes[0].set_title("Loss - 5-fold cross-validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Dice - 5-fold cross-validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print("\nRésultats de la validation croisée :")
    for index, (epoch, loss, dice) in enumerate(
        zip(best_loss_epochs, best_losses, best_dices), start=1
    ):
        print(
            f"Fold {index} : meilleure epoch={epoch}, "
            f"val_loss minimale={loss:.4f}, val_dice maximal={dice:.4f}"
        )

    print(
        f"\nVal_loss minimale moyenne : {np.mean(best_losses):.4f} "
        f"± {np.std(best_losses):.4f}"
    )
    print(
        f"Val_Dice maximal moyen : {np.mean(best_dices):.4f} "
        f"± {np.std(best_dices):.4f}"
    )


# ---------------------------------------------------------------------------
# Évaluation et sauvegarde des prédictions
# ---------------------------------------------------------------------------
def save_predictions(model: Model, dataset: tf.data.Dataset, image_paths: List[str], output_dir: Path) -> List[Tuple[str, float]]:
    """Prédit les masques d'un dataset et sauvegarde les résultats au format TIFF.

    La classe 1 est considérée comme la classe mitochondrie et seuillée à 0.5.
    Le coefficient Dice est calculé individuellement pour chaque image.

    Args:
        model (Model): Modèle Keras entraîné.
        dataset (tf.data.Dataset): Dataset fournissant les images et masques.
        image_paths (List[str]): Chemins des images, dans l'ordre du dataset.
        output_dir (Path): Dossier dans lequel enregistrer les masques prédits.

    Returns:
        List[Tuple[str, float]]: Liste des couples (nom de l'image, Dice).

    Raises:
        RuntimeError: Si le nombre d'images parcourues diffère du nombre de
            chemins fournis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    image_index = 0

    for images, masks in tqdm(dataset, desc="Batch"):
        probabilities = model.predict(images, verbose=0)[..., 1]
        predictions = (probabilities > 0.5).astype(np.uint8)

        for batch_index in range(len(images)):
            image_name = Path(image_paths[image_index]).stem
            image_index += 1

            true_mask = masks[batch_index, ..., 0].numpy()
            dice = stats.dice_coef(predictions[batch_index], true_mask)
            results.append((image_name, float(dice)))

            tifffile.imwrite(
                output_dir / f"{image_name}_pred.tif",
                predictions[batch_index] * 255,
            )

    if image_index != len(image_paths):
        raise RuntimeError(
            f"{image_index} images prédites sur {len(image_paths)} chemins fournis."
        )

    for image_name, dice in results:
        print(f"{image_name} : Dice={dice:.3f}")
    print(f"Dice moyen par image : {np.mean([dice for _, dice in results]):.3f}")
    return results


def main() -> None:
    """Exécute le pipeline complet d'entraînement et d'évaluation.

    Prépare ou recharge le split train/test, effectue la validation croisée,
    entraîne le modèle final, évalue ses performances et sauvegarde les
    prédictions sur le jeu de test puis sur l'ensemble des images."""
    args = parse_args()
    # tf.keras.utils.set_random_seed(args.seed)

    split_path = args.splits or Path("train_test_splits.csv")
    if args.splits is None or not split_path.exists():
        create_train_test_split(
            args.input_dir,
            args.target_dir,
            split_path,
            test_size=args.test_size,
            seed=args.seed,
        )

    train_images, train_masks, test_images, test_masks = load_train_test_paths(
        args.input_dir,
        args.target_dir,
        split_path,
    )
    print(f"Dataset train : {len(train_images)} images")
    print(f"Dataset test  : {len(test_images)} images")

    if not args.load_model:
        # Le jeu de test reste entièrement à l'écart de la validation croisée.
        histories, best_epochs = run_cross_validation(train_images, train_masks, args)
        plot_cv_history(histories, args.output_dir / "cross_validation_curves.png")

        # Nombre d'epochs final choisi à partir des meilleures epochs des folds.
        final_epochs = max(1, int(round(np.mean(best_epochs))))
        print(f"\nEntraînement final sur tout le train pendant {final_epochs} epochs.")

    final_train_dataset = make_dataset(
        train_images,
        train_masks,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
        downsampling=args.downsampling
    )
    test_dataset = make_dataset(
        test_images,
        test_masks,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
        downsampling=args.downsampling
    )

    if not args.load_model:
        final_model = compile_model()
        final_model.summary()
        final_model.fit(final_train_dataset, epochs=final_epochs, verbose=1)

        model_path = args.output_dir / "unet_final.keras"
        final_model.save(model_path)
        print(f"Modèle sauvegardé : {model_path}")
    else:
        final_model = tf.keras.models.load_model(
            args.load_model,
            custom_objects={
                "ce_dice_loss": ce_dice_loss,
                "DiceCoefficient": stats.DiceCoefficient
            }
        )
            
    test_loss, test_accuracy, test_dice = final_model.evaluate(test_dataset, verbose=1)
    print(
        f"Test : loss={test_loss:.4f} | "
        f"accuracy={test_accuracy:.4f} | Dice={test_dice:.4f}"
    )

    save_predictions(
        final_model,
        test_dataset,
        test_images,
        args.output_dir / "predictions_test",
    )

    # Prédictions sur toutes les images avec le modèle final.
    all_images = train_images + test_images
    all_masks = train_masks + test_masks
    all_dataset = make_dataset(
        all_images,
        all_masks,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
        downsampling=args.downsampling
    )
    save_predictions(
        final_model,
        all_dataset,
        all_images,
        args.output_dir / "predictions_all",
    )


if __name__ == "__main__":
    main()
