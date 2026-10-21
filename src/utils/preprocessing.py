"""
Ce module contient des fonctions pour charger, normaliser et convertir des images DM3 ou TIFF,
ainsi que pour convertir des fichiers IMOD (.mod) en masques TIFF binaires.
"""

from pathlib import Path

import numpy as np
import tifffile
from ncempy.io import dm

from imodmodel import ImodModel
from skimage.draw import polygon
import tensorflow as tf

def load_image(path):
    """Charge une image à partir d'un fichier DM3 ou TIFF.
    Renvoie l'image sous forme de tableau numpy 2D."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    if path.suffix.lower() == ".dm3":
        f = dm.dmReader(path)
        image = np.asarray(f["data"])
    elif path.suffix.lower() in [".tif", ".tiff"]:
        image = tifffile.imread(path)

    image = image.astype(np.float32)
    return image


def normalize_image(image: np.ndarray, p_low=1.0, p_high=99.0, clip=True, eps=1e-8) -> np.ndarray:
    """Normalise une image en niveaux de gris selon les percentiles p_low et p_high.
    Si clip=True, les valeurs sont limitées à [0, 1]."""
    img = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(img, [p_low, p_high])
    out = (img - lo) / (hi - lo + eps)
    return np.clip(out, 0.0, 1.0) if clip else out


def convert_to_8bit(image, p_low=None, p_high=None):
    """Convertit un tableau NumPy (uint16, float...) en uint8 [0, 255].

    - p_low / p_high = None : étire entre le min et le max de l'image.
    - sinon : étire entre ces percentiles (par ex. 1 et 99), valeurs extrêmes écrêtées.
    """
    img = np.asarray(image, dtype=np.float32)
    if p_low is None or p_high is None:
        lo, hi = img.min(), img.max()
    else:
        lo, hi = np.percentile(img, [p_low, p_high])
    out = (img - lo) / (hi - lo + 1e-8)
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def binarize_mask(mask):
    """Convertit un masque d'instances en masque binaire (0 ou 255)."""
    return (mask > 0).astype(np.uint8) * 255


def mod_to_tiff(mod_image_path, output_tiff_path = None):
    """Convertit un fichier IMOD (.mod) en masque TIFF binaire."""
    model = ImodModel.from_file(str(mod_image_path))
    height, width = model.header.ymax, model.header.xmax
    # Recherche de l'objet contenant les mitochondries
    mito_objects = [obj for obj in model.objects if obj.header.name.lower() == "mitochondria"]

    if not mito_objects:
        raise ValueError("Objet 'mitochondria' introuvable dans le modèle.")

    # Image binaire de sortie
    mask = np.zeros((height, width), dtype=np.uint8)

    number_of_contours = 0
    number_of_points = 0
    number_of_pixels = 0

    for obj in mito_objects:
        for contour in obj.contours:

            points = np.asarray(contour.points)

            if len(points) < 3:
                continue

            # Les coordonnées IMOD sont données sous la forme (x, y, z).
            # Pour un TIFF 2D, on utilise x et y.
            xy = np.rint(points[:, :2]).astype(int)

            # Séparer les coordonnées x et y
            x = xy[:, 0]
            y = xy[:, 1]

            # Remplir le polygone délimité par le contour
            rr, cc = polygon(y, x, shape=mask.shape)

            mask[rr, cc] = 255

            number_of_contours += 1
            number_of_points += len(points)
            number_of_pixels += len(rr)

    # Inversion verticale
    mask = np.flipud(mask)

    print("\nConversion terminée !")
    print(f"Contours remplis : {number_of_contours}")
    print(f"Points traités : {number_of_points}")
    print(f"Pixels remplis (somme par contour) : {number_of_pixels}")
    print(f"Pixels blancs dans le masque : {np.count_nonzero(mask)}")

    if output_tiff_path:
        output_tiff_path = Path(output_tiff_path)
        output_tiff_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output_tiff_path, mask)
        print(f"Masque enregistré : {output_tiff_path}")

    return mask


def resize_np(img, size, method="bilinear", antialias=False):
    """Redimensionne un tableau NumPy (H, W) ou (H, W, C) avec tf.image.resize.
    size = (new_h, new_w). Renvoie un tableau NumPy de même rang que l'entrée."""
    img = np.asarray(img)
    was_2d = img.ndim == 2
    if was_2d:
        img = img[..., None]                                   # (H, W) -> (H, W, 1)

    out = tf.image.resize(img, size, method=method, antialias=antialias).numpy()

    if was_2d:
        out = out[..., 0]                                      # (h, w, 1) -> (h, w)
    return out

def down_sample(image_path, output_path=None, downsampling=2):
    """Rééchantillonne une image en réduisant sa taille par un facteur de sous-échantillonnage."""
    image = load_image(image_path)
    if not np.any(image):
        return np.zeros((image.shape[0] // downsampling, image.shape[1] // downsampling), dtype=np.uint8)
    
    downsampled_image = convert_to_8bit(image)
    downsampled_image = downsampled_image[::downsampling, ::downsampling]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output_path, downsampled_image)
        print(f"Image rééchantillonnée enregistrée : {output_path}")
    #
    return downsampled_image