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

def convert_to_8bit_v1(image: np.ndarray) -> np.ndarray:
    """Convertit une image en niveaux vers 8 bits."""
    image = image.astype(np.float32)

    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        return np.zeros(image.shape, dtype=np.uint8)

    image_8bit = (image).astype(np.float64) / 256

    return image_8bit.astype(np.uint8)

def convert_to_8bit_v2(image: np.ndarray) -> np.ndarray:
    """
    Convertit une image en niveaux vers 8 bits en utilisant les percentiles 1 et 99 pour la normalisation.
    Cette méthode est plus robuste aux valeurs aberrantes que la normalisation min-max.
    """
    image = image.astype(np.float32)

    min_val, max_val = np.percentile(image, [1, 99])

    if max_val <= min_val:
        return np.zeros(image.shape, dtype=np.uint8)

    image = (image - min_val) / (max_val - min_val)
    image = np.clip(image, 0, 1)

    return (image * 255).round().astype(np.uint8)

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