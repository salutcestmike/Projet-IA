from pathlib import Path

import numpy as np
import tifffile
from ncempy.io import dm



def load_image(path):
    """Charge une image à partir d'un fichier DM3 ou TIFF.
    Renvoie l'image sous forme de tableau numpy 2D."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    if path.suffix.lower() == ".dm3":
        f = dm.dmReader(path)
        image = np.asarray(f["data"])
        image = image.astype(np.float32)
    elif path.suffix.lower() in [".tif", ".tiff"]:
        image = tifffile.imread(path)

    return image

def discretize_image(image):
    """Convertit une image en 8 bits (0-255) après normalisation."""
    p1, p99 = np.percentile(image, [1, 99])
    image = np.clip(
        (image - p1) / (p99 - p1 + 1e-8) * 255,
        0,
        255
    ).astype(np.uint8)

    return image

def binarize_mask(mask):
    """Convertit un masque d'instances en masque binaire (0 ou 255)."""
    return (mask > 0).astype(np.uint8) * 255


