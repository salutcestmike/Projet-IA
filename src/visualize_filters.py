"""
Module pour visualiser et comparer différents filtres appliqués à une image DM3.
Ce script charge une image DM3, applique plusieurs filtres et techniques de traitement d'image,
puis affiche les résultats côte à côte pour comparaison.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

import utils.preprocessing as preprocessing
from scipy.ndimage import median_filter, gaussian_filter
from skimage import exposure, restoration

import argparse


def getArgs():
    parser = argparse.ArgumentParser(
        description="Visualiser et comparer différents filtres appliqués à une image DM3 ou TIFF."
    )
    parser.add_argument(
        "-i", "--input_image",
        type=Path,
        help="Chemin vers l'image DM3 à traiter."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=None,
        help="Dossier de sortie pour les images filtrées."
    )

    return parser.parse_args()

def _main(input_image, output_dir):
    image = preprocessing.load_image(input_image)
    image_norm = preprocessing.convert_to_8bit(image)

    images = {
        "Originale normalisée": image_norm,

        "Filtre médian (3x3)": median_filter(
            image_norm, size=3
        ),

        "Filtre gaussien (sigma=0.7)": gaussian_filter(
            image_norm, sigma=0.7
        ),

        "CLAHE": exposure.equalize_adapthist(
            image_norm,
            kernel_size=64,
            clip_limit=0.01
        ),

        "Filtre bilatéral": restoration.denoise_bilateral(
            image_norm,
            sigma_color=0.05,
            sigma_spatial=2,
            channel_axis=None
        ),

        "Non-Local Means": restoration.denoise_nl_means(
            image_norm,
            h=0.05,
            patch_size=5,
            patch_distance=6,
            fast_mode=True,
            channel_axis=None
        ),

        "Filtre gaussien (sigma=0.7) et CLAHE" : exposure.equalize_adapthist(
            gaussian_filter(image_norm, sigma=0.7),
            kernel_size=64,
            clip_limit=0.01
        )
    }

    n_images = len(images)
    n_cols = min(3, n_images)
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    for ax, (title, img) in zip(axes.flat, images.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Comparaison des filtres pour la détection des mitochondries")
    plt.tight_layout()
    plt.show()


    if output_dir:
        output_dir.mkdir(exist_ok=True)

        for name, img in images.items():
            # Nom de fichier sûr
            filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            output_path = output_dir / f"{filename}.tif"

            tifffile.imwrite(output_path, img)
            print(f"Image sauvegardée : {output_path}")


if __name__ == "__main__":
    args = getArgs()
    _main(args.input_image, args.output_dir)