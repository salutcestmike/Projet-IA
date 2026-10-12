import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

from ncempy.io import dm
from scipy.ndimage import median_filter, gaussian_filter
from skimage import exposure, restoration


def load_dm3(path):
    """Charge une image DM3."""
    f = dm.dmReader(path)
    image = np.asarray(f["data"])
    return image


def normalize_image(image):
    """Normalise les intensités entre 0 et 255 en utilisant les percentiles 1 et 99."""
    image = image.astype(np.float32)

    p1, p99 = np.percentile(image, [1, 99])

    if p99 <= p1:
        raise ValueError("Image constante ou intensités invalides.")

    image = np.clip((image - p1) / (p99 - p1), 0, 1)

    return image


def main():
    path = "data/HCT116-0Gy-3j-2900X-0002.dm3"  # À remplacer par le chemin de ton fichier

    # 1. Charger et normaliser l'image
    image = load_dm3(path)
    image_norm = normalize_image(image)

    # 2. Appliquer les filtres
    images = {
        # "Originale normalisée": image_norm,

        # "Filtre médian (3x3)": median_filter(
        #     image_norm, size=3
        # ),

        # "Filtre gaussien (sigma=0.7)": gaussian_filter(
        #     image_norm, sigma=0.7
        # ),

        # "CLAHE": exposure.equalize_adapthist(
        #     image_norm,
        #     kernel_size=64,
        #     clip_limit=0.01
        # ),

        # "Filtre bilatéral": restoration.denoise_bilateral(
        #     image_norm,
        #     sigma_color=0.05,
        #     sigma_spatial=2,
        #     channel_axis=None
        # ),

        # "Non-Local Means": restoration.denoise_nl_means(
        #     image_norm,
        #     h=0.05,
        #     patch_size=5,
        #     patch_distance=6,
        #     fast_mode=True,
        #     channel_axis=None
        # ),

        "Filtre gaussien (sigma=0.7) et CLAHE" : exposure.equalize_adapthist(
            gaussian_filter(image_norm, sigma=0.7),
            kernel_size=64,
            clip_limit=0.01
        )
    }

    # 3. Afficher tous les résultats dans une seule fenêtre
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for ax, (title, img) in zip(axes.flat, images.items()):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Comparaison des filtres pour la détection des mitochondries")
    plt.tight_layout()
    plt.show()


    output_dir = Path("data/filtered")
    output_dir.mkdir(exist_ok=True)

    for name, img in images.items():
        # Conversion en 8 bits
        img_8bit = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Nom de fichier sûr
        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        output_path = output_dir / f"{filename}.tif"

        tifffile.imwrite(output_path, img_8bit)
        print(f"Image sauvegardée : {output_path}")


if __name__ == "__main__":
    main()