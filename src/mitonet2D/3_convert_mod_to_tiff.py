from pathlib import Path

import numpy as np
import tifffile
from imodmodel import ImodModel
from skimage.draw import line


# ============================================================
# FICHIERS
# ============================================================

IMAGE_PATH = Path("./data/HCT116-0Gy-3j-2900X-0002.tif")
MODEL_PATH =  Path("./data/HCT116-0Gy-3j-2900X-0002.mod")
OUTPUT_PATH = Path("./data/HCT116-0Gy-3j-2900X-0002_mask_mod.tif")


# ============================================================
# CHARGEMENT DE L'IMAGE ORIGINALE
# ============================================================

image = tifffile.imread(IMAGE_PATH)

if image.ndim != 2:
    raise ValueError(
        f"L'image doit être 2D, mais ses dimensions sont {image.shape}."
    )

height, width = image.shape

print(f"Image originale : {width} × {height} pixels")


# ============================================================
# CHARGEMENT DU MODELE IMOD
# ============================================================

model = ImodModel.from_file(str(MODEL_PATH))

# Recherche de l'objet contenant les mitochondries
mito_objects = [
    obj
    for obj in model.objects
    if obj.header.name.lower() == "mitochondria"
]

if not mito_objects:
    raise ValueError("Objet 'mitochondria' introuvable dans le modèle.")

# Image binaire de sortie
contour_image = np.zeros((height, width), dtype=np.uint8)

number_of_contours = 0
number_of_points = 0
number_of_pixels = 0


# ============================================================
# TRACER LES CONTOURS
# ============================================================

for obj in mito_objects:
    for contour in obj.contours:

        points = np.asarray(contour.points)

        if len(points) < 2:
            continue

        # Les coordonnées IMOD sont données sous la forme (x, y, z).
        # Pour un TIFF 2D, on utilise x et y.
        xy = points[:, :2]

        # Conversion en indices de pixels
        xy = np.rint(xy).astype(int)

        # Tracer chaque segment entre deux points successifs
        for i in range(len(xy) - 1):
            x1, y1 = xy[i]
            x2, y2 = xy[i + 1]

            # Ignorer les segments entièrement hors de l'image
            if (
                max(x1, x2) < 0
                or min(x1, x2) >= width
                or max(y1, y2) < 0
                or min(y1, y2) >= height
            ):
                continue

            rr, cc = line(y1, x1, y2, x2)

            # Ne conserver que les pixels à l'intérieur de l'image
            valid = (
                (rr >= 0)
                & (rr < height)
                & (cc >= 0)
                & (cc < width)
            )

            contour_image[rr[valid], cc[valid]] = 255

            number_of_pixels += int(valid.sum())

        # Fermer le contour en reliant le dernier point au premier
        # si le contour n'est pas déjà fermé.
        if not np.array_equal(xy[0], xy[-1]):
            x1, y1 = xy[-1]
            x2, y2 = xy[0]

            if not (
                max(x1, x2) < 0
                or min(x1, x2) >= width
                or max(y1, y2) < 0
                or min(y1, y2) >= height
            ):
                rr, cc = line(y1, x1, y2, x2)

                valid = (
                    (rr >= 0)
                    & (rr < height)
                    & (cc >= 0)
                    & (cc < width)
                )

                contour_image[rr[valid], cc[valid]] = 255

                number_of_pixels += int(valid.sum())

        number_of_contours += 1
        number_of_points += len(points)


# ============================================================
# ENREGISTREMENT DU TIFF
# ============================================================

tifffile.imwrite(OUTPUT_PATH, np.flipud(contour_image))

print("\nConversion terminée !")
print(f"Contours extraits : {number_of_contours}")
print(f"Points traités : {number_of_points}")
print(f"Pixels tracés (comptage des segments) : {number_of_pixels}")
print(f"Fichier créé : {OUTPUT_PATH.resolve()}")
