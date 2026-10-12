from pathlib import Path

import numpy as np
import tifffile
import torch

from empanada.config_loaders import read_yaml
from empanada_napari.inference import Engine2d
from empanada_napari.utils import get_configs

from ncempy.io import dm
import matplotlib.pyplot as plt



# À ADAPTER : dossier contenant les images DM3
INPUT_DIR = Path("./data")

# Dossier de sortie des segmentations
OUTPUT_DIR = INPUT_DIR / "segmentations"

# Modèle à utiliser
MODEL_NAME = "MitoNet_v1"

# Paramètres d'inférence
DOWNSAMPLING = 1
USE_QUANTIZED = True


def load_model():
    """Charge la configuration et initialise MitoNet."""
    configs = get_configs()

    if MODEL_NAME not in configs:
        raise RuntimeError(
            f"Configuration {MODEL_NAME!r} introuvable. "
            f"Modèles disponibles : {list(configs)}"
        )

    config = read_yaml(configs[MODEL_NAME])

    print(f"Chargement du modèle {MODEL_NAME}...")
    print("Le téléchargement des poids peut prendre quelques minutes.")

    model = Engine2d(
        model_config=config,
        inference_scale=DOWNSAMPLING,
        label_divisor=1000,
        nms_threshold=0.1,
        nms_kernel=3,
        confidence_thr=0.5,
        use_gpu=False,
        use_quantized=USE_QUANTIZED,
    )

    return model


def segment_image(model, image):
    """Segmente une image 2D et renvoie un masque d'instances."""
    if image.ndim != 2:
        raise ValueError(
            f"Image attendue en 2D, dimensions reçues : {image.shape}"
        )

    if image.dtype == np.bool_:
        image = image.astype(np.uint8) * 255

    # MitoNet effectue lui-même le prétraitement de l'image.
    prediction = model.infer(image)

    # Renumérote les instances de 1 à N ; 0 reste le fond.
    labels = np.unique(prediction)
    labels = labels[labels != 0]

    mask = np.zeros(prediction.shape, dtype=np.uint32)

    for new_label, old_label in enumerate(labels, start=1):
        mask[prediction == old_label] = new_label

    return mask


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(INPUT_DIR.glob("*.dm3"))
    )

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier DM3 trouvé dans {INPUT_DIR}"
        )

    print(f"{len(files)} image(s) trouvée(s).")
    print(f"PyTorch : {torch.__version__}")
    print("Calcul sur CPU.")

    model = load_model()

    for index, path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] {path.name}")

        f = dm.dmReader(path)
        image = np.asarray(f["data"])
        image = image.astype(np.float32)
        p1, p99 = np.percentile(image, [1, 99])

        image = np.clip(
            (image - p1) / (p99 - p1 + 1e-8) * 255,
            0,
            255
        ).astype(np.uint8)


        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.title("Image 8 bits")
        plt.axis("off")
        plt.show()

        print(
            f"Dimensions : {image.shape} | "
            f"type : {image.dtype} | "
            f"intensité : [{image.min()}, {image.max()}]"
        )

        mask = segment_image(model, image)

        output_path = OUTPUT_DIR / f"{path.stem}_seg.tif"
        tifffile.imwrite(output_path,(mask > 0).astype(np.uint8) * 255)

        n_instances = len(np.unique(mask)) - 1

        print(f"Instances détectées : {n_instances}")
        print(f"Résultat enregistré : {output_path}")

    print("\nSegmentation terminée.")


if __name__ == "__main__":
    main()
