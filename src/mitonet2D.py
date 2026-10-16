"""
Segmentation d'images DM3 ou TIFF en utilisant MitoNet en mode 2D.
"""

from pathlib import Path
import src.utils.preprocessing as preprocessing
import argparse

import numpy as np
import tifffile
import torch

from empanada.config_loaders import read_yaml
from empanada_napari.inference import Engine2d
from empanada_napari.utils import get_configs

from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage import exposure


def getArgs():
    parser = argparse.ArgumentParser(
        description="Segmente des images DM3 ou TIFF en utilisant MitoNet en mode 2D."
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=Path,
        default=Path("./data"),
        help="Dossier contenant les images DM3 ou TIFF à segmenter."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=None,
        help="Dossier de sortie pour les segmentations. "
             "Par défaut, un sous-dossier 'segmentations' est créé dans le dossier 'outputs' à la racine."
    )
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        default="MitoNet_v1",
        help="Nom du modèle MitoNet à utiliser pour la segmentation."
    )
    parser.add_argument(
        "-d", "--downsampling",
        type=int,
        default=1,
        help="Facteur de sous-échantillonnage pour l'inférence (1 = pas de sous-échantillonnage)."
    )
    parser.add_argument(
        "-q", "--use_quantized",
        action="store_true",
        help="Utiliser le modèle quantifié pour l'inférence (plus rapide mais moins précis)."
    )

    return parser.parse_args()


def load_model(model_name, downsampling=1, use_quantized=True):
    """Charge la configuration et initialise MitoNet."""
    configs = get_configs()

    if model_name not in configs:
        raise RuntimeError(
            f"Configuration {model_name!r} introuvable. "
            f"Modèles disponibles : {list(configs)}"
        )

    config = read_yaml(configs[model_name])

    print(f"Chargement du modèle {model_name}...")
    print("Le téléchargement des poids peut prendre quelques minutes.")

    model = Engine2d(
        model_config=config,
        inference_scale=downsampling,
        label_divisor=1000,
        nms_threshold=0.1,
        nms_kernel=3,
        confidence_thr=0.5,
        use_gpu=True if torch.cuda.is_available() else False,
        use_quantized=use_quantized,
    )

    return model


def segment_image(model, image):
    """Segmente une image 2D et renvoie un masque de labels."""
    prediction = model.infer(image)

    # Renumérote les labels de 1 à N ; 0 reste le fond.
    labels = np.unique(prediction)
    labels = labels[labels != 0]

    mask = np.zeros(prediction.shape, dtype=np.uint32)

    for new_label, old_label in enumerate(labels, start=1):
        mask[prediction == old_label] = new_label

    return mask


def _main():
    args = getArgs()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = list(sorted(file for pattern in ("*.dm3", "*.tif", "*.tiff") 
                            for file in args.input_dir.glob(pattern)))

    if not files:
        raise FileNotFoundError(f"Aucun fichier DM3 ou TIFF trouvé dans {args.input_dir}")

    print(f"{len(files)} image(s) trouvée(s).")
    print(f"PyTorch : {torch.__version__}")
    print("Calcul sur CPU.")

    model = load_model(
        model_name=args.model_name,
        downsampling=args.downsampling,
        use_quantized=args.use_quantized
    )

    for index, path in tqdm(enumerate(files, start=1)):
        print(f"\n[{index}/{len(files)}] {path.name}")

        # Charge l'image DM3 ou TIFF
        image = preprocessing.load_image(path) 
        image = preprocessing.convert_to_8bit_v1(image)

        print(
            f"Dimensions : {image.shape} | "
            f"type : {image.dtype} | "
            f"intensité : [{image.min()}, {image.max()}]"
        )

        mask = segment_image(model, image)

        output_path = args.output_dir / f"{path.stem}_seg.tif"
        tifffile.imwrite(output_path,(mask > 0).astype(np.uint8) * 255)

        n_instances = len(np.unique(mask)) - 1

        print(f"Instances détectées : {n_instances}")
        print(f"Résultat enregistré : {output_path}")

    print("\nSegmentation terminée.")


if __name__ == "__main__":
    _main()
