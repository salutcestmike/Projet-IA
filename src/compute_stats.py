"""
Script qui calcule plusieurs métriques statistiques entre plusieurs paires de masques binaires.
"""

import utils.stats as stats 
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
import tifffile
import csv
import utils.preprocessing as preprocessing
from tqdm import tqdm


def getArgs():
    parser = ArgumentParser(
        description="Calcule plusieurs métriques statistiques entre plusieurs paires de masques binaires."
    )
    parser.add_argument(
        "-p", "--pred_folder",
        type=Path,
        help="Dossier contenant les masques prédits."
    )
    parser.add_argument(
        "-r", "--ref_folder",
        type=Path,
        help="Dossier contenant les masques de référence."
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=Path,
        help="Dossier de sortie pour les résultats."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    pred_folder = args.pred_folder
    ref_folder = args.ref_folder
    output_folder = args.output_folder

    (output_folder / Path("stats")).mkdir(parents=True, exist_ok=True)

    with open(output_folder /f"stats/{pred_folder.name}.csv", mode="w", newline="") as csv_file:
        csv.writer = csv.writer(csv_file)
        csv.writer.writerow(["File", "Dice Score"])

        for pred_file in tqdm(pred_folder.glob("*.tif")):
            print(f"Traitement du fichier : {pred_file.name}")
            ref_file = list(ref_folder.glob(f"{pred_file.stem.replace('_seg', '')}*"))
            if not ref_file or len(ref_file) == 0:
                print(f"Fichier de référence manquant pour {pred_file.name}.")
                continue

            ref_file_path = ref_file[0] 
            if ref_file_path.suffix.lower() == ".mod":
                # Convertir le fichier .mod en masque TIFF binaire
                mask_ref = preprocessing.mod_to_tiff(ref_file_path) > 0
            else:
                mask_ref = preprocessing.load_image(ref_file_path) > 0
            mask_pred = np.array(tifffile.imread(pred_file)) > 0

            dice = stats.dice_coef(mask_pred, mask_ref)
            print(f"Coefficient de Dice pour {pred_file.name} : {dice}")
            csv.writer.writerow([pred_file.name, dice])
