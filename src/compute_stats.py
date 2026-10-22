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
        default="",
        help="Dossier de sortie pour les résultats."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    pred_folder = args.pred_folder
    ref_folder = args.ref_folder
    output_folder = args.output_folder

    (output_folder / Path("stats")).mkdir(parents=True, exist_ok=True)

    totals = {
        "IoU": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "FPR" : 0.0,
        "Dice": 0.0
    }

    with open(output_folder /f"stats/{pred_folder.name}.csv", mode="w", newline="") as csv_file:
        csv.writer = csv.writer(csv_file)
        csv.writer.writerow(["File", "IoU", "Precision", "Recall", "FPR", "Dice Score"])

        for pred_file in tqdm(pred_folder.glob("*.tif")):
            print(f"Traitement du fichier : {pred_file.name}")
            ref_file = list(ref_folder.glob(f"{pred_file.stem.replace('_pred', '').replace('_seg', '')}*"))
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

            stats_summary = stats.calculate_metrics(mask_pred, mask_ref)
            totals["IoU"] += stats_summary["IoU"]
            totals["Precision"] += stats_summary["Precision"]
            totals["Recall"] += stats_summary["Recall"]
            totals["FPR"] += stats_summary["FPR"]
            totals["Dice"] += stats_summary["Dice"]

            print(f"{pred_file.name} : IoU : {stats_summary['IoU']} |  Precision : {stats_summary['Precision']} | Recall : {stats_summary['Recall']} | FPR : {stats_summary['FPR']} | Dice : {stats_summary['Dice']} \n")
            csv.writer.writerow([pred_file.name,
                                 totals["IoU"], 
                                 totals["Precision"],
                                 totals["Recall"],
                                 totals["FPR"],
                                 totals["Dice"]])
            
    total = len(list(pred_folder.glob("*.tif")))
    print(f"Moyennes : IoU : {totals['IoU'] / total} |  Precision : {totals['Precision'] / total} | Recall : {totals['Recall'] / total} | FPR : {totals['FPR'] / total} | Dice : {totals['Dice'] / total} ")
