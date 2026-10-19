"""
Script pour convertir des fichiers .mod en fichiers TIFF.
"""

from utils.preprocessing import down_sample
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

def getArgs():
    parser = ArgumentParser(
        description="Convertit des fichiers .mod en fichiers TIFF."
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=Path,
        default=Path("./data/dm3"),
        help="Dossier contenant les fichiers .mod à convertir."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=Path("./data/dm3_red"),
        help="Dossier de sortie pour les fichiers TIFF convertis."
    )
    parser.add_argument(
        "d", "--downsampling",
        type=int,
        default=2,
        help="Facteur de sous-échantillonnage pour l'inférence (1 = pas de sous-échantillonnage)."
    )
    parser.add_argument(
        "-e", "--extension",
        type=str,
        default=".tif",
        help="Extension des fichiers à traiter (par défaut : .tif)."
    )


    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()

    input_dir = args.input_dir
    output_dir = args.output_dir

    files = list(input_dir.glob(f"*.{args.extension}"))

    output_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(files):
        down_sample(path, output_path=output_dir / f"{path.stem}.tif", downsampling=args.downsampling)