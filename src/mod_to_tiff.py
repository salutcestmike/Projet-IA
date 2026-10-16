"""
Script pour convertir des fichiers .mod en fichiers TIFF.
"""

from utils.preprocessing import mod_to_tiff
from pathlib import Path
from argparse import ArgumentParser

def getArgs():
    parser = ArgumentParser(
        description="Convertit des fichiers .mod en fichiers TIFF."
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=Path,
        default=Path("./data/mod"),
        help="Dossier contenant les fichiers .mod à convertir."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=Path("./data/mod_tiff"),
        help="Dossier de sortie pour les fichiers TIFF convertis."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()

    input_dir = args.input_dir
    output_dir = args.output_dir

    files = list(input_dir.glob("*.mod"))

    output_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        mod_to_tiff(path, output_tiff_path=output_dir / f"{path.stem}_mod.tif")