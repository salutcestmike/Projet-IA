# Projet IA — Segmentation des mitochondries

## Sommaire

1. [Structure du projet](#structure-du-projet)
2. [Prérequis](#prérequis)
3. [Organisation des données](#organisation-des-données)
4. [Chaîne de traitement](#chaîne-de-traitement)
   - [1. Segmenter les images](#1-segmenter-les-images-avec-mitonet2dpy)
   - [2. Comparer les filtres](#2-comparer-les-filtres-avec-visualize_filterspy)
   - [3. Convertir les annotations `.mod` en TIFF](#3-convertir-les-annotations-mod-en-tiff-avec-mod_to_tiffpy)
   - [4. Évaluer les segmentations](#4-évaluer-les-segmentations-avec-compute_statspy)
5. [Détails techniques](#détails-techniques)
6. [Remarques et limitations connues](#remarques-et-limitations-connues)

## Structure du projet

```text
Projet-IA/
├── README.md
├── .gitignore                  # ignore data/, output et __pycache__
├── data/                       # images et annotations (non versionné, à créer)
├── outputs/                    # résultats (masques, statistiques) à créer
└── src/
    ├── mitonet2D.py            # segmentation MitoNet 2D
    ├── compute_stats.py        # calcul du Dice entre prédictions et références
    ├── mod_to_tiff.py          # conversion en lot des fichiers .mod
    ├── visualize_filters.py    # comparaison visuelle de filtres de prétraitement
    └── utils/
        ├── preprocessing.py    # chargement, conversion 8 bits, conversion .mod -> masque
        └── stats.py            # coefficient de Dice
```

## Prérequis

- Créer un environnement Python avec les dépendances du projet installées, notamment :
  - avec conda : `numpy`, `tifffile`, `matplotlib`, `scipy`, `scikit-image`, `ncempy`, `tqdm`, `pytorch`,
  - avec pip : `empanada`, `empanada-napari` et `imodmodel`.
- Lancer **toutes** les commandes depuis la racine du projet. Les scripts s'exécutent directement par leur chemin, avec `python src/<script>.py`
- Une connexion internet est nécessaire au premier lancement de `mitonet2D` pour télécharger les poids du modèle.

Exemple d'installation :

```bash
conda create -n projet-ia python=3.10
conda activate projet-ia
conda install -c conda-forge numpy tifffile matplotlib scipy scikit-image ncempy tqdm
conda install pytorch -c pytorch
pip install empanada empanada-napari imodmodel
```

## Organisation des données

Le dossier `data/` n'est pas versionné (cf. `.gitignore`). Une organisation possible :

```text
data/
├── dm3/                        # images brutes à segmenter (.dm3, .tif ou .tiff)
├────── image_01.dm3            
├────── image_02.tif
├────── ...
├── mod/                        # annotations IMOD : image_01.mod, image_02.
├────── image_01.mod
├────── image_02.mod                  
├────── ...
└── mod_tiff/                   # (généré) masques de référence : image_01_mod.tif, ...
```

Les fichiers d'annotation doivent porter le **même nom de base** que l'image correspondante (par exemple `image_01.dm3` ↔ `image_01.mod`), car c'est ainsi que `compute_stats.py` apparie prédictions et références.

## Chaîne de traitement

```text
images .dm3/.tif ──► mitonet2D ──► masques prédits (*_seg.tif) ──┐
                                                                  ├──► compute_stats ──► Dice (CSV)
annotations .mod ──► mod_to_tiff ──► masques de référence ────────┘
```

### 1. Segmenter les images avec `mitonet2D.py`

Le script traite toutes les images `.dm3`, `.tif` et `.tiff` présentes directement dans le dossier d'entrée. Il enregistre pour chaque image un masque binaire TIFF : fond noir (`0`), objets segmentés blancs (`255`).

```bash
python src/mitonet2D.py -i data -o outputs/segmentations
```

Options disponibles :

- `-i`, `--input_dir` : dossier des images ; par défaut `./data`.
- `-o`, `--output_dir` : dossier de sortie des masques (**obligatoire en pratique** : sans cette option, le script s'arrête avec une erreur).
- `-m`, `--model_name` : modèle à utiliser ; par défaut `MitoNet_v1`.
- `-d`, `--downsampling` : facteur d'échelle pour l'inférence ; par défaut `1` (pas de sous-échantillonnage).
- `-q`, `--use_quantized` : utilise le modèle quantifié, potentiellement plus rapide mais moins précis.

Exemple avec le modèle quantifié :

```bash
python src/mitonet2D.py \
  -i data \
  -o outputs/segmentations \
  -m MitoNet_v1 \
  -d 1 \
  -q
```

Les fichiers de sortie sont nommés `<nom_image>_seg.tif`. Le chargement initial du modèle peut prendre quelques minutes.

> Les instances sont numérotées en interne (de 1 à N), mais le fichier enregistré est un masque **binaire** : les instances ne sont pas distinguées dans la sortie.

### 2. Comparer les filtres avec `visualize_filters.py`

Il est intéressant d'appliquer des filtres en prétraitement pour améliorer la détection des mitochondries. Ce script affiche l'image originale convertie en 8 bits et plusieurs variantes filtrées :

| Filtre | Paramètres |
| --- | --- |
| Filtre médian | fenêtre 3×3 |
| Filtre gaussien | σ = 0,7 |
| CLAHE | `kernel_size=64`, `clip_limit=0.01` |
| Filtre bilatéral | `sigma_color=0.05`, `sigma_spatial=2` |
| Non-Local Means | `h=0.05`, `patch_size=5`, `patch_distance=6` |
| Gaussien + CLAHE | combinaison des deux (à revérifier) |

```bash
python src/visualize_filters.py -i chemin/vers/image.dm3
```

Les formats `.dm3`, `.tif` et `.tiff` sont pris en charge. Pour enregistrer les images filtrées dans un dossier :

```bash
python src/visualize_filters.py \
  -i chemin/vers/image.tif \
  -o outputs/filtres
```

Les images sont enregistrées au format TIFF, avec un nom dérivé du titre du filtre (par exemple `filtre_médian_3x3.tif`).

### 3. Convertir les annotations `.mod` en TIFF avec `mod_to_tiff.py`

Les annotations manuelles réalisées sous IMOD sont converties en masques binaires TIFF (`0` / `255`) pour servir de référence.

```bash
python src/mod_to_tiff.py -i data/mod -o data/mod_tiff
```

Options :

- `-i`, `--input_dir` : dossier des fichiers `.mod` ; par défaut `./data/mod`.
- `-o`, `--output_dir` : dossier de sortie ; par défaut `./data/mod_tiff`.

Les fichiers de sortie sont nommés `<nom>_mod.tif`.

### 4. Évaluer les segmentations avec `compute_stats.py`

Le script compare chaque masque prédit à son masque de référence et calcule le **coefficient de Dice** :

$$\text{Dice} = \frac{2\,|P \cap R|}{|P| + |R|}$$

où `P` est le masque prédit et `R` le masque de référence. La valeur est comprise entre `0` (aucun recouvrement) et `1` (recouvrement parfait) ; elle vaut `1` si les deux masques sont vides.

```bash
python src/compute_stats.py \
  -p outputs/segmentations \
  -r data/mod_tiff \
  -o outputs
```

Options :

- `-p`, `--pred_folder` : dossier des masques prédits (fichiers `*.tif`).
- `-r`, `--ref_folder` : dossier des masques de référence. Il peut contenir des masques TIFF **ou directement des fichiers `.mod`**, qui sont alors convertis à la volée.
- `-o`, `--output_folder` : dossier de sortie.

Pour chaque `<nom>_seg.tif`, le script cherche dans le dossier de référence un fichier dont le nom commence par `<nom>`. Les prédictions sans référence sont ignorées avec un message.

Résultat : un fichier CSV `outputs/stats/<nom_du_dossier_de_prédictions>.csv` avec les colonnes `File` et `Dice Score`.

## Détails techniques

- **Chargement** (`preprocessing.load_image`) : lit les `.dm3` via `ncempy` et les `.tif`/`.tiff` via `tifffile`, puis convertit en `float32`.
- **Conversion 8 bits** (`preprocessing.py`) :
  - `convert_to_8bit_v1` : division par 256 (suppose des images 16 bits) ; c'est celle utilisée par `mitonet2D.py`.
  - `convert_to_8bit_v2` : normalisation par les percentiles 1 et 99, plus robuste aux valeurs aberrantes.

## Remarques et limitations connues

- Les scripts ne parcourent pas les sous-dossiers : les images doivent être directement dans le dossier indiqué par `-i`.
- `visualize_filters` sert à examiner l'effet des filtres ; la segmentation MitoNet n'applique pas automatiquement ces filtres avant l'inférence.
- Les masques prédits et de référence doivent avoir la **même taille** pour que le Dice soit calculable.