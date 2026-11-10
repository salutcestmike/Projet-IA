# Projet IA — Segmentation des mitochondries

Ce projet segmente automatiquement les mitochondries dans des images de microscopie électronique (`.dm3`, `.tif`, `.tiff`) de deux façons :

- avec le modèle pré-entraîné **MitoNet** (bibliothèque [empanada](https://github.com/volume-em/empanada-napari)) ;
- avec un **U-Net** entraîné sur nos propres images (Keras/TensorFlow).

Les résultats sont comparés à des annotations manuelles faites sous **IMOD** (`.mod`) à l'aide de plusieurs métriques (Dice, IoU, précision, rappel, FPR).

## Sommaire

1. [Structure du projet](#structure-du-projet)
2. [Prérequis](#prérequis)
3. [Organisation des données](#organisation-des-données)
4. [Chaîne de traitement](#chaîne-de-traitement)
   - [1. Convertir les annotations `.mod` en TIFF](#1-convertir-les-annotations-mod-en-tiff-avec-mod_to_tiffpy)
   - [2. Segmenter avec MitoNet](#2-segmenter-avec-mitonet-avec-mitonet2dpy)
   - [3. Segmenter avec le U-Net](#3-segmenter-avec-le-u-net-avec-unet_keras_v4py)
   - [4. Évaluer les segmentations](#4-évaluer-les-segmentations-avec-compute_statspy)
6. [Remarques et limitations connues](#remarques-et-limitations-connues)

## Structure du projet

```text
Projet-IA/
├── README.md
├── .gitignore                  # ignore data/, outputs, logs, __pycache__ et les fichiers commençant par _
├── unet_job.slurm              # script de soumission SLURM pour l'entraînement du U-Net
├── data/                       # images et annotations (non versionné, à créer)
├── outputs/                    # résultats (masques, modèles, statistiques)
└── src/
    ├── mitonet2D.py            # segmentation avec MitoNet 2D
    ├── unet_keras.py           # entraînement, évaluation et prédiction d'un U-Net
    ├── compute_stats.py        # métriques entre prédictions et références
    ├── mod_to_tiff.py          # conversion en lot des fichiers .mod
    ├── downsample.py           # réduction de la résolution des images
    └── utils/
        ├── preprocessing.py    # chargement, normalisation, conversion .mod -> masque
        └── stats.py            # Dice, IoU, précision, rappel, métrique Keras
```

## Prérequis

- Créer un environnement Python avec les dépendances du projet installées, notamment :
  - avec conda (option `-c conda-forge`) : `numpy`, `tifffile`, `matplotlib`, `scipy`, `scikit-learn`, `imodmodel`, `ncempy`, `tqdm`, `keras`, `tensorflow` ;
  - avec pip : `empanada-napari` et `scikit-image`.
- Lancer **toutes** les commandes depuis la racine du projet. Les scripts s'exécutent directement par leur chemin, avec `python src/<script>.py` (et non `python -m src.<script>`) : `src` n'est pas un paquet et les modules sont importés depuis le dossier du script (`import utils.preprocessing`).
- Une connexion internet est nécessaire au premier lancement de `mitonet2D` pour télécharger les poids du modèle.
- Un GPU est fortement conseillé pour entraîner le U-Net.
- Si des conflits de paquets sont présents lors de l'installation, créer un autre environnement distinct pour empanada-napari est recommandé.

Exemple d'installation :

```bash
conda create -n projet_ia python=3.10
conda activate projet_ia
conda install -c conda-forge numpy tifffile matplotlib scipy scikit-learn ncempy tqdm imodmodel
conda install -c conda-forge tensorflow keras
pip install empanada-napari scikit-image
```

## Organisation des données

Le dossier `data/` n'est pas versionné (cf. `.gitignore`). Organisation attendue :

```text
data/
├── dm3/                        # images brutes (.dm3, .tif ou .tiff)
│   ├── image_01.dm3
│   ├── image_02.tif
│   └── ...
├── mod/                        # annotations IMOD
│   ├── image_01.mod
│   ├── image_02.mod
│   └── ...
└── mod_tiff/                   # (généré) masques de référence
    ├── image_01.tif
    └── ...
```

Les fichiers d'annotation doivent porter le **même nom de base** que l'image correspondante (par exemple `image_01.dm3` ↔ `image_01.mod`) : c'est ainsi que `compute_stats.py` et `unet_keras.py` apparient images et masques.

## Chaîne de traitement

```text
                                   ┌──► mitonet2D ──► masques (*_seg.tif) ──┐
images .dm3/.tif ──────────────────┤                                        ├──► compute_stats ──► métriques (CSV)
                                   └──► unet_keras ──► masques (*_pred.tif) ┤
annotations .mod ──► mod_to_tiff ──► masques de référence ───────────────────┘
```

### 1. Convertir les annotations `.mod` en TIFF avec `mod_to_tiff.py`

Les annotations manuelles réalisées sous IMOD sont converties en masques binaires TIFF (`0` / `255`). Ces masques servent de **vérité terrain** pour l'entraînement du U-Net et pour l'évaluation. Seuls les objets IMOD nommés `mitochondria` (insensible à la casse) sont pris en compte ; chaque contour est rempli comme un polygone.

```bash
python src/mod_to_tiff.py -i data/mod -o data/mod_tiff
```

Options :

- `-i`, `--input_dir` : dossier des fichiers `.mod` ; par défaut `./data/mod`.
- `-o`, `--output_dir` : dossier de sortie ; par défaut `./data/mod_tiff`.

Les fichiers de sortie sont nommés `<nom>.tif`. Si aucun objet `mitochondria` n'existe dans un modèle, une erreur est levée.

### 2. Segmenter avec MitoNet avec `mitonet2D.py`

Le script traite toutes les images `.dm3`, `.tif` et `.tiff` présentes directement dans le dossier d'entrée. Chaque image est normalisée (percentiles 1 et 99), segmentée, puis enregistrée sous forme de masque binaire TIFF : fond noir (`0`), objets segmentés blancs (`255`).

```bash
python src/mitonet2D.py -i data/dm3 -o outputs/segmentations
```

Options :

- `-i`, `--input_dir` : dossier des images ; par défaut `./data`.
- `-o`, `--output_dir` : dossier de sortie des masques (**obligatoire en pratique** : sans cette option, le script s'arrête avec une erreur).
- `-m`, `--model_name` : modèle à utiliser ; par défaut `MitoNet_v1`.
- `-d`, `--downsampling` : facteur d'échelle pour l'inférence ; par défaut `1` (pas de sous-échantillonnage).
- `-q`, `--use_quantized` : utilise le modèle quantifié, potentiellement plus rapide mais moins précis.

Exemple avec le modèle quantifié :

```bash
python src/mitonet2D.py \
  -i data/dm3 \
  -o outputs/segmentations \
  -m MitoNet_v1 \
  -d 1 \
  -q
```

Les fichiers de sortie sont nommés `<nom_image>_seg.tif`. Le chargement initial du modèle peut prendre quelques minutes.

### 3. Segmenter avec le U-Net avec `unet_keras.py`

#### Description

Exemple d'U-Net : `unet_keras_template.py` entraîne un **U-Net 2D** (Keras/TensorFlow) pour la segmentation binaire des mitochondries (fond / mitochondrie), puis l'évalue et produit des prédictions. Un seul lancement exécute tout le pipeline, avec la possibilité de charger un modèle :

1. **Appariement** des images de `data/dm3_tiff` et des masques de `data/mod_tiff` par nom de base (les fichiers sans correspondant sont ignorés, avec un avertissement).
2. **Prétraitement** : chargement, normalisation par les percentiles 1 et 99, redimensionnement à 512 × 512 (interpolation antialiasée pour les images, plus proche voisin pour les masques). Paramètres modifiables.
3. **Découpage** aléatoire (seed `0`) en **70 % entraînement et 30 % test**. Les indices sont sauvegardés dans `train_test_splits.csv`.
4. **Augmentation** (entraînement uniquement) : flips horizontaux et verticaux, rotations de 0°/90°/180°/270°, légère variation de contraste et de luminosité, appliqués de façon identique à l'image et au masque.
5. **Entraînement cross validation à 5 folds** avec :
   - une architecture U-Net à 4 niveaux (8 filtres au premier niveau, doublés à chaque niveau), avec du dropout (0,3) dans les deux niveaux les plus profonds, et une sortie softmax à 2 classes ;
   - une loss combinée **cross entropy + loss de Dice** (`ce_dice_loss`) ;
   - l'optimiseur Adam, et les métriques Accuracy et Dice ;
   - trois callbacks : Early Stopping (patience de 8 epochs sur `val_loss`, restauration des meilleurs poids), sauvegarde du meilleur modèle et réduction du taux d'apprentissage lorsque le modèle stagne (facteur 0,5, patience de 3).
6. **Évaluation** sur le jeu de test (Accuracy, Dice) puis **prédiction** de chaque image (seuil à 0,5) pour les images de test et pour toutes les images.
7. **Courbes d'apprentissage** (Loss et Dice, entraînement et validation) enregistrées en PNG.

### Options disponibles

| Option | Valeur par défaut | Description |
|---|---|---|
| `-i`, `--input_dir` | `./data/dm3_downsample_8` | Dossier des images |
| `-m`, `--target_dir` | `./data/mod_tiff_downsample_8` | Dossier des masques |
| `-p`, `--splits` | `None` | CSV de split existant |
| `-o`, `--output_dir` | `./outputs/run_default` | Dossier de sortie |
| `-b`, `--batch_size` | `4` | Taille des batches |
| `-s`, `--seed` | `0` | Graine aléatoire |
| `-e`, `--max_epochs` | `100` | Nombre maximal d'epochs par fold |
| `-t`, `--test_size` | `0.3` | Proportion réservée au test lors de la création du split |
| `-l`, `--load_model` | `None` | Chemin vers le modèle à charger à la place de l'entraînement |

Afficher l'aide :

```bash
python unet_keras_template.py --help
```

Si aucun chemin `--splits` n'est fourni, le script utilise `train_test_splits.csv`. Si ce fichier n'existe pas, une nouvelle séparation est créée et enregistrée à cet emplacement. Si un CSV est fourni et existe, il est réutilisé.

Préparer d'abord les masques (étape 1), puis lancer depuis la racine du projet.

### Exemple avec des paramètres personnalisés

```bash
python src/unet_keras_template.py \
    --input_dir ./data/images \
    --target_dir ./data/masks \
    --output_dir ./outputs/experiment_01 \
    --batch_size 4 \
    --seed 0 \
    --max_epochs 100 \
    --splits ./train_test_splits.csv \
    --test_size 0.3
```

**Sur le cluster (SLURM)**, le script `unet_job.slurm` réserve 1 GPU, 16 CPU et 64 Go de mémoire pendant 10 h maximum. Créer le dossier de logs avant de soumettre le job :

```bash
mkdir -p logs
sbatch unet_job.slurm
```

Les sorties standard et d'erreur sont écrites dans `logs/projet_ia_maia_<id>.out` et `.err`. Adapter la ligne `python src/...` du fichier `.slurm`.

#### Résultats

Dans `output_dir` (`outputs/run_default/` par défaut) :

```text
outputs/run_default/
├── unet_final.keras                 # modèle final
├── unet_fold_<i>.keras              # modèle du fold <i> de cross validation
├── cross_validation_curves.png      # courbes de Loss et de Dice
├── predictions_test/                # prédictions des images de TEST   : <nom>_pred.tif
└── predictions_all/                 # prédictions de TOUTES les images : <nom>_pred.tif
```

Les masques prédits sont binaires (`0` / `255`). Le script affiche le Dice de chaque image et le Dice moyen. Pour obtenir toutes les métriques, utiliser ensuite `compute_stats.py` (étape 4) sur `predictions_test/` (images de test uniquement, à privilégier pour juger le modèle).

> `predictions_all/` inclut les images d'entraînement et de test : ses scores sont optimistes.

### 4. Évaluer les segmentations avec `compute_stats.py`

Le script compare chaque masque prédit à son masque de référence et calcule plusieurs métriques sur le masque binaire :

| Métrique | Définition |
| --- | --- |
| Dice | `2·TP / (2·TP + FP + FN)` |
| IoU | `TP / (TP + FP + FN)` |
| Précision | `TP / (TP + FP)` |
| Rappel | `TP / (TP + FN)` |
| FPR | `FP / (FP + TN)` |

où `TP`, `FP`, `FN` et `TN` sont les nombres de pixels vrais positifs, faux positifs, faux négatifs et vrais négatifs. Le Dice vaut `1` si les deux masques sont vides.

```bash
python src/compute_stats.py \
  -p outputs/segmentations \
  -r data/mod_tiff \
  -o outputs
```

Options :

- `-p`, `--pred_folder` : dossier des masques prédits (fichiers `*.tif`).
- `-r`, `--ref_folder` : dossier des masques de référence. Il peut contenir des masques TIFF **ou directement des fichiers `.mod`**, qui sont alors convertis à la volée.
- `-o`, `--output_folder` : dossier de sortie ; par défaut le dossier courant.

Pour chaque `<nom>_seg.tif` ou `<nom>_pred.tif`, le script cherche dans le dossier de référence un fichier dont le nom commence par `<nom>`. Les prédictions sans référence sont ignorées avec un message. Les moyennes de chaque métrique sont affichées à la fin.

Résultat : un fichier CSV `outputs/stats/<nom_du_dossier_de_prédictions>.csv` avec les colonnes `File`, `IoU`, `Precision`, `Recall`, `FPR` et `Dice Score`.

Exemple pour évaluer le U-Net sur les images de test :

```bash
python src/compute_stats.py \
  -p outputs/run_default/predictions_test \
  -r data/mod_tiff \
  -o outputs
```


## Remarques et limitations connues

- Les scripts ne parcourent pas les sous-dossiers : les images doivent être directement dans le dossier indiqué.
- Les masques prédits et de référence doivent avoir la **même taille** pour que les métriques soient calculables. 
- Dans `compute_stats.py`, le dossier de référence ne doit contenir que des masques : si une image brute portant le même nom de base s'y trouve, elle risque d'être appariée par erreur.
- Vérifier visuellement les masques produits avant de les utiliser pour des analyses quantitatives.
