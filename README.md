# Projet IA — Segmentation des mitochondries 

## Prérequis

- Créer un environnement Python avec les dépendances du projet installées : notamment `numpy`, `tifffile`, `matplotlib`, `scipy`, `scikit-image`, `ncempy`, `torch`, `empanada`, `empanada-napari` et `imodmodel`. 
- Lancer les commandes depuis la racine du projet.


## 1. Segmenter les images avec `mitonet2D.py`

Le script traite toutes les images `.dm3`, `.tif` et `.tiff` présentes directement dans le dossier d’entrée. Il enregistre pour chaque image un masque binaire TIFF : fond noir (`0`), objets segmentés blancs (`255`).

```bash
python -m src.mitonet2D -i data -o outputs/segmentations
```

Options disponibles :

- `-i`, `--input_dir` : dossier des images ; par défaut `./data`.
- `-o`, `--output_dir` : dossier de sortie des masques (**à préciser**).
- `-m`, `--model_name` : modèle à utiliser ; par défaut `MitoNet_v1`.
- `-d`, `--downsampling` : facteur d’échelle pour l’inférence ; par défaut `1`.
- `-q`, `--use_quantized` : utilise le modèle quantifié, potentiellement plus rapide mais moins précis.

Exemple avec le modèle quantifié :

```bash
python -m src.mitonet2D \
  -i data \
  -o outputs/segmentations \
  -m MitoNet_v1 \
  -d 1 \
  -q
```

Les fichiers de sortie sont nommés `<nom_image>_seg.tif`. Le script effectue l’inférence sur CPU dans sa configuration actuelle. Le chargement initial du modèle peut prendre quelques minutes.

## 2. Comparer les filtres avec `visualize_filters.py`

Il est intéressant d'appliquer des filtres en prétraitement pour améliorer la détection des mitochondries. Ce script affiche l’image originale convertie en 8 bits et plusieurs variantes filtrées : filtre médian, gaussien, CLAHE, bilatéral, Non-Local Means et gaussien + CLAHE (à revérifier).

```bash
python -m src.utils.visualize_filters -i chemin/vers/image.dm3
```

Les formats `.dm3`, `.tif` et `.tiff` sont pris en charge. Pour enregistrer les images filtrées dans un dossier :

```bash
python -m src.utils.visualize_filters \
  -i chemin/vers/image.tif \
  -o outputs/filtres
```

## Remarques

- Les scripts ne parcourent pas les sous-dossiers : les images doivent être directement dans le dossier indiqué par `-i`.
- `visualize_filters` sert à examiner l’effet des filtres ; la segmentation MitoNet n’applique pas automatiquement ces filtres avant l’inférence.
- Vérifier visuellement les masques produits avant de les utiliser pour des analyses quantitatives.
