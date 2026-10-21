#!/usr/bin/env bash

#SBATCH --job-name=projet_ia_maia
#SBATCH --partition=master-bi
#SBATCH --account=master-bi
#SBATCH --output=logs/projet_ia_maia_%j.out
#SBATCH --error=logs/projet_ia_maia_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Arrêter en cas d'erreur
set -e

# Se placer dans le répertoire depuis lequel le job a été soumis
cd "$SLURM_SUBMIT_DIR"

# Charger Conda
source /opt/conda/etc/profile.d/conda.sh

# Activer l'environnement
conda activate projet_ia

# Afficher les informations de calcul
echo "Job ID : $SLURM_JOB_ID"
echo "Nœud : $SLURMD_NODENAME"
echo "Date : $(date)"
echo "Python : $(which python)"
python --version

# Exécuter le script UNet
python src/unet_keras_v6.py

echo "Fin du job : $(date)"