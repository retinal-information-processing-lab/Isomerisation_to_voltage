#!/bin/bash
ENV_NAME="led_calib_env"

# Initialisation de conda pour les scripts shell (indispensable)
eval "$(conda shell.bash hook)"

echo "[INFO] Activation de l'environnement $ENV_NAME..."
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "[ERREUR] Impossible d'activer l'environnement. Verifie qu'il est bien cree."
    read -p "Pressez Entree pour quitter."
    exit 1
fi

echo "[INFO] Lancement de PowerList_to_voltage..."
python3 PowerList_to_voltage.py

# Garde le terminal ouvert si le script Python plante
if [ $? -ne 0 ]; then
    echo "[ERREUR] Le script s'est arrete prematurement."
    read -p "Pressez Entree pour quitter."
fi
