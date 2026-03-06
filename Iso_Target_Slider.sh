#!/bin/bash
ENV_NAME="led_calib_env"

# 1. On définit le chemin vers l'installation de conda
# (Vérifie ce chemin en tapant 'conda info --base' dans ton terminal)
CONDA_PATH="$HOME/anaconda3"

# 2. On charge les fonctions shell de conda
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
else
    echo "Erreur : conda.sh non trouvé dans $CONDA_PATH"
    exit 1
fi

# 3. Maintenant utiliser conda
conda activate $ENV_NAME

echo "[INFO] Lancement de Iso_Target_Slider..."
python3 Iso_Target_Slider.py

# Garde le terminal ouvert si le script Python plante
if [ $? -ne 0 ]; then
    echo "[ERREUR] Le script s'est arrete prematurement."
    read -p "Pressez Entree pour quitter."
fi
