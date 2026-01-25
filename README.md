# 🔆 LED Voltage Calculation Tools

---

## 🚀 Project Overview

Ce projet permet de calculer les voltages nécessaires pour piloter des LEDs en fonction de cibles de puissance ou d'isomérisation. Il automatise le lien entre mesures physiques, spectres d'émission et commandes hardware.

**Outils principaux :**
* **Isomerisation_to_voltage.py** : Calcul basé sur les cibles d'isomérisation (S-cones, M-cones, Rods, Melanopsin).
* **PowerList_to_voltage.py** : Conversion massive d'un fichier `.txt` de puissances vers un `.csv` de tensions.

---


## 📦 Installation (Conda)

Le projet nécessite un environnement Python 3.10 géré par Conda pour la stabilité des bibliothèques scientifiques et graphiques.

### 1. Créer l'environnement
Utilisez le fichier `environment.yml` :
```bash
conda env create -f environment.yml
```

### 2. Contenu de environment.yml
```yaml
name: led_calib_env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - openpyxl
  - matplotlib
  - tqdm
  - plotly
  - pillow
  - scipy
  - tk
```

---

## 🏗️ Utilisation & Lancement

### Lancement Automatisé
Utilisez les scripts fournis pour activer l'environnement et lancer le programme :
* **Windows** : `run_script.bat`
* **Linux** : `run_script.sh` (faire `chmod +x run_script.sh` au préalable)

### Phase de Correction
Au démarrage, le programme affiche l'heure de la dernière modification enregistrée.
* **Pour modifier** : Tapez la nouvelle valeur en mW et validez.
* **Pour conserver** : Appuyez sur **Entrée** sans rien taper.
* **Horodatage** : La date `Last updated` dans `last_correction.txt` ne change **que si** une valeur numérique est réellement modifiée.

---

## 🔬 Logique de Traitement

1.  **Correction Temps Réel** : Ajustement via une mesure à 5V en sortie de fibre. Si aucune valeur n'est saisie, le système utilise les données par défaut de la dernière correction ou, à defaut, de l'Excel.
2.  **Ratio de Transformation** : Calcul dynamique du ratio (mW -> µW/cm²) incluant les atténuations et transformations du microscope (repose uniquement sur le fichier excel de calibration).
3.  **Visualisation Grid** : Pour chaque LED, affichage côte à côte du spectre d'émission (`.pkl`) et de la courbe de calibration calculée.
4.  **Interpolation Linéaire** : Inversion de la courbe de puissance pour trouver la tension exacte via `np.interp`.

---
## 📂 Structure du Projet

* `PowerList_to_voltage.py` / `Isomerisation_to_voltage.py` : Scripts de haut niveau.
* `led_controllers_utils.py` : Fonctions de calcul, parsing Excel et gestion des corrections.
* `colors_utils.py` : Gestion des spectres et des fichiers Pickle.
* `IlluminationData.csv` : Données spectrales des LEDs. (Save a copy as a pkl file for color_utils to work properly while allowing easy modification of leds spectrums)
* `PhotoReceptorData.pkl` : Sensibilités spectrales des opsines.
* `last_correction.txt` : Historique et valeurs de correction persistantes.

---

## 📝 Sortie de données
Le fichier `VoltageList.csv` généré est au format "raw" (sans header). Chaque ligne correspond aux tensions à appliquer simultanément sur les différentes LEDs pour chaque état demandé.
