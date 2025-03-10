# 🔆 LED Voltage Calculation from Isomerisation

## 🚀 Project Overview

This project calculates the required LED voltage based on user-defined **isomerisation** values. The program enables users to:

- Load the latest **LED voltage calibration** data.
- Apply **correction factors** based on current-day LED power measurements (measured at the **optic fiber output** with a **5V output from an Arduino**).
- Compute the required **LED voltage** based on user-specified **isomerisation targets**.

### 🔬 How It Works

The program adjusts LED calibration using:

- 📊 The latest **calibration data** (typically stored in an **Excel file**).
- 🛠 User inputs for **today’s correction factors** and **filter values** (transmittance, **not ND**; see the **filters measurement sheet**).
  - If filter values are unknown, you can fine-tune the **"Direct at 5V"** instead.
  - Set **"Direct at 5V = True"** if filters are **after** the optic fiber output. Otherwise, either a **full calibration** is needed, or the **adjusting factor** will be larger.
- 🔆 **Isomerisation targets** for different **opsins** (e.g., "S-cones", "M-cones", "Rods", "Melanopsin").

The program outputs the necessary **LED voltage values** based on the required **isomerisation** and the corresponding **calibration data**.

---

## 📦 Installation & Dependencies

### ✅ Prerequisites
- **Python 3.9.16** (same version as required for the **analysis pipeline**).

### 📌 Install Required Packages
Use the following `requirements.txt` file to install dependencies:

```plaintext
openpyxl
numpy
matplotlib
tqdm
tkinter
pickle
itertools
plotly
Pillow
scipy
pyinstaller
```


Install them via pip:
```bash
pip install -r requirements.txt
```

> **Note:** This uses the same `color_utils` as the color pipeline.

---

## 🏗️ Building the Executable

To package the script into a standalone executable, run the following command in your folder with active env:

```bash
pyinstaller --onefile --console --distpath . \
  --add-data "PhotoReceptorData.pkl;." \
  --add-data "IlluminationData.pkl;." \
  --add-data "colors_utils.py;." \
  Isomerisation_to_voltage.py
```
pyinstaller --onefile --console --distpath . --add-data "PhotoReceptorData.pkl;." --add-data "IlluminationData.pkl;." --add-data "colors_utils.py;." Isomerisation_to_voltage.py
  
  
Same for PowerList_to_voltage

```bash
pyinstaller --onefile --console --distpath . \
  --add-data "PhotoReceptorData.pkl;." \
  --add-data "IlluminationData.pkl;." \
  --add-data "colors_utils.py;." \
  PowerList_to_voltage.py
```
pyinstaller --onefile --console --distpath . --add-data "PhotoReceptorData.pkl;." --add-data "IlluminationData.pkl;." --add-data "colors_utils.py;." PowerList_to_voltage.py
  
  
### 🔧 Explanation of Parameters:
- `--onefile`: Bundles everything into a single executable.
- `--console`: Ensures a console appears when the program runs.
- `--add-data`: Includes required data files (adjust paths as needed).
  - On **Windows**, use `;` to separate paths.
  - On **Linux**, use `:` if `;` does not work.
- `--distpath .`: Places the executable in the current directory.

## Running the Program

Once you’ve built the executable, run it by following the prompts in the console window. The program will:

- Ask the user to select the calibration file using a file dialog.
- Ask the user to input correction factors for LEDs. This is today's measurement of each LED at 5V (use Arduino, not manually) AT THE OPTIC FIBER OUTPUT!
- Prompt the user for target isomerisation values for different opsins (e.g., "Scones", "Mcones", "Rods", "Mela").
- Display plots of the calibration data and isomerisation, and calculate the required LED voltages for the provided isomerisation values.

### Notes
- Make sure that the data files (PhotoReceptorData.pkl, IlluminationData.pkl, colors_utils.py) are present in the same directory as the executable or properly included during packaging (via --add-data).
- You can create a shortcut to this executable to have a clean portable executable.
- On Linux, you need to create a bash file to execute the program or run it from a terminal with `./Isomerisation_to_voltage`.







