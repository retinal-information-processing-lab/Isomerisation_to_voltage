# LED Stimulus Tools

A set of tools for designing and converting light stimuli for multi-LED systems, bridging isomerization targets, LED power levels, and hardware driving voltages.

---

## Overview

Three programs, each solving a different problem in the stimulus design chain:

| Program | What it does |
|---------|-------------|
| **Ptot_Slider** | Set LED powers (µW/cm²) → see resulting opsin isomerization in real time |
| **Iso_Target_Slider** | Set opsin isomerization targets (R*/s) → solve for the required LED power mix |
| **PowerList_to_Voltage** | Convert a PowerList file (µW/cm²) → VoltageList file (V) using calibration data |

---

## Installation

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

```bash
conda env create -f environment.yml
```

This creates the `led_calib_env` environment with all dependencies (Python 3.10, numpy, scipy, matplotlib, tkinter, pandas, etc.).

> On Linux, also make the launcher scripts executable once:
> ```bash
> chmod +x Ptot_Slider.sh Iso_Target_Slider.sh PowerList_to_Voltage.sh
> ```

---

## Launching the programs

### Double-click (recommended)

| Program | Windows | Linux |
|---------|---------|-------|
| Ptot Slider | `Ptot_Slider.bat` | `Ptot_Slider.sh` |
| Iso Target Slider | `Iso_Target_Slider.bat` | `Iso_Target_Slider.sh` |
| PowerList to Voltage | `PowerList_to_Voltage.bat` | `PowerList_to_Voltage.sh` |

On Linux: right-click the `.sh` file → **Execute as program** (or equivalent in your file manager).

### From terminal

```bash
conda activate led_calib_env
python Ptot_Slider.py
python Iso_Target_Slider.py
python PowerList_to_voltage.py
```

---

## Ptot Slider

**Forward direction: LED powers → isomerization**

Set the power of each LED (µW/cm²) using sliders or by typing directly in the entry fields. The bar chart updates live showing the isomerization contribution of each LED to each opsin (R*/s).

**Interface:**
- **Sliders + entry fields** — one row per LED. Entry fields accept values above the slider maximum.
- **LED checkboxes** (right of each row) — exclude an LED from the calculation without resetting its value.
- **Opsin checkboxes** (above plot) — hide opsins from the bar chart and spectra panel.
- **Bar chart** — stacked bars showing each LED's contribution to total isomerization per opsin.
- **Spectra panel** — opsin sensitivity curves (dashed) overlaid with LED emission fills (opacity ∝ power).
- **Legends** — displayed to the right of the plots.

**Buttons:**
- `Reset (Space)` — restore all sliders to default values.
- `Print values` — open a popup showing LED powers and computed isomerizations.
- `Copy to clipboard` — copy the same summary text.
- `Save plots` — export the current figure (PNG, PDF, SVG).
- `Close` — close the window.

---

## Iso Target Slider

**Inverse direction: isomerization targets → LED power mix**

Set target isomerization rates per opsin (R*/s). The solver finds the LED power mix that best achieves those targets and displays the result live.

**Interface:**
- **Sliders + entry fields** — one row per opsin. Entry fields accept values above the slider maximum.
- **Opsin checkboxes** (right of each row) — exclude an opsin from the solve.
- **Mode selector** — choose the solving strategy (see below). Click `?` for in-app help.
- **LED checkboxes** (above plot) — restrict which LEDs the solver can use.
- **LED power display** — shows the solved power (µW/cm²) for each LED in real time.
- **Bar chart** — achieved isomerization stacked by LED contribution. Dashed lines show the targets.
- **Spectra panel** — same as Ptot Slider, fills show solved power levels.

**Solving modes** (only relevant when more LEDs than opsins):

| Mode | Behaviour |
|------|-----------|
| **Balanced** | Distributes power across all LEDs — no single LED dominates |
| **Sparse** | Uses as few LEDs as possible — most LEDs are driven to zero |
| **Maximize LED** | Pushes the selected LED to its maximum, others compensate |

When fewer LEDs than opsins are active, no exact solution exists and the solver minimises the least-squares error regardless of mode. The system case and residual error are shown next to the mode selector.

**Buttons:**
- `Reset (Space)` — restore all targets to default.
- `Print values` — open a popup showing targets, achieved isomerizations, and solved LED powers.
- `Copy to clipboard` — copy the solved LED powers in **PowerList format** (space-separated floats, one value per LED in order) — ready to paste directly into a PowerList file.
- `Save plots` — export the figure.
- `Close` — close the window.

---

## PowerList to Voltage

**Conversion: LED powers (µW/cm²) → driving voltages (V)**

Interactive command-line tool. Steps:
1. Select a **PowerList file** (`.txt`) via file dialog.
2. Select which **LEDs** to process (numbered list).
3. Enter or confirm the **correction values** — measured power at 5V for each LED (in mW). Press Enter to keep the stored value.
4. Confirm or change the **calibration file** (`.xlsx`).
5. The output **VoltageList** (`.csv`) is saved next to the input file.

**PowerList file format:**

Plain text, one stimulus per line, space- or tab-separated values, one column per LED (no header):

```
10   30   6500
4    9    762
10   9    169
```

**VoltageList output format:**

CSV, same structure, values in volts:

```
0.823400,1.204500,4.981200
0.312100,0.756300,3.201400
```

---

## Data files

| File | Description |
|------|-------------|
| `IlluminationData.csv` | LED emission spectra. Edit this to add or update LEDs. A `.pkl` copy is auto-generated on launch. |
| `PhotoReceptorData.pkl` | Opsin spectral sensitivity curves (Scones, Mcones, RedOpsin, Rods, Mela). |
| `calibration_*.xlsx` | LED power calibration data. Each sheet is named `YYYYMMDD`; the most recent sheet is used automatically. |
| `last_correction.txt` | Persistent correction values (measured power at 5V per LED). Updated automatically when new values are entered. |
| `environment.yml` | Conda environment definition. |

---

## Project structure

```
├── Ptot_Slider.py              Entry point — forward GUI
├── Iso_Target_Slider.py        Entry point — inverse GUI
├── PowerList_to_voltage.py     Entry point — batch conversion
├── colors_utils.py             Spectra, isomerisation matrix, solver, GUI functions
├── led_controllers_utils.py    Calibration loading, voltage interpolation, correction management
├── IlluminationData.csv        LED spectra (editable)
├── PhotoReceptorData.pkl       Opsin sensitivity data
├── calibration_*.xlsx          Calibration data
├── last_correction.txt         Stored correction values
├── environment.yml             Conda environment
├── Ptot_Slider.bat/.sh         Launchers — forward GUI
├── Iso_Target_Slider.bat/.sh   Launchers — inverse GUI
└── PowerList_to_Voltage.bat/.sh  Launchers — batch conversion
```

---

## Updating LED spectra

Edit `IlluminationData.csv` to add or modify LED spectra. The first column must be `x_axis` (wavelength in nm). Each additional column is one LED, named by its identifier (e.g. `Violet`, `Blue`, `Green`).

On next launch, the `.pkl` file is regenerated automatically from the CSV.
