import tkinter as tk
from tkinter import filedialog
import os
import re
from tqdm import tqdm
from led_controllers_utils import (
    load_spectral_dict_from_csv, get_corrections, save_correction_to_txt,
    get_latest_sheet_name, charge_calibration, get_voltages
)
from colors_utils import save_obj


if __name__ == '__main__':
    calibration_file = r"./calibration_5_colors_w_MEA_20230403.xlsx"

    leds_csv = load_spectral_dict_from_csv("IlluminationData.csv")
    pkl_led_file = "./IlluminationData.pkl"
    save_obj(leds_csv, pkl_led_file)

    all_leds = [c for c in leds_csv if c not in ("x_axis", "help")]
    selected_leds = []
    calibration = {}

    try:
                                        ### POWERLIST FILE LOADING ###

        # Open the file dialog
        root = tk.Tk()
        root.withdraw()
        print('Select a PowerList file...')
        powerlist_file = filedialog.askopenfilename(title="Select a PowerList file")
        root.destroy()

        print("Selected file:", powerlist_file)
        
        
                                       ### SELECTING LEDs ###
        led_success = False
        while not led_success:
            print("\nPlease select LEDs :")
            print("\n".join(f"{i}. {led}" for i, led in enumerate(all_leds, 1)))

            user_input = input("\nEnter numbers (separated by space, comma, etc.): ")
            try:
                choices = [int(n) for n in re.findall(r'\d+', user_input)]
                selected_leds = [all_leds[i-1] for i in choices if 1 <= i <= len(all_leds)]
                if selected_leds:
                    print(f"\nYou selected: {', '.join(selected_leds)}")
                    led_success = True
            except Exception:
                print("Invalid selection, please try again.")


                                        ### CORRECTION LOADING ###
        corr_success = False
        while not corr_success:
            try:
                corrections = get_corrections(selected_LEDs=selected_leds)
                save_correction_to_txt(corrections)
                corr_success = True
            except Exception as e:
                print(e)
                print("Error: please provide a proper input.")


                                        ### CALIBRATION FILE LOADING ###
        calib_success = False
        while not calib_success:
            try:
                print(f'Current calibration file:\n{calibration_file}\nSheet: {get_latest_sheet_name(calibration_file)}\n')

                response = input('Change calibration file? [y/n]: ').lower()
                if response in ["yes", "y"]:
                    root = tk.Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename(title="Select a calibration file")
                    root.destroy()
                    if file_path:
                        calibration_file = file_path

                calibration = charge_calibration(calibration_file, corrections, path_spectra=pkl_led_file, selected_LEDs=selected_leds, verbose=True)
                calib_success = True
            except Exception as e:
                print(f"\nError: {e}\n")


                                        ### GENERATING OUTPUT FILE ###
        output_file_path = powerlist_file.replace('.txt', '.csv').replace('PowerList', 'VoltageList')
        total_lines = sum(1 for _ in open(powerlist_file, 'r'))

        try:
            with open(powerlist_file, 'r') as file, open(output_file_path, 'w') as output_file:
                for line in tqdm(file, desc='Converting Powers to Voltage', total=total_lines):
                    Ptot = list(map(float, line.split()))
                    if len(Ptot) != len(selected_leds):
                        raise ValueError(f"Dimension mismatch: {len(Ptot)} columns in file vs {len(selected_leds)} LEDs selected.")
                    voltage = get_voltages(Ptot, calibration, selected_leds)
                    csv_line = ",".join(f"{v:f}" for v in voltage)
                    output_file.write(csv_line + "\n")
        except Exception:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            raise

        print(f"\nOutput file contains {total_lines} colors and saved at:\n{os.path.normpath(output_file_path)}\n")

        
        
    except Exception as e :
        print(f"Unexpected error : \n{e}")
        print("Program will shut down")
        input('Press enter to close...')
           
    
