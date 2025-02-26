import openpyxl as pxl
import numpy as np
import matplotlib.pyplot as plt
from  colors_utils import *
from tqdm import trange
import tkinter as tk
from tkinter import filedialog
import pickle
import os


calibration_file = "/home/guiglaz/Documents/stim generation/calibration_to_volt/calibration_5_colors_w_MEA_20230403.xlsx"

ledDATA_path = './IlluminationData.pkl'


if __name__ == '__main__':
    
    corr = input('Do you want to apply a new correction ? (If no, last correction will be applied)')
    if corr not in ["Yes", 'Y', 'y', 'yes', 'YES', 'Oui','oui','OUI', 'SI', 'Si', 'si']:
        try:
            statbuf = os.stat('./last_correction.pkl')
            utc_plus_1 = timezone(timedelta(hours=1))
            print("Last Correction Date: {}".format(datetime.fromtimestamp(statbuf.st_mtime, tz=utc_plus_1).strftime("%Y-%m-%d %H:%M UTC+1")))
            calibrations = load_obj('./last_correction.pkl')
            read_success = True
        except FileNotFoundError:
            print('No previous correction found, user input needed...')
            read_success = False
            
    if not read_success :
        # Create the root window (it won't be shown)
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        print('Select a calibration file...')
        # Open the file dialog
        file_path = filedialog.askopenfilename(title="Select a calibration file")

        # Print the selected file path
        print("Selected file:", file_path)

        root.quit()

        calibration, _ = charge_calibration(path = file_path, verbose=True)
        save_obj(calibrations, './last_correction.pkl')   
    
    all_leds = list(load_obj(ledDATA_path).keys())

    try :
        all_leds.remove('x_axis')
    except :
        pass 
    try:
        all_leds.remove('help')
    except:
        pass
    
    delimiters = [',', ' ', ';', ':', '+']  # List of possible delimiters
    while True:
        print("\nPlease select LEDs by entering the numbers next to them, multiples possible : ")
        for i, led in enumerate(all_leds, start=1):
            print(f"{i}. {led}")

        try:
            user_input = input("Enter numbers: ")

            # Replace each delimiter with a space
            for delimiter in delimiters:
                user_input = user_input.replace(delimiter, ' ')

            # Split the input by spaces and convert to integers
            choices = [int(x.strip()) for x in user_input.split() if x.strip()]
            selected_leds = [all_leds[i-1] for i in choices if 1 <= i <= len(all_leds)]

            if selected_leds:
                print(f"You selected: {', '.join(selected_leds)}")
                break
            else:
                print("Invalid selection, try again.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by valid delimiters.")
        
    
    opsins = ['Scones', 'Mcones', 'Rods', 'Mela']
    isomerisation_target = {}
    
    for opsin in opsins:
        try :
            isomerisation = float(input(f"Target Isomerisation for {opsin} (if not a number, opsin is ignored) : "))
            isomerisation_target[opsin] = int(isomerisation)
        except:
            print(f"{opsin} is ignored !")
    print("Best Solution found:")
    Ptot_solution = get_mix_color(isomerisation_target, selected_LEDs = selected_leds, ledDATA_path = ledDATA_path)
    print("\n\n\n\n---------------Summary---------------")

    
    plot_isomerisations([Ptot_solution], selected_LEDs = selected_leds, ledDATA_path = ledDATA_path)
    plt.show(block=False)
    print('\nLEDs Voltages')
    get_voltages(Ptot_solution, calibration, selected_leds, verbose = True)
    print("-------------------------------------\n\n")
    input('\nPress enter to finish')
    
           
    
