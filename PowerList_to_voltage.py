import openpyxl as pxl
import numpy as np
import matplotlib.pyplot as plt
from  colors_utils import *
from tqdm import trange
import tkinter as tk
from tkinter import filedialog
import pickle
import os
import time

calibration_file = "/home/guiglaz/Documents/stim generation/calibration_to_volt/calibration_5_colors_w_MEA_20230403.xlsx"


ledDATA_path = './IlluminationData.pkl'


if __name__ == '__main__':
    
    
    # Create the root window (it won't be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    print('Select a PowerList file...')
    # Open the file dialog
    powerlist_file = filedialog.askopenfilename(title="Select a PowerList file")

    # Print the selected file path
    print("Selected file:", powerlist_file)

    root.quit()
    
    corr = input('Do you want to apply a new correction (If no, last correction will be applied)? : ')
    if corr not in ["Yes", 'Y', 'y', 'yes', 'YES', 'Oui','oui','OUI', 'SI', 'Si', 'si']:
        try:
            statbuf = os.stat('./last_correction.pkl')
            utc_plus_1 = timezone(timedelta(hours=1))
            print("Last Correction Date: {}".format(datetime.fromtimestamp(statbuf.st_mtime, tz=utc_plus_1).strftime("%Y-%m-%d %H:%M UTC+1")))
            calibration = load_obj('./last_correction.pkl')
            read_success = True
        except FileNotFoundError:
            print('No previous correction found, user input needed...')
            read_success = False
    else:
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
        save_obj(calibration, './last_correction.pkl')   

    
    all_leds = list(load_obj(ledDATA_path).keys())

    try :
        all_leds.remove('x_axis')
    except :
        pass 
    try:
        all_leds.remove('help')
    except:
        pass
    
    delimiters = [',', ' ', ';', ':', '+','/']  # List of possible delimiters
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
                print(f"\nYou selected: {', '.join(selected_leds)}")
                break
            else:
                print("Invalid selection, try again.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by valid delimiters.")
        
    compressed = False
    compressed = input('Do you want to compress to 8bit (If no, voltage will be float32)? : ')
    if compressed in ["Yes", 'Y', 'y', 'yes', 'YES', 'Oui','oui','OUI', 'SI', 'Si', 'si']:
        compressed = True
        
        
    output_file_path = powerlist_file.replace('PowerList', '')  # Remove 'PowerList' if it exists
    output_file_path = output_file_path.split('.')[0] + 'VoltageList.txt'  # Add '_VoltageList' before the file extension

    tot = 0
    with open(powerlist_file, 'r') as file, open(output_file_path, 'w') as output_file:
        for line in tqdm(file, desc = 'Converting Powers to Voltage : ', total=sum(1 for _ in open(powerlist_file))):
            Ptot = list(map(int, line.split()))

            # Ensure that the number of columns in the line matches the number of selected LEDs
            assert len(Ptot) == len(selected_leds), f"Mismatch: {len(Ptot)} columns in file, but {len(selected_leds)} LEDs selected."

            voltage = get_voltages(Ptot, calibration, selected_leds)  
            
            # Write the calculated voltages to the output file in the proper format
            if compressed:
                output_file.write('\t'.join(map(str, float32_to_uint8(voltage))) + '\n')
            else:
                output_file.write('\t'.join(map(str, voltage.astype(np.float32))) + '\n')
            tot += 1
    print(f"\nOutput file contains   {tot} colors   and saved at :\n{os.path.normpath(output_file_path)} \n")
    
           
    
