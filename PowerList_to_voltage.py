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


def load_obj(name):
    """
        Generic function to load a bin obj with pickle protocol

    Input :
        - name (str) : path to where the obj is
    Output :
        - (python object) : loaded object
        
    Possible mistakes :
        - Wrong path 
    """
    
    if os.path.dirname(os.path.normpath(name)) != '':
        os.makedirs(os.path.dirname(os.path.normpath(name)), exist_ok=True)
    else:
        name = os.path.join(os.getcwd(),os.path.normpath(name))
    if name[-4:]!='.pkl':
        name += '.pkl'
    with open(os.path.normpath(name), 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, name ):
    """
        Generic function to save an obj with pickle protocol

    Input :
        - obj (python var) : object to be saved in binary format
        - name (str) : path to where the obj shoud be saved

    Possible mistakes :
        - Permissions denied, restart notebook from an admin shell
        - Folders aren't callable, change your folders
    """
    
    if os.path.dirname(os.path.normpath(name)) != '':
        os.makedirs(os.path.dirname(os.path.normpath(name)), exist_ok=True)
    else:
        name = os.path.join(os.getcwd(),os.path.normpath(name))
    
    if name[-4:]!='.pkl':
        name += '.pkl'
    with open( os.path.normpath(name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def charge_calibration(path=calibration_file,
                       all_LEDs = ['Violet', 'Blue', 'Green', 'Yellow', 'Red'],
                      verbose = True):

    
    
    print('\nC O R R E C T I O N \n')
    
    correction = {'Red':None,
                  'Yellow':None,
                  'Green':None,
                  'Blue':None,
                  'Violet':None, 
                 }
    for i in range(len(all_LEDs)):
        temp_color_name = all_LEDs[i]
        temp_corr = input('Enter the measured power of the {} LED at 5V (mW): '.format(temp_color_name) )
        try :
            temp_corr = float(temp_corr)
            print('The {} LED power function will be corrected.'.format(temp_color_name) )
            correction[temp_color_name]=temp_corr
        except :
            print('No correction will be applied to the {} LED.'.format(temp_color_name))
            pass

    print('\nF I L T E R S \n')
    
    filters = {'Red':1,
                'Yellow':1,
                'Green':1,
                'Blue':1,
                'Violet':1, 
                 }

    for i in range(len(all_LEDs)):
        temp_color_name = all_LEDs[i]
        temp_filter = input('Enter the filter transmittance value applied to the {} LED: '.format(temp_color_name))
        try :
            temp_filter = float(temp_filter)
            print('The {} LED will be filtered.'.format(temp_color_name))
            filters[temp_color_name]=temp_filter
        except :
            print('No filter will be applied to the {} LED.'.format(temp_color_name))
            pass
    
    all_LEDs = np.flip(all_LEDs)
    wb = pxl.load_workbook(path)
    ws = wb[list(sorted(wb.sheetnames))[-1]]
    #col_number = 5
    #colors = ['625 nm','530 nm','490 nm', '415 nm', '385 nm']
    
    fiber_to_mea_red = {'Red':1,
                        'Yellow':1,
                        'Green':1,
                        'Blue':1,
                        'Violet':1, 
                 }
    np.ones(len(all_LEDs))
    
    calibrations = {'voltages' : np.array([ws.cell(row=i, column=1).value for i in range(4,21)])}
    for col_i in range(0,len(all_LEDs)):
        color_name = all_LEDs[col_i]
        temp_ratio = ws.cell(row=23, column=2+col_i).value/ws.cell(row=20, column=2+col_i).value
        fiber_to_mea_red[color_name]=temp_ratio 
        
        calibrations[color_name]=np.array([ws.cell(row=i, column=2+col_i).value for i in range(4,21)])

        
        if correction[color_name]!=None :
            calibrations[color_name]=calibrations[color_name]*correction[color_name]/calibrations[color_name][-1]
    
        calibrations[color_name]=calibrations[color_name]*filters[color_name]*fiber_to_mea_red[color_name]
        if verbose:
            print(calibrations[color_name])
    min_pow=0
    ind=1
    old_pow=calibrations[all_LEDs[0]][ind]
    while min_pow==0 :
        for col_name in all_LEDs :
            temp_pow=calibrations[col_name][ind]
            min_pow = min(temp_pow,old_pow)
            old_pow=temp_pow
        ind+=1
    if ind>=1 :
        calibrations['voltages']=np.concatenate(([0],calibrations['voltages'][ind:]))
        for col_name in all_LEDs :
            calibrations[col_name]=np.concatenate(([0],calibrations[col_name][ind:]))


    if verbose :
        for col_name in all_LEDs :
        
            plt.plot((calibrations['voltages']),calibrations[col_name], label=col_name)
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('Tension (V)')
            plt.ylabel('Power (µW/cm²)')
            plt.title('{} LED'.format(col_name))
            plt.legend()
            plt.show(block=False)
                 
    return calibrations, fiber_to_mea_red

def get_voltages(Ptot, calibration, all_LEDs = ['Violet', 'Blue', 'Green', 'Yellow', 'Red']):
    voltages = calibration['voltages']
    driving_tension = []
    
    for col_i in range(len(all_LEDs)):
        col_name = all_LEDs[col_i]
        temp_P = float(Ptot[col_i])
        
        # Return 0 voltage if the power is 0
        if temp_P == 0:
            driving_tension.append(0)
            print(f"{col_name}: Power is 0, so voltage is 0 V.")
            continue
        
        calibration_list = calibration[col_name]
        
        if temp_P > calibration_list[-1]:
            raise ValueError('The given power value is too high. Try putting a lower value.')
    
        if temp_P in calibration_list:
            index = np.where(calibration_list == temp_P)
            res_voltage = voltages[index][0]
        else:
            low_ind = 0
            high_ind = len(calibration_list) - 1
            
            for i in range(len(calibration_list)):
                if calibration_list[i] < temp_P and calibration_list[i] > calibration_list[low_ind]:
                    low_ind = i
                if calibration_list[i] > temp_P and calibration_list[i] < calibration_list[high_ind]:
                    high_ind = i
            
            res_voltage = voltages[low_ind] + (voltages[high_ind] - voltages[low_ind]) * (temp_P - calibration_list[low_ind]) / (calibration_list[high_ind] - calibration_list[low_ind])
            driving_tension.append(res_voltage)

            print(f'{col_name}: {res_voltage} V')
            
    driving_tension = np.array(driving_tension)
    return driving_tension



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
        
        
    output_file_path = powerlist_file.replace('PowerList', '')  # Remove 'PowerList' if it exists
    output_file_path = output_file_path.split('.')[0] + 'VoltageList.txt'  # Add '_VoltageList' before the file extension

    with open(powerlist_file, 'r') as file, open(output_file_path, 'w') as output_file:
        for line in file:
            Ptot = list(map(int, line.split()))

            # Ensure that the number of columns in the line matches the number of selected LEDs
            assert len(Ptot) == len(selected_leds), f"Mismatch: {len(Ptot)} columns in file, but {len(selected_leds)} LEDs selected."

            voltage = get_voltages(Ptot, calibration, selected_leds)  
            
            # Write the calculated voltages to the output file in the same format
            output_file.write('\t'.join(map(str, voltage)) + '\n')
   
    input('\nPress enter to finish')
    
           
    
