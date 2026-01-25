from led_controllers_utils import *
from  colors_utils import *




calibration_file = r"./calibration_5_colors_w_MEA_20230403.xlsx"
leds_csv = load_spectral_dict_from_csv("IlluminationData.csv")

pkl_led_file = "./IlluminationData.pkl"
save_obj(leds_csv, pkl_led_file)


all_leds = [c for c in leds_csv if c != "x_axis"]


try :
    all_leds.remove('x_axis')
except :
    pass 
try:
    all_leds.remove('help')
except:
    pass
if __name__ == '__main__':
    try:
                                        ### POWERLIST FILE LOADING ###

        # Create the root window (it won't be shown)
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        print('Select a PowerList file...')
        # Open the file dialog
        powerlist_file = filedialog.askopenfilename(title="Select a PowerList file")

        # Print the selected file path
        print("Selected file:", powerlist_file)

        root.quit()
        
        
                                       ### Selecting Leds ###
        led_success = False
        while not led_success:
            # Affichage compact
            print("\nPlease select LEDs :")
            print("\n".join(f"{i}. {led}" for i, led in enumerate(all_leds, 1)))

            user_input = input("\nEnter numbers (separated by space, comma, etc.): ")
            try :
                # 2. Le "secret" : re.findall extrait tous les nombres, peu importe le délimiteur
                choices = [int(n) for n in re.findall(r'\d+', user_input)]

                # 3. Filtrage et récupération en une ligne
                selected_leds = [all_leds[i-1] for i in choices if 1 <= i <= len(all_leds)]

                if selected_leds:
                    print(f"\nYou selected: {', '.join(selected_leds)}")
                    led_success = True
            except:
                print("Invalid selection, please try again.")
                pass
        
        
        
                                        ### CORRECTION LOADING ###
        corr_success = False
        while not corr_success:
            try:
                corrections = get_corrections(selected_LEDs=selected_leds)
                save_correction_to_txt(corrections)
                corr_success = True
            except Exception as e:
                print(e)
                print("Erreur : Please provide a proper input.")        
        
        

        
                                        ### CALIBRATION FILE LOADING ###

        calib_success = False
        while not calib_success:
            try:
                print(f'Current file for calibration : \n{calibration_file}\nSheet : {get_latest_sheet_name(calibration_file)} \n')

                response = input('Souhaitez-vous changer de fichier ? : ').lower()
                if response in ["yes", "y", "oui", "o", "si"]:
                    root = tk.Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename(title="Select a calibration file")
                    root.destroy()
                    if file_path:
                        calibration_file = file_path

                calibration = charge_calibration(calibration_file, corrections, path_spectra= pkl_led_file, selected_LEDs=selected_leds, verbose=True)
                calib_success = True # Sortie de boucle si le chargement réussit
            except Exception as e:
                print(f"\nErreur : {e}\n") # Affiche l'erreur et recommence la boucle
        
        
                                        ### Generating Output file ###

        # On remplace l'extension et le nom pour le fichier de sortie
        output_file_path = powerlist_file.replace('.txt', '.csv').replace('PowerList', 'VoltageList')

        # Calcul du nombre de lignes pour la barre de progression tqdm
        total_lines = sum(1 for _ in open(powerlist_file, 'r'))

        with open(powerlist_file, 'r') as file, open(output_file_path, 'w') as output_file:
            for line in tqdm(file, desc='Converting Powers to Voltage', total=total_lines):
                # Extraction des puissances de la ligne (split par espace ou tab)
                Ptot = list(map(float, line.split()))

                # Vérification de cohérence entre le fichier texte et les LEDs sélectionnées
                if len(Ptot) != len(selected_leds):
                     raise ValueError(f"Erreur de dimension : {len(Ptot)} colonnes dans le texte vs {len(selected_leds)} LEDs.")

                # Calcul des tensions avec ta fonction habituelle
                # Note : voltage doit être un tableau numpy ou une liste
                voltage = get_voltages(Ptot, calibration, selected_leds)

                # Création de la ligne CSV : on joint les valeurs par des virgules
                # f"{v:f}" permet d'éviter la notation scientifique (ex: 1e-05) pour rester lisible
                csv_line = ",".join(f"{v:f}" for v in voltage)

                # Écriture de la ligne suivie d'un retour à la ligne
                output_file.write(csv_line + "\n")

        print(f"\nOutput file contains   {total_lines} colors   and saved at :\n{os.path.normpath(output_file_path)} \n")

        
        
    except Exception as e :
        print(f"Unexpected error : \n{e}")
        print("Program will shut down")
        input('Press enter to close...')
           
    
