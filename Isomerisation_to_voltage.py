from led_controllers_utils import *
from colors_utils import *

# --- Default Paths & Config ---
calibration_file_default = r"./calibration_5_colors_w_MEA_20230403.xlsx"
ledDATA_path = './IlluminationData.pkl'
illumination_csv = "IlluminationData.csv"

# Load spectral data and sync PKL (Matches PowerList_to_voltage logic)
leds_csv = load_spectral_dict_from_csv(illumination_csv)
save_obj(leds_csv, ledDATA_path)

# Extract LED names excluding axis and help
all_leds = [c for c in leds_csv if c not in ["x_axis", "help"]]

if __name__ == '__main__':
    try:
        # 1. CALIBRATION FILE SELECTION (GUI)
        root = tk.Tk()
        root.withdraw()  # Hide root window
        print('Select a calibration file...')
        
        calibration_file = filedialog.askopenfilename(
            title="Select the Excel calibration file", 
            initialfile=calibration_file_default
        )
        
        if not calibration_file:
            calibration_file = calibration_file_default
        
        print(f"File used: {calibration_file}")
        print(f"Sheet detected: {get_latest_calibration_sheet(calibration_file)}\n")

        # 2. LED SELECTION
        led_success = False
        while not led_success:
            print("\nPlease select LEDs to use:")
            print("\n".join(f"{i}. {led}" for i, led in enumerate(all_leds, 1)))

            user_input = input("\nEnter numbers (separated by space or comma): ")
            try:
                # Splitting input to handle spaces or commas
                indices = [int(x) - 1 for x in re.split(r'[,\s]+', user_input.strip()) if x]
                selected_leds = [all_leds[i] for i in indices if 0 <= i < len(all_leds)]
                
                if selected_leds:
                    print(f"Selected LEDs: {', '.join(selected_leds)}")
                    led_success = True
                else:
                    print("Error: No valid LEDs selected.")
            except:
                print("Invalid input. Please use the list numbers.")

        # 3. CORRECTION PHASE (Interactive mW @ 5V)
        # Display the last recorded update time
        _, last_time = load_correction_from_txt()
        print(f"\nLast correction update: {last_time}")
        
        correction = get_corrections(selected_leds)
        save_correction_to_txt(correction)

        # 4. LOAD CALIBRATION DATA
        # This computes the irradiance curves based on the correction
        calibration = charge_calibration(calibration_file, correction, selected_leds, verbose=True)

        # 5. INPUT ISOMERISATION TARGETS
        # 
        opsins = ['Scones', 'Mcones', 'Rods', 'Mela']
        isomerisation_target = {}
        print("\n--- ISOMERISATION TARGETS ---")
        for opsin in opsins:
            try:
                val = input(f"Target for {opsin} (Press Enter to ignore): ").strip()
                if val:
                    isomerisation_target[opsin] = float(val)
            except ValueError:
                print(f"Invalid value for {opsin}. Opsin ignored.")

        if not isomerisation_target:
            print("No targets entered. Program will exit.")
        else:
            # 6. CALCULATE POWER MIX (The Solver)
            print("\nCalculating best LED combination for targets...")
            Ptot_solution = get_mix_color(isomerisation_target, selected_LEDs=selected_leds, ledDATA_path=ledDATA_path)

            # 7. SUMMARY PLOTS
            print("\n--- SUMMARY ---")
            plot_isomerisations([Ptot_solution], selected_LEDs=selected_leds, ledDATA_path=ledDATA_path)
            plt.show(block=False)

            # 8. VOLTAGE CALCULATION
            # 
            print('\n--- REQUIRED LED VOLTAGES ---')
            # Convert solved power mix to voltages using the calibration curves
            voltage = get_voltages(Ptot_solution, calibration, selected_leds, verbose=True).astype(np.float32)
            
            # Display Hardware conversion (8-bit)
            print(f"\nHardware Conversion (8-bit):")
            for i in range(len(selected_leds)):
                 print(f"{selected_leds[i]:12} : {float32_to_uint8(voltage[i])}")

        print("\n-------------------------------------")
        input('\nPress Enter to finish...')

    except Exception as e:
        print(f"Unexpected error: \n{e}")
        input('\nPress Enter to quit...')