from colors_utils import *
from led_controllers_utils import *


calibration_file = r"./calibration_5_colors_w_MEA_20230403.xlsx"
leds_csv= load_spectral_dict_from_csv("IlluminationData.csv")


pkl_led_file = "./IlluminationData.pkl"
save_obj(leds_csv, pkl_led_file)


selected_LEDs = [ led for led in leds_csv.keys() if led !="x_axis"]
def_val = {led:1 for led in leds_csv.keys()}
max_vals = {led: 5000 if ('Red' in led or '600' in led or "595" in led) else 20 for led in leds_csv.keys()}

if __name__ == '__main__':
    interactive_Ptot_slider(selected_LEDs=selected_LEDs, def_vals=(def_val), max_vals = max_vals)