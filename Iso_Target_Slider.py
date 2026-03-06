from colors_utils import *
from led_controllers_utils import *


leds_csv = load_spectral_dict_from_csv("IlluminationData.csv")

selected_LEDs = [led for led in leds_csv.keys() if led != "x_axis"]
max_vals = {led: 5000 if ('Red' in led or '600' in led or "595" in led) else 20 for led in selected_LEDs}

if __name__ == '__main__':
    interactive_iso_target_slider(selected_LEDs=selected_LEDs, max_vals=max_vals)
