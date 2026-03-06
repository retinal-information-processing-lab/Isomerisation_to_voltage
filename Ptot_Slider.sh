#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PYTHON="$HOME/anaconda3/envs/led_calib_env/bin/python3"

"$ENV_PYTHON" -c "
import subprocess, os
subprocess.Popen(
    ['$ENV_PYTHON', '$SCRIPT_DIR/Ptot_Slider.py'],
    cwd='$SCRIPT_DIR',
    start_new_session=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
"
