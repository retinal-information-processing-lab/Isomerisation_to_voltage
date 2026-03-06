@echo off
set ENV_NAME=led_calib_env

call conda activate %ENV_NAME%

if %errorlevel% neq 0 (
    echo [ERREUR] L'environnement %ENV_NAME% n'existe pas ou conda n'est pas installe.
    pause
    exit /b
)

start "" pythonw Ptot_Slider.py
