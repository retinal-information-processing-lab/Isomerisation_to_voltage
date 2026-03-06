@echo off
set ENV_NAME=led_calib_env

echo [INFO] Activation de l'environnement %ENV_NAME%...
call conda activate %ENV_NAME%

if %errorlevel% neq 0 (
    echo [ERREUR] L'environnement %ENV_NAME% n'existe pas ou conda n'est pas installe.
    pause
    exit /b
)

echo [INFO] Lancement de Iso_Target_Slider...
python Iso_Target_Slider.py

pause
