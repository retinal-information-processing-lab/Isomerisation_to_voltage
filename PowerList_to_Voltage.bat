@echo off
set ENV_NAME=led_calib_env

echo [INFO] Activation de l'environnement %ENV_NAME%...
:: On utilise 'call' pour s'assurer que conda s'initialise bien dans le batch
call conda activate %ENV_NAME%

if %errorlevel% neq 0 (
    echo [ERREUR] L'environnement %ENV_NAME% n'existe pas ou conda n'est pas installe.
    pause
    exit /b
)

echo [INFO] Lancement de PowerList_to_voltage...
python PowerList_to_voltage.py

pause
