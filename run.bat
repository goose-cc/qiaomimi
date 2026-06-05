@REM # python ./main.py --model_type cnn --load_model --index 25
@REM # python ./main.py --model_type unet --load_model --index 0
@REM # python ./main.py --model_type cnn --index 25

@REM close echo when a command is executed.
@echo off

@REM set python and pip path
set "VENV_DIR=.\venv"
set "PYTHON=.\venv\Scripts\python.exe"
set "PIP=.\venv\Scripts\pip.exe"

@REM set train and  test dataset path
set "train_data_dir=./train"
set "test_data_dir=./test"
set "val_data_dir=./val"

set "MODEL_TYPE=%1"
set "LOAD=%2"
set "INDEX=%3"

@REM python env create
if not exist "%VENV_DIR%" (
    echo ========================================
    echo a python virual enviroment need to create
    echo ========================================
    exit /b 1
)

@REM ======================================= data processing =======================================
set "data_process=0"
dir /b /a "%train_data_dir%" 2>nul | findstr . >nul
if errorlevel 1 (
    set "data_process=1"
)

dir /b /a "%test_data_dir%" 2>nul | findstr . >nul
if errorlevel 1 (
    set "data_process=1"
)

dir /b /a "%val_data_dir%" 2>nul | findstr . >nul
if errorlevel 1 (
    set "data_process=1"
)

if "%data_process%"==1 (
    echo ========================================
    echo start generating dataset
    echo ========================================
    %PYTHON% ./data_generate.py
)

@REM =============================================run scripts=====================================
if "%LOAD%"=="load" (
    %PYTHON% ./main.py --model_type %MODEL_TYPE% --load_model --index %INDEX%
) else if "%LOAD%"=="train" (
    %PYTHON% ./main.py --model_type %MODEL_TYPE% --index %INDEX%
)