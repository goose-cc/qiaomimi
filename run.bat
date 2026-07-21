@echo off
set MODEL_TYPE=%1
set MODE=%2
set LOSS_PROFILE=%3
set INDEX=%4

if "%MODEL_TYPE%"=="" set MODEL_TYPE=transformer
if "%MODE%"=="" set MODE=train
if "%LOSS_PROFILE%"=="" set LOSS_PROFILE=base
if "%INDEX%"=="" set INDEX=20

set PYTHON=.\venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

if "%MODE%"=="load" (
    "%PYTHON%" main.py --model_type %MODEL_TYPE% --loss_profile %LOSS_PROFILE% --index %INDEX% --load_model
) else (
    "%PYTHON%" main.py --model_type %MODEL_TYPE% --loss_profile %LOSS_PROFILE% --index %INDEX%
)