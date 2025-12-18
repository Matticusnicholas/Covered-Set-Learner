@echo off
setlocal enabledelayedexpansion

title Neural Network Abbreviated Wheel Learner

echo.
echo ========================================
echo   Neural Network Abbreviated Wheel Learner
echo   CUDA-Accelerated Lottery Optimizer
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    set NEEDS_INSTALL=1
) else (
    echo [OK] Virtual environment exists
    set NEEDS_INSTALL=0
)

:: Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

:: Check if packages are installed by testing imports
python -c "import torch; import pygame; import numpy" >nul 2>&1
if errorlevel 1 (
    set NEEDS_INSTALL=1
)

:: Install dependencies if needed
if !NEEDS_INSTALL!==1 (
    echo.
    echo [SETUP] Installing dependencies...
    echo         This may take a few minutes on first run...
    echo.

    :: Upgrade pip first
    python -m pip install --upgrade pip --quiet

    :: Install PyTorch with CUDA support
    echo [SETUP] Installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
    if errorlevel 1 (
        echo [WARN] CUDA version failed, trying CPU version...
        pip install torch torchvision --quiet
    )

    :: Install other requirements
    echo [SETUP] Installing other dependencies...
    pip install pygame numpy tqdm colorama pillow --quiet

    echo.
    echo [OK] All dependencies installed!
    echo.
) else (
    echo [OK] Dependencies already installed
)

:: Display CUDA status
echo.
python -c "import torch; print('[GPU] CUDA Available:', torch.cuda.is_available()); print('[GPU] Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>nul
echo.

:: Parse command line arguments or use defaults
set ARGS=%*
if "%ARGS%"=="" (
    set ARGS=--pool 36 --draw 5 --match 3
)

echo ========================================
echo   Starting Training...
echo   Args: %ARGS%
echo   Press ESC in the window to stop
echo ========================================
echo.

:: Run the main application
python main.py %ARGS%

:: Deactivate virtual environment
deactivate

echo.
echo ========================================
echo   Training Complete!
echo ========================================
pause
