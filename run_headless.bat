@echo off
setlocal enabledelayedexpansion

title Abbreviated Wheel Learner (Headless)

echo.
echo ========================================
echo   Abbreviated Wheel Learner - HEADLESS
echo   (No visualization - faster training)
echo ========================================
echo.

:: Check if virtual environment exists and is set up
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Running setup first...
    call run.bat --headless %*
    exit /b
)

:: Activate and run
call venv\Scripts\activate.bat

:: Parse arguments or use defaults
set ARGS=%*
if "%ARGS%"=="" (
    set ARGS=--pool 36 --draw 5 --match 3 --gens 500
)

python main.py --headless %ARGS%

deactivate
pause
