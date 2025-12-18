@echo off
setlocal enabledelayedexpansion

title Abbreviated Wheel Learner - Menu

:menu
cls
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║     🎰 NEURAL NETWORK ABBREVIATED WHEEL LEARNER 🎰       ║
echo  ║          CUDA-Accelerated Lottery Optimizer              ║
echo  ╠══════════════════════════════════════════════════════════╣
echo  ║                                                          ║
echo  ║   [1] 3 of 5 from 36  (Default - Quick)                  ║
echo  ║   [2] 4 of 6 from 49  (UK Lotto Style)                   ║
echo  ║   [3] 3 of 6 from 45  (Medium)                           ║
echo  ║   [4] 5 of 6 from 49  (Hard - Many Tickets)              ║
echo  ║   [5] Custom Configuration                               ║
echo  ║                                                          ║
echo  ║   [H] Run Headless (No Graphics)                         ║
echo  ║   [Q] Quit                                               ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
set /p choice="  Select option: "

if "%choice%"=="1" goto run_3of5from36
if "%choice%"=="2" goto run_4of6from49
if "%choice%"=="3" goto run_3of6from45
if "%choice%"=="4" goto run_5of6from49
if "%choice%"=="5" goto custom
if /i "%choice%"=="h" goto headless_menu
if /i "%choice%"=="q" goto quit

echo Invalid option, try again...
timeout /t 2 >nul
goto menu

:run_3of5from36
call run.bat --pool 36 --draw 5 --match 3
goto menu

:run_4of6from49
call run.bat --pool 49 --draw 6 --match 4
goto menu

:run_3of6from45
call run.bat --pool 45 --draw 6 --match 3
goto menu

:run_5of6from49
call run.bat --pool 49 --draw 6 --match 5
goto menu

:custom
cls
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║              CUSTOM WHEEL CONFIGURATION                  ║
echo  ║                                                          ║
echo  ║   Format: "M of D from P"                                ║
echo  ║   - P = Pool (total numbers)                             ║
echo  ║   - D = Draw (numbers per ticket)                        ║
echo  ║   - M = Match (guaranteed matches)                       ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
set /p pool="  Enter POOL size (e.g., 36): "
set /p draw="  Enter DRAW size (e.g., 5): "
set /p match="  Enter MATCH required (e.g., 3): "
set /p gens="  Enter generations (default 500): "

if "%gens%"=="" set gens=500

echo.
echo  Running: %match% of %draw% from %pool%
echo.
call run.bat --pool %pool% --draw %draw% --match %match% --gens %gens%
goto menu

:headless_menu
cls
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║              HEADLESS MODE (No Graphics)                 ║
echo  ║            Faster training, console output only          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
set /p pool="  Enter POOL size (default 36): "
set /p draw="  Enter DRAW size (default 5): "
set /p match="  Enter MATCH required (default 3): "
set /p gens="  Enter generations (default 1000): "

if "%pool%"=="" set pool=36
if "%draw%"=="" set draw=5
if "%match%"=="" set match=3
if "%gens%"=="" set gens=1000

echo.
echo  Running headless: %match% of %draw% from %pool% for %gens% generations
echo.
call run.bat --headless --pool %pool% --draw %draw% --match %match% --gens %gens%
goto menu

:quit
echo.
echo  Goodbye!
exit /b 0
