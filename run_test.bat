@echo off
rem ===========================================================================
rem  run_test.bat - repeats the measurements of the thesis in Isaac Sim 5.1.0
rem
rem    run_test.bat          shows the list of tests and asks for a number
rem                          (this is what happens on a double-click)
rem    run_test.bat 3        runs test 3  (in PowerShell: .\run_test.bat 3)
rem    run_test.bat setup    only finds Isaac Sim and installs the packages
rem
rem  Isaac Sim is found automatically: ISAACSIM_PATH, C:\isaacsim, next to
rem  this folder, on the Desktop or in Downloads. Otherwise the script asks
rem  for the folder once and remembers it in isaacsim_path.txt.
rem  Before running a test, it prints the exact command it runs.
rem ===========================================================================
setlocal
cd /d "%~dp0"
set "MENU="
if "%~1"=="" set "MENU=1"

call :find_isaacsim
if errorlevel 1 goto end_fail
call :check_packages
if errorlevel 1 goto end_fail
if /i "%~1"=="setup" goto end_ok
if defined MENU goto menu

call :run_test "%~1"
exit /b %errorlevel%

:menu
echo.
echo  FSAE RL Racing: repeat the measurements of the thesis
echo  Isaac Sim: %ISAAC%
echo.
echo    1  Recompute the thesis tables from the result files      1 second
echo    2  Watch the final cars race (opens a window)             until closed
echo    3  The final car on the stadium track                     about 3 min
echo    4  The final car on the original track                    about 3 min
echo    5  An unseen track, before and after extra training       about 6 min
echo    6  The classical controller (pure pursuit)                about 4 min
echo    7  The learned single car                                 about 4 min
echo    8  Where a record training score came from                about 30 s
echo    9  Whoever starts slightly ahead wins                     about 2 min
echo   10  The two cars behave identically (mirrored starts)      about 2 min
echo   11  Can the car behind overtake?                           about 2.5 min
echo   12  Staggered-start experiment: driving quality            about 3 min
echo   13  Staggered-start experiment: overtaking                 about 2.5 min
echo   14  Ten-lap benchmark                                      under 1 min
echo   15  Lap telemetry: acceleration, steering, sector times    about 4 min
echo   16  Recompute the thesis tables from your own runs         1 second
echo.
set "CHOICE="
set /p "CHOICE= Type a test number and press Enter (Enter alone quits): "
if not defined CHOICE goto end_ok
if /i "%CHOICE%"=="q" goto end_ok
call :run_test "%CHOICE%"
echo.
pause
goto menu

:end_fail
if defined MENU pause
exit /b 1

:end_ok
exit /b 0


rem ---------------------------------------------------------------- the tests
:run_test
for %%N in (1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16) do if "%~1"=="%%N" goto t%%N
echo.
echo  There is no test "%~1". Choose a number from 1 to 16.
exit /b 1

:t1
call :py 2_python_scripts\check_results.py
exit /b
:t2
call :py 2_python_scripts\g_two_car_racing\run_unknown_track_demo.py
exit /b
:t3
call :py 2_python_scripts\h_evaluation\two_car_race_eval.py stadium 12 2000 100M_milestone
exit /b
:t4
call :py 2_python_scripts\h_evaluation\two_car_race_eval.py original 12 2000 100M_milestone
exit /b
:t5
call :py 2_python_scripts\h_evaluation\two_car_race_eval.py stadium 12 2000 run49 newtrack_run3
exit /b
:t6
call :py 2_python_scripts\h_evaluation\common_metric_eval.py purepursuit
exit /b
:t7
call :py 2_python_scripts\h_evaluation\common_metric_eval.py baseline700
exit /b
:t8
call :py 2_python_scripts\h_evaluation\collapse_reward_diagnostic.py racingline det box
exit /b
:t9
call :py 2_python_scripts\h_evaluation\start_condition_eval.py 12 500 4_results\start_condition_results\start_offsets_100M.csv
exit /b
:t10
call :py 2_python_scripts\h_evaluation\start_condition_eval.py 12 500 4_results\start_condition_results\mirrored_100M.csv 3_trained_models\car_race_ppo_model 1
exit /b
:t11
call :py 2_python_scripts\h_evaluation\stagger_eval.py 4_results\start_condition_results\passing_test_100M_before.csv 3_trained_models\car_race_ppo_model 500 3
exit /b
:t12
call :py 2_python_scripts\h_evaluation\two_car_race_eval.py stadium 12 2000 newtrack_stagger_run3
exit /b
:t13
call :py 2_python_scripts\h_evaluation\stagger_eval.py 4_results\start_condition_results\passing_test_100M_after_staggered.csv 3_trained_models\checkpoints\car_race_ppo_model_newtrack_stagger_run3 500 3
exit /b
:t14
call :py 2_python_scripts\g_two_car_racing\lap_time_benchmark.py 10
exit /b
:t15
call :py 2_python_scripts\h_evaluation\lap_telemetry.py 10
if errorlevel 1 exit /b
call :py 2_python_scripts\h_evaluation\lap_telemetry.py 10 swap
if errorlevel 1 exit /b
call :py 2_python_scripts\h_evaluation\lap_telemetry.py 10 a_ahead=2
if errorlevel 1 exit /b
call :py 2_python_scripts\h_evaluation\plot_lap_telemetry.py
exit /b
:t16
call :py 2_python_scripts\check_results.py
exit /b

:py
echo.
echo  Project folder: %CD%
echo  Command:        "%PY%" %*
echo.
call "%PY%" %*
exit /b


rem ---------------------------------------------------------- finding Isaac Sim
:find_isaacsim
set "ISAAC="
set "SAVED="
if exist "isaacsim_path.txt" set /p SAVED=<"isaacsim_path.txt"
if defined ISAACSIM_PATH call :try "%ISAACSIM_PATH%"
if defined SAVED call :try "%SAVED%"
call :try "C:\isaacsim"
call :try "%~dp0..\isaac-sim-standalone-5.1.0-windows-x86_64"
call :try "%~dp0..\isaacsim"
call :try "C:\isaac-sim"
call :try "C:\isaac-sim-standalone-5.1.0-windows-x86_64"
call :try "%USERPROFILE%\Desktop\isaac-sim-standalone-5.1.0-windows-x86_64"
call :try "%USERPROFILE%\Desktop\isaacsim"
call :try "%USERPROFILE%\Downloads\isaac-sim-standalone-5.1.0-windows-x86_64"
call :try "%USERPROFILE%\isaacsim"
if defined ISAAC goto found

:ask
echo.
echo  Isaac Sim 5.1.0 was not found.
echo  Paste the folder where it is installed (the folder that contains
echo  python.bat) and press Enter. Press Enter alone to cancel.
set "ANSWER="
set /p "ANSWER= Isaac Sim folder: "
if not defined ANSWER exit /b 1
set "ANSWER=%ANSWER:"=%"
call :try "%ANSWER%"
if defined ISAAC goto save
echo  There is no python.bat in "%ANSWER%".
goto ask
:save
>"isaacsim_path.txt" echo %ISAAC%
echo  Saved in isaacsim_path.txt, so you will not be asked again.

:found
set "PY=%ISAAC%\python.bat"
exit /b 0

:try
if defined ISAAC exit /b 0
if exist "%~1\python.bat" set "ISAAC=%~f1"
if defined ISAAC exit /b 0
if /i "%~nx1"=="python.bat" if exist "%~1" set "ISAAC=%~dp1"
if defined ISAAC exit /b 0
rem one folder deeper: Extract All sometimes creates a second folder
if exist "%~1\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat" set "ISAAC=%~f1\isaac-sim-standalone-5.1.0-windows-x86_64"
exit /b 0


rem ------------------------------------------------ the two Python packages
:check_packages
set "SITE=%ISAAC%\kit\python\Lib\site-packages"
if exist "%SITE%\stable_baselines3\" if exist "%SITE%\gymnasium\" exit /b 0
call "%PY%" -c "import stable_baselines3, gymnasium" >nul 2>&1
if not errorlevel 1 exit /b 0
echo.
echo  First run: installing the two Python packages the project uses into
echo  Isaac Sim (stable-baselines3 2.9.0 and gymnasium 1.2.1). This needs an
echo  internet connection and takes a few minutes.
echo.
call "%PY%" -m pip install stable-baselines3==2.9.0 gymnasium==1.2.1
if not errorlevel 1 exit /b 0
echo.
echo  The installation failed; the messages above say why.
echo  After fixing it, run: run_test.bat setup
exit /b 1
