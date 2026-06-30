@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "ENV_NAME=behavior_suite_gui"
set "STEP=starting"

cd /d "%REPO_ROOT%" || goto fail

set "STEP=checking Conda availability"
where conda >nul 2>nul
if errorlevel 1 (
    echo Conda was not found on PATH.
    echo Install Miniforge or Anaconda, then run scripts\install_windows_gui.bat.
    goto fail
)

set "STEP=checking %ENV_NAME% environment"
call conda run -n %ENV_NAME% python -c "import sys; print(sys.executable)"
if errorlevel 1 (
    echo The %ENV_NAME% environment is missing or unusable.
    echo Run scripts\install_windows_gui.bat before launching the GUI.
    goto fail
)

set "STEP=running behavior-suite doctor"
call conda run -n %ENV_NAME% behavior-suite doctor
if errorlevel 1 (
    echo The %ENV_NAME% runtime is unsupported.
    echo Run scripts\install_windows_gui.bat to repair the environment.
    goto fail
)

set "STEP=launching behavior-suite GUI"
call conda run --live-stream -n %ENV_NAME% behavior-suite gui
if errorlevel 1 (
    echo behavior-suite GUI exited with an error.
    goto fail
)

exit /b 0

:fail
echo.
echo Launch failed during: %STEP%
echo If the environment is missing or unsupported, run scripts\install_windows_gui.bat.
if not defined BEHAVIOR_SUITE_NO_PAUSE pause
exit /b 1
