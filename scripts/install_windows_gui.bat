@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "ENV_FILE=%REPO_ROOT%\environment-gui.yml"
set "ENV_NAME=behavior_suite_gui"
set "STEP=starting"

cd /d "%REPO_ROOT%" || goto fail

set "STEP=checking Conda availability"
where conda >nul 2>nul
if errorlevel 1 (
    echo Conda was not found on PATH.
    echo Install Miniforge or Anaconda once, then run this installer again.
    echo Recommended: https://conda-forge.org/download/
    goto fail
)

set "STEP=checking environment-gui.yml"
if not exist "%ENV_FILE%" (
    echo Missing environment file: "%ENV_FILE%"
    goto fail
)

set "STEP=creating or updating %ENV_NAME% Conda environment"
call conda env update -f "%ENV_FILE%" --prune
if errorlevel 1 (
    echo Failed to create or update the %ENV_NAME% Conda environment.
    goto fail
)

set "STEP=removing stale pip-managed PySide6 packages from %ENV_NAME%"
call conda run -n %ENV_NAME% python -m pip uninstall -y PySide6 PySide6_Addons PySide6_Essentials shiboken6
if errorlevel 1 (
    echo Failed to remove stale pip-managed PySide6 packages from %ENV_NAME%.
    goto fail
)

set "STEP=reinstalling Conda-forge PySide6 runtime"
call conda install -n %ENV_NAME% -c conda-forge --force-reinstall "pyside6=6.11.1" -y
if errorlevel 1 (
    echo Failed to force-reinstall Conda-forge PySide6 6.11.1 in %ENV_NAME%.
    goto fail
)

set "STEP=installing behavior_suite editable package"
call conda run -n %ENV_NAME% python -m pip install -e .
if errorlevel 1 (
    echo Failed to install the current checkout into %ENV_NAME%.
    goto fail
)

set "STEP=verifying editable install points at this checkout"
call conda run -n %ENV_NAME% python -c "import sys; from pathlib import Path; import cli.preprocess as p; repo=Path.cwd().resolve(); loc=Path(p.__file__).resolve(); print('behavior-suite CLI module: ' + str(loc)); sys.exit(0 if repo in loc.parents else 1)"
if errorlevel 1 (
    echo behavior-suite in %ENV_NAME% does not resolve to the current checkout: "%CD%"
    goto fail
)

set "STEP=verifying Conda-forge PySide6 Qt runtime"
call conda run -n %ENV_NAME% python -c "from PySide6.QtWidgets import QApplication; print('PySide6 QtWidgets import: ok')"
if errorlevel 1 (
    echo PySide6/QtWidgets import failed in %ENV_NAME%.
    echo The supported Windows GUI runtime uses Conda-forge PySide6.
    echo The installer attempted to remove stale pip-managed PySide6 packages and reinstall Conda-forge PySide6.
    echo The environment remains unable to load QtWidgets.
    goto fail
)

set "STEP=running behavior-suite doctor"
call conda run -n %ENV_NAME% behavior-suite doctor
if errorlevel 1 (
    echo behavior-suite doctor failed in %ENV_NAME%.
    goto fail
)

echo.
echo behavior_suite Windows GUI runtime is ready.
echo Launch with:
echo   scripts\launch_windows_gui.bat
exit /b 0

:fail
echo.
echo Installation failed during: %STEP%
echo Fix the issue above and run scripts\install_windows_gui.bat again.
if not defined BEHAVIOR_SUITE_NO_PAUSE pause
exit /b 1
