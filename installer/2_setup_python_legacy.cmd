@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo  -------------------------------------------------------------
echo    LatticeUrbanWind Installer: Python dependencies - LEGACY
echo    ENJOY! Huanxia Wei - huanxia.wei@u.nus.edu
echo  -------------------------------------------------------------
echo    WARNING: This script installs Python packages globally
echo    on your system Python environment! If you prefer using a
echo    virtual environment, please use "2_setup_python_venv.cmd".
echo  -------------------------------------------------------------
echo.
echo Confirm? [Press Y to proceed, any other key to cancel]
set /p CONFIRM=Proceed with installing Python dependencies? [Y/N]:
if /I not "%CONFIRM%"=="Y" if /I not "%CONFIRM%"=="YES" (
  echo Cancelled
  pause
  exit /b 1
)

REM Use requirements.txt located in the same directory as this script
set "SCRIPT_DIR=%~dp0"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"

if not exist "%REQ_FILE%" (
  echo Cannot find requirements.txt: "%REQ_FILE%"
  goto :confirm_exit
)

REM Search for Python 3 interpreter
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Cannot find Python 3 interpreter in PATH.
  echo Please install Python 3 and ensure it is added to your system PATH.
  goto :confirm_exit
)

REM Ensure pip is available and upgrade if possible
%PY% -m pip --version >nul 2>&1 || %PY% -m ensurepip --upgrade
%PY% -m pip install --upgrade pip

REM Install dependencies, falling back to --user if permissions are restricted
%PY% -m pip install -r "%REQ_FILE%" || %PY% -m pip install --user -r "%REQ_FILE%"

:confirm_exit
echo.
echo Press any key to continue...
pause >nul
endlocal
