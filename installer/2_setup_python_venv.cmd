@echo off

rem Check for admin rights
>nul 2>&1 net session
if %errorlevel% neq 0 (
  echo Applying administrator privileges...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

setlocal

rem Target directory and files
set "VENV_DIR=%LUW_HOME%\.venv"
set "REQ_FILE=%~dp0requirements.txt"

rem Ensure LUW_HOME directory exists
if not exist "%LUW_HOME%" (
  echo %LUW_HOME% not found. Creating it.
  mkdir "%LUW_HOME%" 2>nul
)

rem Detect a usable Python launcher
set "PYCMD="
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PYCMD=py -3"
) else (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (
    set "PYCMD=python"
  )
)

if not defined PYCMD (
  echo Python not found. Please install Python and ensure it is on PATH.
  goto endPause
)

rem Create or reuse the virtual environment
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating virtual environment: %VENV_DIR%
  %PYCMD% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo Failed to create virtual environment.
    goto endPause
  )
) else (
  echo Existing virtual environment detected. Reusing it.
)

rem Upgrade pip inside the venv
echo Upgrading pip inside the virtual environment.
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip

rem Install dependencies
if exist "%REQ_FILE%" (
  echo Installing dependencies from: %REQ_FILE%
  "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQ_FILE%"
) else (
  echo requirements.txt not found next to this script. Skipping dependency installation.
)

echo Done. Virtual environment is at "%VENV_DIR%"

rem Set LUW_PYTHON environment variable
set "LUW_PYTHON=%VENV_DIR%\Scripts\python.exe"
setx /M LUW_PYTHON "%VENV_DIR%\Scripts\python.exe" >nul
echo LUW_PYTHON has been set to "%LUW_PYTHON%"
echo Testing LUW_PYTHON version:
"%LUW_PYTHON%" --version
if errorlevel 1 (
  echo [ERROR] Unsuccessful test of LUW_PYTHON.
) else (
  echo LUW_PYTHON successfully tested.
)
echo  ------------------------------------------------
echo    LatticeUrbanWind Installer: Python venv setup
echo    ENJOY! Huanxia Wei - huanxia.wei@u.nus.edu
echo  ------------------------------------------------

:endPause
echo.
echo Press any key to continue...
pause >nul
exit /b
