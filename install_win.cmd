@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Installer Orchestrator (elevated)

rem ===== Check and elevate privileges =====
rem Use a driver-management command that requires admin to test elevation.
fltmc >nul 2>&1
if errorlevel 1 (
  echo Requesting administrative privileges...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -Verb RunAs -FilePath '%ComSpec%' -ArgumentList '/c','""%~f0"" %*' -WorkingDirectory '%cd%'"
  exit /b
)

rem ===== Initialized as elevated =====
set "EXITCODE=0"
set "__HOLD=1"


rem Normalize working directory to the script's directory.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" || (
  echo Failed to access script directory: %SCRIPT_DIR%
  set "EXITCODE=1"
  goto wait_exit
)

set "INSTALLER_DIR=%SCRIPT_DIR%installer"
if not exist "%INSTALLER_DIR%\" (
  echo Installer directory not found: "%INSTALLER_DIR%"
  set "EXITCODE=1"
  goto wait_exit
)

echo Starting staged execution under elevation...
echo.

rem ===== Stage 0 =====
echo [0/5] Running 0_detect_env.cmd ...
call "%INSTALLER_DIR%\0_detect_env.cmd"
if errorlevel 1 (
  echo Step 0 failed with exit code %errorlevel%. Continuing...
  set "EXITCODE=%errorlevel%"
)


rem ===== Stage 1 =====
echo [1/5] Running 1_env_var.cmd ...
call "%INSTALLER_DIR%\1_env_var.cmd"
if errorlevel 1 (
  echo Step 1 failed with exit code %errorlevel%. Continuing...
  set "EXITCODE=%errorlevel%"
)


rem ===== Stage 2 =====
echo [2/5] Running 2_setup_python_venv.cmd ...
call "%INSTALLER_DIR%\2_setup_python_venv.cmd"
if errorlevel 1 (
  echo Step 2 failed with exit code %errorlevel%. Continuing...
  set "EXITCODE=%errorlevel%"
)


rem ===== Stage 3 =====
echo [3/5] Running 3_compile_cfdcore.cmd ...
call "%INSTALLER_DIR%\3_compile_cfdcore.cmd"
if errorlevel 1 (
  echo Step 3 failed with exit code %errorlevel%. Continuing...
  set "EXITCODE=%errorlevel%"
)

rem ===== Stage 4 =====
echo [4/5] Running 4_testrun.cmd ...
call "%INSTALLER_DIR%\4_testrun.cmd"
if errorlevel 1 (
  echo Step 4 failed with exit code %errorlevel%. Continuing...
  set "EXITCODE=%errorlevel%"
)

echo.
echo All steps completed successfully.
goto wait_exit

:wait_exit
echo.
echo Press any key to exit...
pause >nul

rem Restore previous directory and exit with the recorded code.
popd 2>nul
endlocal & exit /b %EXITCODE%
