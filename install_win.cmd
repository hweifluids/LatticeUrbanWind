@echo off
setlocal EnableExtensions EnableDelayedExpansion
title LatticeUrbanWind Native Installer

fltmc >nul 2>&1
if errorlevel 1 (
  echo Requesting administrative privileges...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -Verb RunAs -FilePath '%ComSpec%' -ArgumentList '/c','""%~f0"" %*' -WorkingDirectory '%cd%'"
  exit /b
)

set "EXITCODE=0"
set "LUW_INSTALL_ORCHESTRATED=1"
set "SCRIPT_DIR=%~dp0"
set "INSTALLER_DIR=%SCRIPT_DIR%installer"
pushd "%SCRIPT_DIR%" || (
  echo Failed to access script directory: %SCRIPT_DIR%
  exit /b 1
)

if not exist "%INSTALLER_DIR%\" (
  echo Installer directory not found: %INSTALLER_DIR%
  set "EXITCODE=1"
  goto finish
)

echo Starting staged native installation under elevation...
echo.

echo [0/5] Running 0_detect_env.cmd ...
call "%INSTALLER_DIR%\0_detect_env.cmd"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" (
  echo Step 0 failed with exit code !STEP_RC!. Continuing...
  set "EXITCODE=!STEP_RC!"
)

echo [1/5] Running 1_env_var.cmd ...
call "%INSTALLER_DIR%\1_env_var.cmd"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" (
  echo Step 1 failed with exit code !STEP_RC!. Continuing...
  set "EXITCODE=!STEP_RC!"
)

echo [2/5] Running 2_build_native.cmd ...
call "%INSTALLER_DIR%\2_build_native.cmd"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" (
  echo Step 2 failed with exit code !STEP_RC!. Continuing...
  set "EXITCODE=!STEP_RC!"
)

echo [3/5] Running 3_compile_cfdcore.cmd ...
call "%INSTALLER_DIR%\3_compile_cfdcore.cmd"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" (
  echo Step 3 failed with exit code !STEP_RC!. Continuing...
  set "EXITCODE=!STEP_RC!"
)

echo [4/5] Running 4_testrun.cmd ...
call "%INSTALLER_DIR%\4_testrun.cmd"
set "STEP_RC=!ERRORLEVEL!"
if not "!STEP_RC!"=="0" (
  echo Step 4 failed with exit code !STEP_RC!. Continuing...
  set "EXITCODE=!STEP_RC!"
)

:finish
echo.
if "%EXITCODE%"=="0" (
  echo All installation steps completed successfully.
) else (
  echo Installation steps completed with failures. Final exit code: %EXITCODE%
)
echo.
echo Press any key to exit...
pause >nul
popd 2>nul
endlocal & exit /b %EXITCODE%
