@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "LUW_ROOT=%%~fI"
set "LAUNCHER=%LUW_ROOT%\bin\luw.exe"
set "RC=0"

if not exist "%LAUNCHER%" (
  echo Native launcher not found: %LAUNCHER%
  set "RC=2"
  goto report
)

echo Probing OpenCL...
"%LAUNCHER%" detect_opencl
set "RC=%ERRORLEVEL%"

:report
echo  --------------------------------------------------
echo    LatticeUrbanWind Installer: OpenCL device check
echo    ENJOY! Huanxia Wei - huanxia.wei@u.nus.edu
echo  --------------------------------------------------

if not defined LUW_INSTALL_ORCHESTRATED (
  echo.
  echo Press any key to continue...
  pause >nul
)
exit /b %RC%
