@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "LUW_ROOT=%%~fI"
set "LUW_HOME=%LUW_ROOT%"
set "LAUNCHER=%LUW_ROOT%\bin\luw.exe"
set "STUDIO=%LUW_ROOT%\bin\luw_studio.exe"
set "CULBM=%LUW_ROOT%\core\cfd_core\CULBM\bin\CULBM.exe"
set "FLUIDX3D=%LUW_ROOT%\core\cfd_core\FluidX3D\bin\FluidX3D.exe"
if not defined LUW_ENABLE_FLUIDX3D set "LUW_ENABLE_FLUIDX3D=OFF"
if /I "%LUW_ENABLE_FLUIDX3D%"=="ON" (
  set "LUW_ENABLE_FLUIDX3D=ON"
) else if /I "%LUW_ENABLE_FLUIDX3D%"=="OFF" (
  set "LUW_ENABLE_FLUIDX3D=OFF"
) else (
  echo LUW_ENABLE_FLUIDX3D must be ON or OFF, got: %LUW_ENABLE_FLUIDX3D%
  exit /b 2
)
set "PATH=%LUW_ROOT%\bin;%SystemRoot%\System32;%SystemRoot%"
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "GDAL_DATA="
set "PROJ_DATA="
set "PROJ_LIB="

if not exist "%LAUNCHER%" (
  echo Native launcher not found: %LAUNCHER%
  exit /b 1
)
if not exist "%CULBM%" (
  echo Default CULBM solver not found: %CULBM%
  exit /b 1
)
if /I "%LUW_ENABLE_FLUIDX3D%"=="ON" if not exist "%FLUIDX3D%" (
  echo Optional FluidX3D solver not found: %FLUIDX3D%
  exit /b 1
)

"%LAUNCHER%" --version
set "VERSION_STATUS=%ERRORLEVEL%"
if not "%VERSION_STATUS%"=="0" exit /b %VERSION_STATUS%

"%LAUNCHER%" --help
set "HELP_STATUS=%ERRORLEVEL%"
if not "%HELP_STATUS%"=="0" exit /b %HELP_STATUS%

"%LAUNCHER%" detect_opencl
set "OPENCL_STATUS=%ERRORLEVEL%"
if "%OPENCL_STATUS%"=="3" (
  echo WARNING: No OpenCL platforms found; native CPU tools remain available.
  set "OPENCL_STATUS=0"
)
if not "%OPENCL_STATUS%"=="0" exit /b %OPENCL_STATUS%

if not exist "%STUDIO%" (
  echo Native GUI not found: %STUDIO%
  exit /b 1
)
"%STUDIO%" -platform offscreen --runtime-self-check
exit /b %ERRORLEVEL%
