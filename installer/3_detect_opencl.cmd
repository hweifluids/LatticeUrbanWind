@echo off
setlocal

rem Use LUW_PYTHON to run an embedded Python script that prints OpenCL platform and device info
if not defined LUW_PYTHON (
  echo LUW_PYTHON is not set. Please run your installer script or set it manually.
  goto end
)
if not exist "%LUW_PYTHON%" (
  echo The path pointed to by LUW_PYTHON does not exist: "%LUW_PYTHON%"
  goto end
)

echo Using interpreter:
"%LUW_PYTHON%" --version
echo.

set "PYFILE=%~dp03_detect_opencl.py"
if not exist "%PYFILE%" (
  echo Missing Python file: "%PYFILE%"
  set "RC=5"
  goto end
)
echo Probing OpenCL...
"%LUW_PYTHON%" "%PYFILE%"
set "RC=%ERRORLEVEL%"

:cleanup

:end

echo  --------------------------------------------------
echo    LatticeUrbanWind Installer: OpenCL device check
echo    ENJOY! Huanxia Wei - huanxia.wei@u.nus.edu
echo  --------------------------------------------------

echo.
echo Press any key to continue...
pause >nul

exit /b %RC%
