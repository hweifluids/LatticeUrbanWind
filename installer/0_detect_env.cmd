:: ===== Windows CMD script: save as check_env.cmd and run in CMD =====
@echo off
if not defined __HOLD set "__HOLD=1" & cmd /k "%~f0" & exit /b
setlocal ENABLEEXTENSIONS DISABLEDELAYEDEXPANSION

echo ============================================
echo [Python environment]
echo ============================================

set "PYEXE="
for /f "usebackq delims=" %%i in (`where python 2^>NUL`) do if not defined PYEXE set "PYEXE=%%i"
if not defined PYEXE (
  for /f "usebackq delims=" %%i in (`where python3 2^>NUL`) do if not defined PYEXE set "PYEXE=%%i"
)

if defined PYEXE (
  echo Python: FOUND
  echo Executable: %PYEXE%
  for /f "usebackq tokens=1,2 delims= " %%v in (`"%PYEXE%" --version 2^>^&1`) do echo Version: %%w
) else (
  echo Python: NOT FOUND on PATH
  for /f "usebackq delims=" %%i in (`where py 2^>NUL`) do set "PYLAUNCHER=%%i"
  if defined PYLAUNCHER (
    echo Found Python launcher: %PYLAUNCHER%
    for /f "usebackq delims=" %%p in (`py -c "import sys; print(sys.executable)" 2^>NUL`) do set "PYEXE=%%p"
    if defined PYEXE (
      echo Default Python from launcher: %PYEXE%
      for /f "usebackq delims=" %%v in (`py -c "import platform; print(platform.python_version())"`) do echo Version: %%v
    )
  )
)

echo.
echo ============================================
echo [Pip environment]
echo ============================================

set "PIPEXE="
for /f "usebackq delims=" %%i in (`where pip 2^>NUL`) do if not defined PIPEXE set "PIPEXE=%%i"
if not defined PIPEXE (
  for /f "usebackq delims=" %%i in (`where pip3 2^>NUL`) do if not defined PIPEXE set "PIPEXE=%%i"
)

if defined PIPEXE (
  echo Pip: FOUND
  echo Executable: %PIPEXE%
  for /f "usebackq tokens=2 delims= " %%v in (`"%PIPEXE%" --version 2^>^&1`) do echo Version: %%v
  for /f "usebackq tokens=1,* delims= " %%a in (`"%PIPEXE%" --version 2^>^&1`) do (
    rem The default output includes "pip X.Y from PATH (python A.B)"
    for /f "tokens=3" %%x in ('%PIPEXE% --version ^| findstr /i /c:" from "') do rem placeholder
  )
  echo Details: 
  "%PIPEXE%" --version
) else (
  echo Pip: not found as a standalone executable
  if defined PYEXE (
    for /f "usebackq delims=" %%v in (`"%PYEXE%" -m pip --version 2^>NUL`) do (
      echo Pip via python -m: FOUND
      echo %%v
    )
  )
)

echo.
echo ============================================
echo [OpenCL environment]
echo ============================================

set "OCL_DLL=%SystemRoot%\System32\OpenCL.dll"
if exist "%OCL_DLL%" (
  echo OpenCL runtime: FOUND
  echo Runtime library: %OCL_DLL%
) else (
  echo OpenCL runtime: not found at %OCL_DLL%
)

echo.
echo Querying Khronos ICD registry entries if present:
reg query "HKLM\SOFTWARE\Khronos\OpenCL\Vendors" 2>NUL
reg query "HKLM\SOFTWARE\WOW6432Node\Khronos\OpenCL\Vendors" 2>NUL

echo.
for /f "usebackq delims=" %%i in (`where clinfo 2^>NUL`) do set "CLINFO=%%i"
if defined CLINFO (
  echo clinfo tool: FOUND
  echo Executable: %CLINFO%
  "%CLINFO%" --version 2>NUL
) else (
  echo clinfo tool: NOT FOUND on PATH
)


echo.
echo Done.
echo.
echo  ------------------------------------------------
echo    LatticeUrbanWind Installer: environment check
echo    ENJOY! Huanxia Wei - huanxia.wei@u.nus.edu
echo  ------------------------------------------------

:endPause
echo.
echo Press any key to continue...
pause >nul
exit /b
