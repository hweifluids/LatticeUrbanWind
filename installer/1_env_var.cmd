@echo off
REM Requires Administrator privileges to write system environment

setlocal EnableExtensions EnableDelayedExpansion

REM Compute parent directory of this script
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "LUW_HOME=%%~fI"

echo Detected parent directory:
echo   %LUW_HOME%
set /p CONFIRM=Set this as system variable LUW_HOME and add %%LUW_HOME%%\bin to system PATH. Proceed? [Y/N]:
if /I not "%CONFIRM%"=="Y" if /I not "%CONFIRM%"=="YES" (
  echo Cancelled
  pause
  exit /b 1
)

REM Write system-level LUW_HOME (absolute path)
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" ^
  /v LUW_HOME /t REG_EXPAND_SZ /d "%LUW_HOME%" /f >nul

REM Read current system PATH
for /f "tokens=1,2,*" %%A in ('
  reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path ^| find /i "Path"
') do set "CUR_PATH=%%C"

REM Append %%LUW_HOME%%\bin only if not present (store as relative variable form)
echo !CUR_PATH! | find /I "%%LUW_HOME%%\bin" >nul
if errorlevel 1 (
  reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" ^
    /v Path /t REG_EXPAND_SZ /d "!CUR_PATH!;%%LUW_HOME%%\bin" /f >nul
)

REM Apply to current session
set "LUW_HOME=%LUW_HOME%"
echo !PATH! | find /I "%LUW_HOME%\bin" >nul
if errorlevel 1 set "PATH=%LUW_HOME%\bin;%PATH%"

echo LUW_HOME=%LUW_HOME%
echo PATH=%PATH%
echo  ----------------------------------------------
echo    LatticeUrbanWind Installer: sysvar setup
echo    ENJOY! Huanxia Wei - huanxia.wei@u.nus.edu
echo  ----------------------------------------------

:endPause
echo.
echo Press any key to continue...
pause >nul
exit /b

