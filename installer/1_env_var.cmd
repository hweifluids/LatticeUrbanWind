@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "LUW_ROOT=%%~fI"

echo Detected parent directory:
echo   %LUW_ROOT%
set /p CONFIRM=Set this as system variable LUW_HOME and add %%LUW_HOME%%\bin to system PATH. Proceed? [Y/N]:
if /I not "%CONFIRM%"=="Y" if /I not "%CONFIRM%"=="YES" (
  echo Cancelled
  exit /b 1
)

reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v LUW_HOME /t REG_EXPAND_SZ /d "%LUW_ROOT%" /f >nul
if errorlevel 1 exit /b %ERRORLEVEL%

set "CUR_PATH="
for /f "tokens=1,2,*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path ^| find /i "Path"') do set "CUR_PATH=%%C"
echo !CUR_PATH! | find /I "%%LUW_HOME%%\bin" >nul
if errorlevel 1 (
  reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path /t REG_EXPAND_SZ /d "!CUR_PATH!;%%LUW_HOME%%\bin" /f >nul
  if errorlevel 1 exit /b !ERRORLEVEL!
)

echo LUW_HOME=%LUW_ROOT%
echo System PATH contains %%LUW_HOME%%\bin
if not defined LUW_INSTALL_ORCHESTRATED pause
exit /b 0
