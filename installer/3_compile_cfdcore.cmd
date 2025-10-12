@echo off
setlocal

rem ===== User settings =====
set "CONFIG=Release"
set "PLATFORM=x64"
set "SOLUTION=%LUW_HOME%\core\cfd_core\FluidX3D\FluidX3D.sln"
rem =========================

rem Validate solution path
if not exist "%SOLUTION%" (
  echo Solution not found: "%SOLUTION%"
  exit /b 1
)

rem Locate MSBuild via vswhere if available
set "MSBUILD="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`
    "%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe
  `) do (
    set "MSBUILD=%%I"
  )
)

rem Fallback to common MSBuild locations
if not defined MSBUILD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe" set "MSBUILD=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
if not defined MSBUILD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"   set "MSBUILD=%ProgramFiles%\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
if not defined MSBUILD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe" set "MSBUILD=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
if not defined MSBUILD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"   set "MSBUILD=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
if not defined MSBUILD if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe" set "MSBUILD=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
if not defined MSBUILD if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"   set "MSBUILD=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"

if not defined MSBUILD (
  echo MSBuild.exe not found. Install Visual Studio Build Tools or run from a Developer Command Prompt.
  exit /b 2
)

echo Using MSBuild: "%MSBUILD%"
"%MSBUILD%" "%SOLUTION%" /t:Build /p:Configuration=%CONFIG%;Platform=%PLATFORM% /m
if errorlevel 1 (
  echo Build failed. ErrorLevel %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)

echo Build succeeded.
echo.
echo  ----------------------------------------------
echo    LatticeUrbanWind Installer: cfdCore Maker
echo    ENJOY! Huanxia Wei - huanxia.wei@u.nus.edu
echo  ----------------------------------------------

:endPause
echo.
echo Press any key to continue...
pause >nul
exit /b
exit /b 0
