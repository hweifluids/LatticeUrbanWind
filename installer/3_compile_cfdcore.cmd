@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ===== User settings =====
set "CONFIG=Release"
set "PLATFORM=x64"
set "SOLUTION=%LUW_HOME%\core\cfd_core\FluidX3D\FluidX3D.sln"
set "PROJECT=%LUW_HOME%\core\cfd_core\FluidX3D\FluidX3D.vcxproj"
rem =========================

rem If LUW_HOME is not set for this session, infer it from the script location.
if not defined LUW_HOME (
  set "SCRIPT_DIR=%~dp0"
  for %%I in ("!SCRIPT_DIR!\..") do set "LUW_HOME=%%~fI"
  set "SOLUTION=!LUW_HOME!\core\cfd_core\FluidX3D\FluidX3D.sln"
  set "PROJECT=!LUW_HOME!\core\cfd_core\FluidX3D\FluidX3D.vcxproj"
)

rem Validate solution path
if not exist "%SOLUTION%" (
  echo Solution not found: "%SOLUTION%"
  exit /b 1
)

rem Locate MSBuild via vswhere if available
set "MSBUILD="
set "VSINSTALL="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`
    "%VSWHERE%" -latest -products * -property installationPath
  `) do (
    set "VSINSTALL=%%I"
  )
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

if not defined VSINSTALL if defined MSBUILD (
  for %%I in ("%MSBUILD%") do set "MSBUILD_DIR=%%~dpI"
  for %%I in ("!MSBUILD_DIR!\..\..\..") do set "VSINSTALL=%%~fI"
)

rem Read the toolset requested by the project and fall back to the newest installed one if needed.
set "PROJECT_TOOLSET="
if exist "%PROJECT%" (
  for /f "tokens=3 delims=<>" %%I in ('findstr /R /C:"<PlatformToolset>.*</PlatformToolset>" "%PROJECT%"') do (
    set "PROJECT_TOOLSET=%%I"
    goto :toolsetRequested
  )
)
:toolsetRequested

set "PLATFORM_TOOLSET="
if defined VSINSTALL if exist "!VSINSTALL!\VC\Auxiliary\Build" (
  if defined PROJECT_TOOLSET if exist "!VSINSTALL!\VC\Auxiliary\Build\Microsoft.VCToolsVersion.!PROJECT_TOOLSET!.default.props" (
    set "PLATFORM_TOOLSET=!PROJECT_TOOLSET!"
  ) else (
    for /f "delims=" %%I in ('dir /b /o-n "!VSINSTALL!\VC\Auxiliary\Build\Microsoft.VCToolsVersion.v*.default.props" 2^>nul') do (
      set "TOOLSET_FILE=%%~nI"
      set "PLATFORM_TOOLSET=!TOOLSET_FILE:Microsoft.VCToolsVersion.=!"
      set "PLATFORM_TOOLSET=!PLATFORM_TOOLSET:.default=!"
      goto :toolsetResolved
    )
  )
)
:toolsetResolved

echo Using MSBuild: "%MSBUILD%"
if defined PROJECT_TOOLSET if defined PLATFORM_TOOLSET if /I not "!PROJECT_TOOLSET!"=="!PLATFORM_TOOLSET!" (
  echo Project requests PlatformToolset "!PROJECT_TOOLSET!", using installed toolset "!PLATFORM_TOOLSET!".
)

set "MSBUILD_PROPS=/p:Configuration=%CONFIG%;Platform=%PLATFORM%"
if defined PLATFORM_TOOLSET set "MSBUILD_PROPS=!MSBUILD_PROPS!;PlatformToolset=!PLATFORM_TOOLSET!"

"%MSBUILD%" "%SOLUTION%" /t:Build !MSBUILD_PROPS! /m
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
