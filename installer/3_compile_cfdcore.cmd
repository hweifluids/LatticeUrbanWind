@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "LUW_ROOT=%%~fI"
set "CULBM_ROOT=%LUW_ROOT%\core\cfd_core\CULBM"
set "CULBM_BUILD=%CULBM_ROOT%\build-installer"
set "SOLUTION=%LUW_ROOT%\core\cfd_core\FluidX3D\FluidX3D.sln"
set "PROJECT=%LUW_ROOT%\core\cfd_core\FluidX3D\FluidX3D.vcxproj"
if not defined LUW_CFD_BUILD_JOBS (
  if defined LUW_BUILD_JOBS (
    set "LUW_CFD_BUILD_JOBS=!LUW_BUILD_JOBS!"
  ) else (
    set "LUW_CFD_BUILD_JOBS=2"
  )
)
if not defined LUW_CULBM_REQUIRE_MPI set "LUW_CULBM_REQUIRE_MPI=OFF"
if not defined LUW_ENABLE_FLUIDX3D set "LUW_ENABLE_FLUIDX3D=OFF"
if /I "!LUW_ENABLE_FLUIDX3D!"=="ON" (
  set "LUW_ENABLE_FLUIDX3D=ON"
) else if /I "!LUW_ENABLE_FLUIDX3D!"=="OFF" (
  set "LUW_ENABLE_FLUIDX3D=OFF"
) else (
  echo LUW_ENABLE_FLUIDX3D must be ON or OFF, got: !LUW_ENABLE_FLUIDX3D!
  exit /b 2
)

set "VSINSTALL="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
)
where cl >nul 2>&1
if errorlevel 1 if defined VSINSTALL if exist "!VSINSTALL!\Common7\Tools\VsDevCmd.bat" (
  echo Initializing MSVC environment: !VSINSTALL!
  call "!VSINSTALL!\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
  if errorlevel 1 (
    echo Failed to initialize the Visual Studio C++ environment.
    exit /b 2
  )
)

where cmake >nul 2>&1
if errorlevel 1 (
  echo CMake was not found on PATH; CULBM is the default solver and cannot be built.
  exit /b 2
)
if not exist "%CULBM_ROOT%\CMakeLists.txt" (
  echo CULBM source tree not found: %CULBM_ROOT%
  exit /b 1
)

echo Configuring the default CULBM solver...
if defined LUW_CULBM_GENERATOR (
  call cmake -S "%CULBM_ROOT%" -B "%CULBM_BUILD%" -G "!LUW_CULBM_GENERATOR!" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCULBM_REQUIRE_MPI="!LUW_CULBM_REQUIRE_MPI!"
) else (
  call cmake -S "%CULBM_ROOT%" -B "%CULBM_BUILD%" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCULBM_REQUIRE_MPI="!LUW_CULBM_REQUIRE_MPI!"
)
if errorlevel 1 exit /b %ERRORLEVEL%
call cmake --build "%CULBM_BUILD%" --config Release --target CULBM --parallel "!LUW_CFD_BUILD_JOBS!"
if errorlevel 1 exit /b %ERRORLEVEL%
if not exist "%CULBM_ROOT%\bin\CULBM.exe" (
  echo CULBM build completed without producing: %CULBM_ROOT%\bin\CULBM.exe
  exit /b 1
)
echo CULBM default solver build succeeded.

if /I not "!LUW_ENABLE_FLUIDX3D!"=="ON" (
  echo FluidX3D build skipped. Set LUW_ENABLE_FLUIDX3D=ON to build that separate optional backend.
  if not defined LUW_INSTALL_ORCHESTRATED pause
  exit /b 0
)

if not exist "%SOLUTION%" (
  echo Solution not found: %SOLUTION%
  exit /b 1
)

set "MSBUILD="
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe`) do set "MSBUILD=%%I"
)
if not defined MSBUILD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe" set "MSBUILD=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
if not defined MSBUILD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" set "MSBUILD=%ProgramFiles%\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
if not defined MSBUILD (
  echo MSBuild.exe not found. Install Visual Studio Build Tools.
  exit /b 2
)

set "PROJECT_TOOLSET="
for /f "tokens=3 delims=^<^>" %%I in ('findstr /R /C:"^<PlatformToolset^>.*^</PlatformToolset^>" "%PROJECT%"') do if not defined PROJECT_TOOLSET set "PROJECT_TOOLSET=%%I"
set "PLATFORM_TOOLSET="
if defined VSINSTALL if exist "!VSINSTALL!\VC\Auxiliary\Build" (
  if defined PROJECT_TOOLSET if exist "!VSINSTALL!\VC\Auxiliary\Build\Microsoft.VCToolsVersion.!PROJECT_TOOLSET!.default.props" set "PLATFORM_TOOLSET=!PROJECT_TOOLSET!"
  if not defined PLATFORM_TOOLSET for /f "delims=" %%I in ('dir /b /o-n "!VSINSTALL!\VC\Auxiliary\Build\Microsoft.VCToolsVersion.v*.default.props" 2^>nul') do if not defined PLATFORM_TOOLSET (
    set "TOOLSET_FILE=%%~nI"
    set "PLATFORM_TOOLSET=!TOOLSET_FILE:Microsoft.VCToolsVersion.=!"
    set "PLATFORM_TOOLSET=!PLATFORM_TOOLSET:.default=!"
  )
)

set "PROPERTIES=/p:Configuration=Release;Platform=x64"
if defined PLATFORM_TOOLSET set "PROPERTIES=!PROPERTIES!;PlatformToolset=!PLATFORM_TOOLSET!"
echo Using MSBuild: %MSBUILD%
echo CFD build concurrency: !LUW_CFD_BUILD_JOBS!
"%MSBUILD%" "%SOLUTION%" /t:Build !PROPERTIES! /m:!LUW_CFD_BUILD_JOBS!
if errorlevel 1 exit /b %ERRORLEVEL%

echo FluidX3D optional solver build succeeded.
if not defined LUW_INSTALL_ORCHESTRATED pause
exit /b 0
