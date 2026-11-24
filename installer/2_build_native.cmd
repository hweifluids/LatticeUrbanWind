@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "LUW_ROOT=%%~fI"
set "BUILD_DIR=%LUW_ROOT%\build"
if not defined LUW_BUILD_JOBS set "LUW_BUILD_JOBS=2"
if not defined LUW_ENABLE_FLUIDX3D set "LUW_ENABLE_FLUIDX3D=OFF"
if /I "!LUW_ENABLE_FLUIDX3D!"=="ON" (
  set "LUW_ENABLE_FLUIDX3D=ON"
) else if /I "!LUW_ENABLE_FLUIDX3D!"=="OFF" (
  set "LUW_ENABLE_FLUIDX3D=OFF"
) else (
  echo LUW_ENABLE_FLUIDX3D must be ON or OFF, got: !LUW_ENABLE_FLUIDX3D!
  exit /b 2
)
if not defined VCPKG_MAX_CONCURRENCY set "VCPKG_MAX_CONCURRENCY=%LUW_BUILD_JOBS%"

set "LUW_VCPKG_ROOT=%VCPKG_ROOT%"
if not defined LUW_VCPKG_ROOT if exist "%LUW_ROOT%\third_party\vcpkg\scripts\buildsystems\vcpkg.cmake" set "LUW_VCPKG_ROOT=%LUW_ROOT%\third_party\vcpkg"

rem LUW_CMAKE_PREFIX_PATH is the explicit installer override.  Preserve a
rem caller-provided CMAKE_PREFIX_PATH and add the repository's common sibling
rem dependency layout plus a conventional C:\Qt MSVC installation.
set "LUW_NATIVE_PREFIX_PATH=%LUW_CMAKE_PREFIX_PATH%"
if not defined LUW_NATIVE_PREFIX_PATH set "LUW_NATIVE_PREFIX_PATH=%CMAKE_PREFIX_PATH%"
set "LUW_QT_PREFIX="
if defined SystemDrive for /d %%V in ("%SystemDrive%\Qt\*") do for /d %%Q in ("%%~fV\msvc*_64") do if exist "%%~fQ\lib\cmake\Qt6\Qt6Config.cmake" set "LUW_QT_PREFIX=%%~fQ"
if defined LUW_QT_PREFIX (
  if defined LUW_NATIVE_PREFIX_PATH (
    set "LUW_NATIVE_PREFIX_PATH=!LUW_NATIVE_PREFIX_PATH!;!LUW_QT_PREFIX!"
  ) else (
    set "LUW_NATIVE_PREFIX_PATH=!LUW_QT_PREFIX!"
  )
)
for %%I in ("%LUW_ROOT%\..\deps\VTK-install") do set "LUW_SIBLING_VTK=%%~fI"
if exist "!LUW_SIBLING_VTK!\lib\cmake" (
  if defined LUW_NATIVE_PREFIX_PATH (
    set "LUW_NATIVE_PREFIX_PATH=!LUW_NATIVE_PREFIX_PATH!;!LUW_SIBLING_VTK!"
  ) else (
    set "LUW_NATIVE_PREFIX_PATH=!LUW_SIBLING_VTK!"
  )
)

where cmake >nul 2>&1
if errorlevel 1 (
  echo CMake was not found on PATH.
  exit /b 2
)

rem install_win.cmd is normally launched from an ordinary elevated shell, not
rem a Visual Studio developer prompt.  Initialize the newest installed MSVC
rem environment before CMake selects its default Ninja generator.
where cl >nul 2>&1
if errorlevel 1 (
  set "LUW_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if exist "!LUW_VSWHERE!" (
    set "LUW_VS_INSTALL="
    for /f "usebackq tokens=*" %%I in (`"!LUW_VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "LUW_VS_INSTALL=%%I"
    if defined LUW_VS_INSTALL if exist "!LUW_VS_INSTALL!\Common7\Tools\VsDevCmd.bat" (
      echo Initializing MSVC environment: !LUW_VS_INSTALL!
      call "!LUW_VS_INSTALL!\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
      if errorlevel 1 (
        echo Failed to initialize the Visual Studio C++ environment.
        exit /b 2
      )
    )
  )
)

where cl >nul 2>&1 || where clang-cl >nul 2>&1 || where g++ >nul 2>&1 || where clang++ >nul 2>&1
if errorlevel 1 (
  echo No supported C++ compiler was found. Install the Visual Studio C++ workload, clang, or GCC.
  exit /b 2
)

set "LUW_CONFIGURE_ARGS=-DCMAKE_BUILD_TYPE=Release -DLUW_ENABLE_FLUIDX3D=!LUW_ENABLE_FLUIDX3D!"
if defined LUW_NATIVE_PREFIX_PATH (
  echo Native dependency prefixes: !LUW_NATIVE_PREFIX_PATH!
  set LUW_CONFIGURE_ARGS=!LUW_CONFIGURE_ARGS! "-DCMAKE_PREFIX_PATH=!LUW_NATIVE_PREFIX_PATH!"
)
if defined LUW_VCPKG_ROOT (
  if not exist "%LUW_VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
    echo vcpkg toolchain was not found under: %LUW_VCPKG_ROOT%
    exit /b 2
  )
  echo vcpkg root: !LUW_VCPKG_ROOT!
  set LUW_CONFIGURE_ARGS=!LUW_CONFIGURE_ARGS! "-DCMAKE_TOOLCHAIN_FILE=!LUW_VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_OVERLAY_PORTS=%LUW_ROOT%\third_party\ports"
) else (
  echo vcpkg root was not detected. CMake will use native packages already visible in the configured prefixes.
  echo Set VCPKG_ROOT to a bootstrapped vcpkg checkout if the manifest dependencies are not installed system-wide.
)
call cmake -S "%LUW_ROOT%" -B "%BUILD_DIR%" !LUW_CONFIGURE_ARGS!
if errorlevel 1 exit /b %ERRORLEVEL%
call cmake --build "%BUILD_DIR%" --config Release --parallel "%LUW_BUILD_JOBS%"
if errorlevel 1 exit /b %ERRORLEVEL%
call cmake --install "%BUILD_DIR%" --config Release --prefix "%LUW_ROOT%"
if errorlevel 1 exit /b %ERRORLEVEL%

echo Native LUW build installed to: %LUW_ROOT%\bin
if not defined LUW_INSTALL_ORCHESTRATED pause
exit /b 0
