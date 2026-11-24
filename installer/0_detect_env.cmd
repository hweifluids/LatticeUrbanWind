@echo off
setlocal EnableExtensions

echo ============================================
echo [Native build environment]
echo ============================================

where cmake >nul 2>&1
if errorlevel 1 (
  echo CMake: NOT FOUND on PATH
  set "RC=2"
) else (
  echo CMake: FOUND
  cmake --version
  set "RC=0"
)

where cl >nul 2>&1 && echo MSVC compiler: FOUND
where clang-cl >nul 2>&1 && echo clang-cl compiler: FOUND
where g++ >nul 2>&1 && echo GNU C++ compiler: FOUND

echo.
echo ============================================
echo [OpenCL environment]
echo ============================================
if exist "%SystemRoot%\System32\OpenCL.dll" (
  echo OpenCL runtime: FOUND
  echo Runtime library: %SystemRoot%\System32\OpenCL.dll
) else (
  echo OpenCL runtime: NOT FOUND at %SystemRoot%\System32\OpenCL.dll
)
reg query "HKLM\SOFTWARE\Khronos\OpenCL\Vendors" 2>nul
reg query "HKLM\SOFTWARE\WOW6432Node\Khronos\OpenCL\Vendors" 2>nul
where clinfo >nul 2>&1 && clinfo --version

echo.
echo  ------------------------------------------------
echo    LatticeUrbanWind Installer: environment check
echo    ENJOY! Huanxia Wei - huanxia.wei@u.nus.edu
echo  ------------------------------------------------
if not defined LUW_INSTALL_ORCHESTRATED pause
exit /b %RC%
