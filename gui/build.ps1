param(
    [string]$QtRoot = "C:\Qt\6.8.0\msvc2022_64",
    [string]$VtkRoot = "C:\1_Development\deps\VTK-install",
    [string]$Config = "Release",
    [switch]$Deploy
)

$ErrorActionPreference = "Stop"

$cmake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$sourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $sourceDir "build-vs2022"
$vtkDir = Join-Path $VtkRoot "lib\cmake\vtk-9.3"
$qtDir = Join-Path $QtRoot "lib\cmake\Qt6"

& $cmake `
    -S $sourceDir `
    -B $buildDir `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_PREFIX_PATH="$QtRoot;$VtkRoot" `
    -DQt6_DIR="$qtDir" `
    -DVTK_DIR="$vtkDir"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $cmake --build $buildDir --config $Config -- /m
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Deploy) {
    & (Join-Path $sourceDir "deploy.ps1") -QtRoot $QtRoot -VtkRoot $VtkRoot -Config $Config
    exit $LASTEXITCODE
}

exit 0
