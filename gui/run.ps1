param(
    [string]$QtRoot = "C:\Qt\6.8.0\msvc2022_64",
    [string]$VtkRoot = "C:\1_Development\deps\VTK-install",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$exe = Join-Path $PSScriptRoot "build-vs2022\$Config\luw_studio.exe"
if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe"
}

$env:PATH = (Join-Path $QtRoot "bin") + ";" + (Join-Path $VtkRoot "bin") + ";" + $env:PATH
& $exe
