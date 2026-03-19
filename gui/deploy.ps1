param(
    [string]$QtRoot = "C:\Qt\6.8.0\msvc2022_64",
    [string]$VtkRoot = "C:\1_Development\deps\VTK-install",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$targetDir = Join-Path $PSScriptRoot "build-vs2022\$Config"
$exe = Join-Path $targetDir "luw_studio.exe"
$windeployqt = Join-Path $QtRoot "bin\windeployqt.exe"
$vtkBin = Join-Path $VtkRoot "bin"

if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe"
}
if (-not (Test-Path $windeployqt)) {
    throw "windeployqt not found: $windeployqt"
}
if (-not (Test-Path $vtkBin)) {
    throw "VTK bin directory not found: $vtkBin"
}

& $windeployqt --release --force --compiler-runtime $exe
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Get-ChildItem $vtkBin -Filter *.dll | ForEach-Object {
    Copy-Item $_.FullName -Destination (Join-Path $targetDir $_.Name) -Force
}

Write-Host "Deployment completed in $targetDir"
