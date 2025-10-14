# runluw.ps1
# Run FluidX3D using LUW_HOME environment variable and forward all arguments.

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsFromUser
)

# Build the executable path from LUW_HOME
$exe = Join-Path -Path $env:LUW_HOME -ChildPath 'core\cfd_core\FluidX3D\bin\FluidX3D.exe'

# Validate path existence
if (-not (Test-Path -Path $exe -PathType Leaf)) {
    Write-Error "FluidX3D.exe not found at: $exe. Please verify LUW_HOME and the installation layout."
    exit 1
}

# Run inside the executable directory to keep relative resources working
$exeDir = Split-Path -Path $exe -Parent
Push-Location $exeDir
try {
    if ($ArgsFromUser -and $ArgsFromUser.Count -gt 0) {
        & $exe @ArgsFromUser
    } else {
        & $exe
    }
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

exit $code
