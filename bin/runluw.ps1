# runluw.ps1
# Run FluidX3D using LUW_HOME. First argument can be a .luw/.luwdg/.luwpf path.

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

# Resolve first argument to absolute path before changing working directory.
# This allows: runluw conf.luw from the case folder.
$forwardArgs = @()
if ($ArgsFromUser -and $ArgsFromUser.Count -gt 0) {
    $firstArg = $ArgsFromUser[0]
    if (-not [string]::IsNullOrWhiteSpace($firstArg)) {
        try {
            if (Test-Path -LiteralPath $firstArg) {
                $forwardArgs += (Resolve-Path -LiteralPath $firstArg).Path
            } else {
                $forwardArgs += [System.IO.Path]::GetFullPath((Join-Path -Path (Get-Location).Path -ChildPath $firstArg))
            }
        } catch {
            $forwardArgs += $firstArg
        }
    }
    if ($ArgsFromUser.Count -gt 1) {
        $forwardArgs += $ArgsFromUser[1..($ArgsFromUser.Count - 1)]
    }
}

# Run inside the executable directory to keep relative resources working
$exeDir = Split-Path -Path $exe -Parent
Push-Location $exeDir
try {
    if ($forwardArgs -and $forwardArgs.Count -gt 0) {
        & $exe @forwardArgs
    } else {
        & $exe
    }
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

exit $code
