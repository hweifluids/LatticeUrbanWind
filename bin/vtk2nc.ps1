param([Parameter(Mandatory=$false, Position=0)][string]$ConfPath)

if (-not $env:LUW_HOME) { Write-Error "LUW_HOME is not set"; exit 2 }

$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$target = Join-Path $env:LUW_HOME "core\tools_core\vtk2nc.py"
if (-not (Test-Path $target)) { Write-Error "Target script not found: $target"; exit 2 }

if (-not $ConfPath -or $ConfPath.Trim() -eq "") {
    $cwd = Get-Location
    $candidate = Join-Path $cwd.Path "conf.luw"
    if (Test-Path $candidate) {
        $ConfPath = $candidate
    } else {
        $cands = Get-ChildItem -Path $cwd.Path -Filter *.luw -File | Select-Object -First 2
        if ($cands.Count -eq 1) {
            $ConfPath = $cands[0].FullName
        } else {
            Write-Error "Usage: vtk2nc <path-to-conf.luw>. For auto mode, no conf.luw in current directory and no unique *.luw found."
            exit 2
        }
    }
}

try {
    $ConfPath = (Resolve-Path -LiteralPath $ConfPath).Path
} catch {
    Write-Error "Conf file not found: $ConfPath"
    exit 2
}

& $py $target $ConfPath
