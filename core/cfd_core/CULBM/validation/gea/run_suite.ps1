param(
    [string]$Executable = (
        Join-Path $PSScriptRoot '..\..\build\CULBM-gea-validation.exe'
    ),
    [string]$ResultsRoot = (
        Join-Path $PSScriptRoot 'results'
    ),
    [ValidateSet('all', 'rising-thermal', 'density-current')]
    [string]$CaseFilter = 'all',
    [switch]$MainOnly
)

$ErrorActionPreference = 'Stop'
$Executable = [System.IO.Path]::GetFullPath($Executable)
$ResultsRoot = [System.IO.Path]::GetFullPath($ResultsRoot)

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "GEA validation executable not found: $Executable"
}

$mainRuns = @(
    [pscustomobject]@{ Group='main'; Case='rising-thermal'; H=250.0; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='main'; Case='density-current'; H=400.0; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='main'; Case='rising-thermal'; H=125.0; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='main'; Case='density-current'; H=200.0; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='main'; Case='rising-thermal'; H=62.5; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='main'; Case='density-current'; H=100.0; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='main'; Case='rising-thermal'; H=31.25; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='main'; Case='density-current'; H=50.0; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='main'; Case='rising-thermal'; H=15.625; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='main'; Case='density-current'; H=25.0; Dt=0.1; Thermal='D3Q19'; Viscosity='dynamic' }
)

$sensitivityRuns = @(
    [pscustomobject]@{ Group='sensitivity'; Case='rising-thermal'; H=15.625; Dt=0.05; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='sensitivity'; Case='density-current'; H=25.0; Dt=0.05; Thermal='D3Q19'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='sensitivity'; Case='rising-thermal'; H=15.625; Dt=0.1; Thermal='D3Q7'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='sensitivity'; Case='density-current'; H=25.0; Dt=0.1; Thermal='D3Q7'; Viscosity='dynamic' }
    [pscustomobject]@{ Group='sensitivity'; Case='rising-thermal'; H=31.25; Dt=0.1; Thermal='D3Q19'; Viscosity='numeric' }
    [pscustomobject]@{ Group='sensitivity'; Case='density-current'; H=50.0; Dt=0.1; Thermal='D3Q19'; Viscosity='numeric' }
)

$runs = @($mainRuns)
if (-not $MainOnly) {
    $runs += $sensitivityRuns
}
if ($CaseFilter -ne 'all') {
    $runs = @($runs | Where-Object { $_.Case -eq $CaseFilter })
}
if ($runs.Count -eq 0) {
    throw "No runs selected for CaseFilter=$CaseFilter"
}

function Convert-Token {
    param([double]$Value)
    return $Value.ToString(
        '0.############',
        [System.Globalization.CultureInfo]::InvariantCulture
    ).Replace('.', 'p')
}

function Get-GeaRunName {
    param([pscustomobject]$Run)
    $caseToken = $Run.Case.Replace('-', '_')
    $hToken = Convert-Token $Run.H
    $dtToken = Convert-Token $Run.Dt
    $thermalToken = $Run.Thermal.ToLowerInvariant()
    return "${caseToken}_h${hToken}_dt${dtToken}_${thermalToken}_$($Run.Viscosity)"
}

function Start-GeaRun {
    param(
        [pscustomobject]$Run,
        [int]$Gpu
    )

    $name = Get-GeaRunName $Run
    $runDirectory = Join-Path (Join-Path $ResultsRoot $Run.Group) $name
    if (Test-Path -LiteralPath $runDirectory) {
        $existing = @(Get-ChildItem -LiteralPath $runDirectory -Force)
        if ($existing.Count -ne 0) {
            throw (
                "Refusing to reuse non-empty run directory (stale output risk): {0}" -f
                $runDirectory
            )
        }
    }
    New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null

    $stdout = Join-Path $runDirectory 'console.stdout.log'
    $stderr = Join-Path $runDirectory 'console.stderr.log'
    $argumentList = @(
        '--case', $Run.Case,
        '--h', $Run.H.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--dt', $Run.Dt.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--sample-seconds', '5',
        '--thermal', $Run.Thermal,
        '--viscosity', $Run.Viscosity,
        '--output', $runDirectory
    )

    Write-Host (
        "START gpu={0} group={1} case={2} h={3} dt={4} thermal={5} viscosity={6}" -f
        $Gpu, $Run.Group, $Run.Case, $Run.H, $Run.Dt,
        $Run.Thermal, $Run.Viscosity
    )
    $process = Start-Process `
        -FilePath $Executable `
        -ArgumentList $argumentList `
        -Environment @{ CUDA_VISIBLE_DEVICES = "$Gpu" } `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr

    return [pscustomobject]@{
        Run = $Run
        Gpu = $Gpu
        Name = $name
        Directory = $runDirectory
        Stdout = $stdout
        Stderr = $stderr
        Process = $process
        StartTime = Get-Date
        LastProgress = ''
    }
}

function Wait-GeaBatch {
    param([object[]]$Batch)

    while ($true) {
        $allFinished = $true
        foreach ($item in $Batch) {
            $item.Process.Refresh()
            if (-not $item.Process.HasExited) {
                $allFinished = $false
            }
            if (Test-Path -LiteralPath $item.Stdout) {
                $progress = Get-Content -LiteralPath $item.Stdout |
                    Where-Object { $_ -like 'GEA_PROGRESS*' } |
                    Select-Object -Last 1
                if ($progress -and $progress -ne $item.LastProgress) {
                    Write-Host "STATUS gpu=$($item.Gpu) run=$($item.Name) $progress"
                    $item.LastProgress = $progress
                }
            }
        }
        if ($allFinished) {
            break
        }
        Start-Sleep -Seconds 5
    }

    $records = @()
    foreach ($item in $Batch) {
        $item.Process.WaitForExit()
        $elapsed = ((Get-Date) - $item.StartTime).TotalSeconds
        $stderrTail = ''
        if (Test-Path -LiteralPath $item.Stderr) {
            $stderrTail = (
                Get-Content -LiteralPath $item.Stderr |
                Select-Object -Last 8
            ) -join [Environment]::NewLine
        }
        Write-Host (
            "DONE gpu={0} run={1} exit={2} elapsed_s={3:F1}" -f
            $item.Gpu, $item.Name, $item.Process.ExitCode, $elapsed
        )
        if ($item.Process.ExitCode -ne 0) {
            Write-Warning (
                "GEA run failed: {0} (exit {1})`n{2}" -f
                $item.Name, $item.Process.ExitCode, $stderrTail
            )
        }
        $records += [pscustomobject]@{
            group = $item.Run.Group
            case = $item.Run.Case
            h_m = $item.Run.H
            dt_s = $item.Run.Dt
            thermal = $item.Run.Thermal
            viscosity = $item.Run.Viscosity
            gpu = $item.Gpu
            exit_code = $item.Process.ExitCode
            status = if ($item.Process.ExitCode -eq 0) { 'passed' } else { 'failed' }
            elapsed_s = [math]::Round($elapsed, 3)
            output_directory = $item.Directory
            error_tail = $stderrTail
        }
    }
    return $records
}

New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
foreach ($run in $runs) {
    $candidate = Join-Path (
        Join-Path $ResultsRoot $run.Group
    ) (Get-GeaRunName $run)
    if (Test-Path -LiteralPath $candidate) {
        $existing = @(Get-ChildItem -LiteralPath $candidate -Force)
        if ($existing.Count -ne 0) {
            throw (
                "Refusing to start suite because a run directory is non-empty: {0}" -f
                $candidate
            )
        }
    }
}
$manifest = @()
for ($index = 0; $index -lt $runs.Count; $index += 2) {
    $batch = @()
    $batch += Start-GeaRun -Run $runs[$index] -Gpu 0
    if ($index + 1 -lt $runs.Count) {
        $batch += Start-GeaRun -Run $runs[$index + 1] -Gpu 1
    }
    $manifest += Wait-GeaBatch -Batch $batch
}

$manifestName = if ($CaseFilter -eq 'all') {
    'suite_manifest.csv'
} else {
    "suite_manifest_$($CaseFilter.Replace('-', '_')).csv"
}
$manifestPath = Join-Path $ResultsRoot $manifestName
$manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation
$failureCount = @($manifest | Where-Object { $_.exit_code -ne 0 }).Count
Write-Output "SUITE_COMPLETE runs=$($manifest.Count) failures=$failureCount manifest=$manifestPath"
