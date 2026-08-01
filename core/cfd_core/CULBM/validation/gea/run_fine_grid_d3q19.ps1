param(
    [string]$Executable = (
        Join-Path $PSScriptRoot '..\..\build\CULBM-gea-validation.exe'
    ),
    [string]$ResultsRoot = (
        Join-Path $PSScriptRoot 'results'
    ),
    [ValidateSet('primary-control', 'low-mach', 'all')]
    [string]$RunSet = 'primary-control',
    [string]$ExpectedExecutableSha256 = (
        'EF9DEEE7EAB5FA96219409BDAAC11847B6BA21B1D1F491078F9D4C9A8424D0F7'
    )
)

$ErrorActionPreference = 'Stop'
$Executable = [System.IO.Path]::GetFullPath($Executable)
$ResultsRoot = [System.IO.Path]::GetFullPath($ResultsRoot)

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "GEA validation executable not found: $Executable"
}

$actualExecutableSha256 = (
    Get-FileHash -LiteralPath $Executable -Algorithm SHA256
).Hash.ToUpperInvariant()
if (
    $ExpectedExecutableSha256 -and
    $actualExecutableSha256 -ne $ExpectedExecutableSha256.ToUpperInvariant()
) {
    throw (
        "Executable SHA-256 mismatch: expected={0}, actual={1}, path={2}" -f
        $ExpectedExecutableSha256, $actualExecutableSha256, $Executable
    )
}

# The h/2, dt=0.1 runs extend the published-grid sequence directly, but double
# the lattice Mach number relative to the prior finest grid.  The dt=0.05 runs
# preserve approximately the prior lattice Mach and isolate that effect.
$runs = @(
    [pscustomobject]@{
        Group='main'; Case='rising-thermal'; H=7.8125; Dt=0.1
        Thermal='D3Q19'; Viscosity='dynamic'; WriteSlices=$true
    }
    [pscustomobject]@{
        Group='main'; Case='density-current'; H=12.5; Dt=0.1
        Thermal='D3Q19'; Viscosity='dynamic'; WriteSlices=$true
    }
    [pscustomobject]@{
        Group='sensitivity'; Case='rising-thermal'; H=7.8125; Dt=0.05
        Thermal='D3Q19'; Viscosity='dynamic'; WriteSlices=$false
    }
    [pscustomobject]@{
        Group='sensitivity'; Case='density-current'; H=12.5; Dt=0.05
        Thermal='D3Q19'; Viscosity='dynamic'; WriteSlices=$false
    }
    [pscustomobject]@{
        Group='sensitivity'; Case='rising-thermal'; H=7.8125; Dt=0.025
        Thermal='D3Q19'; Viscosity='dynamic'; WriteSlices=$false
    }
    [pscustomobject]@{
        Group='sensitivity'; Case='density-current'; H=12.5; Dt=0.025
        Thermal='D3Q19'; Viscosity='dynamic'; WriteSlices=$false
    }
)
if ($RunSet -eq 'primary-control') {
    $runs = @($runs | Where-Object { $_.Dt -ge 0.05 })
} elseif ($RunSet -eq 'low-mach') {
    $runs = @($runs | Where-Object { $_.Dt -eq 0.025 })
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
    if (-not $Run.WriteSlices) {
        $argumentList += '--no-slices'
    }

    Write-Host (
        "START gpu={0} group={1} case={2} h={3} dt={4} slices={5}" -f
        $Gpu, $Run.Group, $Run.Case, $Run.H, $Run.Dt, $Run.WriteSlices
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
                if (
                    $progress -and
                    $progress -match 'eta_s=[0-9.eE+-]+$' -and
                    $progress -ne $item.LastProgress
                ) {
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
                Select-Object -Last 12
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
            write_slices = $item.Run.WriteSlices
            gpu = $item.Gpu
            executable_sha256 = $actualExecutableSha256
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
$manifestName = if ($RunSet -eq 'low-mach') {
    'suite_manifest_fine_d3q19_low_mach.csv'
} elseif ($RunSet -eq 'all') {
    'suite_manifest_fine_d3q19_all.csv'
} else {
    'suite_manifest_fine_d3q19.csv'
}
$manifestPath = Join-Path $ResultsRoot $manifestName
if (Test-Path -LiteralPath $manifestPath) {
    throw "Refusing to overwrite existing fine-grid manifest: $manifestPath"
}
foreach ($run in $runs) {
    $candidate = Join-Path (
        Join-Path $ResultsRoot $run.Group
    ) (Get-GeaRunName $run)
    if (Test-Path -LiteralPath $candidate) {
        $existing = @(Get-ChildItem -LiteralPath $candidate -Force)
        if ($existing.Count -ne 0) {
            throw (
                "Refusing to start because a run directory is non-empty: {0}" -f
                $candidate
            )
        }
    }
}

$manifest = @()
for ($index = 0; $index -lt $runs.Count; $index += 2) {
    $batch = @(
        Start-GeaRun -Run $runs[$index] -Gpu 0
        Start-GeaRun -Run $runs[$index + 1] -Gpu 1
    )
    $manifest += Wait-GeaBatch -Batch $batch
}

$manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation
$failureCount = @($manifest | Where-Object { $_.exit_code -ne 0 }).Count
Write-Output (
    "FINE_GRID_SUITE_COMPLETE runs={0} failures={1} sha256={2} manifest={3}" -f
    $manifest.Count, $failureCount, $actualExecutableSha256, $manifestPath
)
