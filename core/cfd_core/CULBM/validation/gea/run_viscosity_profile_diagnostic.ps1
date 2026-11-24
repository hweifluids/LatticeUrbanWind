param(
    [string]$Executable = (
        Join-Path $PSScriptRoot '..\..\build\CULBM-gea-validation.exe'
    ),
    [string]$ResultsRoot = (
        Join-Path $PSScriptRoot 'results\viscosity_profile'
    )
)

$ErrorActionPreference = 'Stop'
$Executable = [System.IO.Path]::GetFullPath($Executable)
$ResultsRoot = [System.IO.Path]::GetFullPath($ResultsRoot)

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "GEA validation executable not found: $Executable"
}

# The GEA hydrostatic profile gives approximately 0.90 kg/m^3 at z=3 km
# and 0.55 kg/m^3 at z=8 km. With mu_a=15 Pa s these correspond to
# nu=16.67 and 27.27 m^2/s, compared with the primary ground-density
# mapping nu=12.91 m^2/s.
$runs = @(
    [pscustomobject]@{ Label='rho_z3km'; ReferenceDensity=0.90; Gpu=0 }
    [pscustomobject]@{ Label='rho_z8km'; ReferenceDensity=0.55; Gpu=1 }
)

foreach ($run in $runs) {
    $runDirectory = Join-Path $ResultsRoot (
        "rising_thermal_h31p25_dt0p1_d3q19_dynamic_$($run.Label)"
    )
    if (Test-Path -LiteralPath $runDirectory) {
        $existing = @(Get-ChildItem -LiteralPath $runDirectory -Force)
        if ($existing.Count -ne 0) {
            throw (
                "Refusing to reuse non-empty run directory (stale output risk): {0}" -f
                $runDirectory
            )
        }
    }
}

New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
$active = @()
foreach ($run in $runs) {
    $runDirectory = Join-Path $ResultsRoot (
        "rising_thermal_h31p25_dt0p1_d3q19_dynamic_$($run.Label)"
    )
    New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
    $stdout = Join-Path $runDirectory 'console.stdout.log'
    $stderr = Join-Path $runDirectory 'console.stderr.log'
    $arguments = @(
        '--case', 'rising-thermal',
        '--h', '31.25',
        '--dt', '0.1',
        '--sample-seconds', '5',
        '--thermal', 'D3Q19',
        '--viscosity', 'dynamic',
        '--rho-ref', $run.ReferenceDensity.ToString(
            [System.Globalization.CultureInfo]::InvariantCulture),
        '--output', $runDirectory
    )
    Write-Host (
        "START gpu={0} label={1} rho_ref={2} nu={3:F6}" -f
        $run.Gpu, $run.Label, $run.ReferenceDensity,
        (15.0 / $run.ReferenceDensity)
    )
    $process = Start-Process `
        -FilePath $Executable `
        -ArgumentList $arguments `
        -Environment @{ CUDA_VISIBLE_DEVICES = "$($run.Gpu)" } `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr
    $active += [pscustomobject]@{
        Run = $run
        Directory = $runDirectory
        Stdout = $stdout
        Stderr = $stderr
        Process = $process
        StartTime = Get-Date
        LastProgress = ''
    }
}

while ($true) {
    $allFinished = $true
    foreach ($item in $active) {
        $item.Process.Refresh()
        if (-not $item.Process.HasExited) {
            $allFinished = $false
        }
        if (Test-Path -LiteralPath $item.Stdout) {
            $progress = Get-Content -LiteralPath $item.Stdout |
                Where-Object { $_ -like 'GEA_PROGRESS*' } |
                Select-Object -Last 1
            if ($progress -and $progress -ne $item.LastProgress) {
                Write-Host (
                    "STATUS gpu={0} run={1} {2}" -f
                    $item.Run.Gpu, $item.Run.Label, $progress
                )
                $item.LastProgress = $progress
            }
        }
    }
    if ($allFinished) {
        break
    }
    Start-Sleep -Seconds 2
}

$manifest = @()
foreach ($item in $active) {
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
        $item.Run.Gpu, $item.Run.Label, $item.Process.ExitCode, $elapsed
    )
    if ($item.Process.ExitCode -ne 0) {
        Write-Warning (
            "Viscosity diagnostic failed: {0} (exit {1})`n{2}" -f
            $item.Run.Label, $item.Process.ExitCode, $stderrTail
        )
    }
    $manifest += [pscustomobject]@{
        case = 'rising-thermal'
        h_m = 31.25
        dt_s = 0.1
        thermal = 'D3Q19'
        rho_ref_kg_m3 = $item.Run.ReferenceDensity
        nu_m2_s = 15.0 / $item.Run.ReferenceDensity
        gpu = $item.Run.Gpu
        exit_code = $item.Process.ExitCode
        status = if ($item.Process.ExitCode -eq 0) { 'passed' } else { 'failed' }
        elapsed_s = [math]::Round($elapsed, 3)
        output_directory = $item.Directory
        error_tail = $stderrTail
    }
}

$manifestPath = Join-Path $ResultsRoot 'viscosity_profile_manifest.csv'
$manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation
$failureCount = @($manifest | Where-Object { $_.exit_code -ne 0 }).Count
Write-Output (
    "VISCOSITY_PROFILE_COMPLETE runs={0} failures={1} manifest={2}" -f
    $manifest.Count, $failureCount, $manifestPath
)
