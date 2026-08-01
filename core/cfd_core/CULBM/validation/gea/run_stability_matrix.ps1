param(
    [string]$SrtExecutable = (
        Join-Path $PSScriptRoot '..\..\build\CULBM-gea-validation.exe'
    ),
    [string]$TrtExecutable = (
        Join-Path $PSScriptRoot '..\..\build\CULBM-gea-validation-trt.exe'
    ),
    [string]$ResultsRoot = (
        Join-Path $PSScriptRoot 'results\stability'
    )
)

$ErrorActionPreference = 'Stop'
$SrtExecutable = [System.IO.Path]::GetFullPath($SrtExecutable)
$TrtExecutable = [System.IO.Path]::GetFullPath($TrtExecutable)
$ResultsRoot = [System.IO.Path]::GetFullPath($ResultsRoot)

foreach ($executable in @($SrtExecutable, $TrtExecutable)) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "GEA validation executable not found: $executable"
    }
}

$runs = @(
    [pscustomobject]@{
        Collision='SRT'; Executable=$SrtExecutable
        Case='rising-thermal'; H=250.0; Dt=1.0
    }
    [pscustomobject]@{
        Collision='SRT'; Executable=$SrtExecutable
        Case='density-current'; H=400.0; Dt=1.0
    }
    [pscustomobject]@{
        Collision='TRT'; Executable=$TrtExecutable
        Case='rising-thermal'; H=250.0; Dt=0.1
    }
    [pscustomobject]@{
        Collision='TRT'; Executable=$TrtExecutable
        Case='density-current'; H=400.0; Dt=0.1
    }
)

function Convert-Token {
    param([double]$Value)
    return $Value.ToString(
        '0.############',
        [System.Globalization.CultureInfo]::InvariantCulture
    ).Replace('.', 'p')
}

function Get-StabilityRunName {
    param([pscustomobject]$Run)
    $caseToken = $Run.Case.Replace('-', '_')
    $hToken = Convert-Token $Run.H
    $dtToken = Convert-Token $Run.Dt
    $collisionToken = $Run.Collision.ToLowerInvariant()
    return "${caseToken}_h${hToken}_dt${dtToken}_${collisionToken}"
}

function Start-StabilityRun {
    param(
        [pscustomobject]$Run,
        [int]$Gpu
    )

    $name = Get-StabilityRunName $Run
    $runDirectory = Join-Path $ResultsRoot $name
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
    $arguments = @(
        '--case', $Run.Case,
        '--h', $Run.H.ToString(
            [System.Globalization.CultureInfo]::InvariantCulture),
        '--dt', $Run.Dt.ToString(
            [System.Globalization.CultureInfo]::InvariantCulture),
        '--sample-seconds', '5',
        '--thermal', 'D3Q19',
        '--viscosity', 'dynamic',
        '--output', $runDirectory
    )
    Write-Host (
        "START gpu={0} collision={1} case={2} h={3} dt={4}" -f
        $Gpu, $Run.Collision, $Run.Case, $Run.H, $Run.Dt
    )
    $process = Start-Process `
        -FilePath $Run.Executable `
        -ArgumentList $arguments `
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

function Wait-StabilityBatch {
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
                    Write-Host (
                        "STATUS gpu={0} run={1} {2}" -f
                        $item.Gpu, $item.Name, $progress
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
                "Stability run failed: {0} (exit {1})`n{2}" -f
                $item.Name, $item.Process.ExitCode, $stderrTail
            )
        }
        $records += [pscustomobject]@{
            collision = $item.Run.Collision
            case = $item.Run.Case
            h_m = $item.Run.H
            dt_s = $item.Run.Dt
            gpu = $item.Gpu
            exit_code = $item.Process.ExitCode
            status = if ($item.Process.ExitCode -eq 0) {
                'passed'
            } else {
                'failed'
            }
            elapsed_s = [math]::Round($elapsed, 3)
            output_directory = $item.Directory
            error_tail = $stderrTail
        }
    }
    return $records
}

New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
foreach ($run in $runs) {
    $candidate = Join-Path $ResultsRoot (Get-StabilityRunName $run)
    if (Test-Path -LiteralPath $candidate) {
        $existing = @(Get-ChildItem -LiteralPath $candidate -Force)
        if ($existing.Count -ne 0) {
            throw (
                "Refusing to start stability matrix because a run directory is non-empty: {0}" -f
                $candidate
            )
        }
    }
}
$manifest = @()
for ($index = 0; $index -lt $runs.Count; $index += 2) {
    $batch = @(
        Start-StabilityRun -Run $runs[$index] -Gpu 0
        Start-StabilityRun -Run $runs[$index + 1] -Gpu 1
    )
    $manifest += Wait-StabilityBatch -Batch $batch
}

$manifestPath = Join-Path $ResultsRoot 'stability_manifest.csv'
$manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation
$failureCount = @(
    $manifest | Where-Object { $_.exit_code -ne 0 }
).Count
Write-Output (
    "STABILITY_COMPLETE runs={0} failures={1} manifest={2}" -f
    $manifest.Count, $failureCount, $manifestPath
)
