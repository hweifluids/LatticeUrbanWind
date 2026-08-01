param(
    [ValidateSet("Release", "RelWithDebInfo", "Debug")]
    [string]$Configuration = "Release",
    [string]$Generator = "Visual Studio 17 2022"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$IsVisualStudio = $Generator.StartsWith("Visual Studio")
$BuildDir = Join-Path $Root $(if ($IsVisualStudio) { "build-msvc" } else { "build" })
$ConfigureArgs = @("-S", $Root, "-B", $BuildDir, "-G", $Generator)
if ($IsVisualStudio) {
    $ConfigureArgs += @("-A", "x64")
}
else {
    $ConfigureArgs += "-DCMAKE_BUILD_TYPE=$Configuration"
}

cmake @ConfigureArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $BuildDir --config $Configuration --parallel
exit $LASTEXITCODE
