[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Command,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgumentsFromUser
)

if (-not $env:LUW_HOME) {
    Write-Error "LUW_HOME is not set"
    exit 2
}

$launcher = Join-Path $env:LUW_HOME "bin\luw.exe"
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    Write-Error "Native launcher not found: $launcher"
    exit 2
}

& $launcher $Command @ArgumentsFromUser
exit $LASTEXITCODE
