#Requires -Version 5.1
param(
    [string]$Configuration  = "Development",
    [string]$Platform       = "Win64",
    [string]$Filter         = "[unit]",
    [switch]$SkipBuild,
    [int]   $TimeoutMinutes = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot  = Split-Path -Parent $ScriptDir
$UprojectPath = Join-Path $ProjectRoot "ReinBalance\ReinBalance.uproject"
$BuildBat     = "C:\UnrealEngine\UE_5.4\Engine\Build\BatchFiles\Build.bat"

# Build
if (-not $SkipBuild)
{
    Write-Host ">> Building ReinBalanceLogicTests ($Platform $Configuration)..." -ForegroundColor Cyan
    & cmd /c "`"$BuildBat`" ReinBalanceLogicTests $Platform $Configuration -Project=`"$UprojectPath`"" 2>&1
    if ($LASTEXITCODE -ne 0)
    {
        Write-Error "Build failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
    Write-Host ">> Build succeeded." -ForegroundColor Green
}

# Find exe
$BinRoot = Join-Path $ProjectRoot "ReinBalance\Binaries\$Platform"
$Exe = Get-ChildItem -Path $BinRoot -Filter "ReinBalanceLogicTests.exe" -Recurse -ErrorAction SilentlyContinue |
       Select-Object -First 1

if (-not $Exe)
{
    Write-Error "ReinBalanceLogicTests.exe not found under $BinRoot. Build first."
    exit 1
}

Write-Host ">> Running: $($Exe.FullName)" -ForegroundColor Cyan
Write-Host "   Filter : $Filter"

# Run
$Timeout = $TimeoutMinutes * 60
& $Exe.FullName --log --debug --timeout=$Timeout -r console $Filter --extra-args -stdout 2>&1
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0)
{
    Write-Host ">> All tests passed." -ForegroundColor Green
}
else
{
    Write-Host ">> Tests FAILED (exit code $ExitCode)." -ForegroundColor Red
}

exit $ExitCode
