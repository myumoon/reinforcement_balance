#Requires -Version 5.1
param(
    [string]$Configuration  = "Development",
    [string]$Platform       = "Win64",
    [string]$Filter         = "[unit]",
    [string]$EngineRoot     = "",
    [switch]$SkipBuild,
    [int]   $TimeoutMinutes = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve engine root: -EngineRoot > $env:UE_ROOT > common locations
if (-not $EngineRoot) {
    if ($env:UE_ROOT) {
        $EngineRoot = $env:UE_ROOT
    } else {
        $Candidates = @(
            "C:\UnrealEngine\UE_5.4",
            "C:\Program Files\Epic Games\UE_5.4"
        )
        foreach ($C in $Candidates) {
            if (Test-Path $C) { $EngineRoot = $C; break }
        }
    }
}
if (-not $EngineRoot) {
    Write-Error "Engine root not found. Set UE_ROOT env var or pass -EngineRoot 'C:\path\to\UE_5.4'."
    exit 1
}

$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot  = (Get-Item $ScriptDir).Parent.Parent.FullName
$UprojectPath = Join-Path $ProjectRoot "ReinBalance\ReinBalance.uproject"
$BuildBat     = Join-Path $EngineRoot "Engine\Build\BatchFiles\Build.bat"

# Build
if (-not $SkipBuild)
{
    Write-Host ">> Building ReinBalanceLogicTests ($Platform $Configuration)..." -ForegroundColor Cyan
    Write-Host "   Engine: $EngineRoot"
    & cmd /c "`"$BuildBat`" ReinBalanceLogicTests $Platform $Configuration -Project=`"$UprojectPath`"" 2>&1
    if ($LASTEXITCODE -ne 0)
    {
        Write-Error "Build failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
    Write-Host ">> Build succeeded." -ForegroundColor Green
}

# Resolve exe name.
# UE naming convention:
#   Development -> ReinBalanceLogicTests.exe
#   Other       -> ReinBalanceLogicTests-<Platform>-<Configuration>.exe
$ExeName = if ($Configuration -eq "Development") {
    "ReinBalanceLogicTests.exe"
} else {
    "ReinBalanceLogicTests-$Platform-$Configuration.exe"
}

$BinRoot = Join-Path $ProjectRoot "ReinBalance\Binaries\$Platform"
$Exe = Get-ChildItem -Path $BinRoot -Filter $ExeName -Recurse -ErrorAction SilentlyContinue |
       Select-Object -First 1

if (-not $Exe)
{
    Write-Error "$ExeName not found under $BinRoot. Build first."
    exit 1
}

Write-Host ">> Running: $($Exe.FullName)" -ForegroundColor Cyan
Write-Host "   Filter : $Filter"

# Run. --timeout unit is minutes in LowLevelTestsRunner.
& $Exe.FullName --log --debug --timeout=$TimeoutMinutes -r console $Filter --extra-args -stdout 2>&1
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
