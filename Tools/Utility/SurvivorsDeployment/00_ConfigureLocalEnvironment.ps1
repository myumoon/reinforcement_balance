#requires -Version 5.1

[CmdletBinding()]
param(
    [string]$RepositoryRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'SurvivorsDeployment.Common.ps1')

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = Get-RepositoryRoot
}
else {
    $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
}

$envDirectory = Join-Path $RepositoryRoot '.env'
$envPath = Join-Path $envDirectory 'survivors_deployment.env'
$examplePath = Join-Path $PSScriptRoot 'survivors_deployment.env.example'
New-Item -ItemType Directory -Path $envDirectory -Force | Out-Null

if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
    Copy-Item -LiteralPath $examplePath -Destination $envPath
    Write-Host 'Created .env/survivors_deployment.env from the empty template.'
}
else {
    Write-Host 'Existing .env/survivors_deployment.env was preserved.'
}

$settings = Read-DeploymentEnv -Path $envPath
$keys = @(
    'REINBALANCE_PYTHON',
    'SURVIVORS_ARTIFACT_PRIMARY_ROOT',
    'SURVIVORS_ARTIFACT_BACKUP_ROOT',
    'SURVIVORS_EVIDENCE_ROOT',
    'VAMPIRE_SURVIVORS_EXE',
    'SURVIVORS_CANONICAL_SAVE',
    'SURVIVORS_TARGET_PROFILE'
)
$missing = @($keys | Where-Object { -not $settings.Contains($_) -or [string]::IsNullOrWhiteSpace([string]$settings[$_]) })
if ($missing.Count -gt 0) {
    Write-Host 'Edit these empty setting names in the local env file:'
    $missing | ForEach-Object { Write-Host "  $_" }
}
else {
    Write-Host 'All deployment setting names are populated.'
}

if (Test-Path -LiteralPath (Join-Path $RepositoryRoot '.git')) {
    & git -C $RepositoryRoot check-ignore -q -- '.env/survivors_deployment.env'
    if ($LASTEXITCODE -eq 0) {
        Write-Host '.env/survivors_deployment.env is ignored by Git.'
    }
    else {
        Write-Host 'WARNING: .env/survivors_deployment.env is not ignored by Git.'
    }
}
else {
    Write-Host 'Git ignore check skipped for a temporary test root.'
}

Write-Host 'Edit the local env file, then run 01_SetupDeploymentEnvironment.bat.'
