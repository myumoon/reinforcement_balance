# 目的: 確認済み annotation から下書き用検出器を学習する。
# 前提: Python と artifact root は survivors_deployment.env から取得する。
[CmdletBinding()]
param(
    [string]$EnvFile,
    [string]$WorkRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$commonPath = Join-Path $PSScriptRoot '..\SurvivorsDeployment\SurvivorsDeployment.Common.ps1'
. $commonPath

$repositoryRoot = Get-RepositoryRoot
if ([string]::IsNullOrWhiteSpace($EnvFile)) {
    $EnvFile = Join-Path $repositoryRoot '.env\survivors_deployment.env'
}
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    throw "Deployment env file was not found: '$EnvFile'. Specify it with -EnvFile."
}

$settings = Read-DeploymentEnv -Path $EnvFile
$pythonPath = Get-RequiredDeploymentSetting -Settings $settings -Name 'REINBALANCE_PYTHON'
$recordRoot = Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_ARTIFACT_PRIMARY_ROOT'
if ([string]::IsNullOrWhiteSpace($WorkRoot)) {
    $WorkRoot = Join-Path $recordRoot 'annotation_work'
}

$workRootPath = [IO.Path]::GetFullPath($WorkRoot)
$cliPath = Join-Path $repositoryRoot 'Tools/Deployment/train_survivors_prelabel_detector.py'
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'REINBALANCE_PYTHON must be an existing absolute file.'
}
$arguments = @($cliPath, '--work-root', $workRootPath)
Push-Location $repositoryRoot
try {
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 では native stderr を終了コード判定まで継続させる。
        $ErrorActionPreference = 'Continue'
        & $pythonPath @arguments
        $pythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}
finally {
    Pop-Location
}

if ($pythonExitCode -ne 0) {
    throw "Configured Python command failed with exit code $pythonExitCode."
}
