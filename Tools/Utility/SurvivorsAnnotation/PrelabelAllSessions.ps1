# 目的: annotation_work の全 session に下書き生成を順に実行する。
# 前提: Python と artifact root は survivors_deployment.env から取得する。
[CmdletBinding()]
param(
    [string]$EnvFile,
    [string]$WorkRoot,
    [switch]$RefreshUnchecked
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
    throw "設定ファイルが見つかりません: '$EnvFile'。-EnvFile で既存のファイルを指定してください。"
}

$settings = Read-DeploymentEnv -Path $EnvFile
$pythonPath = Get-RequiredDeploymentSetting -Settings $settings -Name 'REINBALANCE_PYTHON'
$recordRoot = Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_ARTIFACT_PRIMARY_ROOT'
if ([string]::IsNullOrWhiteSpace($WorkRoot)) {
    $WorkRoot = Join-Path $recordRoot 'annotation_work'
}

$workRootPath = [IO.Path]::GetFullPath($WorkRoot)
$sessions = @(Get-ChildItem -LiteralPath $workRootPath -Directory -Filter 'session-*' | Sort-Object -Property Name)
if ($sessions.Count -eq 0) {
    throw "WorkRoot '$workRootPath' に session-* ディレクトリがありません。"
}

$cliPath = Join-Path $repositoryRoot 'Tools/Deployment/prelabel_survivors_frames.py'
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'REINBALANCE_PYTHON must be an existing absolute file.'
}
$failedSessions = @()
foreach ($session in $sessions) {
    $arguments = @($cliPath, '--work-root', $workRootPath, '--session-id', $session.Name)
    if ($RefreshUnchecked) {
        $arguments += '--refresh-unchecked'
    }
    try {
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
    }
    catch {
        $failedSessions += $session.Name
        Write-Warning ("session {0} の下書き生成に失敗しました: {1}" -f $session.Name, $_.Exception.Message)
    }
}

if ($failedSessions.Count -gt 0) {
    throw "下書き生成に失敗した session: $($failedSessions -join ', ')"
}
