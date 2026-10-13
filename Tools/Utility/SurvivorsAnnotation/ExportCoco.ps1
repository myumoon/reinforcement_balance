# 目的: annotation_work の session ごとに COCO JSON を出力する。
# 出力先は artifact root 配下の coco フォルダに session ID ごとに分ける。
[CmdletBinding()]
param(
    [string]$EnvFile,
    [string]$WorkRoot,
    [string]$SessionId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$deploymentCommonPath = Join-Path $PSScriptRoot '..\SurvivorsDeployment\SurvivorsDeployment.Common.ps1'
. $deploymentCommonPath
$annotationCommonPath = Join-Path $PSScriptRoot 'SurvivorsAnnotation.Common.ps1'
. $annotationCommonPath

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
$recordRootPath = [IO.Path]::GetFullPath($recordRoot)
if ([string]::IsNullOrWhiteSpace($WorkRoot)) {
    $WorkRoot = Join-Path $recordRootPath 'annotation_work'
}
$workRootPath = [IO.Path]::GetFullPath($WorkRoot)

if ([string]::IsNullOrWhiteSpace($SessionId)) {
    $sessionIds = @(
        Get-ChildItem -LiteralPath $workRootPath -Directory -Filter 'session-*' |
            Sort-Object -Property Name |
            ForEach-Object { $_.Name }
    )
}
else {
    $sessionIds = @()
    foreach ($requestedId in $SessionId.Split(',')) {
        $requestedId = $requestedId.Trim()
        if ($requestedId -notmatch '^session-[A-Za-z0-9_-]+$') {
            throw "不正な session ID です: '$requestedId'。session- で始まる英数字、_、- の ID を指定してください。"
        }
        $sessionIds += $requestedId
    }
}

if ($sessionIds.Count -eq 0) {
    throw "WorkRoot '$workRootPath' に export 対象の session-* ディレクトリがありません。"
}

Write-Host '処理対象 session:'
foreach ($sessionId in $sessionIds) {
    Write-Host ("  {0}" -f $sessionId)
}

$cliPath = Join-Path $repositoryRoot 'Tools/Deployment/export_survivors_annotations_coco.py'
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'REINBALANCE_PYTHON must be an existing absolute file.'
}

$failedSessions = @()
foreach ($sessionId in $sessionIds) {
    $outputDir = Join-Path (Join-Path $recordRootPath 'coco') $sessionId
    $arguments = @(
        $cliPath,
        '--work-root', $workRootPath,
        '--store-root', $recordRootPath,
        '--session-id', $sessionId,
        '--output-dir', $outputDir
    )

    try {
        Invoke-AnnotationPython -PythonPath $pythonPath -ArgumentList $arguments -RepositoryRoot $repositoryRoot
    }
    catch {
        $failedSessions += $sessionId
        Write-Warning ("session {0} の COCO 出力に失敗しました: {1}" -f $sessionId, $_.Exception.Message)
    }
}

if ($failedSessions.Count -gt 0) {
    throw "COCO 出力に失敗した session: $($failedSessions -join ', ')"
}
