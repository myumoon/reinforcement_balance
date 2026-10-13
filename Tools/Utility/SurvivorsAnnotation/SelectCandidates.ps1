# 目的: 未配置の収録 session からアノテーション候補を選ぶ。
# WorkRoot に配置済みの session は既定で除外し、指定時だけ再実行する。
[CmdletBinding()]
param(
    [string]$EnvFile,
    [string]$WorkRoot,
    [string]$SessionId,
    [int]$Count,
    [int]$Stride
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
    $captureRoot = Join-Path $recordRootPath 'capture_sessions'
    $sessionIds = @(
        Get-ChildItem -LiteralPath $captureRoot -Directory -Filter 'session-*' |
            Sort-Object -Property Name |
            Where-Object { -not (Test-Path -LiteralPath (Join-Path $workRootPath $_.Name)) } |
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
    Write-Host '新しい session はありません。'
    exit 0
}

Write-Host '処理対象 session:'
foreach ($sessionId in $sessionIds) {
    Write-Host ("  {0}" -f $sessionId)
}

$cliPath = Join-Path $repositoryRoot 'Tools/Deployment/select_survivors_annotation_candidates.py'
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'REINBALANCE_PYTHON must be an existing absolute file.'
}

$failedSessions = @()
foreach ($sessionId in $sessionIds) {
    $arguments = @(
        $cliPath,
        '--store-root', $recordRootPath,
        '--session-id', $sessionId,
        '--work-root', $workRootPath
    )
    if ($PSBoundParameters.ContainsKey('Count')) {
        $arguments += @('--count', [string]$Count)
    }
    if ($PSBoundParameters.ContainsKey('Stride')) {
        $arguments += @('--stride', [string]$Stride)
    }

    try {
        Invoke-AnnotationPython -PythonPath $pythonPath -ArgumentList $arguments -RepositoryRoot $repositoryRoot
    }
    catch {
        $failedSessions += $sessionId
        Write-Warning ("session {0} の候補抽出に失敗しました: {1}" -f $sessionId, $_.Exception.Message)
    }
}

if ($failedSessions.Count -gt 0) {
    throw "候補抽出に失敗した session: $($failedSessions -join ', ')"
}
