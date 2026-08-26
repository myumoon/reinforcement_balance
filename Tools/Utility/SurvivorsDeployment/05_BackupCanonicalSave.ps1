#requires -Version 5.1

[CmdletBinding()]
param(
    [string]$EnvPath,
    [string]$CanonicalSavePath,
    [string]$BackupRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'SurvivorsDeployment.Common.ps1')

$resolvedSource = $CanonicalSavePath
try {
    if ([string]::IsNullOrWhiteSpace($CanonicalSavePath) -or [string]::IsNullOrWhiteSpace($BackupRoot)) {
        if ([string]::IsNullOrWhiteSpace($EnvPath)) {
            $EnvPath = Join-Path (Get-RepositoryRoot) '.env\survivors_deployment.env'
        }
        $settings = Read-DeploymentEnv -Path ([IO.Path]::GetFullPath($EnvPath))
        if ([string]::IsNullOrWhiteSpace($CanonicalSavePath)) {
            $resolvedSource = Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_CANONICAL_SAVE'
        }
        if ([string]::IsNullOrWhiteSpace($BackupRoot)) {
            $BackupRoot = Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_CANONICAL_SAVE_BACKUP_ROOT'
        }
    }

    $result = Invoke-CanonicalSaveBackup -SourcePath $resolvedSource -BackupRoot $BackupRoot
    [Console]::Out.WriteLine(($result | ConvertTo-Json -Depth 4))
    exit 0
}
catch {
    $failure = [ordered]@{
        schema_version = 'survivors.canonical-save-backup.v1'
        status = 'failure'
        source_path = $resolvedSource
        backup_path = $null
        source_hash = $null
        backup_hash = $null
        size = $null
        collected_at = [DateTimeOffset]::UtcNow.ToString('o')
        match = $false
        blocking_reason = $_.Exception.Message
        record_path = $null
    }
    [Console]::Out.WriteLine(($failure | ConvertTo-Json -Depth 4))
    exit 1
}
