#requires -Version 5.1

[CmdletBinding()]
param(
    [string]$RepositoryRoot,
    [switch]$DryRun
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

$logLines = @('workflow=02_PrepareArtifactStore', 'status=failed')
try {
    $settings = Read-DeploymentEnv -Path (Join-Path $RepositoryRoot '.env\survivors_deployment.env')
    $primary = Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_ARTIFACT_PRIMARY_ROOT'
    $backup = Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_ARTIFACT_BACKUP_ROOT'
    $evidence = Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_EVIDENCE_ROOT'
    $primary = Assert-PathOutsideRepository -Path $primary -RepositoryRoot $RepositoryRoot
    $backup = Assert-PathOutsideRepository -Path $backup -RepositoryRoot $RepositoryRoot
    $evidence = Assert-PathOutsideRepository -Path $evidence -RepositoryRoot $RepositoryRoot

    $roots = Test-ArtifactStoreRoots -PrimaryRoot $primary -BackupRoot $backup -DryRun:$DryRun
    if (-not $DryRun) {
        if (Test-Path -LiteralPath $evidence -PathType Leaf) {
            throw 'Evidence root already exists as a file.'
        }
        New-Item -ItemType Directory -Path $evidence -Force | Out-Null

        $manageScript = Join-Path $RepositoryRoot 'Tools\Artifacts\manage_artifacts.py'
        if (-not (Test-Path -LiteralPath $manageScript -PathType Leaf)) {
            throw 'Tools/Artifacts/manage_artifacts.py is missing.'
        }
        $python = if ($settings.Contains('REINBALANCE_PYTHON') -and -not [string]::IsNullOrWhiteSpace([string]$settings['REINBALANCE_PYTHON'])) {
            [string]$settings['REINBALANCE_PYTHON']
        }
        else {
            $command = Get-Command python -ErrorAction SilentlyContinue
            if ($null -eq $command) { throw 'Python is required to verify the artifact store.' }
            $command.Source
        }
        Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -Arguments @(
            $manageScript,
            '--store-root', $primary,
            'check-backup-root',
            '--backup-root', $backup
        ) | Out-Null
    }

    $mode = if ($DryRun) { 'dry-run' } else { 'prepared' }
    $logLines = @(
        'workflow=02_PrepareArtifactStore',
        "status=$mode",
        "primary_basename=$(Split-Path -Leaf $roots.PrimaryRoot)",
        "backup_basename=$(Split-Path -Leaf $roots.BackupRoot)",
        "evidence_basename=$(Split-Path -Leaf $evidence)",
        'different_volume_check=passed',
        'backup_check=passed'
    )
    Write-OperatorLog -Name 'prepare-artifact-store' -Lines $logLines | Out-Null
    if ($DryRun) {
        Write-Host 'Dry-run completed: store paths were validated and no folders were created.'
    }
    else {
        Write-Host 'Artifact stores and evidence root prepared.'
    }
}
catch {
    try { Write-OperatorLog -Name 'prepare-artifact-store' -Lines $logLines | Out-Null } catch { }
    throw 'Artifact store preparation failed. Check the external operator log.'
}
