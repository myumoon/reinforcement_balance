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

$logLines = @('workflow=04_VerifyDeploymentSmoke', 'status=failed')
try {
    $captureScript = Join-Path $RepositoryRoot 'Tools\Deployment\capture_survivors.py'
    if (-not (Test-Path -LiteralPath $captureScript -PathType Leaf)) {
        throw 'PR #308 is not merged: Tools/Deployment/capture_survivors.py is unavailable. Run this workflow after PR #308 merges.'
    }

    $settings = Read-DeploymentEnv -Path (Join-Path $RepositoryRoot '.env\survivors_deployment.env')
    $python = Get-RequiredDeploymentSetting -Settings $settings -Name 'REINBALANCE_PYTHON'
    $primary = Assert-PathOutsideRepository -Path (Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_ARTIFACT_PRIMARY_ROOT') -RepositoryRoot $RepositoryRoot
    if (-not (Test-Path -LiteralPath $primary -PathType Container)) {
        throw 'Primary artifact store does not exist. Run 02_PrepareArtifactStore first.'
    }

    $sessionId = 'synthetic-smoke-{0}' -f (Get-Date -Format 'yyyyMMdd-HHmmss')
    $published = @(Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -CaptureOutput -Arguments @(
        $captureScript,
        '--store-root', $primary,
        '--session-id', $sessionId,
        '--duration-sec', '1',
        '--synthetic'
    ))
    $publishedJson = $published | ForEach-Object { try { $_ | ConvertFrom-Json } catch { $null } } | Where-Object { $null -ne $_ } | Select-Object -Last 1
    if ($null -eq $publishedJson -or $publishedJson.status -ne 'PUBLISHED' -or $publishedJson.formal_dataset_eligible -ne $false) {
        throw 'Synthetic smoke output was not formally ineligible.'
    }

    $dryRun = @(Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -CaptureOutput -Arguments @(
        $captureScript,
        '--store-root', $primary,
        '--session-id', ('synthetic-dry-run-{0}' -f (Get-Date -Format 'yyyyMMdd-HHmmss')),
        '--duration-sec', '1',
        '--synthetic',
        '--dry-run'
    ))
    if (($dryRun -join "`n") -notmatch 'DRY_RUN_OK') {
        throw 'Synthetic dry-run did not report DRY_RUN_OK.'
    }

    Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -Arguments @('-m', 'pytest', 'Tools/Deployment/tests/test_capture_dataset.py', 'Tools/Deployment/tests/test_capture_annotation.py', '-q') | Out-Null
    Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -Arguments @('-m', 'pytest', 'Tools/Deployment/tests', '-q') | Out-Null

    $logLines = @(
        'workflow=04_VerifyDeploymentSmoke',
        'status=success',
        'synthetic_capture=passed',
        'formal_dataset_eligible=false',
        'synthetic_dry_run=passed',
        'deployment_tests=passed'
    )
    Write-OperatorLog -Name 'verify-deployment-smoke' -Lines $logLines | Out-Null
    Write-Host 'Synthetic deployment smoke completed; formal_dataset_eligible=false.'
    Write-Host 'No live game capture was started.'
}
catch {
    try { Write-OperatorLog -Name 'verify-deployment-smoke' -Lines $logLines | Out-Null } catch { }
    if ($_.Exception.Message -like 'PR #308 is not merged:*') {
        throw $_.Exception.Message
    }
    throw 'Deployment smoke verification failed. Check the external operator log.'
}
