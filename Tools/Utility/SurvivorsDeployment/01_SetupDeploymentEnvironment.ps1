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

function Assert-PinnedLockFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw 'Tools/Deployment/requirements.lock is missing.'
    }
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith('#')) { continue }
        if ($trimmed -notmatch '^[A-Za-z0-9_.-]+==[^=\s]+$') {
            throw 'requirements.lock contains an unpinned dependency.'
        }
    }
}

$logLines = @('workflow=01_SetupDeploymentEnvironment', 'status=failed')
try {
    $settings = Read-DeploymentEnv -Path (Join-Path $RepositoryRoot '.env\survivors_deployment.env')
    $python = Get-RequiredDeploymentSetting -Settings $settings -Name 'REINBALANCE_PYTHON'
    $commonPath = Join-Path $RepositoryRoot 'Tools\Common'
    $deploymentPath = Join-Path $RepositoryRoot 'Tools\Deployment'
    $lockPath = Join-Path $deploymentPath 'requirements.lock'
    if (-not (Test-Path -LiteralPath (Join-Path $commonPath 'pyproject.toml') -PathType Leaf)) {
        throw 'Tools/Common packaging metadata is missing.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $deploymentPath 'pyproject.toml') -PathType Leaf)) {
        throw 'PR #308 deployment packaging metadata is missing.'
    }
    Assert-PinnedLockFile $lockPath

    Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -Arguments @('-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '-e', $commonPath)
    Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -Arguments @('-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '-e', $deploymentPath)
    Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -Arguments @('-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--requirement', $lockPath)
    Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -Arguments @('-c', "from reinbalance_survivors_contracts.canonical_json import canonical_hash; print('reinbalance_survivors_contracts import ok')")
    Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -Arguments @('--version')
    Invoke-ConfiguredPython -PythonPath $python -WorkingDirectory $RepositoryRoot -Arguments @('-c', "import pytest; print('pytest ' + pytest.__version__)")

    $logLines = @(
        'workflow=01_SetupDeploymentEnvironment',
        'status=success',
        'common_editable=ok',
        'deployment_editable=ok',
        'lock_install=ok',
        'contracts_import=ok'
    )
    Write-OperatorLog -Name 'setup-deployment-environment' -Lines $logLines | Out-Null
    Write-Host 'Deployment Python environment setup completed.'
    Write-Host 'The operator log was written outside the repository.'
}
catch {
    try { Write-OperatorLog -Name 'setup-deployment-environment' -Lines $logLines | Out-Null } catch { }
    throw 'Deployment Python environment setup failed. Check the external operator log.'
}
