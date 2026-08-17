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

function Invoke-IsolatedUtility {
    param(
        [Parameter(Mandatory)][string]$SourceScript,
        [Parameter(Mandatory)][string[]]$CopyNames,
        [Parameter(Mandatory)][string]$OutputPattern,
        [Parameter(Mandatory)][string]$OutputName,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory)][string]$EvidenceRoot
    )

    $runRoot = Join-Path $EvidenceRoot ('.utility-run-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
    try {
        foreach ($copyName in $CopyNames) {
            Copy-Item -LiteralPath (Join-Path (Split-Path -Parent $SourceScript) $copyName) -Destination $runRoot
        }
        $runScript = Join-Path $runRoot (Split-Path -Leaf $SourceScript)
        $null = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runScript @Arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw 'Existing evidence utility returned a failure.'
        }
        $generated = @(Get-ChildItem -LiteralPath $runRoot -File -Filter $OutputPattern -Recurse)
        if ($generated.Count -ne 1) {
            throw 'Existing evidence utility did not produce its expected output.'
        }
        $destination = Join-Path $EvidenceRoot $OutputName
        Move-Item -LiteralPath $generated[0].FullName -Destination $destination -Force
        return $destination
    }
    finally {
        if (Test-Path -LiteralPath $runRoot) {
            Remove-Item -LiteralPath $runRoot -Recurse -Force
        }
    }
}

function Publish-EvidenceArtifact {
    param(
        [Parameter(Mandatory)][string]$PythonPath,
        [Parameter(Mandatory)][string]$ManageScript,
        [Parameter(Mandatory)][string]$StoreRoot,
        [Parameter(Mandatory)][string]$LogicalId,
        [Parameter(Mandatory)][string]$SourcePath,
        [Parameter(Mandatory)][string]$MediaType,
        [Parameter(Mandatory)][string]$RepositoryRoot
    )

    $output = @(Invoke-ConfiguredPython -PythonPath $PythonPath -WorkingDirectory $RepositoryRoot -CaptureOutput -Arguments @(
            $ManageScript,
            '--store-root', $StoreRoot,
            'put',
            '--logical-id', $LogicalId,
            '--source', $SourcePath,
            '--media-type', $MediaType
        ))
    $references = foreach ($line in $output) {
        try {
            $candidate = ([string]$line).Trim() | ConvertFrom-Json
        }
        catch {
            continue
        }
        if ($null -ne $candidate -and $candidate.PSObject.Properties.Name -contains 'store_uri') {
            $candidate
        }
    }
    $reference = $references | Select-Object -Last 1
    if ($null -eq $reference -or [string]$reference.store_uri -notmatch '^artifact://sha256/[0-9a-f]{64}$') {
        throw 'Artifact registration did not return a valid artifact reference.'
    }
    return $reference
}

$logLines = @('workflow=03_CollectTargetEvidence', 'status=failed')
try {
    $settings = Read-DeploymentEnv -Path (Join-Path $RepositoryRoot '.env\survivors_deployment.env')
    $primary = Assert-PathOutsideRepository -Path (Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_ARTIFACT_PRIMARY_ROOT') -RepositoryRoot $RepositoryRoot
    $evidence = Assert-PathOutsideRepository -Path (Get-RequiredDeploymentSetting -Settings $settings -Name 'SURVIVORS_EVIDENCE_ROOT') -RepositoryRoot $RepositoryRoot
    $exe = Get-NormalizedDeploymentPath (Get-RequiredDeploymentSetting -Settings $settings -Name 'VAMPIRE_SURVIVORS_EXE')
    if (-not (Test-Path -LiteralPath $primary -PathType Container)) {
        throw 'Primary artifact store does not exist. Run 02_PrepareArtifactStore first.'
    }
    if (Test-Path -LiteralPath $evidence -PathType Leaf) {
        throw 'Evidence root exists as a file.'
    }
    New-Item -ItemType Directory -Path $evidence -Force | Out-Null
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw 'VAMPIRE_SURVIVORS_EXE must point to an existing file.'
    }

    $machineInfoScript = Join-Path $RepositoryRoot 'Tools\Utility\MachineInfo\GetMachineInfo.ps1'
    $machineInfoModule = Join-Path $RepositoryRoot 'Tools\Utility\MachineInfo\MachineInfo.psm1'
    $metadataScript = Join-Path $RepositoryRoot 'Tools\Utility\VampireSurvivorsMetadata\GetVampireSurvivorsExecutableMetadata.ps1'
    foreach ($utility in @($machineInfoScript, $machineInfoModule, $metadataScript)) {
        if (-not (Test-Path -LiteralPath $utility -PathType Leaf)) {
            throw 'An existing evidence utility is missing.'
        }
    }

    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $machineInfoOutput = Invoke-IsolatedUtility `
        -SourceScript $machineInfoScript `
        -CopyNames @('GetMachineInfo.ps1', 'MachineInfo.psm1') `
        -OutputPattern 'machine_info_*.json' `
        -OutputName ("machine_info_$stamp.json") `
        -EvidenceRoot $evidence
    $metadataOutput = Invoke-IsolatedUtility `
        -SourceScript $metadataScript `
        -CopyNames @('GetVampireSurvivorsExecutableMetadata.ps1') `
        -OutputPattern 'vampire_survivors_executable_metadata.json' `
        -OutputName ("vampire_survivors_executable_metadata_$stamp.json") `
        -Arguments @('-ExePath', $exe) `
        -EvidenceRoot $evidence

    $python = if ($settings.Contains('REINBALANCE_PYTHON') -and -not [string]::IsNullOrWhiteSpace([string]$settings['REINBALANCE_PYTHON'])) {
        [string]$settings['REINBALANCE_PYTHON']
    }
    else { $null }
    $manageScript = Join-Path $RepositoryRoot 'Tools\Artifacts\manage_artifacts.py'
    if (-not (Test-Path -LiteralPath $manageScript -PathType Leaf)) {
        throw 'Tools/Artifacts/manage_artifacts.py is missing.'
    }
    if ([string]::IsNullOrWhiteSpace($python)) {
        $command = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $command) { throw 'Python is required to register evidence.' }
        $python = $command.Source
    }

    $machineInfoRef = Publish-EvidenceArtifact `
        -PythonPath $python `
        -ManageScript $manageScript `
        -StoreRoot $primary `
        -LogicalId "target-evidence/$stamp/machine-info" `
        -SourcePath $machineInfoOutput `
        -MediaType 'application/json' `
        -RepositoryRoot $RepositoryRoot
    $metadataRef = Publish-EvidenceArtifact `
        -PythonPath $python `
        -ManageScript $manageScript `
        -StoreRoot $primary `
        -LogicalId "target-evidence/$stamp/executable-metadata" `
        -SourcePath $metadataOutput `
        -MediaType 'application/json' `
        -RepositoryRoot $RepositoryRoot

    $records = [ordered]@{
        schema_version = 'survivors.target-evidence.v1'
        collected_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        executable = [ordered]@{
            basename = Split-Path -Leaf $exe
            canonical_hash = Get-CanonicalHash -Path $exe -PythonPath $python
        }
        machine_info = [ordered]@{
            basename = Split-Path -Leaf $machineInfoOutput
            canonical_hash = Get-CanonicalHash -Path $machineInfoOutput -PythonPath $python
            artifact_ref = $machineInfoRef
        }
        executable_metadata = [ordered]@{
            basename = Split-Path -Leaf $metadataOutput
            canonical_hash = Get-CanonicalHash -Path $metadataOutput -PythonPath $python
            artifact_ref = $metadataRef
        }
    }

    foreach ($optional in @(
            @{ Name = 'SURVIVORS_CANONICAL_SAVE'; Label = 'canonical_save' },
            @{ Name = 'SURVIVORS_TARGET_PROFILE'; Label = 'target_profile' }
        )) {
        if ($settings.Contains($optional.Name) -and -not [string]::IsNullOrWhiteSpace([string]$settings[$optional.Name])) {
            $path = Assert-PathOutsideRepository -Path ([string]$settings[$optional.Name]) -RepositoryRoot $RepositoryRoot
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "$($optional.Name) must point to an existing file."
            }
            $records[$optional.Label] = [ordered]@{
                basename = Split-Path -Leaf $path
                canonical_hash = Get-CanonicalHash -Path $path -PythonPath $python
            }
        }
    }

    $manifestPath = Join-Path $evidence "target-evidence-$stamp.json"
    $records | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    $manifestRef = Publish-EvidenceArtifact `
        -PythonPath $python `
        -ManageScript $manageScript `
        -StoreRoot $primary `
        -LogicalId "target-evidence/$stamp/manifest" `
        -SourcePath $manifestPath `
        -MediaType 'application/json' `
        -RepositoryRoot $RepositoryRoot

    & git -C $RepositoryRoot diff --quiet -- Tools/Deployment/configs
    if ($LASTEXITCODE -ne 0) {
        throw 'Tracked deployment configuration changed during evidence collection.'
    }
    $logLines = @(
        'workflow=03_CollectTargetEvidence',
        'status=success',
        "evidence_basename=$(Split-Path -Leaf $manifestPath)",
        "executable_basename=$(Split-Path -Leaf $exe)",
        'canonical_hash=passed',
        'machine_info_artifact=passed',
        'executable_metadata_artifact=passed',
        "manifest_artifact_uri=$($manifestRef.store_uri)",
        'tracked_config_check=passed'
    )
    Write-OperatorLog -Name 'collect-target-evidence' -Lines $logLines | Out-Null
    Write-Host 'Target evidence collection completed.'
    Write-Host 'Evidence payloads were written to the configured external roots.'
}
catch {
    try { Write-OperatorLog -Name 'collect-target-evidence' -Lines $logLines | Out-Null } catch { }
    throw 'Target evidence collection failed. Check the external operator log.'
}
