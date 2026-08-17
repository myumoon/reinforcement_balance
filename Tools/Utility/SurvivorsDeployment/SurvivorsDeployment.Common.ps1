#requires -Version 5.1

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:SurvivorsDeploymentScriptRoot = $PSScriptRoot
$script:SurvivorsDeploymentAllowedKeys = @(
    'REINBALANCE_PYTHON',
    'SURVIVORS_ARTIFACT_PRIMARY_ROOT',
    'SURVIVORS_ARTIFACT_BACKUP_ROOT',
    'SURVIVORS_EVIDENCE_ROOT',
    'VAMPIRE_SURVIVORS_EXE',
    'SURVIVORS_CANONICAL_SAVE',
    'SURVIVORS_TARGET_PROFILE'
)

function Get-RepositoryRoot {
    [CmdletBinding()]
    param()

    return [IO.Path]::GetFullPath((Join-Path $script:SurvivorsDeploymentScriptRoot '..\..\..'))
}

function Read-DeploymentEnv {
    [CmdletBinding()]
    param(
        [string]$Path = (Join-Path (Get-RepositoryRoot) '.env\survivors_deployment.env')
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw 'survivors_deployment.env was not found. Run 00_ConfigureLocalEnvironment first.'
    }

    $allowed = @{}
    foreach ($key in $script:SurvivorsDeploymentAllowedKeys) {
        $allowed[$key] = $true
    }

    $settings = [ordered]@{}
    $lineNumber = 0
    foreach ($line in Get-Content -LiteralPath $Path) {
        $lineNumber++
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith('#')) {
            continue
        }
        if ($trimmed -notmatch '^([^=\s][^=]*)=(.*)$') {
            throw "Invalid deployment env syntax at line $lineNumber."
        }

        $key = $Matches[1].Trim()
        $value = $Matches[2].Trim()
        if (-not $allowed.ContainsKey($key)) {
            throw "Unknown deployment env key '$key'."
        }
        if ($settings.Contains($key)) {
            throw "Duplicate deployment env key '$key'."
        }
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $settings[$key] = $value
    }

    return $settings
}

function Get-RequiredDeploymentSetting {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [hashtable]$Settings,
        [Parameter(Mandatory)]
        [string]$Name
    )

    if (-not $Settings.ContainsKey($Name) -or [string]::IsNullOrWhiteSpace([string]$Settings[$Name])) {
        throw "Required deployment setting '$Name' is empty."
    }
    return [string]$Settings[$Name]
}

function Get-NormalizedDeploymentPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw 'Deployment path must not be empty.'
    }
    if (-not [IO.Path]::IsPathRooted($Path)) {
        throw 'Deployment paths must be absolute.'
    }
    $fullPath = [IO.Path]::GetFullPath($Path)
    $pathRoot = [IO.Path]::GetPathRoot($fullPath)
    if ($fullPath.Equals($pathRoot, [StringComparison]::OrdinalIgnoreCase)) {
        return $pathRoot
    }
    return $fullPath.TrimEnd('\', '/')
}

function Assert-PathOutsideRepository {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [string]$RepositoryRoot = (Get-RepositoryRoot)
    )

    $candidate = Get-NormalizedDeploymentPath $Path
    $root = Get-NormalizedDeploymentPath $RepositoryRoot
    $prefix = $root + [IO.Path]::DirectorySeparatorChar
    if ($candidate.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Deployment output paths must be outside the repository.'
    }
    return $candidate
}

function Get-NearestExistingPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    $current = Get-NormalizedDeploymentPath $Path
    while (-not (Test-Path -LiteralPath $current) -and $current -ne [IO.Path]::GetPathRoot($current)) {
        $current = Split-Path -Parent $current
    }
    if (-not (Test-Path -LiteralPath $current)) {
        throw 'Unable to inspect the volume for a deployment path.'
    }
    return $current
}

function Get-DeploymentVolumeIdentity {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    $existing = Get-NearestExistingPath $Path
    $root = [IO.Path]::GetPathRoot($existing)
    if ($root -notmatch '^[A-Za-z]:\\$') {
        throw 'Deployment stores must resolve to a Windows volume.'
    }
    $driveLetter = $root.Substring(0, 1)

    try {
        $volume = Get-Volume -DriveLetter $driveLetter -ErrorAction Stop
        if ($volume.UniqueId) {
            return [string]$volume.UniqueId
        }
        if ($volume.SerialNumber) {
            return "serial:$($volume.SerialNumber)"
        }
    }
    catch {
        # Win32_LogicalDisk below is available on Windows PowerShell 5.1 hosts
        # where the Storage module is unavailable.
    }

    try {
        $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='$driveLetter`:'" -ErrorAction Stop
        if ($disk.VolumeSerialNumber) {
            return "serial:$($disk.VolumeSerialNumber)"
        }
    }
    catch {
        # Convert the platform failure into one stable operator-facing error.
    }

    $volumeText = @(& cmd.exe /c "vol $driveLetter`:" 2>$null)
    foreach ($line in $volumeText) {
        if ([string]$line -match '(?i)Volume Serial Number is\s+([0-9A-F]{4}-[0-9A-F]{4})') {
            return "serial:$($Matches[1].ToUpperInvariant())"
        }
    }
    throw 'Unable to determine the deployment volume identity.'
}

function Test-ArtifactStoreRoots {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$PrimaryRoot,
        [Parameter(Mandatory)]
        [string]$BackupRoot,
        [switch]$DryRun,
        [scriptblock]$VolumeResolver
    )

    $primary = Get-NormalizedDeploymentPath $PrimaryRoot
    $backup = Get-NormalizedDeploymentPath $BackupRoot
    if ($primary.Equals($backup, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Primary and backup artifact roots must not be the same path.'
    }
    if ($null -eq $VolumeResolver) {
        $VolumeResolver = { param([string]$Path) Get-DeploymentVolumeIdentity $Path }
    }

    $primaryVolume = [string](& $VolumeResolver $primary)
    $backupVolume = [string](& $VolumeResolver $backup)
    if ([string]::IsNullOrWhiteSpace($primaryVolume) -or [string]::IsNullOrWhiteSpace($backupVolume) -or $primaryVolume.Equals($backupVolume, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Primary and backup artifact roots must be on different volumes/devices.'
    }

    if (-not $DryRun) {
        foreach ($root in @($primary, $backup)) {
            if (Test-Path -LiteralPath $root -PathType Leaf) {
                throw 'An artifact store root already exists as a file.'
            }
            if (-not (Test-Path -LiteralPath $root)) {
                New-Item -ItemType Directory -Path $root -Force | Out-Null
            }
        }
    }

    return [pscustomobject]@{
        PrimaryRoot   = $primary
        BackupRoot    = $backup
        PrimaryVolume = $primaryVolume
        BackupVolume  = $backupVolume
        DryRun        = [bool]$DryRun
    }
}

function Invoke-ConfiguredPython {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$PythonPath,
        [Parameter(Mandatory)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = (Get-RepositoryRoot),
        [switch]$CaptureOutput
    )

    if (-not [IO.Path]::IsPathRooted($PythonPath) -or -not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw 'REINBALANCE_PYTHON must be an existing absolute file.'
    }
    $output = @()
    Push-Location $WorkingDirectory
    try {
        $output = @(& $PythonPath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "Configured Python command failed with exit code $exitCode."
    }
    if ($CaptureOutput) {
        return $output
    }
    $output | Write-Output
}

function Get-CanonicalHash {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [string]$PythonPath
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw 'The file to hash does not exist.'
    }
    if ([string]::IsNullOrWhiteSpace($PythonPath)) {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $python) {
            throw 'Python is required for canonical hashing.'
        }
        $PythonPath = $python.Source
    }

    $commonSource = Join-Path (Get-RepositoryRoot) 'Tools\Common\src'
    $oldPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = $commonSource
    try {
        $code = @'
import sys
from reinbalance_survivors_contracts.canonical_json import canonical_hash

data = open(sys.argv[1], 'rb').read()
print(canonical_hash({'bytes_hex': data.hex()}))
'@
        $result = @(Invoke-ConfiguredPython -PythonPath $PythonPath -Arguments @('-c', $code, $Path) -WorkingDirectory (Get-RepositoryRoot) -CaptureOutput)
    }
    finally {
        $env:PYTHONPATH = $oldPythonPath
    }
    $hash = ($result | ForEach-Object { [string]$_ } | Where-Object { $_ -match '^[0-9a-f]{64}$' } | Select-Object -Last 1)
    if ([string]::IsNullOrWhiteSpace($hash)) {
        throw 'Canonical hash command returned an invalid result.'
    }
    return $hash
}

function Write-OperatorLog {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string[]]$Lines
    )

    if ($Name -notmatch '^[A-Za-z0-9_.-]+$') {
        throw 'Operator log name contains unsupported characters.'
    }
    $localAppData = [Environment]::GetFolderPath('LocalApplicationData')
    $logRoot = $null
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($localAppData)) {
        $candidates += Join-Path $localAppData 'ReinBalance\DeploymentLogs'
    }
    $candidates += Join-Path ([IO.Path]::GetTempPath()) 'ReinBalance\DeploymentLogs'
    foreach ($candidate in $candidates) {
        try {
            New-Item -ItemType Directory -Path $candidate -Force -ErrorAction Stop | Out-Null
            $logRoot = $candidate
            break
        }
        catch {
            # A locked-down host may deny LOCALAPPDATA; use an external temp log.
        }
    }
    if ([string]::IsNullOrWhiteSpace($logRoot)) {
        throw 'No external operator log directory is writable.'
    }
    $path = Join-Path $logRoot ("{0}_{1}.log" -f $Name, (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $safeLines = @($Lines | ForEach-Object { [string]$_ })
    Set-Content -LiteralPath $path -Value $safeLines -Encoding UTF8
    return $path
}
