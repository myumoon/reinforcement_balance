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
    'SURVIVORS_CANONICAL_SAVE_BACKUP_ROOT',
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

function Assert-LocalFixedNtfsPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    $normalized = Get-NormalizedDeploymentPath $Path
    $root = [IO.Path]::GetPathRoot($normalized)
    if ($root -notmatch '^[A-Za-z]:\\$') {
        throw 'Backup root must be on a local fixed NTFS volume.'
    }
    try {
        $drive = New-Object IO.DriveInfo($root)
        if (-not $drive.IsReady -or $drive.DriveType -ne [IO.DriveType]::Fixed -or $drive.DriveFormat -ne 'NTFS') {
            throw 'unsupported volume'
        }
    }
    catch {
        throw 'Backup root must be on a local fixed NTFS volume.'
    }
}

function Assert-NoReparsePointPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    $current = Get-NearestExistingPath $Path
    $root = [IO.Path]::GetPathRoot($current)
    while ($true) {
        $item = Get-Item -LiteralPath $current -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Backup root must not traverse a reparse point.'
        }
        if ($current.Equals($root, [StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $current = Split-Path -Parent $current
    }
}

function Invoke-CanonicalSaveBackup {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$SourcePath,
        [Parameter(Mandatory)]
        [string]$BackupRoot,
        [string]$GenerationName,
        [DateTimeOffset]$CollectedAt = [DateTimeOffset]::UtcNow,
        [scriptblock]$HashProvider,
        [scriptblock]$VolumeValidator
    )

    $source = Get-NormalizedDeploymentPath $SourcePath
    $backupRootPath = Get-NormalizedDeploymentPath $BackupRoot
    if ($source.Equals($backupRootPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Canonical save source and backup root must not be the same path.'
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw 'SURVIVORS_CANONICAL_SAVE must point to an existing regular file.'
    }
    $sourceItem = Get-Item -LiteralPath $source -Force
    if (($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'SURVIVORS_CANONICAL_SAVE must point to an existing regular file.'
    }
    if (Test-Path -LiteralPath $backupRootPath -PathType Leaf) {
        throw 'Canonical save backup root must be a directory.'
    }

    if ($null -eq $VolumeValidator) {
        $VolumeValidator = { param([string]$Path) Assert-LocalFixedNtfsPath -Path $Path }
    }
    $null = & $VolumeValidator $backupRootPath
    Assert-NoReparsePointPath -Path $backupRootPath

    if ([string]::IsNullOrWhiteSpace($GenerationName)) {
        $GenerationName = '{0}_{1}' -f $CollectedAt.ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ'), [guid]::NewGuid().ToString('N').Substring(0, 8)
    }
    if ($GenerationName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
        throw 'Backup generation name contains unsupported characters.'
    }
    if ($null -eq $HashProvider) {
        $HashProvider = {
            param([string]$Path)
            return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }

    New-Item -ItemType Directory -Path $backupRootPath -Force | Out-Null
    Assert-NoReparsePointPath -Path $backupRootPath
    $generationPath = Join-Path $backupRootPath $GenerationName
    if (Test-Path -LiteralPath $generationPath) {
        throw "Backup generation '$GenerationName' already exists."
    }

    $generationCreated = $false
    $published = $false
    $tempPath = $null
    $recordTempPath = $null
    try {
        New-Item -ItemType Directory -Path $generationPath -ErrorAction Stop | Out-Null
        $generationCreated = $true
        $sourceName = Split-Path -Leaf $source
        $backupPath = Join-Path $generationPath $sourceName
        $recordPath = Join-Path $generationPath 'backup-result.json'
        if ($sourceName.Equals('backup-result.json', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Canonical save basename conflicts with the backup result filename.'
        }
        $tempPath = Join-Path $generationPath ('.{0}.{1}.tmp' -f $sourceName, [guid]::NewGuid().ToString('N'))
        [IO.File]::Copy($source, $tempPath, $false)

        $sourceHash = [string](& $HashProvider $source)
        $tempHash = [string](& $HashProvider $tempPath)
        if ($sourceHash -notmatch '^[0-9A-Fa-f]{64}$' -or $tempHash -notmatch '^[0-9A-Fa-f]{64}$') {
            throw 'SHA-256 provider returned an invalid hash.'
        }
        $sourceHash = $sourceHash.ToLowerInvariant()
        $tempHash = $tempHash.ToLowerInvariant()
        if (-not $sourceHash.Equals($tempHash, [StringComparison]::Ordinal)) {
            throw 'Canonical save backup blocked: source and temporary copy SHA-256 mismatch.'
        }
        $sourceSize = (Get-Item -LiteralPath $source).Length
        if ($sourceSize -ne (Get-Item -LiteralPath $tempPath).Length) {
            throw 'Canonical save backup blocked: source and temporary copy size mismatch.'
        }

        [IO.File]::Move($tempPath, $backupPath)
        $tempPath = $null
        $backupHash = [string](& $HashProvider $backupPath)
        if ($backupHash -notmatch '^[0-9A-Fa-f]{64}$' -or -not $sourceHash.Equals($backupHash.ToLowerInvariant(), [StringComparison]::Ordinal)) {
            throw 'Canonical save backup blocked: finalized backup SHA-256 mismatch.'
        }

        $result = [ordered]@{
            schema_version = 'survivors.canonical-save-backup.v1'
            status = 'success'
            source_path = $source
            backup_path = $backupPath
            source_hash = $sourceHash
            backup_hash = $backupHash.ToLowerInvariant()
            size = [long]$sourceSize
            collected_at = $CollectedAt.ToUniversalTime().ToString('o')
            match = $true
            blocking_reason = $null
            record_path = $recordPath
        }
        $recordTempPath = Join-Path $generationPath ('.backup-result.{0}.tmp' -f [guid]::NewGuid().ToString('N'))
        $utf8NoBom = New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($recordTempPath, ($result | ConvertTo-Json -Depth 4), $utf8NoBom)
        [IO.File]::Move($recordTempPath, $recordPath)
        $recordTempPath = $null
        $published = $true
        return [pscustomobject]$result
    }
    finally {
        if ($null -ne $tempPath -and [IO.File]::Exists($tempPath)) {
            [IO.File]::Delete($tempPath)
        }
        if ($null -ne $recordTempPath -and [IO.File]::Exists($recordTempPath)) {
            [IO.File]::Delete($recordTempPath)
        }
        if (-not $published -and $generationCreated -and [IO.Directory]::Exists($generationPath) -and @(Get-ChildItem -LiteralPath $generationPath -Force).Count -eq 0) {
            [IO.Directory]::Delete($generationPath)
        }
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
