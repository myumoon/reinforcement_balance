#requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Assert-Equal {
    param(
        [object]$Actual,
        [object]$Expected,
        [string]$Message
    )

    if ($Actual -ne $Expected) {
        throw "$Message. Expected '$Expected', got '$Actual'."
    }
}

function Assert-Throws {
    param(
        [scriptblock]$Script,
        [string]$Message
    )

    $thrown = $false
    try {
        & $Script
    }
    catch {
        $thrown = $true
    }
    Assert-True $thrown $Message
}

function Get-TestPythonPath {
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $candidates += Join-Path $env:CONDA_PREFIX 'python.exe'
    }
    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $condaCommand) {
        $candidates += @(& $condaCommand.Source run -n reinbalance python -c 'import sys; print(sys.executable)' 2>$null)
    }
    $candidates += Join-Path ([Environment]::GetFolderPath('UserProfile')) 'anaconda3\envs\reinbalance\python.exe'
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        $candidates += $pythonCommand.Source
    }
    foreach ($candidate in ($candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        $probe = @(& $candidate -c 'import sys; print(sys.version_info >= (3, 11))' 2>$null)
        if ($LASTEXITCODE -eq 0 -and ($probe -join '').Trim() -eq 'True') {
            return $candidate
        }
    }
    throw 'A Python 3.11+ interpreter is required for the canonical hash regression test.'
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..'))
$commonPath = Join-Path $PSScriptRoot '..\SurvivorsDeployment.Common.ps1'
Assert-True (Test-Path -LiteralPath $commonPath -PathType Leaf) 'Common helper is missing.'
. $commonPath

Assert-True (Test-Path -LiteralPath (Join-Path $repoRoot '.gitignore') -PathType Leaf) 'Repository root is incorrect.'
& git -C $repoRoot check-ignore -q -- '.env/survivors_deployment.env'
Assert-Equal $LASTEXITCODE 0 '.env/survivors_deployment.env must be ignored'
Assert-True (-not (Test-Path -LiteralPath (Join-Path $repoRoot '.env/survivors_deployment.env'))) 'A real env file must not be tracked by this test.'

$examplePath = Join-Path $PSScriptRoot '..\survivors_deployment.env.example'
Assert-True (Test-Path -LiteralPath $examplePath -PathType Leaf) 'Environment template is missing.'
$exampleText = Get-Content -LiteralPath $examplePath -Raw
Assert-True ($exampleText -notmatch '(?im)^[A-Z]:\\') 'Template must not contain a drive-rooted path.'
Assert-True ($exampleText -notmatch '%USERPROFILE%|%APPDATA%|password|token|secret') 'Template must not contain user paths or secrets.'
foreach ($key in @(
        'REINBALANCE_PYTHON',
        'SURVIVORS_ARTIFACT_PRIMARY_ROOT',
        'SURVIVORS_ARTIFACT_BACKUP_ROOT',
        'SURVIVORS_EVIDENCE_ROOT',
        'VAMPIRE_SURVIVORS_EXE',
        'SURVIVORS_CANONICAL_SAVE',
        'SURVIVORS_TARGET_PROFILE'
    )) {
    Assert-True ($exampleText -match "(?m)^$key=\s*$") "Template key $key must have an empty value."
}

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("survivors-deployment-common-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
try {
    $envFile = Join-Path $tempRoot 'survivors_deployment.env'
    @'
# ignored
REINBALANCE_PYTHON = "C:\\Python311\\python.exe"
SURVIVORS_EVIDENCE_ROOT='D:\\evidence root'
SURVIVORS_CANONICAL_SAVE=
'@ | Set-Content -LiteralPath $envFile -Encoding UTF8

    $settings = Read-DeploymentEnv -Path $envFile
    Assert-Equal $settings['REINBALANCE_PYTHON'] 'C:\\Python311\\python.exe' 'Parser should trim and remove double quotes'
    Assert-Equal $settings['SURVIVORS_EVIDENCE_ROOT'] 'D:\\evidence root' 'Parser should remove single quotes'
    Assert-Equal $settings['SURVIVORS_CANONICAL_SAVE'] '' 'Parser should preserve an empty value'
    Assert-Equal (Get-RequiredDeploymentSetting -Settings $settings -Name 'REINBALANCE_PYTHON') 'C:\\Python311\\python.exe' 'Required setting should be returned'

    @'
UNKNOWN_KEY=value
'@ | Set-Content -LiteralPath $envFile -Encoding UTF8
    Assert-Throws { Read-DeploymentEnv -Path $envFile } 'Unknown env keys must be rejected'

    @'
REINBALANCE_PYTHON=one
REINBALANCE_PYTHON=two
'@ | Set-Content -LiteralPath $envFile -Encoding UTF8
    Assert-Throws { Read-DeploymentEnv -Path $envFile } 'Duplicate env keys must be rejected'

    @'
REINBALANCE_PYTHON
'@ | Set-Content -LiteralPath $envFile -Encoding UTF8
    Assert-Throws { Read-DeploymentEnv -Path $envFile } 'Malformed env rows must be rejected'

    $originalLocation = (Get-Location).Path
    Push-Location $tempRoot
    try {
        Assert-Equal (Get-RepositoryRoot) $repoRoot 'Repository root must not depend on the current directory'
    }
    finally {
        Pop-Location
    }
    Assert-Equal (Get-Location).Path $originalLocation 'The parser test must restore its current directory'

    Assert-Throws {
        Assert-PathOutsideRepository -Path (Join-Path $repoRoot 'Tools') -RepositoryRoot $repoRoot
    } 'Repository paths must be rejected as artifact destinations'
    Assert-PathOutsideRepository -Path (Join-Path $tempRoot 'external') -RepositoryRoot $repoRoot
    Assert-Throws {
        Get-NormalizedDeploymentPath -Path 'relative-external-store'
    } 'Deployment paths must reject relative values'

    $binaryPath = Join-Path $tempRoot 'hash-fixture.bin'
    [IO.File]::WriteAllBytes($binaryPath, [byte[]](0, 1, 2, 255))
    $binaryHex = (([IO.File]::ReadAllBytes($binaryPath) | ForEach-Object { '{0:x2}' -f $_ }) -join '')
    $canonicalJson = '{"bytes_hex":"' + $binaryHex + '"}'
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $expectedHash = ([BitConverter]::ToString($sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonicalJson)))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
    $configuredPython = Get-TestPythonPath
    Assert-Equal (Get-CanonicalHash -Path $binaryPath -PythonPath $configuredPython) $expectedHash 'Canonical file hash must match target_audit byte hashing'

    $fakeVolumeResolver = {
        param([string]$Path)
        if ($Path -match 'primary') { return 'volume-primary' }
        return 'volume-backup'
    }
    Assert-Throws {
        Test-ArtifactStoreRoots -PrimaryRoot (Join-Path $tempRoot 'same') -BackupRoot (Join-Path $tempRoot 'same') -VolumeResolver $fakeVolumeResolver
    } 'The same primary and backup path must be rejected'
    $roots = Test-ArtifactStoreRoots `
        -PrimaryRoot (Join-Path $tempRoot 'primary') `
        -BackupRoot (Join-Path $tempRoot 'backup') `
        -DryRun `
        -VolumeResolver $fakeVolumeResolver
    Assert-Equal $roots.PrimaryVolume 'volume-primary' 'Primary volume should be returned'
    Assert-Equal $roots.BackupVolume 'volume-backup' 'Backup volume should be returned'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $tempRoot 'primary'))) 'Dry-run must not create primary root'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $tempRoot 'backup'))) 'Dry-run must not create backup root'

    Assert-Throws {
        Invoke-ConfiguredPython -PythonPath (Join-Path $tempRoot 'missing-python.exe') -Arguments @('--version')
    } 'A missing configured Python must be rejected before invocation'
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

Write-Host 'Survivors deployment common tests passed.'
