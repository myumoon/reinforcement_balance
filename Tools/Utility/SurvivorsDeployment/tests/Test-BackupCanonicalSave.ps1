#requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Assert-Equal {
    param([object]$Actual, [object]$Expected, [string]$Message)
    if ($Actual -ne $Expected) { throw "$Message. Expected '$Expected', got '$Actual'." }
}

function Assert-Throws {
    param([scriptblock]$Script, [string]$Pattern, [string]$Message)
    try {
        & $Script
    }
    catch {
        Assert-True ($_.Exception.Message -match $Pattern) "$Message. Unexpected error: $($_.Exception.Message)"
        return
    }
    throw "$Message. No error was raised."
}

function Invoke-BackupCli {
    param([string[]]$Arguments)

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $scriptPath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = ($output -join "`n")
    }
}

$scriptRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $scriptRoot '05_BackupCanonicalSave.ps1'
$commonPath = Join-Path $scriptRoot 'SurvivorsDeployment.Common.ps1'
. $commonPath

$exampleText = Get-Content -LiteralPath (Join-Path $scriptRoot 'survivors_deployment.env.example') -Raw
Assert-True ($exampleText -match '(?m)^SURVIVORS_CANONICAL_SAVE_BACKUP_ROOT=\s*$') 'The env template must expose an empty canonical save backup root.'

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('survivors-canonical-save-backup-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot | Out-Null
try {
    $source = Join-Path $tempRoot 'SaveData'
    [IO.File]::WriteAllBytes($source, [Text.Encoding]::ASCII.GetBytes('abc'))
    $volumeValidator = { param([string]$Path) }

    $successRoot = Join-Path $tempRoot 'success'
    $collectedAt = [DateTimeOffset]::Parse('2026-08-24T01:02:03Z')
    $result = Invoke-CanonicalSaveBackup `
        -SourcePath $source `
        -BackupRoot $successRoot `
        -GenerationName '20260824T010203000Z_test' `
        -CollectedAt $collectedAt `
        -VolumeValidator $volumeValidator

    $expectedHash = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    $expectedBackup = Join-Path $successRoot '20260824T010203000Z_test\SaveData'
    Assert-Equal $result.schema_version 'survivors.canonical-save-backup.v1' 'Success must use the versioned result schema'
    Assert-Equal $result.status 'success' 'Only a verified backup may report success'
    Assert-Equal $result.source_path ([IO.Path]::GetFullPath($source)) 'The result must record the normalized source path'
    Assert-Equal $result.backup_path ([IO.Path]::GetFullPath($expectedBackup)) 'The result must record the generation path'
    Assert-Equal $result.source_hash $expectedHash 'The result must record the source SHA-256'
    Assert-Equal $result.backup_hash $expectedHash 'The result must record the backup SHA-256'
    Assert-Equal $result.size 3 'The result must record the source byte size'
    Assert-Equal $result.collected_at '2026-08-24T01:02:03.0000000+00:00' 'The result must record a UTC collection time'
    Assert-True $result.match 'A successful result must record a hash match'
    Assert-True (Test-Path -LiteralPath $expectedBackup -PathType Leaf) 'The verified backup must be finalized'
    Assert-Equal ([Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($expectedBackup))) 'abc' 'The backup bytes must match the source fixture'
    $record = Get-Content -LiteralPath $result.record_path -Raw | ConvertFrom-Json
    Assert-Equal $record.backup_path $result.backup_path 'The JSON sidecar must record the finalized backup path'
    Assert-Equal $record.source_hash $expectedHash 'The JSON sidecar must record the verified source hash'
    Assert-True $record.match 'The JSON sidecar must record the hash match'

    $mismatchRoot = Join-Path $tempRoot 'mismatch'
    $mismatchHasher = {
        param([string]$Path)
        if ([IO.Path]::GetFileName($Path) -match '\.tmp$') { return ('b' * 64) }
        return ('a' * 64)
    }
    Assert-Throws {
        Invoke-CanonicalSaveBackup `
            -SourcePath $source `
            -BackupRoot $mismatchRoot `
            -GenerationName 'mismatch_generation' `
            -HashProvider $mismatchHasher `
            -VolumeValidator $volumeValidator
    } 'SHA-256 mismatch' 'A copied temporary file with a different hash must be rejected'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $mismatchRoot 'mismatch_generation\SaveData'))) 'A hash mismatch must not publish a backup'
    Assert-Equal ([Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($source))) 'abc' 'A hash mismatch must not modify the source'

    $protectedRoot = Join-Path $tempRoot 'protected'
    $protectedGeneration = Join-Path $protectedRoot 'existing_generation'
    New-Item -ItemType Directory -Path $protectedGeneration -Force | Out-Null
    $protectedBackup = Join-Path $protectedGeneration 'SaveData'
    [IO.File]::WriteAllText($protectedBackup, 'sentinel')
    Assert-Throws {
        Invoke-CanonicalSaveBackup `
            -SourcePath $source `
            -BackupRoot $protectedRoot `
            -GenerationName 'existing_generation' `
            -VolumeValidator $volumeValidator
    } 'already exists' 'An existing generation must be protected from overwrite'
    Assert-Equal ([IO.File]::ReadAllText($protectedBackup)) 'sentinel' 'An existing backup must remain unchanged'

    Assert-Throws {
        Invoke-CanonicalSaveBackup `
            -SourcePath (Join-Path $tempRoot 'missing-save') `
            -BackupRoot (Join-Path $tempRoot 'missing-input') `
            -GenerationName 'missing_input' `
            -VolumeValidator $volumeValidator
    } 'existing regular file' 'A missing canonical save must fail closed'

    Assert-Throws {
        Invoke-CanonicalSaveBackup `
            -SourcePath $source `
            -BackupRoot $source `
            -GenerationName 'same_path' `
            -VolumeValidator $volumeValidator
    } 'same path' 'The source and backup root must not be the same path'

    Assert-Throws {
        Invoke-CanonicalSaveBackup `
            -SourcePath $source `
            -BackupRoot (Join-Path $tempRoot 'unsupported-volume') `
            -GenerationName 'unsupported_volume' `
            -VolumeValidator { throw 'Backup root must be on a local fixed NTFS volume.' }
    } 'local fixed NTFS' 'An unsupported backup volume must fail before copying'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $tempRoot 'unsupported-volume'))) 'Volume rejection must not create the backup root'

    $missingEnv = Join-Path $tempRoot 'missing-setting.env'
    @(
        "SURVIVORS_CANONICAL_SAVE=$source",
        'SURVIVORS_CANONICAL_SAVE_BACKUP_ROOT='
    ) | Set-Content -LiteralPath $missingEnv -Encoding UTF8
    $missingSetting = Invoke-BackupCli -Arguments @('-EnvPath', $missingEnv)
    Assert-True ($missingSetting.ExitCode -ne 0) 'The CLI must fail when the backup root setting is missing'
    $missingSettingJson = $missingSetting.Output | ConvertFrom-Json
    Assert-Equal $missingSettingJson.status 'failure' 'A configuration failure must return failure JSON'
    Assert-True ([string]$missingSettingJson.blocking_reason -match 'SURVIVORS_CANONICAL_SAVE_BACKUP_ROOT') 'Failure JSON must identify the missing setting'

    $missingInput = Invoke-BackupCli -Arguments @(
        '-CanonicalSavePath', (Join-Path $tempRoot 'does-not-exist'),
        '-BackupRoot', (Join-Path $tempRoot 'cli-missing-input')
    )
    Assert-True ($missingInput.ExitCode -ne 0) 'The CLI must fail when the configured source is missing'
    $missingInputJson = $missingInput.Output | ConvertFrom-Json
    Assert-Equal $missingInputJson.status 'failure' 'A missing input must return failure JSON'
    Assert-True ([string]$missingInputJson.blocking_reason -match 'existing regular file') 'Failure JSON must include a blocking reason for the missing input'
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

Write-Host 'Canonical save backup tests passed.'
