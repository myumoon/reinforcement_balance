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
    throw 'A Python 3.11+ interpreter is required for the deployment workflow integration test.'
}

function Invoke-Workflow {
    param(
        [Parameter(Mandatory)][string]$ScriptPath,
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [string[]]$Arguments = @()
    )

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath -RepositoryRoot $RepositoryRoot @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    [pscustomobject]@{
        ExitCode = $exitCode
        Output   = ($output -join "`n")
    }
}

function Write-TestEnv {
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [hashtable]$Values = @{}
    )

    $envDir = Join-Path $RepositoryRoot '.env'
    New-Item -ItemType Directory -Path $envDir -Force | Out-Null
    $keys = @(
        'REINBALANCE_PYTHON',
        'SURVIVORS_ARTIFACT_PRIMARY_ROOT',
        'SURVIVORS_ARTIFACT_BACKUP_ROOT',
        'SURVIVORS_EVIDENCE_ROOT',
        'VAMPIRE_SURVIVORS_EXE',
        'SURVIVORS_CANONICAL_SAVE',
        'SURVIVORS_TARGET_PROFILE'
    )
    $lines = foreach ($key in $keys) {
        $value = if ($Values.ContainsKey($key)) { [string]$Values[$key] } else { '' }
        "$key=$value"
    }
    Set-Content -LiteralPath (Join-Path $envDir 'survivors_deployment.env') -Value $lines -Encoding UTF8
}

$scriptRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = [IO.Path]::GetFullPath((Join-Path $scriptRoot '..\..\..'))
$workflowNames = @(
    '00_ConfigureLocalEnvironment',
    '01_SetupDeploymentEnvironment',
    '02_PrepareArtifactStore',
    '03_CollectTargetEvidence',
    '04_VerifyDeploymentSmoke'
)

foreach ($name in $workflowNames) {
    $batPath = Join-Path $scriptRoot "$name.bat"
    $ps1Path = Join-Path $scriptRoot "$name.ps1"
    Assert-True (Test-Path -LiteralPath $batPath -PathType Leaf) "$name.bat is missing."
    Assert-True (Test-Path -LiteralPath $ps1Path -PathType Leaf) "$name.ps1 is missing."
    $bat = Get-Content -LiteralPath $batPath -Raw
    Assert-True ($bat -match '%~dp0') "$name.bat must resolve its script from %~dp0."
    Assert-True ($bat -match [regex]::Escape("$name.ps1")) "$name.bat must launch its matching PowerShell script."
    Assert-True ($bat -match '(?im)^pause\s*$') "$name.bat must pause for Explorer users."
    Assert-True ($bat -match '(?im)^exit /b %exitCode%\s*$') "$name.bat must return the PowerShell exit code."
}

$readmePath = Join-Path $scriptRoot 'README.md'
Assert-True (Test-Path -LiteralPath $readmePath -PathType Leaf) 'Operator README is missing.'
$readme = Get-Content -LiteralPath $readmePath -Raw
foreach ($requiredText in @(
        '.env/survivors_deployment.env',
        '別ボリューム',
        'PR #308',
        'D04',
        'formal_dataset_eligible=false',
        '%LOCALAPPDATA%\ReinBalance\DeploymentLogs',
        '00_ConfigureLocalEnvironment.bat',
        '04_VerifyDeploymentSmoke.bat'
    )) {
    Assert-True ($readme -match [regex]::Escape($requiredText)) "README must document '$requiredText'."
}

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("survivors-deployment-scripts-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$beforeGenerated = @(Get-ChildItem -LiteralPath $scriptRoot -Recurse -File -Include '*.log', '*.json', '*.png', '*.mp4' -ErrorAction SilentlyContinue | ForEach-Object FullName)
try {
    $configure = Invoke-Workflow -ScriptPath (Join-Path $scriptRoot '00_ConfigureLocalEnvironment.ps1') -RepositoryRoot $tempRoot
    Assert-Equal $configure.ExitCode 0 '00 workflow should create a missing local env file'
    $envPath = Join-Path $tempRoot '.env\survivors_deployment.env'
    Assert-True (Test-Path -LiteralPath $envPath -PathType Leaf) '00 workflow must create .env/survivors_deployment.env'
    $initialEnv = Get-Content -LiteralPath $envPath -Raw
    Assert-True ($initialEnv -match '(?m)^REINBALANCE_PYTHON=\s*$') '00 workflow must copy the empty template'

    'REINBALANCE_PYTHON=sentinel-python' | Set-Content -LiteralPath $envPath -Encoding UTF8
    $configureAgain = Invoke-Workflow -ScriptPath (Join-Path $scriptRoot '00_ConfigureLocalEnvironment.ps1') -RepositoryRoot $tempRoot
    Assert-Equal $configureAgain.ExitCode 0 '00 workflow should validate an existing env file'
    Assert-Equal (Get-Content -LiteralPath $envPath -Raw) "REINBALANCE_PYTHON=sentinel-python`r`n" '00 workflow must not overwrite an existing env file'
    Assert-True ($configureAgain.Output -notmatch 'sentinel-python') '00 workflow must not print configured values'

    $missingPython = Invoke-Workflow -ScriptPath (Join-Path $scriptRoot '01_SetupDeploymentEnvironment.ps1') -RepositoryRoot $tempRoot
    Assert-True ($missingPython.ExitCode -ne 0) '01 workflow must fail when REINBALANCE_PYTHON is missing'

    Write-TestEnv -RepositoryRoot $tempRoot -Values @{ REINBALANCE_PYTHON = (Join-Path $tempRoot 'not-a-python.exe') }
    $invalidPython = Invoke-Workflow -ScriptPath (Join-Path $scriptRoot '01_SetupDeploymentEnvironment.ps1') -RepositoryRoot $tempRoot
    Assert-True ($invalidPython.ExitCode -ne 0) '01 workflow must fail when configured Python does not exist'
    Assert-True ($invalidPython.Output -notmatch 'not-a-python\.exe') '01 workflow must not print configured path values'
    $setupText = Get-Content -LiteralPath (Join-Path $scriptRoot '01_SetupDeploymentEnvironment.ps1') -Raw
    Assert-True ($setupText -match 'Tools\\Common' -and $setupText -match 'Tools\\Deployment' -and $setupText -match 'requirements\.lock') '01 workflow must use fixed repository package paths and lock file'
    Assert-True ($setupText -match 'Invoke-ConfiguredPython') '01 workflow must use configured Python rather than PATH Python'
    Assert-True ($setupText -notmatch '--user') '01 workflow must not install with --user'

    $storeText = Get-Content -LiteralPath (Join-Path $scriptRoot '02_PrepareArtifactStore.ps1') -Raw
    Assert-True ($storeText -match 'Test-ArtifactStoreRoots' -and $storeText -match 'check-backup-root' -and $storeText -match 'DryRun') '02 workflow must validate and prepare both stores'
    Write-TestEnv -RepositoryRoot $tempRoot -Values @{
        SURVIVORS_ARTIFACT_PRIMARY_ROOT = (Join-Path $repoRoot 'Tools\Deployment\forbidden-store')
        SURVIVORS_ARTIFACT_BACKUP_ROOT = (Join-Path $tempRoot 'backup')
        SURVIVORS_EVIDENCE_ROOT = (Join-Path $tempRoot 'evidence')
    }
    $repositoryStore = Invoke-Workflow -ScriptPath (Join-Path $scriptRoot '02_PrepareArtifactStore.ps1') -RepositoryRoot $tempRoot -Arguments @('-DryRun')
    Assert-True ($repositoryStore.ExitCode -ne 0) '02 workflow must reject a store inside the repository'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $tempRoot 'backup'))) '02 dry-run failure must not create a backup directory'

    $evidenceText = Get-Content -LiteralPath (Join-Path $scriptRoot '03_CollectTargetEvidence.ps1') -Raw
    Assert-True ($evidenceText -match 'GetMachineInfo\.ps1' -and $evidenceText -match 'GetVampireSurvivorsExecutableMetadata\.ps1' -and $evidenceText -match 'Get-CanonicalHash') '03 workflow must call existing evidence utilities and shared canonical hashing'
    Assert-True ($evidenceText -notmatch 'Get-FileHash') '03 workflow must not implement canonical hashing with raw Get-FileHash'
    Assert-True ($evidenceText -match 'artifact_ref' -and $evidenceText -match 'store_uri' -and $evidenceText -match 'machine-info' -and $evidenceText -match 'executable-metadata') '03 workflow must register generated evidence files and record their artifact references'
    $commonText = Get-Content -LiteralPath (Join-Path $scriptRoot 'SurvivorsDeployment.Common.ps1') -Raw
    Assert-True ($commonText -match 'IsPathRooted') 'Deployment path normalization must reject relative paths'
    $evidenceRoot = Join-Path $tempRoot 'evidence'
    $primaryRoot = Join-Path $tempRoot 'primary'
    New-Item -ItemType Directory -Path $evidenceRoot, $primaryRoot -Force | Out-Null
    Write-TestEnv -RepositoryRoot $tempRoot -Values @{
        SURVIVORS_ARTIFACT_PRIMARY_ROOT = $primaryRoot
        SURVIVORS_EVIDENCE_ROOT = $evidenceRoot
        VAMPIRE_SURVIVORS_EXE = $tempRoot
    }
    $invalidExe = Invoke-Workflow -ScriptPath (Join-Path $scriptRoot '03_CollectTargetEvidence.ps1') -RepositoryRoot $tempRoot
    Assert-True ($invalidExe.ExitCode -ne 0) '03 workflow must reject a configured directory as the executable'
    Assert-True ($invalidExe.Output -notmatch [regex]::Escape($tempRoot)) '03 workflow must not print configured path values'
    Write-TestEnv -RepositoryRoot $tempRoot -Values @{
        SURVIVORS_ARTIFACT_PRIMARY_ROOT = $primaryRoot
        SURVIVORS_EVIDENCE_ROOT = $evidenceRoot
        VAMPIRE_SURVIVORS_EXE = 'relative\path\VampireSurvivors.exe'
    }
    $relativeExe = Invoke-Workflow -ScriptPath (Join-Path $scriptRoot '03_CollectTargetEvidence.ps1') -RepositoryRoot $tempRoot
    Assert-True ($relativeExe.ExitCode -ne 0) '03 workflow must reject a relative path for VAMPIRE_SURVIVORS_EXE'

    $repoEnvPath = Join-Path $repoRoot '.env\survivors_deployment.env'
    $repoEnvDirectory = Split-Path -Parent $repoEnvPath
    $savedRepoEnv = $null
    if (Test-Path -LiteralPath $repoEnvPath -PathType Leaf) {
        $savedRepoEnv = Get-Content -LiteralPath $repoEnvPath -Raw
    }
    $artifactRunRoot = Join-Path ([IO.Path]::GetTempPath()) ('survivors-deployment-artifact-run-' + [guid]::NewGuid().ToString('N'))
    $artifactPrimary = Join-Path $artifactRunRoot 'primary'
    $artifactEvidence = Join-Path $artifactRunRoot 'evidence'
    $fakeExe = Join-Path $artifactRunRoot 'VampireSurvivors.exe'
    $testPython = Get-TestPythonPath
    New-Item -ItemType Directory -Path $artifactPrimary, $artifactEvidence -Force | Out-Null
    [IO.File]::WriteAllBytes($fakeExe, [byte[]](1, 2, 3, 4))
    try {
        New-Item -ItemType Directory -Path $repoEnvDirectory -Force | Out-Null
        @(
            "REINBALANCE_PYTHON=$testPython",
            "SURVIVORS_ARTIFACT_PRIMARY_ROOT=$artifactPrimary",
            'SURVIVORS_ARTIFACT_BACKUP_ROOT=',
            "SURVIVORS_EVIDENCE_ROOT=$artifactEvidence",
            "VAMPIRE_SURVIVORS_EXE=$fakeExe",
            'SURVIVORS_CANONICAL_SAVE=',
            'SURVIVORS_TARGET_PROFILE='
        ) | Set-Content -LiteralPath $repoEnvPath -Encoding UTF8
        $oldPythonPath = $env:PYTHONPATH
        $env:PYTHONPATH = Join-Path $repoRoot 'Tools\Common\src'
        try {
            $evidenceRun = Invoke-Workflow -ScriptPath (Join-Path $scriptRoot '03_CollectTargetEvidence.ps1') -RepositoryRoot $repoRoot
        }
        finally {
            $env:PYTHONPATH = $oldPythonPath
        }
        Assert-Equal $evidenceRun.ExitCode 0 '03 workflow should register all generated evidence files'
        $manifest = @(Get-ChildItem -LiteralPath $artifactEvidence -Filter 'target-evidence-*.json' -File)
        Assert-Equal $manifest.Count 1 '03 workflow must create one target evidence manifest'
        $manifestJson = Get-Content -LiteralPath $manifest[0].FullName -Raw | ConvertFrom-Json
        foreach ($record in @($manifestJson.machine_info, $manifestJson.executable_metadata)) {
            Assert-True (-not [string]::IsNullOrWhiteSpace([string]$record.canonical_hash)) 'Evidence record must include a canonical hash'
            Assert-True (-not [string]::IsNullOrWhiteSpace([string]$record.artifact_ref.store_uri)) 'Evidence record must include an artifact URI'
            Assert-True (([string]$record.artifact_ref.sha256).Length -eq 64) 'Evidence record must include an artifact content hash'
        }
        $primaryObjects = @(Get-ChildItem -LiteralPath (Join-Path $artifactPrimary 'objects\sha256') -Recurse -File)
        Assert-True ($primaryObjects.Count -ge 3) '03 workflow must publish manifest and generated evidence objects'
    }
    finally {
        if ($null -eq $savedRepoEnv) {
            if (Test-Path -LiteralPath $repoEnvPath) { Remove-Item -LiteralPath $repoEnvPath -Force }
        }
        else {
            Set-Content -LiteralPath $repoEnvPath -Value $savedRepoEnv -Encoding UTF8
        }
        if (Test-Path -LiteralPath $artifactRunRoot) {
            Remove-Item -LiteralPath $artifactRunRoot -Recurse -Force
        }
    }

    $smokeText = Get-Content -LiteralPath (Join-Path $scriptRoot '04_VerifyDeploymentSmoke.ps1') -Raw
    Assert-True ($smokeText -match '--synthetic' -and $smokeText -match '--duration-sec' -and $smokeText -match '--store-root') '04 workflow must use the safe synthetic CLI'
    Assert-True ($smokeText -notmatch '--duration-seconds') '04 workflow must use the current duration-sec option'
    $smokeRoot = Join-Path $tempRoot 'smoke-root'
    New-Item -ItemType Directory -Path $smokeRoot -Force | Out-Null
    Write-TestEnv -RepositoryRoot $tempRoot -Values @{ SURVIVORS_ARTIFACT_PRIMARY_ROOT = $smokeRoot; REINBALANCE_PYTHON = (Join-Path $tempRoot 'missing-python.exe') }
    $missingCapture = Invoke-Workflow -ScriptPath (Join-Path $scriptRoot '04_VerifyDeploymentSmoke.ps1') -RepositoryRoot $tempRoot
    Assert-True ($missingCapture.ExitCode -ne 0) '04 workflow must fail before execution when PR #308 capture CLI is absent'
    Assert-True ($missingCapture.Output -match 'PR #308') '04 workflow must explain that PR #308 is required'
    Assert-True ($missingCapture.Output -notmatch [regex]::Escape($tempRoot)) '04 workflow must not print configured path values'
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

$afterGenerated = @(Get-ChildItem -LiteralPath $scriptRoot -Recurse -File -Include '*.log', '*.json', '*.png', '*.mp4' -ErrorAction SilentlyContinue | ForEach-Object FullName)
Assert-Equal (($afterGenerated | Sort-Object) -join "`n") (($beforeGenerated | Sort-Object) -join "`n") 'Deployment utility tests must not leave generated output in the repository'

Write-Host 'Survivors deployment script tests passed.'
