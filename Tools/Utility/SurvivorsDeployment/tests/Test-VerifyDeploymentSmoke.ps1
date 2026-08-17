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

function Invoke-Workflow {
    param(
        [Parameter(Mandatory)][string]$ScriptPath,
        [Parameter(Mandatory)][string]$RepositoryRoot
    )

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath -RepositoryRoot $RepositoryRoot 2>&1)
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

function Find-OperatorLogContaining {
    param(
        [Parameter(Mandatory)][string]$Text,
        [Parameter(Mandatory)][datetime]$NotBeforeUtc
    )

    $roots = @(
        (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'ReinBalance\DeploymentLogs'),
        (Join-Path ([IO.Path]::GetTempPath()) 'ReinBalance\DeploymentLogs')
    ) | Select-Object -Unique
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root -PathType Container)) { continue }
        $match = Get-ChildItem -LiteralPath $root -Filter 'verify-deployment-smoke_*.log' -File |
            Where-Object { $_.LastWriteTimeUtc -ge $NotBeforeUtc } |
            Sort-Object LastWriteTimeUtc -Descending |
            Where-Object { (Get-Content -LiteralPath $_.FullName -Raw) -match [regex]::Escape($Text) } |
            Select-Object -First 1
        if ($null -ne $match) { return $match.FullName }
    }
    return $null
}

function Test-PytestFailureLogging {
    param(
        [Parameter(Mandatory)][string]$Workflow,
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][string]$FixtureDeployment,
        [Parameter(Mandatory)][string]$FakeCalls,
        [Parameter(Mandatory)][string]$FailureStage,
        [Parameter(Mandatory)][int]$FailureExitCode
    )

    $sentinel = 'EXPECTED_PYTEST_FAILURE_' + [guid]::NewGuid().ToString('N')
    $inheritedPythonPath = 'inherited-python-path'
    $expectedPythonPath = $FixtureDeployment + [IO.Path]::PathSeparator + $inheritedPythonPath
    $savedPythonPath = $env:PYTHONPATH
    $savedFailureStage = $env:FAKE_PYTEST_FAILURE_STAGE
    $savedSentinel = $env:FAKE_PYTEST_SENTINEL
    $logSearchStart = (Get-Date).ToUniversalTime().AddSeconds(-1)
    $failureLog = $null
    try {
        if (Test-Path -LiteralPath $FakeCalls) { Remove-Item -LiteralPath $FakeCalls -Force }
        $env:PYTHONPATH = $inheritedPythonPath
        $env:FAKE_PYTEST_FAILURE_STAGE = $FailureStage
        $env:FAKE_PYTEST_SENTINEL = $sentinel
        $result = Invoke-Workflow -ScriptPath $Workflow -RepositoryRoot $RepositoryRoot

        Assert-True ($result.ExitCode -ne 0) "04 workflow must fail when $FailureStage fails"
        Assert-True (Test-Path -LiteralPath $FakeCalls -PathType Leaf) "04 workflow must invoke the configured Python fixture. Output: $($result.Output)"
        $pytestCalls = @(Get-Content -LiteralPath $FakeCalls | Where-Object { $_ -match 'ARGS=-m pytest' })
        $expectedCallCount = if ($FailureStage -eq 'capture_annotation_tests') { 1 } else { 2 }
        Assert-Equal $pytestCalls.Count $expectedCallCount "04 workflow must stop after the failing $FailureStage stage"
        foreach ($call in $pytestCalls) {
            Assert-True ($call.StartsWith("PYTHONPATH=$expectedPythonPath|")) '04 workflow must prepend Tools/Deployment while preserving the inherited PYTHONPATH'
        }
        $failureLog = Find-OperatorLogContaining -Text $sentinel -NotBeforeUtc $logSearchStart
        Assert-True (-not [string]::IsNullOrWhiteSpace($failureLog)) '04 workflow must write stderr from pytest to the external operator log'
        $failureLogText = Get-Content -LiteralPath $failureLog -Raw
        Assert-True ($failureLogText -match "(?m)^failure_stage=$FailureStage\r?$") '04 workflow must log the failing pytest stage'
        Assert-True ($failureLogText -match "(?m)^pytest_exit_code=$FailureExitCode\r?$") '04 workflow must log the actual pytest exit code'
        Assert-True ($failureLogText -match '(?m)^pytest_output_begin\r?$') '04 workflow must mark the beginning of pytest output'
        Assert-True ($failureLogText -match '(?m)^pytest_output_end\r?$') '04 workflow must mark the end of pytest output'
    }
    finally {
        $env:PYTHONPATH = $savedPythonPath
        $env:FAKE_PYTEST_FAILURE_STAGE = $savedFailureStage
        $env:FAKE_PYTEST_SENTINEL = $savedSentinel
        if ([string]::IsNullOrWhiteSpace($failureLog)) {
            $failureLog = Find-OperatorLogContaining -Text $sentinel -NotBeforeUtc $logSearchStart
        }
        if (-not [string]::IsNullOrWhiteSpace($failureLog) -and (Test-Path -LiteralPath $failureLog -PathType Leaf)) {
            Remove-Item -LiteralPath $failureLog -Force
        }
    }
}

$scriptRoot = Split-Path -Parent $PSScriptRoot
$workflow = Join-Path $scriptRoot '04_VerifyDeploymentSmoke.ps1'
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('verify-deployment-smoke-' + [guid]::NewGuid().ToString('N'))
$repositoryRoot = Join-Path $tempRoot 'repository'
$fixtureDeployment = Join-Path $repositoryRoot 'Tools\Deployment'
$fixtureStore = Join-Path $tempRoot 'artifact-store'
$envDirectory = Join-Path $repositoryRoot '.env'
New-Item -ItemType Directory -Path $fixtureDeployment, $fixtureStore, $envDirectory -Force | Out-Null
try {
    New-Item -ItemType File -Path (Join-Path $fixtureDeployment 'capture_survivors.py') -Force | Out-Null
    $fakePython = Join-Path $tempRoot 'fake-python.cmd'
    $fakeCalls = Join-Path $tempRoot 'fake-python-calls.txt'
    @'
@echo off
setlocal
>> "%~dp0fake-python-calls.txt" echo PYTHONPATH=%PYTHONPATH%^|ARGS=%*
echo %* | findstr /C:"--dry-run" >nul && (echo DRY_RUN_OK& exit /b 0)
echo %* | findstr /C:"test_capture_dataset.py" >nul && (if "%FAKE_PYTEST_FAILURE_STAGE%"=="capture_annotation_tests" (echo %FAKE_PYTEST_SENTINEL% 1>&2& exit /b 5) else (echo capture annotation tests passed& exit /b 0))
echo %* | findstr /C:"Tools/Deployment/tests -q" >nul && (echo %FAKE_PYTEST_SENTINEL% 1>&2& exit /b 7)
echo {"status":"PUBLISHED","formal_dataset_eligible":false}
exit /b 0
'@ | Set-Content -LiteralPath $fakePython -Encoding ASCII
    @(
        "REINBALANCE_PYTHON=$fakePython",
        "SURVIVORS_ARTIFACT_PRIMARY_ROOT=$fixtureStore",
        'SURVIVORS_ARTIFACT_BACKUP_ROOT=',
        'SURVIVORS_EVIDENCE_ROOT=',
        'VAMPIRE_SURVIVORS_EXE=',
        'SURVIVORS_CANONICAL_SAVE=',
        'SURVIVORS_TARGET_PROFILE='
    ) | Set-Content -LiteralPath (Join-Path $envDirectory 'survivors_deployment.env') -Encoding UTF8

    Test-PytestFailureLogging -Workflow $workflow -RepositoryRoot $repositoryRoot -FixtureDeployment $fixtureDeployment -FakeCalls $fakeCalls -FailureStage 'capture_annotation_tests' -FailureExitCode 5
    Test-PytestFailureLogging -Workflow $workflow -RepositoryRoot $repositoryRoot -FixtureDeployment $fixtureDeployment -FakeCalls $fakeCalls -FailureStage 'deployment_tests' -FailureExitCode 7
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

Write-Host 'Verify deployment smoke regression test passed.'
