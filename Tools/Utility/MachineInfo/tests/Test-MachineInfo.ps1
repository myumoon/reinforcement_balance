#requires -Version 5.1

$ErrorActionPreference = 'Stop'

$modulePath = Join-Path $PSScriptRoot '..\MachineInfo.psm1'
Import-Module -Name $modulePath -Force

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

$gpu = @(Convert-NvidiaSmiOutput -Lines @(
    'NVIDIA GeForce RTX 3080 Ti, 12288 MiB, 610.47'
))

Assert-Equal $gpu.Count 1 'A valid nvidia-smi row should produce one GPU record'
Assert-Equal $gpu[0].Name 'NVIDIA GeForce RTX 3080 Ti' 'GPU name should be preserved'
Assert-Equal $gpu[0].VRAM_GiB 12 'VRAM MiB should be converted to GiB'
Assert-Equal $gpu[0].DriverVersion '610.47' 'Driver version should be preserved'
Assert-Equal $gpu[0].Source 'nvidia-smi' 'The source should identify nvidia-smi'

$invalid = @(Convert-NvidiaSmiOutput -Lines @('malformed output'))
Assert-Equal $invalid.Count 0 'Malformed nvidia-smi output should be ignored'

Write-Host 'MachineInfo tests passed.'
