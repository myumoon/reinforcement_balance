#requires -Version 5.1

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$outputDir = Join-Path $scriptDir 'output'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$errors = [System.Collections.Generic.List[string]]::new()

try {
    $gpu = @(
        Get-CimInstance -ClassName Win32_VideoController |
            Select-Object Name,
                @{Name = 'VRAM_GiB'; Expression = {
                    if ($null -eq $_.AdapterRAM) { $null }
                    else { [math]::Round($_.AdapterRAM / 1GB, 2) }
                }},
                DriverVersion,
                DriverDate
    )
}
catch {
    $gpu = @()
    $errors.Add("GPU information failed: $($_.Exception.Message)")
}

try {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem |
        Select-Object Caption, Version, BuildNumber
}
catch {
    $os = $null
    $errors.Add("OS information failed: $($_.Exception.Message)")
}

$logPixels = $null
$dpi = $null
$scalePercent = $null
$scaleSource = $null

try {
    if ($null -eq ('MachineInfoDisplayScaleNative' -as [type])) {
        Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;

public static class MachineInfoDisplayScaleNative
{
    [DllImport("user32.dll")]
    public static extern uint GetDpiForSystem();
}
'@
    }

    $dpi = [MachineInfoDisplayScaleNative]::GetDpiForSystem()
    if ($dpi -gt 0) {
        $logPixels = [int]$dpi
        $scalePercent = [math]::Round(([double]$dpi / 96) * 100, 0)
        $scaleSource = 'user32.GetDpiForSystem'
    }
}
catch {
    # Fall back to the registry when the Windows API is unavailable.
}

if ($null -eq $scalePercent) {
    try {
        $desktopSettings = Get-ItemProperty -Path 'HKCU:\Control Panel\Desktop'
        $logPixels = $desktopSettings.LogPixels
        if ($null -ne $logPixels) {
            $dpi = [int]$logPixels
            $scalePercent = [math]::Round(([double]$logPixels / 96) * 100, 0)
            $scaleSource = 'HKCU:\Control Panel\Desktop\LogPixels'
        }
    }
    catch {
        $errors.Add("DPI information failed: $($_.Exception.Message)")
    }
}

if ($null -eq $scalePercent) {
    $errors.Add('DPI information was not available. Check Windows Settings manually.')
}

$result = [ordered]@{
    schema_version  = 'machine_info.v1'
    collected_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    computer_name   = $env:COMPUTERNAME
    gpu             = $gpu
    os              = $os
    display_scaling = [ordered]@{
        log_pixels             = $logPixels
        dpi                    = $dpi
        scale_percent          = $scalePercent
        source                 = $scaleSource
        scope                  = 'system/primary display'
        manual_check_required  = ($null -eq $scalePercent)
        manual_check_location  = 'Windows Settings > System > Display > Scale'
    }
    errors          = @($errors)
}

$timestamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$outputPath = Join-Path $outputDir "machine_info_$timestamp.json"

$result |
    ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $outputPath -Encoding UTF8

Write-Host "MachineInfo collection completed."
Write-Host "Output: $outputPath"
if ($errors.Count -gt 0) {
    Write-Warning 'Some information could not be collected. Check the errors field in the JSON.'
}
