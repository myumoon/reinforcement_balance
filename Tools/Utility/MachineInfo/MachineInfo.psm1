function Convert-NvidiaSmiOutput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [string[]]$Lines
    )

    foreach ($line in $Lines) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        $fields = $line -split '\s*,\s*', 3
        if ($fields.Count -ne 3) {
            continue
        }

        $memoryText = $fields[1].Trim() -replace '\s*MiB$', ''
        if ($memoryText -notmatch '^\d+(?:\.\d+)?$') {
            continue
        }

        [pscustomobject]@{
            Name          = $fields[0].Trim()
            VRAM_GiB      = [math]::Round(([double]$memoryText / 1024), 2)
            DriverVersion = $fields[2].Trim()
            DriverDate    = $null
            Source        = 'nvidia-smi'
        }
    }
}

function Get-NvidiaGpuInfo {
    [CmdletBinding()]
    param()

    $command = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return @()
    }

    $lines = @(
        & $command.Source --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null
    )

    return @(Convert-NvidiaSmiOutput -Lines $lines)
}

Export-ModuleMember -Function Convert-NvidiaSmiOutput, Get-NvidiaGpuInfo
