#requires -Version 5.1

[CmdletBinding()]
param(
    [string]$ExePath = 'C:\Program Files (x86)\Steam\steamapps\common\Vampire Survivors\VampireSurvivors.exe'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) {
    throw "ゲーム実行ファイルが見つかりません: $ExePath"
}

$file = Get-Item -LiteralPath $ExePath
$versionInfo = $file.VersionInfo
$sha256 = (Get-FileHash -LiteralPath $ExePath -Algorithm SHA256).Hash

$result = [ordered]@{
    collected_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    executable       = [ordered]@{
        path                = $file.FullName
        file_version        = $versionInfo.FileVersion
        product_version     = $versionInfo.ProductVersion
        sha256              = $sha256
        size_bytes          = $file.Length
        last_write_time_utc = $file.LastWriteTimeUtc.ToString('o')
    }
}

$outputPath = Join-Path $PSScriptRoot 'vampire_survivors_executable_metadata.json'

$result |
    ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $outputPath -Encoding UTF8

Write-Host "取得完了しました。"
Write-Host "結果: $outputPath"
Write-Host ""
$result | ConvertTo-Json -Depth 5
