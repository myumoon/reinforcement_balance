# 目的: アノテーション用 Python CLI を安全に実行する。
# stderr の出力にかかわらず終了コードで成否を判定し、呼び出し元の状態を戻す。

# 目的: Python CLI を終了コードで判定して実行する。
# native stderr で処理を中断せず、作業場所と ErrorActionPreference を元に戻す。
function Invoke-AnnotationPython {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$PythonPath,
        [Parameter(Mandatory)]
        [string[]]$ArgumentList,
        [Parameter(Mandatory)]
        [string]$RepositoryRoot
    )

    Push-Location -LiteralPath $RepositoryRoot
    try {
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $PythonPath @ArgumentList
            $pythonExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
    }
    finally {
        Pop-Location
    }

    if ($pythonExitCode -ne 0) {
        throw "Configured Python command failed with exit code $pythonExitCode."
    }
}
