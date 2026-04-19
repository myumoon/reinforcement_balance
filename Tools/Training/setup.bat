@echo off
setlocal

set ENV_NAME=reinbalance
set SCRIPT_DIR=%~dp0

conda --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda が見つかりません。Anaconda をインストールしてください。
    echo https://www.anaconda.com/download
    exit /b 1
)

conda env list 2>nul | findstr /C:"%ENV_NAME%" >nul
if not errorlevel 1 (
    echo [INFO] conda環境 '%ENV_NAME%' は既に存在します。
    echo パッケージを更新する場合は以下を実行してください:
    echo   conda activate %ENV_NAME%
    echo   pip install -r %SCRIPT_DIR%requirements.txt
    exit /b 0
)

echo [INFO] conda環境 '%ENV_NAME%' を作成しています...
conda create -n %ENV_NAME% python=3.11 -y
if errorlevel 1 (
    echo [ERROR] conda環境の作成に失敗しました。
    exit /b 1
)

echo.
echo Setup complete.
echo 以下のコマンドでパッケージをインストールしてください:
echo   conda activate %ENV_NAME%
echo   pip install -r %SCRIPT_DIR%requirements.txt
echo.
echo 次回以降の有効化:
echo   conda activate %ENV_NAME%

endlocal
