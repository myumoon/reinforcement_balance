@echo off
setlocal

set SCRIPT_DIR=%~dp0

rem pyenv-win は python コマンドをシムで管理する (.python-version を参照)
rem py -3.11 ではなく python を使う
python --version 2>nul | findstr /C:"Python 3.11" >nul
if errorlevel 1 (
    echo [ERROR] Python 3.11 が見つかりません。
    echo.
    echo pyenv-win を使用している場合:
    echo   pyenv install 3.11.9
    echo   ^(その後このスクリプトを再実行してください^)
    echo.
    echo python.org からインストールした場合:
    echo   インストール時に "Add python.exe to PATH" を有効にしてください
    exit /b 1
)

python -m venv "%SCRIPT_DIR%.venv"
call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r "%SCRIPT_DIR%requirements.txt"

echo.
echo Setup complete. 次回以降は以下で有効化してください:
echo   %SCRIPT_DIR%.venv\Scripts\activate.bat

endlocal
