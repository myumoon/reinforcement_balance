@echo off
setlocal

set SCRIPT_DIR=%~dp0

py -3.11 -m venv "%SCRIPT_DIR%.venv"
if errorlevel 1 (
    echo [ERROR] Python 3.11 が見つかりません。pyenv-win または python.org からインストールしてください。
    exit /b 1
)

call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r "%SCRIPT_DIR%requirements.txt"

echo.
echo Setup complete. 次回以降は以下で有効化してください:
echo   %SCRIPT_DIR%.venv\Scripts\activate.bat

endlocal
