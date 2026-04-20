@echo off
chcp 65001 >nul
setlocal

set SCRIPT_DIR=%~dp0

python --version 2>nul | findstr /C:"Python 3.11" >nul
if errorlevel 1 (
    echo Python 3.11 is required. Please install it from https://www.python.org/downloads/release/python-3110/ and ensure it is added to your PATH.
    exit /b 1
)

python -m venv "%SCRIPT_DIR%.venv"
call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r "%SCRIPT_DIR%requirements.txt"

endlocal
