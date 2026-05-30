@echo off
for /f "delims=" %%i in ('wsl wslpath -u "%~dp0"') do set SCRIPT_DIR=%%i
wsl bash "%SCRIPT_DIR%setup_wsl.sh"
pause
