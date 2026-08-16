@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp000_ConfigureLocalEnvironment.ps1"
set "exitCode=%ERRORLEVEL%"
echo.
if not "%exitCode%"=="0" echo FAILED: exit code %exitCode%
pause
exit /b %exitCode%
