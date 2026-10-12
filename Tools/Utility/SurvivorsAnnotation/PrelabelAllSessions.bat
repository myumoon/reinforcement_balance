@echo off
rem Generate drafts for all sessions through PowerShell.
rem Pass startup arguments to the matching PowerShell script.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0PrelabelAllSessions.ps1" %*
set "exitCode=%ERRORLEVEL%"
echo.
if not "%exitCode%"=="0" echo FAILED: exit code %exitCode%
pause
exit /b %exitCode%
