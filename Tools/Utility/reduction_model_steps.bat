@echo off
setlocal

set REDUCTION_STEPS=1000000
set SCRIPT_DIR=%~dp0

if "%~1"=="" (
    echo [ERROR] 対象フォルダを指定してください。
    echo Usage: reduction_model_steps.bat ^<model_steps フォルダのパス^>
    echo Example: reduction_model_steps.bat "..\Training\runs\survivors\v12\train\10_is1_passive_bootstrap_5\work\model_steps"
    exit /b 1
)

python "%SCRIPT_DIR%reduction_model_steps\reduction_model_steps.py" --dir "%~1" --reduction-steps %REDUCTION_STEPS%
