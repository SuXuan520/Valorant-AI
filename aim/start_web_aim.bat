@echo off
cd /d "%~dp0"

echo ==========================================
echo   Aim Web Panel - Launcher
echo   URL: http://127.0.0.1:5000
echo ==========================================
echo.

REM Activate conda environment "aimenv"
call C:\Users\ShadowCrane\Anaconda3\Scripts\activate.bat aimenv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda env 'aimenv'.
    pause
    exit /b 1
)

python --version
echo.
echo Starting web_aim.py ...
echo.
python web_aim.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Process exited with code: %errorlevel%
    pause
)
