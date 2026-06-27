@echo off
cd /d "%~dp0"

echo ==========================================
echo   GVInput HID Mouse Test - Launcher
echo   Tests relative mouse movement via HID
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
echo Starting gvinput_wrapper.py ...
echo.
python gvinput_wrapper.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Process exited with code: %errorlevel%
    pause
)

pause
