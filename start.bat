@echo off
REM ===================================================================
REM  3D AI Zoomquilt Engine - ONE-CLICK START
REM  First run: installs Python 3.11 / Git / ffmpeg (if missing) and
REM  builds the environment. Then opens the hub where you install /
REM  pick an image-gen backend + model and launch everything.
REM ===================================================================
cd /d "%~dp0"

REM --- first run: build the environment ---
if not exist ".venv\Scripts\python.exe" (
    echo First run - setting up. This installs prerequisites and may take a while.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 (
        echo.
        echo Setup did not finish. If it just installed Python/Git/ffmpeg,
        echo close this window and run start.bat again to complete setup.
        pause
        exit /b 1
    )
)

REM --- open the hub (install backend, download model, launch backend + app) ---
if exist ".venv\Scripts\pythonw.exe" (
    ".venv\Scripts\pythonw.exe" installer.py
) else (
    python installer.py
)
