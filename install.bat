@echo off
REM 3D AI Zoomquilt Engine - backend installer & launcher
cd /d "%~dp0"
echo Opening the image-gen backend installer...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" installer.py
) else (
    python installer.py
)
if errorlevel 1 pause
