@echo off
REM ===================================================================
REM  3D AI Zoomquilt Engine - one-click setup (works from stock Windows)
REM  Installs Python 3.11 / Git / ffmpeg if missing (via winget), then
REM  builds the Python 3.11 venv with CUDA torch + dependencies.
REM ===================================================================
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
echo.
pause
