@echo off
REM 3D AI Zoomquilt Engine launcher
cd /d "%~dp0"
echo Starting 3D AI Zoomquilt Engine...
echo (Make sure your Stable Diffusion WebUI is running with --api)

REM Prefer the project venv (has CUDA torch); fall back to system python.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" zoomquilt3d.py
) else (
    echo NOTE: .venv not found - running with system Python.
    echo       Passes 2/3 need torch; run setup.bat first if they fail.
    python zoomquilt3d.py
)
if errorlevel 1 pause
