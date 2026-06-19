@echo off
REM Generate a themed prompt bank for the Zoomquilt app.
cd /d "%~dp0"
echo === Zoomquilt theme prompt-bank generator ===
echo Uses your local Ollama if running, otherwise an offline generator.
echo.
set "THEME=%*"
if "%THEME%"=="" set /p THEME=Enter a theme (e.g. cyberpunk megacity):
if "%THEME%"=="" (
    echo No theme entered.
    pause
    exit /b 1
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" theme_gen.py %THEME%
) else (
    python theme_gen.py %THEME%
)
echo.
pause
