@echo off
REM ===================================================================
REM  3D AI Zoomquilt Engine - one-time environment setup
REM  Builds a Python 3.11 venv with CUDA torch + transformers, because
REM  the system Python (3.14) has no torch wheels yet.
REM ===================================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo   3D AI Zoomquilt Engine - environment setup
echo ============================================================

REM --- Find a compatible Python (prefer 3.11, then 3.12, then 3.10) ---
set "PYEXE="
for %%V in (3.11 3.12 3.10) do (
    if not defined PYEXE (
        py -%%V --version >nul 2>&1 && set "PYEXE=py -%%V"
    )
)
if not defined PYEXE (
    echo.
    echo ERROR: Could not find Python 3.10, 3.11, or 3.12 via the 'py' launcher.
    echo        torch does not yet publish wheels for Python 3.13/3.14.
    echo        Install Python 3.11 from python.org and re-run setup.bat.
    pause
    exit /b 1
)
echo Using interpreter: %PYEXE%

REM --- Create venv ---
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PYEXE% -m venv .venv
    if errorlevel 1 ( echo Failed to create venv. & pause & exit /b 1 )
) else (
    echo Reusing existing .venv
)

set "VPY=.venv\Scripts\python.exe"

echo Upgrading pip ...
"%VPY%" -m pip install --upgrade pip

echo Installing CUDA torch (cu121) ... this is a large download.
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo.
    echo WARNING: CUDA torch install failed. Retrying with the default CPU wheel.
    echo          Passes 2 and 3 will run on CPU ^(slow^) without CUDA.
    "%VPY%" -m pip install torch
)

echo Installing remaining requirements ...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo Failed installing requirements. & pause & exit /b 1 )

echo.
echo ============================================================
echo   Setup complete. Verifying torch / CUDA ...
echo ============================================================
"%VPY%" -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

echo.
echo Done. Launch the app with:  launch.bat
pause
endlocal
