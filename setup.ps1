# ===================================================================
#  3D AI Zoomquilt Engine - bulletproof setup (stock Windows -> ready)
#  Ensures Python 3.11, Git and ffmpeg (via winget), then builds the
#  Python 3.11 venv with CUDA torch + the app's dependencies.
# ===================================================================
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Refresh-Path {
    $m = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $u = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = ($m, $u -join ";")
}
function Have($cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}
function Py311-OK {
    try { & py -3.11 --version *> $null; return ($LASTEXITCODE -eq 0) }
    catch { return $false }
}
function Winget-Install($id, $name) {
    Write-Host ">> Installing $name ..." -ForegroundColor Cyan
    winget install -e --id $id --accept-source-agreements --accept-package-agreements --disable-interactivity
    Refresh-Path
}

Write-Host "============================================================"
Write-Host "  3D AI Zoomquilt Engine - setup"
Write-Host "============================================================"

if (-not (Have "winget")) {
    Write-Host "winget (Windows Package Manager) was not found." -ForegroundColor Yellow
    Write-Host "Install 'App Installer' from the Microsoft Store, then re-run setup.bat."
    Write-Host "(Or install Python 3.11, Git and ffmpeg manually.)"
    exit 1
}

# --- Python 3.11 (torch has no wheels for 3.13/3.14) ---
if (Py311-OK) { Write-Host "Python 3.11 found." -ForegroundColor Green }
else {
    Winget-Install "Python.Python.3.11" "Python 3.11"
    if (-not (Py311-OK)) {
        Write-Host "Python 3.11 still not visible in this session." -ForegroundColor Yellow
        Write-Host "Close this window and run setup.bat again to finish." -ForegroundColor Yellow
        exit 1
    }
}

# --- Git (to clone image-gen backends) ---
if (Have "git") { Write-Host "Git found." -ForegroundColor Green }
else { Winget-Install "Git.Git" "Git" }

# --- ffmpeg (to stitch the mp4) ---
if (Have "ffmpeg") { Write-Host "ffmpeg found." -ForegroundColor Green }
else { Winget-Install "Gyan.FFmpeg" "ffmpeg" }

# --- venv + deps ---
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host ">> Creating Python 3.11 virtual environment ..." -ForegroundColor Cyan
    & py -3.11 -m venv .venv
}
$vpy = ".\.venv\Scripts\python.exe"
Write-Host ">> Upgrading pip ..." -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip
Write-Host ">> Installing CUDA torch (cu121) - large download ..." -ForegroundColor Cyan
& $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "CUDA torch failed; falling back to CPU wheel (passes 2/3 will be slow)." -ForegroundColor Yellow
    & $vpy -m pip install torch
}
Write-Host ">> Installing app requirements ..." -ForegroundColor Cyan
& $vpy -m pip install -r requirements.txt

Write-Host "============================================================"
Write-Host "  Verifying torch / CUDA ..." -ForegroundColor Cyan
& $vpy -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
Write-Host ""
Write-Host "Setup complete. Run  launch.bat  to start." -ForegroundColor Green
