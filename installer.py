#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
installer.py  -  Backend installer & launcher for the 3D AI Zoomquilt Engine.

A small Tkinter GUI that installs and launches your choice of high-performance
image-generation backends. The Zoomquilt app talks to a local Stable Diffusion
WebUI over the AUTOMATIC1111-compatible REST API (`/sdapi/v1/...`) on port 7860,
so the "plug-and-play" backends below work with it directly; the others are
included because they're excellent generators (a small adapter is noted where
the API differs).

  - Pick a backend, click Install -> it git-clones into  backends/<name>/  and
    writes a  start-api.bat  configured to serve the API on 127.0.0.1:7860.
  - First launch self-installs that backend's own dependencies (can take a
    while - it's downloading torch etc.). Subsequent launches are fast.
  - Recommended *fast* models for scenery are listed; drop them in the backend's
    models folder for few-step (Turbo / Lightning / Hyper-SD) generation.

Run:  install.bat     (or  python installer.py )
"""

import os
import sys
import queue
import shutil
import threading
import subprocess

try:
    import requests
except ImportError:
    requests = None

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

HERE = os.path.dirname(os.path.abspath(__file__))
BACKENDS_DIR = os.path.join(HERE, "backends")
API_PORT = 7860

# ---------------------------------------------------------------------------
#  Backend registry
# ---------------------------------------------------------------------------
# type drives how start-api.bat is generated:
#   a1111  -> self-installing WebUI; we set COMMANDLINE_ARGS=--api and call webui.bat
#   sdnext -> webui.bat --api
#   fooocus-> run.bat --listen (limited API)
#   comfy  -> venv + torch + requirements, main.py --listen (native API)
BACKENDS = [
    {
        "key": "Forge",
        "title": "Stable Diffusion WebUI Forge",
        "desc": "Fast, low-VRAM A1111 fork. Flux + SDXL + SD1.5. "
                "Recommended - plug-and-play with Zoomquilt.",
        "url": "https://github.com/lllyasviel/stable-diffusion-webui-forge",
        "type": "a1111", "plug": True,
    },
    {
        "key": "A1111",
        "title": "AUTOMATIC1111 WebUI",
        "desc": "The classic. Broadest extension ecosystem. "
                "Plug-and-play (/sdapi/v1 API).",
        "url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui",
        "type": "a1111", "plug": True,
    },
    {
        "key": "SD.Next",
        "title": "SD.Next (vladmandic)",
        "desc": "Performance-focused, many model backends. "
                "A1111-compatible API - plug-and-play.",
        "url": "https://github.com/vladmandic/sdnext",
        "type": "sdnext", "plug": True,
    },
    {
        "key": "Fooocus",
        "title": "Fooocus",
        "desc": "Dead-simple SDXL with excellent defaults. "
                "Great standalone; limited API.",
        "url": "https://github.com/lllyasviel/Fooocus",
        "type": "fooocus", "plug": False,
    },
    {
        "key": "ComfyUI",
        "title": "ComfyUI",
        "desc": "Most flexible + fastest. Native Flux / SD3.5. "
                "Native API (needs a small adapter for Zoomquilt).",
        "url": "https://github.com/comfyanonymous/ComfyUI",
        "type": "comfy", "plug": False,
    },
]

# Downloadable fast models. `file` is the preferred filename; if it's missing
# from the repo at download time we fall back to the largest .safetensors.
# kind -> subfolder under <backend>/models/ (Stable-diffusion or Lora).
FAST_MODELS = [
    {"key": "SDXL-Turbo (standalone checkpoint)",
     "repo": "stabilityai/sdxl-turbo",
     "file": "sd_xl_turbo_1.0_fp16.safetensors",
     "kind": "checkpoint", "note": "1-2 step, works alone, ~6.9 GB"},
    {"key": "SD-Turbo (standalone, smaller)",
     "repo": "stabilityai/sd-turbo",
     "file": "sd_turbo.safetensors",
     "kind": "checkpoint", "note": "SD2.1 turbo, great at 512-640, ~2.5 GB"},
    {"key": "SDXL-Lightning 4-step (LoRA)",
     "repo": "ByteDance/SDXL-Lightning",
     "file": "sdxl_lightning_4step_lora.safetensors",
     "kind": "lora", "note": "add to any SDXL base, 4 step, ~390 MB"},
    {"key": "Hyper-SD SDXL 8-step (LoRA)",
     "repo": "ByteDance/Hyper-SD",
     "file": "Hyper-SDXL-8steps-lora.safetensors",
     "kind": "lora", "note": "add to any SDXL base, best quality/step, ~700 MB"},
    {"key": "LCM-LoRA SD1.5",
     "repo": "latent-consistency/lcm-lora-sdv1-5",
     "file": "pytorch_lora_weights.safetensors",
     "kind": "lora", "note": "add to any SD1.5 base, ~135 MB"},
]
HF_BASE = "https://huggingface.co"


def resolve_model_file(repo, preferred):
    """Confirm `preferred` exists in the repo; else pick the largest
    .safetensors. Returns a filename or None."""
    try:
        r = requests.get(f"https://huggingface.co/api/models/{repo}",
                         timeout=15)
        r.raise_for_status()
        sibs = r.json().get("siblings", [])
        names = [s.get("rfilename", "") for s in sibs]
        safes = [n for n in names if n.endswith(".safetensors")]
        if preferred in names:
            return preferred
        if not safes:
            return None
        # heuristics: prefer a name containing the preferred stem, else first
        stem = os.path.splitext(preferred)[0].lower()
        for n in safes:
            if stem in n.lower():
                return n
        return safes[0]
    except Exception:
        return preferred   # try the preferred name directly


def _start_api_bat(b):
    """Return the contents of the start-api.bat for a given backend dict."""
    t = b["type"]
    if t == "a1111":
        # Self-installing WebUI. Force a torch-compatible Python (3.10/3.11):
        # these WebUIs pin old torch that has NO wheels for 3.13/3.14, so a
        # default 3.14 venv fails on "Couldn't install torch".
        return (
            "@echo off\r\n"
            "cd /d \"%~dp0\"\r\n"
            "set \"PYTHON=\"\r\n"
            "call :findpy 3.11\r\n"
            "call :findpy 3.10\r\n"
            "call :findpy 3.12\r\n"
            "if not defined PYTHON echo WARNING: no Python 3.10-3.12 found - "
            "Forge needs one (3.13/3.14 lack torch wheels).\r\n"
            "echo Starting {title} with API on 127.0.0.1:{port}  "
            "(Python=%PYTHON%)\r\n"
            "echo (first run downloads dependencies - this can take a while)\r\n"
            "set COMMANDLINE_ARGS=--api --api-log --port {port}\r\n"
            "call webui.bat\r\n"
            "pause\r\n"
            "exit /b\r\n"
            ":findpy\r\n"
            "if defined PYTHON exit /b\r\n"
            "py -%~1 --version >nul 2>&1 || exit /b\r\n"
            "for /f \"delims=\" %%p in ('py -%~1 -c \"import sys;print(sys.executable)\"') do set \"PYTHON=%%p\"\r\n"
            "exit /b\r\n"
        ).format(title=b["title"], port=API_PORT)
    if t == "sdnext":
        return (
            "@echo off\r\n"
            "cd /d \"%~dp0\"\r\n"
            "echo Starting SD.Next with API on 127.0.0.1:{port} ...\r\n"
            "call webui.bat --api --port {port}\r\n"
            "pause\r\n"
        ).format(port=API_PORT)
    if t == "fooocus":
        return (
            "@echo off\r\n"
            "cd /d \"%~dp0\"\r\n"
            "echo Starting Fooocus (API is limited) ...\r\n"
            "call run.bat --listen\r\n"
            "pause\r\n"
        )
    if t == "comfy":
        return (
            "@echo off\r\n"
            "cd /d \"%~dp0\"\r\n"
            "echo Starting ComfyUI on 127.0.0.1:{port} ...\r\n"
            "if not exist venv\\Scripts\\python.exe (\r\n"
            "  echo Creating venv + installing torch/deps (one time) ...\r\n"
            "  py -3.11 -m venv venv || python -m venv venv\r\n"
            "  venv\\Scripts\\python.exe -m pip install --upgrade pip\r\n"
            "  venv\\Scripts\\python.exe -m pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121\r\n"
            "  venv\\Scripts\\python.exe -m pip install -r requirements.txt\r\n"
            ")\r\n"
            "venv\\Scripts\\python.exe main.py --listen 127.0.0.1 --port {port}\r\n"
            "pause\r\n"
        ).format(port=API_PORT)
    return "@echo off\r\necho Unknown backend type.\r\npause\r\n"


class InstallerApp:
    def __init__(self, root):
        self.root = root
        root.title("Zoomquilt - Image-Gen Backend Installer")
        root.geometry("820x620")
        root.minsize(720, 560)

        self.events = queue.Queue()
        self.worker = None
        self.choice = tk.StringVar(value=BACKENDS[0]["key"])

        self._build_ui()
        self.root.after(100, self._drain)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Choose an image-generation backend to install",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(main,
                  text="The Zoomquilt app connects over the A1111 API on "
                       f"port {API_PORT}. ⭐ = plug-and-play with Zoomquilt.",
                  foreground="gray").pack(anchor="w", pady=(0, 8))

        # Backend cards
        cards = ttk.Frame(main)
        cards.pack(fill="x")
        for b in BACKENDS:
            row = ttk.Frame(cards)
            row.pack(fill="x", pady=2)
            badge = " ⭐" if b["plug"] else ""
            ttk.Radiobutton(row, value=b["key"], variable=self.choice,
                            text=f"{b['title']}{badge}").pack(side="left")
            ttk.Label(row, text="  " + b["desc"], foreground="#555").pack(
                side="left")

        # Action buttons
        act = ttk.Frame(main)
        act.pack(fill="x", pady=(10, 4))
        self.install_btn = ttk.Button(act, text="Install selected",
                                      command=self.install)
        self.install_btn.pack(side="left")
        self.launch_btn = ttk.Button(act, text="Launch (start API)",
                                     command=self.launch)
        self.launch_btn.pack(side="left", padx=(6, 0))
        ttk.Button(act, text="Open backends folder",
                   command=self.open_folder).pack(side="left", padx=(6, 0))
        self.status = ttk.Label(act, text="", foreground="gray")
        self.status.pack(side="right")

        # Downloadable fast models (into the selected backend's models folder)
        rec = ttk.LabelFrame(main, text="FAST models - select (Ctrl/Shift) then "
                                        "Download into the chosen backend",
                             padding=6)
        rec.pack(fill="x", pady=(6, 6))
        mrow = ttk.Frame(rec)
        mrow.pack(fill="x")
        self.models_list = tk.Listbox(mrow, selectmode="extended", height=5,
                                      exportselection=False)
        for mdl in FAST_MODELS:
            self.models_list.insert("end", f"{mdl['key']}  ({mdl['note']})")
        self.models_list.pack(side="left", fill="x", expand=True)
        self.dl_btn = ttk.Button(mrow, text="Download selected",
                                 command=self.download_models)
        self.dl_btn.pack(side="left", padx=(6, 0), anchor="n")
        ttk.Label(rec, text="Checkpoints work standalone; LoRAs need a base "
                            "model (the Turbo checkpoints double as bases).",
                  foreground="#777", font=("Segoe UI", 8)).pack(anchor="w",
                                                                pady=(2, 0))

        # Log
        self.log = scrolledtext.ScrolledText(main, height=12, wrap="word",
                                             font=("Consolas", 8))
        self.log.pack(fill="both", expand=True)
        self._log("Ready. Requires Git (git --version). "
                  "First launch of a backend installs its own dependencies.")

    # ----- helpers ---------------------------------------------------------
    def _log(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def _selected(self):
        return next(b for b in BACKENDS if b["key"] == self.choice.get())

    def _dest(self, b):
        return os.path.join(BACKENDS_DIR, b["key"])

    # ----- install ---------------------------------------------------------
    def install(self):
        if self.worker and self.worker.is_alive():
            return
        if shutil.which("git") is None:
            messagebox.showerror("Git missing",
                                 "Git is required. Install from git-scm.com.")
            return
        b = self._selected()
        dest = self._dest(b)
        # A real clone is detected by .git - NOT just a non-empty folder, since
        # downloading models first creates dest/models/ before the clone.
        if os.path.isdir(os.path.join(dest, ".git")):
            if not messagebox.askyesno(
                    "Already installed",
                    f"{b['title']} already cloned at:\n{dest}\n\n"
                    "Re-write the start-api.bat only (keep the clone)?"):
                return
            self._finish_setup(b, dest)
            return

        self.install_btn.config(state="disabled")
        self.status.config(text="cloning...")
        self._log(f"\n=== Installing {b['title']} ===")
        self.worker = threading.Thread(target=self._clone, args=(b, dest),
                                       daemon=True)
        self.worker.start()

    def _clone(self, b, dest):
        os.makedirs(BACKENDS_DIR, exist_ok=True)
        # If dest already has content (e.g. pre-downloaded models), git clone
        # would refuse it. Clone into a temp dir, then merge into dest so the
        # downloads are preserved.
        merge = os.path.isdir(dest) and os.listdir(dest)
        target = (dest + "__cloning") if merge else dest
        if merge and os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        try:
            proc = subprocess.Popen(
                ["git", "clone", "--progress", b["url"], target],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:
                self.events.put(("log", line.rstrip()))
            proc.wait()
            if proc.returncode != 0:
                self.events.put(("log", "ERROR: git clone failed."))
                self.events.put(("done", None))
                return
            if merge:
                self.events.put(("log", "  merging clone with existing files "
                                        "(keeping downloaded models)..."))
                self._merge_into(target, dest)
                shutil.rmtree(target, ignore_errors=True)
        except Exception as e:
            self.events.put(("log", f"ERROR: {e}"))
            self.events.put(("done", None))
            return
        self.events.put(("setup", (b, dest)))

    @staticmethod
    def _merge_into(src, dst):
        """Move everything from src into dst; existing files in dst win."""
        for root, _dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            target_dir = os.path.join(dst, rel) if rel != "." else dst
            os.makedirs(target_dir, exist_ok=True)
            for fn in files:
                d = os.path.join(target_dir, fn)
                if not os.path.exists(d):
                    shutil.move(os.path.join(root, fn), d)

    def _finish_setup(self, b, dest):
        """Write start-api.bat (runs on the Tk thread)."""
        bat = os.path.join(dest, "start-api.bat")
        try:
            with open(bat, "w", encoding="utf-8") as f:
                f.write(_start_api_bat(b))
            self._log(f"Wrote launcher: {bat}")
            self._log(f"DONE. Launch '{b['title']}' to serve the API on "
                      f"127.0.0.1:{API_PORT}, then Generate in Zoomquilt.")
            if not b["plug"]:
                self._log("NOTE: this backend's API isn't A1111-compatible "
                          "out of the box - use it standalone, or add an "
                          "adapter to feed Zoomquilt.")
            self.status.config(text="installed")
        except Exception as e:
            self._log(f"ERROR writing launcher: {e}")
        self.install_btn.config(state="normal")

    # ----- launch ----------------------------------------------------------
    def launch(self):
        b = self._selected()
        bat = os.path.join(self._dest(b), "start-api.bat")
        if not os.path.isfile(bat):
            messagebox.showinfo("Not installed",
                                f"Install {b['title']} first.")
            return
        try:
            os.startfile(bat)            # opens in its own console window
            self._log(f"Launched {b['title']} ({bat}). "
                      "Wait for 'Running on local URL' in its window.")
        except Exception as e:
            messagebox.showerror("Launch failed", str(e))

    def open_folder(self):
        os.makedirs(BACKENDS_DIR, exist_ok=True)
        try:
            os.startfile(BACKENDS_DIR)
        except Exception:
            pass

    # ----- model downloads -------------------------------------------------
    def download_models(self):
        if requests is None:
            messagebox.showerror("Missing", "`requests` not installed.")
            return
        sel = list(self.models_list.curselection())
        if not sel:
            messagebox.showinfo("Pick models", "Select one or more models.")
            return
        b = self._selected()
        dest = self._dest(b)
        models = [FAST_MODELS[i] for i in sel]
        self.dl_btn.config(state="disabled")
        self.status.config(text="downloading...")
        self._log(f"\n=== Downloading {len(models)} model(s) into "
                  f"{b['title']} ===")
        threading.Thread(target=self._download_worker,
                         args=(dest, models), daemon=True).start()

    def _download_worker(self, backend_dir, models):
        for mdl in models:
            try:
                self._download_one(backend_dir, mdl)
            except Exception as e:
                self.events.put(("log", f"  {mdl['key']}: FAILED - {e}"))
        self.events.put(("log", "Model downloads finished."))
        self.events.put(("dldone", None))

    def _download_one(self, backend_dir, mdl):
        sub = "Lora" if mdl["kind"] == "lora" else "Stable-diffusion"
        folder = os.path.join(backend_dir, "models", sub)
        os.makedirs(folder, exist_ok=True)
        self.events.put(("log", f"  {mdl['key']}: resolving in {mdl['repo']}..."))
        fname = resolve_model_file(mdl["repo"], mdl["file"]) or mdl["file"]
        out = os.path.join(folder, os.path.basename(fname))
        if os.path.isfile(out) and os.path.getsize(out) > 1_000_000:
            self.events.put(("log", f"  {mdl['key']}: already present, skip."))
            return
        url = f"{HF_BASE}/{mdl['repo']}/resolve/main/{fname}?download=true"
        self.events.put(("log", f"  downloading {fname} -> models/{sub}/"))
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = last = 0
        tmp = out + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total and done - last > total * 0.1:
                    last = done
                    self.events.put(("log",
                        f"    {mdl['key']}: {done / 1e9:.2f}/"
                        f"{total / 1e9:.2f} GB ({done * 100 // total}%)"))
        os.replace(tmp, out)
        self.events.put(("log", f"  {mdl['key']}: done -> {out}"))

    # ----- event pump ------------------------------------------------------
    def _drain(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "setup":
                    self._finish_setup(*payload)
                elif kind == "done":
                    self.install_btn.config(state="normal")
                    self.status.config(text="")
                elif kind == "dldone":
                    self.dl_btn.config(state="normal")
                    self.status.config(text="models ready")
        except queue.Empty:
            pass
        self.root.after(100, self._drain)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    InstallerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
