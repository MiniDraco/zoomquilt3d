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

# Recommended fast models for scenery (few-step). Filenames + where they live.
FAST_MODELS = [
    ("SDXL-Lightning 4-step UNet", "ByteDance/SDXL-Lightning",
     "models/Stable-diffusion (or checkpoints) - 4-step, deployable"),
    ("Hyper-SD LoRA (1-8 step)", "ByteDance/Hyper-SD",
     "models/Lora - add to any SD/SDXL checkpoint, best quality/step"),
    ("SD/SDXL-Turbo", "stabilityai/sdxl-turbo",
     "models/Stable-diffusion - 1-2 step, fastest"),
    ("FLUX.1-schnell", "black-forest-labs/FLUX.1-schnell",
     "models/Stable-diffusion - 4-step Flux, top detail"),
]


def _start_api_bat(b):
    """Return the contents of the start-api.bat for a given backend dict."""
    t = b["type"]
    if t == "a1111":
        # Self-installing WebUI: set API flags then hand off to webui.bat.
        return (
            "@echo off\r\n"
            "cd /d \"%~dp0\"\r\n"
            "echo Starting {title} with API on 127.0.0.1:{port} ...\r\n"
            "echo (first run downloads dependencies - this can take a while)\r\n"
            "set COMMANDLINE_ARGS=--api --api-log --port {port}\r\n"
            "call webui.bat\r\n"
            "pause\r\n"
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

        # Recommended models
        rec = ttk.LabelFrame(main, text="Recommended FAST models for scenery "
                                        "(few-step = faster gen)", padding=6)
        rec.pack(fill="x", pady=(6, 6))
        for name, repo, where in FAST_MODELS:
            ttk.Label(rec, text=f"• {name}  -  huggingface.co/{repo}",
                      font=("Segoe UI", 9)).pack(anchor="w")
            ttk.Label(rec, text=f"     {where}", foreground="gray",
                      font=("Segoe UI", 8)).pack(anchor="w")

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
        if os.path.isdir(dest) and os.listdir(dest):
            if not messagebox.askyesno(
                    "Already exists",
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
        try:
            proc = subprocess.Popen(
                ["git", "clone", "--progress", b["url"], dest],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:
                self.events.put(("log", line.rstrip()))
            proc.wait()
            if proc.returncode != 0:
                self.events.put(("log", "ERROR: git clone failed."))
                self.events.put(("done", None))
                return
        except Exception as e:
            self.events.put(("log", f"ERROR: {e}"))
            self.events.put(("done", None))
            return
        self.events.put(("setup", (b, dest)))

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
