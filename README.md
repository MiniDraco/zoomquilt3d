# 3D AI Zoomquilt Engine

A local, single-file desktop app that automates an **infinite-zoom ("zoomquilt") video pipeline** on top of a local Stable Diffusion WebUI, then lifts the flat frames into a **3D parallax fly-through** and stitches an MP4 — all on your own GPU.

Tuned for an **RTX 3060 12GB**, but works on any CUDA card.

## What it does

A four-pass pipeline, all local (the only network call is to your WebUI):

| Pass | Stage | Output |
|------|-------|--------|
| **1** | 2D outpaint loop — each frame shrinks the last and the model paints a new border | `output/frames` |
| **2** | Depth estimation (Depth-Anything-V2) | `output/depths` |
| **3** | 3D camera warp (`grid_sample` radial parallax, depth-detrended to stay centered) | `output/3d_warps` |
| **4** | Zoom-tween interpolation + upscale + ffmpeg stitch | `renders/zoomquilt_<timestamp>.mp4` |

Highlights:
- **Dynamic timeline** of prompt keyframes with backward-fill, a 🎲 dice that pulls from a 5,000-prompt landscape bank, and a settable keyframe interval.
- **Small-gen → HQ-upscale**: generate at a small fast resolution, upscale to 720p/1080p/1440p/4K with correct aspect (Cover/Fit) via ffmpeg lanczos, or a neural upscaler (R-ESRGAN via the WebUI).
- **Smooth slow-motion** via geometric frame interpolation (no extra generation), sub-pixel-accurate at the zoom center.
- **Zoom In / Zoom Out** direction, FPS, Speed, and a live "final resolution / length" readout.
- **Fast-model controls** — pick the checkpoint, steps, CFG, sampler from the app; one-click ⚡ Fast preset for Turbo/Lightning/Hyper-SD/LCM models.
- **Backend health poll** — auto-detects when your WebUI is online and auto-loads its styles/upscalers.
- **Swappable prompt banks** — a `prompts/` folder of themed sets (landscapes, noir, space, trippy, creepy) in a dropdown; the 🎲 dice / Roll All draw from the chosen one. Drop in your own `.txt` and it appears automatically.
- **Theme generator** — type a theme ("+ New theme…" in the app, or `make-theme.bat`) and it writes a fresh prompt bank using your local Ollama (offline fallback included).
- **Batch / Infinite** — Bulk-N runs, ∞ Infinite, ⚄ Randomize-prompts-each-run, and a cooldown between runs, for an endless gallery of unique zoomquilts.
- **Dark themed UI** — collapsible, resizable panels (black / purple / orange).
- Multi-GPU aware (auto-pins the highest-VRAM card for the ML passes).

## Requirements

- **Python 3.11** (torch wheels; 3.13/3.14 are not yet supported by torch)
- A CUDA GPU
- **ffmpeg** on PATH (for the MP4 stitch)
- A local **Stable Diffusion WebUI** with the API enabled (see the installer below)

## Quick start

**Double-click `start.bat`.** That's it.

On the first run it installs Python 3.11 / Git / ffmpeg if they're missing (via `winget`) and builds the environment, then opens the hub. In the hub:

1. **Install** an image-gen backend (pick one of 5)
2. **Download** a fast model into it
3. **🚀 Launch backend + app** — pick your model and it brings both up

In the app: it auto-detects when the backend is online → set keyframe prompts → **Generate Zoomquilt**. The finished video lands in `renders/`.

> Power-user shortcuts: `launch.bat` runs just the app (when a backend's already up); `make-theme.bat` generates a new themed prompt bank.

### Backends

The hub git-clones and configures your choice of:

| Backend | Plug-and-play* | Notes |
|---------|:---:|-------|
| **SD WebUI Forge** | ⭐ | Fast, low-VRAM, Flux/SDXL/SD1.5 — recommended |
| **AUTOMATIC1111** | ⭐ | The classic, broadest ecosystem |
| **SD.Next** | ⭐ | Performance-focused |
| **Fooocus** | | Easiest SDXL; limited API |
| **ComfyUI** | | Most flexible/fastest; native API (needs an adapter for this app) |

\* Plug-and-play = exposes the AUTOMATIC1111 `/sdapi/v1/` API the app speaks, on port 7860.

For fast scenery generation, drop a few-step model (SDXL-Lightning, Hyper-SD, SD/SDXL-Turbo, or FLUX.1-schnell) into the backend's models folder and use the ⚡ Fast preset.

## Files

| File | Purpose |
|------|---------|
| **`start.bat`** | **One-click: first-run setup + the hub (install backend, model, launch)** |
| `zoomquilt3d.py` | The app (single file) |
| `installer.py` | The hub GUI (opened by start.bat) |
| `setup.ps1` | First-run bootstrap (winget Python 3.11 / Git / ffmpeg + venv) |
| `prompts/` | Prompt banks (`*.txt`, one prompt per line) shown in the dropdown |
| `build_prompt_bank.py` | Regenerates `prompts/landscapes.txt` (5,000 prompts) |
| `build_theme_banks.py` | Regenerates the noir/space/trippy/creepy banks |
| `theme_gen.py` / `make-theme.bat` | Generate a new themed bank from a typed theme (Ollama) |
| `launch.bat` | Run just the app (when a backend is already up) |

## License

MIT — see [LICENSE](LICENSE).

---
*Built iteratively with [Claude Code](https://claude.com/claude-code).*
