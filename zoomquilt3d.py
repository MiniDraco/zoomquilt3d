#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D AI Zoomquilt Engine  -  local, single-file, RTX 3060 12GB tuned.

A Tkinter desktop app that automates an infinite-zoom ("zoomquilt") video
pipeline on top of a local Stable Diffusion WebUI (AUTOMATIC1111 / Forge)
instance, then lifts the flat frames into a 3D parallax fly-through.

PIPELINE
  Pass 1  (CPU + WebUI GPU)  2D outpaint loop ............ -> /frames
  Pass 2  (local GPU)        Depth-Anything-V2 depth maps . -> /depths
  Pass 3  (local GPU)        grid_sample 3D camera warp ... -> /3d_warps

Everything runs locally. The only network call is to the WebUI API URL.

REQUIREMENTS
  pip install requests pillow numpy torch transformers
  # torch with CUDA, e.g.:
  # pip install torch --index-url https://download.pytorch.org/whl/cu121

  A running Stable Diffusion WebUI launched with the API enabled:
      webui.bat --api
  (Default endpoint: http://127.0.0.1:7860)

The heavy ML deps (torch / transformers) are imported lazily so the GUI
still opens and Pass 1 still runs even if they are not installed yet.
"""

import os
import io
import sys
import time
import json
import base64
import queue
import threading
import traceback
from datetime import datetime

# ---- Hard GUI / imaging deps (fail loudly if missing) ----------------------
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageTk
except ImportError:  # pragma: no cover
    print("FATAL: Pillow is required.  pip install pillow", file=sys.stderr)
    raise


# ===========================================================================
#  Configuration constants
# ===========================================================================
GEN_RES = 640                 # default gen size (square). Small = fast gen;
                              # the upscale pass takes it up to the target.
GEN_SIZES = [384, 512, 640, 768, 896, 1024]
# Output target presets: label -> (width, height) or None for "source size".
OUTPUT_PRESETS = {
    "Source (square)": None,
    "720p  (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080),
    "1440p (2560x1440)": (2560, 1440),
    "4K    (3840x2160)": (3840, 2160),
    "Square 2048": (2048, 2048),
}
DEFAULT_API = "http://127.0.0.1:7860"
DEFAULT_ZOOM = 0.7
DEFAULT_FRAMES = 60
FRAMES_PAD = 30               # auto-mode: frames of "play" added per keyframe
SD_STEPS = 20
SD_DENOISE = 0.75             # seed-frame denoise (frame 0 uses 1.0)
OUTPAINT_DENOISE = 0.82       # border denoise - high enough to fill, low
                              # enough to follow the scene's colors
MASK_FEATHER = 48             # gaussian blur radius for seamless mask blend
SEAM_OVERLAP = 40             # px of the pasted edge also repainted (blend ring)
EMPTY_STD_THRESHOLD = 7.0     # border std-dev below this => "empty/flat" frame
MAX_FRAME_RETRIES = 2         # re-roll a frame this many times if it comes back empty
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
WARP_STRENGTH = 0.10          # how aggressively foreground expands in Pass 3
                              # (lower = gentler parallax, less orbit/wobble)

PREVIEW_MAX = 460             # preview panel size in px
HEALTH_INTERVAL_MS = 8000     # backend liveness ping period

OUT_FRAMES = "frames"
OUT_DEPTHS = "depths"
OUT_WARPS = "3d_warps"
OUT_TWEENS = "tweens"
TWEEN_DEFAULT = 4             # in-between frames synthesized per real pair


def select_best_gpu():
    """
    On multi-GPU boxes, torch's default cuda:0 may be the *weakest* card.
    Query nvidia-smi and pin CUDA_VISIBLE_DEVICES to the GPU with the most
    VRAM *before* torch is imported, so the depth model lands on the big card
    (e.g. the RTX 3060 12GB rather than a 6GB GTX 1660S).

    Returns a human-readable note, or None if no selection was made / possible.
    Respects a CUDA_VISIBLE_DEVICES the user set themselves.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        gpus = []
        for line in out.stdout.strip().splitlines():
            idx, name, mem = (p.strip() for p in line.split(",", 2))
            gpus.append((int(idx), name, int(mem)))
        if len(gpus) < 2:
            return None  # single GPU - nothing to choose
        best = max(gpus, key=lambda g: g[2])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best[0])
        return f"Pinned GPU {best[0]} ({best[1]}, {best[2]} MiB) for ML passes."
    except Exception:
        return None


# ===========================================================================
#  Pipeline backend  (no Tk references - pushes events onto a queue)
# ===========================================================================
class PipelineCancelled(Exception):
    """Raised internally to unwind the worker thread on user Stop."""


def format_eta(seconds):
    """Human ETA: '8s', '1m12s', '1h03m'."""
    if seconds is None or seconds < 0:
        return "--"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class EtaTracker:
    """
    Rolling ETA from the duration of the last `window` completed items.
    Recent timings track real throughput better than a cumulative average
    (the depth model load, VRAM warmup, etc. skew early frames).
    """

    def __init__(self, window=5):
        self.window = window
        self.durations = []
        self._last = None

    def start(self):
        self._last = time.time()

    def tick(self):
        now = time.time()
        if self._last is not None:
            self.durations.append(now - self._last)
            if len(self.durations) > self.window:
                self.durations.pop(0)
        self._last = now

    def eta(self, remaining):
        if not self.durations or remaining <= 0:
            return None
        avg = sum(self.durations) / len(self.durations)
        return avg * remaining


class ZoomquiltPipeline:
    """
    Headless orchestrator for the three passes.  Communicates with the GUI
    exclusively through a thread-safe event queue so Tkinter is only ever
    touched from the main thread.

    Event dicts pushed onto self.events:
        {"type": "log",      "msg": str}
        {"type": "progress", "value": float(0..100), "label": str}
        {"type": "preview",  "image": PIL.Image}
        {"type": "done",     "ok": bool, "msg": str}
    """

    def __init__(self, config, events, cancel_flag):
        self.cfg = config
        self.events = events
        self.cancel = cancel_flag      # threading.Event
        self.out_dir = config["out_dir"]
        self.gen_res = int(config.get("gen_res", GEN_RES))

    # ----- event helpers ---------------------------------------------------
    def log(self, msg):
        self.events.put({"type": "log", "msg": msg})

    def progress(self, value, label=""):
        self.events.put({"type": "progress", "value": value, "label": label})

    def preview(self, image):
        # Always hand off a copy - the worker keeps mutating its own buffers.
        try:
            self.events.put({"type": "preview", "image": image.copy()})
        except Exception:
            pass

    def _check_cancel(self):
        if self.cancel.is_set():
            raise PipelineCancelled()

    def _prepare_dir(self, path):
        """
        Create `path` and remove any stale *.png from a previous run, so each
        pass starts clean. Mixed old/new files cause count mismatches between
        passes (missing depths -> skipped warps -> broken stitch). Locked files
        that can't be deleted are skipped (a same-named new file overwrites).
        """
        os.makedirs(path, exist_ok=True)
        removed = 0
        for fn in os.listdir(path):
            if fn.lower().endswith(".png"):
                try:
                    os.remove(os.path.join(path, fn))
                    removed += 1
                except OSError:
                    pass
        if removed:
            self.log(f"  cleared {removed} stale file(s) from "
                     f"{os.path.basename(path)}/")

    # ----- prompt resolution ----------------------------------------------
    @staticmethod
    def resolve_prompt(keyframes, frame_index):
        """
        Automatic fallback resolver: if frame_index has no explicit keyframe,
        scan backward to the closest lower keyframe and reuse its prompt.
        `keyframes` is a dict {int frame: str prompt}.
        """
        if frame_index in keyframes:
            return keyframes[frame_index]
        candidates = [f for f in keyframes if f <= frame_index]
        if not candidates:
            # No lower keyframe (shouldn't happen - frame 0 is enforced).
            return keyframes[min(keyframes)]
        return keyframes[max(candidates)]

    # ======================================================================
    #  Mask / canvas math
    # ======================================================================
    def _build_outpaint_inputs(self, prev_image):
        """
        Given the previous full-res frame, produce (init_image, mask, offset,
        shrunk_size) for an img2img outpaint step.

          * shrink prev by zoom factor (Lanczos)
          * background = a BLURRED full-canvas upscale of the previous frame,
            so the border carries the scene's own colors instead of flat gray.
            This is what kills the "empty gray ring" frames and gives the model
            real context to extend outward.
          * paste the sharp shrunk frame dead-center on that background
          * mask: WHITE(255) outer border = repaint -> BLACK(0) protected center,
            but the protected box is INSET by SEAM_OVERLAP so a ring around the
            pasted edge is also repainted -> the model dissolves the hard seam.
          * heavy gaussian feather gives a gradient (not a step) so denoise
            ramps smoothly from full at the border to zero at the center.
        """
        g = self.gen_res
        zoom = self.cfg["zoom"]
        shrunk_size = max(8, int(round(g * zoom)))
        offset = (g - shrunk_size) // 2

        shrunk = prev_image.resize((shrunk_size, shrunk_size), Image.LANCZOS)

        # Scene-colored background (no gray): upscale prev to full frame and
        # blur it hard so the border is plausible, low-detail continuation.
        bg = prev_image.resize((g, g), Image.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(g // 12))
        canvas = bg.copy()
        canvas.paste(shrunk, (offset, offset))

        # Mask with an inset protected box (overlap ring) + feather.
        mask = Image.new("L", (g, g), 255)
        draw = ImageDraw.Draw(mask)
        overlap = min(SEAM_OVERLAP, shrunk_size // 3)
        inner = [offset + overlap, offset + overlap,
                 offset + shrunk_size - overlap, offset + shrunk_size - overlap]
        if inner[2] > inner[0] and inner[3] > inner[1]:
            draw.rectangle(inner, fill=0)
        mask = mask.filter(ImageFilter.GaussianBlur(MASK_FEATHER))

        return canvas, mask, offset, shrunk_size

    def _border_is_empty(self, image, offset, inner_size):
        """
        Detect a failed outpaint: if the repainted border ring has almost no
        texture (near-flat), the model left it gray/empty. Returns True so the
        caller can re-roll the frame.
        """
        if np is None:
            return False
        arr = np.asarray(image.convert("L"), dtype=np.float32)
        h, w = arr.shape
        keep = np.ones((h, w), dtype=bool)
        y0 = max(0, offset)
        x0 = max(0, offset)
        keep[y0:y0 + inner_size, x0:x0 + inner_size] = False
        border = arr[keep]
        if border.size == 0:
            return False
        return float(border.std()) < EMPTY_STD_THRESHOLD

    def _load_start_image(self, path, g):
        """
        Load an optional user-supplied seed image and fit it to a g x g square
        (center-crop 'cover' so the zoom subject stays centered). Returns a PIL
        image, or None if no path / unreadable.
        """
        if not path or not os.path.isfile(path):
            return None
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            self.log(f"  WARN: could not open start image ({e}); generating "
                     "frame 0 instead.")
            return None
        w, h = img.size
        scale = g / min(w, h)                       # cover
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
        nw, nh = img.size
        left, top = (nw - g) // 2, (nh - g) // 2
        return img.crop((left, top, left + g, top + g))

    @staticmethod
    def _b64_png(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _decode_b64_image(b64_str):
        raw = base64.b64decode(b64_str.split(",", 1)[-1])
        return Image.open(io.BytesIO(raw)).convert("RGB")

    # ======================================================================
    #  PASS 1 - 2D outpaint loop
    # ======================================================================
    def pass1_outpaint(self):
        if requests is None:
            raise RuntimeError("`requests` not installed - cannot reach WebUI API.")

        api = self.cfg["api"].rstrip("/")
        url = f"{api}/sdapi/v1/img2img"
        total = self.cfg["total_frames"]
        keyframes = self.cfg["keyframes"]
        frames_dir = os.path.join(self.out_dir, OUT_FRAMES)
        self._prepare_dir(frames_dir)

        styles = self.cfg.get("styles", [])
        if styles:
            self.log(f"PASS 1  -  applying styles: {', '.join(styles)}")
        g = self.gen_res
        self.log(f"PASS 1  -  2D outpaint  ({total} frames @ {g}px)")
        prev_image = None
        eta = EtaTracker()
        eta.start()

        start_img = self._load_start_image(self.cfg.get("start_image", ""), g)
        if start_img is not None:
            self.log("  using supplied start image as the seed frame (0)")
        self.log(f"  sampling: {self.cfg.get('steps', SD_STEPS)} steps, "
                 f"cfg {self.cfg.get('cfg_scale', 7)}, "
                 f"sampler {self.cfg.get('sampler', 'Euler a')}"
                 + (f", model {self.cfg.get('model')}"
                    if self.cfg.get("model") else ""))

        for i in range(total):
            self._check_cancel()
            prompt = self.resolve_prompt(keyframes, i)
            negative = self.cfg["negative"]

            # Frame 0 from a user image: skip SD entirely, just seed the chain.
            if i == 0 and start_img is not None:
                start_img.save(os.path.join(frames_dir, "frame_0000.png"))
                prev_image = start_img
                eta.tick()
                self.preview(start_img)
                self.progress(1 / total * 100.0, f"Pass 1: seed image 1/{total}")
                continue

            if i == 0:
                # Seed frame: blank canvas + solid white mask -> full generation.
                init = Image.new("RGB", (g, g), (128, 128, 128))
                mask = Image.new("L", (g, g), 255)
                denoise = 1.0
                off, ssize = 0, g
            else:
                init, mask, off, ssize = self._build_outpaint_inputs(prev_image)
                denoise = OUTPAINT_DENOISE

            result = None
            for attempt in range(MAX_FRAME_RETRIES + 1):
                self._check_cancel()
                payload = {
                    "init_images": [self._b64_png(init)],
                    "mask": self._b64_png(mask),
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "styles": styles,              # Forge/A1111 prompt styles
                    "steps": int(self.cfg.get("steps", SD_STEPS)),
                    "denoising_strength": denoise,
                    "cfg_scale": float(self.cfg.get("cfg_scale", 7)),
                    "width": g,
                    "height": g,
                    "sampler_name": self.cfg.get("sampler", "Euler a"),
                    # Mask is already feathered in PIL; keep API blur small so
                    # the soft edges don't compound and bleed the center.
                    "mask_blur": 4,
                    # 1 = original: the border starts from the scene-colored
                    # blurred background, so denoise extends the existing scene
                    # (seamless) instead of inventing unrelated content. The
                    # blurred bg means this never leaves flat gray.
                    "inpainting_fill": 1,
                    "inpaint_full_res": False,     # use whole image as context
                    "inpaint_full_res_padding": 32,
                    "inpainting_mask_invert": 0,   # white = repaint
                    "resize_mode": 0,
                    "seed": -1,                    # random each attempt
                }
                # Optional: switch the Forge checkpoint per request (e.g. to a
                # Turbo/Lightning model) without touching the WebUI manually.
                model = self.cfg.get("model", "")
                if model:
                    payload["override_settings"] = {
                        "sd_model_checkpoint": model}
                    payload["override_settings_restore_afterwards"] = False

                tag = f" (retry {attempt})" if attempt else ""
                self.log(f"  frame {i:03d}{tag}  denoise={denoise:.2f}  "
                         f"prompt='{prompt[:44]}'")
                try:
                    resp = requests.post(url, json=payload, timeout=600)
                    resp.raise_for_status()
                except requests.exceptions.ConnectionError:
                    raise RuntimeError(
                        f"Could not connect to WebUI at {api}. "
                        "Is it running with --api ?")
                except requests.exceptions.Timeout:
                    raise RuntimeError(f"WebUI timed out generating frame {i}.")

                data = resp.json()
                if not data.get("images"):
                    raise RuntimeError(f"WebUI returned no image for frame {i}: "
                                       f"{json.dumps(data)[:200]}")

                result = self._decode_b64_image(data["images"][0])
                if result.size != (g, g):
                    result = result.resize((g, g), Image.LANCZOS)

                # Empty-frame guard (only meaningful for outpaint frames).
                if i == 0 or not self._border_is_empty(result, off, ssize):
                    break
                self.log(f"    -> border looks empty/flat, re-rolling frame {i}")

            out_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            result.save(out_path)
            prev_image = result

            eta.tick()
            self.preview(result)
            self.progress((i + 1) / total * 100.0,
                          f"Pass 1: outpaint {i + 1}/{total}  -  "
                          f"ETA {format_eta(eta.eta(total - (i + 1)))}")

        self.log(f"PASS 1 complete -> {frames_dir}")

    # ======================================================================
    #  PASS 2 - depth estimation (Depth-Anything-V2)
    # ======================================================================
    def pass2_depth(self):
        # Lazy import - GUI/Pass1 should not require torch to be installed.
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except ImportError as e:
            raise RuntimeError(
                "Pass 2 needs torch + transformers.  "
                "pip install torch transformers  (" + str(e) + ")")

        # Free VRAM held by anything before loading the depth model.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = 0 if torch.cuda.is_available() else -1
        self.log(f"PASS 2  -  loading {DEPTH_MODEL} "
                 f"({'cuda:0' if device == 0 else 'cpu'}) ...")

        try:
            depth_pipe = hf_pipeline(
                task="depth-estimation",
                model=DEPTH_MODEL,
                device=device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load depth model: {e}")

        frames_dir = os.path.join(self.out_dir, OUT_FRAMES)
        depths_dir = os.path.join(self.out_dir, OUT_DEPTHS)
        self._prepare_dir(depths_dir)

        frame_files = sorted(
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(".png"))
        if not frame_files:
            raise RuntimeError("No frames found for depth estimation.")

        total = len(frame_files)
        self.log(f"PASS 2  -  {total} frames to process")
        eta = EtaTracker()
        eta.start()
        for idx, fname in enumerate(frame_files):
            self._check_cancel()
            img = Image.open(os.path.join(frames_dir, fname)).convert("RGB")
            out = depth_pipe(img)
            depth = out["depth"]                       # PIL grayscale 'L'

            # Normalize to full 0..255 so warp uses the entire range.
            depth_np = np.asarray(depth).astype(np.float32)
            dmin, dmax = depth_np.min(), depth_np.max()
            if dmax > dmin:
                depth_np = (depth_np - dmin) / (dmax - dmin) * 255.0
            depth_img = Image.fromarray(depth_np.astype(np.uint8), mode="L")

            base = os.path.splitext(fname)[0]
            depth_img.save(os.path.join(depths_dir, f"{base}_depth.png"))

            eta.tick()
            self.preview(depth_img.convert("RGB"))
            self.progress((idx + 1) / total * 100.0,
                          f"Pass 2: depth {idx + 1}/{total}  -  "
                          f"ETA {format_eta(eta.eta(total - (idx + 1)))}")

        del depth_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        written = len([f for f in os.listdir(depths_dir)
                       if f.lower().endswith(".png")])
        if written != total:
            self.log(f"  WARN: wrote {written} depth maps for {total} frames "
                     "- some frames may have been lost mid-run.")
        self.log(f"PASS 2 complete ({written} depth maps) -> {depths_dir}")

    # ======================================================================
    #  PASS 3 - 3D camera warp (grid_sample parallax)
    # ======================================================================
    def pass3_warp(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            raise RuntimeError("Pass 3 needs torch.  pip install torch  (" + str(e) + ")")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"PASS 3  -  3D camera warp on {device}")

        frames_dir = os.path.join(self.out_dir, OUT_FRAMES)
        depths_dir = os.path.join(self.out_dir, OUT_DEPTHS)
        warps_dir = os.path.join(self.out_dir, OUT_WARPS)
        self._prepare_dir(warps_dir)

        frame_files = sorted(
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(".png"))
        if not frame_files:
            raise RuntimeError("No frames found for warping.")

        total = len(frame_files)

        # Build the static normalized coordinate grid once: range [-1, 1].
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.gen_res, device=device),
            torch.linspace(-1.0, 1.0, self.gen_res, device=device),
            indexing="ij",
        )
        base_grid = torch.stack((xs, ys), dim=-1)        # (H, W, 2) -> (x, y)

        strength = float(self.cfg.get("warp_strength", WARP_STRENGTH))
        detrend = bool(self.cfg.get("warp_detrend", True))
        if strength <= 0:
            self.log("  3D depth off (strength 0) - frames copied unwarped.")
        # Precompute centered coord planes for depth detrending.
        g = self.gen_res
        yy, xx = np.mgrid[0:g, 0:g].astype(np.float32)
        plane_x = xx / g - 0.5
        plane_y = yy / g - 0.5
        eta = EtaTracker()
        eta.start()

        for idx, fname in enumerate(frame_files):
            self._check_cancel()
            base = os.path.splitext(fname)[0]
            depth_path = os.path.join(depths_dir, f"{base}_depth.png")
            if not os.path.exists(depth_path):
                self.log(f"  WARN: missing depth for {fname}, skipping warp.")
                continue

            # --- frame tensor: (1, 3, H, W) in [-1, 1] ---
            img = Image.open(os.path.join(frames_dir, fname)).convert("RGB")
            img_np = np.asarray(img).astype(np.float32) / 127.5 - 1.0
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

            # --- depth tensor: (H, W) in [0, 1], bright = foreground/near ---
            depth = Image.open(depth_path).convert("L")
            depth_np = np.asarray(depth).astype(np.float32) / 255.0
            if detrend:
                # Remove the best-fit tilt plane: Depth-Anything biases the
                # bottom "near"/top "far", which makes the warp pull off-center
                # and the zoom appear to ORBIT. Subtracting the linear gradient
                # keeps the centered near/far bowl (the real 3D) but kills the
                # directional bias. plane_x/plane_y are zero-mean so the fit
                # coefficients decouple.
                a = float((plane_x * depth_np).sum() /
                          ((plane_x * plane_x).sum() + 1e-6))
                b = float((plane_y * depth_np).sum() /
                          ((plane_y * plane_y).sum() + 1e-6))
                depth_np = depth_np - (a * plane_x + b * plane_y)
                lo, hi = depth_np.min(), depth_np.max()
                if hi > lo:
                    depth_np = (depth_np - lo) / (hi - lo)
            depth_t = torch.from_numpy(depth_np).to(device)

            # Radial parallax: contract the sampling grid toward the center so
            # content moves OUTWARD on screen (camera dollies forward). Bright
            # (near) pixels contract more than dark (far) pixels -> they race
            # toward the borders faster, producing depth parallax.
            scale = 1.0 - strength * depth_t            # (H, W)
            sample_grid = base_grid * scale.unsqueeze(-1)   # (H, W, 2)
            sample_grid = sample_grid.unsqueeze(0)          # (1, H, W, 2)

            warped = F.grid_sample(
                img_t, sample_grid,
                mode="bilinear", padding_mode="reflection",
                align_corners=True,
            )

            out_np = ((warped.squeeze(0).permute(1, 2, 0).clamp(-1, 1)
                       .cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            out_img = Image.fromarray(out_np, mode="RGB")
            out_img.save(os.path.join(warps_dir, f"{base}_warp.png"))

            eta.tick()
            self.preview(out_img)
            self.progress((idx + 1) / total * 100.0,
                          f"Pass 3: warp {idx + 1}/{total}  -  "
                          f"ETA {format_eta(eta.eta(total - (idx + 1)))}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.log(f"PASS 3 complete -> {warps_dir}")

    # ======================================================================
    #  Zoom tweening - synthesize smooth in-between frames (no SD, ~free)
    # ======================================================================
    def _zoom_in_center(self, img, m):
        """Magnify `img` about its center by factor m>=1 (crop center, upscale)."""
        g = self.gen_res
        if m <= 1.0001:
            return img.copy()
        crop = max(1, int(round(g / m)))
        left = (g - crop) // 2
        box = (left, left, left + crop, left + crop)
        return img.crop(box).resize((g, g), Image.LANCZOS)

    def _tween_frame(self, outer, inner, t, z):
        """
        One in-between frame at phase t in (0,1) between a wider frame `outer`
        and `inner` (== the central z-fraction of `outer`). The whole frame
        zooms in continuously while the sharp `inner` image crossfades up from
        the center - registered so the inner content sits exactly where it is
        baked into `outer`. Pure resampling: no model, milliseconds per frame.
        """
        g = self.gen_res
        c = (g - 1) / 2.0                   # exact pixel center
        m = z ** (-t)                       # screen magnification 1 -> 1/z

        # Zoom `outer` in by m about the EXACT center via a sub-pixel affine
        # (output->input coords). Integer crop+resize rounded the zoom center
        # by +-0.5px each frame, so the convergence point shimmered/ghosted.
        inv = 1.0 / m
        base = outer.transform(
            (g, g), Image.AFFINE,
            (inv, 0.0, c * (1.0 - inv), 0.0, inv, c * (1.0 - inv)),
            resample=Image.BICUBIC)
        if m > 1.02:
            pct = int(min(160, (m - 1.0) * 340))
            base = base.filter(ImageFilter.UnsharpMask(
                radius=2, percent=pct, threshold=2))

        # Place `inner` at scale s, ALSO sub-pixel centered with the same affine
        # convention, so it registers exactly onto the inner content already
        # baked into `base` -> no double-image (blur) and no shimmer (jitter).
        s = z ** (1.0 - t)                  # inner footprint fraction (z -> 1)
        invs = 1.0 / s
        inner_layer = inner.transform(
            (g, g), Image.AFFINE,
            (invs, 0.0, c * (1.0 - invs), 0.0, invs, c * (1.0 - invs)),
            resample=Image.BICUBIC)

        sN = s * g
        feather = max(2.0, sN * 0.05)
        half = max(1.0, sN / 2.0 - feather)   # keep blurred mask on valid inner
        mask = Image.new("L", (g, g), 0)
        ImageDraw.Draw(mask).rectangle(
            [c - half, c - half, c + half, c + half], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
        alpha = int(round(t * 255))        # temporal crossfade weight
        mask = mask.point(lambda p: p * alpha // 255)
        return Image.composite(inner_layer, base, mask)

    def _make_tweens(self, ordered_paths, count):
        """
        Expand an ordered list of frame paths (outer -> inner, i.e. zoom-IN
        order) by inserting `count` tween frames between each adjacent pair.
        Returns the full expanded list of absolute paths (real + tween).
        """
        if count <= 0 or len(ordered_paths) < 2:
            return list(ordered_paths)

        tween_dir = os.path.join(self.out_dir, OUT_TWEENS)
        self._prepare_dir(tween_dir)
        z = float(self.cfg.get("zoom", DEFAULT_ZOOM))
        pairs = len(ordered_paths) - 1
        expanded = []
        counter = 0
        eta = EtaTracker()
        eta.start()

        outer = Image.open(ordered_paths[0]).convert("RGB")
        for i in range(pairs):
            self._check_cancel()
            inner = Image.open(ordered_paths[i + 1]).convert("RGB")
            expanded.append(ordered_paths[i])
            for k in range(1, count + 1):
                t = k / (count + 1)
                tw = self._tween_frame(outer, inner, t, z)
                p = os.path.join(tween_dir, f"tween_{counter:06d}.png")
                tw.save(p)
                expanded.append(p)
                counter += 1
            outer = inner
            eta.tick()
            self.preview(self._zoom_in_center(inner, 1.0))
            self.progress((i + 1) / pairs * 100.0,
                          f"Tweening pair {i + 1}/{pairs}  -  "
                          f"ETA {format_eta(eta.eta(pairs - (i + 1)))}")
        expanded.append(ordered_paths[-1])   # final inner frame
        self.log(f"  tweened {pairs} pairs x{count} -> "
                 f"{len(expanded)} total frames")
        return expanded

    # ======================================================================
    #  Upscale helpers
    # ======================================================================
    @staticmethod
    def _scale_filter(target, fit):
        """
        ffmpeg -vf string to upscale square frames to `target` (w, h) keeping
        aspect via cover (fill+center-crop) or fit (letterbox-pad). Lanczos
        scaler. Returns None if target is None (keep source size).
        """
        if not target:
            return None
        w, h = target
        if fit == "fit":   # contain: scale down to fit, pad the rest
            return (f"scale={w}:{h}:force_original_aspect_ratio=decrease:"
                    f"flags=lanczos,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
        # cover: scale up to fill, crop the overflow (default)
        return (f"scale={w}:{h}:force_original_aspect_ratio=increase:"
                f"flags=lanczos,crop={w}:{h},setsar=1")

    def _neural_upscale(self, seq, factor, upscaler):
        """
        Enlarge each frame in `seq` via the WebUI's extras upscaler (e.g.
        R-ESRGAN) for real detail, writing results to output/upscaled and
        returning the new path list. Falls back to the original `seq` on any
        error (ffmpeg's lanczos scale still runs afterward).
        """
        if requests is None:
            self.log("  neural upscale skipped - `requests` missing.")
            return seq
        api = self.cfg["api"].rstrip("/")
        url = f"{api}/sdapi/v1/extra-single-image"
        up_dir = os.path.join(self.out_dir, "upscaled")
        self._prepare_dir(up_dir)
        self.log(f"PASS 4b  -  neural upscale x{factor} via '{upscaler}' "
                 f"({len(seq)} frames)")
        out_paths = []
        eta = EtaTracker()
        eta.start()
        for idx, p in enumerate(seq):
            self._check_cancel()
            try:
                with open(p, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("utf-8")
                payload = {
                    "image": b64,
                    "upscaler_1": upscaler,
                    "upscaling_resize": factor,
                    "resize_mode": 0,
                }
                resp = requests.post(url, json=payload, timeout=300)
                resp.raise_for_status()
                img = self._decode_b64_image(resp.json()["image"])
                op = os.path.join(up_dir, f"up_{idx:06d}.png")
                img.save(op)
                out_paths.append(op)
            except Exception as e:
                self.log(f"  neural upscale failed on frame {idx}: {e} "
                         "- falling back to lanczos for the rest.")
                return seq
            eta.tick()
            self.progress((idx + 1) / len(seq) * 100.0,
                          f"Upscaling {idx + 1}/{len(seq)}  -  "
                          f"ETA {format_eta(eta.eta(len(seq) - (idx + 1)))}")
        return out_paths

    # ======================================================================
    #  PASS 4 - stitch warped frames into an MP4 (optional, ffmpeg)
    # ======================================================================
    def pass4_stitch(self):
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.log("PASS 4 skipped - ffmpeg not found on PATH.")
            return

        warps_dir = os.path.join(self.out_dir, OUT_WARPS)
        frame_files = sorted(
            f for f in os.listdir(warps_dir)
            if f.lower().endswith(".png"))
        if not frame_files:
            self.log("PASS 4 skipped - no warped frames to stitch.")
            return

        fps = self.cfg.get("fps", 30)
        tween = int(self.cfg.get("tween", 0))
        speed = float(self.cfg.get("speed", 1.0)) or 1.0

        # Slow-down strategy: INTERPOLATE, don't duplicate. Holding each frame
        # 2x for speed 0.5 only shows 15fps of unique content -> stepped/jittery
        # motion on a fast zoom. Instead, fold the slowdown into the tween
        # density so every output frame is a unique in-between (true `fps`,
        # smooth). Bonus: more tweens per pair = smaller zoom step each = LESS
        # blur pulse too. Speed > 1 (faster) subsamples the final sequence.
        subsample = 1
        if speed < 1.0:
            eff_tween = max(tween, int(round((tween + 1) / speed)) - 1)
        else:
            eff_tween = tween
            if speed > 1.0:
                subsample = max(1, int(round(speed)))
        if eff_tween != tween:
            self.log(f"  speed {speed:g} -> tween {tween} raised to {eff_tween} "
                     "(smooth interpolation, not frame-holding)")

        # Build the zoom-IN order (outer -> inner): frames are generated
        # innermost-first, so the widest frame is last. In this order each
        # adjacent pair is (outer, its central z-region) - exactly what the
        # tween math expects.
        zoomin_paths = [os.path.join(warps_dir, fn)
                        for fn in reversed(frame_files)]

        if eff_tween > 0:
            self.log(f"PASS 4a  -  tweening x{eff_tween} (smoothing, no SD)")
            seq = self._make_tweens(zoomin_paths, eff_tween)
        else:
            seq = zoomin_paths
        if subsample > 1:
            seq = seq[::subsample]

        # Direction just chooses playback order of the (already tweened) seq.
        zoom_in = self.cfg.get("direction", "in") == "in"
        if not zoom_in:                       # zoom OUT = play inner -> outer
            seq = list(reversed(seq))

        # Resolve the output target + optional neural upscale.
        target = OUTPUT_PRESETS.get(self.cfg.get("out_res"), None)
        upscaler = self.cfg.get("upscaler", "")
        if upscaler and upscaler.lower() not in ("", "lanczos", "none"):
            # enlarge enough to cover the target's long edge with detail
            long_edge = max(target) if target else self.gen_res
            factor = max(1, min(4, -(-long_edge // self.gen_res)))  # ceil, cap 4
            seq = self._neural_upscale(seq, factor, upscaler)

        vf = self._scale_filter(target, self.cfg.get("fit", "cover"))
        # Save each render with a timestamped name in a persistent 'renders'
        # dir (sibling of output/) that is NEVER cleared between runs, so a
        # new generation can't purge previous videos.
        render_dir = os.path.join(os.path.dirname(self.out_dir), "renders")
        os.makedirs(render_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_mp4 = os.path.join(render_dir, f"zoomquilt_{stamp}.mp4")
        res_note = (f"{target[0]}x{target[1]} {self.cfg.get('fit','cover')}"
                    if target else f"{self.gen_res}px source")

        # Every frame in `seq` is unique; play them all at full `fps` (the
        # slowdown is already baked into the frame count via interpolation),
        # so there's no frame duplication and no stepped/jittery motion.
        length_s = len(seq) / float(fps)
        self.log(f"PASS 4  -  stitching {len(seq)} frames @ {fps}fps "
                 f"-> {res_note}, ~{length_s:.1f}s "
                 f"({'zoom IN' if zoom_in else 'zoom OUT'})")

        # Concat list, uniform 1/fps per frame -> exact CFR, all unique frames.
        list_path = os.path.join(self.out_dir, "_concat_list.txt")
        dur = 1.0 / float(fps)
        lines = ["ffconcat version 1.0"]
        for p in seq:
            lines.append(f"file '{p.replace(chr(92), '/')}'")
            lines.append(f"duration {dur:.6f}")
        # Repeat the last frame so its duration isn't dropped by the demuxer.
        lines.append(f"file '{seq[-1].replace(chr(92), '/')}'")
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-r", str(fps),
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            out_mp4,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=1800)
            if proc.returncode != 0:
                self.log("ffmpeg error:\n" + proc.stderr[-600:])
                return
        except Exception as e:
            self.log(f"PASS 4 failed: {e}")
            return
        finally:
            try:
                os.remove(list_path)
            except OSError:
                pass
        self.last_render = out_mp4
        self.log(f"PASS 4 complete -> {out_mp4}")

    # ======================================================================
    #  Driver
    # ======================================================================
    def run(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            self.log(f"Output directory: {self.out_dir}")
            self.pass1_outpaint()
            gpu_note = select_best_gpu()
            if gpu_note:
                self.log(gpu_note)
            self.pass2_depth()
            self.pass3_warp()
            if self.cfg.get("stitch", True):
                self.pass4_stitch()
            self.progress(100.0, "All passes complete")
            render = getattr(self, "last_render", None)
            done_msg = (f"Zoomquilt saved -> {render}" if render
                        else "Zoomquilt frames complete. See "
                             f"{os.path.join(self.out_dir, OUT_WARPS)}")
            self.events.put({"type": "done", "ok": True, "msg": done_msg})
        except PipelineCancelled:
            self.log("Cancelled by user.")
            self.events.put({"type": "done", "ok": False, "msg": "Cancelled."})
        except Exception as e:
            self.log("ERROR: " + str(e))
            self.log(traceback.format_exc())
            self.events.put({"type": "done", "ok": False, "msg": str(e)})


# ===========================================================================
#  GUI
# ===========================================================================
class KeyframeRow:
    """One timeline layer: frame index, prompt text, delete button."""

    def __init__(self, parent, app, frame_index=0, prompt=""):
        self.app = app
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", pady=2)

        ttk.Label(self.frame, text="Frame:").pack(side="left")
        self.frame_var = tk.StringVar(value=str(frame_index))
        self.frame_entry = ttk.Entry(self.frame, width=5,
                                     textvariable=self.frame_var)
        self.frame_entry.pack(side="left", padx=(2, 6))
        # Editing a frame index re-sizes Total Frames when Auto is on.
        self.frame_var.trace_add("write", lambda *a: self.app._recompute_total())

        self.prompt_entry = ttk.Entry(self.frame, width=42)
        self.prompt_entry.insert(0, prompt)
        self.prompt_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        # Dice: fill this row with a random prompt from the bank.
        # Use a BMP die glyph (U+2684) + a font that has it - the astral
        # emoji die (U+1F3B2) renders as a tofu box in Tk on Windows.
        self.dice_btn = tk.Button(self.frame, text="⚄", width=2,
                                  font=("Segoe UI Symbol", 11),
                                  command=self.roll)
        self.dice_btn.pack(side="left", padx=(0, 4))

        self.del_btn = ttk.Button(self.frame, text="Delete", width=7,
                                  command=self.delete)
        self.del_btn.pack(side="left")

    def roll(self):
        prompt = self.app.random_prompt()
        if prompt:
            self.prompt_entry.delete(0, "end")
            self.prompt_entry.insert(0, prompt)

    def delete(self):
        self.app.remove_row(self)
        self.frame.destroy()

    def get(self):
        """Return (frame_index:int, prompt:str) or raise ValueError."""
        idx = int(self.frame_var.get().strip())
        return idx, self.prompt_entry.get().strip()


class ZoomquiltApp:
    def __init__(self, root):
        self.root = root
        root.title("3D AI Zoomquilt Engine  -  RTX 3060 12GB")
        root.geometry("1180x720")
        root.minsize(1040, 640)

        self.events = queue.Queue()
        self.cancel_flag = threading.Event()
        self.worker = None
        self.rows = []
        self._preview_ref = None      # keep PhotoImage alive
        self._suppress_total = False  # guard auto<->manual feedback loop
        self.prompt_bank = self._load_prompt_bank()

        self._build_ui()
        # Seed with a mandatory Frame 0 row + one downstream keyframe.
        self.add_row(0, "a glowing cosmic nebula, deep space, ultra detailed, "
                        "vibrant colors, cinematic")
        self.add_row(30, "an ancient ornate temple interior, intricate carvings, "
                         "volumetric light")

        self.root.after(100, self._drain_events)
        # Health poll: pings the backend, drives the alive/offline indicator,
        # and auto-loads styles/upscalers when it first comes online.
        self._health_alive = None
        self._poll_running = True
        self.root.after(600, self._health_poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- prompt bank -----------------------------------------------------
    def _load_prompt_bank(self):
        """Load prompts.txt (one prompt per line) sitting next to this script."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "prompts.txt")
        try:
            with open(path, encoding="utf-8") as f:
                bank = [ln.strip() for ln in f if ln.strip()]
            return bank
        except OSError:
            return []

    def random_prompt(self):
        """Return one random prompt from the bank, or None if empty."""
        import random as _random
        if not self.prompt_bank:
            messagebox.showwarning(
                "No prompt bank",
                "prompts.txt not found. Run build_prompt_bank.py to "
                "generate the 5000-prompt landscape bank.")
            return None
        return _random.choice(self.prompt_bank)

    def roll_all(self):
        """Fill every keyframe row with a fresh random prompt."""
        for row in self.rows:
            row.roll()

    def _fast_preset(self):
        """Dial in few-step settings for Turbo/Lightning/Hyper-SD/LCM models."""
        self.steps_var.set(6)
        self.cfg_var.set(2.0)
        # Prefer an LCM sampler if the backend offers one.
        for v in self.sampler_combo.cget("values"):
            if "lcm" in str(v).lower():
                self.sampler_var.set(v)
                break
        self._log("Fast preset: 6 steps, CFG 2.0. Pick a Turbo/Lightning/"
                  "Hyper-SD/LCM model (or add its LoRA) for best results.")

    # ----- start image + final-output readout ------------------------------
    def _browse_start_image(self):
        path = filedialog.askopenfilename(
            title="Choose a starting image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
                       ("All files", "*.*")])
        if path:
            self.start_image_var.set(path)

    def _update_final_label(self):
        """Live readout: final resolution, frame count, and video length."""
        try:
            preset = OUTPUT_PRESETS.get(self.out_res_var.get())
            res = (f"{preset[0]}x{preset[1]}" if preset
                   else f"{int(self.gen_var.get())}²")
            frames = int(self.frames_var.get())
            tween = int(self.tween_var.get())
            fps = max(1, int(self.fps_var.get()))
            speed = float(self.speed_var.get()) or 1.0
            # Slowdown is realized as extra interpolated frames (see pass4).
            sub = 1
            if speed < 1.0:
                eff = max(tween, int(round((tween + 1) / speed)) - 1)
            else:
                eff = tween
                if speed > 1.0:
                    sub = max(1, int(round(speed)))
            seq = (frames + max(0, frames - 1) * eff) // sub
            length = seq / fps
            self.final_lbl.config(
                text=f"→ Final: {res} @ {fps}fps x{speed:g}  ·  "
                     f"{seq} frames  ·  ~{length:.1f}s")
        except (ValueError, tk.TclError):
            pass

    # ----- UI construction -------------------------------------------------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # ---- LEFT: control panel ----
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        # Global controls
        glob = ttk.LabelFrame(left, text="Global Settings", padding=8)
        glob.pack(fill="x", pady=(0, 8))

        api_row = ttk.Frame(glob)
        api_row.pack(fill="x", pady=2)
        ttk.Label(api_row, text="WebUI API URL:").pack(side="left")
        self.api_var = tk.StringVar(value=DEFAULT_API)
        ttk.Entry(api_row, textvariable=self.api_var, width=32).pack(
            side="left", padx=6)
        ttk.Button(api_row, text="Test Connection",
                   command=self.test_connection).pack(side="left")
        self.conn_lbl = ttk.Label(api_row, text="?", foreground="gray")
        self.conn_lbl.pack(side="left", padx=6)

        # Model / sampling row - switch checkpoint + drop steps for fast
        # (Turbo/Lightning/Hyper-SD/LCM) models. Scenery needs few steps.
        ms_row = ttk.Frame(glob)
        ms_row.pack(fill="x", pady=2)
        ttk.Label(ms_row, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value="(current)")
        self.model_combo = ttk.Combobox(ms_row, width=20, state="readonly",
                                        textvariable=self.model_var,
                                        values=["(current)"])
        self.model_combo.pack(side="left", padx=(4, 8))
        ttk.Label(ms_row, text="Steps:").pack(side="left")
        self.steps_var = tk.IntVar(value=SD_STEPS)
        ttk.Spinbox(ms_row, from_=1, to=50, width=4,
                    textvariable=self.steps_var).pack(side="left", padx=(2, 8))
        ttk.Label(ms_row, text="CFG:").pack(side="left")
        self.cfg_var = tk.DoubleVar(value=7.0)
        ttk.Spinbox(ms_row, from_=1.0, to=15.0, increment=0.5, width=4,
                    format="%.1f", textvariable=self.cfg_var).pack(
            side="left", padx=(2, 8))
        ttk.Label(ms_row, text="Sampler:").pack(side="left")
        self.sampler_var = tk.StringVar(value="Euler a")
        self.sampler_combo = ttk.Combobox(ms_row, width=12, state="readonly",
                                         textvariable=self.sampler_var,
                                         values=["Euler a"])
        self.sampler_combo.pack(side="left", padx=(4, 8))
        ttk.Button(ms_row, text="⚡ Fast", width=7,
                   command=self._fast_preset).pack(side="left")

        # Total frames: slider + editable spinbox + auto-grow toggle.
        # Auto mode keeps Total Frames = highest keyframe index + FRAMES_PAD,
        # so each added prompt adds ~30 frames of "play". Manual edits (drag
        # or type) switch off Auto; re-checking Auto recomputes.
        tf_row = ttk.Frame(glob)
        tf_row.pack(fill="x", pady=4)
        ttk.Label(tf_row, text="Total Frames:").pack(side="left")
        self.frames_var = tk.IntVar(value=DEFAULT_FRAMES)
        self.auto_frames = tk.BooleanVar(value=True)
        ttk.Spinbox(tf_row, from_=2, to=3000, width=6,
                    textvariable=self.frames_var).pack(side="right")
        ttk.Checkbutton(tf_row, text="Auto", variable=self.auto_frames,
                        command=self._recompute_total).pack(side="right",
                                                            padx=(6, 4))
        ttk.Scale(tf_row, from_=2, to=600, orient="horizontal",
                  variable=self.frames_var).pack(
            side="left", fill="x", expand=True, padx=6)
        self.frames_var.trace_add("write", self._on_total_changed)

        # Zoom factor: slider + editable spinbox override (type an exact value).
        zf_row = ttk.Frame(glob)
        zf_row.pack(fill="x", pady=4)
        ttk.Label(zf_row, text="Zoom Factor:").pack(side="left")
        self.zoom_var = tk.DoubleVar(value=DEFAULT_ZOOM)
        ttk.Spinbox(zf_row, from_=0.30, to=0.95, increment=0.01, width=6,
                    format="%.2f", textvariable=self.zoom_var).pack(side="right")
        ttk.Scale(zf_row, from_=0.30, to=0.95, orient="horizontal",
                  variable=self.zoom_var,
                  command=lambda v: self.zoom_var.set(round(float(v), 2))).pack(
            side="left", fill="x", expand=True, padx=6)

        # Gen Size: the (small) square resolution frames are generated at.
        # Small = fast SD gen; the upscale pass brings it up to the target.
        gs_row = ttk.Frame(glob)
        gs_row.pack(fill="x", pady=4)
        ttk.Label(gs_row, text="Gen Size:").pack(side="left")
        self.gen_var = tk.IntVar(value=GEN_RES)
        ttk.Combobox(gs_row, width=6, state="readonly",
                     textvariable=self.gen_var,
                     values=[str(s) for s in GEN_SIZES]).pack(side="left",
                                                              padx=6)
        ttk.Label(gs_row, text="px square  (smaller = faster gen)",
                  foreground="gray").pack(side="left")

        # Negative prompt
        neg_row = ttk.Frame(glob)
        neg_row.pack(fill="x", pady=2)
        ttk.Label(neg_row, text="Negative:").pack(side="left")
        self.neg_var = tk.StringVar(
            value="blurry, low quality, watermark, text, border, frame")
        ttk.Entry(neg_row, textvariable=self.neg_var).pack(
            side="left", fill="x", expand=True, padx=6)

        # Output / video row
        vid_row = ttk.Frame(glob)
        vid_row.pack(fill="x", pady=(4, 0))
        self.stitch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(vid_row, text="Stitch MP4 (ffmpeg)",
                        variable=self.stitch_var).pack(side="left")
        ttk.Label(vid_row, text="FPS:").pack(side="left", padx=(12, 2))
        self.fps_var = tk.IntVar(value=30)
        ttk.Spinbox(vid_row, from_=8, to=60, width=5,
                    textvariable=self.fps_var).pack(side="left")
        # Direction: Zoom In (reversed frames) vs Zoom Out (forward).
        ttk.Label(vid_row, text="Direction:").pack(side="left", padx=(12, 2))
        self.direction_var = tk.StringVar(value="Zoom In")
        ttk.Combobox(vid_row, width=9, state="readonly",
                     textvariable=self.direction_var,
                     values=["Zoom In", "Zoom Out"]).pack(side="left")
        # Tween: synthesized in-between frames per real pair (0 = off).
        ttk.Label(vid_row, text="Tween:").pack(side="left", padx=(12, 2))
        self.tween_var = tk.IntVar(value=TWEEN_DEFAULT)
        ttk.Spinbox(vid_row, from_=0, to=16, width=4,
                    textvariable=self.tween_var).pack(side="left")
        # Speed: <1 = slower (holds frames longer, no extra generation).
        ttk.Label(vid_row, text="Speed:").pack(side="left", padx=(12, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(vid_row, from_=0.10, to=3.0, increment=0.05, width=5,
                    format="%.2f", textvariable=self.speed_var).pack(side="left")
        # 3D Depth: parallax warp strength (0 = flat/off, higher = more pop
        # but more orbit/wobble). Detrended to stay centered.
        ttk.Label(vid_row, text="3D Depth:").pack(side="left", padx=(12, 2))
        self.warp_var = tk.DoubleVar(value=WARP_STRENGTH)
        ttk.Spinbox(vid_row, from_=0.0, to=0.30, increment=0.02, width=5,
                    format="%.2f", textvariable=self.warp_var).pack(side="left")

        # Start image (optional): seeds frame 0 instead of generating it.
        si_row = ttk.Frame(glob)
        si_row.pack(fill="x", pady=(4, 0))
        ttk.Label(si_row, text="Start Image:").pack(side="left")
        self.start_image_var = tk.StringVar(value="")
        ttk.Entry(si_row, textvariable=self.start_image_var).pack(
            side="left", fill="x", expand=True, padx=6)
        ttk.Button(si_row, text="Browse", width=7,
                   command=self._browse_start_image).pack(side="left")
        ttk.Button(si_row, text="X", width=2,
                   command=lambda: self.start_image_var.set("")).pack(
            side="left", padx=(4, 0))

        # Output / upscale row: target resolution + aspect fit + upscaler.
        up_row = ttk.Frame(glob)
        up_row.pack(fill="x", pady=(4, 0))
        ttk.Label(up_row, text="Output:").pack(side="left")
        self.out_res_var = tk.StringVar(value="1080p (1920x1080)")
        ttk.Combobox(up_row, width=18, state="readonly",
                     textvariable=self.out_res_var,
                     values=list(OUTPUT_PRESETS.keys())).pack(side="left",
                                                              padx=(4, 8))
        ttk.Label(up_row, text="Fit:").pack(side="left")
        self.fit_var = tk.StringVar(value="Cover (crop)")
        ttk.Combobox(up_row, width=12, state="readonly",
                     textvariable=self.fit_var,
                     values=["Cover (crop)", "Fit (pad)"]).pack(side="left",
                                                                padx=(4, 8))
        ttk.Label(up_row, text="Upscaler:").pack(side="left")
        self.upscaler_var = tk.StringVar(value="Lanczos (fast)")
        self.upscaler_combo = ttk.Combobox(
            up_row, width=20, state="readonly",
            textvariable=self.upscaler_var, values=["Lanczos (fast)"])
        self.upscaler_combo.pack(side="left", padx=(4, 0))

        # Live "final output" readout (resolution / frames / length).
        self.final_lbl = ttk.Label(glob, text="", foreground="#1a7f37",
                                   font=("Segoe UI", 9, "bold"))
        self.final_lbl.pack(anchor="w", pady=(4, 0))
        for v in (self.out_res_var, self.fps_var, self.tween_var,
                  self.frames_var, self.speed_var, self.gen_var):
            v.trace_add("write", lambda *a: self._update_final_label())
        self._update_final_label()

        # ---- Forge Styles (sniffed from the API) ----
        st = ttk.LabelFrame(left, text="Forge Styles", padding=6)
        st.pack(fill="x", pady=(0, 8))

        st_bar = ttk.Frame(st)
        st_bar.pack(fill="x")
        ttk.Button(st_bar, text="Sniff Styles from API",
                   command=self.fetch_styles).pack(side="left")
        ttk.Button(st_bar, text="Clear", width=6,
                   command=lambda: self.styles_list.selection_clear(0, "end")
                   ).pack(side="left", padx=(6, 0))
        self.styles_status = ttk.Label(st_bar, text="not loaded",
                                       foreground="gray")
        self.styles_status.pack(side="right")

        st_body = ttk.Frame(st)
        st_body.pack(fill="x", pady=(4, 0))
        st_scroll = ttk.Scrollbar(st_body, orient="vertical")
        self.styles_list = tk.Listbox(st_body, selectmode="extended", height=5,
                                      exportselection=False,
                                      yscrollcommand=st_scroll.set)
        st_scroll.config(command=self.styles_list.yview)
        self.styles_list.pack(side="left", fill="x", expand=True)
        st_scroll.pack(side="right", fill="y")
        ttk.Label(st, text="Ctrl/Shift-click to pick multiple; applied to "
                           "every frame.", foreground="gray").pack(
            anchor="w", pady=(2, 0))

        # ---- Timeline Layer Manager ----
        tl = ttk.LabelFrame(left, text="Timeline Layer Manager", padding=6)
        tl.pack(fill="both", expand=True, pady=(0, 8))

        tl_bar = ttk.Frame(tl)
        tl_bar.pack(fill="x", pady=(0, 4))
        ttk.Button(tl_bar, text="[ + Add Keyframe ]",
                   command=lambda: self.add_row()).pack(side="left")
        ttk.Button(tl_bar, text="⚄ Roll All",
                   command=self.roll_all).pack(side="left", padx=(6, 0))
        # Interval: frames between auto-placed keyframes (and Auto-total step).
        ttk.Label(tl_bar, text="every").pack(side="left", padx=(10, 2))
        self.interval_var = tk.IntVar(value=FRAMES_PAD)
        ttk.Spinbox(tl_bar, from_=2, to=600, width=5,
                    textvariable=self.interval_var,
                    command=self._recompute_total).pack(side="left")
        ttk.Label(tl_bar, text="frames").pack(side="left", padx=(2, 0))
        ttk.Label(tl_bar,
                  text=f"{len(self.prompt_bank)} prompts in bank").pack(
            side="right")

        # Scrollable region for keyframe rows
        canvas = tk.Canvas(tl, highlightthickness=0, height=200)
        scrollbar = ttk.Scrollbar(tl, orient="vertical", command=canvas.yview)
        self.rows_container = ttk.Frame(canvas)
        self.rows_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.rows_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ---- Action buttons + progress ----
        act = ttk.Frame(left)
        act.pack(fill="x", pady=(0, 6))
        self.gen_btn = ttk.Button(act, text="GENERATE ZOOMQUILT",
                                  command=self.start_generation)
        self.gen_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(act, text="Stop", width=8,
                                   command=self.stop_generation,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))

        self.progress = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.status_lbl = ttk.Label(left, text="Idle.", anchor="w")
        self.status_lbl.pack(fill="x", pady=(2, 4))

        # ---- Log box ----
        self.logbox = scrolledtext.ScrolledText(left, height=8, wrap="word",
                                                font=("Consolas", 8))
        self.logbox.pack(fill="both", expand=False)

        # ---- RIGHT: preview panel ----
        right = ttk.LabelFrame(main, text="Live Preview", padding=8)
        right.pack(side="right", fill="both", padx=(8, 0))
        self.preview_canvas = tk.Canvas(right, width=PREVIEW_MAX,
                                        height=PREVIEW_MAX, bg="#101014",
                                        highlightthickness=1,
                                        highlightbackground="#333")
        self.preview_canvas.pack()
        self.preview_canvas.create_text(
            PREVIEW_MAX // 2, PREVIEW_MAX // 2,
            text="preview", fill="#444", font=("Segoe UI", 14),
            tags="placeholder")

    # ----- row management --------------------------------------------------
    def _row_indices(self):
        """Valid integer frame indices currently in the timeline."""
        out = []
        for r in self.rows:
            try:
                out.append(int(r.frame_var.get()))
            except (ValueError, tk.TclError):
                pass
        return out

    def _interval(self):
        """Settable spacing between auto-placed keyframes (default FRAMES_PAD)."""
        try:
            return max(1, int(self.interval_var.get()))
        except (ValueError, tk.TclError, AttributeError):
            return FRAMES_PAD

    def add_row(self, frame_index=None, prompt=""):
        if frame_index is None:
            existing = self._row_indices()
            frame_index = (max(existing) + self._interval()) if existing else 0
        row = KeyframeRow(self.rows_container, self, frame_index, prompt)
        self.rows.append(row)
        self._recompute_total()

    def remove_row(self, row):
        if row in self.rows:
            self.rows.remove(row)
        self._recompute_total()

    # ----- total-frames auto management ------------------------------------
    def _recompute_total(self):
        """If Auto is on, size Total Frames to the highest keyframe + padding."""
        if not getattr(self, "auto_frames", None) or not self.auto_frames.get():
            return
        idxs = self._row_indices()
        target = (max(idxs) if idxs else 0) + self._interval()
        target = max(2, min(3000, target))
        self._suppress_total = True
        try:
            self.frames_var.set(target)
        finally:
            self._suppress_total = False

    def _on_total_changed(self, *_):
        """A manual edit (slider drag or spinbox typing) cancels Auto."""
        if not self._suppress_total and getattr(self, "auto_frames", None):
            self.auto_frames.set(False)

    # ----- connection test -------------------------------------------------
    def test_connection(self):
        if requests is None:
            self._set_conn(False, "no `requests`")
            return
        api = self.api_var.get().rstrip("/")
        self._set_conn(None, "...")

        def _probe():
            try:
                r = requests.get(f"{api}/sdapi/v1/options", timeout=8)
                ok = r.status_code == 200
                self.root.after(0, lambda: self._set_conn(
                    ok, "connected" if ok else f"HTTP {r.status_code}"))
            except Exception as e:
                err = str(e)[:30]
                self.root.after(0, lambda m=err: self._set_conn(False, m))

        threading.Thread(target=_probe, daemon=True).start()

    def _set_conn(self, ok, text):
        color = {True: "green", False: "red", None: "gray"}[ok]
        self.conn_lbl.config(text=text, foreground=color)

    # ----- backend health poll (alive check + styles auto-load) ------------
    def _health_poll(self):
        """Periodic non-blocking ping; reschedules itself."""
        if not getattr(self, "_poll_running", True):
            return
        if requests is not None:
            api = self.api_var.get().rstrip("/")

            def _probe():
                alive = False
                try:
                    r = requests.get(f"{api}/sdapi/v1/options", timeout=4)
                    alive = (r.status_code == 200)
                except Exception:
                    alive = False
                try:
                    self.root.after(0, lambda: self._on_health(alive))
                except Exception:
                    pass

            threading.Thread(target=_probe, daemon=True).start()
        try:
            self.root.after(HEALTH_INTERVAL_MS, self._health_poll)
        except Exception:
            pass

    def _on_health(self, alive):
        prev = self._health_alive
        self._health_alive = alive
        if alive:
            self.conn_lbl.config(text="● gen online", foreground="green")
            if prev is not True:           # just (re)connected -> load styles
                self._log("Backend online - loading styles/upscalers...")
                self.fetch_styles()
        else:
            self.conn_lbl.config(text="○ gen offline", foreground="red")
            if prev is True:
                self._log("Backend went offline.")

    # ----- styles sniffing -------------------------------------------------
    def fetch_styles(self):
        """Pull the WebUI's saved prompt styles from /sdapi/v1/prompt-styles."""
        if requests is None:
            self.styles_status.config(text="no `requests`", foreground="red")
            return
        api = self.api_var.get().rstrip("/")
        self.styles_status.config(text="sniffing...", foreground="gray")

        def _probe():
            try:
                r = requests.get(f"{api}/sdapi/v1/prompt-styles", timeout=10)
                r.raise_for_status()
                data = r.json()
                names = [s.get("name", "") for s in data if s.get("name")]
                # Drop the WebUI's built-in "None" placeholder style.
                names = [n for n in names if n and n.lower() != "none"]
                self.root.after(0, lambda: self._populate_styles(names))
            except Exception as e:
                err = str(e)[:34]
                self.root.after(0, lambda m=err: self.styles_status.config(
                    text=m, foreground="red"))
            # Also pull the available upscalers for the output combobox.
            try:
                ru = requests.get(f"{api}/sdapi/v1/upscalers", timeout=10)
                ru.raise_for_status()
                ups = [u.get("name", "") for u in ru.json() if u.get("name")]
                ups = [u for u in ups if u and u.lower() != "none"]
                self.root.after(0, lambda: self._populate_upscalers(ups))
            except Exception:
                pass
            # Checkpoints (use 'title' - what sd_model_checkpoint accepts).
            try:
                rm = requests.get(f"{api}/sdapi/v1/sd-models", timeout=10)
                rm.raise_for_status()
                models = [m.get("title", "") for m in rm.json()
                          if m.get("title")]
                self.root.after(0, lambda: self._populate_models(models))
            except Exception:
                pass
            # Samplers.
            try:
                rs = requests.get(f"{api}/sdapi/v1/samplers", timeout=10)
                rs.raise_for_status()
                samps = [s.get("name", "") for s in rs.json() if s.get("name")]
                self.root.after(0, lambda: self._populate_samplers(samps))
            except Exception:
                pass

        threading.Thread(target=_probe, daemon=True).start()

    def _populate_upscalers(self, names):
        values = ["Lanczos (fast)"] + names
        self.upscaler_combo.config(values=values)

    def _populate_models(self, titles):
        self.model_combo.config(values=["(current)"] + titles)

    def _populate_samplers(self, names):
        if names:
            self.sampler_combo.config(values=names)

    def _populate_styles(self, names):
        self.styles_list.delete(0, "end")
        for n in names:
            self.styles_list.insert("end", n)
        self.styles_status.config(
            text=f"{len(names)} styles", foreground="green")

    def _selected_styles(self):
        return [self.styles_list.get(i)
                for i in self.styles_list.curselection()]

    # ----- gather + validate config ---------------------------------------
    def _collect_config(self):
        keyframes = {}
        for r in self.rows:
            try:
                idx, prompt = r.get()
            except ValueError:
                raise ValueError("Every keyframe needs a valid integer "
                                 "frame index.")
            if idx < 0:
                raise ValueError("Frame indices must be >= 0.")
            if not prompt:
                raise ValueError(f"Frame {idx} has an empty prompt.")
            keyframes[idx] = prompt          # last wins on duplicate index

        if 0 not in keyframes:
            raise ValueError("You must configure a keyframe for Frame 0 "
                             "to establish the initial seed frame.")

        total = int(self.frames_var.get())
        # Clamp any keyframe beyond the timeline - it just won't be reached,
        # but at least one keyframe (Frame 0) is guaranteed in range.
        return {
            "api": self.api_var.get().strip(),
            "total_frames": total,
            "zoom": float(self.zoom_var.get()),
            "negative": self.neg_var.get().strip(),
            "keyframes": keyframes,
            "styles": self._selected_styles(),
            "stitch": bool(self.stitch_var.get()),
            "fps": int(self.fps_var.get()),
            "direction": "in" if self.direction_var.get() == "Zoom In" else "out",
            "tween": int(self.tween_var.get()),
            "warp_strength": float(self.warp_var.get()),
            "gen_res": int(self.gen_var.get()),
            "steps": int(self.steps_var.get()),
            "cfg_scale": float(self.cfg_var.get()),
            "sampler": self.sampler_var.get(),
            "model": ("" if self.model_var.get() == "(current)"
                      else self.model_var.get()),
            "out_res": self.out_res_var.get(),
            "fit": "fit" if self.fit_var.get().startswith("Fit") else "cover",
            "speed": float(self.speed_var.get()),
            "start_image": self.start_image_var.get().strip(),
            "upscaler": ("" if self.upscaler_var.get().startswith("Lanczos")
                         else self.upscaler_var.get()),
            "out_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "output"),
        }

    # ----- start / stop ----------------------------------------------------
    def start_generation(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self._collect_config()
        except ValueError as e:
            messagebox.showerror("Invalid configuration", str(e))
            return

        self.cancel_flag.clear()
        self.progress.config(value=0)
        self.logbox.delete("1.0", "end")
        self.gen_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._log(f"=== Run started {datetime.now():%H:%M:%S} ===")
        self._log(f"Keyframes: {sorted(cfg['keyframes'])}")

        pipeline = ZoomquiltPipeline(cfg, self.events, self.cancel_flag)
        self.worker = threading.Thread(target=pipeline.run, daemon=True)
        self.worker.start()

    def stop_generation(self):
        if self.worker and self.worker.is_alive():
            self.cancel_flag.set()
            self.status_lbl.config(text="Stopping after current step...")

    # ----- event pump (runs on Tk main thread) ----------------------------
    def _drain_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                t = ev["type"]
                if t == "log":
                    self._log(ev["msg"])
                elif t == "progress":
                    self.progress.config(value=ev["value"])
                    if ev.get("label"):
                        self.status_lbl.config(text=ev["label"])
                elif t == "preview":
                    self._show_preview(ev["image"])
                elif t == "done":
                    self._on_done(ev["ok"], ev["msg"])
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _on_done(self, ok, msg):
        self.gen_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_lbl.config(text=msg)
        self._log("=== " + ("DONE: " if ok else "ENDED: ") + msg + " ===")
        if ok:
            messagebox.showinfo("Zoomquilt complete", msg)

    def _log(self, msg):
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")

    def _show_preview(self, pil_image):
        img = pil_image.copy()
        img.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
        self._preview_ref = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            PREVIEW_MAX // 2, PREVIEW_MAX // 2, image=self._preview_ref)

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Quit",
                                       "A run is in progress. Stop and quit?"):
                return
            self.cancel_flag.set()
        self._poll_running = False
        self.root.destroy()


# ===========================================================================
def main():
    if np is None:
        print("FATAL: numpy is required.  pip install numpy", file=sys.stderr)
        return
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    ZoomquiltApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
