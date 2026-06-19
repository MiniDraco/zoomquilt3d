#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theme_gen.py  -  type a theme, get a prompt bank for the Zoomquilt app.

  python theme_gen.py "cyberpunk megacity"
  python theme_gen.py underwater ruins --count 600

Writes prompts/<slug>.txt (one prompt per line). It shows up in the app's
"Prompt set" dropdown automatically.

How it generates:
  * PRIMARY - a local LLM via Ollama (http://localhost:11434). Best quality;
    the model invents varied, theme-appropriate scenes. Pick the model with
    --model, else the first installed model is used.
  * FALLBACK - if Ollama isn't reachable, a combinatorial generator injects the
    theme into generic scene/atmosphere/light/mood/style pools. Works offline.

Flags:
  --count N        target number of prompts (default 400)
  --model NAME     Ollama model (default: first available)
  --ollama URL     Ollama base URL (default http://localhost:11434)
  --out PATH       output file (default prompts/<slug>.txt)
"""

import os
import re
import sys
import json
import random
import argparse

try:
    import requests
except ImportError:
    requests = None

HERE = os.path.dirname(os.path.abspath(__file__))

TAILS = [
    "ultra detailed, cinematic lighting, masterpiece",
    "intricate detail, volumetric light, sharp focus",
    "highly detailed, dramatic lighting, 8k, masterpiece",
    "atmospheric depth, beautiful composition, award-winning",
]


# ---------------------------------------------------------------------------
#  Cleaning
# ---------------------------------------------------------------------------
def clean_line(line):
    s = line.strip()
    # strip leading numbering / bullets / quotes
    s = re.sub(r'^\s*(?:\d+[\.\)]|[-*•])\s*', '', s)
    s = s.strip().strip('"\'`').strip()
    if len(s) < 20 or len(s) > 400:
        return None
    if s.lower().startswith(("here are", "sure", "okay", "prompt", "theme:")):
        return None
    return s


def slugify(theme):
    s = re.sub(r'[^a-z0-9]+', '-', theme.lower()).strip('-')
    return s or "theme"


# ---------------------------------------------------------------------------
#  Ollama backend
# ---------------------------------------------------------------------------
def ollama_models(base):
    r = requests.get(f"{base}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", []) if m.get("name")]


def ollama_generate(base, model, theme, per_call):
    instruction = (
        "You are writing prompts for an AI image generator that makes "
        "infinite-zoom 'zoomquilt' videos.\n"
        f"Theme: {theme}.\n"
        f"Write exactly {per_call} DISTINCT, vivid, single-line image prompts. "
        "Each must describe a different scene that fits the theme and works "
        "when you zoom into its center (a clear focal subject, depth toward the "
        "middle). Each line = scene + atmosphere + lighting + a couple of "
        "style/quality words. No numbering, no preamble, no blank lines - just "
        "one prompt per line."
    )
    payload = {
        "model": model, "prompt": instruction, "stream": False,
        "options": {"temperature": 1.05, "top_p": 0.95},
    }
    r = requests.post(f"{base}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    text = r.json().get("response", "")
    return [c for c in (clean_line(ln) for ln in text.splitlines()) if c]


def gen_ollama(theme, count, base, model, per_call):
    models = ollama_models(base)
    if not models:
        raise RuntimeError("Ollama has no models installed (try: ollama pull llama3).")
    model = model or models[0]
    print(f"Using Ollama model '{model}' at {base}")
    out, seen = [], set()
    max_calls = max(6, (count // max(1, per_call)) * 4)
    for i in range(max_calls):
        if len(out) >= count:
            break
        try:
            batch = ollama_generate(base, model, theme, per_call)
        except Exception as e:
            print(f"  call {i + 1} failed: {e}")
            continue
        new = 0
        for p in batch:
            k = p.lower()
            if k not in seen:
                seen.add(k)
                out.append(p)
                new += 1
        print(f"  call {i + 1}: +{new} (total {len(out)}/{count})")
        if new == 0 and i > 3:
            break
    return out[:count]


# ---------------------------------------------------------------------------
#  Offline combinatorial fallback
# ---------------------------------------------------------------------------
PLACES = [
    "city", "temple", "forest", "cavern", "palace", "ruins", "marketplace",
    "tower", "tunnel", "garden", "shrine", "laboratory", "cathedral",
    "wasteland", "harbor", "valley", "arena", "library", "station",
    "sanctuary", "village", "fortress", "vault", "observatory", "alley",
    "throne room", "courtyard", "canyon", "bridge", "gateway",
]
SCENE_TEMPLATES = [
    "a {theme} {place}", "a {place} in a {theme} world",
    "the heart of a {theme} {place}", "a vast {theme} {place}",
    "an ancient {theme} {place}", "a glowing {theme} {place}",
]
ATMOS = ["mist drifting through", "light pouring from the center",
         "particles floating in the air", "fog rolling low",
         "haze catching the light", "dust suspended in beams",
         "steam curling upward", "shafts of light cutting across"]
LIGHT = ["dramatic rim lighting", "soft glowing ambiance", "harsh shadows",
         "neon glow", "golden backlight", "moody low light",
         "luminous central glow", "cinematic volumetric light"]
MOOD = ["awe-inspiring", "mysterious", "serene", "epic", "dreamlike",
        "ominous", "majestic", "otherworldly"]
STYLE = ["highly detailed digital painting", "cinematic concept art",
         "hyperreal 8k render", "atmospheric matte painting",
         "epic fantasy artwork", "moody illustration"]


def gen_offline(theme, count):
    rng = random.Random(sum(ord(c) for c in theme))
    out = set()
    attempts = 0
    while len(out) < count and attempts < count * 80:
        attempts += 1
        scene = rng.choice(SCENE_TEMPLATES).format(
            theme=theme, place=rng.choice(PLACES))
        p = (f"{scene}, {rng.choice(ATMOS)}, {rng.choice(LIGHT)}, "
             f"{rng.choice(MOOD)}, {rng.choice(STYLE)}, {rng.choice(TAILS)}")
        out.add(p[:1].upper() + p[1:])
    return sorted(out)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("theme", nargs="*", help="theme text, e.g. cyberpunk megacity")
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--model", default=None)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--per-call", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    theme = " ".join(args.theme).strip()
    if not theme:
        try:
            theme = input("Enter a theme: ").strip()
        except EOFError:
            theme = ""
    if not theme:
        print("No theme given.")
        return

    prompts = []
    base = args.ollama.rstrip("/")
    if requests is not None:
        try:
            requests.get(f"{base}/api/tags", timeout=4).raise_for_status()
            print(f"Theme: {theme!r}  ->  asking local LLM for ~{args.count} prompts...")
            prompts = gen_ollama(theme, args.count, base, args.model,
                                 args.per_call)
        except Exception as e:
            print(f"Ollama unavailable ({e}); using offline generator.")
    else:
        print("`requests` not installed; using offline generator.")

    if len(prompts) < max(20, args.count // 4):
        print(f"Filling out with the offline generator (had {len(prompts)}).")
        extra = gen_offline(theme, args.count - len(prompts))
        seen = {p.lower() for p in prompts}
        prompts += [p for p in extra if p.lower() not in seen]

    prompts = prompts[:args.count]
    out = args.out or os.path.join(HERE, "prompts", f"{slugify(theme)}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(prompts) + "\n")
    print(f"\nWrote {len(prompts)} prompts -> {out}")
    print("Open the app's 'Prompt set' dropdown and pick it.")


if __name__ == "__main__":
    main()
