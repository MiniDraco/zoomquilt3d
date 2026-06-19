#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_theme_banks.py  -  generate themed prompt banks for the Zoomquilt app.

Writes one .txt per theme into the prompts/ folder (noir, space, trippy,
creepy). Each is built combinatorially from curated pools so the dice / Roll
All have lots of variety to draw from. Tuned for zoom-into-the-center scenes.

Run:   python build_theme_banks.py
"""

import os
import random

TARGET = 1500          # unique prompts per theme
SEED = 11

TAILS = [
    "ultra detailed, cinematic lighting, masterpiece",
    "intricate detail, volumetric light, sharp focus",
    "highly detailed, dramatic lighting, 8k, masterpiece",
    "atmospheric depth, beautiful composition, award-winning",
]

THEMES = {
    "noir": {
        "scene": [
            "a rain-slicked city street at midnight", "a dim detective's office",
            "a smoky underground jazz club", "a fog-drowned harbor dock",
            "a neon-lit back alley", "an empty train platform at 3am",
            "a room striped by venetian-blind shadows", "a lonely all-night diner",
            "a shadowed tenement stairwell", "a rooftop above the sleeping city",
            "a deserted subway car", "a cramped interrogation room",
            "a wet cobblestone lane under a single streetlamp",
            "a smoke-filled poker den", "a derelict theatre lobby",
            "a moonlit pier slick with rain", "a glass-walled corner office at night",
            "an abandoned warehouse lit by one bulb", "a hotel hallway of locked doors",
            "a flickering cinema marquee in the rain", "a parking garage in deep shadow",
            "a telephone booth glowing in the fog", "a morgue corridor",
            "a record store after hours", "a dockside bar with a dying neon sign",
            "a courthouse staircase in hard shadow", "a vintage car under a streetlight",
            "a foggy bridge vanishing into darkness", "a back-room speakeasy",
            "a newspaper office at deadline, smoke hanging",
        ],
        "atmos": [
            "rain streaking every surface", "cigarette smoke curling in the light",
            "thick fog rolling through", "wet reflections on the pavement",
            "dust drifting in a single beam", "steam rising from a manhole",
            "venetian-blind stripes across the wall", "a flickering neon glow",
            "shadows pooling in every corner", "rain hammering the window",
        ],
        "light": [
            "harsh single key light", "a bare bulb swinging overhead",
            "neon spilling through the blinds", "a streetlamp cutting the fog",
            "moonlight through a dirty window", "headlights sweeping the wall",
            "a desk lamp in the dark", "the red glow of a distant sign",
        ],
        "mood": ["moody and mysterious", "tense and melancholic", "lonely and cold",
                 "ominous and quiet", "shadowy and dangerous", "world-weary"],
        "style": [
            "black-and-white film noir", "high-contrast chiaroscuro",
            "1940s noir cinematography", "grainy monochrome film still",
            "moody black and white photography", "classic detective-film aesthetic",
            "stark noir matte painting",
        ],
    },
    "space": {
        "scene": [
            "a vast glowing nebula", "a spiral galaxy seen edge-on",
            "a ringed gas giant over its icy moons", "a dense asteroid field",
            "an expanding supernova remnant", "a wormhole tearing the starfield",
            "a globular cluster of a thousand suns", "a dark nebula pierced by newborn stars",
            "a comet streaking past a blue planet", "the blazing core of a quasar",
            "a frozen alien lake under twin suns", "a galactic core ablaze with light",
            "a derelict starship corridor drifting in the void",
            "an orbital ring station above a storm world", "a crystalline asteroid catching starlight",
            "a black hole's glowing accretion disk", "an alien ocean world from orbit",
            "a lunar base under a giant ringed planet", "a field of drifting ice comets",
            "a colossal space elevator rising to orbit", "a shimmering aurora over a methane sea",
            "a starfield reflected in a dome city", "a molten lava planet's terminator line",
            "a swarm of solar sails near a red giant", "a cathedral-like nebula of dust pillars",
            "a shattered moon ring around a desert world", "the bridge of a vast generation ship",
            "a binary star system at sunset", "a glittering ring of orbital debris",
            "a deep-space relay beacon among the stars",
        ],
        "atmos": [
            "cosmic gas swirling", "stardust drifting", "ion trails glowing",
            "plasma arcs flickering", "distant galaxies scattered across the dark",
            "solar wind shimmering", "frozen vapor catching the light",
            "ribbons of aurora overhead", "meteors streaking past",
            "luminous nebula clouds rolling",
        ],
        "light": [
            "blazing starlight", "the glow of twin suns", "cold blue stellar light",
            "the fiery rim of a red giant", "soft nebula luminescence",
            "harsh unfiltered sunlight in vacuum", "the eerie glow of an accretion disk",
            "bioluminescent aurora light",
        ],
        "mood": ["vast and awe-inspiring", "serene and infinite", "lonely and sublime",
                 "majestic and timeless", "mysterious and cold", "epic and grand"],
        "style": [
            "photoreal deep-space render", "epic sci-fi concept art",
            "hyperreal 8k astrophotography", "luminous cosmic matte painting",
            "cinematic space vista", "detailed hard-sci-fi illustration",
            "ethereal astronomical artwork",
        ],
    },
    "trippy": {
        "scene": [
            "a kaleidoscopic fractal tunnel", "a melting rainbow landscape",
            "an infinite mirror room", "a liquid-chrome dreamscape",
            "a garden of blooming fractal flowers", "a swirling vortex of color",
            "a geometric mandala temple", "a dripping psychedelic jungle",
            "an iridescent crystal cavern", "impossible Escher architecture",
            "a tunnel of pulsing neon hexagons", "a floating island of melting clocks",
            "a sea of undulating fractal waves", "a cathedral grown from glowing fungi",
            "a spiral staircase into a fractal sky", "a forest of translucent jelly trees",
            "a city folding into itself endlessly", "a kaleidoscope desert of mirrored dunes",
            "a galaxy made of stained glass", "a river of liquid light",
            "a chamber of breathing geometric walls", "a meadow of eyeball flowers",
            "an ocean of mercury under two moons", "a temple of rotating sacred geometry",
            "a labyrinth of glowing fractal coral", "a sky raining luminous paint",
            "a tunnel of spinning op-art spirals", "a void blooming with neon fractals",
            "a melting checkerboard horizon", "an aurora made of liquid prisms",
        ],
        "atmos": [
            "colors melting and flowing", "fractals blooming endlessly",
            "geometry breathing and shifting", "light refracting into rainbows",
            "patterns rippling outward", "shapes folding into themselves",
            "iridescent mist swirling", "neon trails smearing through the air",
            "kaleidoscopic reflections multiplying", "liquid light dripping",
        ],
        "light": [
            "pulsing neon glow", "iridescent rainbow light", "blacklight UV shimmer",
            "prismatic refracted beams", "glowing bioluminescence",
            "shifting holographic light", "saturated candy-bright light",
            "luminous fractal radiance",
        ],
        "mood": ["surreal and hypnotic", "euphoric and dreamlike", "mind-bending",
                 "wondrous and strange", "hallucinatory", "cosmic and otherworldly"],
        "style": [
            "psychedelic fractal art", "vibrant DMT-style illustration",
            "vaporwave dreamscape", "iridescent surreal render",
            "kaleidoscopic digital painting", "trippy op-art masterpiece",
            "hyper-saturated visionary art",
        ],
    },
    "creepy": {
        "scene": [
            "an abandoned asylum corridor", "a foggy graveyard at midnight",
            "a derelict hospital ward", "a doll-filled attic in the dark",
            "a twisted dead forest", "a flickering haunted-mansion hallway",
            "a blood-red ritual chamber", "a decaying carnival at night",
            "a damp stone basement", "an eldritch cathedral of bone",
            "a child's nursery long abandoned", "a drowned village under black water",
            "a morgue with a flickering light", "a cabin deep in a whispering wood",
            "a staircase descending into pitch black", "a hall of cracked antique mirrors",
            "a rusted slaughterhouse", "a fog-choked moor with standing stones",
            "an overgrown mausoleum", "a derelict ship's flooded hold",
            "a sanatorium bathroom of broken tiles", "a clocktower full of crows",
            "a circus tent rotting in the dark", "a wax museum of melted figures",
            "a sunken chapel lit by candles", "a butcher's freezer of swaying hooks",
            "a swamp of gnarled black roots", "a derelict subway tunnel dripping water",
            "a farmhouse with every door ajar", "an attic crawlspace behind the wall",
        ],
        "atmos": [
            "fog creeping along the floor", "shadows shifting where they shouldn't",
            "dust hanging in stale air", "a flickering light buzzing",
            "cobwebs draped over everything", "black water dripping in the dark",
            "a cold draft stirring tattered curtains", "silhouettes lurking in doorways",
            "candle flames guttering", "something just out of sight",
        ],
        "light": [
            "a single flickering bulb", "pale moonlight through broken glass",
            "a guttering candle", "a red emergency light pulsing",
            "a flashlight beam cutting the dark", "sickly green gloom",
            "lightning flashing through a window", "the cold blue of dawn",
        ],
        "mood": ["dreadful and eerie", "unsettling and tense", "haunting and silent",
                 "ominous and wrong", "sinister and cold", "foreboding"],
        "style": [
            "atmospheric horror art", "eerie gothic illustration",
            "unsettling cinematic still", "dark dreadful matte painting",
            "found-footage horror aesthetic", "grim desaturated horror render",
            "nightmarish detailed artwork",
        ],
    },
}

TEMPLATES = [
    "{scene}, {atmos}, {light}, {mood}, {style}, {tail}",
    "{style} of {scene}, {atmos}, {light}, {tail}",
    "{scene} with {atmos}, {light}, {mood}, {style}, {tail}",
    "{scene}, {light}, {atmos}, {mood}, {tail}",
]


def build(theme, pools, rng):
    prompts = set()
    attempts = 0
    while len(prompts) < TARGET and attempts < TARGET * 100:
        attempts += 1
        p = rng.choice(TEMPLATES).format(
            scene=rng.choice(pools["scene"]),
            atmos=rng.choice(pools["atmos"]),
            light=rng.choice(pools["light"]),
            mood=rng.choice(pools["mood"]),
            style=rng.choice(pools["style"]),
            tail=rng.choice(TAILS),
        )
        prompts.add(p[:1].upper() + p[1:])
    return sorted(prompts)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "prompts")
    os.makedirs(out_dir, exist_ok=True)
    for theme, pools in THEMES.items():
        rng = random.Random(SEED + sum(ord(c) for c in theme))
        prompts = build(theme, pools, rng)
        path = os.path.join(out_dir, f"{theme}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(prompts) + "\n")
        print(f"{theme}: {len(prompts)} prompts -> {path}")


if __name__ == "__main__":
    main()
