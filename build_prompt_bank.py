#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_prompt_bank.py  -  generate prompts.txt for the 3D AI Zoomquilt Engine.

Produces 5000 unique, creative landscape prompts (one per line) by sampling
from curated component pools across several sentence templates. Tuned for
infinite-zoom / outpaint imagery: evocative, depth-rich, painterly scenes.

Run once:   python build_prompt_bank.py
Output:     prompts.txt   (next to this script)
"""

import os
import random

TARGET = 5000
SEED = 7   # deterministic output so re-runs are reproducible

# ---------------------------------------------------------------------------
#  Component pools
# ---------------------------------------------------------------------------

# Core scene / biome / landform  (the subject)
SCENES = [
    "a windswept alpine meadow", "a bioluminescent mangrove swamp",
    "an endless rolling lavender field", "a shattered glacier canyon",
    "a sunken temple lagoon", "a floating archipelago in the clouds",
    "a crystalline salt flat", "a fog-drowned redwood forest",
    "a volcanic obsidian shoreline", "a terraced rice valley",
    "a coral reef cathedral", "a moss-cloaked stone monastery",
    "a desert of glass dunes", "an aurora-lit tundra",
    "a thundering basalt waterfall", "a mirror-still mountain lake",
    "a canyon of striped sandstone", "a frozen waterfall grotto",
    "a sea of drifting paper lanterns", "a petrified ancient forest",
    "a geothermal hot-spring terrace", "an overgrown ruined cathedral",
    "a cliffside hanging garden", "a meteor-cratered badland",
    "a kelp forest sunbeam cavern", "a marble quarry filled with rain",
    "a sprawling mushroom forest", "a winter birch grove",
    "a tidal cave of glowing pools", "a fractured ice shelf at dawn",
    "an emerald jungle waterfall basin", "a wheat field under storm clouds",
    "a chalk-white coastal cliff", "a river delta seen from above",
    "a snow-buried pine valley", "a cracked dry lakebed",
    "a labyrinth of slot canyons", "a poppy field at golden hour",
    "a sea stack archway at low tide", "a misty terraced tea plantation",
    "a frozen fjord between black peaks", "a glowing cave of crystal spires",
    "a vast sand sea of crescent dunes", "a flooded gothic library",
    "a cliff village clinging to a gorge", "a star-reflecting reflecting pool",
    "a meadow of fireflies at dusk", "a canyon river carving red rock",
    "an iceberg drifting through still fog", "a cherry-blossom mountain pass",
    "a derelict greenhouse reclaimed by vines", "a vast cosmic nebula",
    "a sunflower field stretching to the horizon", "a moonlit dune sea",
    "a thunderstorm rolling over the plains", "a quiet bamboo grove",
    "a glacier-fed turquoise river", "a windmill-dotted polder",
    "a forest of giant ferns", "a rainbow-banded badlands",
    "a sea cave open to the sunset", "a frostbitten alpine pass",
    "a lily-covered swamp at twilight", "a hillside of ancient olive trees",
    "a coastal lighthouse on jagged rocks", "a valley of floating stone islands",
    "a crater lake ringed by volcanoes", "an autumn maple canyon",
    "a moonlit beach of black sand", "a vineyard on terraced hills",
    "a frozen marsh under the northern lights", "a jungle ziggurat lost in mist",
    "a desert oasis ringed by palms", "a snowy mountain shrine",
    "a field of tall golden grass", "a tide pool reef at dawn",
    "a canyon of hoodoos and spires", "a drowned forest in a clear lake",
    "an ice cave glowing blue", "a meadow beneath a double rainbow",
    "a cliff of nesting seabirds", "a misty valley of waterfalls",
    "a frozen lake cracked into mosaic", "a savanna under a vast sky",
    "a fjord village at blue hour", "a canyon spanned by a stone bridge",
    "a forest clearing struck by sunbeams", "a coastline of towering arches",
    "a high plateau of wind-carved rock", "a marsh of glowing will-o-wisps",
    "a snowfield rippled by wind", "a rainforest canopy above the clouds",
    "a dune valley with a lone caravan", "a riverbank of weeping willows",
    "a volcanic caldera lake", "a terraced waterfall of travertine pools",
    "a foggy moor dotted with standing stones", "a crystal geode cavern",
    "a coastal cliff path in the rain", "a sun-bleached coral atoll",
    "a meadow of swaying cottongrass", "a ravine bridged by ancient roots",
]

# A distinctive secondary feature or focal element
FEATURES = [
    "a lone gnarled tree", "scattered glowing crystals",
    "a winding stone staircase", "a distant ringed planet in the sky",
    "drifting seed pods of light", "a half-buried marble statue",
    "floating islands tethered by vines", "a flock of paper-white birds",
    "a ruined archway veiled in mist", "shafts of volumetric god rays",
    "a meandering silver river", "clusters of luminous mushrooms",
    "a weathered wooden bridge", "swirling clouds of fireflies",
    "a colossal ancient tree", "petals carried on the wind",
    "a still reflecting pool", "wisps of low-hanging fog",
    "a cascade of terraced pools", "delicate frost patterns",
    "drifting jellyfish in the air", "a spiral of migrating birds",
    "a moss-covered monolith", "bioluminescent plankton",
    "a crumbling watchtower", "ribbons of aurora overhead",
    "a field of glowing orbs", "a serpentine mountain path",
    "translucent crystal spires", "a sunken stone idol",
    "drifting dandelion seeds", "a lantern-lit dock",
    "schools of fish overhead", "a meteor streaking across the sky",
    "softly glowing pollen", "a hidden cave mouth",
    "a chain of stepping stones", "vines heavy with blossoms",
    "a curtain of falling water", "wind-bent silver grass",
]

# Time of day + light quality
LIGHT = [
    "at golden hour", "under a blazing sunset", "at misty dawn",
    "in the cold blue of twilight", "beneath a full harvest moon",
    "under a sky of swirling stars", "in soft overcast light",
    "at the first light of morning", "under a stormy dusk sky",
    "in dappled afternoon sunlight", "beneath an emerald aurora",
    "under a sky streaked with comets", "in the pale glow before sunrise",
    "at high noon with hard shadows", "under a violet dusk",
    "in the silver light of a moonlit night", "during a fiery dawn",
    "beneath a cloudless cobalt sky", "in the amber light of late autumn",
    "under shifting curtains of northern lights", "in eerie green storm light",
    "at the last ember of sunset", "under a pearl-grey morning haze",
    "beneath a sky of rolling thunderheads", "in warm honeyed backlight",
]

# Weather / atmosphere
WEATHER = [
    "with rolling banks of fog", "after a fresh rainfall",
    "in a gentle snowfall", "with mist rising off the water",
    "under gathering storm clouds", "in crisp clear air",
    "with a light drizzle shimmering", "amid swirling autumn leaves",
    "with heat haze rippling the horizon", "in a sudden sun shower",
    "with frost glittering on every surface", "under a veil of sea spray",
    "as low clouds drift through the valley", "in still, breathless calm",
    "with petals drifting through the air", "as lightning flickers in the distance",
    "with dew clinging to everything", "in a soft golden dust haze",
    "as a rainbow arcs overhead", "with steam curling from the ground",
]

# Color palette / mood tone
PALETTE = [
    "in rich teal and gold tones", "in a warm autumnal palette",
    "in cool blues and silvers", "in vivid emerald and jade",
    "in dusky rose and violet hues", "in monochrome moody greys",
    "in fiery orange and crimson", "in soft pastel gradients",
    "in deep midnight blues", "in luminous neon accents",
    "in earthy ochre and umber", "in icy cyan and white",
    "in saturated tropical colors", "in muted sepia tones",
    "in iridescent opal shimmer", "in bold complementary contrasts",
    "in candy-bright surreal colors", "in faded vintage film tones",
    "in glowing amber and bronze", "in cold steel and pale blue",
]

# Rendering style / medium
STYLE = [
    "highly detailed digital painting", "epic fantasy concept art",
    "dreamy surreal illustration", "photorealistic landscape photography",
    "ethereal matte painting", "lush studio-ghibli style scenery",
    "cinematic wide-angle render", "atmospheric oil painting",
    "intricate ink and watercolor", "luminous fantasy artwork",
    "hyperreal 8k render", "soft impressionist brushwork",
    "moody cinematic still", "vibrant storybook illustration",
    "sweeping panoramic vista", "delicate gouache painting",
    "otherworldly sci-fi landscape", "rich textured digital matte",
    "serene minimalist composition", "grand romantic landscape painting",
]

# Mood / feeling
MOOD = [
    "serene and contemplative", "vast and awe-inspiring",
    "mysterious and dreamlike", "peaceful and warm",
    "epic and sublime", "haunting and quiet",
    "magical and otherworldly", "lonely and beautiful",
    "tranquil and meditative", "wondrous and surreal",
    "majestic and timeless", "intimate and gentle",
    "wild and untamed", "calm and luminous",
]

# Universal quality tags appended to every prompt
TAILS = [
    "ultra detailed, cinematic lighting, masterpiece",
    "intricate detail, volumetric light, sharp focus",
    "highly detailed, dramatic lighting, 8k, masterpiece",
    "stunning composition, atmospheric depth, ultra detailed",
    "rich detail, beautiful lighting, award-winning",
]

# Sentence templates (index-based fills). Each yields a different rhythm so
# 5000 prompts don't all read identically.
TEMPLATES = [
    "{scene}, {feature}, {light}, {weather}, {palette}, {style}, {mood}, {tail}",
    "{scene} {light}, {weather}, with {feature}, {palette}, {style}, {tail}",
    "{style} of {scene}, {feature}, {light}, {mood}, {tail}",
    "{scene}, {palette}, {feature} {light}, {weather}, {mood}, {tail}",
    "{scene} with {feature}, {weather} {light}, {style}, {palette}, {tail}",
]


def lower_first(s):
    return s[:1].lower() + s[1:] if s else s


def build():
    rng = random.Random(SEED)
    prompts = set()
    attempts = 0
    max_attempts = TARGET * 80

    while len(prompts) < TARGET and attempts < max_attempts:
        attempts += 1
        tpl = rng.choice(TEMPLATES)
        p = tpl.format(
            scene=rng.choice(SCENES),
            feature=rng.choice(FEATURES),
            light=rng.choice(LIGHT),
            weather=rng.choice(WEATHER),
            palette=rng.choice(PALETTE),
            style=rng.choice(STYLE),
            mood=rng.choice(MOOD),
            tail=rng.choice(TAILS),
        )
        # Tidy stray capitalization from templates that start with a fill.
        p = p[:1].upper() + p[1:]
        prompts.add(p)

    return sorted(prompts)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "prompts.txt")
    prompts = build()
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(prompts) + "\n")
    combos = (len(SCENES) * len(FEATURES) * len(LIGHT) * len(WEATHER) *
              len(PALETTE) * len(STYLE) * len(MOOD) * len(TAILS) *
              len(TEMPLATES))
    print(f"Wrote {len(prompts)} unique prompts -> {out}")
    print(f"(theoretical combination space: ~{combos:,})")


if __name__ == "__main__":
    main()
