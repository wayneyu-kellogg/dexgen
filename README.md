# DexGen — Multi-Attribute Conditional Diffusion for Sprite Synthesis

A conditional diffusion model that generates Pokémon-style sprites from categorical attributes such as type, color, and body shape.

## Overview

DexGen trains a Conditional DDIM (Denoising Diffusion Implicit Model) with a U-Net backbone on sprites and metadata sourced from [PokeAPI](https://pokeapi.co/docs/v2#pokemon-section). At inference time, a user selects attributes via a web UI and the model synthesizes a novel sprite conditioned on those inputs.

## Features

- **Automated data pipeline** — fetches sprites and categorical metadata directly from PokeAPI
- **Conditional U-Net** — attribute embeddings injected into the U-Net to guide the reverse diffusion process
- **DDIM sampling** — fast, deterministic inference
- **Interactive gallery GUI** — dropdown menus for attribute selection; displays generated sprites alongside generation history

## Architectur (TBD)

```
Attributes (type, color, shape)
        │
  Embedding Layers
        │
   Condition Vector
        │
U-Net (DDPM backbone)  ←──  Noisy image xₜ
        │
   Predicted noise εθ
        │
  DDIM Sampler  →  Generated sprite x₀
```

## Data

Sprites and metadata are sourced from [PokeAPI](https://pokeapi.co/). The pipeline downloads:
- Front-facing sprites (96×96 px)
- Categorical attributes: Primary Type, Color, Body Shape

