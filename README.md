# DexGen — Multi-Attribute Conditional Diffusion for Sprite Synthesis

A conditional diffusion model that generates Pokémon-style sprites from categorical attributes: type, color, and body shape.

## Overview

DexGen trains a Conditional DDIM (Denoising Diffusion Implicit Model) with a U-Net backbone on 1,024 sprites and metadata sourced from [PokeAPI](https://pokeapi.co/docs/v2#pokemon-section). At inference time, the model synthesizes a novel 64×64 sprite conditioned on a user-specified combination of attributes.

## Architecture

- **ConditionEmbedding** — embeds three categorical attributes into a single condition vector.
- **SinusoidalPositionEmbeddings** — encodes the diffusion timestep as a fixed sinusoidal vector of dimension `cond_dim`; added element-wise to the attribute condition vector.
- **ResBlock** — `Conv2d → GroupNorm(8) → SiLU → + cond_proj → Conv2d → GroupNorm(8) → SiLU → + residual`; condition is projected via `Linear(cond_dim, out_channels)` and broadcast over spatial dims
- **U-Net** — encoder [64→64→128→256] with 2× strided downsampling, bottleneck ResBlock, decoder [512→256→384→128→192→64] with bilinear upsampling and skip connections, output `Conv2d(64, 3)`
- **DiffusionSchedule** — cosine noise schedule (T=1000, offset=0.008): `α̅ₜ = cos²(((t/T + s)/(1+s)) · π/2)`, forward process `xₜ = √α̅ₜ · x₀ + √(1−α̅ₜ) · ε`
- **DDIM Sampler** — deterministic reverse: predict `x̂₀`, clamp to `[−1, 1]`, re-derive `ε`, step to `xₜ₋₁`

## Data

Sprites and metadata sourced from [PokeAPI](https://pokeapi.co/). The dataset contains **1,024 Pokémon** (Gen 1–9) with:
- Front-facing sprites converted to RGB on a white background, resized to **64×64 px**
- **18 primary types**: grass, fire, water, bug, normal, poison, electric, ground, fairy, fighting, psychic, rock, ghost, ice, dragon, dark, steel, flying
- **10 colors**: green, red, blue, white, brown, yellow, purple, pink, gray, black
- **14 body shapes**: quadruped, upright, armor, squiggle, bug-wings, wings, humanoid, legs, blob, heads, tentacles, arms, fish, ball

Training augmentations: random horizontal flip, color jitter (brightness/contrast/saturation ±0.2), random rotation ±15°.

## Training

| Hyperparameter | Value |
|---|---|
| Image size | 64×64 |
| Batch size | 64 |
| Epochs | 500 |
| Optimizer | Adam, lr=1e-4 |
| Learning Rate | 1e-4 |
| Condition dim | 128 |
| Diffusion steps T | 1,000 |
| Loss | MSE on predicted noise |

Hyperparameters were selected via grid search over `lr ∈ {1e-3, 1e-4}`, `cond_dim ∈ {32, 64, 128}`, `batch_size ∈ {32, 64}`, `num_epochs ∈ {50, 100, 200, 500}`, then the final model was trained at `cond_dim=128` for 500 epochs.

## Saved Checkpoint

`dexgen.pt` stores:
- `model_state_dict` — base model weights
- `ema_model_state_dict` — EMA-smoothed weights (used for inference)
- `type_to_idx`, `color_to_idx`, `shape_to_idx` — vocabulary mappings
- `cond_dim` — condition vector dimensionality (128)

## Fetching the Data

Sprite images and metadata are not included in the repository. Fetch them from [PokeAPI](https://pokeapi.co/) by running:

```sh
python data/fetch.py
```

This downloads `data/sprites/*.png` (front-facing sprites for Gen 1–9) and writes `data/pokemon.csv` with type, color, and shape metadata. Fetching all 1,025 Pokémon takes a few minutes; progress is printed every 50 entries.

## Running the Model

### Option 1 — Run the notebook sequentially

Open `model_training.ipynb` and run all cells top to bottom. This will download the data, train the model from scratch, and generate a sample image at the end. **Skip the Load Checkpoint cell** — running it will override the freshly trained model with the saved weights.

### Option 2 — Load the checkpoint and run inference

Open `model_training.ipynb` and run the **Load Checkpoint** cell, then the inference cell below it. This skips training entirely and generates sprites directly from `dexgen.pt`.

Valid attribute values:
- **type**: grass, fire, water, bug, normal, poison, electric, ground, fairy, fighting, psychic, rock, ghost, ice, dragon, dark, steel, flying
- **color**: green, red, blue, white, brown, yellow, purple, pink, gray, black
- **shape**: quadruped, upright, armor, squiggle, bug-wings, wings, humanoid, legs, blob, heads, tentacles, arms, fish, ball

## Sample Images

<!-- Add generated sample images and their conditions here -->

## Dependencies

```
torch torchvision pandas Pillow numpy matplotlib cairosvg
```
