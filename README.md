# DexGen — Multi-Attribute Conditional Diffusion for Sprite Synthesis

DexGen is a conditional generative model that synthesizes novel Pokémon-style sprites from user-specified categorical attributes: primary type, color, and body shape. The model combines a U-Net denoising backbone with a cosine-schedule DDIM sampler, trained on 1,024 sprites sourced from PokeAPI. At inference time, any combination of 18 types × 10 colors × 14 shapes can be queried to produce a unique 64×64 sprite.

## Installation & Run Instructions

```bash
pip install torch torchvision pandas Pillow numpy matplotlib cairosvg
python data/fetch.py
```

`fetch.py` downloads `data/sprites/*.png` and writes `data/pokemon.csv`. Fetching all Pokémon takes a few minutes; progress is printed every 50 entries.

**Option 1 — Train from scratch:** Open `model_training.ipynb` and run all cells top to bottom. Skip the Grid Search and Load Checkpoint cells.

**Option 2 — Load checkpoint and run inference:** Run the Load Checkpoint cell, then the inference cell below it.

Valid attribute values:
- **type**: grass, fire, water, bug, normal, poison, electric, ground, fairy, fighting, psychic, rock, ghost, ice, dragon, dark, steel, flying
- **color**: green, red, blue, white, brown, yellow, purple, pink, gray, black
- **shape**: quadruped, upright, armor, squiggle, bug-wings, wings, humanoid, legs, blob, heads, tentacles, arms, fish, ball

## Results

| Image | Type | Color | Shape |
|---|---|---|---|
|<img width="64" height="64" alt="water_blue_armor" src="https://github.com/user-attachments/assets/c87bbec0-1a1e-4f8f-ae7d-18a6b8a5eb34" />|Water|Blue|Arms|
|<img width="64" height="64" alt="fire_red_heads" src="https://github.com/user-attachments/assets/b1722b0a-3983-4c98-bfec-96f5bfdc7b3f" />|Fire|Red|Heads|
|<img width="64" height="64" alt="dragon_blue_armor" src="https://github.com/user-attachments/assets/f60d38d7-2a90-4b72-b667-440400b4e9f5" />|Dragon|Blue|Armor|
|<img width="64" height="64" alt="electric_yellow_bug-wings" src="https://github.com/user-attachments/assets/9b7fbc2e-06cb-40f7-a729-d532d4f4df5a" />|Electric|Yellow|Wings|
|<img width="64" height="64" alt="ghost_black_blob" src="https://github.com/user-attachments/assets/bdcec1ed-5181-400d-988a-6735e4ce1f5f" />|Ghost|Black|Blob|

## Extra Criteria

**Hyperparameter Tuning Strategies** — A full grid search was run over learning rate {1e-3, 1e-4}, condition dimension {32, 64, 128}, batch size {32, 64}, and training epochs {50, 100, 200, 500} before committing to the final configuration (cond_dim=128, lr=1e-4, 500 epochs). This identified that a larger condition embedding meaningfully improved generation quality.

**Creative Latent Space Exploration** — The model conditions on three independent categorical attributes simultaneously. Each attribute gets its own embedding layer, projected into a shared condition vector and fused with sinusoidal timestep embeddings. This allows compositional generation: any valid (type, color, shape) triple maps to a distinct region of the learned distribution, enabling structured latent space traversal across 2,520 possible attribute combinations.

## Difficulties

**Small dataset with high attribute diversity** — 1,024 samples spread across 18 types and 14 shapes left many attribute combinations with very few examples. Attempted to address with aggressive augmentation (horizontal flip, color jitter, random rotation) and EMA smoothing to stabilize training and reduce overfitting.

**Unstable DDIM denoising** — Early sampling runs produced washed-out or noisy images because the predicted x̂₀ occasionally drifted outside the valid pixel range, compounding across reverse steps. Fixed by clamping x̂₀ ∈ [−1, 1] at each step and re-deriving the noise vector from the clamped estimate before computing the next step.

**Conditioning architecture design** — Injecting per-attribute semantics and a per-step timestep signal into every ResBlock required some thought around architecture. The solution was to project the condition vector (attribute + time) into each ResBlock's channel dimension and broadcast it spatially, rather than concatenating to the image channels.

**Time constraints** — The original plan included a Gallery GUI for interactive sprite generation. Due to time constraints, the GUI was not completed; focus shifted instead to deeper investment in hyperparameter tuning and creative latent space exploration.
