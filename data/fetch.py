import csv
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://pokeapi.co/api/v2"
TOTAL = 1025
SPRITE_DIR = Path(__file__).parent / "sprites"
CSV_OUT = Path(__file__).parent / "pokemon.csv"
CSV_FIELDS = ["id", "name", "primary_type", "secondary_type", "color", "shape", "sprite_file"]


def _get(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)


def download_sprite(url, path):
    if path.exists():
        return
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    path.write_bytes(r.content)


def fetch_pokemon(pokemon_id):
    poke = _get(f"{BASE_URL}/pokemon/{pokemon_id}")
    if poke is None:
        return None

    sprite_url = poke["sprites"].get("front_default")
    if not sprite_url:
        return None

    species = _get(f"{BASE_URL}/pokemon-species/{pokemon_id}")
    if species is None:
        return None

    types = sorted(poke["types"], key=lambda t: t["slot"])
    primary_type = types[0]["type"]["name"]
    secondary_type = types[1]["type"]["name"] if len(types) > 1 else ""

    sprite_path = SPRITE_DIR / f"{pokemon_id:04d}.png"
    download_sprite(sprite_url, sprite_path)

    return {
        "id": pokemon_id,
        "name": poke["name"],
        "primary_type": primary_type,
        "secondary_type": secondary_type,
        "color": species["color"]["name"],
        "shape": species["shape"]["name"] if species.get("shape") else "",
        "sprite_file": f"sprites/{sprite_path.name}",
    }


def main():
    SPRITE_DIR.mkdir(exist_ok=True)
    rows = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_pokemon, i): i for i in range(1, TOTAL + 1)}
        for i, future in enumerate(as_completed(futures), 1):
            pokemon_id = futures[future]
            result = future.result()
            if result:
                rows.append(result)
            if i % 50 == 0 or i == TOTAL:
                print(f"  {i}/{TOTAL} processed, {len(rows)} valid so far")

    rows.sort(key=lambda r: r["id"])

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} Pokémon saved to {CSV_OUT}")


if __name__ == "__main__":
    main()
