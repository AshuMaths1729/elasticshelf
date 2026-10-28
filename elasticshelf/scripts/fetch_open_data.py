"""Fetch open-source data layers (graceful offline fallback).

Sources (all free, no API key):
  1. Open Food Facts API  - real product names/barcodes/images for catalog enrichment
  2. Open-Meteo API       - real hourly weather (seed for the simulator's weather walk)

Writes: data/open_catalog.json, data/weather.json
Run:  python -m scripts.fetch_open_data   (from elasticshelf/)
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

QUERIES = ["chicken breast", "greek yogurt", "sourdough", "salmon", "orange juice"]


def fetch_openfoodfacts(page_size: int = 6):
    import urllib.request

    out = []
    for q in QUERIES:
        url = ("https://world.openfoodfacts.org/cgi/search.pl?search_terms="
               + q.replace(" ", "+") + f"&search_simple=1&action=process&json=1&page_size={page_size}")
        try:
            with urllib.request.urlopen(url, timeout=12) as r:
                payload = json.loads(r.read().decode("utf-8", "replace"))
        except Exception as exc:
            print(f"  [skip] OpenFoodFacts '{q}': {exc}")
            continue
        for p in payload.get("products", [])[:page_size]:
            out.append({"query": q,
                        "product_name": p.get("product_name") or p.get("generic_name"),
                        "brands": p.get("brands"), "code": p.get("code"),
                        "image_url": p.get("image_url"),
                        "categories": p.get("categories")})
    return [o for o in out if o["product_name"]]


def fetch_weather(lat: float = 37.77, lon: float = -122.41):
    import urllib.request

    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&hourly=temperature_2m,precipitation,weathercode&forecast_days=3&timezone=auto")
    with urllib.request.urlopen(url, timeout=12) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def main() -> None:
    DATA.mkdir(exist_ok=True)
    print("Fetching Open Food Facts sample...")
    try:
        products = fetch_openfoodfacts()
    except Exception as exc:
        print(f"  OpenFoodFacts failed: {exc}")
        products = []
    (DATA / "open_catalog.json").write_text(json.dumps(products, indent=2))
    print(f"  saved {len(products)} products -> data/open_catalog.json")

    print("Fetching Open-Meteo weather...")
    try:
        weather = fetch_weather()
        (DATA / "weather.json").write_text(json.dumps(weather, indent=2))
        print("  saved hourly forecast -> data/weather.json")
    except Exception as exc:
        print(f"  Open-Meteo failed (offline?): {exc}")
        (DATA / "weather.json").write_text(json.dumps({"fallback": True, "note": str(exc)}))


if __name__ == "__main__":
    main()
