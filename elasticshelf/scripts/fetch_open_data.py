"""Fetch open-source data layers (graceful offline fallback).

Sources (all free, no API key):
  1. Open Food Facts API  - real product names/barcodes/images for catalog enrichment
  2. Open-Meteo API       - real hourly weather for the store location; the
      simulator steps through this series (falls back to a random walk offline)

Writes: data/open_catalog.json, data/weather.json
Run:  python -m scripts.fetch_open_data [--lat 37.77 --lon -122.41]   (from elasticshelf/)
      STORE_LAT / STORE_LON env vars provide the defaults.
"""
from __future__ import annotations

import argparse
import json
import os
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


def _has_usable_forecast(path: Path) -> bool:
    try:
        hourly = json.loads(path.read_text()).get("hourly", {})
        return bool(hourly.get("temperature_2m")) and \
            len(hourly["temperature_2m"]) == len(hourly.get("precipitation", []))
    except Exception:
        return False


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float,
                    default=float(os.getenv("STORE_LAT", "37.77")))
    ap.add_argument("--lon", type=float,
                    default=float(os.getenv("STORE_LON", "-122.41")))
    args = ap.parse_args(argv)
    DATA.mkdir(exist_ok=True)
    print("Fetching Open Food Facts sample...")
    try:
        products = fetch_openfoodfacts()
    except Exception as exc:
        print(f"  OpenFoodFacts failed: {exc}")
        products = []
    (DATA / "open_catalog.json").write_text(json.dumps(products, indent=2))
    print(f"  saved {len(products)} products -> data/open_catalog.json")

    print(f"Fetching Open-Meteo weather for ({args.lat}, {args.lon})...")
    try:
        weather = fetch_weather(args.lat, args.lon)
        (DATA / "weather.json").write_text(json.dumps(weather, indent=2))
        print("  saved hourly forecast -> data/weather.json")
    except Exception as exc:
        print(f"  Open-Meteo failed (offline?): {exc}")
        if _has_usable_forecast(DATA / "weather.json"):
            print("  keeping existing data/weather.json")
        else:
            (DATA / "weather.json").write_text(json.dumps({"fallback": True, "note": str(exc)}))


if __name__ == "__main__":
    main()
