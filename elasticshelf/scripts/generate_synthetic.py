"""Generate synthetic grocery catalog + 10k transactions (offline, seeded).

Writes:
  data/catalog.json      - ~24 perishable SKUs with expiry/stock/cost/price
  data/transactions.csv  - 10,000 POS transactions for training / demo replay
Modalities covered: structured POS, expiry time-series, category text.
Run:  python -m scripts.generate_synthetic   (from elasticshelf/)
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

CATALOG_SPEC = [
    # sku, name, category, base_price, cost, qty, hours_to_expiry
    ("ORG-CHK-01", "Organic Chicken Breast", "meat", 12.99, 8.50, 45, 72),
    ("GRB-BEF-02", "Ground Beef 500g", "meat", 9.49, 6.10, 38, 48),
    ("SLM-FIL-03", "Salmon Fillet", "meat", 15.99, 10.20, 22, 36),
    ("BRR-MIX-04", "Mixed Berries 300g", "produce", 6.49, 3.20, 30, 30),
    ("AVO-HAS-05", "Hass Avocado (4pk)", "produce", 5.99, 2.90, 55, 60),
    ("SPN-BAB-06", "Baby Spinach 200g", "produce", 3.99, 1.80, 40, 42),
    ("MLK-OAT-07", "Oat Milk 1L", "dairy", 4.29, 2.40, 52, 96),
    ("YGR-GRK-08", "Greek Yogurt 500g", "dairy", 5.49, 2.90, 44, 84),
    ("CHS-CHD-09", "Cheddar Block 400g", "dairy", 7.29, 4.10, 33, 110),
    ("BRD-SRD-10", "Sourdough Loaf", "bakery", 4.99, 2.10, 28, 24),
    ("CRS-BTR-11", "Butter Croissant (4pk)", "bakery", 5.49, 2.30, 24, 20),
    ("BAG-BLU-12", "Blueberry Bagels (6pk)", "bakery", 4.79, 2.00, 26, 40),
    ("SUS-SET-13", "Sushi Set (12pc)", "prepared", 11.99, 6.50, 18, 14),
    ("SLD-CSR-14", "Caesar Salad Bowl", "prepared", 8.49, 4.20, 20, 18),
    ("PST-PRM-15", "Pasta Primavera Tray", "prepared", 9.99, 5.10, 16, 26),
    ("OJ-FRS-16", "Fresh Orange Juice 1L", "drinks", 5.99, 2.80, 36, 52),
    ("KMB-GNG-17", "Ginger Kombucha 750ml", "drinks", 4.49, 2.10, 30, 90),
    ("CHP-KE-18", "Kale Chips 120g", "snacks", 3.79, 1.60, 48, 100),
    ("HUM-CLS-19", "Classic Hummus 300g", "snacks", 4.19, 1.90, 42, 66),
    ("EGG-FRE-20", "Free-Range Eggs (12)", "dairy", 6.99, 3.80, 50, 104),
    ("TOM-VIN-21", "Vine Tomatoes 500g", "produce", 3.49, 1.50, 58, 46),
    ("MUS-SHI-22", "Shiitake Mushrooms 200g", "produce", 4.99, 2.40, 25, 38),
    ("PRW-GRL-23", "Garlic Prawns 300g", "meat", 13.49, 8.80, 15, 28),
    ("CAK-CRT-24", "Carrot Cake Slice", "bakery", 3.99, 1.70, 22, 16),
]


def build_catalog(now: datetime):
    items = []
    for sku, name, cat, price, cost, qty, hrs in CATALOG_SPEC:
        items.append({
            "sku": sku, "name": name, "category": cat,
            "quantity_on_shelf": qty, "base_price": price,
            "current_price": price, "cost_basis": cost,
            "expiration_timestamp": (now + timedelta(hours=hrs)).isoformat(),
        })
    return items


def gen_transactions(catalog, n: int = 10_000, seed: int = 7):
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n):
        item = rng.choice(catalog)
        # Older transactions spread over past 14 days.
        ts = now - timedelta(hours=rng.uniform(0, 14 * 24))
        # Elastic quantity: discounts move more units.
        discount = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.25, 0.40])
        price = round(item["base_price"] * (1 - discount), 2)
        # Realistic grocery elasticity (~2): discounts lift demand modestly.
        qty = max(1, int(rng.expovariate(1 / (1.5 + discount * 2.5))))
        rows.append({
            "txn_id": f"TXN-{i:06d}", "timestamp": ts.isoformat(),
            "sku": item["sku"], "category": item["category"],
            "qty": qty, "unit_price": price,
            "discount_pct": round(discount * 100, 1),
            "footfall_index": round(rng.uniform(0.3, 1.6), 2),
            "temperature_c": round(rng.uniform(2, 34), 1),
        })
    return rows


def main() -> None:
    DATA.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    catalog = build_catalog(now)
    (DATA / "catalog.json").write_text(json.dumps(catalog, indent=2))
    txns = gen_transactions(catalog)
    with open(DATA / "transactions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(txns[0].keys()))
        w.writeheader()
        w.writerows(txns)
    print(f"Wrote {len(catalog)} SKUs -> data/catalog.json")
    print(f"Wrote {len(txns)} transactions -> data/transactions.csv")


if __name__ == "__main__":
    main()
