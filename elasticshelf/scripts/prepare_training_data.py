"""Prepare DQN training data: real download + calibrated synthetic fit.

1. Tries to download the UCI Online Retail dataset (real transactions) and
   derives an hourly footfall curve + quantity stats from it.
2. Always fits per-category demand parameters (base_qty, elasticity) from
   data/transactions.csv via least squares on (discount -> qty).

Writes: data/demand_model.json (+ data/uci_online_retail.xlsx when fetched)
Run:  python -m scripts.prepare_training_data   (from elasticshelf/)
"""
from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

UCI_CANDIDATES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
]


def try_download_uci() -> Path | None:
    import urllib.request

    for url in UCI_CANDIDATES:
        try:
            print(f"Trying {url} ...")
            tmp = DATA / "_uci_download.tmp"
            urllib.request.urlretrieve(url, tmp)
            if url.endswith(".zip") or zipfile.is_zipfile(tmp):
                with zipfile.ZipFile(tmp) as z:
                    xlsx = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
                    target = DATA / "uci_online_retail.xlsx"
                    with z.open(xlsx) as src, open(target, "wb") as dst:
                        dst.write(src.read())
            else:
                target = DATA / "uci_online_retail.xlsx"
                tmp.rename(target)
            if tmp.exists():
                tmp.unlink()
            print(f"  saved -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
            return target
        except Exception as exc:
            print(f"  failed: {exc}")
    return None


def uci_stats(path: Path, sample_every: int = 5) -> dict | None:
    """Hourly invoice curve + quantity stats from the UCI workbook (sampled)."""
    try:
        import openpyxl
    except ImportError:
        print("  openpyxl missing, skipping UCI parse")
        return None
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    hourly = defaultdict(int)
    qtys: list[float] = []
    n = 0
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True)):
        if i % sample_every:
            continue
        try:
            qty = float(row[3] or 0)
            price = float(row[5] or 0)
            dt = row[4]
        except (TypeError, ValueError, IndexError):
            continue
        if qty <= 0 or price <= 0:
            continue
        n += 1
        qtys.append(qty)
        if isinstance(dt, datetime):
            hourly[dt.hour] += 1
    wb.close()
    if n < 100:
        return None
    curve = np.array([hourly.get(h, 0) for h in range(24)], dtype=float)
    curve = curve / max(curve.mean(), 1e-6)
    print(f"  parsed {n} sampled UCI rows")
    return {"hourly_curve": [round(float(v), 3) for v in curve],
            "qty_mean": round(float(np.mean(qtys)), 3),
            "qty_std": round(float(np.std(qtys)), 3)}


def fit_elasticity() -> tuple[dict, list[float]]:
    rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    hours = defaultdict(list)
    with open(DATA / "transactions.csv") as f:
        for r in csv.DictReader(f):
            d = float(r["discount_pct"]) / 100.0
            rows[r["category"]].append((d, float(r["qty"])))
            try:
                h = datetime.fromisoformat(r["timestamp"]).hour
                hours[h].append(float(r["qty"]))
            except ValueError:
                pass
    cats = {}
    for cat, pts in rows.items():
        d = np.array([p[0] for p in pts])
        q = np.array([p[1] for p in pts])
        m, b = np.polyfit(d, q, 1)
        base = float(np.clip(b, 0.3, 8.0))
        cats[cat] = {"base_qty": round(base, 3),
                     "elasticity": round(float(np.clip(m / max(b, 1e-6), 0.0, 10.0)), 3)}
    curve = np.array([np.mean(hours.get(h, [1.0])) for h in range(24)], dtype=float)
    curve = curve / max(curve.mean(), 1e-6)
    return cats, [round(float(v), 3) for v in curve]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    if not (DATA / "transactions.csv").exists():
        raise SystemExit("data/transactions.csv missing — run scripts.generate_synthetic first")
    print("Fitting elasticity from data/transactions.csv ...")
    cats, synth_curve = fit_elasticity()
    for cat, p in sorted(cats.items()):
        print(f"  {cat:10s} base_qty={p['base_qty']:<6} elasticity={p['elasticity']}")
    model: dict = {"categories": cats, "hourly_curve": synth_curve,
                   "sources": {"elasticity": "synthetic:transactions.csv",
                               "hourly_curve": "synthetic:transactions.csv"}}
    uci_path = DATA / "uci_online_retail.xlsx"
    if not uci_path.exists():
        dl = try_download_uci()
        if dl is not None:
            uci_path = dl
    if uci_path.exists():
        stats = uci_stats(uci_path)
        if stats:
            model["hourly_curve"] = stats["hourly_curve"]
            model["uci_qty"] = {"mean": stats["qty_mean"], "std": stats["qty_std"]}
            model["sources"]["hourly_curve"] = "real:UCI Online Retail"
    (DATA / "demand_model.json").write_text(json.dumps(model, indent=2))
    print("Wrote data/demand_model.json")


if __name__ == "__main__":
    main()
