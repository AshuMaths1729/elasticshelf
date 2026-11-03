"""ElasticShelf MVP backend: FastAPI + simulator + ESL/POS WebSocket stream."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import (InventoryItem, InventorySyncIn, PricingEvaluateOut,
                     RestockIn, SimControl, SimStatus)
from .simulator import Simulator
from .store import Store

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (stdlib-only). Real environment wins; inline
    ` # comments` are stripped; missing file is silently skipped."""
    try:
        text = path.read_text()
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if " #" in val and not val.startswith(("'", '"')):
            val = val.split(" #", 1)[0].rstrip()
        val = val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="ElasticShelf MVP", version="0.1.0")
store = Store(store_id=os.getenv("STORE_ID", "ST-4092"))
sim = Simulator(store)
subscribers: Set[WebSocket] = set()
_loop_task = None


# -- startup data --------------------------------------------------------
def _reanchor_expiries(rows: List[InventoryItem]) -> List[InventoryItem]:
    """catalog.json stores absolute expiries from when it was generated, but
    the sim clock restarts at today 09:00 — a catalog generated days ago
    would start with most SKUs already expired. Shift every expiry by one
    delta so the generator's hours-to-expiry hold relative to sim_time."""
    from datetime import timedelta
    from scripts.generate_synthetic import CATALOG_SPEC
    spec_hrs = {sku: hrs for sku, *_, hrs in CATALOG_SPEC}
    deltas = sorted(sim.sim_time + timedelta(hours=spec_hrs[r.sku]) - r.expiration_timestamp
                    for r in rows if r.sku in spec_hrs)
    if not deltas:
        return rows
    delta = deltas[len(deltas) // 2]
    for r in rows:
        r.expiration_timestamp = r.expiration_timestamp + delta
    return rows


def _load_seed_catalog() -> None:
    catalog = DATA_DIR / "catalog.json"
    if catalog.exists():
        rows = [InventoryItem(**row) for row in json.loads(catalog.read_text())]
        for item in _reanchor_expiries(rows):
            store.upsert(item)
        return
    # Minimal fallback so the API works before scripts are run.
    from datetime import timedelta, timezone
    now = datetime.now(timezone.utc)
    for i, (sku, name, cat, price, cost, qty, hrs) in enumerate([
        ("ORG-CHK-01", "Organic Chicken Breast", "meat", 12.99, 8.50, 45, 72),
        ("BRR-MIX-02", "Mixed Berries 300g", "produce", 6.49, 3.20, 30, 30),
        ("MLK-OAT-03", "Oat Milk 1L", "dairy", 4.29, 2.40, 52, 96),
    ]):
        store.upsert(InventoryItem(sku=sku, name=name, category=cat,
                                   quantity_on_shelf=qty, base_price=price,
                                   current_price=price, cost_basis=cost,
                                   expiration_timestamp=now + timedelta(hours=hrs)))


@app.on_event("startup")
async def _startup() -> None:
    _load_seed_catalog()
    global _loop_task
    _loop_task = asyncio.create_task(_sim_loop())


async def _sim_loop() -> None:
    while True:
        await asyncio.sleep(sim.tick_seconds)
        if sim.running:
            try:
                result = await asyncio.to_thread(sim.tick)
                await _broadcast({"type": "tick", **result,
                                  "metrics": _metrics()})
            except Exception:
                pass


async def _broadcast(msg: dict) -> None:
    dead: List[WebSocket] = []
    payload = jsonable_encoder(msg)
    for ws in list(subscribers):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        subscribers.discard(ws)


def _metrics() -> dict:
    return {"revenue": store.revenue, "units_sold": store.units_sold,
            "units_wasted": store.units_wasted,
            "skus": len(store.items), "active_skus": sim.active_skus(),
            "store_open": sim.is_open, "open_hour": sim.open_hour,
            "close_hour": sim.close_hour,
            "revenue_baseline": sim.baseline["revenue"],
            "units_wasted_baseline": sim.baseline["units_wasted"],
            "profit": sim.net_profit(), "profit_baseline": sim.baseline_profit(),
            "markdown_cost": store.markdown_cost,
            "waste_cost": store.waste_cost,
            "waste_cost_baseline": sim.baseline["waste_cost"],
            "history": sim.history,
            "sim_time": sim.sim_time.isoformat(), "tick": sim.tick_count,
            "running": sim.running}


# -- guide endpoints ------------------------------------------------------
@app.post("/api/v1/inventory/sync")
async def inventory_sync(body: InventorySyncIn):
    for row in body.items:
        prev = store.items.get(row.sku)
        store.upsert(InventoryItem(
            sku=row.sku, name=row.name, category=row.category or (prev.category if prev else "general"),
            quantity_on_shelf=row.quantity_on_shelf, base_price=row.base_price,
            current_price=prev.current_price if prev else row.base_price,
            cost_basis=row.cost_basis, expiration_timestamp=row.expiration_timestamp,
            image_url=row.image_url or (prev.image_url if prev else None),
            last_reasoning=prev.last_reasoning if prev else None))
        if prev and prev.quantity_on_shelf > 0 and row.quantity_on_shelf == 0:
            await asyncio.to_thread(sim.note_depletion, row.sku, prev.quantity_on_shelf)
        if row.quantity_on_shelf > 0 and (not prev or prev.quantity_on_shelf == 0):
            sim.depletion.pop(row.sku, None)
            sim.stocked_tick[row.sku] = sim.tick_count
        if row.quantity_on_shelf > 0:
            sim.stock_peak[row.sku] = max(sim.stock_peak.get(row.sku, 0),
                                          row.quantity_on_shelf)
        sim.baseline_upsert(row.sku, row.quantity_on_shelf, row.base_price,
                            row.expiration_timestamp, row.cost_basis)
    return {"status": "success", "store_id": body.store_id, "items_synced": len(body.items)}


@app.post("/api/v1/pricing/evaluate", response_model=PricingEvaluateOut)
async def pricing_evaluate():
    result = await asyncio.to_thread(sim.evaluate_pricing)
    await _broadcast({"type": "pricing", **result, "metrics": _metrics()})
    return result


# -- sim + read endpoints --------------------------------------------------
@app.get("/api/v1/inventory")
async def inventory_list():
    return {"store_id": store.store_id, "items": [i.dict() for i in store.all()]}


@app.get("/api/v1/context")
async def context_now():
    snap = sim.snapshots[-1].dict() if sim.snapshots else None
    return {"sim_time": sim.sim_time.isoformat(), "footfall_index": sim.footfall(),
            "temperature_c": sim.temperature_c, "precipitation_mm": sim.precipitation_mm,
            "event": sim.current_event(), "last_snapshot": snap,
            "store_open": sim.is_open, "weather_source": sim.weather_source,
            "engine": sim.fx.backend, "llm_provider": sim.ctx.provider,
            "llm_model": sim.ctx.model}


@app.get("/api/v1/sim/status", response_model=SimStatus)
async def sim_status():
    return SimStatus(running=sim.running, tick=sim.tick_count, sim_time=sim.sim_time,
                     tick_seconds=sim.tick_seconds, store_id=store.store_id,
                     skus=len(store.items), active_skus=sim.active_skus(),
                     store_open=sim.is_open, open_hour=sim.open_hour,
                     close_hour=sim.close_hour, revenue=store.revenue,
                     units_sold=store.units_sold, units_wasted=store.units_wasted,
                     revenue_baseline=sim.baseline["revenue"],
                     units_wasted_baseline=sim.baseline["units_wasted"],
                     profit=sim.net_profit(), profit_baseline=sim.baseline_profit(),
                     markdown_cost=store.markdown_cost,
                     waste_cost=store.waste_cost,
                     waste_cost_baseline=sim.baseline["waste_cost"],
                     history=sim.history)


@app.post("/api/v1/sim/start")
async def sim_start():
    sim.start()
    return {"running": True}


@app.post("/api/v1/sim/stop")
async def sim_stop():
    sim.stop()
    return {"running": False}


@app.post("/api/v1/sim/tick")
async def sim_tick_once():
    result = await asyncio.to_thread(sim.tick)
    await _broadcast({"type": "tick", **result, "metrics": _metrics()})
    return result


@app.post("/api/v1/sim/control")
async def sim_control(body: SimControl):
    if body.tick_seconds:
        sim.tick_seconds = max(0.2, min(30.0, body.tick_seconds))
    if body.hours_per_tick:
        sim.hours_per_tick = max(0.25, min(12.0, body.hours_per_tick))
    if body.open_hour is not None:
        sim.open_hour = max(0, min(23, body.open_hour))
    if body.close_hour is not None:
        sim.close_hour = max(0, min(23, body.close_hour))
    return {"tick_seconds": sim.tick_seconds, "hours_per_tick": sim.hours_per_tick,
            "open_hour": sim.open_hour, "close_hour": sim.close_hour,
            "store_open": sim.is_open}


@app.post("/api/v1/sim/reset")
async def sim_reset():
    sim.reset()
    store.items.clear()
    _load_seed_catalog()
    await _broadcast({"type": "reset", "metrics": _metrics()})
    return {"status": "reset"}


@app.get("/api/v1/pos/feed")
async def pos_feed():
    return {"checkouts": sim.pos_feed[-20:][::-1]}


@app.get("/api/v1/forecast")
async def forecast():
    return {"tick": sim.tick_count, "sim_time": sim.sim_time.isoformat(),
            "window_ticks": max(1, min(24, sim.tick_count)),
            "items": sim.forecast()}


@app.post("/api/v1/inventory/restock")
async def inventory_restock(body: RestockIn):
    try:
        row = sim.restock(body.sku, body.quantity, body.expiration_timestamp)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown sku {body.sku}")
    return {"status": "success", **jsonable_encoder(row)}


# -- websocket --------------------------------------------------------------
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    subscribers.add(ws)
    try:
        await ws.send_json(jsonable_encoder({"type": "hello", "metrics": _metrics(),
                                             "items": [i.dict() for i in store.all()]}))
        while True:
            await ws.receive_text()  # keepalive; client may ping
    except WebSocketDisconnect:
        pass
    finally:
        subscribers.discard(ws)


# -- frontend -----------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"service": "ElasticShelf MVP", "docs": "/docs"}
