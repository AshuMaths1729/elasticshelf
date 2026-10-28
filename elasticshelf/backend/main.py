"""ElasticShelf MVP backend: FastAPI + simulator + ESL/POS WebSocket stream."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import (InventoryItem, InventorySyncIn, PricingEvaluateOut,
                     SimControl, SimStatus)
from .simulator import Simulator
from .store import Store

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="ElasticShelf MVP", version="0.1.0")
store = Store(store_id=os.getenv("STORE_ID", "ST-4092"))
sim = Simulator(store)
subscribers: Set[WebSocket] = set()
_loop_task = None


# -- startup data --------------------------------------------------------
def _load_seed_catalog() -> None:
    catalog = DATA_DIR / "catalog.json"
    if catalog.exists():
        for row in json.loads(catalog.read_text()):
            store.upsert(InventoryItem(**row))
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
            result = sim.tick()
            await _broadcast({"type": "tick", **result,
                              "metrics": _metrics()})


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
            "skus": len(store.items),
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
    return {"status": "success", "store_id": body.store_id, "items_synced": len(body.items)}


@app.post("/api/v1/pricing/evaluate", response_model=PricingEvaluateOut)
async def pricing_evaluate():
    result = sim.evaluate_pricing()
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
            "engine": sim.fx.backend, "llm_provider": sim.ctx.provider,
            "llm_model": sim.ctx.model}


@app.get("/api/v1/sim/status", response_model=SimStatus)
async def sim_status():
    return SimStatus(running=sim.running, tick=sim.tick_count, sim_time=sim.sim_time,
                     tick_seconds=sim.tick_seconds, store_id=store.store_id,
                     skus=len(store.items), revenue=store.revenue,
                     units_sold=store.units_sold, units_wasted=store.units_wasted)


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
    result = sim.tick()
    await _broadcast({"type": "tick", **result, "metrics": _metrics()})
    return result


@app.post("/api/v1/sim/control")
async def sim_control(body: SimControl):
    if body.tick_seconds:
        sim.tick_seconds = max(0.2, min(30.0, body.tick_seconds))
    if body.hours_per_tick:
        sim.hours_per_tick = max(0.25, min(12.0, body.hours_per_tick))
    return {"tick_seconds": sim.tick_seconds, "hours_per_tick": sim.hours_per_tick}


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
