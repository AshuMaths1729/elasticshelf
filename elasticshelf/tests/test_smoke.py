"""Focused regression tests for the ElasticShelf MVP.

Covers the fixed bug (WebSocket payloads must JSON-serialize datetimes) plus
the two guide endpoints. Runs with plain python, no test framework needed:

    python -m tests.test_smoke        (from elasticshelf/, venv active)
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def _client():
    c = TestClient(app)
    c.__enter__()
    return c


def test_ws_hello_serializes():
    c = _client()
    try:
        with c.websocket_connect("/ws/stream") as ws:
            hello = ws.receive_json()
        assert hello["type"] == "hello", hello
        assert len(hello["items"]) >= 3, "seed catalog missing"
        assert isinstance(hello["items"][0]["expiration_timestamp"], str)
    finally:
        c.__exit__(None, None, None)


def test_tick_and_pricing_loop():
    c = _client()
    try:
        t = c.post("/api/v1/sim/tick").json()
        assert t["tick"] >= 1
        assert "checkouts" in t and "pricing" in t
        p = c.post("/api/v1/pricing/evaluate").json()
        assert p["status"] == "success"
        assert p["hardware_sync_status"] == "acknowledged"
        audits = t["pricing"]["audit_log"] + p["audit_log"]
        assert audits, "expected at least one price push in tick+evaluate"
        for a in audits:
            assert "llm_reasoning" in a, a
            assert "llama_reasoning" not in a, a
    finally:
        c.__exit__(None, None, None)


def test_inventory_sync_roundtrip():
    c = _client()
    try:
        r = c.post("/api/v1/inventory/sync", json={
            "store_id": "ST-4092",
            "items": [{"sku": "TST-99", "name": "Test Beans",
                       "quantity_on_shelf": 7, "base_price": 2.99,
                       "cost_basis": 1.50,
                       "expiration_timestamp": "2026-10-30T23:59:59Z"}]})
        assert r.json()["items_synced"] == 1
        skus = [i["sku"] for i in c.get("/api/v1/inventory").json()["items"]]
        assert "TST-99" in skus
    finally:
        c.__exit__(None, None, None)


def test_elasticity_price_bounds():
    import random

    from backend.elasticity import ElasticityEngine
    e = ElasticityEngine()
    rng = random.Random(0)
    for _ in range(200):
        base = rng.uniform(2, 16)
        cost = base * rng.uniform(0.45, 0.7)
        cur = rng.uniform(cost, base * 1.5)
        p, a, _ = e.decide(cur, base, cost, rng.randint(0, 60),
                           rng.uniform(0, 120), rng.uniform(-1, 1))
        assert cost - 1e-9 <= p <= base * 1.5 + 1e-9, (p, cost, base)
        assert a in (0, 1, 2, 3), a


if __name__ == "__main__":
    test_ws_hello_serializes()
    test_tick_and_pricing_loop()
    test_inventory_sync_roundtrip()
    test_elasticity_price_bounds()
    print("ALL SMOKE TESTS PASSED")
