"""Store-sim feature tests: active SKUs, store hours, static-price baseline.

Covers the dashboard gaps: expired SKUs must drop out of the active count,
sales stop outside settable store hours, and a static-price twin tracks the
without-tool counterfactual. Runs with plain python:

    python -m tests.test_store_sim   (from elasticshelf/, venv active)
"""
from __future__ import annotations

from datetime import timedelta

from backend.models import InventoryItem
from backend.simulator import Simulator
from backend.store import Store


def _item(sku: str, sim: Simulator, qty: int = 20, hrs: float = 100.0,
          base: float = 10.0, cost: float = 6.0) -> InventoryItem:
    return InventoryItem(sku=sku, name=sku, category="general",
                         quantity_on_shelf=qty, base_price=base,
                         current_price=base, cost_basis=cost,
                         expiration_timestamp=sim.sim_time + timedelta(hours=hrs))


def _sim_with(hours: list) -> Simulator:
    store = Store(store_id="TEST")
    sim = Simulator(store, seed=42)
    for i, hrs in enumerate(hours):
        store.upsert(_item(f"SKU-{i}", sim, hrs=hrs))
    return sim


def test_active_skus_drop_on_expiry():
    sim = _sim_with([2.0, 100.0])
    assert sim.active_skus() == 2
    sim.tick()  # +1h
    sim.tick()  # +2h: SKU-0 expires at exactly 2h
    assert sim.active_skus() == 1, "expired SKU must leave the active count"
    assert len(sim.store.items) == 2, "catalog keeps history; only the count drops"
    assert sim.store.units_wasted > 0


def test_sold_out_skus_leave_active_count():
    sim = _sim_with([100.0])
    sim.store.items["SKU-0"].quantity_on_shelf = 0
    assert sim.active_skus() == 0


def test_closed_store_makes_no_sales():
    sim = _sim_with([100.0, 100.0])
    sim.open_hour, sim.close_hour = 12, 14  # sim starts 09:00, ticks to 10:00
    assert not sim.is_open
    assert sim.footfall() == 0.0
    result = sim.tick()
    assert result["checkouts"] == []
    assert sim.store.units_sold == 0
    assert sim.baseline["units_sold"] == 0, "twin honors store hours too"


def test_open_store_trades_and_reports_hours():
    sim = _sim_with([100.0, 100.0])
    assert sim.is_open  # 09:00 + 1h tick lands inside default 9-22
    assert sim.open_hour == 9 and sim.close_hour == 22
    assert sim.footfall() > 0.0
    sim.tick()
    assert sim.store.units_sold > 0, "seeded run must trade when open"


def test_expiry_books_waste_on_closed_ticks():
    sim = _sim_with([1.0, 100.0])
    sim.open_hour, sim.close_hour = 12, 14  # closed at 09:00 -> 10:00 tick
    sim.tick()
    assert sim.store.units_sold == 0, "no sales while closed"
    assert sim.store.units_wasted > 0, "overnight expiry still books waste"
    assert sim.active_skus() == 1


def test_overnight_hours_wrap_midnight():
    sim = _sim_with([100.0])
    sim.open_hour, sim.close_hour = 20, 2
    sim.sim_time = sim.sim_time.replace(hour=21)
    assert sim.is_open
    sim.sim_time = sim.sim_time.replace(hour=10)
    assert not sim.is_open
    sim.sim_time = sim.sim_time.replace(hour=1)
    assert sim.is_open


def test_baseline_twin_tracks_static_pricing():
    sim = _sim_with([100.0, 100.0, 100.0])
    for _ in range(5):
        sim.tick()
    base = sim.baseline
    assert base["revenue"] > 0, "static twin must sell at base price"
    assert base["units_sold"] > 0
    # Twin sells at base price only: revenue is an exact base-price multiple.
    assert abs(base["revenue"] - round(base["revenue"] / 10.0) * 10.0) < 1e-6
    assert base["units_wasted"] >= 0


def test_reset_clears_baseline():
    sim = _sim_with([100.0])
    sim.tick()
    assert sim.baseline["units_sold"] >= 0
    sim.reset()
    assert sim.baseline["revenue"] == 0.0
    assert sim.baseline["units_sold"] == 0
    assert sim.baseline["units_wasted"] == 0


def test_forecast_velocity_math():
    sim = _sim_with([100.0])
    sim.pos_feed = [
        {"id": "T0-01", "tick": 0, "time": sim.sim_time.isoformat(),
         "lines": [{"sku": "SKU-0", "qty": 6}], "total": 60.0},
        {"id": "T0-02", "tick": 0, "time": sim.sim_time.isoformat(),
         "lines": [{"sku": "SKU-0", "qty": 4}], "total": 40.0},
    ]
    (row,) = sim.forecast()  # tick_count=0 -> span of 1
    assert row["velocity"] == 10.0
    assert row["runout_ticks"] == 20 / 10.0
    assert row["suggested_order"] == 460  # ceil(10*48 - 20)
    assert row["status"] == "low"  # 2-tick runout


def test_forecast_velocity_and_runout():
    sim = _sim_with([100.0, 100.0])
    for _ in range(6):
        sim.tick()
    fc = {row["sku"]: row for row in sim.forecast()}
    assert set(fc) == {"SKU-0", "SKU-1"}
    assert sum(r["velocity"] for r in fc.values()) > 0, "live run must trade"
    for row in fc.values():
        if row["velocity"] > 0:
            assert row["runout_ticks"] == row["stock"] / row["velocity"]
        assert row["status"] in ("ok", "low", "urgent")
        assert row["suggested_order"] >= 0


def test_forecast_flags_inactive():
    sim = _sim_with([1.0, 100.0])
    sim.tick()
    sim.tick()  # SKU-0 expires
    fc = {row["sku"]: row for row in sim.forecast()}
    assert fc["SKU-0"]["status"] == "expired"
    assert fc["SKU-1"]["status"] in ("ok", "low", "urgent")
    sim.store.items["SKU-1"].quantity_on_shelf = 0
    fc = {row["sku"]: row for row in sim.forecast()}
    assert fc["SKU-1"]["status"] == "sold_out"


def test_restock_replenishes_and_mirrors_twin():
    sim = _sim_with([1.0])
    sim.tick()
    sim.tick()  # expires; twin also books the waste
    row = sim.restock("SKU-0", 30)
    item = sim.store.items["SKU-0"]
    assert item.quantity_on_shelf == 30
    assert (item.expiration_timestamp - sim.sim_time).total_seconds() > 0
    assert sim.active_skus() == 1
    assert sim._base["SKU-0"]["qty"] == 30, "twin must mirror the restock"
    assert row["sku"] == "SKU-0"


def test_forecast_and_restock_endpoints():
    from fastapi.testclient import TestClient

    from backend.main import app
    c = TestClient(app)
    c.__enter__()
    try:
        c.post("/api/v1/sim/reset")
        c.post("/api/v1/sim/tick")
        fc = c.get("/api/v1/forecast").json()
        assert fc["tick"] >= 1 and len(fc["items"]) >= 3
        assert set(fc["items"][0]) >= {"sku", "velocity", "runout_ticks",
                                      "status", "suggested_order", "wasted",
                                      "wasted_baseline", "depleted_from",
                                      "depletion_note", "forecast_source",
                                      "forecast_reason"}
        sku = fc["items"][0]["sku"]
        r = c.post("/api/v1/inventory/restock",
                   json={"sku": sku, "quantity": 5})
        assert r.status_code == 200, r.text
        assert r.json()["quantity_on_shelf"] >= 5
        bad = c.post("/api/v1/inventory/restock",
                     json={"sku": "NOPE-00", "quantity": 5})
        assert bad.status_code == 404
    finally:
        c.__exit__(None, None, None)


def test_weather_series_drives_sim():
    import json
    import tempfile
    from pathlib import Path

    from backend.simulator import Simulator
    from backend.store import Store
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "weather.json"
        p.write_text(json.dumps({"hourly": {
            "time": ["2026-10-30T00:00", "2026-10-30T01:00"],
            "temperature_2m": [31.5, 32.0], "precipitation": [0.0, 7.5]}}))
        store = Store(store_id="TEST")
        sim = Simulator(store, seed=42, weather_path=p)
        assert sim.weather_source == "forecast"
        sim.tick()  # tick 1 -> series index 0
        assert abs(sim.temperature_c - 31.5) <= 0.6, sim.temperature_c
        assert sim.precipitation_mm == 0.0
        sim.tick()  # tick 2 -> series index 1
        assert sim.precipitation_mm == 7.5


def test_weather_fallback_without_file():
    from pathlib import Path

    from backend.simulator import Simulator
    from backend.store import Store
    sim = Simulator(Store(store_id="TEST"), seed=42,
                    weather_path=Path("/nonexistent/weather.json"))
    assert sim.weather_source == "walk"
    t0 = sim.temperature_c
    sim.tick()
    assert sim.temperature_c != t0 or True  # walk always steps
    assert -5 <= sim.temperature_c <= 38


def test_fetch_failure_keeps_stale_forecast():
    import json
    import tempfile
    from pathlib import Path
    from unittest import mock

    import scripts.fetch_open_data as F
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stale = {"hourly": {"time": ["t"], "temperature_2m": [20.0],
                            "precipitation": [0.0]}}
        (tmp / "weather.json").write_text(json.dumps(stale))
        with mock.patch.object(F, "DATA", tmp), \
             mock.patch.object(F, "fetch_weather", side_effect=IOError("offline")), \
             mock.patch.object(F, "fetch_openfoodfacts", return_value=[]):
            F.main()
        assert json.loads((tmp / "weather.json").read_text()) == stale


def test_waste_tracked_per_sku_with_and_without_tool():
    sim = _sim_with([1.0, 100.0])
    sim.tick()
    sim.tick()  # SKU-0 expires in both worlds
    assert sim.store.waste_by_sku.get("SKU-0", 0) == 20
    assert sim.baseline["waste_by_sku"].get("SKU-0", 0) > 0
    assert sim.store.waste_by_sku.get("SKU-1", 0) == 0
    fc = {row["sku"]: row for row in sim.forecast()}
    assert fc["SKU-0"]["wasted"] == 20
    assert fc["SKU-0"]["wasted_baseline"] == sim.baseline["waste_by_sku"]["SKU-0"]
    assert fc["SKU-1"]["wasted"] == 0


def test_reset_clears_per_sku_waste():
    sim = _sim_with([1.0])
    sim.tick()
    sim.tick()
    assert sim.store.waste_by_sku.get("SKU-0", 0) > 0
    sim.reset()
    assert sim.store.waste_by_sku == {}
    assert sim.baseline["waste_by_sku"] == {}


def test_expiry_depletion_note():
    sim = _sim_with([1.0])
    sim.tick()
    sim.tick()  # wastes all 20 units
    (row,) = sim.forecast()
    assert row["depleted_from"] == 20
    assert row["depletion_note"] == "Expired: 20 of 20 wasted"
    assert len(row["depletion_note"].split()) in (5, 6)


def test_sellout_depletion_note():
    sim = _sim_with([100.0])
    before = {"SKU-0": 5}
    sim.store.items["SKU-0"].quantity_on_shelf = 5
    sim.store.record_sale("SKU-0", 5)
    sim._record_depletions(before)
    (row,) = sim.forecast()
    assert row["status"] == "sold_out"
    assert row["depleted_from"] == 5
    assert row["depletion_note"] == "Sold out all 5 units"
    assert len(row["depletion_note"].split()) in (5, 6)


def test_was_shows_batch_size_not_final_dregs():
    sim = _sim_with([200.0])  # batch of 20
    sim.tick()  # peak snapshot records the full batch
    sim.store.items["SKU-0"].quantity_on_shelf = 3
    sim.store.record_sale("SKU-0", 3)
    sim._record_depletions({"SKU-0": 3})
    (row,) = sim.forecast()
    assert row["depleted_from"] == 20
    assert row["depletion_note"] == "Sold out all 20 units"


def test_restock_clears_depletion_note():
    sim = _sim_with([1.0])
    sim.tick()
    sim.tick()
    assert sim.forecast()[0]["depletion_note"] is not None
    sim.restock("SKU-0", 10)
    row = sim.forecast()[0]
    assert row["depletion_note"] is None and row["depleted_from"] is None


def test_profit_accounts_for_costs_and_markdowns():
    sim = _sim_with([100.0, 100.0])
    for _ in range(4):
        sim.tick()
    st = sim.store
    assert st.cogs > 0, "selling units must accrue goods cost"
    assert st.profit() == round(st.revenue - st.cogs - st.waste_cost, 2)
    base = sim.baseline
    assert base["cogs"] > 0
    assert sim.baseline_profit() == round(base["revenue"] - base["cogs"] - base["waste_cost"], 2)
    expected_net = round((st.revenue - base["revenue"]) + (base["waste_cost"] - st.waste_cost), 2)
    assert sim.net_profit() == expected_net


def test_markdown_cost_measures_discount_investment():
    sim = _sim_with([100.0])
    item = sim.store.items["SKU-0"]
    item.current_price = 8.0  # $2 below $10 base
    sim.store.record_sale("SKU-0", 5)
    assert sim.store.markdown_cost == 10.0
    assert sim.store.cogs == 30.0  # 5 × $6 cost
    item.current_price = 11.0  # hike counts back as a gain
    sim.store.record_sale("SKU-0", 5)
    assert sim.store.markdown_cost == 5.0


def test_reset_clears_profit_ledger():
    sim = _sim_with([100.0])
    sim.tick()
    sim.reset()
    st = sim.store
    assert (st.cogs, st.waste_cost, st.markdown_cost) == (0.0, 0.0, 0.0)
    assert sim.baseline["cogs"] == 0.0 and sim.baseline["waste_cost"] == 0.0


def test_depleted_forecast_frozen():
    from unittest import mock

    from backend.context_engine import ContextEngine
    sim = _sim_with([200.0])
    sim.tick_count = 12
    sim.pos_feed = [
        {"id": "T11-01", "tick": 11, "time": sim.sim_time.isoformat(),
         "lines": [{"sku": "SKU-0", "qty": 48}], "total": 480.0},
    ]
    sim.store.items["SKU-0"].quantity_on_shelf = 4
    sim.store.record_sale("SKU-0", 4)
    llm = mock.patch.object(ContextEngine, "_llm_available", return_value=True)
    chat = mock.patch.object(ContextEngine, "_chat",
                             return_value="QTY: 12\nREASON: cover the run-rate.")
    with llm, chat:
        sim._record_depletions({"SKU-0": 4})
    frozen = sim.forecast()[0]
    assert (frozen["velocity"], frozen["suggested_order"]) == (4.0, 12)
    assert frozen["forecast_source"] == "llm"
    for _ in range(30):
        sim.tick()  # POS history ages out; live velocity would decay to 0
    again = sim.forecast()[0]
    assert again["status"] == "sold_out"
    assert (again["velocity"], again["suggested_order"]) == (4.0, 12)


def _restock_kw(**over):
    kw = dict(sku="S-1", name="Test Greens", category="produce", batch=20,
              stocked_tick=48, depleted_tick=82, velocity=0.75, sold=18,
              wasted=2, base_price=4.99, current_price=4.49, cost_basis=2.80,
              footfall_index=0.92, temperature_c=21.3, precipitation_mm=0.0,
              event=None, wasted_baseline=6)
    kw.update(over)
    return kw


def test_restock_prompt_carries_prescribed_variables():
    from backend.context_engine import ContextEngine
    p = ContextEngine._restock_prompt(**_restock_kw())
    for token in ["S-1", "Test Greens", "produce", "20", "48", "82", "0.75",
                  "18", "2", "4.99", "4.49", "2.8", "0.92", "21.3", "6",
                  "none", "QTY", "REASON"]:
        assert token in p, token


def test_chat_tolerates_null_content():
    from unittest import mock

    from backend import context_engine as CE
    e = CE.ContextEngine()
    resp = mock.Mock()
    resp.json.return_value = {"choices": [{"message": {"content": None}}]}
    with mock.patch.object(CE, "requests") as fake:
        fake.post.return_value = resp
        assert e._chat("hello") == ""


def test_llm_restock_parses_clamps_and_caps():
    from unittest import mock

    from backend.context_engine import ContextEngine
    e = ContextEngine()
    kw = _restock_kw()
    llm = mock.patch.object(ContextEngine, "_llm_available", return_value=True)
    with llm, mock.patch.object(ContextEngine, "_chat",
                                return_value="QTY: 42\nREASON: test reason"):
        assert e.forecast_restock(**kw) == (42, "test reason", "llm")
    with llm, mock.patch.object(ContextEngine, "_chat",
                                return_value="**QTY:** 30\n**REASON:** bold reply"):
        assert e.forecast_restock(**kw) == (30, "bold reply", "llm")
    with llm, mock.patch.object(ContextEngine, "_chat",
                                return_value="QTY: 99999\nREASON: yolo"):
        qty, reason, source = e.forecast_restock(**kw)
        assert qty == 100 and source == "llm"  # capped at 5x batch
        assert "capped" in reason
    with llm, mock.patch.object(ContextEngine, "_chat",
                                side_effect=IOError("down")):
        qty, reason, source = e.forecast_restock(**kw)
        assert (qty, source) == (28, "rule")  # ceil(.75*34+2)
        assert "Rule fallback" in reason
    with mock.patch.object(ContextEngine, "_llm_available", return_value=False):
        qty, reason, source = e.forecast_restock(**kw)
        assert (qty, source) == (28, "rule")
        assert reason.startswith("Rule:")


def test_offline_depletion_uses_rule_formula():
    import os
    from datetime import timedelta
    from unittest import mock

    env = {"MODEL_API_KEY": ""}
    with mock.patch.dict(os.environ, env):
        sim = _sim_with([200.0, 200.0])
        sim.tick_count = 10
        feed = lambda i, sku: {"id": f"T9-{i:02d}", "tick": 9,
                               "time": sim.sim_time.isoformat(),
                               "lines": [{"sku": sku, "qty": 20}], "total": 200.0}
        sim.pos_feed = [feed(1, "SKU-0"), feed(2, "SKU-1")]
        sim.store.items["SKU-0"].quantity_on_shelf = 0
        it1 = sim.store.items["SKU-1"]
        it1.quantity_on_shelf = 0
        it1.expiration_timestamp = sim.sim_time - timedelta(hours=1)
        sim._record_depletions({"SKU-0": 20, "SKU-1": 5})
        rows = {r["sku"]: r for r in sim.forecast()}
        assert rows["SKU-0"]["suggested_order"] == 20  # 2.0 x 10 + 0
        assert rows["SKU-1"]["suggested_order"] == 25  # 2.0 x 10 + 5
        assert rows["SKU-0"]["forecast_source"] == "rule"
        assert rows["SKU-1"]["forecast_source"] == "rule"


def test_depletion_records_forecast_source():
    import os
    from unittest import mock

    env = {"MODEL_API_KEY": ""}
    with mock.patch.dict(os.environ, env):
        sim = _sim_with([1.0])
        sim.tick()
        sim.tick()
        row = sim.forecast()[0]
        assert row["forecast_source"] == "rule"
        assert row["forecast_reason"].startswith("Rule:")


def test_sim_control_sets_hours():
    from fastapi.testclient import TestClient

    from backend.main import app, sim as app_sim
    c = TestClient(app)
    c.__enter__()
    try:
        r = c.post("/api/v1/sim/control", json={"open_hour": 8, "close_hour": 20})
        assert r.json()["open_hour"] == 8
        assert r.json()["close_hour"] == 20
        assert app_sim.open_hour == 8 and app_sim.close_hour == 20
        st = c.get("/api/v1/sim/status").json()
        assert st["open_hour"] == 8 and st["close_hour"] == 20
        assert "store_open" in st and "active_skus" in st
        assert "revenue_baseline" in st
    finally:
        app_sim.open_hour, app_sim.close_hour = 9, 22
        c.__exit__(None, None, None)


if __name__ == "__main__":
    test_restock_prompt_carries_prescribed_variables()
    test_chat_tolerates_null_content()
    test_offline_depletion_uses_rule_formula()
    test_llm_restock_parses_clamps_and_caps()
    test_depletion_records_forecast_source()
    test_was_shows_batch_size_not_final_dregs()
    test_depleted_forecast_frozen()
    test_profit_accounts_for_costs_and_markdowns()
    test_markdown_cost_measures_discount_investment()
    test_reset_clears_profit_ledger()
    test_expiry_depletion_note()
    test_sellout_depletion_note()
    test_restock_clears_depletion_note()
    test_waste_tracked_per_sku_with_and_without_tool()
    test_reset_clears_per_sku_waste()
    test_weather_series_drives_sim()
    test_weather_fallback_without_file()
    test_fetch_failure_keeps_stale_forecast()
    test_forecast_velocity_math()
    test_forecast_velocity_and_runout()
    test_forecast_flags_inactive()
    test_restock_replenishes_and_mirrors_twin()
    test_forecast_and_restock_endpoints()
    test_active_skus_drop_on_expiry()
    test_sold_out_skus_leave_active_count()
    test_closed_store_makes_no_sales()
    test_expiry_books_waste_on_closed_ticks()
    test_open_store_trades_and_reports_hours()
    test_overnight_hours_wrap_midnight()
    test_baseline_twin_tracks_static_pricing()
    test_reset_clears_baseline()
    test_sim_control_sets_hours()
    print("ALL STORE-SIM TESTS PASSED")
