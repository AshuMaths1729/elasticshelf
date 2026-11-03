"""Realism regression tests: shared demand pool, margin floor, adversarial env.

    python -m tests.test_realism    (from elasticshelf/, venv active)
"""
from __future__ import annotations


def test_shared_pool_bounds_total_sales():
    """One footfall pool per tick: forcing the pool to 5 caps total units
    sold across ALL SKUs at 5 (previously each SKU drew its own demand)."""
    from backend import main as M
    if not M.store.items:
        M._load_seed_catalog()
    M.sim._poisson = lambda mu, rng=None: 5
    before = M.store.units_sold
    res = M.sim.tick()
    sold = sum(line["qty"] for t in res["checkouts"] for line in t["lines"])
    assert sold <= 5, sold
    assert M.store.units_sold - before == sold


def test_margin_floor():
    """Engine never prices below cost + 5%, even on an urgent markdown."""
    from backend.elasticity import ElasticityEngine
    for policy in ("ppo", "rule", "tuned-rule", "mpc", "dqn"):
        e = ElasticityEngine(policy=policy)
        p, a, _ = e.decide(10.0, 10.0, 8.0, 50, 6, -1.0)
        assert p >= 8.0 * 1.05 - 1e-9, (policy, p)


def test_short_dated_env_is_adversarial():
    from backend.pricing_env import DemandEnv
    demand = {"categories": {"x": {"base_qty": 1.5, "elasticity": 1.3}},
              "hourly_curve": [1.0] * 24}
    env = DemandEnv(demand, seed=3, short_dated=True)
    for _ in range(20):
        env.reset()
        assert 30 <= env.stock <= 60, env.stock
        assert 6 <= env.hours <= 48, env.hours
    std = DemandEnv(demand, seed=3)
    std.reset()
    assert std.hours >= 12


def test_rollout_reports_unit_weighted_waste():
    from backend.pricing_env import DemandEnv, rollout_stats
    demand = {"categories": {"x": {"base_qty": 1.5, "elasticity": 1.3}},
              "hourly_curve": [1.0] * 24}
    env = DemandEnv(demand, seed=1, short_dated=True)
    s = rollout_stats(env, lambda st, e: 2, n=10)  # hold: should waste a lot
    assert 0.0 <= s["waste_units_rate"] <= 1.0
    assert s["waste_units_rate"] > 0.3, s  # short-dated + hold = doomed


def _hours_sim():
    from backend.simulator import Simulator
    from backend.store import Store
    sim = Simulator(Store(store_id="T"), seed=1)
    sim.open_hour, sim.close_hour = 9, 22
    return sim


def test_selling_hours_skip_closed_time():
    """Policy clock = open hours to expiry (store trades 9-22)."""
    from datetime import timedelta
    sim = _hours_sim()  # sim_time is 09:00
    assert sim.selling_hours_until(sim.sim_time + timedelta(hours=10)) == 10
    # 09:00 -> next day 09:00: 13 open hours, 11 closed.
    assert sim.selling_hours_until(sim.sim_time + timedelta(hours=24)) == 13
    assert sim.selling_hours_until(sim.sim_time - timedelta(hours=1)) == 0
    sim.open_hour = sim.close_hour = 0  # 24h store
    assert sim.selling_hours_until(sim.sim_time + timedelta(hours=24)) == 24


def test_no_repricing_while_closed():
    """evaluate_pricing (also the /pricing/evaluate endpoint) holds every
    price while closed: zero footfall, a markdown would only burn spacing."""
    from datetime import timedelta
    from backend.models import InventoryItem
    sim = _hours_sim()
    sim.sim_time = sim.sim_time.replace(hour=23)
    sim.store.upsert(InventoryItem(
        sku="A", name="A", category="dairy", quantity_on_shelf=40,
        base_price=10.0, current_price=10.0, cost_basis=5.0,
        expiration_timestamp=sim.sim_time + timedelta(hours=8)))
    res = sim.evaluate_pricing()
    assert res["status"] == "closed" and res["updates_pushed"] == 0
    assert sim.store.get("A").current_price == 10.0


def test_late_ladder_and_off():
    from backend.elasticity import ElasticityEngine
    e = ElasticityEngine(policy="rule")
    # late ladder: <=12h forces at least -10%.
    p, a, _ = e.decide(10.0, 10.0, 5.0, 10, 10, 0.8, waste_ladder="late")
    assert a == 0 and p == 9.0, (p, a)
    # 'off' leaves the policy's own choice (rule holds safe stock).
    p, a, _ = e.decide(10.0, 10.0, 5.0, 10, 100, 0.8, waste_ladder="off")
    assert a == 2 and p == 10.0, (p, a)


def test_catalog_reanchored_to_sim_clock():
    """A stale catalog.json must not start the run with expired SKUs."""
    from backend import main as M
    M.store.items.clear()
    M._load_seed_catalog()
    from scripts.generate_synthetic import CATALOG_SPEC
    spec = {sku: hrs for sku, *_, hrs in CATALOG_SPEC}
    for it in M.store.all():
        hrs = (it.expiration_timestamp - M.sim.sim_time).total_seconds() / 3600
        assert hrs > 0, (it.sku, hrs)
        if it.sku in spec:
            assert abs(hrs - spec[it.sku]) < 1.0, (it.sku, hrs, spec[it.sku])


if __name__ == "__main__":
    test_shared_pool_bounds_total_sales()
    test_margin_floor()
    test_short_dated_env_is_adversarial()
    test_rollout_reports_unit_weighted_waste()
    test_selling_hours_skip_closed_time()
    test_no_repricing_while_closed()
    test_late_ladder_and_off()
    test_catalog_reanchored_to_sim_clock()
    print("ALL REALISM TESTS PASSED")
