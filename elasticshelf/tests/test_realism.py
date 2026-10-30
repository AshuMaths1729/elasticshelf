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
    M.sim._poisson = lambda mu: 5
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


if __name__ == "__main__":
    test_shared_pool_bounds_total_sales()
    test_margin_floor()
    test_short_dated_env_is_adversarial()
    test_rollout_reports_unit_weighted_waste()
    print("ALL REALISM TESTS PASSED")
