"""Tradeoff-layer regression tests: tuned rules, MPC, lambda env.

Covers the margin-vs-waste controls without touching the DQN:
parametrized rule_policy, waste_weight/shaping env options, the MPC
policy, and the tune_rule search helpers. Runs with plain python:

    python -m tests.test_tradeoff   (from elasticshelf/, venv active)
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path


def _demand():
    return {"categories": {"x": {"base_qty": 1.5, "elasticity": 1.3}},
            "hourly_curve": [1.0] * 24}


def test_rule_defaults_reproduce_legacy():
    from backend.elasticity import RULE_DEFAULTS, rule_policy
    rng = random.Random(0)
    for _ in range(500):
        pr = rng.uniform(0.5, 1.2)
        st = rng.randint(0, 60)
        hrs = rng.uniform(0, 120)
        dem = rng.uniform(-1, 1)
        assert rule_policy(pr, st, hrs, dem) == \
            rule_policy(pr, st, hrs, dem, params=dict(RULE_DEFAULTS))
    assert rule_policy(1.0, 50, 6, 0.0) == 0    # urgent -> -10%
    assert rule_policy(1.0, 5, 100, 0.0) == 2   # safe -> hold
    assert rule_policy(0.9, 5, 100, 1.0) == 3   # surging demand -> +5%


def test_rule_params_change_behavior():
    from backend.elasticity import rule_policy
    calm = {"hrs_urgent": 6.0, "hrs_soon": 12.0, "hrs_watch": 24.0,
            "hi_thresh": 4.5, "lo_thresh": 2.5}
    assert rule_policy(1.0, 50, 20, 0.0) == 0          # default: -10%
    assert rule_policy(1.0, 50, 20, 0.0, params=calm) == 2  # tuned calm: hold


def test_env_defaults_preserve_dqn():
    from backend.pricing_env import SHAPING, WASTE_WEIGHT, DemandEnv
    assert WASTE_WEIGHT == 2.0 and SHAPING == 0.0
    env = DemandEnv(_demand(), seed=1)
    assert env.waste_weight == 2.0 and env.shaping == 0.0
    try:
        import scripts.train_dqn as T
    except (ImportError, SystemExit):
        return  # torch missing: train script unusable, nothing further to check
    assert T.DemandEnv is DemandEnv
    assert T.GAMMA == 0.95 and T.WASTE_WEIGHT == 2.0 and T.SHAPING == 0.0


def test_waste_weight_penalizes_waste():
    import numpy as np

    from backend.pricing_env import DemandEnv, rollout_stats
    rewards = {}
    for w in (0.5, 8.0):
        random.seed(5)
        np.random.seed(5)
        env = DemandEnv(_demand(), seed=5, waste_weight=w)
        stats = rollout_stats(env, lambda s, e: 2, n=20)  # fixed hold policy
        rewards[w] = stats["reward"]
        assert stats["margin"] >= stats["reward"]  # penalty only subtracts
    assert rewards[0.5] >= rewards[8.0]


def test_mpc_tradeoff_knob():
    from backend.elasticity import mpc_action
    kw = dict(price=10.0, base_price=10.0, cost=6.0, stock=40,
              hours_to_expiry=30, demand_mod=0.0, base_qty=1.5,
              elasticity=1.4, footfall=1.0)
    lax = mpc_action(**kw, waste_weight=0.1)
    strict = mpc_action(**kw, waste_weight=10.0)
    assert lax in (0, 1, 2, 3) and strict in (0, 1, 2, 3)
    assert strict <= lax  # more waste-averse -> deeper-or-equal markdown
    assert mpc_action(**{**kw, "stock": 0}) == 2
    assert mpc_action(**{**kw, "hours_to_expiry": 0}) == 2


def test_engine_policies():
    import random as _random

    from backend.elasticity import ElasticityEngine
    e = ElasticityEngine(policy="rule")
    assert e.backend == "rules"
    _, _, note = e.decide(10.0, 10.0, 6.0, 50, 6, 0.0)
    assert note == "rule-policy"
    e = ElasticityEngine(policy="mpc", waste_weight=4.0)
    assert e.backend == "mpc"
    _, _, note = e.decide(10.0, 10.0, 6.0, 50, 6, 0.0, category="meat",
                          footfall=1.0, hour=12)
    assert note == "mpc-w4"
    e = ElasticityEngine(policy="tuned-rule", rule_params={"hi_thresh": 4.0})
    assert e.backend == "rules-tuned"
    _, _, note = e.decide(10.0, 10.0, 6.0, 50, 6, 0.0)
    assert note == "rule-tuned"
    from unittest import mock
    with mock.patch.dict("os.environ", {"ELASTICITY_POLICY": "auto"}):
        e = ElasticityEngine()  # default unchanged: DQN first when available
        assert e.backend in ("pytorch-dqn", "pytorch-dqn-lite", "rules")
    # MPC price bounds mirror the smoke test exactly.
    mpc = ElasticityEngine(policy="mpc")
    rng = _random.Random(0)
    for _ in range(50):
        base = rng.uniform(2, 16)
        cost = base * rng.uniform(0.45, 0.7)
        cur = rng.uniform(cost, base * 1.5)
        p, a, _ = mpc.decide(cur, base, cost, rng.randint(0, 60),
                             rng.uniform(0, 120), rng.uniform(-1, 1))
        assert cost - 1e-9 <= p <= base * 1.5 + 1e-9, (p, cost, base)
        assert a in (0, 1, 2, 3), a


def test_tune_rule_smoke():
    from scripts.tune_rule import evaluate_mpc, evaluate_rule, sample_params, tune
    rng = random.Random(0)
    p = sample_params(rng)
    assert p["hrs_urgent"] < p["hrs_soon"] < p["hrs_watch"]
    assert p["lo_thresh"] < p["hi_thresh"]
    demand = _demand()
    s = evaluate_rule(demand, p, seed=7, waste_weight=2.0, shaping=0.0, rollouts=5)
    assert set(s) >= {"reward", "margin", "waste_rate", "sellout_rate"}
    m = evaluate_mpc(demand, seed=7, waste_weight=2.0, shaping=0.0,
                     rollouts=5, horizon=24)
    assert m["waste_rate"] >= 0.0
    res = tune(demand, 2.0, 0.0, trials=4, rollouts=5, seed=7, verbose=False)
    assert res["best"]["score"] >= res["baseline"]["reward"] - 1e-9
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rule_params.json"
        out.write_text(json.dumps({"params": res["best"]["params"]}))
        from backend.elasticity import load_rule_params
        loaded = load_rule_params(str(out))
        assert loaded["hi_thresh"] == res["best"]["params"]["hi_thresh"]


if __name__ == "__main__":
    test_rule_defaults_reproduce_legacy()
    test_rule_params_change_behavior()
    test_env_defaults_preserve_dqn()
    test_waste_weight_penalizes_waste()
    test_mpc_tradeoff_knob()
    test_engine_policies()
    test_tune_rule_smoke()
    print("ALL TRADEOFF TESTS PASSED")
