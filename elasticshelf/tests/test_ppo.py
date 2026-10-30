"""PPO regression tests: actor-critic, engine integration, short training.

Runs with plain python, no test framework needed:

    python -m tests.test_ppo        (from elasticshelf/, venv active)

Without torch the torch-dependent tests skip (return early), mirroring
test_tradeoff.py's handling of the optional DQN training script.
"""
from __future__ import annotations

import random


def _demand():
    return {"categories": {"x": {"base_qty": 1.5, "elasticity": 1.3}},
            "hourly_curve": [1.0] * 24}


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def test_actor_critic_shapes():
    if not _has_torch():
        return
    import torch

    from backend.elasticity import ActorCritic
    net = ActorCritic()
    logits, value = net(torch.zeros((5, 4)))
    assert logits.shape == (5, 4), logits.shape
    assert value.shape == (5,), value.shape
    assert net.act_greedy([1.0, 0.5, 0.5, 0.0]) in (0, 1, 2)  # +5% masked out


def test_engine_ppo_bounds():
    from backend.elasticity import ElasticityEngine
    e = ElasticityEngine(policy="ppo")
    # Trained weights present -> pytorch-ppo; absent/no torch -> rule fallback.
    # Either way decisions must be valid and respect the shared price bounds.
    assert e.backend in ("pytorch-ppo", "ppo-rules-fallback"), e.backend
    rng = random.Random(0)
    for _ in range(200):
        base = rng.uniform(2, 16)
        cost = base * rng.uniform(0.45, 0.7)
        cur = rng.uniform(cost, base * 1.5)
        p, a, note = e.decide(cur, base, cost, rng.randint(0, 60),
                              rng.uniform(0, 120), rng.uniform(-1, 1))
        assert cost - 1e-9 <= p <= base * 1.5 + 1e-9, (p, cost, base)
        assert a in (0, 1, 2, 3), a
        assert note in ("pytorch-ppo", "ppo-rules-fallback"), note


def test_ppo_short_training_runs():
    if not _has_torch():
        return
    from scripts.train_ppo import evaluate_ppo, evaluate_rule, train_ppo
    demand = _demand()
    net = train_ppo(demand, iterations=2, rollout_steps=256, seed=3, verbose=False)
    ppo_stats = evaluate_ppo(net, demand, seed=5)
    rule_stats = evaluate_rule(demand, seed=5)
    for stats in (ppo_stats, rule_stats):
        assert set(stats) >= {"reward", "margin", "waste_rate", "sellout_rate"}
        assert 0.0 <= stats["waste_rate"] <= 1.0
        assert 0.0 <= stats["sellout_rate"] <= 1.0
    # A 2-iteration run is a plumbing check, not a quality claim: just
    # require a finite reward in a sane range for this environment.
    assert ppo_stats["reward"] == ppo_stats["reward"]  # not NaN
    assert abs(ppo_stats["reward"]) < 1e6


def test_never_increases_and_expired_never_repriced():
    """Business invariants, enforced by ElasticityEngine for every policy:
    prices only ever decrease or hold, and expired stock is never repriced."""
    from backend.elasticity import ElasticityEngine
    rng = random.Random(1)
    for policy in ("ppo", "rule", "tuned-rule", "mpc", "dqn"):
        e = ElasticityEngine(policy=policy)
        for _ in range(100):
            base = rng.uniform(2, 16)
            cost = base * rng.uniform(0.45, 0.7)
            cur = round(rng.uniform(cost, base), 2)  # shelf prices are cent values
            hrs = rng.uniform(0.1, 120)
            p, a, _ = e.decide(cur, base, cost, rng.randint(0, 60),
                               hrs, rng.uniform(-1, 1))
            assert p <= cur + 1e-9, (policy, p, cur)   # markdown-only
            assert a != 3, (policy, a)                 # +5% never served
            # Expired (and exactly-at-expiry): price frozen, hold action.
            p2, a2, note2 = e.decide(cur, base, cost, 30, rng.uniform(-48, 0), 0.5)
            assert p2 == round(cur, 2) and a2 == 2 and note2 == "expired-hold", \
                (policy, p2, a2, note2)


def test_expiry_ladder_forces_markdown():
    """Waste-minimisation ladder: near expiry the engine markdowns even
    when the underlying policy would hold (all policies, served level)."""
    from backend.elasticity import ElasticityEngine
    for policy in ("ppo", "rule", "tuned-rule", "mpc", "dqn"):
        e = ElasticityEngine(policy=policy)
        # 20h left, calm demand: ladder forces a markdown below current.
        p, a, _ = e.decide(10.0, 10.0, 5.0, 10, 20, 0.8)
        assert a == 0 and p < 10.0, (policy, p, a)
        # 40h left with heavy stock: at least -5%.
        p, a, _ = e.decide(10.0, 10.0, 5.0, 40, 40, 0.8)
        assert a in (0, 1) and p < 10.0, (policy, p, a)
        # Early ladder profile (live simulator): 72h left, stock 30 -> markdown.
        p, a, _ = e.decide(10.0, 10.0, 5.0, 30, 72, 0.8, waste_ladder="early")
        assert a in (0, 1) and p < 10.0, (policy, p, a)
        # Far from expiry with light stock: ladder does not force anything
        # (price may hold, but per markdown-only it may never rise).
        p, a, _ = e.decide(10.0, 10.0, 5.0, 5, 100, 0.0)
        assert p <= 10.0, (policy, p)


def test_gae_shapes():
    if not _has_torch():
        return
    from scripts.train_ppo import compute_gae
    adv, returns = compute_gae([1.0, 1.0, 1.0], [0.0, 0.0, 1.0],
                               [0.5, 0.5, 0.5], last_value=0.0)
    assert len(adv) == len(returns) == 3
    # Terminal step: advantage is reward - value (no bootstrap past done).
    assert abs(float(adv[2]) - 0.5) < 1e-6


if __name__ == "__main__":
    test_actor_critic_shapes()
    test_engine_ppo_bounds()
    test_ppo_short_training_runs()
    test_never_increases_and_expired_never_repriced()
    test_expiry_ladder_forces_markdown()
    test_gae_shapes()
    print("ALL PPO TESTS PASSED")
