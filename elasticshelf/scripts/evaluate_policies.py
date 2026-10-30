"""Served-policy evaluation: revenue/profit vs units wasted, done honestly.

Every policy is served through ElasticityEngine.decide (same markdown-only,
expiry-ladder and margin-floor invariants), on:
  - standard DemandEnv episodes (stock-feasible expiries), and
  - adversarial short-dated episodes (30-60 units, 6-48h to expiry —
    doomed unless cleared fast; underrepresented in training draws.
    Below ~24h the expiry ladder saturates every policy to -10%, so
    this window is where policy differences still show).

Reports means across seeds with unit-weighted waste (actual leftover
units / initial units from rollout_stats' waste_units_rate), revenue
and margin. Writes JSON with --out.

Run:  python -m scripts.evaluate_policies [--seeds 5] [--episodes 100]
      python -m scripts.evaluate_policies --out data/policy_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.elasticity import ElasticityEngine
from backend.pricing_env import DemandEnv, rollout_stats

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
POLICIES = ("ppo", "dqn", "tuned-rule", "mpc", "rule")


def evaluate(policy: str, demand: dict, seeds: int, episodes: int,
             short_dated: bool) -> dict:
    eng = ElasticityEngine(policy=policy)
    agg = []
    for seed in range(seeds):
        env = DemandEnv(demand, seed=1000 + seed, short_dated=short_dated)
        agg.append(rollout_stats(env, lambda s, e: eng.decide(
            e.price, e.base_price, e.cost, e.stock, e.hours, e.dem,
            category=e.cat,
            footfall=demand["hourly_curve"][e.hour % 24], hour=e.hour)[1],
            n=episodes))
    keys = agg[0].keys()
    return {k: round(sum(a[k] for a in agg) / len(agg), 3) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    demand = json.loads((DATA / "demand_model.json").read_text())
    report = {"seeds": args.seeds, "episodes_per_seed": args.episodes,
              "standard": {}, "adversarial_short_dated": {}}
    for scenario, short in (("standard", False), ("adversarial_short_dated", True)):
        for policy in POLICIES:
            s = evaluate(policy, demand, args.seeds, args.episodes, short)
            report[scenario][policy] = s
            print(f"{scenario:26s} {policy:10s} waste_units={s['waste_units_rate']:.3f} "
                  f"sellout={s['sellout_rate']:.2f} revenue={s['revenue']:.2f} "
                  f"margin={s['margin']:.2f} reward={s['reward']:+.2f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
