"""Tune the rule policy and map the margin-vs-waste Pareto frontier.

Random search over rule thresholds, evaluated on the exact DQN training
dynamics (backend.pricing_env). No torch needed, DQN files untouched.

Writes: data/rule_params.json   (best params for --waste-weight; serve with
                                 ELASTICITY_POLICY=tuned-rule)
        data/rule_pareto.json   (only with --sweep: per-lambda comparison of
                                 default rule vs tuned rule vs MPC)
Run:  python -m scripts.tune_rule [--trials 80]   (from elasticshelf/)
      python -m scripts.tune_rule --sweep 0.5,1,2,4,8 --include-mpc
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from backend.elasticity import RULE_DEFAULTS, mpc_action, rule_policy
from backend.pricing_env import SHAPING, WASTE_WEIGHT, DemandEnv, rollout_stats

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

# Search ranges for the tuned subset; urgency weights stay at legacy values.
RANGES = {
    "hrs_urgent": (6.0, 24.0),
    "hrs_soon": (18.0, 48.0),
    "hrs_watch": (36.0, 96.0),
    "stock_big": (20.0, 50.0),
    "dem_gain": (0.0, 4.0),
    "hi_thresh": (2.0, 4.5),
    "lo_thresh": (0.5, 2.5),
    "recover_thresh": (-3.0, -0.5),
}


def sample_params(rng: random.Random) -> dict:
    p = {k: rng.uniform(*RANGES[k]) for k in RANGES}
    hrs = sorted([p["hrs_urgent"], p["hrs_soon"], p["hrs_watch"]])
    p["hrs_urgent"], p["hrs_soon"], p["hrs_watch"] = hrs
    if p["lo_thresh"] >= p["hi_thresh"]:
        p["lo_thresh"], p["hi_thresh"] = p["hi_thresh"], p["lo_thresh"]
    p["stock_big"] = round(p["stock_big"])
    return {k: round(float(v), 3) for k, v in p.items()}


def evaluate_rule(demand: dict, params: dict | None, seed: int,
                  waste_weight: float, shaping: float, rollouts: int) -> dict:
    random.seed(seed)  # paired episodes: every candidate sees the same draws
    env = DemandEnv(demand, seed=seed, waste_weight=waste_weight, shaping=shaping)
    return rollout_stats(
        env,
        lambda s, e: rule_policy(e.price / e.base_price, e.stock, e.hours, e.dem,
                                 params=params),
        n=rollouts)


def evaluate_mpc(demand: dict, seed: int, waste_weight: float, shaping: float,
                 rollouts: int, horizon: int) -> dict:
    random.seed(seed)
    env = DemandEnv(demand, seed=seed, waste_weight=waste_weight, shaping=shaping)
    curve = demand["hourly_curve"]

    def acts(s, e):
        return mpc_action(e.price, e.base_price, e.cost, e.stock, e.hours, e.dem,
                          waste_weight=waste_weight, horizon=horizon,
                          base_qty=e.base_qty, elasticity=e.elast,
                          footfall=curve[e.hour % 24])

    return rollout_stats(env, acts, n=rollouts)


def tune(demand: dict, waste_weight: float, shaping: float, trials: int,
         rollouts: int, seed: int, verbose: bool = True) -> dict:
    rng = random.Random(seed)
    baseline = evaluate_rule(demand, None, seed, waste_weight, shaping, rollouts)
    best = {"params": dict(RULE_DEFAULTS), "stats": baseline, "score": baseline["reward"]}
    for t in range(1, trials + 1):
        params = sample_params(rng)
        stats = evaluate_rule(demand, params, seed, waste_weight, shaping, rollouts)
        if stats["reward"] > best["score"]:
            best = {"params": params, "stats": stats, "score": stats["reward"]}
        if verbose and (t % 10 == 0 or t == trials):
            print(f"  trial {t}/{trials}  best_reward={best['score']:+.2f} "
                  f"waste={best['stats']['waste_rate']:.2f}", flush=True)
    return {"baseline": baseline, "best": best}


def pick_constrained(candidates: list, waste_target: float) -> dict | None:
    feasible = [c for c in candidates if c["stats"]["waste_rate"] <= waste_target]
    if not feasible:
        return None
    return max(feasible, key=lambda c: c["stats"]["margin"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--rollouts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--waste-weight", type=float, default=WASTE_WEIGHT)
    ap.add_argument("--shaping", type=float, default=SHAPING)
    ap.add_argument("--waste-target", type=float, default=None,
                    help="if set, served params are the best-margin feasible trial")
    ap.add_argument("--sweep", type=str, default="",
                    help="comma-separated waste weights for a Pareto sweep")
    ap.add_argument("--include-mpc", action="store_true")
    ap.add_argument("--mpc-horizon", type=int, default=120)
    ap.add_argument("--out", type=str, default=str(DATA / "rule_params.json"))
    ap.add_argument("--pareto", type=str, default=str(DATA / "rule_pareto.json"))
    args = ap.parse_args()
    if not (DATA / "demand_model.json").exists():
        raise SystemExit("data/demand_model.json missing — run scripts.prepare_training_data first")
    demand = json.loads((DATA / "demand_model.json").read_text())

    lambdas = [float(x) for x in args.sweep.split(",") if x.strip()] or [args.waste_weight]
    pareto = []
    for w in lambdas:
        print(f"Tuning rule for waste_weight={w:g} "
              f"({args.trials} trials x {args.rollouts} rollouts) ...")
        res = tune(demand, w, args.shaping, args.trials, args.rollouts, args.seed)
        entry: dict = {"waste_weight": w, "rule_default": res["baseline"],
                       "tuned_rule": {"params": res["best"]["params"],
                                      "stats": res["best"]["stats"]}}
        print(f"  default: reward={res['baseline']['reward']:+.2f} "
              f"waste={res['baseline']['waste_rate']:.2f}")
        print(f"  tuned  : reward={res['best']['stats']['reward']:+.2f} "
              f"waste={res['best']['stats']['waste_rate']:.2f}")
        if args.include_mpc:
            mpc_stats = evaluate_mpc(demand, args.seed, w, args.shaping,
                                     args.rollouts, args.mpc_horizon)
            entry["mpc"] = {"stats": mpc_stats}
            print(f"  mpc    : reward={mpc_stats['reward']:+.2f} "
                  f"waste={mpc_stats['waste_rate']:.2f}")
        pareto.append(entry)

    if len(pareto) > 1:
        Path(args.pareto).write_text(json.dumps(pareto, indent=2))
        print(f"Wrote {args.pareto}")

    main_entry = next((e for e in pareto if e["waste_weight"] == args.waste_weight),
                      pareto[0])
    served_params: dict = main_entry["tuned_rule"]["params"]
    constrained = None
    if args.waste_target is not None:
        cands = [{"params": main_entry["tuned_rule"]["params"],
                  "stats": main_entry["tuned_rule"]["stats"]},
                 {"params": dict(RULE_DEFAULTS),
                  "stats": main_entry["rule_default"]}]
        constrained = pick_constrained(cands, args.waste_target)
        if constrained is not None:
            served_params = constrained["params"]
            print(f"Waste target {args.waste_target}: serving "
                  f"margin={constrained['stats']['margin']:+.2f} "
                  f"waste={constrained['stats']['waste_rate']:.2f}")
        else:
            print(f"Waste target {args.waste_target}: INFEASIBLE at "
                  f"waste_weight={main_entry['waste_weight']}, serving best reward")
    out = {"method": "tuned-rule", "waste_weight": main_entry["waste_weight"],
           "shaping": args.shaping, "seed": args.seed, "trials": args.trials,
           "rollouts": args.rollouts, "waste_target": args.waste_target,
           "baseline": main_entry["rule_default"],
           "best": main_entry["tuned_rule"],
           "best_constrained": constrained,
           "params": served_params}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out} — serve with ELASTICITY_POLICY=tuned-rule")


if __name__ == "__main__":
    main()
