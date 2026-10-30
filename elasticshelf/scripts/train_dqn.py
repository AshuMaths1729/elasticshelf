"""Train a DQN dynamic-pricing agent for the Elasticity Engine.

Environment: single-SKU lifecycle. Each episode draws a product (category,
price, cost, stock, hours-to-expiry) and steps hour-by-hour; the agent picks
price deltas from [-10%, -5%, 0%, +5%]. Demand is sampled from the fitted
elasticity model (scripts/prepare_training_data.py). Reward = realized margin
per hour minus a terminal waste penalty for expired leftover stock.

Writes: data/dqn_weights.pt, data/dqn_meta.json
Run:  python -m scripts.train_dqn [--episodes 2500]   (from elasticshelf/)

Tradeoff options (--waste-weight, --gamma, --shaping) retrain with a
different margin-vs-waste objective. Non-default runs write suffixed files
(data/dqn_weights_w<W>_g<G>.pt) and never overwrite the main DQN.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("torch is required: pip install -r requirements.txt")

from backend.elasticity import ACTIONS, QNet, rule_policy, state_features
from backend.pricing_env import (DEMAND_SCALE, PRICE_CAP, SHAPING,
                                 WASTE_WEIGHT, DemandEnv, rollout_stats)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

GAMMA = 0.95
LR = 1e-3
BUFFER = 20_000
BATCH = 128
TARGET_SYNC = 300
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 15_000


def train(episodes: int, seed: int = 1, gamma: float = GAMMA,
          waste_weight: float = WASTE_WEIGHT, shaping: float = SHAPING) -> QNet:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    demand = json.loads((DATA / "demand_model.json").read_text())
    env = DemandEnv(demand, seed=seed, waste_weight=waste_weight, shaping=shaping)
    net, tgt = QNet(), QNet()
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf: deque = deque(maxlen=BUFFER)
    steps = 0
    ep_rewards: list[float] = []
    for ep in range(1, episodes + 1):
        s = env.reset()
        done, total = False, 0.0
        while not done:
            eps = EPS_END + (EPS_START - EPS_END) * max(0.0, 1.0 - steps / EPS_DECAY_STEPS)
            if random.random() < eps:
                a = random.randrange(4)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(net(torch.tensor([s], dtype=torch.float32))).item())
            ns, r, done = env.step(a)
            buf.append((s, a, r, ns, done))
            s, total, steps = ns, total + r, steps + 1
            if len(buf) >= BATCH:
                batch = random.sample(buf, BATCH)
                sb = torch.tensor([b[0] for b in batch], dtype=torch.float32)
                ab = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
                rb = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                nsb = torch.tensor([b[3] for b in batch], dtype=torch.float32)
                db = torch.tensor([b[4] for b in batch], dtype=torch.float32)
                q = net(sb).gather(1, ab).squeeze(1)
                with torch.no_grad():
                    target = rb + gamma * (1 - db) * tgt(nsb).max(1).values
                loss = nn.functional.mse_loss(q, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if steps % TARGET_SYNC == 0:
                    tgt.load_state_dict(net.state_dict())
        ep_rewards.append(total)
        if ep % 250 == 0 or ep == 1:
            print(f"  ep {ep}/{episodes}  avg100={np.mean(ep_rewards[-100:]):+.2f}  eps={eps:.2f}",
                  flush=True)
    return net


def _rollout_stats(env: DemandEnv, acts, n: int = 200) -> dict:
    """Legacy alias of backend.pricing_env.rollout_stats (adds a margin key)."""
    return rollout_stats(env, acts, n=n)


def evaluate_dqn(net: QNet, demand: dict, seed: int = 999,
                 waste_weight: float = WASTE_WEIGHT,
                 shaping: float = SHAPING) -> dict:
    env = DemandEnv(demand, seed=seed, waste_weight=waste_weight, shaping=shaping)
    net.eval()
    with torch.no_grad():
        return _rollout_stats(
            env, lambda s, e: int(torch.argmax(net(torch.tensor([s], dtype=torch.float32))).item()))


def evaluate_rule(demand: dict, seed: int = 999, rule_params: dict | None = None,
                  waste_weight: float = WASTE_WEIGHT,
                  shaping: float = SHAPING) -> dict:
    env = DemandEnv(demand, seed=seed, waste_weight=waste_weight, shaping=shaping)
    return _rollout_stats(
        env, lambda s, e: rule_policy(e.price / e.base_price, e.stock, e.hours, e.dem,
                                      params=rule_params))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--waste-weight", type=float, default=WASTE_WEIGHT)
    ap.add_argument("--shaping", type=float, default=SHAPING)
    args = ap.parse_args()
    if not (DATA / "demand_model.json").exists():
        raise SystemExit("data/demand_model.json missing — run scripts.prepare_training_data first")
    print(f"Training DQN for {args.episodes} episodes "
          f"(gamma={args.gamma} waste_weight={args.waste_weight} shaping={args.shaping}) ...")
    net = train(args.episodes, seed=args.seed, gamma=args.gamma,
                waste_weight=args.waste_weight, shaping=args.shaping)
    demand = json.loads((DATA / "demand_model.json").read_text())
    dqn_stats = evaluate_dqn(net, demand, waste_weight=args.waste_weight,
                             shaping=args.shaping)
    rule_stats = evaluate_rule(demand, waste_weight=args.waste_weight,
                               shaping=args.shaping)
    print(f"Eval DQN : {dqn_stats}")
    print(f"Eval rule: {rule_stats}")
    default_run = (args.gamma == GAMMA and args.waste_weight == WASTE_WEIGHT
                   and args.shaping == SHAPING)
    if default_run:
        weights_path, meta_path = DATA / "dqn_weights.pt", DATA / "dqn_meta.json"
    else:
        tag = f"w{args.waste_weight:g}_g{args.gamma}_s{args.shaping:g}"
        weights_path, meta_path = DATA / f"dqn_weights_{tag}.pt", DATA / f"dqn_meta_{tag}.json"
        print(f"Non-default tradeoff run — main DQN files left untouched.")
    torch.save(net.state_dict(), weights_path)
    meta_path.write_text(json.dumps({
        "episodes": args.episodes, "seed": args.seed, "gamma": args.gamma,
        "waste_weight": args.waste_weight, "shaping": args.shaping,
        "eval_dqn": dqn_stats, "eval_rule": rule_stats,
        "actions": list(ACTIONS)}, indent=2))
    print(f"Wrote {weights_path.name} + {meta_path.name}")


if __name__ == "__main__":
    main()
