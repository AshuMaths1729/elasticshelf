"""Train a DQN dynamic-pricing agent for the Elasticity Engine.

Environment: single-SKU lifecycle. Each episode draws a product (category,
price, cost, stock, hours-to-expiry) and steps hour-by-hour; the agent picks
price deltas from [-10%, -5%, 0%, +5%]. Demand is sampled from the fitted
elasticity model (scripts/prepare_training_data.py). Reward = realized margin
per hour minus a terminal waste penalty for expired leftover stock.

Writes: data/dqn_weights.pt, data/dqn_meta.json
Run:  python -m scripts.train_dqn [--episodes 2500]   (from elasticshelf/)
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

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

GAMMA = 0.95
LR = 1e-3
BUFFER = 20_000
BATCH = 128
TARGET_SYNC = 300
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 15_000
DEMAND_SCALE = 0.5  # tuned so full-price sales roughly match typical stock
PRICE_CAP = 1.5  # max price as a multiple of base (matches serve-time cap)


class DemandEnv:
    def __init__(self, demand: dict, seed: int = 0):
        self.demand = demand
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        cats = list(self.demand["categories"].keys())
        self.cat = random.choice(cats)
        p = self.demand["categories"][self.cat]
        self.base_qty = p["base_qty"]
        self.elast = p["elasticity"]
        self.base_price = float(self.rng.uniform(2.0, 16.0))
        self.cost = round(self.base_price * float(self.rng.uniform(0.45, 0.70)), 2)
        self.price = self.base_price
        self.stock = int(self.rng.integers(10, 61))
        # Feasible cover: hours scale with stock (no doomed 55-units-in-8h draws).
        self.hours = float(np.clip(self.stock * self.rng.uniform(0.8, 2.5), 12, 120))
        self.hour = int(self.rng.integers(0, 24))
        self.dem = float(self.rng.uniform(-0.5, 0.5))
        return self._state()

    def _state(self):
        return state_features(self.price / self.base_price, self.stock, self.hours, self.dem)

    def step(self, action: int):
        self.price = round(min(self.base_price * PRICE_CAP,
                               max(self.cost, self.price * (1 + ACTIONS[action]))), 2)
        # Signed elasticity: discounts lift demand, hikes cut it (no free lunch).
        rel = max(-0.5, min(0.6, (self.base_price - self.price) / self.base_price))
        foot = self.demand["hourly_curve"][self.hour % 24]
        lam = max(0.0, DEMAND_SCALE * self.base_qty
                  * max(0.05, 1 + self.elast * rel) * foot * (1 + 0.5 * self.dem))
        sold = min(self.stock, int(self.rng.poisson(lam)))
        self.stock -= sold
        reward = (self.price - self.cost) * sold / 10.0
        self.hours -= 1.0
        self.hour += 1
        self.dem = float(np.clip(self.dem + self.rng.normal(0, 0.15), -1, 1))
        done = self.hours <= 0 or self.stock <= 0
        if done and self.hours <= 0 and self.stock > 0:
            reward -= self.cost * self.stock * 2.0 / 10.0  # 2x-cost waste penalty
        return self._state(), reward, done


def train(episodes: int, seed: int = 1) -> QNet:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    demand = json.loads((DATA / "demand_model.json").read_text())
    env = DemandEnv(demand, seed=seed)
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
                    target = rb + GAMMA * (1 - db) * tgt(nsb).max(1).values
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


def _rollout_stats(env: DemandEnv, acts) -> dict:
    totals, sellouts, wastes, min_ratios = [], [], [], []
    for _ in range(200):
        s = env.reset()
        start_stock = env.stock
        min_ratio, done, total = 1.0, False, 0.0
        while not done:
            a = acts(s, env)
            s, r, done = env.step(a)
            total += r
            min_ratio = min(min_ratio, env.price / env.base_price)
        totals.append(total)
        sellouts.append(1.0 if env.stock <= 0 else 0.0)
        wastes.append(env.stock / max(start_stock, 1))
        min_ratios.append(min_ratio)
    return {"reward": round(float(np.mean(totals)), 2),
            "sellout_rate": round(float(np.mean(sellouts)), 2),
            "waste_rate": round(float(np.mean(wastes)), 2),
            "avg_deepest_discount": round(1 - float(np.mean(min_ratios)), 2)}


def evaluate_dqn(net: QNet, demand: dict, seed: int = 999) -> dict:
    env = DemandEnv(demand, seed=seed)
    net.eval()
    with torch.no_grad():
        return _rollout_stats(
            env, lambda s, e: int(torch.argmax(net(torch.tensor([s], dtype=torch.float32))).item()))


def evaluate_rule(demand: dict, seed: int = 999) -> dict:
    env = DemandEnv(demand, seed=seed)
    return _rollout_stats(
        env, lambda s, e: rule_policy(e.price / e.base_price, e.stock, e.hours, e.dem))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if not (DATA / "demand_model.json").exists():
        raise SystemExit("data/demand_model.json missing — run scripts.prepare_training_data first")
    print(f"Training DQN for {args.episodes} episodes ...")
    net = train(args.episodes, seed=args.seed)
    demand = json.loads((DATA / "demand_model.json").read_text())
    dqn_stats = evaluate_dqn(net, demand)
    rule_stats = evaluate_rule(demand)
    print(f"Eval DQN : {dqn_stats}")
    print(f"Eval rule: {rule_stats}")
    torch.save(net.state_dict(), DATA / "dqn_weights.pt")
    (DATA / "dqn_meta.json").write_text(json.dumps({
        "episodes": args.episodes, "seed": args.seed, "gamma": GAMMA,
        "eval_dqn": dqn_stats, "eval_rule": rule_stats,
        "actions": list(ACTIONS)}, indent=2))
    print("Wrote data/dqn_weights.pt + data/dqn_meta.json")


if __name__ == "__main__":
    main()
