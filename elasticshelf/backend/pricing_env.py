"""Shared single-SKU pricing environment (torch-free).

Same dynamics the DQN was trained on (see scripts/train_dqn.py): each episode
draws a product (category, price, cost, stock, hours-to-expiry) and steps
hour-by-hour; the agent picks price deltas from [-10%, -5%, 0%, +5%].

Reward = realized margin per hour minus a terminal waste penalty for expired
leftover stock, plus an optional dense expiry-risk shaping term:

    reward = margin - waste_weight * cost * leftover / 10   (terminal only)
             - shaping * stock / hours_left * cost / 10      (every step)

Defaults (waste_weight=2.0, shaping=0.0) reproduce DQN training exactly.
Rule tuning and MPC evaluation import from here so they never need torch.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .elasticity import ACTIONS, state_features

DEMAND_SCALE = 0.5  # tuned so full-price sales roughly match typical stock
PRICE_CAP = 1.5  # max price as a multiple of base (matches serve-time cap)
WASTE_WEIGHT = 2.0  # terminal penalty = weight * cost * leftover / 10
SHAPING = 0.0  # dense expiry-risk coefficient (0 = legacy behavior)


class DemandEnv:
    def __init__(self, demand: dict, seed: int = 0,
                 waste_weight: float = WASTE_WEIGHT,
                 shaping: float = SHAPING):
        self.demand = demand
        self.rng = np.random.default_rng(seed)
        self.waste_weight = waste_weight
        self.shaping = shaping
        self.last_margin = 0.0
        self.reset()

    def reset(self):
        cats = list(self.demand["categories"].keys())
        # Use the env's own seeded RNG (was: global random.choice, which made
        # rollouts non-reproducible and cross-policy comparisons noisy).
        self.cat = cats[int(self.rng.integers(0, len(cats)))]
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
        self.last_margin = 0.0
        self.last_revenue = 0.0
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
        margin = (self.price - self.cost) * sold / 10.0
        self.last_margin = margin
        self.last_revenue = self.price * sold / 10.0
        reward = margin
        if self.shaping > 0 and self.stock > 0 and self.hours > 1:
            reward -= self.shaping * (self.stock / self.hours) * (self.cost / 10.0)
        self.hours -= 1.0
        self.hour += 1
        self.dem = float(np.clip(self.dem + self.rng.normal(0, 0.15), -1, 1))
        done = self.hours <= 0 or self.stock <= 0
        if done and self.hours <= 0 and self.stock > 0:
            reward -= self.waste_weight * self.cost * self.stock / 10.0
        return self._state(), reward, done


def rollout_stats(env: DemandEnv, acts: Callable, n: int = 200) -> Dict:
    """Roll out `acts(state, env) -> action` n times; summarize margin/waste."""
    totals, margins, revenues, sellouts, wastes, min_ratios = [], [], [], [], [], []
    for _ in range(n):
        s = env.reset()
        start_stock = env.stock
        min_ratio, done, total, margin, revenue = 1.0, False, 0.0, 0.0, 0.0
        while not done:
            a = acts(s, env)
            s, r, done = env.step(a)
            total += r
            margin += env.last_margin
            revenue += env.last_revenue
            min_ratio = min(min_ratio, env.price / env.base_price)
        totals.append(total)
        margins.append(margin)
        revenues.append(revenue)
        sellouts.append(1.0 if env.stock <= 0 else 0.0)
        wastes.append(env.stock / max(start_stock, 1))
        min_ratios.append(min_ratio)
    return {"reward": round(float(np.mean(totals)), 2),
            "margin": round(float(np.mean(margins)), 2),
            "revenue": round(float(np.mean(revenues)), 2),
            "sellout_rate": round(float(np.mean(sellouts)), 2),
            "waste_rate": round(float(np.mean(wastes)), 2),
            "avg_deepest_discount": round(1 - float(np.mean(min_ratios)), 2)}
