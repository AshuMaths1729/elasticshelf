"""Module C: PyTorch Elasticity Engine (DQN dynamic pricing).

State:  [price_ratio, stock_norm, hours_to_expiry_norm, demand_modifier]
Action: price delta in [-10%, -5%, 0%, +5%] (index 0..3).

Backends, best first:
1. "pytorch-dqn"      — torch installed + data/dqn_weights.pt trained
                        (see scripts/train_dqn.py).
2. "pytorch-dqn-lite" — torch installed, no weights: rule-distilled MLP.
3. "rules"            — torch missing: transparent rule policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

ACTIONS = (-0.10, -0.05, 0.0, 0.05)
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "dqn_weights.pt"

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False


def state_features(price_ratio: float, stock: int, hours_to_expiry: float,
                   demand_mod: float) -> List[float]:
    price_ratio = max(0.2, min(1.5, price_ratio))
    stock_n = max(0.0, min(1.0, stock / 60.0))
    hrs_n = max(0.0, min(1.0, hours_to_expiry / 120.0))
    dem = max(-1.0, min(1.0, demand_mod))
    return [price_ratio, stock_n, hrs_n, dem]


def rule_policy(price_ratio: float, stock: int, hours_to_expiry: float, demand_mod: float) -> int:
    """Transparent fallback: returns action index 0..3."""
    urgency = 0.0
    if hours_to_expiry <= 12:
        urgency += 3
    elif hours_to_expiry <= 24:
        urgency += 2
    elif hours_to_expiry <= 48:
        urgency += 1
    if stock >= 40:
        urgency += 1
    urgency -= demand_mod * 2.0  # strong demand cancels urgency
    if price_ratio < 0.7:
        urgency -= 1  # already deeply discounted
    if urgency >= 3:
        return 0  # -10%
    if urgency >= 1.5:
        return 1  # -5%
    if urgency <= -1.5 and price_ratio < 1.0:
        return 3  # +5% recover margin on surging demand
    return 2  # hold


if HAS_TORCH:

    class QNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 4),
            )

        def forward(self, x):
            return self.net(x)

    def _train_qnet(epochs: int = 60) -> "QNet":
        """Distill the rule policy into the MLP on synthetic states."""
        import random

        net = QNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            batch_x, batch_y = [], []
            for _ in range(256):
                pr = random.uniform(0.5, 1.1)
                st = random.randint(0, 60)
                hrs = random.uniform(0, 120)
                dem = random.uniform(-1, 1)
                batch_x.append(state_features(pr, st, hrs, dem))
                batch_y.append(rule_policy(pr, st, hrs, dem))
            xb = torch.tensor(batch_x, dtype=torch.float32)
            yb = torch.tensor(batch_y, dtype=torch.long)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
        net.eval()
        return net


class ElasticityEngine:
    def __init__(self) -> None:
        self.backend = "rules"
        self._net = None
        if HAS_TORCH:
            try:
                if WEIGHTS_PATH.exists():
                    self._net = QNet()
                    self._net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
                    self._net.eval()
                    self.backend = "pytorch-dqn"
                else:
                    self._net = _train_qnet()
                    self.backend = "pytorch-dqn-lite"
            except Exception:
                self._net = None
                self.backend = "rules"

    def decide(self, current_price: float, base_price: float, cost_basis: float,
               stock: int, hours_to_expiry: float, demand_mod: float) -> Tuple[float, int, str]:
        """Returns (new_price, action_idx, policy_note). Never prices below cost."""
        price_ratio = current_price / max(base_price, 1e-6)
        if self._net is not None:
            with torch.no_grad():
                q = self._net(torch.tensor([state_features(price_ratio, stock, hours_to_expiry, demand_mod)],
                                           dtype=torch.float32))
                action = int(torch.argmax(q, dim=1).item())
            note = self.backend
        else:
            action = rule_policy(price_ratio, stock, hours_to_expiry, demand_mod)
            note = "rule-policy"
        floor = round(cost_basis * 1.0, 2)
        cap = round(base_price * 1.5, 2)  # same cap the DQN was trained with
        new_price = round(min(cap, max(floor, current_price * (1 + ACTIONS[action]))), 2)
        if new_price == current_price and action != 2:
            action = 2  # rounding/floor absorbed the move -> report hold
        return new_price, action, note
