"""Module C: PyTorch Elasticity Engine (DQN dynamic pricing).

State:  [price_ratio, stock_norm, hours_to_expiry_norm, demand_modifier]
Action: price delta in [-10%, -5%, 0%, +5%] (index 0..3).

Backends, best first:
1. "pytorch-dqn"      — torch installed + data/dqn_weights.pt trained
                        (see scripts/train_dqn.py).
2. "pytorch-dqn-lite" — torch installed, no weights: rule-distilled MLP.
3. "rules"            — torch missing: transparent rule policy.

Opt-in policies for margin-vs-waste control (no retraining needed):
- policy="rule"       — always the default rule (ignores DQN weights).
- policy="tuned-rule" — rule with data/rule_params.json (scripts/tune_rule.py).
- policy="mpc"        — model-predictive rollout on the fitted demand model;
                        waste_weight tunes the tradeoff live.
- policy="ppo"        — Proximal Policy Optimization actor-critic
                        (scripts/train_ppo.py, data/ppo_weights.pt); falls
                        back to the transparent rule policy when torch or
                        trained weights are unavailable.
  Select via ElasticityEngine(policy=...) or the ELASTICITY_POLICY /
  ELASTICITY_WASTE_WEIGHT env vars. Default "auto" keeps legacy behavior.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ACTIONS = (-0.10, -0.05, 0.0, 0.05)
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "dqn_weights.pt"
PPO_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "ppo_weights.pt"
RULE_PARAMS_PATH = Path(__file__).resolve().parent.parent / "data" / "rule_params.json"
DEMAND_PATH = Path(__file__).resolve().parent.parent / "data" / "demand_model.json"

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


RULE_DEFAULTS: Dict[str, float] = {
    "hrs_urgent": 12.0, "w_urgent": 3.0,
    "hrs_soon": 24.0, "w_soon": 2.0,
    "hrs_watch": 48.0, "w_watch": 1.0,
    "stock_big": 40.0, "w_stock": 1.0,
    "dem_gain": 2.0,
    "deep_ratio": 0.7, "w_deep": 1.0,
    "hi_thresh": 3.0, "lo_thresh": 1.5, "recover_thresh": -1.5,
}


def rule_policy(price_ratio: float, stock: int, hours_to_expiry: float, demand_mod: float,
                params: Optional[Dict[str, float]] = None) -> int:
    """Transparent fallback: returns action index 0..3.

    params=None (or {}) reproduces the legacy thresholds exactly; pass a
    dict of RULE_DEFAULTS overrides (see scripts/tune_rule.py) to retune.
    """
    p = dict(RULE_DEFAULTS)
    if params:
        for k, v in params.items():
            if k in p:
                p[k] = float(v)
    urgency = 0.0
    if hours_to_expiry <= p["hrs_urgent"]:
        urgency += p["w_urgent"]
    elif hours_to_expiry <= p["hrs_soon"]:
        urgency += p["w_soon"]
    elif hours_to_expiry <= p["hrs_watch"]:
        urgency += p["w_watch"]
    if stock >= p["stock_big"]:
        urgency += p["w_stock"]
    urgency -= demand_mod * p["dem_gain"]  # strong demand cancels urgency
    if price_ratio < p["deep_ratio"]:
        urgency -= p["w_deep"]  # already deeply discounted
    if urgency >= p["hi_thresh"]:
        return 0  # -10%
    if urgency >= p["lo_thresh"]:
        return 1  # -5%
    if urgency <= p["recover_thresh"] and price_ratio < 1.0:
        return 3  # +5% recover margin on surging demand
    return 2  # hold


def load_rule_params(path: Optional[str] = None) -> Dict[str, float]:
    """Load tuned rule params; unknown keys ignored, missing file -> defaults."""
    params = dict(RULE_DEFAULTS)
    candidate = Path(path) if path else RULE_PARAMS_PATH
    try:
        if candidate.exists():
            raw = json.loads(candidate.read_text())
            inner = raw.get("params", raw) if isinstance(raw, dict) else {}
            if isinstance(inner, dict):
                for k, v in inner.items():
                    if k in params:
                        params[k] = float(v)
    except Exception:
        pass
    return params


_demand_model_cache: Optional[dict] = None


def _load_demand_model() -> Optional[dict]:
    global _demand_model_cache
    if _demand_model_cache is not None:
        return _demand_model_cache
    try:
        _demand_model_cache = json.loads(DEMAND_PATH.read_text())
    except Exception:
        _demand_model_cache = None
    return _demand_model_cache


def demand_for_category(category: Optional[str] = None) -> Tuple[float, float]:
    """(base_qty, elasticity) for a category, or the cross-category average."""
    model = _load_demand_model()
    try:
        cats = (model or {}).get("categories", {})
        if category and category in cats:
            c = cats[category]
            return float(c["base_qty"]), float(c["elasticity"])
        if cats:
            bq = sum(float(c["base_qty"]) for c in cats.values()) / len(cats)
            el = sum(float(c["elasticity"]) for c in cats.values()) / len(cats)
            return bq, el
    except Exception:
        pass
    return 1.5, 1.4


def hourly_factor(hour: Optional[int] = None) -> float:
    """Footfall multiplier for an hour of day (1.0 when unknown)."""
    if hour is None:
        return 1.0
    try:
        curve = (_load_demand_model() or {}).get("hourly_curve")
        if curve:
            return float(curve[int(hour) % 24]) or 1.0
    except Exception:
        pass
    return 1.0


def mpc_action(price: float, base_price: float, cost: float, stock: int,
               hours_to_expiry: float, demand_mod: float,
               waste_weight: float = 2.0, horizon: int = 120,
               base_qty: Optional[float] = None, elasticity: Optional[float] = None,
               footfall: float = 1.0, demand_scale: float = 0.5,
               price_cap: float = 1.5) -> int:
    """One-step model-predictive pick: try each action, roll the expected
    demand forward (hold afterwards), keep the best margin - waste trade.

    Uses the same Poisson-mean demand as training; deterministic, no torch.
    Ties prefer hold (index 2) so equal-value markdowns are not pushed.
    """
    if stock <= 0 or hours_to_expiry <= 0:
        return 2
    if base_qty is None or elasticity is None:
        avg_qty, avg_el = demand_for_category()
        base_qty = avg_qty if base_qty is None else base_qty
        elasticity = avg_el if elasticity is None else elasticity
    base = max(base_price, 1e-6)
    dem = max(-1.0, min(1.0, demand_mod))
    steps = max(1, min(int(math.ceil(hours_to_expiry)), max(1, int(horizon))))
    expires_inside = hours_to_expiry <= max(1, int(horizon))
    best_a, best_v = 2, float("-inf")
    for a in (2, 3, 1, 0):  # hold wins ties
        p = min(base * price_cap, max(cost, price * (1 + ACTIONS[a])))
        rel = max(-0.5, min(0.6, (base - p) / base))
        lam = max(0.0, demand_scale * base_qty
                  * max(0.05, 1 + elasticity * rel) * footfall * (1 + 0.5 * dem))
        s = float(stock)
        v = 0.0
        for _ in range(steps):
            sold = min(s, lam)
            s -= sold
            v += (p - cost) * sold / 10.0
            if s <= 1e-9:
                s = 0.0
                break
        if s > 0 and expires_inside:
            v -= waste_weight * cost * s / 10.0
        if v > best_v + 1e-9:
            best_a, best_v = a, v
    return best_a


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

    class ActorCritic(nn.Module):
        """PPO actor-critic: shared trunk, policy (logits) + value heads.

        Same 4-dim state / 4-action space as the DQN so PPO, DQN, rules,
        tuned-rule and MPC are directly comparable on DemandEnv rollouts.
        Trained by scripts/train_ppo.py; served when policy="ppo".
        """

        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
            )
            self.policy_head = nn.Linear(64, 4)
            self.value_head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.trunk(x)
            return self.policy_head(h), self.value_head(h).squeeze(-1)

        def act_greedy(self, state) -> int:
            with torch.no_grad():
                logits, _ = self(torch.tensor([state], dtype=torch.float32))
                return int(torch.argmax(logits, dim=1).item())

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
    def __init__(self, policy: str = "auto",
                 rule_params: Optional[dict] = None,
                 waste_weight: Optional[float] = None,
                 mpc_horizon: int = 120) -> None:
        if policy == "auto":
            policy = os.getenv("ELASTICITY_POLICY", "auto")
        self._policy = (policy or "auto").lower()
        if waste_weight is None:
            try:
                waste_weight = float(os.getenv("ELASTICITY_WASTE_WEIGHT", "2.0"))
            except ValueError:
                waste_weight = 2.0
        self._waste_weight = float(waste_weight)
        self._mpc_horizon = int(mpc_horizon)
        if isinstance(rule_params, dict):
            merged = dict(RULE_DEFAULTS)
            for k, v in rule_params.items():
                if k in merged:
                    merged[k] = float(v)
            self._rule_params = merged
        elif rule_params:
            self._rule_params = load_rule_params(str(rule_params))
        elif self._policy == "tuned-rule":
            self._rule_params = load_rule_params()
        else:
            self._rule_params = dict(RULE_DEFAULTS)
        self.backend = "rules"
        self._net = None
        self._ppo_net = None
        if self._policy in ("auto", "dqn"):
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
        elif self._policy == "mpc":
            self.backend = "mpc"
        elif self._policy == "tuned-rule":
            self.backend = "rules-tuned"
        elif self._policy == "ppo":
            if HAS_TORCH:
                try:
                    if PPO_WEIGHTS_PATH.exists():
                        self._ppo_net = ActorCritic()
                        self._ppo_net.load_state_dict(
                            torch.load(PPO_WEIGHTS_PATH, map_location="cpu"))
                        self._ppo_net.eval()
                        self.backend = "pytorch-ppo"
                    else:
                        self.backend = "ppo-rules-fallback"
                except Exception:
                    self._ppo_net = None
                    self.backend = "ppo-rules-fallback"

    def decide(self, current_price: float, base_price: float, cost_basis: float,
               stock: int, hours_to_expiry: float, demand_mod: float,
               category: Optional[str] = None, footfall: Optional[float] = None,
               hour: Optional[int] = None, base_qty: Optional[float] = None,
               elasticity: Optional[float] = None) -> Tuple[float, int, str]:
        """Returns (new_price, action_idx, policy_note). Never prices below cost."""
        price_ratio = current_price / max(base_price, 1e-6)
        if self._policy == "mpc":
            if base_qty is None or elasticity is None:
                cbq, cel = demand_for_category(category)
                base_qty = cbq if base_qty is None else base_qty
                elasticity = cel if elasticity is None else elasticity
            ff = footfall if footfall is not None else hourly_factor(hour)
            action = mpc_action(current_price, base_price, cost_basis, stock,
                                hours_to_expiry, demand_mod,
                                waste_weight=self._waste_weight,
                                horizon=self._mpc_horizon,
                                base_qty=base_qty, elasticity=elasticity,
                                footfall=ff)
            note = f"mpc-w{self._waste_weight:g}"
        elif self._policy in ("rule", "tuned-rule"):
            action = rule_policy(price_ratio, stock, hours_to_expiry, demand_mod,
                                 params=self._rule_params)
            note = "rule-tuned" if self._policy == "tuned-rule" else "rule-policy"
        elif self._policy == "ppo" and self._ppo_net is not None:
            action = self._ppo_net.act_greedy(
                state_features(price_ratio, stock, hours_to_expiry, demand_mod))
            note = "pytorch-ppo"
        elif self._net is not None:
            with torch.no_grad():
                q = self._net(torch.tensor([state_features(price_ratio, stock, hours_to_expiry, demand_mod)],
                                           dtype=torch.float32))
                action = int(torch.argmax(q, dim=1).item())
            note = self.backend
        else:
            action = rule_policy(price_ratio, stock, hours_to_expiry, demand_mod)
            note = "ppo-rules-fallback" if self._policy == "ppo" else "rule-policy"
        # Cent-safe bounds: never below cost, never above 1.5x base.
        floor = math.ceil(cost_basis * 100 - 1e-9) / 100
        cap = math.floor(base_price * 1.5 * 100 + 1e-9) / 100
        new_price = round(min(cap, max(floor, current_price * (1 + ACTIONS[action]))), 2)
        if new_price == current_price and action != 2:
            action = 2  # rounding/floor absorbed the move -> report hold
        return new_price, action, note
