"""Train PPO directly in the LIVE multi-SKU shared-pool simulator.

Why this exists: the single-SKU DemandEnv (scripts/train_ppo.py) cannot
teach allocation — in the live store 24 SKUs share one footfall pool, so
a markdown only wins demand stolen from other SKUs. Policies trained
single-SKU barely move live waste when swapped (22.7% shipped vs 21.2%
best waste-tuned variant). This trainer runs episodes of the actual
Simulator (shared pool, fitted elasticity, spaced markdowns disabled
during training so actions execute as chosen) with a shared ActorCritic
choosing per-SKU actions, and per-SKU credit assignment:

    reward_sku(t) = margin from that SKU's sales in the next tick / 10
                    - waste_weight * cost * units of it wasted / 10

GAE is computed per-SKU trajectory, then all transitions are batched for
the usual clipped PPO update (markdown-only mask, as served).

Run:  python -m scripts.train_ppo_live [--iterations 12] [--waste-weight 8]
      (writes /tmp-style candidate via --out; use --write-main to ship)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
except ImportError:
    raise SystemExit("torch is required: pip install -r requirements.txt")

from backend.elasticity import ActorCritic, state_features
from backend.models import InventoryItem
from backend.simulator import Simulator
from backend.store import Store
from scripts.generate_synthetic import CATALOG_SPEC
from scripts.train_ppo import compute_gae

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

GAMMA = 0.95


def run_episode(net: ActorCritic, seed: int, training: bool,
                waste_weight: float):
    store = Store()
    sim = Simulator(store, seed=seed)
    from datetime import timedelta
    costs = {}
    items_by_sku = {}
    for sku, name, cat, price, cost, qty, hrs in CATALOG_SPEC:
        costs[sku] = cost
        item = InventoryItem(
            sku=sku, name=name, category=cat, quantity_on_shelf=qty,
            base_price=price, current_price=price, cost_basis=cost,
            expiration_timestamp=sim.sim_time + timedelta(hours=hrs))
        items_by_sku[sku] = item
        store.upsert(item)

    wasted_now: dict = {}
    orig_waste = store.record_waste

    def rec_waste(sku, qty):
        if qty:
            wasted_now[sku] = wasted_now.get(sku, 0) + qty
        return orig_waste(sku, qty)

    store.record_waste = rec_waste

    pending: dict = {}       # sku -> (state, action, logprob, value)
    trajectories: dict = {sku: [] for sku in items_by_sku}

    # Simulator.evaluate_pricing calls fx.decide per item in store order;
    # bind sku by wrapping decide with the item loop order each tick is
    # fragile, so instead patch fx to an object keyed by a queue the
    # simulator fills in store.all() order (its iteration order).
    class _FX:
        backend = "ppo-live-train"
        queue: list = []

        def decide(self, cur, base, cost, stock, hrs, mod, category=None,
                   footfall=None, hour=None, **kw):
            sku = self.queue.pop(0) if self.queue else ""
            state = state_features(cur / max(base, 1e-6), stock, hrs, mod)
            with torch.no_grad():
                logits, value = net(torch.tensor([state], dtype=torch.float32))
                logits = logits.clone()
                logits[:, 3] = float("-inf")
                if training:
                    dist = Categorical(logits=logits)
                    a_t = dist.sample()
                    action = int(a_t.item())
                    pending[sku] = (state, action, float(dist.log_prob(a_t).item()),
                                    float(value.item()))
                else:
                    action = int(torch.argmax(logits, dim=1).item())
            floor = float(np.ceil(cost * 1.05 * 100 - 1e-9) / 100)
            from backend.elasticity import ACTIONS
            new_price = round(min(cur, max(floor, cur * (1 + ACTIONS[action]))), 2)
            if new_price == cur and action != 2:
                action = 2
            return new_price, action, "ppo-live"

    fx = _FX()
    sim.fx = fx  # type: ignore
    orig_eval = sim.evaluate_pricing

    def eval_with_queue(*a, **k):
        # Match evaluate_pricing's iteration: it calls decide for every
        # non-expired item (even stock 0), in store order.
        fx.queue = [i.sku for i in store.all()
                    if (i.expiration_timestamp - sim.sim_time).total_seconds() > 0]
        return orig_eval(*a, **k)

    sim.evaluate_pricing = eval_with_queue  # type: ignore

    max_exp = max(i.expiration_timestamp for i in store.all())
    ticks = 0
    while sim.sim_time <= max_exp and ticks < 200 and any(
            i.quantity_on_shelf > 0 for i in store.all()):
        wasted_now.clear()
        sim._last_push_tick.clear()  # actions execute as chosen in training
        result = sim.tick()
        ticks += 1
        if training:
            margin_by_sku: dict = {}
            for t in result["checkouts"]:
                for line in t["lines"]:
                    sku = line["sku"]
                    margin_by_sku[sku] = margin_by_sku.get(sku, 0.0) + \
                        line["qty"] * (line["unit_price"] - costs[sku]) / 10.0
            for sku, (state, action, lp, value) in list(pending.items()):
                r = margin_by_sku.get(sku, 0.0) \
                    - waste_weight * costs[sku] * wasted_now.get(sku, 0) / 10.0
                item = items_by_sku[sku]
                done = item.quantity_on_shelf <= 0
                trajectories[sku].append(
                    {"s": state, "a": action, "lp": lp, "r": r, "v": value,
                     "done": float(done)})
                if done:
                    del pending[sku]
    # Flush any still-pending transitions with zero reward (terminal).
    if training:
        for sku, (state, action, lp, value) in pending.items():
            trajectories[sku].append(
                {"s": state, "a": action, "lp": lp, "r": 0.0, "v": value,
                 "done": 1.0})
    return {"store": store, "trajectories": trajectories, "ticks": ticks}


def ppo_update(net, trajectories, opt, gamma=GAMMA, clip=0.2, epochs=4,
               minibatch=512, lr=None):
    states, actions, logprobs, advs, rets = [], [], [], [], []
    for traj in trajectories.values():
        if not traj:
            continue
        adv, ret = compute_gae([t["r"] for t in traj],
                               [t["done"] for t in traj],
                               [t["v"] for t in traj], 0.0, gamma=gamma)
        for t, a_, r_ in zip(traj, adv, ret):
            states.append(t["s"]); actions.append(t["a"])
            logprobs.append(t["lp"]); advs.append(float(a_)); rets.append(float(r_))
    if not states:
        return 0
    s_t = torch.tensor(np.array(states), dtype=torch.float32)
    a_t = torch.tensor(actions, dtype=torch.int64)
    lp_t = torch.tensor(logprobs, dtype=torch.float32)
    adv_t = torch.tensor(advs, dtype=torch.float32)
    ret_t = torch.tensor(rets, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    n = len(states)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, minibatch):
            mb = idx[i:i + minibatch]
            logits, value = net(s_t[mb])
            logits = logits.clone()
            logits[:, 3] = float("-inf")
            dist = Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(a_t[mb]) - lp_t[mb])
            surr = torch.min(ratio * adv_t[mb],
                             torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb])
            loss = (-surr.mean()
                    + 0.5 * nn.functional.mse_loss(value, ret_t[mb])
                    - 0.01 * dist.entropy().mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
    return n


def evaluate(net, seeds=(1, 42, 99), waste_weight=8.0):
    ws, rs = [], []
    for seed in seeds:
        out = run_episode(net, seed, training=False, waste_weight=waste_weight)
        ws.append(out["store"].units_wasted)
        rs.append(out["store"].revenue)
    return sum(ws) / len(ws), sum(rs) / len(rs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--episodes-per-iter", type=int, default=3)
    ap.add_argument("--waste-weight", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", type=str, default=str(DATA / "ppo_weights.pt"),
                    help="weights to start from (shipped PPO by default)")
    ap.add_argument("--out", type=str, default="/tmp/es-sweep/ppo_live.pt")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    net = ActorCritic()
    init = Path(args.init)
    if init.exists():
        net.load_state_dict(torch.load(init, map_location="cpu"))
        print(f"Init from {init.name}")
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    base_w, base_r = evaluate(net)
    print(f"Eval before (greedy, live sim): waste={base_w:.1f} revenue={base_r:.2f}", flush=True)
    for it in range(1, args.iterations + 1):
        trajectories: dict = {}
        for ep in range(args.episodes_per_iter):
            out = run_episode(net, seed=args.seed * 1000 + it * 17 + ep,
                              training=True, waste_weight=args.waste_weight)
            for sku, traj in out["trajectories"].items():
                trajectories.setdefault(sku, []).extend(traj)
        # Per-SKU GAE needs per-episode trajectories; our merge concatenates
        # episodes per SKU with done flags at boundaries (GAE handles via
        # done masking in compute_gae calls inside ppo_update per list —
        # concatenated lists are treated as one trajectory per SKU key,
        # with done=1.0 at each episode end flushing advantage).
        n = ppo_update(net, trajectories, opt)
        w, r = evaluate(net)
        print(f"  iter {it}/{args.iterations} transitions={n} waste={w:.1f} revenue={r:.2f}", flush=True)
    net.eval()
    torch.save(net.state_dict(), args.out)
    print(f"Wrote {args.out}  final waste={w:.1f} revenue={r:.2f} (before: waste={base_w:.1f} revenue={base_r:.2f})")


if __name__ == "__main__":
    main()
