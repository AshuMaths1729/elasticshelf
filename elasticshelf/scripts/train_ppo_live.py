"""Train PPO in the LIVE simulator (shared shopper pool + store hours)
with a joint profit + waste reward and a cross-SKU state.

Why: the single-SKU DemandEnv PPO (scripts/train_ppo.py) never sees the
shared pool or opening hours, so live it needed hand-set guards (expiry
ladder, waste gate). This trainer learns those trade-offs directly.

State (backend.elasticity.live_state_features, 7 dims): price/base,
stock, selling (open) hours to expiry, demand modifier, projected
leftover / stock (Simulator.projected_leftover), relative pool share,
hours until close.

Reward, per SKU decision (credited from the decision until the SKU's
next decision; /10 scale):

    r = margin on its sales  -  waste_weight * cost * units it wasted
        + team_weight * (store-wide contribution that tick) / live SKUs

waste_weight = 1 is exactly store profit; > 1 adds an explicit waste
penalty (profit maximisation and waste minimisation in one reward). The
team term charges each SKU for demand it steals from the others in the
shared pool (a markdown that only reallocates shoppers earns nothing).

Served exactly as at run time: Simulator.evaluate_pricing in "learned"
mode (no ladder / gate, markdown spacing checked before the policy is
asked, markdown-only, cost+5% floor, closed hours skipped).

Pipeline:  1) behaviour-clone the waste-first gate (warm start)
           2) fit the value head only (policy frozen)
           3) PPO fine-tune; keep the best checkpoint on selection seeds
              by the training objective (profit - (w-1) * waste cost).

Run:  python -m scripts.train_ppo_live --waste-weight 2 --out data/ppo_live_weights.pt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
except ImportError:
    raise SystemExit("torch is required: pip install -r requirements.txt")

from backend.elasticity import (ACTIONS, LIVE_STATE_DIM, ActorCritic,
                                ElasticityEngine)
from backend.models import InventoryItem
from backend.simulator import Simulator
from backend.store import Store
from scripts.generate_synthetic import CATALOG_SPEC
from scripts.train_ppo import compute_gae

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
SELECT_SEEDS = (1, 7, 42, 99, 123)
torch.set_num_threads(1)


def build_sim(seed: int, jitter: bool) -> Simulator:
    rng = random.Random(seed * 7919 + 13)
    store = Store()
    sim = Simulator(store, seed=seed)
    for sku, name, cat, price, cost, qty, hrs in CATALOG_SPEC:
        if jitter:
            qty = max(5, int(round(qty * rng.uniform(0.7, 1.4))))
            hrs = hrs * rng.uniform(0.8, 1.25)
        store.upsert(InventoryItem(
            sku=sku, name=name, category=cat, quantity_on_shelf=qty,
            base_price=price, current_price=price, cost_basis=cost,
            expiration_timestamp=sim.sim_time + timedelta(hours=hrs)))
    sim.live_profile = "learned"
    sim.waste_gate = False
    return sim


class TrainFX:
    """Drop-in for ElasticityEngine inside the simulator: samples (train),
    argmax (eval), or the waste-first gate (teacher, for cloning)."""
    wants_live_state = True

    def __init__(self, net, mode: str, teacher=None):
        self.net, self.mode, self.teacher = net, mode, teacher
        self.backend = f"ppo-live-{mode}"
        self.records = []            # (sku, state, action, logprob, value)

    def decide(self, cur, base, cost, stock, hrs, mod, live_state=None,
               sku=None, **kw):
        if self.mode == "teacher":
            action = self.teacher(cur, base, cost, stock, hrs, mod, live_state)
            lp = v = 0.0
        else:
            with torch.no_grad():
                logits, value = self.net(torch.tensor([live_state], dtype=torch.float32))
                logits = logits.clone()
                logits[:, 3] = float("-inf")      # markdown-only
                if self.mode == "train":
                    dist = Categorical(logits=logits)
                    a_t = dist.sample()
                    action, lp = int(a_t.item()), float(dist.log_prob(a_t).item())
                else:
                    action, lp = int(torch.argmax(logits, dim=1).item()), 0.0
                v = float(value.item())
        floor = float(np.ceil(cost * 1.05 * 100 - 1e-9) / 100)
        new_price = round(min(cur, max(floor, cur * (1 + ACTIONS[action]))), 2)
        executed = action if new_price != cur or action == 2 else 2
        self.records.append((sku, list(live_state), action, lp, v, executed))
        return new_price, executed, self.backend


def make_teacher(push=0.03, deep=0.4):
    """Waste-first serving (gate on top of the 4-dim PPO, ladder off),
    expressed on the live state so it can be cloned."""
    eng = ElasticityEngine(policy="ppo")

    def teacher(cur, base, cost, stock, hrs, mod, s):
        ratio = s[4]
        if ratio <= 0:
            return 2
        kw = {}
        if ratio >= deep:
            kw["min_action"] = 0
        elif ratio >= push:
            kw["min_action"] = 1
        _, a, _ = eng.decide(cur, base, cost, stock, hrs, mod,
                             waste_ladder="off", **kw)
        return a
    return teacher


def run_episode(fx: TrainFX, seed: int, jitter: bool, waste_weight: float,
                team_weight: float, collect: bool, own_weight: float = 1.0):
    sim = build_sim(seed, jitter)
    sim.fx = fx  # type: ignore
    store = sim.store
    costs = {i.sku: i.cost_basis for i in store.all()}
    open_tr: dict = {}
    traj: dict = {sku: [] for sku in costs}
    tick_total = [0.0]

    def credit(sku, r):
        tick_total[0] += r
        if sku in open_tr:
            open_tr[sku]["r"] += own_weight * r

    orig_sale, orig_waste = store.record_sale, store.record_waste

    def rec_sale(sku, qty):
        price = store.items[sku].current_price
        out = orig_sale(sku, qty)
        credit(sku, qty * (price - costs[sku]) / 10.0)
        return out

    def rec_waste(sku, qty):
        q = min(qty, store.items[sku].quantity_on_shelf)
        orig_waste(sku, qty)
        if q:
            credit(sku, -waste_weight * costs[sku] * q / 10.0)

    store.record_sale, store.record_waste = rec_sale, rec_waste

    def close(sku, done):
        t = open_tr.pop(sku)
        t["done"] = float(done)
        traj[sku].append(t)

    max_exp = max(i.expiration_timestamp for i in store.all())
    while sim.sim_time <= max_exp + timedelta(hours=2):
        fx.records.clear()
        tick_total[0] = 0.0
        sim.tick()
        # Team term: this tick's store-wide contribution, shared by every
        # transition that was open while it was earned (sales precede the
        # pricing step inside tick()).
        if collect and team_weight and open_tr:
            share = team_weight * tick_total[0] / len(open_tr)
            for t in open_tr.values():
                t["r"] += share
        for sku, state, action, lp, v, _ in fx.records:
            if not collect:
                continue
            if sku in open_tr:
                close(sku, done=False)
            open_tr[sku] = {"s": state, "a": action, "lp": lp, "v": v, "r": 0.0}
        for sku in list(open_tr):
            it = store.items[sku]
            if it.quantity_on_shelf <= 0 or it.expiration_timestamp <= sim.sim_time:
                close(sku, done=True)
    for sku in list(open_tr):
        close(sku, done=True)
    return sim, traj


def score(sim: Simulator, waste_weight: float) -> dict:
    st = sim.store
    profit = st.revenue - st.cogs - st.waste_cost
    return {"wasted": st.units_wasted, "revenue": round(st.revenue, 2),
            "profit": round(profit, 2),
            "objective": round(profit - (waste_weight - 1) * st.waste_cost, 2)}


def evaluate(net, seeds, waste_weight):
    fx = TrainFX(net, "eval")
    rows = [score(run_episode(fx, s, False, waste_weight, 0.0, False)[0], waste_weight)
            for s in seeds]
    return {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in rows[0]}


def flatten(trajs, gamma):
    S, A, LP, ADV, RET = [], [], [], [], []
    for traj in trajs:
        for seq in traj.values():
            if not seq:
                continue
            adv, ret = compute_gae([t["r"] for t in seq], [t["done"] for t in seq],
                                   [t["v"] for t in seq], 0.0, gamma=gamma)
            for t, a_, r_ in zip(seq, adv, ret):
                S.append(t["s"]); A.append(t["a"]); LP.append(t["lp"])
                ADV.append(float(a_)); RET.append(float(r_))
    return (torch.tensor(np.array(S), dtype=torch.float32),
            torch.tensor(A, dtype=torch.int64),
            torch.tensor(LP, dtype=torch.float32),
            torch.tensor(ADV, dtype=torch.float32),
            torch.tensor(RET, dtype=torch.float32))


def ppo_update(net, opt, batch, policy: bool, clip=0.2, epochs=4, mb=256,
               ent=0.01, target_kl=0.02):
    s_t, a_t, lp_t, adv_t, ret_t = batch
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    n = len(s_t)
    if not policy:
        # Value warm-up: fit the value head only, so the cloned policy
        # (shared trunk) is not disturbed before PPO starts.
        opt = torch.optim.Adam(net.value_head.parameters(), lr=1e-3)
    for _ in range(epochs):
        if policy:
            with torch.no_grad():
                lg, _ = net(s_t)
                lg[:, 3] = float("-inf")
                kl = (lp_t - Categorical(logits=lg).log_prob(a_t)).mean().item()
            if kl > target_kl:
                break  # early stop: policy moved enough this iteration
        idx = torch.randperm(n)
        for i in range(0, n, mb):
            j = idx[i:i + mb]
            logits, value = net(s_t[j])
            vloss = nn.functional.mse_loss(value, ret_t[j])
            if policy:
                logits = logits.clone()
                logits[:, 3] = float("-inf")
                dist = Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(a_t[j]) - lp_t[j])
                surr = torch.min(ratio * adv_t[j],
                                 torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[j])
                loss = -surr.mean() + 0.5 * vloss - ent * dist.entropy().mean()
            else:
                loss = vloss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
    return n


def behaviour_clone(net, episodes: int, seed: int, epochs: int = 40):
    fx = TrainFX(None, "teacher", make_teacher())
    X, Y = [], []
    for ep in range(episodes):
        fx.records.clear()
        sim = build_sim(seed * 1000 + ep, jitter=True)
        sim.fx = fx  # type: ignore
        max_exp = max(i.expiration_timestamp for i in sim.store.all())
        while sim.sim_time <= max_exp + timedelta(hours=2):
            sim.tick()
            for _, state, _, _, _, executed in fx.records:
                X.append(state); Y.append(executed)
            fx.records.clear()
    x = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.int64)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        idx = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            j = idx[i:i + 256]
            logits, _ = net(x[j])
            logits = logits.clone()
            logits[:, 3] = -1e9
            loss = nn.functional.cross_entropy(logits, y[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits, _ = net(x)
        logits[:, 3] = -1e9
        acc = (logits.argmax(1) == y).float().mean().item()
    return len(x), acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waste-weight", type=float, default=2.0)
    ap.add_argument("--team-weight", type=float, default=1.0)
    ap.add_argument("--own-weight", type=float, default=1.0,
                    help="weight on a SKU's own margin/waste (0 = pure store-level reward)")
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--episodes-per-iter", type=int, default=4)
    ap.add_argument("--bc-episodes", type=int, default=12)
    ap.add_argument("--value-iters", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-bc", action="store_true")
    ap.add_argument("--out", type=str, default=str(DATA / "ppo_live_weights.pt"))
    ap.add_argument("--meta", type=str, default=None)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.time()
    net = ActorCritic(LIVE_STATE_DIM)
    log = {"args": vars(args), "iters": []}
    if not args.no_bc:
        n, acc = behaviour_clone(net, args.bc_episodes, args.seed)
        log["bc"] = {"samples": n, "accuracy": round(acc, 3)}
        print(f"BC warm start: {n} samples, teacher-match {acc:.1%}", flush=True)
    ev = evaluate(net, SELECT_SEEDS, args.waste_weight)
    print(f"init  {ev}", flush=True)
    best, best_state = ev, {k: v.clone() for k, v in net.state_dict().items()}
    log["init"] = ev
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for it in range(1, args.iterations + 1):
        trajs = []
        fx = TrainFX(net, "train")
        for ep in range(args.episodes_per_iter):
            _, tr = run_episode(fx, args.seed * 100000 + it * 31 + ep, True,
                                args.waste_weight, args.team_weight, True,
                                own_weight=args.own_weight)
            trajs.append(tr)
        batch = flatten(trajs, args.gamma)
        policy = it > args.value_iters
        n = ppo_update(net, opt, batch, policy=policy, ent=args.ent)
        ev = evaluate(net, SELECT_SEEDS, args.waste_weight)
        log["iters"].append({"it": it, "n": n, "policy": policy, **ev})
        mark = ""
        if policy and ev["objective"] > best["objective"]:
            best, best_state = ev, {k: v.clone() for k, v in net.state_dict().items()}
            mark = "  *best"
        print(f"iter {it:3d} n={n:5d} {'ppo ' if policy else 'vfit'} {ev}{mark}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    torch.save(net.state_dict(), args.out)
    log["best_selection"] = best
    print(f"Wrote {args.out}  best on selection seeds: {best}", flush=True)
    if args.meta:
        Path(args.meta).write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
