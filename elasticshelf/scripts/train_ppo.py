"""Train a PPO dynamic-pricing agent for the Elasticity Engine.

Alternative to the DQN (scripts/train_dqn.py): Proximal Policy Optimization
with a shared-trunk actor-critic (backend.elasticity.ActorCritic), trained
on the exact same single-SKU lifecycle environment (backend.pricing_env)
and reward as the DQN, so the three learned/rule policies are directly
comparable via rollout_stats on identical seeds.

Why PPO here (hypotheses the eval below tests):
- On-policy updates + clipped surrogate -> stabler learning than DQN's
  bootstrapped Q targets on this small (4-dim state, 4-action) problem,
  where replay-buffer staleness can chase a moving demand distribution.
- Stochastic policy + entropy bonus -> smoother exploration of the
  markdown/hold tradeoff instead of epsilon-greedy jumps.
- No target network / replay buffer -> less code, fewer hyperparameters.

Writes: data/ppo_weights.pt, data/ppo_meta.json
Run:  python -m scripts.train_ppo [--iterations 80]   (from elasticshelf/)
Serve: ELASTICITY_POLICY=ppo  (backend shows "pytorch-ppo")

Non-default --waste-weight/--gamma runs write suffixed files and never
overwrite the main PPO weights (same convention as train_dqn.py).
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

from backend.elasticity import ACTIONS, ActorCritic, rule_policy
from backend.pricing_env import SHAPING, WASTE_WEIGHT, DemandEnv, rollout_stats

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

GAMMA = 0.95
GAE_LAMBDA = 0.95
CLIP = 0.2
LR = 3e-3
EPOCHS = 4
MINIBATCH = 256
ROLLOUT_STEPS = 2048
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


def collect_rollout(net: ActorCritic, env: DemandEnv, state, steps: int):
    """Run `steps` env steps with the stochastic policy; return batch + state."""
    states, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
    s = state
    for _ in range(steps):
        with torch.no_grad():
            logits, value = net(torch.tensor([s], dtype=torch.float32))
            # Markdown-only business rule: mask the +5% action (index 3)
            # so PPO explores/learns only -10% / -5% / hold.
            logits = logits.clone()
            logits[:, 3] = float("-inf")
            dist = Categorical(logits=logits)
            a = dist.sample()
        ns, r, done = env.step(int(a.item()))
        states.append(s)
        actions.append(int(a.item()))
        logprobs.append(float(dist.log_prob(a).item()))
        rewards.append(float(r))
        dones.append(float(done))
        values.append(float(value.item()))
        s = env.reset() if done else ns
    with torch.no_grad():
        _, last_value = net(torch.tensor([s], dtype=torch.float32))
    return (states, actions, logprobs, rewards, dones, values,
            float(last_value.item()), s)


def compute_gae(rewards, dones, values, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    adv = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    vals = list(values) + [last_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t + 1] * (1.0 - dones[t]) - vals[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


def train_ppo(demand: dict, iterations: int = 80, rollout_steps: int = ROLLOUT_STEPS,
              seed: int = 1, gamma: float = GAMMA, waste_weight: float = WASTE_WEIGHT,
              shaping: float = SHAPING, lr: float = LR, clip: float = CLIP,
              epochs: int = EPOCHS, minibatch: int = MINIBATCH,
              verbose: bool = True) -> ActorCritic:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = DemandEnv(demand, seed=seed, waste_weight=waste_weight, shaping=shaping)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    state = env.reset()
    for it in range(1, iterations + 1):
        states, actions, logprobs, rewards, dones, values, last_value, state = \
            collect_rollout(net, env, state, rollout_steps)
        adv, returns = compute_gae(rewards, dones, values, last_value, gamma=gamma)
        s_t = torch.tensor(np.array(states), dtype=torch.float32)
        a_t = torch.tensor(actions, dtype=torch.int64)
        lp_t = torch.tensor(logprobs, dtype=torch.float32)
        adv_t = torch.tensor(adv, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n = len(states)
        for _ in range(epochs):
            idx = torch.randperm(n)
            for i in range(0, n, minibatch):
                mb = idx[i:i + minibatch]
                logits, value = net(s_t[mb])
                logits = logits.clone()
                logits[:, 3] = float("-inf")  # markdown-only mask (as in rollout)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(a_t[mb])
                ratio = torch.exp(new_lp - lp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb]
                loss = (-torch.min(surr1, surr2).mean()
                        + VALUE_COEF * nn.functional.mse_loss(value, ret_t[mb])
                        - ENTROPY_COEF * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                opt.step()
        if verbose and (it % 10 == 0 or it == 1):
            print(f"  iter {it}/{iterations}  rollout_reward={sum(rewards):+.2f}", flush=True)
    net.eval()
    return net


def evaluate_ppo(net: ActorCritic, demand: dict, seed: int = 999,
                 waste_weight: float = WASTE_WEIGHT, shaping: float = SHAPING) -> dict:
    env = DemandEnv(demand, seed=seed, waste_weight=waste_weight, shaping=shaping)
    return rollout_stats(env, lambda s, e: net.act_greedy(s))


def evaluate_rule(demand: dict, seed: int = 999, waste_weight: float = WASTE_WEIGHT,
                  shaping: float = SHAPING) -> dict:
    env = DemandEnv(demand, seed=seed, waste_weight=waste_weight, shaping=shaping)
    return rollout_stats(
        env, lambda s, e: rule_policy(e.price / e.base_price, e.stock, e.hours, e.dem))


def evaluate_dqn(demand: dict, seed: int = 999, waste_weight: float = WASTE_WEIGHT,
                 shaping: float = SHAPING) -> dict | None:
    """Head-to-head vs the shipped DQN when its weights are present."""
    weights = DATA / "dqn_weights.pt"
    if not weights.exists():
        return None
    try:
        from backend.elasticity import QNet
        net = QNet()
        net.load_state_dict(torch.load(weights, map_location="cpu"))
        net.eval()
        env = DemandEnv(demand, seed=seed, waste_weight=waste_weight, shaping=shaping)
        with torch.no_grad():
            return rollout_stats(
                env, lambda s, e: int(torch.argmax(
                    net(torch.tensor([s], dtype=torch.float32))).item()))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=80)
    ap.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--waste-weight", type=float, default=WASTE_WEIGHT)
    ap.add_argument("--shaping", type=float, default=SHAPING)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--clip", type=float, default=CLIP)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--minibatch", type=int, default=MINIBATCH)
    args = ap.parse_args()
    if not (DATA / "demand_model.json").exists():
        raise SystemExit("data/demand_model.json missing — run scripts.prepare_training_data first")
    demand = json.loads((DATA / "demand_model.json").read_text())
    print(f"Training PPO for {args.iterations} iterations x {args.rollout_steps} steps "
          f"(gamma={args.gamma} waste_weight={args.waste_weight} shaping={args.shaping}) ...")
    net = train_ppo(demand, iterations=args.iterations, rollout_steps=args.rollout_steps,
                    seed=args.seed, gamma=args.gamma, waste_weight=args.waste_weight,
                    shaping=args.shaping, lr=args.lr, clip=args.clip,
                    epochs=args.epochs, minibatch=args.minibatch)
    ppo_stats = evaluate_ppo(net, demand, waste_weight=args.waste_weight, shaping=args.shaping)
    rule_stats = evaluate_rule(demand, waste_weight=args.waste_weight, shaping=args.shaping)
    dqn_stats = evaluate_dqn(demand, waste_weight=args.waste_weight, shaping=args.shaping)
    print(f"Eval PPO : {ppo_stats}")
    print(f"Eval rule: {rule_stats}")
    print(f"Eval DQN : {dqn_stats}")
    default_run = (args.gamma == GAMMA and args.waste_weight == WASTE_WEIGHT
                   and args.shaping == SHAPING)
    if default_run:
        weights_path, meta_path = DATA / "ppo_weights.pt", DATA / "ppo_meta.json"
    else:
        tag = f"w{args.waste_weight:g}_g{args.gamma}_s{args.shaping:g}"
        weights_path, meta_path = DATA / f"ppo_weights_{tag}.pt", DATA / f"ppo_meta_{tag}.json"
        print("Non-default tradeoff run — main PPO files left untouched.")
    torch.save(net.state_dict(), weights_path)
    meta_path.write_text(json.dumps({
        "algorithm": "ppo", "markdown_only": True, "expired_never_repriced": True,
        "expiry_ladder": "standard <=24h -10% / <=48h&stock>=30 -5%; live simulator uses early <=36h / <=96h&stock>=25 (in ElasticityEngine/Simulator)",
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps, "seed": args.seed, "gamma": args.gamma,
        "gae_lambda": GAE_LAMBDA, "clip": args.clip, "lr": args.lr,
        "waste_weight": args.waste_weight, "shaping": args.shaping,
        "eval_ppo": ppo_stats, "eval_rule": rule_stats, "eval_dqn": dqn_stats,
        "actions": list(ACTIONS)}, indent=2))
    print(f"Wrote {weights_path.name} + {meta_path.name}")


if __name__ == "__main__":
    main()
