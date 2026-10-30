# ElasticShelf MVP — live store simulation

Headless pricing intelligence + simulated ESL fleet and POS, implementing
`technical_implementation_guide.md`. The full-auto store sim advances footfall,
weather, events, POS checkouts, and the context-model/PyTorch repricing loop
every tick (default 2s = 1 sim-hour), pushing markdowns to a live ESL wall
over WebSocket.

## Architecture

| Module | Guide | Implementation |
|---|---|---|
| A. Ingestion & Normalization | `POST /api/v1/inventory/sync` | `backend/main.py` + `backend/models.py` — POS/ERP payloads normalized to the `InventoryItem` schema |
| B. Context Engine | LLaMA 3 + RAG | `backend/context_engine.py` — `muse-spark-1.3-contributor` (OpenAI-compatible endpoint) over a keyword-retrieval corpus; deterministic mock fallback offline |
| C. Elasticity Engine | PyTorch RL / DQN | `backend/elasticity.py` + `backend/pricing_env.py` — trained DQN (`data/dqn_weights.pt`), rule-distilled lite net, transparent rules, or opt-in `rule` / `tuned-rule` / `mpc` policies (see below) |
| D. ESL Bridge | hardware payloads | `backend/esl_bridge.py` — generic + Vusion/ Pricer-shaped payloads, ack-logged |
| Store simulator | demo harness | `backend/simulator.py` — clock, footfall, weather walk, scripted events, elastic POS sales, repricing cooldowns |
| Storefront | ESL fleet + POS | `frontend/` — e-ink-style ESL wall, live checkout feed, context panel, push audit (`/ws/stream`, REST fallback) |

Pricing loop per SKU per tick: context engine emits a demand modifier in
[-1, +1] with one-line reasoning → elasticity engine maps
(price, stock, hours-to-expiry, modifier) to −10/−5/0/+5% → ESL Bridge pushes
changes (price floor = cost, cap = 1.5× base, max one markdown per 3 ticks
unless expiring within 24h). The policy is selectable via `ELASTICITY_POLICY`
(`auto` = DQN first, the default; `dqn` / `rule` / `tuned-rule` / `mpc`).

## Quickstart

```
cd elasticshelf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.generate_synthetic   # 24 SKUs + 10k transactions (offline)
python -m scripts.fetch_open_data      # Open Food Facts + Open-Meteo (needs net; skips gracefully)
uvicorn backend.main:app --port 8000
```

Open http://127.0.0.1:8000 in a browser, press **Start**.

## Data layer (hybrid)

| Modality | Source | File |
|---|---|---|
| POS transactions (10k) | synthetic, seeded, realistic elasticity (~1–2) | `data/transactions.csv` |
| Catalog w/ expiry + stock | synthetic, seeded (24 perishable SKUs) | `data/catalog.json` |
| Real transactions (~106k) | UCI Online Retail download | `data/uci_online_retail.xlsx` |
| Demand model | elasticity fit + real UCI hourly curve | `data/demand_model.json` |
| Trained DQN | `scripts/train_dqn.py` (3500 eps) | `data/dqn_weights.pt`, `data/dqn_meta.json` |
| Tuned rule params | `scripts/tune_rule.py` (random search, no torch) | `data/rule_params.json` |
| Margin↔waste frontier | `scripts/tune_rule.py --sweep` | `data/rule_pareto.json` |
| Product names/images | Open Food Facts API | `data/open_catalog.json` |
| Weather forecast | Open-Meteo API | `data/weather.json` |
| Footfall sensor stream | simulator (time-of-day × weather × events) | live |
| Local events calendar | simulator script (market, match day, festival, parade) | live |

Regenerate/refresh: `generate_synthetic` (offline) → `fetch_open_data`
(optional net) → `prepare_training_data` (UCI + elasticity fit) → `train_dqn`
→ `tune_rule` (tuned-rule params + optional Pareto sweep, no torch needed).

## DQN training

`scripts/train_dqn.py` trains a Deep Q-Network (MLP 4→64→64→4, replay buffer
20k, batch 128, target sync every 300 steps, γ=0.95, ε 1.0→0.05) on a
single-SKU lifecycle environment: each episode draws a product and steps
hour-by-hour; demand is sampled from the fitted elasticity model and the real
UCI footfall curve; reward = realized margin per hour minus a 2×-cost waste
penalty on expired leftover stock.

Latest eval (`data/dqn_meta.json`, 200 held-out episodes, 3500 training eps):

| Policy | Reward | Sellout | Waste | Deepest discount |
|---|---|---|---|---|
| DQN | **+10.09** | 56% | 12% | 14% |
| Rules | +4.88 | 78% | 7% | 29% |

The DQN earns ~2× the margin-adjusted return with shallower discounts by
holding price on safe stock and marking down only at-risk inventory
(spot-checked: expiring overstock → −5%, demand surge → +5%, cost floor → hold).
Retrain: `python -m scripts.train_dqn --episodes 3500`.

## Pricing policies (margin-vs-waste tradeoff)

No retraining needed — pick a policy with `ELASTICITY_POLICY`:

| Policy | Backend | Behavior |
|---|---|---|
| `auto` (default) | DQN → lite → rules | best available backend |
| `dqn` | trained DQN only | falls back to rules if torch/weights missing |
| `rule` | transparent thresholds | legacy urgency-score policy |
| `tuned-rule` | `data/rule_params.json` | random-searched thresholds, served from file |
| `mpc` | model-predictive rollout | one-step lookahead on the fitted demand model; `ELASTICITY_WASTE_WEIGHT` tunes margin↔waste live |
| `ppo` | trained PPO actor-critic | `data/ppo_weights.pt` (see below); falls back to rules (`ppo-rules-fallback`) if torch/weights missing |

`scripts/tune_rule.py` searches rule thresholds on the exact DQN training
dynamics (`backend/pricing_env.py`, shared with `train_dqn.py`), so results
are directly comparable to the DQN:

```
python -m scripts.tune_rule --trials 80            # writes data/rule_params.json
python -m scripts.tune_rule --sweep 0.5,1,2,4,8 --include-mpc   # writes data/rule_pareto.json
```

Sweep snapshot (`data/rule_pareto.json`, reward at each waste weight λ):

| λ | Default rule | Tuned rule | MPC |
|---|---|---|---|
| 0.5 | +6.73 | +11.15 | +12.18 |
| 1.0 | +6.31 | +10.66 | +11.43 |
| 2.0 | +5.47 | +10.11 | +8.75 |
| 4.0 | +3.78 | +9.08 | +7.04 |
| 8.0 | +0.41 | +7.02 | +2.53 |

At the default λ=2 the tuned rule matches the DQN (+10.1 vs +10.1) with
lower waste (4% vs 12%), while MPC wins when waste is cheap and the tuned
rule wins when waste is expensive. The `policy` field on each `pricing`
event records which backend priced that SKU (`pytorch-dqn`, `rule-tuned`,
`mpc-w2`, …).

## PPO (experimental alternative to DQN)

`scripts/train_ppo.py` trains a Proximal Policy Optimization actor-critic
(shared-trunk MLP 4→64→64 with policy + value heads, clipped surrogate,
GAE λ=0.95) on the *same* `DemandEnv`, reward, and held-out evaluation as
the DQN, so `data/ppo_meta.json` reports PPO vs rule vs DQN `rollout_stats`
on identical seeds (reward, margin, sellout, waste, deepest discount):

```
python -m scripts.train_ppo --iterations 300   # writes data/ppo_weights.pt + data/ppo_meta.json (shipped weights)
ELASTICITY_POLICY=ppo uvicorn backend.main:app --port 8000
python -m tests.test_ppo
```

Where PPO is expected to help vs the DQN/rules (confirmed or refuted by
each retrain's `ppo_meta.json` comparison): stabler on-policy updates on
this small 4-state/4-action problem, smoother exploration of the
markdown-vs-hold tradeoff via a stochastic policy, and no replay buffer /
target network to tune. Where it may not: sample efficiency (the DQN
reuses replay) and the tuned rule's low waste rate — PPO should only
replace the default if it beats tuned-rule on reward *and* waste.

Latest eval (`data/ppo_meta.json`, one reproducible protocol —
`DemandEnv(seed=999)` × 200 rollouts for every row; revenue and profit
(margin) maximised subject to aggressively minimised waste):

All rows below are **served** through `ElasticityEngine.decide` (same
markdown-only, expiry-ladder, cost+5% margin-floor invariants for every
policy). From `scripts/evaluate_policies.py` (`data/policy_eval.json`,
5 seeds × 100 episodes per seed, unit-weighted waste = actual leftover
units / initial units):

| Policy (served) | Revenue | Profit (margin) | Reward | Sellout | **Units wasted** |
|---|---|---|---|---|---|
| **PPO + ladder** | **26.84** | **8.56** | +7.30 | 85% | 3.3% |
| DQN + ladder | 25.13 | 8.30 | +7.09 | 84% | 3.3% |
| Rule + ladder | 24.30 | 6.47 | +5.60 | 87% | **2.6%** |
| Tuned rule + ladder | 23.74 | 6.84 | +5.69 | 84% | 3.4% |
| MPC + ladder | 21.68 | 3.79 | +2.42 | 87% | 3.3% |

On **adversarial short-dated** episodes (30–60 units, 6–48h to expiry)
every policy wastes **37–39%** — demand physics dominates (max sell
rate at the floor vs stock), MPC marginally best (36.9%). No pricing
policy can clear stock that cannot be sold in time; that waste must be
prevented at ordering/stocking, not pricing.

**Realism fixes (after a reward-hacking audit found live "0 waste" was
a simulator artifact):** (1) the live simulator now draws demand from
ONE shared footfall pool (`HOURLY_SHOPPER_UNITS`, allocated across SKUs
by fitted category elasticity ~1–2) — discounts reallocate demand, they
no longer create shoppers per-SKU; (2) markdowns are spaced (one per
SKU per 6 ticks, final-6h excepted) and the engine floor is cost + 5%,
so hourly repricing cannot compound straight to bare cost; (3)
evaluation reports unit-weighted waste across seeds plus the
adversarial scenario above. Under the fixed simulator the same live
catalog run wastes **20.3%** served (PPO + early ladder below) vs
**29.7%** at forced-hold (5 seeds; revenue $3,060 vs $3,797 — waste
reduction still costs margin; anyone claiming both free should be
distrusted). Earlier
"0 units wasted live" numbers were produced by the old per-SKU
demand model and are retracted.

**Waste optimisation round 2 (live simulator is the target metric).**
Diagnosis: waste concentrates in high-stock mid-life SKUs (avocado
23/55, tomatoes 22/58 in one run) — a stock-cover problem. Searched:
(i) SB3-canonical PPO settings (lr 3e-4–1e-3, 10 epochs, minibatch
64) and waste×8–16 + shaping retrains — best raw candidate
(`waste×16, shaping 0.1`) took live waste 22.7% → 21.2%; (ii) PPO
trained directly in the multi-SKU shared-pool simulator
(`scripts/train_ppo_live.py`, per-SKU waste credit assignment) —
raw waste 262 → 239 units/episode, but served-identical once the
ladder/spacing stack applies; (iii) served ladder variants — the
winner is the **early ladder** the live simulator now requests
(`waste_ladder="early"`: ≤36h → −10%, ≤96h & stock ≥25 → −5%):
live units wasted **22.7% → 18.2%** on seeds {1,42,99} (20.3%
across 5 seeds; seed variance is ±3pts) for ~9% less revenue than
the standard-ladder served policy on the same seeds. Cover-ratio
and even-earlier ladders wasted more or earned less, and the early
ladder is deliberately *not* the engine default: in the hourly-step
training env it fire-sales (revenue −22%), because only the live
simulator spaces markdowns to one per 6 ticks. Structural limit:
with a shared pool, pricing mostly *reallocates* demand — the
remaining waste is bounded by total footfall vs stock/expiry
schedule, i.e. an ordering problem (see adversarial results).

**Business invariants (all policies, enforced in `ElasticityEngine.decide`):**
prices are *markdown-only* — the +5% action is masked for PPO in training
and at serve time, and clamped to hold for DQN/rule/MPC — and expired
stock (`hours_to_expiry <= 0`) is never repriced (`expired-hold`; the
simulator also skips expired SKUs entirely, including the LLM call).
Earlier versions allowed +5% "margin recovery" on safe stock; that was
removed because shelf prices must only ever go down.

## Context engine

Default model `muse-spark-1.3-contributor` via `LLAMA_BASE_URL` +
`SPARK_API_KEY` (OpenAI-compatible `/chat/completions`, strict two-line
`MODIFIER:`/`REASON:` format). Groq/Together remain available through
`LLAMA_PROVIDER`. Retrieval is a dependency-free keyword scorer over weather /
event / history notes (FAISS stand-in). With no credentials the engine runs a
deterministic mock (rain/footfall/heat/event/expiry heuristics), so the demo
always works offline. Provider + model are shown in the dashboard footer and
`GET /api/v1/context`.

## Simulation loop

- Clock starts at 09:00; each tick = 1 sim-hour (adjustable via
  `POST /api/v1/sim/control`, 0.25–12h).
- Footfall = time-of-day curve × rain penalty × event effects (match-day
  surge, festival diversion) × noise.
- Per-SKU sales = Poisson demand with signed price elasticity (discounts lift,
  hikes cut), +40% urgency under 24h to expiry; expired leftovers → waste.
- Every tick runs the full pricing loop and broadcasts `checkouts`, `pricing`
  (with `llm_reasoning` audit), `context`, and `metrics` on `/ws/stream`.

## API reference

- `POST /api/v1/inventory/sync` — ingest POS/ERP inventory
  `{store_id, items: [{sku, name, quantity_on_shelf, base_price, cost_basis, expiration_timestamp, category?, image_url?}]}`
- `POST /api/v1/pricing/evaluate` — run context+DQN loop, push ESL updates →
  `{status, updates_pushed, hardware_sync_status, audit_log: [{sku, old_price, new_price, llm_reasoning}]}`
- `GET /api/v1/inventory` — current shelf state (prices, stock, reasoning)
- `GET /api/v1/context` — footfall, weather, event, engine, llm provider/model
- `GET /api/v1/pos/feed` — recent checkout tickets
- `POST /api/v1/sim/{start,stop,tick,reset,control}` · `GET /api/v1/sim/status`
- `WS /ws/stream` — `hello` snapshot then `tick` / `pricing` / `reset` events

## Configuration

Copy `.env.example` to `.env`:

| Var | Default | Purpose |
|---|---|---|
| `LLAMA_PROVIDER` | `spark` | `spark` \| `groq` \| `together` \| `none` (mock) |
| `LLAMA_MODEL` | `muse-spark-1.3-contributor` | context model id |
| `LLAMA_BASE_URL` | — | OpenAI-compatible base URL for `spark` |
| `SPARK_API_KEY` / `GROQ_API_KEY` / `TOGETHER_API_KEY` | — | provider credentials |
| `STORE_ID` | `ST-4092` | demo store id |
| `SIM_TICK_SECONDS` | `2.0` | wall-clock seconds per tick (also live-adjustable) |
| `ELASTICITY_POLICY` | `auto` | `auto` \| `dqn` \| `rule` \| `tuned-rule` \| `mpc` |
| `ELASTICITY_WASTE_WEIGHT` | `2.0` | waste penalty λ for `mpc` (live tradeoff knob) |

`.env` is loaded automatically at startup (stdlib loader, real env wins).

## Project layout

```
elasticshelf/
  backend/      main.py (API+WS+static)  models.py  store.py
                simulator.py  context_engine.py  elasticity.py  esl_bridge.py
                pricing_env.py (shared DemandEnv for train/tune/eval)
  frontend/     index.html  app.js  styles.css
  scripts/      generate_synthetic.py  fetch_open_data.py
                prepare_training_data.py  train_dqn.py  tune_rule.py
  tests/        test_smoke.py (API + WS + pricing + elasticity bounds)
                test_tradeoff.py (tuned rules, MPC, lambda env)
  data/         catalog, transactions, UCI workbook, demand model,
                DQN weights, tuned rule params, Pareto frontier
```

## Tests

```
python -m tests.test_smoke
python -m tests.test_tradeoff
```

`test_smoke` covers: WebSocket hello serialization, tick + pricing loop with
the `llm_reasoning` audit contract, inventory sync round-trip, and 200
randomized elasticity bound checks (never below cost, never above 1.5× base,
valid actions). `test_tradeoff` covers: tuned-rule parametrization (defaults
reproduce legacy behavior), waste-weight/shaping env options, the MPC policy
knob, engine policy selection, and `tune_rule` search helpers. No test
framework required.

## Demo script (2 min)

1. Open the page, press **Start** — watch checkouts flow and ESL prices move.
2. Wait for a rain burst or event banner — note demand reasoning on labels.
3. Press **Step +1h** a few times to force near-expiry markdowns.
4. Show the ESL Push Audit + revenue/waste metrics as the payoff.
