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
| C. Elasticity Engine | PyTorch RL / DQN | `backend/elasticity.py` — trained DQN (`data/dqn_weights.pt`), rule-distilled lite net, or transparent rules |
| D. ESL Bridge | hardware payloads | `backend/esl_bridge.py` — generic + Vusion/ Pricer-shaped payloads, ack-logged |
| Store simulator | demo harness | `backend/simulator.py` — clock, footfall, weather walk, scripted events, elastic POS sales, repricing cooldowns |
| Storefront | ESL fleet + POS | `frontend/` — e-ink-style ESL wall, live checkout feed, context panel, push audit (`/ws/stream`, REST fallback) |

Pricing loop per SKU per tick: context engine emits a demand modifier in
[-1, +1] with one-line reasoning → DQN maps
(price, stock, hours-to-expiry, modifier) to −10/−5/0/+5% → ESL Bridge pushes
changes (price floor = cost, cap = 1.5× base, max one markdown per 3 ticks
unless expiring within 24h).

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
| Trained DQN | `scripts/train_dqn.py` (2500 eps) | `data/dqn_weights.pt`, `data/dqn_meta.json` |
| Product names/images | Open Food Facts API | `data/open_catalog.json` |
| Weather forecast | Open-Meteo API | `data/weather.json` |
| Footfall sensor stream | simulator (time-of-day × weather × events) | live |
| Local events calendar | simulator script (market, match day, festival, parade) | live |

Regenerate/refresh: `generate_synthetic` (offline) → `fetch_open_data`
(optional net) → `prepare_training_data` (UCI + elasticity fit) → `train_dqn`.

## DQN training

`scripts/train_dqn.py` trains a Deep Q-Network (MLP 4→64→64→4, replay buffer
20k, batch 128, target sync every 300 steps, γ=0.95, ε 1.0→0.05) on a
single-SKU lifecycle environment: each episode draws a product and steps
hour-by-hour; demand is sampled from the fitted elasticity model and the real
UCI footfall curve; reward = realized margin per hour minus a 2×-cost waste
penalty on expired leftover stock.

Latest eval (`data/dqn_meta.json`, 200 held-out episodes):

| Policy | Reward | Sellout | Waste | Deepest discount |
|---|---|---|---|---|
| DQN | **+10.73** | 64% | 7% | 19% |
| Rules | +6.34 | 82% | 5% | 26% |

The DQN earns ~70% more margin-adjusted return with shallower discounts by
holding price on safe stock and marking down only at-risk inventory
(spot-checked: expiring overstock → −5%, demand surge → +5%, cost floor → hold).
Retrain: `python -m scripts.train_dqn --episodes 2500`.

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

## Project layout

```
elasticshelf/
  backend/      main.py (API+WS+static)  models.py  store.py
                simulator.py  context_engine.py  elasticity.py  esl_bridge.py
  frontend/     index.html  app.js  styles.css
  scripts/      generate_synthetic.py  fetch_open_data.py
                prepare_training_data.py  train_dqn.py
  tests/        test_smoke.py (API + WS + pricing + elasticity bounds)
  data/         catalog, transactions, UCI workbook, demand model, DQN weights
```

## Tests

```
python -m tests.test_smoke
```

Covers: WebSocket hello serialization, tick + pricing loop with the
`llm_reasoning` audit contract, inventory sync round-trip, and 200 randomized
elasticity bound checks (never below cost, never above 1.5× base, valid
actions). No test framework required.

## Demo script (2 min)

1. Open the page, press **Start** — watch checkouts flow and ESL prices move.
2. Wait for a rain burst or event banner — note demand reasoning on labels.
3. Press **Step +1h** a few times to force near-expiry markdowns.
4. Show the ESL Push Audit + revenue/waste metrics as the payoff.
