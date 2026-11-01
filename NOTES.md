# ElasticShelf — project notes

Meta AI Hackathon: headless dynamic-pricing intelligence for grocery ESLs,
plus a live simulated store (ESL fleet + POS) for the demo.

- Repo: https://github.com/AshuMaths1729/elasticshelf (public, `main`)
- App lives in `elasticshelf/`; start here: `README.md`
- Docs: `technical_implementation_guide.md`, `Market Study.docx`, `Pitch Deck.pptx`

## Key decisions

- Brand: **ElasticShelf** (renamed from OptiShelf everywhere in code).
- Context model: **muse-spark-1.3** via Meta Model API (`muse` provider,
  OpenAI-compatible endpoint, `MODEL_API_KEY` + `MUSE_BASE_URL`);
  deterministic mock fallback keeps the demo running offline.
- Audit field: **`llm_reasoning`** (guide + code).
- Elasticity: trained **DQN** (`elasticshelf/data/dqn_weights.pt`,
  eval +10.7 vs rules +6.3); falls back to distilled-lite, then rules.
- Pricing guards: floor = cost, cap = 1.5× base, max one markdown per 3 ticks
  unless expiring within 24h.

## Data pipeline (in order)

1. `python -m scripts.generate_synthetic` — 24 SKUs + 10k POS rows (offline)
2. `python -m scripts.fetch_open_data` — Open Food Facts + Open-Meteo (net, optional)
3. `python -m scripts.prepare_training_data` — UCI download + elasticity fit
4. `python -m scripts.train_dqn --episodes 2500` — trains `data/dqn_weights.pt` //trained DQN for 3.5K episodes

## Everyday commands (from `elasticshelf/`, venv active)

- Run: `uvicorn backend.main:app --port 8000` → http://127.0.0.1:8000
- Test: `python -m tests.test_smoke`
- Env: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Secrets: copy `.env.example` to `.env` (never commit it)

## Environment quirks

- Python 3.9 on Apple Silicon; torch 2.8 CPU wheel works fine.
- No `gh` CLI (brew permissions blocked); GitHub pushes use per-command
  auth headers, never stored in the remote URL.
- Git identity is repo-scoped (`AshuMaths1729` + noreply email).
- The dev server only lives during a work session (macOS has no detach);
  restart it with the Run command above.

## left gaps
- [done] can't see that active SKUs getting less as they get expired
- [done] the time for simulation should be bound by store timings which should be settable in the simulation screen
- [done] there's no sku forecasting tab to buy stocks when they are inactive or wasted
- [done] add explanations for store context parameters
- [done] weather shouldn't be hardcoded, it should be fetched by some API or MCP call via agent based on user (store)'s location in conjunction to offline weather predicitons
- [done] explanation behind logic of "deterministic mock fallback keeps the demo running offline."
- [done] saw that price was hiked when footfall was low, how was that rule learned? what is the policy learned by DQN, maybe train DQN on a million scale data for reducing error rate
- [done] we should also show the revenue without the tool engaged as a small vs without-tool-revenue
- [done] show number of wastage with or without the tool
- [done] to show demand forecase for all SKUs when they are depleted, should show them in a different tab. These should get updated when the shole stock for a SKU gets depleted.
- [done] forecast and depleted tabs are shown but they populated data under shelf-labels, they should change to new tab view instead.
- [done] show previous stock quantity with 5-6 word small reason
- [done] what is suggest as 0 for a product
- [done] research about and add a metric to show that overall revenue is still positive though we have some revenue loss in lieu of wastage minimization.
- [done] I see that the pricing and demand forecasts run even after one update for a sku, that shouldn't be the case. Once a SKU gets depleted or expired -> it has to be restocked -> the forecast engine should see all the parameters and forecast a restock quantity -> this forecasted number shouldn't change once it has been predicted. 
- [done] Once admin clicks restock with suggested or arbitrary chosen number, the item should be removed from forecast tab and re-populate under shelf-labels tab. 
- [done] Also there shouldn't be two tabs separately for forecast and demand. Once a SKU is gone from shelf its gone, it doesn't matter if it expired or got depleted. There should only be one tab showing - how much was qnty in previous stock, how much is suggestion based on various parameters.
- "Was" parameter should be like previous SKU total quantity, like for example Caesar salad it shoud be 20 and then predicted as something. Why I see single digit number like 1 or 3.
- update env variables
- the lost/expired variables should be treated as negatiations during forecasts.
