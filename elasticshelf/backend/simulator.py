"""Full-auto store simulator: footfall + weather + events + POS sales loop.

Each tick advances the simulated clock by `hours_per_tick` (default 1h),
draws footfall from a time-of-day curve modulated by weather/events,
converts footfall into per-SKU sales (price-elastic), expires stock past
its timestamp, then runs the pricing pipeline and emits ESL/POS events.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .context_engine import ContextEngine
from .elasticity import ElasticityEngine, demand_for_category
from .esl_bridge import ESLBridge
from .models import ContextSnapshot, InventoryItem
from .store import Store


EVENTS = [
    (6, "Morning farmers market nearby"),
    (30, "Stadium match day"),
    (54, "Weekend food festival downtown"),
    (80, "Local holiday parade"),
]

DAY_CURVE = [0.15, 0.1, 0.08, 0.08, 0.12, 0.3, 0.6, 0.9, 1.0, 1.0,
             0.95, 1.1, 1.25, 1.1, 0.95, 0.9, 1.0, 1.2, 1.35, 1.15,
             0.9, 0.65, 0.4, 0.25]

# Shared shopper pool: expected total units demanded per sim-hour at
# footfall 1.0, shared across ALL SKUs (one store, one footfall — a
# discount on one SKU reallocates demand, it does not create shoppers).
HOURLY_SHOPPER_UNITS = 9.0


class Simulator:
    def __init__(self, store: Store, seed: int = 42) -> None:
        self.store = store
        self.rng = random.Random(seed)
        self.ctx = ContextEngine()
        self.fx = ElasticityEngine()
        self.esl = ESLBridge()
        self.running = False
        self.tick_count = 0
        self.tick_seconds = 2.0
        self.hours_per_tick = 1.0
        self.sim_time = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
        self._last_push_tick: Dict[str, int] = {}
        self.temperature_c = 22.0
        self.precipitation_mm = 0.0
        self.pos_feed: List[Dict] = []
        self.snapshots: List[ContextSnapshot] = []

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def reset(self) -> None:
        self.tick_count = 0
        self.sim_time = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
        self._last_push_tick.clear()
        self.temperature_c = 22.0
        self.precipitation_mm = 0.0
        self.pos_feed.clear()
        self.snapshots.clear()
        self.store.revenue = 0.0
        self.store.units_sold = 0
        self.store.units_wasted = 0

    # -- tick -----------------------------------------------------------
    def current_event(self) -> Optional[str]:
        for start_tick, name in EVENTS:
            if start_tick <= self.tick_count < start_tick + 8:
                return name
        return None

    def step_weather(self) -> None:
        # Random walk with occasional rain bursts (Open-Meteo-seeded in spirit).
        self.temperature_c = round(max(-5, min(38, self.temperature_c + self.rng.uniform(-1.5, 1.5))), 1)
        if self.rng.random() < 0.18:
            self.precipitation_mm = round(self.rng.uniform(1, 12), 1)
        else:
            self.precipitation_mm = round(max(0.0, self.precipitation_mm - 2.0), 1)

    def footfall(self) -> float:
        hour = self.sim_time.hour
        base = DAY_CURVE[hour % 24]
        if self.precipitation_mm >= 5:
            base *= 0.7
        elif self.precipitation_mm >= 1:
            base *= 0.88
        ev = self.current_event()
        if ev and ("match" in ev.lower()):
            base *= 1.3
        if ev and ("festival" in ev.lower()):
            base *= 0.75
        return round(base * self.rng.uniform(0.9, 1.1), 3)

    def _poisson(self, mu: float) -> int:
        """Knuth Poisson draw on the sim RNG (mu stays small here)."""
        if mu <= 0:
            return 0
        import math as _math
        threshold = _math.exp(-mu)
        k, p = 0, 1.0
        while p > threshold:
            k += 1
            p *= self.rng.random()
        return k - 1

    def tick(self) -> Dict:
        self.sim_time += timedelta(hours=self.hours_per_tick)
        self.tick_count += 1
        self.step_weather()
        ff = self.footfall()
        event = self.current_event()

        # Shared demand pool (realism fix): total units this hour come from
        # ONE footfall-driven pool; SKUs compete for it by fitted category
        # elasticity (~1-2), not an independent 2.2+9*rel firehose each.
        eligible, weights = [], []
        for item in self.store.all():
            if item.quantity_on_shelf <= 0:
                continue
            hrs_left = (item.expiration_timestamp - self.sim_time).total_seconds() / 3600.0
            if hrs_left <= 0:
                self.store.record_waste(item.sku, item.quantity_on_shelf)
                continue
            rel = max(-0.5, min(0.6, (item.base_price - item.current_price) / item.base_price))
            base_qty, elast = demand_for_category(item.category)
            w = base_qty * max(0.05, 1.0 + elast * rel) * (1.4 if hrs_left < 24 else 1.0)
            eligible.append(item)
            weights.append(w)

        sales: List[Dict] = []
        if eligible:
            # Store-level lift from the average discount (bounded): price
            # cuts grow the total basket modestly, never multiplicatively.
            tot_w = sum(weights)
            avg_lift = sum(w * max(-0.5, min(0.6, (i.base_price - i.current_price) / i.base_price))
                           for i, w in zip(eligible, weights)) / max(tot_w, 1e-9)
            mu = ff * HOURLY_SHOPPER_UNITS * max(0.7, min(1.3, 1.0 + avg_lift))
            pool = self._poisson(mu * self.hours_per_tick)
            qty_by_sku: Dict[str, int] = {}
            for _ in range(pool):
                live = [(i, w) for i, w in zip(eligible, weights)
                        if qty_by_sku.get(i.sku, 0) < i.quantity_on_shelf]
                if not live:
                    break
                pick = self.rng.choices([i for i, _ in live],
                                        weights=[w for _, w in live])[0]
                qty_by_sku[pick.sku] = qty_by_sku.get(pick.sku, 0) + 1
            for item in eligible:
                qty = qty_by_sku.get(item.sku, 0)
                if qty > 0:
                    total = self.store.record_sale(item.sku, qty)
                    sales.append({"sku": item.sku, "name": item.name, "qty": qty,
                                  "unit_price": item.current_price, "total": total})

        checkouts = self._group_checkouts(sales)
        self.pos_feed.extend(checkouts)
        self.pos_feed = self.pos_feed[-60:]

        pricing = self.evaluate_pricing(footfall_index=ff, event=event)
        snap = ContextSnapshot(
            sim_time=self.sim_time, footfall_index=ff,
            temperature_c=self.temperature_c, precipitation_mm=self.precipitation_mm,
            event=event, demand_modifier=pricing.get("avg_modifier", 0.0))
        self.snapshots.append(snap)
        self.snapshots = self.snapshots[-120:]
        return {"tick": self.tick_count, "sim_time": self.sim_time.isoformat(),
                "checkouts": checkouts, "pricing": pricing, "context": snap.dict()}

    def _group_checkouts(self, sales: List[Dict]) -> List[Dict]:
        # Bundle individual SKU sales into a few plausible baskets.
        checkouts, basket, total = [], [], 0.0
        for s in sales:
            basket.append(s)
            total += s["total"]
            if len(basket) >= self.rng.randint(1, 4):
                checkouts.append({"id": f"T{self.tick_count}-{len(checkouts)+1:02d}",
                                  "time": self.sim_time.isoformat(),
                                  "lines": basket, "total": round(total, 2)})
                basket, total = [], 0.0
        if basket:
            checkouts.append({"id": f"T{self.tick_count}-{len(checkouts)+1:02d}",
                              "time": self.sim_time.isoformat(),
                              "lines": basket, "total": round(total, 2)})
        return checkouts

    # -- pricing pipeline -----------------------------------------------
    def evaluate_pricing(self, footfall_index: Optional[float] = None,
                         event: Optional[str] = None) -> Dict:
        ff = footfall_index if footfall_index is not None else self.footfall()
        event = event if event is not None else self.current_event()
        audit, pushes, mods = [], [], []
        for item in self.store.all():
            hrs_left = (item.expiration_timestamp - self.sim_time).total_seconds() / 3600.0
            if hrs_left <= 0:
                # Expired: never repriced (also skips a wasted LLM call).
                # The tick sales stage writes the stock off as waste.
                item.last_reasoning = "Expired — repricing disabled."
                continue
            mod, reasoning, llm_used = self.ctx.demand_modifier(
                item.sku, item.name, item.category, hrs_left, item.quantity_on_shelf,
                ff, self.temperature_c, self.precipitation_mm, event)
            mods.append(mod)
            new_price, action, policy = self.fx.decide(
                item.current_price, item.base_price, item.cost_basis,
                item.quantity_on_shelf, hrs_left, mod,
                category=item.category, footfall=ff, hour=self.sim_time.hour,
                waste_ladder="early")
            item.last_reasoning = reasoning
            # Markdown spacing (realism fix): at most one markdown per SKU
            # per 6 ticks, so hourly repricing cannot compound -10% steps
            # straight to the floor; only the final 6h may re-markdown
            # every tick for last-chance clearance.
            cooled = (self.tick_count - self._last_push_tick.get(item.sku, -99)) < 6
            if cooled and new_price < item.current_price and hrs_left > 6:
                new_price, action = item.current_price, 2
            if new_price != item.current_price:
                self._last_push_tick[item.sku] = self.tick_count
                old = item.current_price
                item.current_price = new_price
                payload = self.esl.push(item.sku, new_price, old)
                pushes.append(payload)
                audit.append({"sku": item.sku, "old_price": old, "new_price": new_price,
                              "llm_reasoning": reasoning, "llm_used": llm_used,
                              "policy": policy, "action": action})
        return {"status": "success", "updates_pushed": len(pushes),
                "hardware_sync_status": "acknowledged", "audit_log": audit,
                "avg_modifier": round(sum(mods) / len(mods), 3) if mods else 0.0,
                "engine": self.fx.backend}
