"""Full-auto store simulator: footfall + weather + events + POS sales loop.

Each tick advances the simulated clock by `hours_per_tick` (default 1h),
draws footfall from a time-of-day curve modulated by weather/events,
converts footfall into per-SKU sales (price-elastic), expires stock past
its timestamp, then runs the pricing pipeline and emits ESL/POS events.
"""
from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .context_engine import ContextEngine
from .elasticity import ElasticityEngine, demand_for_category
from .esl_bridge import ESLBridge
from .models import ContextSnapshot, InventoryItem
from .store import Store

WEATHER_PATH = Path(__file__).resolve().parent.parent / "data" / "weather.json"


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
    def __init__(self, store: Store, seed: int = 42,
                 weather_path: Optional[Path] = None) -> None:
        self.store = store
        self.rng = random.Random(seed)
        self._wx_temps: List[float] = []
        self._wx_precip: List[float] = []
        self.weather_source = "walk"
        self._load_weather_series(weather_path or WEATHER_PATH)
        self.ctx = ContextEngine()
        self.fx = ElasticityEngine()
        self.esl = ESLBridge()
        self.running = False
        self.tick_count = 0
        self.tick_seconds = 2.0
        self.hours_per_tick = 1.0
        self.open_hour = 9
        self.close_hour = 22
        self.sim_time = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
        self._last_push_tick: Dict[str, int] = {}
        self.temperature_c = 22.0
        self.precipitation_mm = 0.0
        self.pos_feed: List[Dict] = []
        self.snapshots: List[ContextSnapshot] = []
        # Static-price twin ("without tool" baseline): same SKUs and demand
        # draws, fixed base prices. Lazily snapshotted on the first tick so
        # the startup catalog is captured; runs on its own RNG stream.
        self._base: Dict[str, Dict] = {}
        self._base_ready = False
        self._base_rng = random.Random(987)
        self.baseline: Dict = {"revenue": 0.0, "units_sold": 0,
                               "units_wasted": 0, "waste_by_sku": {},
                               "cogs": 0.0, "waste_cost": 0.0}
        self.depletion: Dict[str, Dict] = {}
        self.stock_peak: Dict[str, int] = {}
        # Serve-time pricing config (see evaluate_pricing). Hours fed to the
        # policy are *selling* hours (store open) unless disabled.
        # Defaults selected on the merged store-hours + shared-pool sim
        # (10 held-out seeds, PPO): late ladder + 6-tick spacing + selling
        # hours wasted 32.5% (hold 40.3%) at ~hold revenue; the old early
        # ladder wasted 33.8% but cost ~$430 of profit.
        self.waste_ladder = os.getenv("ELASTICITY_WASTE_LADDER", "late")
        self.selling_hours = os.getenv("ELASTICITY_SELLING_HOURS", "1") != "0"
        self.markdown_spacing = int(os.getenv("ELASTICITY_MARKDOWN_SPACING", "6"))
        self.stocked_tick: Dict[str, int] = {}
        self.history: List[Dict] = [{"tick": 0, "profit": 0.0, "waste_avoided": 0}]

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
        self.store.waste_by_sku.clear()
        self.store.cogs = 0.0
        self.store.waste_cost = 0.0
        self.store.markdown_cost = 0.0
        self._base.clear()
        self._base_ready = False
        self._base_rng.seed(987)
        self.baseline = {"revenue": 0.0, "units_sold": 0, "units_wasted": 0,
                         "waste_by_sku": {}, "cogs": 0.0, "waste_cost": 0.0}
        self.depletion.clear()
        self.stock_peak.clear()
        self.stocked_tick.clear()
        self.history = [{"tick": 0, "profit": 0.0, "waste_avoided": 0}]

    # -- tick -----------------------------------------------------------
    def current_event(self) -> Optional[str]:
        for start_tick, name in EVENTS:
            if start_tick <= self.tick_count < start_tick + 8:
                return name
        return None

    def _load_weather_series(self, path: Path) -> None:
        """Load the Open-Meteo hourly forecast (scripts/fetch_open_data.py).
        Missing or fallback-shaped files leave the random walk in place."""
        try:
            hourly = json.loads(Path(path).read_text()).get("hourly", {})
            temps = [float(v) for v in hourly.get("temperature_2m", [])]
            precip = [float(v) for v in hourly.get("precipitation", [])]
        except Exception:
            return
        if temps and len(temps) == len(precip):
            self._wx_temps, self._wx_precip = temps, precip
            self.weather_source = "forecast"

    def step_weather(self) -> None:
        if self._wx_temps:
            # Real hourly forecast, cycled per tick, plus small sensor jitter.
            i = (self.tick_count - 1) % len(self._wx_temps)
            self.temperature_c = round(max(-5, min(38, self._wx_temps[i]
                                                  + self.rng.uniform(-0.5, 0.5))), 1)
            self.precipitation_mm = round(max(0.0, self._wx_precip[i]), 1)
            return
        # Random walk with occasional rain bursts (offline fallback).
        self.temperature_c = round(max(-5, min(38, self.temperature_c + self.rng.uniform(-1.5, 1.5))), 1)
        if self.rng.random() < 0.18:
            self.precipitation_mm = round(self.rng.uniform(1, 12), 1)
        else:
            self.precipitation_mm = round(max(0.0, self.precipitation_mm - 2.0), 1)

    @property
    def is_open(self) -> bool:
        o, c = self.open_hour % 24, self.close_hour % 24
        if o == c:
            return True  # 24h store
        h = self.sim_time.hour % 24
        if c < o:  # overnight wrap, e.g. 20 -> 02
            return h >= o or h < c
        return o <= h < c

    def selling_hours_until(self, expiry: datetime) -> float:
        """Open (trading) hours between now and `expiry`; demand is zero
        while closed, so this is the clock the pricing policy should see."""
        o, c = self.open_hour % 24, self.close_hour % 24
        total = (expiry - self.sim_time).total_seconds() / 3600.0
        if total <= 0 or o == c:
            return max(0.0, total)
        t, left, hours = self.sim_time, total, 0.0
        while left > 0:
            step = min(1.0, left)
            h = t.hour
            open_ = (h >= o or h < c) if c < o else (o <= h < c)
            if open_:
                hours += step
            t += timedelta(hours=step)
            left -= step
        return hours

    def active_skus(self) -> int:
        return sum(1 for i in self.store.all()
                   if i.quantity_on_shelf > 0
                   and (i.expiration_timestamp - self.sim_time).total_seconds() > 0)

    def footfall(self) -> float:
        if not self.is_open:
            return 0.0
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

    def _poisson(self, mu: float, rng: Optional[random.Random] = None) -> int:
        """Knuth Poisson draw (mu stays small here); defaults to the sim RNG."""
        rng = rng or self.rng
        if mu <= 0:
            return 0
        import math as _math
        threshold = _math.exp(-mu)
        k, p = 0, 1.0
        while p > threshold:
            k += 1
            p *= rng.random()
        return k - 1

    @staticmethod
    def _allocate_pool(pool: int, stock: List, weights: List[float],
                       rng: random.Random) -> Dict[str, int]:
        """Hand `pool` shopper-units to SKUs by weight, capped at stock."""
        qty: Dict[str, int] = {}
        for _ in range(pool):
            live = [(sku, w) for (sku, cap), w in zip(stock, weights)
                    if qty.get(sku, 0) < cap]
            if not live:
                break
            pick = rng.choices([k for k, _ in live], weights=[w for _, w in live])[0]
            qty[pick] = qty.get(pick, 0) + 1
        return qty

    def tick(self) -> Dict:
        self.sim_time += timedelta(hours=self.hours_per_tick)
        self.tick_count += 1
        self.step_weather()
        self._baseline_ensure()
        ff = self.footfall()
        event = self.current_event()

        prev_qty = {}
        for i in self.store.all():
            prev_qty[i.sku] = i.quantity_on_shelf
            self.stock_peak[i.sku] = max(self.stock_peak.get(i.sku, 0),
                                         i.quantity_on_shelf)

        # Shared demand pool (realism fix): total units this hour come from
        # ONE footfall-driven pool; SKUs compete for it by fitted category
        # elasticity (~1-2), not an independent 2.2+9*rel firehose each.
        eligible, weights = [], []
        for item in self.store.all():
            if item.quantity_on_shelf <= 0:
                continue
            hrs_left = (item.expiration_timestamp - self.sim_time).total_seconds() / 3600.0
            if hrs_left <= 0:
                # Perishables expire overnight too; waste is booked every tick.
                self.store.record_waste(item.sku, item.quantity_on_shelf)
                continue
            if not self.is_open:
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
            qty_by_sku = self._allocate_pool(
                self._poisson(mu * self.hours_per_tick, self.rng),
                [(i.sku, i.quantity_on_shelf) for i in eligible], weights, self.rng)
            for item in eligible:
                qty = qty_by_sku.get(item.sku, 0)
                if qty > 0:
                    total = self.store.record_sale(item.sku, qty)
                    sales.append({"sku": item.sku, "name": item.name, "qty": qty,
                                  "unit_price": item.current_price, "total": total})
        self._baseline_tick(ff, trade=self.is_open)

        checkouts = self._group_checkouts(sales)
        self.pos_feed.extend(checkouts)
        self.pos_feed = self.pos_feed[-60:]
        self._record_depletions(prev_qty)

        if self.is_open:
            pricing = self.evaluate_pricing(footfall_index=ff, event=event)
            avg_mod = pricing.get("avg_modifier", 0.0)
        else:
            # Dormant overnight: no sales, no repricing on zero footfall.
            pricing = {"status": "closed", "updates_pushed": 0,
                       "hardware_sync_status": "acknowledged", "audit_log": [],
                       "avg_modifier": 0.0, "engine": self.fx.backend}
            avg_mod = self.snapshots[-1].demand_modifier if self.snapshots else 0.0
        snap = ContextSnapshot(
            sim_time=self.sim_time, footfall_index=ff,
            temperature_c=self.temperature_c, precipitation_mm=self.precipitation_mm,
            event=event, demand_modifier=avg_mod,
            weather_source=self.weather_source)
        self.snapshots.append(snap)
        self.snapshots = self.snapshots[-120:]
        self.history.append({
            "tick": self.tick_count,
            "profit": self.net_profit(),
            "waste_avoided": self.baseline["units_wasted"] - self.store.units_wasted,
        })
        self.history = self.history[-240:]
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
                                  "tick": self.tick_count,
                                  "time": self.sim_time.isoformat(),
                                  "lines": basket, "total": round(total, 2)})
                basket, total = [], 0.0
        if basket:
            checkouts.append({"id": f"T{self.tick_count}-{len(checkouts)+1:02d}",
                              "tick": self.tick_count,
                              "time": self.sim_time.isoformat(),
                              "lines": basket, "total": round(total, 2)})
        return checkouts

    # -- forecast & restock ------------------------------------------------
    def _velocity_window(self, window_ticks: int = 24):
        """(span, sku -> units sold) over the last `window_ticks` of POS."""
        span = max(1, min(window_ticks, self.tick_count))
        cutoff = self.tick_count - span
        sold: Dict[str, int] = {}
        for t in self.pos_feed:
            if t.get("tick", 0) <= cutoff:
                continue
            for line in t["lines"]:
                sold[line["sku"]] = sold.get(line["sku"], 0) + line["qty"]
        return span, sold

    def forecast(self, window_ticks: int = 24) -> List[Dict]:
        """Per-SKU demand velocity, runout and suggested order over the last
        `window_ticks` ticks of POS history (fewer when the run is young).
        Depleted SKUs report the snapshot frozen at depletion time."""
        span, sold = self._velocity_window(window_ticks)
        rows = []
        for item in self.store.all():
            dep = self.depletion.get(item.sku)
            if dep is not None and "suggested_order" in dep:
                vel, runout, suggested = dep["velocity"], None, dep["suggested_order"]
            else:
                vel = round(sold.get(item.sku, 0) / span, 3)
                runout = item.quantity_on_shelf / vel if vel > 0 else None
                suggested = max(0, math.ceil(vel * 48 - item.quantity_on_shelf)) if vel > 0 else 0
            hrs = (item.expiration_timestamp - self.sim_time).total_seconds() / 3600.0
            if hrs <= 0:
                status = "expired"
            elif item.quantity_on_shelf <= 0:
                status = "sold_out"
            elif hrs < 24:
                status = "urgent"
            elif runout is not None and runout < 24:
                status = "low"
            else:
                status = "ok"
            rows.append({"sku": item.sku, "name": item.name, "category": item.category,
                         "stock": item.quantity_on_shelf, "base_price": item.base_price,
                         "current_price": item.current_price,
                         "hours_left": round(hrs, 1), "velocity": vel,
                         "runout_ticks": runout, "status": status,
                         "suggested_order": suggested,
                         "wasted": self.store.waste_by_sku.get(item.sku, 0),
                         "wasted_baseline": self.baseline["waste_by_sku"].get(item.sku, 0),
                         "depleted_from": dep["from_qty"] if dep else None,
                         "depletion_note": dep["note"] if dep else None,
                         "forecast_source": dep.get("forecast_source") if dep else None,
                         "forecast_reason": dep.get("forecast_reason") if dep else None})
        return rows

    def note_depletion(self, sku: str, leftover: int) -> None:
        """Record how a SKU hit zero stock (previous batch size + short
        reason) and freeze its restock forecast: velocity and suggestion are
        snapshotted once and never recomputed until the SKU is restocked."""
        if leftover <= 0 or sku in self.depletion:
            return
        item = self.store.items.get(sku)
        if item is None or item.quantity_on_shelf > 0:
            return
        batch = max(self.stock_peak.pop(sku, 0), leftover)
        plural = "" if batch == 1 else "s"
        hrs = (item.expiration_timestamp - self.sim_time).total_seconds() / 3600.0
        if hrs <= 0:
            note = f"Expired: {leftover} of {batch} wasted"
        else:
            note = f"Sold out all {batch} unit{plural}"
        span, sold = self._velocity_window()
        vel = round(sold.get(sku, 0) / span, 3)
        waste_cover = leftover if hrs <= 0 else 0
        qty, reason, source = self.ctx.forecast_restock(
            sku=sku, name=item.name, category=item.category, batch=batch,
            stocked_tick=self.stocked_tick.get(sku, 0),
            depleted_tick=self.tick_count, velocity=vel,
            sold=batch - waste_cover, wasted=waste_cover,
            base_price=item.base_price, current_price=item.current_price,
            cost_basis=item.cost_basis, footfall_index=self.footfall(),
            temperature_c=self.temperature_c,
            precipitation_mm=self.precipitation_mm, event=self.current_event(),
            wasted_baseline=self.baseline["waste_by_sku"].get(sku, 0))
        self.depletion[sku] = {"from_qty": batch, "tick": self.tick_count,
                               "note": note, "velocity": vel,
                               "suggested_order": qty,
                               "forecast_source": source,
                               "forecast_reason": reason}

    def _record_depletions(self, before: Dict[str, int]) -> None:
        todo = []
        for sku, was in before.items():
            item = self.store.items.get(sku)
            if item is not None and item.quantity_on_shelf <= 0 and was > 0 and sku not in self.depletion:
                todo.append((sku, was))
        if len(todo) <= 1:
            for sku, was in todo:
                self.note_depletion(sku, was)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
                list(pool.map(lambda pair: self.note_depletion(*pair), todo))

    def restock(self, sku: str, quantity: int, expiration=None) -> Dict:
        """Add fresh units; expired lines get a new 72h shelf life unless the
        caller passes an explicit expiration. Mirrored into the twin."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        item = self.store.items[sku]  # KeyError -> 404 at the API layer
        revived = item.quantity_on_shelf <= 0
        item.quantity_on_shelf += quantity
        self.depletion.pop(sku, None)
        self.stock_peak[sku] = max(self.stock_peak.get(sku, 0),
                                   item.quantity_on_shelf)
        if revived:
            self.stocked_tick[sku] = self.tick_count
        if expiration is not None:
            item.expiration_timestamp = expiration
        elif (item.expiration_timestamp - self.sim_time).total_seconds() <= 0:
            item.expiration_timestamp = self.sim_time + timedelta(hours=72)
        self.baseline_restock(sku, quantity, item.expiration_timestamp)
        twin_qty = self._base.get(sku, {}).get("qty")
        return {"sku": sku, "quantity_on_shelf": item.quantity_on_shelf,
                "expiration_timestamp": item.expiration_timestamp,
                "twin_qty": twin_qty}

    # -- static-price twin ("without tool" baseline) -------------------------
    def baseline_profit(self) -> float:
        b = self.baseline
        return round(b["revenue"] - b["cogs"] - b["waste_cost"], 2)

    def net_profit(self) -> float:
        """Net gain over static pricing: (tool_revenue - no_tool_revenue) + waste_avoided_costs."""
        return round((self.store.revenue - self.baseline["revenue"])
                     + (self.baseline["waste_cost"] - self.store.waste_cost), 2)

    def _baseline_ensure(self) -> None:
        if self._base_ready:
            return
        for item in self.store.all():
            self._base[item.sku] = {"qty": item.quantity_on_shelf,
                                    "base": item.base_price,
                                    "cost": item.cost_basis,
                                    "expiry": item.expiration_timestamp}
        self._base_ready = True

    def baseline_upsert(self, sku: str, quantity: int, base_price: float,
                        expiration, cost_basis: float = 0.0) -> None:
        """Mirror an inventory sync (absolute quantity) into the twin."""
        if not self._base_ready:
            return
        prev_cost = self._base.get(sku, {}).get("cost", 0.0)
        self._base[sku] = {"qty": max(0, quantity), "base": base_price,
                           "cost": cost_basis or prev_cost,
                           "expiry": expiration}

    def baseline_restock(self, sku: str, add_qty: int, expiration) -> None:
        """Mirror a restock (added quantity) into the twin."""
        if not self._base_ready or sku not in self._base:
            return
        self._base[sku]["qty"] += max(0, add_qty)
        self._base[sku]["expiry"] = expiration

    def _baseline_tick(self, ff: float, trade: bool = True) -> None:
        # Static-price twin: same shared-pool demand model as the live store,
        # fixed at base price (rel = 0), on its own RNG stream.
        eligible, weights = [], []
        for item in self.store.all():
            twin = self._base.get(item.sku)
            if not twin or twin["qty"] <= 0:
                continue
            hrs_left = (twin["expiry"] - self.sim_time).total_seconds() / 3600.0
            if hrs_left <= 0:
                self.baseline["units_wasted"] += twin["qty"]
                wb = self.baseline["waste_by_sku"]
                wb[item.sku] = wb.get(item.sku, 0) + twin["qty"]
                self.baseline["waste_cost"] = round(
                    self.baseline["waste_cost"] + twin["qty"] * twin["cost"], 2)
                twin["qty"] = 0
                continue
            if not trade:
                continue
            base_qty, _ = demand_for_category(item.category)
            eligible.append((item.sku, twin["qty"]))
            weights.append(base_qty * (1.4 if hrs_left < 24 else 1.0))
        if not eligible:
            return
        pool = self._poisson(ff * HOURLY_SHOPPER_UNITS * self.hours_per_tick,
                             self._base_rng)
        for sku, qty in self._allocate_pool(pool, eligible, weights,
                                            self._base_rng).items():
            twin = self._base[sku]
            twin["qty"] -= qty
            self.baseline["revenue"] = round(
                self.baseline["revenue"] + qty * twin["base"], 2)
            self.baseline["units_sold"] += qty
            self.baseline["cogs"] = round(
                self.baseline["cogs"] + qty * twin["cost"], 2)

    # -- pricing pipeline -----------------------------------------------
    def evaluate_pricing(self, footfall_index: Optional[float] = None,
                         event: Optional[str] = None) -> Dict:
        ff = footfall_index if footfall_index is not None else self.footfall()
        event = event if event is not None else self.current_event()
        audit, pushes, mods = [], [], []
        if not self.is_open:
            # Closed: zero footfall, so a markdown sells nothing and would
            # only burn the SKU's markdown-spacing window. Hold everything.
            return {"status": "closed", "updates_pushed": 0,
                    "hardware_sync_status": "acknowledged", "audit_log": [],
                    "avg_modifier": 0.0, "engine": self.fx.backend}
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
            # Policy/ladder see selling hours: PPO/DQN were trained on a
            # 24h-trading env, so clock hours overstate time left to sell.
            sell_h = (self.selling_hours_until(item.expiration_timestamp)
                      if self.selling_hours else hrs_left)
            if sell_h <= 0:
                item.last_reasoning = (f"{reasoning} No selling hours left "
                                       "before expiry — price held.")
                continue
            new_price, action, policy = self.fx.decide(
                item.current_price, item.base_price, item.cost_basis,
                item.quantity_on_shelf, sell_h, mod,
                category=item.category, footfall=ff, hour=self.sim_time.hour,
                waste_ladder=self.waste_ladder)
            item.last_reasoning = reasoning
            # Markdown spacing (realism fix): at most one markdown per SKU
            # per 6 ticks, so hourly repricing cannot compound -10% steps
            # straight to the floor; only the final 6h may re-markdown
            # every tick for last-chance clearance.
            cooled = (self.tick_count - self._last_push_tick.get(item.sku, -99)) < self.markdown_spacing
            if cooled and new_price < item.current_price and sell_h > 6:
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
