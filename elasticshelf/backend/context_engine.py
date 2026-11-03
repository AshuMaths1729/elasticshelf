"""Module B: Context Engine (Muse + lightweight retrieval).

MVP behavior:
- Keeps a small in-memory corpus of contextual docs (weather notes, events,
  historical pricing footnotes) and retrieves the most relevant ones per SKU
  with a dependency-free keyword scorer (TF-IDF when sklearn is installed).
- Calls the configured Muse context model (default muse-spark-1.3 via the Meta
  Model API Chat Completions endpoint) when MODEL_API_KEY is set.
- Falls back to a deterministic mock reasoner offline so the demo always runs.

The engine outputs a structured demand modifier in [-1, +1] plus a
human-readable reasoning string per SKU.
"""
from __future__ import annotations

import math
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

_TOKEN = re.compile(r"[a-z0-9]+")


class ContextEngine:
    def __init__(self) -> None:
        self.provider = "muse"
        self.model_key = os.getenv("MODEL_API_KEY", "")
        self.base_url = os.getenv("MUSE_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
        self.model = os.getenv("MUSE_MODEL", "muse-spark-1.3")
        self.corpus: List[Dict] = []
        self._seed_corpus()

    # -- corpus ---------------------------------------------------------
    def _seed_corpus(self) -> None:
        self.corpus = [
            {"id": "wx-storm", "text": "Storm warning: heavy rain tomorrow, footfall expected to drop 20-30%, BBQ and picnic demand falls."},
            {"id": "wx-heat", "text": "Heatwave: cold drinks, salads, ice cream demand rises; hot prepared foods soften."},
            {"id": "ev-match", "text": "Local stadium match day: evening footfall surge near store, snacks and drinks sell out faster."},
            {"id": "ev-fest", "text": "Weekend food festival downtown: footfall diverted away from grocery aisles on Saturday."},
            {"id": "hist-chicken", "text": "History: organic chicken needed a 40% discount to clear 50 units on Thursdays with low footfall."},
            {"id": "hist-berries", "text": "History: berries spoil fast; discounts earlier than 48h before expiry cut waste by half."},
        ]

    def add_document(self, doc_id: str, text: str) -> None:
        self.corpus = [d for d in self.corpus if d["id"] != doc_id]
        self.corpus.append({"id": doc_id, "text": text})

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Keyword-overlap retrieval (FAISS stand-in for the MVP)."""
        q = set(_TOKEN.findall(query.lower()))
        scored = []
        for doc in self.corpus:
            toks = set(_TOKEN.findall(doc["text"].lower()))
            overlap = len(q & toks)
            scored.append((overlap, doc))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [d for s, d in scored[:k] if s > 0] or self.corpus[:k]

    # -- demand modifier ------------------------------------------------
    def demand_modifier(
        self,
        sku: str,
        name: str,
        category: str,
        hours_to_expiry: float,
        stock: int,
        footfall_index: float,
        temperature_c: float,
        precipitation_mm: float,
        event: Optional[str],
        use_llm: bool = False,
    ) -> Tuple[float, str, bool]:
        """Returns (modifier in [-1,1], reasoning, llm_used)."""
        query = f"{name} {category} {event or ''} temp {temperature_c} rain {precipitation_mm} footfall {footfall_index}"
        docs = self.retrieve(query)
        context = " ".join(d["text"] for d in docs)
        if use_llm and self._llm_available():
            try:
                return self._llm_call(
                    sku, name, category, hours_to_expiry, stock,
                    footfall_index, temperature_c, precipitation_mm, event, context,
                )
            except Exception as exc:  # fall through to mock
                mock_mod, mock_why = self._mock_reason(
                    name, category, hours_to_expiry, stock, footfall_index,
                    temperature_c, precipitation_mm, event)
                return mock_mod, f"{mock_why} (LLM fallback: {exc})", False
        mod, why = self._mock_reason(
            name, category, hours_to_expiry, stock, footfall_index,
            temperature_c, precipitation_mm, event)
        return mod, why, False

    def _llm_available(self) -> bool:
        return bool(requests is not None and self.model_key and self.base_url)

    def _chat_endpoint(self):
        return (f"{self.base_url}/chat/completions",
                {"Authorization": f"Bearer {self.model_key}"}, self.model)

    def _chat(self, prompt: str) -> str:
        url, headers, model = self._chat_endpoint()
        resp = requests.post(url, headers=headers, json={
            # reasoning_effort=low keeps Muse's hidden reasoning short
            # (~0.4-0.6k tokens, ~2s per call); without it the model burns
            # 2.7k-3.8k tokens thinking before the two-line answer lands.
            "model": model, "temperature": 0.2, "max_tokens": 1600,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=30)
        resp.raise_for_status()
        choices = resp.json().get("choices") or [{}]
        message = choices[0].get("message") or {}
        return message.get("content") or ""

    def _llm_call(self, sku, name, category, hours_to_expiry, stock,
                  footfall_index, temperature_c, precipitation_mm, event, context):
        prompt = (
            "You are ElasticShelf, a grocery pricing analyst. Reply with exactly two lines:\n"
            "MODIFIER: <float from -1.0 (demand collapsing, discount harder) to +1.0 (demand surging, hold price)>\n"
            "REASON: <one short sentence>\n\n"
            f"SKU: {sku} ({name}, {category}). Hours to expiry: {hours_to_expiry:.1f}. Stock: {stock}. "
            f"Footfall index: {footfall_index:.2f}. Temp: {temperature_c:.1f}C. Rain mm: {precipitation_mm:.1f}. "
            f"Event: {event or 'none'}. Retrieved context: {context}"
        )
        mod, reason = self._parse_llm(self._chat(prompt))
        return mod, reason, True

    @staticmethod
    def _parse_llm(text: str) -> Tuple[float, str]:
        mod = 0.0
        reason = text.strip().replace("\n", " ")[:220]
        for line in text.splitlines():
            if "MODIFIER" in line.upper():
                nums = re.findall(r"-?\d+\.?\d*", line)
                if nums:
                    mod = max(-1.0, min(1.0, float(nums[0])))
            if line.upper().startswith("REASON"):
                reason = line.split(":", 1)[-1].strip()[:220]
        return mod, reason or "Muse demand assessment applied."

    # -- restock forecast (Muse LLM primary, rule fallback) ---------------
    RESTOCK_PROMPT_TEMPLATE = (
        "You are ElasticShelf, a grocery replenishment analyst. Reply with exactly two lines:\n"
        "QTY: <non-negative integer: units to order>\n"
        "REASON: <one short sentence>\n\n"
        "SKU: {sku} ({name}, {category}).\n"
        "LAST_BATCH: {batch} units, stocked tick {stocked_tick}, depleted tick {depleted_tick} "
        "(lifespan {lifespan} ticks).\n"
        "VELOCITY: {velocity} units/tick over recent history.\n"
        "SOLD: {sold} units. WASTED: {wasted} units (sell-through {sell_through}).\n"
        "PRICE: base {base_price}, last {current_price}, cost {cost_basis}.\n"
        "CONTEXT NOW: footfall {footfall_index}, temp {temperature_c}C, "
        "rain {precipitation_mm}mm, event: {event}.\n"
        "NO_TOOL_WASTED: {wasted_baseline} units (what static pricing lost on this SKU).\n"
        "RULE_ORDER: {rule_qty} units (sell-through rule; wasted units are "
        "a sign of over-ordering, not demand).\n\n"
        "Order enough to cover a similar lifespan at the current run-rate, "
        "adjusted for waste risk, margin headroom, and context."
    )

    @staticmethod
    def _rule_restock_qty(velocity: float, lifespan_ticks: int, waste_cover: int) -> int:
        return math.ceil(max(0.0, velocity) * max(1, lifespan_ticks) + max(0, waste_cover))

    @classmethod
    def _restock_prompt(cls, **kw) -> str:
        sold, wasted = kw["sold"], kw["wasted"]
        filled = dict(kw)
        filled["event"] = kw["event"] or "none"
        filled["lifespan"] = max(1, kw["depleted_tick"] - kw.get("stocked_tick", 0))
        filled["sell_through"] = f"{sold / max(1, sold + wasted):.0%}"
        filled.setdefault("rule_qty", "n/a")
        return cls.RESTOCK_PROMPT_TEMPLATE.format(**filled)

    @staticmethod
    def _parse_forecast(text: str) -> Tuple[int, str]:
        qty = 0
        reason = text.strip().replace("\n", " ")[:220]
        for raw in text.splitlines():
            stripped = raw.strip()
            bare = stripped.lstrip("*# ").upper()
            if bare.startswith("QTY"):
                nums = re.findall(r"-?\d+", stripped)
                if nums:
                    qty = max(0, int(nums[0]))
            if bare.startswith("REASON"):
                reason = stripped.split(":", 1)[-1].strip().strip("* ")[:220]
        return qty, reason or "LLM restock forecast applied."

    def forecast_restock(self, **kw) -> Tuple[int, str, str]:
        """Predict a restock qty with Muse. Returns (qty, reasoning, source)
        where source is "llm" or "rule". Offline or on model error the rule
        (velocity x lifespan + waste cover) keeps a forecast available;
        LLM answers are capped at 5x the last batch."""
        lifespan = max(1, kw["depleted_tick"] - kw.get("stocked_tick", 0))
        rule = kw.pop("rule", None)  # (qty, reason) from the simulator's order policy
        if rule is not None:
            rule_qty, rule_reason = rule
        else:
            rule_qty = self._rule_restock_qty(kw["velocity"], lifespan, kw["wasted"])
            rule_reason = (f"Rule: {kw['velocity']}/tick x {lifespan} ticks "
                           f"+ {kw['wasted']} waste cover.")
        if not self._llm_available():
            return rule_qty, rule_reason, "rule"
        kw["rule_qty"] = rule_qty
        try:
            qty, reason = self._parse_forecast(self._chat(self._restock_prompt(**kw)))
        except Exception as exc:
            return rule_qty, f"Rule fallback (LLM error: {exc}).", "rule"
        cap = 5 * max(kw["batch"], 1)
        if qty > cap:
            return cap, reason + " (capped at 5x last batch).", "llm"
        return qty, reason, "llm"

    # -- deterministic mock (offline-safe) -------------------------------
    @staticmethod
    def _mock_reason(name, category, hours_to_expiry, stock, footfall_index,
                     temperature_c, precipitation_mm, event) -> Tuple[float, str]:
        mod = 0.0
        bits = []
        if precipitation_mm >= 5:
            mod -= 0.45
            bits.append(f"heavy rain ({precipitation_mm:.0f}mm) cutting footfall")
        elif precipitation_mm >= 1:
            mod -= 0.2
            bits.append("light rain softening footfall")
        if footfall_index < 0.8:
            mod -= 0.25
            bits.append(f"footfall index low ({footfall_index:.2f})")
        elif footfall_index > 1.25:
            mod += 0.3
            bits.append(f"footfall surge ({footfall_index:.2f})")
        cat = category.lower()
        if temperature_c >= 30 and cat in ("drinks", "dairy", "produce"):
            mod += 0.25
            bits.append("heatwave lifting cold-category demand")
        if temperature_c < 5 and cat in ("bakery", "prepared"):
            mod += 0.15
            bits.append("cold snap favoring warm foods")
        if event:
            el = event.lower()
            if "match" in el or "game" in el:
                mod += 0.35 if cat in ("snacks", "drinks", "prepared") else 0.1
                bits.append(f"event boost ({event})")
            elif "festival" in el:
                mod -= 0.3
                bits.append(f"footfall diverted by {event}")
            else:
                bits.append(f"local event: {event}")
        if hours_to_expiry < 24:
            mod -= 0.2
            bits.append(f"only {hours_to_expiry:.0f}h to expiry")
        if stock >= 40:
            bits.append(f"high stock ({stock}u)")
        mod = max(-1.0, min(1.0, round(mod, 3)))
        if not bits:
            bits.append("normal trading conditions")
        return mod, f"{name}: " + ", ".join(bits) + "."
