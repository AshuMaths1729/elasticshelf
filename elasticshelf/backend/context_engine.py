"""Module B: Context Engine (muse-spark-1.3-contributor + lightweight retrieval).

MVP behavior:
- Keeps a small in-memory corpus of contextual docs (weather notes, events,
  historical pricing footnotes) and retrieves the most relevant ones per SKU
  with a dependency-free keyword scorer (TF-IDF when sklearn is installed).
- Calls the configured context model (default muse-spark-1.3-contributor via
  an OpenAI-compatible endpoint, or Groq / Together) when credentials are set.
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
        self.provider = os.getenv("LLAMA_PROVIDER", "spark").lower()
        self.groq_key = os.getenv("GROQ_API_KEY", "")
        self.together_key = os.getenv("TOGETHER_API_KEY", "")
        self.spark_key = os.getenv("SPARK_API_KEY", "")
        self.base_url = os.getenv("LLAMA_BASE_URL", "").rstrip("/")
        self.model = os.getenv("LLAMA_MODEL", "muse-spark-1.3-contributor")
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
    ) -> Tuple[float, str, bool]:
        """Returns (modifier in [-1,1], reasoning, llm_used)."""
        query = f"{name} {category} {event or ''} temp {temperature_c} rain {precipitation_mm} footfall {footfall_index}"
        docs = self.retrieve(query)
        context = " ".join(d["text"] for d in docs)
        if self._llm_available():
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
        if requests is None:
            return False
        if self.provider == "groq" and self.groq_key:
            return True
        if self.provider == "together" and self.together_key:
            return True
        if self.provider == "spark" and self.spark_key and self.base_url:
            return True
        return False

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
        if self.provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.groq_key}"}
            model = self.model
        elif self.provider == "spark":
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.spark_key}"}
            model = self.model
        else:
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.together_key}"}
            model = self.model if "llama" in self.model.lower() else "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        resp = requests.post(url, headers=headers, json={
            "model": model, "temperature": 0.2, "max_tokens": 120,
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=15)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        mod, reason = self._parse_llm(text)
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
        return mod, reason or "LLaMA demand assessment applied."

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
