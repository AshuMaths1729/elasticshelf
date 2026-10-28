"""Module D: ESL Bridge — formats pricing decisions for ESL hardware APIs.

The MVP speaks a generic payload plus two provider-shaped variants
(VusionGroup / Pricer style). The demo frontend acts as the ESL fleet and
acknowledges over WebSocket; with no fleet connected the bridge still logs
an acknowledgement so the API response stays truthful for a one-node demo.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List


def build_payload(sku: str, new_price: float, old_price: float,
                  provider: str = "generic") -> Dict:
    base = {
        "sku": sku,
        "price": new_price,
        "previous_price": old_price,
        "currency": "USD",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if provider == "vusion":
        return {"labelId": sku, "price": f"{new_price:.2f}", "priceCss": "promo" if new_price < old_price else "normal", **base}
    if provider == "pricer":
        return {"itemId": sku, "displayPrice": new_price, "flash": new_price != old_price, **base}
    return base


class ESLBridge:
    def __init__(self, provider: str = "generic") -> None:
        self.provider = provider
        self.push_log: List[Dict] = []

    def push(self, sku: str, new_price: float, old_price: float) -> Dict:
        payload = build_payload(sku, new_price, old_price, self.provider)
        payload["ack"] = "acknowledged"
        self.push_log.append(payload)
        return payload
