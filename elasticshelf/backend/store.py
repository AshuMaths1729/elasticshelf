"""In-memory store (hackathon MVP). Swappable for Postgres+Redis later."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from .models import InventoryItem


class Store:
    def __init__(self, store_id: str = "ST-4092"):
        self.store_id = store_id
        self.items: Dict[str, InventoryItem] = {}
        self.revenue: float = 0.0
        self.units_sold: int = 0
        self.units_wasted: int = 0
        self.audit_log: List[dict] = []

    def upsert(self, item: InventoryItem) -> None:
        self.items[item.sku] = item

    def get(self, sku: str) -> InventoryItem:
        return self.items[sku]

    def all(self) -> List[InventoryItem]:
        return list(self.items.values())

    def record_sale(self, sku: str, qty: int) -> float:
        item = self.items[sku]
        qty = max(0, min(qty, item.quantity_on_shelf))
        item.quantity_on_shelf -= qty
        total = round(qty * item.current_price, 2)
        self.revenue = round(self.revenue + total, 2)
        self.units_sold += qty
        return total

    def record_waste(self, sku: str, qty: int) -> None:
        item = self.items[sku]
        qty = max(0, min(qty, item.quantity_on_shelf))
        item.quantity_on_shelf -= qty
        self.units_wasted += qty

    def now(self) -> datetime:
        return datetime.now(timezone.utc)
