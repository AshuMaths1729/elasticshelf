"""Pydantic schemas for the ElasticShelf MVP (mirrors technical_implementation_guide.md)."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class InventoryItemIn(BaseModel):
    sku: str
    name: str
    quantity_on_shelf: int = Field(ge=0)
    base_price: float = Field(gt=0)
    cost_basis: float = Field(ge=0)
    expiration_timestamp: datetime
    category: Optional[str] = None
    image_url: Optional[str] = None


class InventorySyncIn(BaseModel):
    store_id: str
    items: List[InventoryItemIn]


class InventoryItem(BaseModel):
    sku: str
    name: str
    category: str = "general"
    quantity_on_shelf: int
    base_price: float
    current_price: float
    cost_basis: float
    expiration_timestamp: datetime
    image_url: Optional[str] = None
    last_reasoning: Optional[str] = None


class PriceAuditEntry(BaseModel):
    sku: str
    old_price: float
    new_price: float
    llm_reasoning: str = ""


class PricingEvaluateOut(BaseModel):
    status: str = "success"
    updates_pushed: int = 0
    hardware_sync_status: str = "acknowledged"
    audit_log: List[PriceAuditEntry] = []


class ContextSnapshot(BaseModel):
    sim_time: datetime
    footfall_index: float  # 0..~2, 1.0 = normal
    temperature_c: float
    precipitation_mm: float
    event: Optional[str] = None
    demand_modifier: float = 0.0  # -1..+1 from LLaMA/mock


class SimStatus(BaseModel):
    running: bool
    tick: int
    sim_time: datetime
    tick_seconds: float
    store_id: str
    skus: int
    revenue: float
    units_sold: int
    units_wasted: int


class SimControl(BaseModel):
    tick_seconds: Optional[float] = None
    hours_per_tick: Optional[float] = None
