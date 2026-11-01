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


class RestockIn(BaseModel):
    sku: str
    quantity: int = Field(gt=0)
    expiration_timestamp: Optional[datetime] = None


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
    demand_modifier: float = 0.0  # -1..+1 from Muse/mock
    weather_source: str = "walk"  # "forecast" when the Open-Meteo series drives the sim


class SimStatus(BaseModel):
    running: bool
    tick: int
    sim_time: datetime
    tick_seconds: float
    store_id: str
    skus: int
    active_skus: int = 0
    store_open: bool = True
    open_hour: int = 9
    close_hour: int = 22
    revenue: float
    units_sold: int
    units_wasted: int
    revenue_baseline: float = 0.0
    units_wasted_baseline: int = 0
    profit: float = 0.0
    profit_baseline: float = 0.0
    markdown_cost: float = 0.0
    waste_cost: float = 0.0
    waste_cost_baseline: float = 0.0
    history: List[Dict] = []


class SimControl(BaseModel):
    tick_seconds: Optional[float] = None
    hours_per_tick: Optional[float] = None
    open_hour: Optional[int] = None
    close_hour: Optional[int] = None
