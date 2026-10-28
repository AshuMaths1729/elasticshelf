# ElasticShelf AI: Technical Implementation & Architecture
**Meta AI Hackathon 2026 | Engineering Documentation**

## 1. System Overview
ElasticShelf operates as a "headless" intelligence layer. It sits between a retailer's legacy data systems (POS, ERP) and their edge hardware (Electronic Shelf Labels - ESLs). The system digests real-time inventory and external contextual data, calculates optimal price elasticity using PyTorch, and pushes markdown updates via REST API.

## 2. Tech Stack (Hackathon Prototype)
*   **Core Backend:** Python 3.11, FastAPI (for low-latency, asynchronous endpoints)
*   **Contextual AI:** Meta LLaMA 3 (70B or 8B depending on local vs. cloud inference limits), hosted via Together AI, Groq, or local vLLM.
*   **Vector Database:** FAISS (Facebook AI Similarity Search) for fast retrieval of historical pricing and local event data.
*   **Elasticity Engine:** PyTorch (Reinforcement Learning / Deep Q-Network for dynamic pricing).
*   **Database:** PostgreSQL (Transactional data) + Redis (Pub/Sub and Caching).

---

## 3. Core Modules & Architecture

### Module A: Data Ingestion & Normalization
Acts as the middleware to bypass legacy POS lock-in. 
*   **Function:** Receives batch CSVs or webhook streams of inventory data (SKU, quantity, expiration date, cost).
*   **Normalization:** Converts varied grocer data into a standardized ElasticShelf JSON schema.

### Module B: The Context Engine (LLaMA 3 + RAG)
Traditional pricing models ignore the outside world. This module gives the system situational awareness.
*   **Data Sources:** Weather APIs, Local event scrapers, Social sentiment.
*   **Pipeline:** 
    1. Unstructured data is embedded and stored in **FAISS**.
    2. When inventory approaches expiration, a Retrieval-Augmented Generation (RAG) query is sent to **LLaMA 3**.
    3. LLaMA outputs a structured "Demand Modifier" (e.g., "Heavy rain tomorrow, footfall will drop 20%, accelerate discount on BBQ meats by 15%").

### Module C: PyTorch Elasticity Engine
The mathematical core of the dynamic pricing.
*   **Architecture:** A lightweight Reinforcement Learning (RL) agent built in PyTorch.
*   **State:** Current Price, Remaining Stock, Hours to Expiration, LLaMA Demand Modifier.
*   **Action:** Adjust price by $[-10\%, -5\%, 0\%, +5\%]$.
*   **Reward:** Maximizing cleared inventory while maintaining the highest possible profit margin.

### Module D: Edge Delivery (ESL Bridge)
*   **Function:** Translates the PyTorch pricing recommendation into the specific payload required by the ESL hardware provider's API (e.g., VusionGroup, Pricer).

---

## 4. Primary API Endpoints

### 1. Ingest Inventory Update
`POST /api/v1/inventory/sync`
Receives inventory updates from the grocer's POS or ERP.

**Payload:**
```json
{
  "store_id": "ST-4092",
  "items": [
    {
      "sku": "ORG-CHK-01",
      "name": "Organic Chicken Breast",
      "quantity_on_shelf": 45,
      "base_price": 12.99,
      "cost_basis": 8.50,
      "expiration_timestamp": "2026-10-30T23:59:59Z"
    }
  ]
}
```

### 2. Trigger Pricing Evaluation (Cron or Event-Driven)
`POST /api/v1/pricing/evaluate`
Triggers the LLaMA/PyTorch pipeline to evaluate active inventory and push necessary updates to the ESL hardware.

**Response:**
```json
{
  "status": "success",
  "updates_pushed": 14,
  "hardware_sync_status": "acknowledged",
  "audit_log": [
    {
      "sku": "ORG-CHK-01",
      "old_price": 12.99,
      "new_price": 10.49,
      "llm_reasoning": "High stock, 72hrs to expiration, localized storm warning reducing expected foot traffic."
    }
  ]
}
```

## 5. Hackathon Execution Plan
To successfully build this within a hackathon timeframe:
1.  **Mock the Hardware:** Do not try to integrate with real ESL hardware. Build a simple React frontend that mimics a digital shelf label to visually prove the API works.
2.  **Use Synthetic Data:** Generate 10,000 rows of fake grocery transactions with varying expiry dates to train the PyTorch model.
3.  **Prompt Engineer LLaMA:** Instead of fine-tuning, use strict system prompts and few-shot examples with LLaMA 3 to ensure it consistently outputs JSON-formatted demand modifiers.