/* ElasticShelf storefront: WebSocket stream + REST fallback. */
const $ = (id) => document.getElementById(id);
const state = { items: new Map(), running: false, ws: null, pollTimer: null, history: [{ tick: 0, profit: 0, waste_avoided: 0 }] };

const money = (n) => "$" + Number(n).toFixed(2);

function hoursLeft(iso, simIso) {
  const end = new Date(iso).getTime();
  const now = simIso ? new Date(simIso).getTime() : Date.now();
  return (end - now) / 3600000;
}

function renderShelf(simIso) {
  const shelf = $("shelf");
  shelf.innerHTML = "";
  const items = [...state.items.values()].sort((a, b) => a.sku.localeCompare(b.sku));
  if (!items.length) { shelf.innerHTML = '<div class="empty">No items in catalog.</div>'; return; }
  for (const it of items) {
    const hrs = hoursLeft(it.expiration_timestamp, simIso);
    const disc = it.base_price > 0 ? (it.base_price - it.current_price) / it.base_price : 0;
    const dead = it.quantity_on_shelf <= 0 || hrs < 0;
    const status = hrs < 0 ? "expired" : (it.quantity_on_shelf <= 0 ? "sold out" : "");
    const card = document.createElement("div");
    card.className = "esl" + (dead ? " dead" : (disc > 0.005 ? " promo" : "") + (hrs < 24 ? " urgent" : ""));
    card.id = "esl-" + it.sku;
    const stockPct = Math.max(0, Math.min(100, (it.quantity_on_shelf / 60) * 100));
    card.innerHTML =
      `<div class="sku">${it.sku} · ${it.category}</div>` +
      `<div class="name">${escapeHtml(it.name)}</div>` +
      (status ? `<div class="status">${status}</div>` : "") +
      `<div class="prices"><span class="now">${money(it.current_price)}</span>` +
      (disc > 0.005 ? `<span class="was">${money(it.base_price)}</span><span class="save">-${Math.round(disc * 100)}%</span>` : "") +
      `</div>` +
      `<div class="meta"><span>stock <b>${it.quantity_on_shelf}</b></span>` +
      `<span>expires <b class="${hrs < 24 ? "expiry-soon" : ""}">${hrs < 0 ? "expired" : Math.round(hrs) + "h"}</b></span></div>` +
      `<div class="stockbar"><i style="width:${stockPct}%"></i></div>` +
      (it.last_reasoning ? `<div class="why">${escapeHtml(it.last_reasoning.slice(0, 110))}</div>` : "");
    shelf.appendChild(card);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

function signedMoney(n) {
  return (n < 0 ? "-" : "+") + money(Math.abs(n));
}

function renderMetrics(m) {
  if (!m) return;
  $("mRev").textContent = money(m.revenue);
  $("mSold").textContent = m.units_sold;
  $("mWaste").textContent = m.units_wasted;
  $("mSkus").textContent = (m.active_skus != null ? m.active_skus : m.skus) + " / " + m.skus;
  if (m.revenue_baseline != null) {
    const d = m.revenue - m.revenue_baseline;
    const pct = m.revenue_baseline > 0 ? " (" + (d < 0 ? "-" : "+") + Math.abs(d / m.revenue_baseline * 100).toFixed(1) + "%)" : "";
    $("mRevBase").textContent = "no-tool " + money(m.revenue_baseline) + " · Δ " + signedMoney(d) + pct;
  }
  if (m.units_wasted_baseline != null) {
    const avoided = m.units_wasted_baseline - m.units_wasted;
    let saved = "";
    if (m.waste_cost != null && m.waste_cost_baseline != null) {
      saved = " · saved " + signedMoney(m.waste_cost_baseline - m.waste_cost);
    }
    $("mWasteBase").textContent = "no-tool " + m.units_wasted_baseline + " · avoided " + avoided + saved;
  }
  if (m.profit != null) {
    const pf = $("mProfit");
    pf.textContent = (m.profit < 0 ? "-" : "") + money(Math.abs(m.profit));
    pf.className = "v " + (m.profit >= 0 ? "good" : "bad");
    $("mProfitBase").textContent = "no-tool revenue vs revenue + waste cost saved";
  }
  renderTrendChart(m);
  if (m.sim_time) $("clock").textContent = new Date(m.sim_time).toLocaleString();
  $("tick").textContent = m.tick;
  if (m.store_open != null) {
    const b = $("openBadge");
    b.textContent = m.store_open ? "open" : "closed";
    b.classList.toggle("closed", !m.store_open);
  }
  setRunning(!!m.running);
}

function renderTrendChart(m) {
  const svg = $("trendChart");
  if (!svg) return;
  if (Array.isArray(m.history) && m.history.length) {
    state.history = m.history.slice();
  } else if (m.tick != null && m.profit != null) {
    const wa = (m.units_wasted_baseline != null && m.units_wasted != null)
      ? (m.units_wasted_baseline - m.units_wasted) : 0;
    const last = state.history[state.history.length - 1];
    if (!last || m.tick === 0) {
      state.history = [{ tick: 0, profit: m.profit || 0, waste_avoided: wa }];
    } else if (m.tick > last.tick) {
      state.history.push({ tick: m.tick, profit: m.profit, waste_avoided: wa });
    } else if (m.tick === last.tick) {
      last.profit = m.profit;
      last.waste_avoided = wa;
    }
  }
  const pts = state.history;
  const lastPt = pts[pts.length - 1] || { tick: 0, profit: 0, waste_avoided: 0 };
  if ($("chProfitVal")) $("chProfitVal").textContent = (lastPt.profit < 0 ? "-" : "") + money(Math.abs(lastPt.profit));
  if ($("chWasteVal")) $("chWasteVal").textContent = lastPt.waste_avoided + "u";

  const W = 336, H = 148;
  const padL = 44, padR = 36, padT = 12, padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const tMin = pts[0].tick;
  const tMax = Math.max(tMin + 1, lastPt.tick);
  const xOf = (t) => padL + ((t - tMin) / (tMax - tMin)) * plotW;

  const pVals = pts.map((d) => d.profit);
  let pMin = Math.min(0, ...pVals);
  let pMax = Math.max(10, ...pVals);
  if (pMax - pMin < 1e-6) pMax = pMin + 10;
  const yProfit = (v) => padT + (1 - (v - pMin) / (pMax - pMin)) * plotH;

  const wVals = pts.map((d) => d.waste_avoided);
  let wMin = Math.min(0, ...wVals);
  let wMax = Math.max(5, ...wVals);
  if (wMax - wMin < 1e-6) wMax = wMin + 5;
  const yWaste = (v) => padT + (1 - (v - wMin) / (wMax - wMin)) * plotH;

  const pPoly = pts.map((d) => `${xOf(d.tick).toFixed(1)},${yProfit(d.profit).toFixed(1)}`).join(" ");
  const wPoly = pts.map((d) => `${xOf(d.tick).toFixed(1)},${yWaste(d.waste_avoided).toFixed(1)}`).join(" ");
  const yZero = yProfit(0);
  const pArea = `${xOf(pts[0].tick).toFixed(1)},${yZero.toFixed(1)} ${pPoly} ${xOf(lastPt.tick).toFixed(1)},${yZero.toFixed(1)}`;

  const fmtMoneyShort = (v) => (v < 0 ? "-$" : "$") + Math.abs(v).toFixed(Math.abs(v) >= 100 ? 0 : 1);
  const midTick = Math.round((tMin + tMax) / 2);

  svg.innerHTML =
    `<line x1="${padL}" y1="${padT}" x2="${W - padR}" y2="${padT}" stroke="#2e382e" stroke-width="1"/>` +
    `<line x1="${padL}" y1="${padT + plotH / 2}" x2="${W - padR}" y2="${padT + plotH / 2}" stroke="#2e382e" stroke-dasharray="2,3" stroke-width="1"/>` +
    `<line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="#2e382e" stroke-width="1"/>` +
    (pMin < 0 ? `<line x1="${padL}" y1="${yZero.toFixed(1)}" x2="${W - padR}" y2="${yZero.toFixed(1)}" stroke="#3d4b3d" stroke-dasharray="3,2" stroke-width="1"/>` : "") +
    `<polygon points="${pArea}" fill="rgba(127,208,140,0.12)"/>` +
    `<polyline fill="none" stroke="#7fd08c" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${pPoly}"/>` +
    `<polyline fill="none" stroke="#e8b64c" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${wPoly}"/>` +
    `<circle cx="${xOf(lastPt.tick).toFixed(1)}" cy="${yProfit(lastPt.profit).toFixed(1)}" r="3" fill="#7fd08c"/>` +
    `<circle cx="${xOf(lastPt.tick).toFixed(1)}" cy="${yWaste(lastPt.waste_avoided).toFixed(1)}" r="3" fill="#e8b64c"/>` +
    `<text x="${padL - 4}" y="${padT + 4}" text-anchor="end" fill="#7fd08c" font-size="9.5">${fmtMoneyShort(pMax)}</text>` +
    `<text x="${padL - 4}" y="${padT + plotH / 2 + 3}" text-anchor="end" fill="#7fd08c" font-size="9" opacity="0.75">${fmtMoneyShort((pMax + pMin) / 2)}</text>` +
    `<text x="${padL - 4}" y="${padT + plotH}" text-anchor="end" fill="#7fd08c" font-size="9.5">${fmtMoneyShort(pMin)}</text>` +
    `<text x="${W - padR + 4}" y="${padT + 4}" text-anchor="start" fill="#e8b64c" font-size="9.5">${Math.round(wMax)}u</text>` +
    `<text x="${W - padR + 4}" y="${padT + plotH / 2 + 3}" text-anchor="start" fill="#e8b64c" font-size="9" opacity="0.75">${Math.round((wMax + wMin) / 2)}u</text>` +
    `<text x="${W - padR + 4}" y="${padT + plotH}" text-anchor="start" fill="#e8b64c" font-size="9.5">${Math.round(wMin)}u</text>` +
    `<text x="${padL}" y="${H - 5}" text-anchor="start" fill="#a9b6a4" font-size="9.5">t${tMin}h</text>` +
    (tMax > tMin + 1 ? `<text x="${xOf(midTick).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="#a9b6a4" font-size="9">t${midTick}h</text>` : "") +
    `<text x="${W - padR}" y="${H - 5}" text-anchor="end" fill="#a9b6a4" font-size="9.5">t${lastPt.tick}h</text>`;
}

function renderContext(c) {
  if (!c) return;
  $("cFoot").textContent = c.footfall_index != null ? Number(c.footfall_index).toFixed(2) : "—";
  $("cTemp").textContent = c.temperature_c != null ? Number(c.temperature_c).toFixed(1) + "°C" : "—";
  $("cRain").textContent = c.precipitation_mm != null ? Number(c.precipitation_mm).toFixed(1) : "—";
  $("cMod").textContent = c.demand_modifier != null ? Number(c.demand_modifier).toFixed(2) : "—";
  const ev = $("cEvent");
  if (c.event) { ev.hidden = false; ev.textContent = "Event: " + c.event; }
  else { ev.hidden = true; }
  if (c.weather_source) {
    $("ctxLegend").textContent = "Footfall 1.0 ≈ average day · Mod −1 collapse → discount, +1 surge → hold · wx: " +
      (c.weather_source === "forecast" ? "live forecast" : "offline walk");
  }
}

function renderCheckouts(checkouts) {
  if (!checkouts || !checkouts.length) return;
  const pos = $("pos");
  const empty = pos.querySelector(".empty");
  if (empty) empty.remove();
  for (const t of checkouts) {
    const d = document.createElement("div");
    d.className = "ticket";
    d.innerHTML = `<div class="thead"><span>${t.id}</span><span class="total">${money(t.total)}</span></div>` +
      "<ul>" + t.lines.map((l) => `<li>${l.qty}× ${escapeHtml(l.name)} @ ${money(l.unit_price)}</li>`).join("") + "</ul>";
    pos.prepend(d);
  }
  while (pos.children.length > 30) pos.lastChild.remove();
}

function renderAudit(entries) {
  if (!entries || !entries.length) return;
  const box = $("audit");
  const empty = box.querySelector(".empty");
  if (empty) empty.remove();
  for (const a of entries) {
    const d = document.createElement("div");
    d.className = "entry";
    const dir = a.new_price < a.old_price ? "down" : "up";
    const arrow = a.new_price < a.old_price ? "↓" : "↑";
    d.innerHTML = `<div class="p"><b>${a.sku}</b> <span class="${dir}">${arrow} ${money(a.old_price)} → ${money(a.new_price)}</span></div>` +
      (a.llm_reasoning ? `<div class="r">${escapeHtml(a.llm_reasoning.slice(0, 160))}</div>` : "");
    box.prepend(d);
    const card = $("esl-" + a.sku);
    if (card) { card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash"); }
  }
  while (box.children.length > 30) box.lastChild.remove();
}

const STATUS_RANK = { expired: 0, sold_out: 1, urgent: 2, low: 3, ok: 4 };

const SUGGEST_TIP = "Restock forecast frozen at depletion — predicted by Muse, rule fallback offline. Hover a value for its reasoning.";

function showTab(which) {
  const fc = which === "forecast";
  $("shelf").hidden = fc;
  $("forecast").hidden = !fc;
  $("tabShelf").classList.toggle("active", !fc);
  $("tabForecast").classList.toggle("active", fc);
  if (fc) loadForecast();
}

async function loadForecast() {
  try {
    const fc = await fetch("/api/v1/forecast").then((r) => r.json());
    const rows = (fc.items || []).filter((r) => r.status === "expired" || r.status === "sold_out")
      .sort((a, b) => (STATUS_RANK[a.status] - STATUS_RANK[b.status]) || (b.velocity - a.velocity));
    $("tabForecast").textContent = rows.length ? "Forecast & restock (" + rows.length + ")" : "Forecast & restock";
    $("fcNote").textContent = "Tick " + fc.tick + " · " + rows.length + " SKU(s) off shelf · forecast frozen at depletion · restocking returns the SKU to the shelf.";
    const body = $("fcBody");
    body.innerHTML = "";
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="7" class="empty">Nothing to reorder — every SKU is on the shelf.</td></tr>';
      state.depleted = new Set();
      return;
    }
    const seen = state.depleted || new Set();
    const now = new Set();
    for (const r of rows) {
      now.add(r.sku);
      const tr = document.createElement("tr");
      if (!seen.has(r.sku)) tr.className = "fresh";
      const was = r.depleted_from != null ? r.depleted_from : "—";
      const note = r.depletion_note ? `<div class="depnote">${escapeHtml(r.depletion_note)}</div>` : "";
      const tip = r.forecast_reason
        ? ((r.forecast_source === "llm" ? "Muse forecast: " : "Rule fallback: ") + r.forecast_reason)
        : SUGGEST_TIP;
      tr.innerHTML =
        `<td class="sku">${r.sku}</td>` +
        `<td>${was}${note}</td>` +
        `<td>${Number(r.velocity).toFixed(2)}</td><td>${r.wasted}</td><td>${r.wasted_baseline}</td>` +
        `<td title="${escapeHtml(tip)}">${r.suggested_order}</td>` +
        `<td><input type="number" min="1" value="${Math.max(1, r.suggested_order)}" id="ord-${r.sku}">` +
        `<button data-sku="${r.sku}">Restock</button></td>`;
      body.appendChild(tr);
    }
    body.querySelectorAll("button").forEach((b) => { b.onclick = () => restockSku(b.dataset.sku, "ord-"); });
    state.depleted = now;
  } catch (e) { /* offline; retry on next tick */ }
}

async function restockSku(sku, prefix) {
  const qty = Math.max(1, Number($(prefix + sku).value) || 1);
  const r = await fetch("/api/v1/inventory/restock", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sku, quantity: qty }) });
  if (r.ok) { loadForecast(); refreshAll(); }
}

function setRunning(r) {
  state.running = r;
  $("btnToggle").textContent = r ? "Pause" : "Start";
  const b = $("liveBadge");
  b.textContent = r ? "● live" : "paused";
  b.classList.toggle("live", r);
}

function setConn(ok) {
  const c = $("conn");
  c.textContent = ok ? "live" : "retrying";
  c.className = "v";
  c.id = "conn";
  c.style.color = ok ? "var(--accent)" : "var(--danger)";
}

async function refreshAll() {
  try {
    const [inv, status, ctx] = await Promise.all([
      fetch("/api/v1/inventory").then((r) => r.json()),
      fetch("/api/v1/sim/status").then((r) => r.json()),
      fetch("/api/v1/context").then((r) => r.json()),
    ]);
    for (const it of inv.items || []) state.items.set(it.sku, it);
    renderShelf(status.sim_time);
    renderMetrics({ ...status });
    if (document.activeElement !== $("openHour")) $("openHour").value = status.open_hour;
    if (document.activeElement !== $("closeHour")) $("closeHour").value = status.close_hour;
    renderContext({ footfall_index: ctx.footfall_index, temperature_c: ctx.temperature_c, precipitation_mm: ctx.precipitation_mm, event: ctx.event, weather_source: ctx.weather_source, demand_modifier: ctx.last_snapshot ? ctx.last_snapshot.demand_modifier : null });
    $("engineBadge").textContent = "engine: " + (ctx.engine || "?");
    $("footEngine").textContent = "engine: " + (ctx.engine || "?") + " · llm: " + (ctx.llm_provider || "none") + "/" + (ctx.llm_model || "?");
  } catch (e) { /* offline; ws will retry */ }
}

function onMessage(msg) {
  if (msg.type === "hello") {
    for (const it of msg.items || []) state.items.set(it.sku, it);
    renderShelf(msg.metrics && msg.metrics.sim_time);
    renderMetrics(msg.metrics);
    refreshAll();
  } else if (msg.type === "tick") {
    renderCheckouts(msg.checkouts);
    renderAudit((msg.pricing && msg.pricing.audit_log) || []);
    renderContext(msg.context);
    renderMetrics(msg.metrics);
    refreshInventoryQuiet(msg.metrics && msg.metrics.sim_time);
    if (!$("forecast").hidden) loadForecast();
  } else if (msg.type === "pricing") {
    renderAudit(msg.audit_log || []);
    renderMetrics(msg.metrics);
    refreshInventoryQuiet();
  } else if (msg.type === "reset") {
    $("pos").innerHTML = '<div class="empty">No sales yet — start the simulation.</div>';
    $("audit").innerHTML = '<div class="empty">No price pushes yet.</div>';
    refreshAll();
  }
}

let invTimer = null;
async function refreshInventoryQuiet(simIso) {
  if (invTimer) return;
  invTimer = setTimeout(async () => {
    invTimer = null;
    try {
      const inv = await fetch("/api/v1/inventory").then((r) => r.json());
      for (const it of inv.items || []) state.items.set(it.sku, it);
      const st = await fetch("/api/v1/sim/status").then((r) => r.json());
      renderShelf(simIso || st.sim_time);
    } catch (e) { /* ignore */ }
  }, 400);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(proto + "://" + location.host + "/ws/stream");
  state.ws = ws;
  ws.onopen = () => { setConn(true); };
  ws.onmessage = (ev) => { try { onMessage(JSON.parse(ev.data)); } catch (e) {} };
  ws.onclose = () => { setConn(false); setTimeout(connect, 2000); };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}

$("btnToggle").onclick = async () => {
  const url = state.running ? "/api/v1/sim/stop" : "/api/v1/sim/start";
  await fetch(url, { method: "POST" });
  setRunning(!state.running);
};
$("btnStep").onclick = async () => {
  const r = await fetch("/api/v1/sim/tick", { method: "POST" }).then((r) => r.json());
  onMessage({ type: "tick", ...r, metrics: r.metrics || null });
  refreshAll();
};
$("btnPrice").onclick = async () => {
  const r = await fetch("/api/v1/pricing/evaluate", { method: "POST" }).then((r) => r.json());
  renderAudit(r.audit_log || []);
  refreshAll();
};
$("btnReset").onclick = async () => { await fetch("/api/v1/sim/reset", { method: "POST" }); refreshAll(); };
$("speed").onchange = async (e) => {
  await fetch("/api/v1/sim/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tick_seconds: Number(e.target.value) }) });
};
async function pushHours() {
  const o = Math.max(0, Math.min(23, Number($("openHour").value)));
  const c = Math.max(0, Math.min(23, Number($("closeHour").value)));
  const r = await fetch("/api/v1/sim/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ open_hour: o, close_hour: c }) }).then((r) => r.json());
  $("openHour").value = r.open_hour;
  $("closeHour").value = r.close_hour;
}
$("openHour").onchange = pushHours;
$("closeHour").onchange = pushHours;
$("tabShelf").onclick = () => showTab("shelf");
$("tabForecast").onclick = () => showTab("forecast");

connect();
refreshAll();
setInterval(() => { if (!state.ws || state.ws.readyState !== 1) refreshAll(); }, 5000);
setInterval(() => { if (state.ws && state.ws.readyState === 1) state.ws.send("ping"); }, 25000);
