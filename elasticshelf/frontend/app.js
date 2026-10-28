/* ElasticShelf storefront: WebSocket stream + REST fallback. */
const $ = (id) => document.getElementById(id);
const state = { items: new Map(), running: false, ws: null, pollTimer: null };

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
    const card = document.createElement("div");
    card.className = "esl" + (disc > 0.005 ? " promo" : "") + (hrs < 24 ? " urgent" : "");
    card.id = "esl-" + it.sku;
    const stockPct = Math.max(0, Math.min(100, (it.quantity_on_shelf / 60) * 100));
    card.innerHTML =
      `<div class="sku">${it.sku} · ${it.category}</div>` +
      `<div class="name">${escapeHtml(it.name)}</div>` +
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

function renderMetrics(m) {
  if (!m) return;
  $("mRev").textContent = money(m.revenue);
  $("mSold").textContent = m.units_sold;
  $("mWaste").textContent = m.units_wasted;
  $("mSkus").textContent = m.skus;
  if (m.sim_time) $("clock").textContent = new Date(m.sim_time).toLocaleString();
  $("tick").textContent = m.tick;
  setRunning(!!m.running);
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
    renderContext({ footfall_index: ctx.footfall_index, temperature_c: ctx.temperature_c, precipitation_mm: ctx.precipitation_mm, event: ctx.event, demand_modifier: ctx.last_snapshot ? ctx.last_snapshot.demand_modifier : null });
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

connect();
refreshAll();
setInterval(() => { if (!state.ws || state.ws.readyState !== 1) refreshAll(); }, 5000);
setInterval(() => { if (state.ws && state.ws.readyState === 1) state.ws.send("ping"); }, 25000);
