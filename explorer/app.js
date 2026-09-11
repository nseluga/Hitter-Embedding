// Matchup Explorer. All math runs here from the raw W(0,0) matrices in data/matchups.json.
// Matchup effect = w - hitter's BF-weighted avg (row) - pitcher's avg over hitters (col) + grand mean,
// the same double-centering as src/analysis/matchup_explorer_data.double_center.
const $ = s => document.querySelector(s);
const G = {}, pitLabel = {}, hitLabel = {}, labelToP = {}, labelToH = {};
let D, H, P, PCT = {};
const AX = ["fb_velo", "fb_height", "brk_share"];
const PRESETS = {
  "Hard throwers (top 20% velo)": [80, 100, 0, 100, 0, 100],
  "Soft tossers (bottom 20% velo)": [0, 20, 0, 100, 0, 100],
  "Elevators (top 20% FB height)": [0, 100, 80, 100, 0, 100],
  "Low-zone (bottom 20% FB height)": [0, 100, 0, 20, 0, 100],
  "Breaking-ball heavy (top 20%)": [0, 100, 0, 100, 80, 100],
  "Fastball heavy (bottom 20% brk)": [0, 100, 0, 100, 0, 20],
  "Power elevators (velo & height top 40%)": [60, 100, 60, 100, 0, 100],
  "Soft breakers (velo bottom 40%, brk top 40%)": [0, 40, 0, 100, 60, 100],
};
const S = {
  tab: "matchup",
  m: { p: null, hs: [], sort: "pred" },
  l: { mode: "hitters", metric: "eff", minpa: 300, stand: "", n: 15, hand: "R", preset: Object.keys(PRESETS)[0], rng: PRESETS["Hard throwers (top 20% velo)"], p1: null, h1: null, hand1: "R", minbf: 500 },
  h: { hand: "R", rows: "top", n: 30, cols: "types", cell: "eff", hs: [], ps: [], sort: null },
};

// ---------- data ----------
function decode(name, o) {
  const bin = atob(o.w), [n, m] = o.shape, w = new Float32Array(n * m), { lo, hi } = D.meta.quant;
  for (let k = 0; k < n * m; k++) w[k] = ((bin.charCodeAt(2 * k) | (bin.charCodeAt(2 * k + 1) << 8)) / 65535) * (hi - lo) + lo;
  const tot = o.bf.reduce((a, b) => a + b, 0), p = o.bf.map(b => b / tot);
  const r = new Float64Array(n), c = new Float64Array(m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const v = w[i * m + j]; r[i] += v * p[j]; c[j] += v / n; }
  let gm = 0; for (let j = 0; j < m; j++) gm += c[j] * p[j];
  G[name] = { w, n, m, r, c, gm, stand: name[0], hand: name[7],
    bIdx: new Map(o.batters.map((b, i) => [b, i])), pIdx: new Map(o.pitchers.map((q, j) => [q, j])) };
}
function groupFor(bid, hand) {
  const L = G[`LHB_vs_${hand}HP`], R = G[`RHB_vs_${hand}HP`], inL = L.bIdx.has(bid), inR = R.bIdx.has(bid);
  if (inL && inR) return hand === "R" ? L : R; // switch hitters bat opposite the pitcher
  return inL ? L : inR ? R : null;
}
function cell(bid, pkey) {
  const g = groupFor(bid, pkey[0]); if (!g) return null;
  const j = g.pIdx.get(+pkey.slice(1)); if (j === undefined) return null;
  const i = g.bIdx.get(bid), v = g.w[i * g.m + j];
  return { pred: v, eff: v - g.r[i] - g.c[j] + g.gm, avg: g.r[i], side: g.stand };
}
function hitterAvg(bid, hand) { const g = groupFor(bid, hand); return g ? { avg: g.r[g.bIdx.get(bid)], side: g.stand } : null; }
function pitchersOf(hand) { return Object.keys(P).filter(k => k[0] === hand); }
function buildPct(hand) { // BF-weighted within-hand percentile, as in the prototype
  const ks = pitchersOf(hand).filter(k => AX.every(a => P[k][a] != null) && P[k].bf > 0);
  const tot = ks.reduce((s, k) => s + P[k].bf, 0), out = {};
  for (const a of AX) {
    let cum = 0; out[a] = {};
    [...ks].sort((x, y) => P[x][a] - P[y][a]).forEach(k => { cum += P[k].bf; out[a][k] = cum / tot; });
  }
  out.keys = ks; out.tot = tot; return out;
}
function typeKeys(hand, rng) {
  const T = PCT[hand];
  return T.keys.filter(k => AX.every((a, i) => T[a][k] >= rng[2 * i] / 100 - 1e-12 && T[a][k] <= rng[2 * i + 1] / 100 + 1e-12));
}
function typeScore(bid, hand, keys) {
  const g = groupFor(bid, hand); if (!g) return null;
  const i = g.bIdx.get(bid); let se = 0, sp = 0, sw = 0;
  for (const k of keys) {
    const j = g.pIdx.get(+k.slice(1)); if (j === undefined) continue;
    const v = g.w[i * g.m + j], wt = P[k].bf;
    se += (v - g.r[i] - g.c[j] + g.gm) * wt; sp += v * wt; sw += wt;
  }
  return sw ? { eff: se / sw, pred: sp / sw, avg: g.r[i], side: g.stand } : null;
}
const pa = (bid, hand) => (H[bid] ? H[bid][hand === "R" ? "pa_R" : "pa_L"] : 0) || 0;
const hitters = () => Object.keys(H).map(Number);

// ---------- formatting ----------
const woba = v => v == null ? "–" : v.toFixed(3).replace(/^0/, "");
const signed = v => { const r = Math.round(v * 1000) || 0; return (r > 0 ? "+" : "") + r; };
const pts = v => v == null ? "–" : `<span class="${v > 0.0005 ? "pos" : v < -0.0005 ? "neg" : ""}">${signed(v)}</span>`;
const esc = s => String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
function color(t) { // t in [-1,1]: red -> neutral -> green
  const x = Math.max(-1, Math.min(1, t)), a = [247, 246, 243], b = x < 0 ? [214, 120, 108] : [104, 181, 134], f = Math.abs(x);
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * f)).join(",")})`;
}
function table(cols, rows) {
  return `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function chips(el, items, onRemove) {
  el.innerHTML = items.map((t, i) => `<span class="chip">${esc(t)}<button data-i="${i}" title="Remove">×</button></span>`).join("");
  el.querySelectorAll("button").forEach(b => b.onclick = () => onRemove(+b.dataset.i));
}

// ---------- tabs ----------
function renderMatchup() {
  const m = S.m, out = $("#m-out");
  chips($("#m-chips"), m.hs.map(b => hitLabel[b]), i => { m.hs.splice(i, 1); render(); });
  $("#m-pitcher").value = m.p ? pitLabel[m.p] : ""; $("#m-sort").value = m.sort;
  if (!m.p) { out.innerHTML = `<span class="muted">Pick a pitcher, then add hitters.</span>`; return; }
  const q = P[m.p], hand = m.p[0], pid = +m.p.slice(1);
  const vs = ["L", "R"].map(s => { const g = G[`${s}HB_vs_${hand}HP`], j = g.pIdx.get(pid); return j === undefined ? null : `vs ${s}HB ${woba(g.c[j])}`; }).filter(Boolean);
  const pct = PCT[hand], attr = q.fb_velo == null ? "no pitch attributes" :
    `FB ${q.fb_velo.toFixed(1)} mph (${Math.round(pct.fb_velo[m.p] * 100)}th pct), FB height ${q.fb_height.toFixed(2)} ft (${Math.round(pct.fb_height[m.p] * 100)}th), breaking share ${Math.round(q.brk_share * 100)}% (${Math.round(pct.brk_share[m.p] * 100)}th)` +
    `, ${q.bf.toLocaleString()} training BF${q.bf < 500 ? " (small sample: the model knows him less well)" : ""}`;
  let rows = m.hs.map(b => ({ b, c: cell(b, m.p) }));
  if (m.sort !== "none") rows.sort((x, y) => (y.c ? y.c[m.sort] : -9) - (x.c ? x.c[m.sort] : -9));
  const max = Math.max(0.45, ...rows.map(r => r.c ? r.c.pred : 0));
  out.innerHTML = `<div><b>${esc(q.name)}</b> <span class="muted">${hand}HP · ${attr}</span></div>
    <div class="hint">Average projected wOBA allowed: ${vs.join(" · ")}</div><div class="scroll" style="margin-top:10px">` +
    (rows.length ? table(["Hitter", "Bats", "Projected wOBA", "", `His avg vs ${hand}HP`, "Expected vs this pitcher", "Matchup effect (pts)"],
      rows.map(({ b, c }) => c ? [esc(H[b].name), c.side, `<b>${woba(c.pred)}</b>`, `<span class="bar" style="width:${Math.round(c.pred / max * 160)}px;background:var(--acc)"></span>`, woba(c.avg), woba(c.pred - c.eff), pts(c.eff)]
        : [esc(H[b].name), "–", "not in data for this pitcher", "", "", "", ""]))
      + `<div class="hint">Who should start: sort by projected wOBA. Expected = his average adjusted for how tough this pitcher is on all hitters from his side. Matchup effect = projected − expected: positive means the model thinks this pitcher suits him better than his overall numbers suggest.</div>`
      : `<span class="muted">Add hitters above.</span>`) + `</div>`;
}

function typeCtl(el, st, onChange) {
  const names = ["Velo", "FB height", "Brk share"];
  el.innerHTML = `<label>Pitcher hand<select data-k="hand"><option>R</option><option>L</option></select></label>
    <label>Pitcher type<select data-k="preset">${Object.keys(PRESETS).map(p => `<option>${p}</option>`).join("")}<option>Custom</option></select></label>` +
    names.map((n, a) => `<label>${n} pct<span><input type="number" data-r="${2 * a}" min="0" max="100" step="5"> – <input type="number" data-r="${2 * a + 1}" min="0" max="100" step="5"></span></label>`).join("");
  el.querySelector("[data-k=hand]").value = st.hand; el.querySelector("[data-k=preset]").value = st.preset;
  el.querySelectorAll("[data-r]").forEach(inp => { inp.value = st.rng[+inp.dataset.r]; inp.onchange = () => { st.rng = [...st.rng]; st.rng[+inp.dataset.r] = +inp.value; st.preset = "Custom"; onChange(); }; });
  el.querySelector("[data-k=hand]").onchange = e => { st.hand = e.target.value; onChange(); };
  el.querySelector("[data-k=preset]").onchange = e => { st.preset = e.target.value; if (PRESETS[st.preset]) st.rng = [...PRESETS[st.preset]]; onChange(); };
}

function renderLeader() {
  const l = S.l;
  for (const k of ["mode", "metric", "minpa", "stand", "n", "hand1", "minbf"]) $(`#l-${k}`).value = l[k];
  $("#l-p1").value = l.p1 ? pitLabel[l.p1] : ""; $("#l-h1").value = l.h1 ? hitLabel[l.h1] : "";
  const isType = l.mode === "hitters";
  $("#l-type-ctl").hidden = !isType;
  $("#l-p1w").hidden = l.mode !== "hitters1"; $("#l-h1w").hidden = $("#l-hand1w").hidden = $("#l-minbfw").hidden = l.mode !== "pitchers";
  $("#l-minpa").parentElement.hidden = $("#l-stand").parentElement.hidden = l.mode === "pitchers";
  if (isType) typeCtl($("#l-type-ctl"), l, render);
  let rows = [], cols, info = "", hand;
  if (l.mode === "pitchers") {
    hand = l.hand1;
    if (l.h1) rows = pitchersOf(hand).filter(k => P[k].bf >= l.minbf).map(k => ({ k, c: cell(l.h1, k) })).filter(r => r.c);
    cols = ["Pitcher", "FB mph", "FB ht", "Brk %", "Projected wOBA", "Effect (pts)"];
    info = l.h1 ? `${esc(H[l.h1].name)} against ${rows.length} ${hand}HP with ≥${l.minbf} training BF. His average vs ${hand}HP: ${woba(hitterAvg(l.h1, hand)?.avg)}.` : "Pick a hitter.";
    rows = rows.map(({ k, c }) => ({ v: c[l.metric], cells: [esc(P[k].name), P[k].fb_velo?.toFixed(1) ?? "–", P[k].fb_height?.toFixed(2) ?? "–", P[k].brk_share == null ? "–" : Math.round(P[k].brk_share * 100), woba(c.pred), pts(c.eff)] }));
  } else {
    let f;
    if (isType) {
      hand = l.hand; const keys = typeKeys(hand, l.rng), T = PCT[hand];
      const bfShare = keys.reduce((s, k) => s + P[k].bf, 0) / T.tot;
      const ex = [...keys].sort((a, b) => P[b].bf - P[a].bf).slice(0, 4).map(k => P[k].name);
      info = `${keys.length} ${hand}HP, ${Math.round(bfShare * 100)}% of batters faced. Most-used: ${esc(ex.join(", "))}.`;
      f = b => keys.length ? typeScore(b, hand, keys) : null;
    } else {
      hand = l.p1 ? l.p1[0] : "R";
      info = l.p1 ? `${esc(P[l.p1].name)} (${hand}HP).` : "Pick a pitcher.";
      f = b => l.p1 ? cell(b, l.p1) : null;
    }
    rows = hitters().filter(b => pa(b, hand) >= l.minpa).map(b => ({ b, c: f(b) })).filter(r => r.c && (!l.stand || r.c.side === l.stand))
      .map(({ b, c }) => ({ v: c[l.metric], cells: [esc(H[b].name), c.side, pts(c.eff), woba(c.pred), woba(c.avg), pa(b, hand)] }));
    cols = ["Hitter", "Bats", "Effect (pts)", "Projected wOBA", `Avg vs ${hand}HP`, "Prior PA"];
    info += ` ${rows.length} hitters pass the filters.`;
  }
  $("#l-typeinfo").innerHTML = info;
  rows.sort((a, b) => b.v - a.v);
  const n = Math.min(l.n, Math.ceil(rows.length / 2));
  $("#l-best").innerHTML = `<b>Best</b>` + (n ? table(["#", ...cols], rows.slice(0, n).map((r, i) => [i + 1, ...r.cells])) : "");
  $("#l-worst").innerHTML = `<b>Worst</b>` + (n ? table(["#", ...cols], rows.slice(-n).reverse().map((r, i) => [rows.length - i, ...r.cells])) : "");
}

function renderHeat() {
  const h = S.h;
  for (const k of ["hand", "rows", "n", "cols", "cell"]) $(`#h-${k}`).value = h[k];
  chips($("#h-chips"), [...h.hs.map(b => "H: " + hitLabel[b]), ...h.ps.map(k => "P: " + pitLabel[k])], i => {
    if (i < h.hs.length) h.hs.splice(i, 1); else h.ps.splice(i - h.hs.length, 1); render();
  });
  const hand = h.hand;
  let rowIds;
  if (h.rows === "top") rowIds = hitters().filter(b => groupFor(b, hand)).sort((a, b) => pa(b, hand) - pa(a, hand)).slice(0, h.n);
  else if (h.rows === "list") rowIds = h.hs.filter(b => groupFor(b, hand));
  else {
    const [attr, dir] = h.rows.split(":");
    rowIds = hitters().filter(b => groupFor(b, hand) && pa(b, hand) >= 300 && H[b][attr] != null)
      .sort((a, b) => (dir === "hi" ? -1 : 1) * (H[a][attr] - H[b][attr])).slice(0, h.n);
  }
  const cols = h.cols === "types"
    ? Object.entries(PRESETS).map(([name, rng]) => { const keys = typeKeys(hand, rng); return { name: name.replace(/ \(.*/, ""), full: name, f: b => typeScore(b, hand, keys) }; })
    : h.ps.filter(k => k[0] === hand).map(k => ({ name: P[k].name, full: P[k].name, f: b => cell(b, k) }));
  if (!rowIds.length || !cols.length) {
    $("#h-out").innerHTML = `<span class="muted">${!cols.length ? `Add ${hand}HP pitchers to your list, or switch columns to pitcher types.` : "No hitters for these rows. Add hitters to your list."}</span>`; return;
  }
  const M = rowIds.map(b => cols.map(c => c.f(b)));
  let order = rowIds.map((_, i) => i);
  if (h.sort != null && h.sort < cols.length) order.sort((a, b) => (M[b][h.sort]?.[h.cell] ?? -9) - (M[a][h.sort]?.[h.cell] ?? -9));
  const vals = M.flat().filter(Boolean).map(c => c.pred), mean = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
  const tone = c => h.cell === "eff" ? c.eff / 0.02 : (c.pred - mean) / 0.06;
  $("#h-out").innerHTML = `<table class="hm"><thead><tr><th></th>${cols.map((c, j) => `<th class="col" data-j="${j}" title="${esc(c.full)} (click to sort)">${esc(c.name)}${h.sort === j ? " ▼" : ""}</th>`).join("")}</tr></thead><tbody>` +
    order.map(i => `<tr><td class="name">${esc(H[rowIds[i]].name)} <span class="muted">${groupFor(rowIds[i], hand).stand}</span></td>` +
      M[i].map((c, j) => c ? `<td style="background:${color(tone(c))}" title="${esc(H[rowIds[i]].name)} vs ${esc(cols[j].full)}: effect ${Math.round(c.eff * 1000)} pts, projected ${woba(c.pred)}, his avg ${woba(c.avg)}">${h.cell === "eff" ? signed(c.eff) : woba(c.pred)}</td>` : `<td>–</td>`).join("") + `</tr>`).join("") +
    `</tbody></table><div class="hint">${h.cell === "eff" ? "Green = better than his usual vs this hand, red = worse (wOBA points; color saturates at ±20)." : "Color relative to the table mean."}</div>`;
  document.querySelectorAll(".hm th.col").forEach(th => th.onclick = () => { h.sort = h.sort === +th.dataset.j ? null : +th.dataset.j; render(); });
}

function render() {
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.t === S.tab));
  document.querySelectorAll("main section").forEach(s => s.hidden = s.id !== "t-" + S.tab);
  ({ matchup: renderMatchup, leader: renderLeader, heat: renderHeat, about: () => {} })[S.tab]();
  history.replaceState(null, "", "#" + encodeURIComponent(JSON.stringify(S)));
}

// ---------- wiring ----------
function wire() {
  document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => { S.tab = b.dataset.t; render(); });
  const pick = (el, map, fn) => el.onchange = () => { const v = map[el.value]; if (v !== undefined) fn(v); };
  pick($("#m-pitcher"), labelToP, k => { S.m.p = k; render(); });
  pick($("#m-hitter"), labelToH, b => { if (!S.m.hs.includes(b)) S.m.hs.push(b); $("#m-hitter").value = ""; render(); });
  $("#m-sort").onchange = e => { S.m.sort = e.target.value; render(); };
  for (const k of ["mode", "metric", "stand", "hand1"]) $(`#l-${k}`).onchange = e => { S.l[k] = e.target.value; render(); };
  for (const k of ["minpa", "n", "minbf"]) $(`#l-${k}`).onchange = e => { S.l[k] = +e.target.value; render(); };
  pick($("#l-p1"), labelToP, k => { S.l.p1 = k; render(); });
  pick($("#l-h1"), labelToH, b => { S.l.h1 = b; render(); });
  for (const k of ["hand", "rows", "cols", "cell"]) $(`#h-${k}`).onchange = e => { S.h[k] = e.target.value; S.h.sort = null; render(); };
  $("#h-n").onchange = e => { S.h.n = +e.target.value; render(); };
  pick($("#h-addh"), labelToH, b => { if (!S.h.hs.includes(b)) S.h.hs.push(b); $("#h-addh").value = ""; S.h.rows = "list"; render(); });
  pick($("#h-addp"), labelToP, k => { if (!S.h.ps.includes(k)) S.h.ps.push(k); $("#h-addp").value = ""; S.h.cols = "list"; S.h.hand = k[0]; render(); });
}
function labels() {
  const dup = (arr) => { const c = {}; arr.forEach(x => c[x] = (c[x] || 0) + 1); return c; };
  const pc = dup(Object.values(P).map(q => q.name));
  for (const [k, q] of Object.entries(P)) { const t = `${q.name} (${k[0]}HP)${pc[q.name] > 1 ? ` #${k.slice(1)}` : ""}`; pitLabel[k] = t; labelToP[t] = k; }
  const hc = dup(Object.values(H).map(x => x.name));
  for (const b of hitters()) {
    const sides = new Set(Object.values(G).filter(g => g.bIdx.has(b)).map(g => g.stand));
    const t = `${H[b].name} (${sides.size > 1 ? "S" : [...sides][0]})${hc[H[b].name] > 1 ? ` #${b}` : ""}`; hitLabel[b] = t; labelToH[t] = b;
  }
  $("#dl-pitchers").innerHTML = Object.values(pitLabel).sort().map(t => `<option value="${esc(t)}">`).join("");
  $("#dl-hitters").innerHTML = Object.values(hitLabel).sort().map(t => `<option value="${esc(t)}">`).join("");
}
function selftest() { // reference values from the Python prototype / npz
  const close = (a, b, tol) => Math.abs(a - b) <= tol;
  const g = G.RHB_vs_RHP, trout = typeScore(545361, "R", typeKeys("R", PRESETS["Hard throwers (top 20% velo)"]));
  const checks = [
    ["cell RHB_vs_RHP[Trout, Colón] = 0.45344", close(g.w[g.bIdx.get(545361) * g.m + g.pIdx.get(112526)], 0.45344, 1e-4)],
    ["Trout vs hard throwers = -12.65 pts", close(trout.eff * 1000, -12.6475, 0.05)],
  ];
  const msg = checks.map(([n, ok]) => `${ok ? "PASS" : "FAIL"} ${n}`).join(" · ");
  console.log("selftest:", msg); $("#banner").textContent = "selftest: " + msg;
}

async function init() {
  try {
    D = await (await fetch("data/matchups.json")).json();
  } catch (e) { $("#loading").textContent = "Could not load data/matchups.json. Serve this folder over http (python -m http.server)."; return; }
  H = D.hitters; P = D.pitchers;
  H = Object.fromEntries(Object.entries(H).map(([k, v]) => [+k, v]));
  for (const [name, o] of Object.entries(D.groups)) decode(name, o);
  PCT = { R: buildPct("R"), L: buildPct("L") };
  labels(); wire();
  try { Object.assign(S, JSON.parse(decodeURIComponent(location.hash.slice(1)))); } catch (e) { /* fresh state */ }
  if (!S.m.p) { // friendly default: a well-known matchup
    const k = Object.keys(P).find(k => P[k].name === "Gerrit Cole"); S.m.p = k || null;
    S.m.hs = [545361, 592450, 665742, 660271].filter(b => H[b]);
  }
  $("#meta").textContent = `${D.meta.build}. Data generated ${D.meta.generated}. ${D.meta.note}`;
  $("#loading").hidden = true; render();
  if (location.search.includes("selftest")) selftest();
}
init();
