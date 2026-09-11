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
  l: { mode: "hitters", metric: "eff", minpa: 300, stand: "", n: 15, hand: "R", preset: Object.keys(PRESETS)[0], rng: PRESETS["Hard throwers (top 20% velo)"], p1: null, h1: null, hand1: "R", minbf: 500, ex: "", exlo: 0, exhi: 100 },
  h: { b: 545361, hand: "R", x: "fb_up", y: "brk_down", z: "fb_velo", cell: "eff", zlo: 0, zhi: 100 },
};
// every pitcher trait: [label, unit, decimals, [low name, high name]]. Horizontal traits are hand-mirrored (+ = glove or arm side as named).
const AXN = {
  fb_velo: ["FB velo", "mph", 1, ["Soft", "Hard"]],
  fb_height: ["FB height", "ft", 2, ["Low FB", "High FB"]],
  brk_share: ["Breaking share", "%", 0, ["Fastball-heavy", "Breaking-heavy"]],
  fb_ivb: ["FB ride (vertical break)", "in", 1, ["Sinking FB", "Riding FB"]],
  fb_run: ["FB arm-side run", "in", 1, ["Little run", "Big run"]],
  fb_spin: ["FB spin", "rpm", 0, ["Low spin", "High spin"]],
  si_share: ["Sinker share of FBs", "%", 0, ["Four-seam", "Sinker"]],
  rel_height: ["Release height", "ft", 2, ["Low release", "High release"]],
  extension: ["Extension", "ft", 2, ["Short ext.", "Long ext."]],
  rel_side: ["Release side (arm side)", "ft", 2, ["Over the top", "Wide / sidearm"]],
  off_share: ["Offspeed share", "%", 0, ["Few offspeed", "Offspeed-heavy"]],
  velo_gap: ["Velo gap (FB − other pitches)", "mph", 1, ["Small gap", "Big gap"]],
  mix_depth: ["Mix depth (effective # pitches)", "pitches", 1, ["Narrow mix", "Deep mix"]],
  brk_sweep: ["Breaking-ball sweep", "in", 1, ["Little sweep", "Big sweep"]],
  zone_rate: ["Zone rate", "%", 0, ["Nibbler", "Attacks zone"]],
  glove_side: ["Location (glove side +)", "in", 1, ["Arm side", "Glove side"]],
  fb_up: ["FB up (top third of zone or above)", "%", 0, ["FB low", "FB up"]],
  brk_down: ["Breaking balls below zone", "%", 0, ["Brk in zone", "Brk buried"]],
  brk_height: ["Breaking-ball height", "ft", 2, ["Low breaking", "High breaking"]],
  off_down: ["Offspeed below zone", "%", 0, ["Offspeed in zone", "Offspeed buried"]],
};
const ALL = Object.keys(AXN), SHARE = new Set(["brk_share", "si_share", "off_share", "zone_rate", "fb_up", "brk_down", "off_down"]);
const fmtAx = (a, v) => SHARE.has(a) ? Math.round(v * 100) + "%" : v.toFixed(AXN[a][2]);
const axOpts = (blank) => (blank ? `<option value="">${blank}</option>` : "") + ALL.map(a => `<option value="${a}">${AXN[a][0]}</option>`).join("");

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
  for (const a of ALL) { // percentile among pitchers that have this trait (sweep is missing for a few)
    const ka = ks.filter(k => P[k][a] != null), ta = ka.reduce((s, k) => s + P[k].bf, 0);
    let cum = 0; out[a] = {};
    ka.sort((x, y) => P[x][a] - P[y][a]).forEach(k => { cum += P[k].bf; out[a][k] = cum / ta; });
  }
  out.keys = ks; out.tot = tot; return out;
}
const inPct = (T, a, k, lo, hi) => T[a][k] != null && T[a][k] >= lo / 100 - 1e-12 && T[a][k] <= hi / 100 + 1e-12;
function typeKeys(hand, rng, ex) { // ex = optional extra filter {a, lo, hi}
  const T = PCT[hand];
  return T.keys.filter(k => AX.every((a, i) => inPct(T, a, k, rng[2 * i], rng[2 * i + 1])) && (!ex || !ex.a || inPct(T, ex.a, k, ex.lo, ex.hi)));
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
function color(t) { // t in [-1,1]: blue (bad for hitter) -> neutral -> red (good)
  const x = Math.max(-1, Math.min(1, t)), a = [247, 246, 243], b = x < 0 ? [47, 102, 144] : [180, 55, 50], f = Math.abs(x);
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
  const more = q.fb_velo == null ? "" : ALL.filter(a => !AX.includes(a) && q[a] != null)
    .map(a => `${AXN[a][0]} ${fmtAx(a, q[a])}${SHARE.has(a) ? "" : " " + AXN[a][1]} (${Math.round(pct[a][m.p] * 100)}th)`).join(" · ");
  let rows = m.hs.map(b => ({ b, c: cell(b, m.p) }));
  if (m.sort !== "none") rows.sort((x, y) => (y.c ? y.c[m.sort] : -9) - (x.c ? x.c[m.sort] : -9));
  const max = Math.max(0.45, ...rows.map(r => r.c ? r.c.pred : 0));
  out.innerHTML = `<div><b>${esc(q.name)}</b> <span class="muted">${hand}HP · ${attr}</span></div>
    ${more ? `<div class="hint">${more}</div>` : ""}<div class="hint">Average projected wOBA allowed: ${vs.join(" · ")}</div><div class="scroll" style="margin-top:10px">` +
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
    names.map((n, a) => `<label>${n} pct<span><input type="number" data-r="${2 * a}" min="0" max="100" step="5"> – <input type="number" data-r="${2 * a + 1}" min="0" max="100" step="5"></span></label>`).join("") +
    `<label>Also filter by<select data-k="ex">${axOpts("(none)")}</select></label>` +
    (st.ex ? `<label>pct<span><input type="number" data-e="exlo" min="0" max="100" step="5"> – <input type="number" data-e="exhi" min="0" max="100" step="5"></span></label>` : "");
  el.querySelector("[data-k=hand]").value = st.hand; el.querySelector("[data-k=preset]").value = st.preset; el.querySelector("[data-k=ex]").value = st.ex;
  el.querySelector("[data-k=ex]").onchange = e => { st.ex = e.target.value; onChange(); };
  el.querySelectorAll("[data-e]").forEach(inp => { inp.value = st[inp.dataset.e]; inp.onchange = () => { st[inp.dataset.e] = +inp.value; onChange(); }; });
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
      hand = l.hand; const keys = typeKeys(hand, l.rng, { a: l.ex, lo: l.exlo, hi: l.exhi }), T = PCT[hand];
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

function quantile(sorted, q) { const i = (sorted.length - 1) * q, lo = Math.floor(i); return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (i - lo); }
function profileGrid(bid, hand, xa, ya, keys, n = 30) { // same kernel as src/analysis/matchup_figures.smooth2d
  const pts = keys.map(k => ({ k, x: P[k][xa], y: P[k][ya], w: P[k].bf, c: cell(bid, k) })).filter(p => p.c);
  if (pts.length < 10) return null;
  const axis = f => { const v = pts.map(f).sort((a, b) => a - b), m = v.reduce((s, t) => s + t, 0) / v.length;
    const sd = Math.sqrt(v.reduce((s, t) => s + (t - m) ** 2, 0) / v.length);
    return { g: Array.from({ length: n }, (_, i) => quantile(v, 0.05 + 0.9 * i / (n - 1))), h: 0.35 * sd }; };
  const X = axis(p => p.x), Y = axis(p => p.y);
  const Kx = X.g.map(gx => pts.map(p => Math.exp(-0.5 * ((gx - p.x) / X.h) ** 2)));
  const Ky = Y.g.map(gy => pts.map(p => Math.exp(-0.5 * ((gy - p.y) / Y.h) ** 2)));
  const eff = [], pred = [], den = []; let mx = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let se = 0, sp = 0, sd = 0;
    pts.forEach((p, t) => { const w = p.w * Ky[i][t] * Kx[j][t]; se += w * p.c.eff; sp += w * p.c.pred; sd += w; });
    eff.push(se / sd); pred.push(sp / sd); den.push(sd); mx = Math.max(mx, sd);
  }
  return { pts, X, Y, n, eff, pred, sup: den.map(d => d / mx) };
}
let PROF = null; // last drawn grid, for hover
function renderProfile() {
  const h = S.h, hand = h.hand, z = h.z;
  $("#h-b").value = h.b ? hitLabel[h.b] : "";
  for (const k of ["hand", "cell", "zlo", "zhi"]) $(`#h-${k}`).value = h[k];
  for (const k of ["x", "y", "z"]) { const el = $(`#h-${k}`); if (!el.options.length) el.innerHTML = axOpts(k === "z" ? "(none)" : ""); el.value = h[k]; }
  $("#h-zr").hidden = !z;
  const cv = $("#h-cv"), ctx = cv.getContext("2d"), quad = $("#h-quad");
  const msg = t => { cv.hidden = true; PROF = null; $("#h-legend").innerHTML = `<span class="muted">${t}</span>`; quad.innerHTML = ""; };
  if (!h.b) return msg("Pick a hitter.");
  if (h.x === h.y) return msg("Pick two different axes.");
  const avg = hitterAvg(h.b, hand); if (!avg) return msg(`${esc(H[h.b].name)} is not in the data against ${hand}HP.`);
  const T = PCT[hand], keys = T.keys.filter(k => P[k][h.x] != null && P[k][h.y] != null && (!z || inPct(T, z, k, h.zlo, h.zhi)));
  const R = profileGrid(h.b, hand, h.x, h.y, keys); if (!R) return msg("Too few pitchers in that filter range.");
  cv.hidden = false;
  const W = Math.min(640, cv.parentElement.clientWidth || 640), Hh = Math.round(W * 0.78), m = { l: 58, r: 10, t: 10, b: 40 };
  const dpr = window.devicePixelRatio || 1; cv.width = W * dpr; cv.height = Hh * dpr; cv.style.width = W + "px"; cv.style.height = Hh + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, Hh);
  const pw = W - m.l - m.r, ph = Hh - m.t - m.b, { X, Y, n } = R;
  const sx = v => m.l + (v - X.g[0]) / (X.g[n - 1] - X.g[0]) * pw, sy = v => m.t + ph - (v - Y.g[0]) / (Y.g[n - 1] - Y.g[0]) * ph;
  const tone = v => h.cell === "eff" ? v / 0.02 : (v - 0.320) / 0.08;
  const css = getComputedStyle(document.body);
  const edges = g => g.map((v, i) => i ? (g[i - 1] + v) / 2 : v).concat(g[n - 1]); // quantile grid is uneven: cells span midpoints
  const ex = edges(X.g).map(sx), ey = edges(Y.g).map(sy);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const k = i * n + j; ctx.fillStyle = R.sup[k] < 0.08 ? "#cfccc5" : color(tone(R[h.cell][k]));
    ctx.fillRect(ex[j], ey[i + 1], ex[j + 1] - ex[j] + 0.5, ey[i] - ey[i + 1] + 0.5);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
  ctx.fillStyle = "rgba(43,47,53,.35)";
  for (const p of R.pts) { ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), 1.6, 0, 7); ctx.fill(); }
  ctx.restore();
  ctx.fillStyle = css.getPropertyValue("--muted"); ctx.font = "11px system-ui"; ctx.textAlign = "center";
  for (let t = 0; t < 5; t++) { const j = Math.round(t * (n - 1) / 4); ctx.fillText(fmtAx(h.x, X.g[j]), sx(X.g[j]), Hh - m.b + 16); }
  ctx.fillText(`${hand}HP ${AXN[h.x][0]} (${AXN[h.x][1]}) →`, m.l + pw / 2, Hh - 6);
  ctx.textAlign = "right";
  for (let t = 0; t < 5; t++) { const i = Math.round(t * (n - 1) / 4); ctx.fillText(fmtAx(h.y, Y.g[i]), m.l - 12, sy(Y.g[i]) + 4); }
  ctx.save(); ctx.translate(12, m.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(`${AXN[h.y][0]} (${AXN[h.y][1]}) →`, 0, 0); ctx.restore();
  PROF = { R, sx, sy };
  const bfShare = keys.reduce((s, k) => s + P[k].bf, 0) / T.tot;
  $("#h-legend").innerHTML = `<b>${esc(H[h.b].name)}</b> (${avg.side}HB) vs ${R.pts.length} ${hand}HP (${Math.round(bfShare * 100)}% of batters faced). His average vs ${hand}HP: ${woba(avg.avg)}. ` +
    (h.cell === "eff" ? `Red = better than his usual, blue = worse. Fixed scale, saturates at ±20 wOBA points.` : `Red = high projected wOBA, blue = low. Fixed scale, .240 to .400, same for every hitter.`);
  // four boxes: both axes split at the BF-weighted median, among the pitchers on the map (filter kept)
  const box = (xh, yh) => { const ks = keys.filter(k => (T[h.x][k] > 0.5) === !!xh && (T[h.y][k] > 0.5) === !!yh), c = ks.length ? typeScore(h.b, hand, ks) : null;
    return c ? `<td style="background:${color(tone(c[h.cell]))}"><b>${signed(c.eff)}</b> pts<br>${woba(c.pred)}<br><span style="font-size:11px">${ks.length} pitchers</span></td>` : `<td>–</td>`; };
  const [xl, xh] = AXN[h.x][3], [yl, yh] = AXN[h.y][3];
  quad.innerHTML = `<b>Four pitcher types</b> <span class="muted">(each axis split at the ${hand}HP median)</span>
    <table class="quad" style="max-width:520px;margin-top:8px"><thead><tr><th></th><th>${xl}</th><th>${xh}</th></tr></thead><tbody>
    <tr><td class="name">${yh}</td>${box(0, 1)}${box(1, 1)}</tr><tr><td class="name">${yl}</td>${box(0, 0)}${box(1, 0)}</tr></tbody></table>
    <div class="hint">Each box: matchup effect (pts), then projected wOBA, averaged over the pitchers in that type by batters faced.</div>`;
}
function profileHover(e) {
  const tip = $("#h-tip"); if (!PROF) return;
  const r = e.target.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  let best = null, bd = 64;
  for (const p of PROF.R.pts) { const d = (PROF.sx(p.x) - mx) ** 2 + (PROF.sy(p.y) - my) ** 2; if (d < bd) { bd = d; best = p; } }
  if (!best) { tip.hidden = true; return; }
  const q = P[best.k];
  const h = S.h, shown = [...new Set(["fb_velo", h.x, h.y, h.z].filter(Boolean))];
  tip.innerHTML = `<b>${esc(q.name)}</b> · ${shown.map(a => `${AXN[a][0]} ${fmtAx(a, q[a])}${SHARE.has(a) ? "" : " " + AXN[a][1]}`).join(", ")}<br>Projected ${woba(best.c.pred)} · effect ${pts(best.c.eff)} · ${q.bf.toLocaleString()} BF`;
  tip.hidden = false; tip.style.left = Math.min(mx + 12, r.width - 260) + "px"; tip.style.top = (my + 12) + "px";
}

function render() {
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.t === S.tab));
  document.querySelectorAll("main section").forEach(s => s.hidden = s.id !== "t-" + S.tab);
  ({ matchup: renderMatchup, leader: renderLeader, profile: renderProfile, about: () => {} })[S.tab]();
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
  for (const k of ["hand", "x", "y", "z", "cell"]) $(`#h-${k}`).onchange = e => { S.h[k] = e.target.value; render(); };
  for (const k of ["zlo", "zhi"]) $(`#h-${k}`).onchange = e => { S.h[k] = +e.target.value; render(); };
  pick($("#h-b"), labelToH, b => { S.h.b = b; render(); });
  $("#h-cv").onmousemove = profileHover; $("#h-cv").onmouseleave = () => $("#h-tip").hidden = true;
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
    ["Trout profile grid = Python smooth2d", (R => close(R.eff[15 * 30 + 15], 0.000708, 5e-5) && close(R.eff[5 * 30 + 25], -0.000237, 5e-5))(profileGrid(545361, "R", "fb_height", "brk_share", PCT.R.keys))],
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
  const h0 = S.h, l0 = S.l;
  try { Object.assign(S, JSON.parse(decodeURIComponent(location.hash.slice(1)))); } catch (e) { /* fresh state */ }
  if (!S.h || !S.h.x) S.h = h0; // links from the old heatmap tab
  if (!("z" in S.h)) S.h.z = ["fb_velo", "fb_height", "brk_share"].find(a => a !== S.h.x && a !== S.h.y); // links from before the z picker
  S.l = { ...l0, ...S.l };
  if (S.tab === "heat") S.tab = "profile";
  if (!S.m.p) { // friendly default: a well-known matchup
    const k = Object.keys(P).find(k => P[k].name === "Gerrit Cole"); S.m.p = k || null;
    S.m.hs = [545361, 592450, 665742, 660271].filter(b => H[b]);
  }
  $("#meta").textContent = `${D.meta.build}. Data generated ${D.meta.generated}. ${D.meta.note}`;
  $("#loading").hidden = true; render();
  if (location.search.includes("selftest")) selftest();
}
init();
