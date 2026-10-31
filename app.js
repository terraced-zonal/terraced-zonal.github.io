// TerracedZonal web demo: spinning spheres with Voronoi cells coloured by
// hexagonal distortion (heat map) or by neighbour count (pentagons/heptagons).
"use strict";

const ELEV = 22 * Math.PI / 180;           // camera elevation above the equator
const SPEEDS = { pause: 0, slow: 60, fast: 5 }; // seconds per revolution (0: paused)
const HALF_X = 3.2, HALF_Y = 2.4;          // zoom window, in units of unit area per point
const VMAX = { aniso: 0.20, perim: 0.07 }; // colour-bar maxima
// Candidate colour schemes (choose with ?palette=A..E): heat-map stops, the
// pentagon / heptagon colours, and the cell-boundary colour.
const PALETTES = {
  A: { name: "white \u2192 yellow \u2192 orange; teal / purple",
       stops: ["#ffffff", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014"],
       pent: "#00a3a3", hept: "#7b3294", edge: "rgba(120,120,120,0.55)", zedge: "#9a9a9a" },
  B: { name: "white \u2192 gray; orange / blue (Okabe\u2013Ito)",
       stops: ["#ffffff", "#e6e6e6", "#c8c8c8", "#a8a8a8", "#858585", "#636363"],
       pent: "#e69f00", hept: "#0072b2", edge: "rgba(70,80,95,0.55)", zedge: "#56606e" },
  C: { name: "cividis (light end); magenta / green",
       stops: ["#fdfdf6", "#fde9a0", "#e9d26a", "#c4b765", "#a19c70", "#7f8078"],
       pent: "#cc3399", hept: "#1b9e77", edge: "rgba(110,110,110,0.55)", zedge: "#8c8c8c" },
  D: { name: "white \u2192 blue; orange / purple",
       stops: ["#ffffff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd"],
       pent: "#e66100", hept: "#5d3a9b", edge: "rgba(110,110,110,0.55)", zedge: "#8c8c8c" },
  E: { name: "current (yellow \u2192 brown), gray boundaries; red / blue",
       stops: ["#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"],
       pent: "#e3412f", hept: "#1a73e8", edge: "rgba(120,120,120,0.55)", zedge: "#9a9a9a" },
};
const PAL = PALETTES[new URLSearchParams(location.search).get("palette")] || PALETTES.E;
const YLORBR = PAL.stops;
const shade = (h, f) => "#" + [1, 3, 5].map(i => Math.round(parseInt(h.slice(i, i + 2), 16) * f)
  .toString(16).padStart(2, "0")).join("");
// Colour by number of sides; <= 4 share one shade and 8+ another, matching the legend.
const DEFECT = { 3: shade(PAL.pent, 0.7), 4: shade(PAL.pent, 0.7), 5: PAL.pent, 6: "#f3efe6",
                 7: PAL.hept, 8: shade(PAL.hept, 0.7), 9: shade(PAL.hept, 0.7) };
const BOX_COLOR = "rgba(90,90,90,0.45)";   // zoom box on the sphere: gray, nearly transparent

// Camera basis: view direction d, screen right r, screen up u (right-handed).
const D = [Math.cos(ELEV), 0, Math.sin(ELEV)];
const R = [0, 1, 0];
const U = [-Math.sin(ELEV), 0, Math.cos(ELEV)];

const state = { mode: "heat", metric: "perim", outline: true, tIdx: 5, speed: "slow", angle: 0, zoomScale: 1,
                manifest: null, cache: {}, cards: [] };

const hex2rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const YLORBR_RGB = YLORBR.map(hex2rgb);
function heatRGB(v) {
  const x = Math.max(0, Math.min(1, v)) * (YLORBR_RGB.length - 1);
  const i = Math.min(Math.floor(x), YLORBR_RGB.length - 2), f = x - i;
  const a = YLORBR_RGB[i], b = YLORBR_RGB[i + 1];
  return a.map((c, k) => Math.round(c + f * (b[k] - c)));
}
function heatColor(v) { return `rgb(${heatRGB(v).join(",")})`; }
const pack = ([r, g, b]) => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
function defectColor(deg) {
  if (deg <= 3) return DEFECT[3];
  if (deg >= 9) return DEFECT[9];
  return DEFECT[deg];
}

// ---------------------------------------------------------------- data ---
async function loadSet(tIdx) {
  const target = state.manifest.targets[tIdx];
  const recs = await Promise.all(state.manifest.constructions.map(async c => {
    const file = c.files[String(target)].file;
    if (!state.cache[file]) {
      const r = await fetch("data/" + file);
      const j = await r.json();
      if (j.large) {
        const bin = async (suffix, Type) => new Type(await (await fetch(`data/${j.bin}_${suffix}`)).arrayBuffer());
        [j.pts, j.perim, j.aniso, j.deg] = await Promise.all([
          bin("pts.f32", Float32Array), bin("perim.f32", Float32Array),
          bin("aniso.f32", Float32Array), bin("deg.u8", Uint8Array)]);
        j.band_verts = Float32Array.from(j.band_verts);
      } else {
        j.pts = Float32Array.from(j.pts);
        j.verts = Float32Array.from(j.verts);
      }
      state.cache[file] = j;
    }
    return state.cache[file];
  }));
  return recs;
}

// --------------------------------------------------------------- cards ---
function makeCards() {
  const row = document.getElementById("row");
  state.manifest.constructions.forEach(c => {
    const card = document.createElement("div");
    card.className = "card" + (c.key === "tz" ? " tz" : "");
    card.innerHTML =
      (c.key === "tz" ? '<div class="badge">this paper</div>' : "") +
      `<div class="title">${c.name}</div>` +
      `<div class="family">${c.family} &middot; <span class="nval"></span></div>` +
      '<canvas class="sphere"></canvas><canvas class="zoom"></canvas>' +
      '<div class="stats"></div>';
    row.appendChild(card);
    state.cards.push({ el: card, key: c.key, sphere: card.querySelector(".sphere"),
                       zoom: card.querySelector(".zoom"), stats: card.querySelector(".stats"),
                       nval: card.querySelector(".nval"), rec: null });
  });
}

function sizeCanvas(cv, aspect) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  const h = Math.round(w * aspect);
  cv.style.height = h + "px";
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  return [cv.width, cv.height];
}

function cellColor(rec, k) {
  if (state.mode === "defect") return defectColor(rec.deg[k]);
  return heatColor(rec[state.metric][k] / VMAX[state.metric]);
}

function updateStats() {
  state.cards.forEach(card => {
    const rec = card.rec;
    if (!rec) return;
    card.nval.textContent = "N = " + rec.n.toLocaleString();
    const s = rec.summary;
    if (state.mode === "defect") {
      const pct = 100 * (s.n5 + s.n7 + s.n_other) / rec.n;
      card.stats.innerHTML = `<b style="color:${DEFECT[5]}">${s.n5.toLocaleString()}</b> pentagons, ` +
        `<b style="color:${DEFECT[7]}">${s.n7.toLocaleString()}</b> heptagons` +
        (s.n_other ? `, ${s.n_other} other` : "") +
        `<br><b>${pct < 1 ? pct.toFixed(2) : pct.toFixed(1)}%</b> of cells are not hexagons`;
    } else {
      if (state.metric === "aniso") {
        card.stats.innerHTML = `mean anisotropy: <b>${(100 * s.mean_aniso).toFixed(2)}%</b>`;
      } else {
        const f = 100 * s.frac_perim_gt1;
        card.stats.innerHTML = `mean perimeter excess: <b>${(100 * s.mean_perim).toFixed(2)}%</b>` +
          `<br><b>${f < 1 ? f.toFixed(2) : f.toFixed(1)}%</b> of cells more than 1% above P<sub>hex</sub>`;
      }
    }
  });
}

// ------------------------------------------------------------- drawing ---
function rotZ(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]];
}

// Per-point packed colours for large N (recomputed when the colouring changes).
function pointColors(rec) {
  const key = `${state.mode}|${state.metric}|${state.outline}`;
  if (rec._ckey === key) return rec._colors;
  const n = rec.n, out = new Uint32Array(n), vmax = VMAX[state.metric], vals = rec[state.metric];
  const defPacked = {};
  for (const d of [3, 4, 5, 6, 7, 8, 9]) defPacked[d] = pack(hex2rgb(DEFECT[d]));
  for (let k = 0; k < n; k++) {
    const d = Math.max(3, Math.min(9, rec.deg[k]));
    if (state.mode === "defect" || (state.outline && d !== 6)) out[k] = defPacked[d];
    else out[k] = pack(heatRGB(vals[k] / vmax));
  }
  rec._ckey = key;
  rec._colors = out;
  return out;
}

function drawSphereLarge(card, angle, ctx, w, h, rad, cx, cy) {
  const rec = card.rec, P = rec.pts, n = rec.n;
  const colors = pointColors(rec);
  const img = ctx.createImageData(w, h);
  const buf = new Uint32Array(img.data.buffer);
  const ca = Math.cos(angle), sa = Math.sin(angle);
  // Splat size: slightly larger than one cell so that the disc has no gaps.
  const s = Math.max(2, Math.min(5, Math.ceil(rad * Math.sqrt(4 * Math.PI / n) * 1.25)));
  const half = Math.floor(s / 2);
  // Draw back to front is unnecessary (only the front hemisphere is drawn), but
  // defect points are written last so that they stay visible at sub-pixel scale.
  for (let pass = 0; pass < 2; pass++) {
    for (let k = 0; k < n; k++) {
      const isDef = rec.deg[k] !== 6;
      if ((pass === 0) === (isDef && (state.mode === "defect" || state.outline))) continue;
      const x = ca * P[3 * k] - sa * P[3 * k + 1], y = sa * P[3 * k] + ca * P[3 * k + 1], z = P[3 * k + 2];
      if (x * D[0] + y * D[1] + z * D[2] <= 0.01) continue;
      const px = Math.round(cx + rad * (x * R[0] + y * R[1] + z * R[2])) - half;
      const py = Math.round(cy - rad * (x * U[0] + y * U[1] + z * U[2])) - half;
      const c = colors[k];
      for (let dy = 0; dy < s; dy++) {
        const row = (py + dy) * w;
        if (py + dy < 0 || py + dy >= h) continue;
        for (let dx = 0; dx < s; dx++) {
          const xx = px + dx;
          if (xx >= 0 && xx < w) buf[row + xx] = c;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawSphere(card, angle) {
  const rec = card.rec, cv = card.sphere;
  const [w, h] = sizeCanvas(cv, 1.0);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const rad = 0.46 * Math.min(w, h), cx = w / 2, cy = h / 2;
  if (rec.large) {
    drawSphereLarge(card, angle, ctx, w, h, rad, cx, cy);
    drawSphereOverlay(ctx, rec.n, rad, cx, cy);
    return;
  }
  const ca = Math.cos(angle), sa = Math.sin(angle);
  const V = rec.verts, P = rec.pts, nv = V.length / 3;
  // Screen coordinates of all Voronoi vertices.
  const sx = new Float32Array(nv), sy = new Float32Array(nv);
  for (let i = 0; i < nv; i++) {
    const x = ca * V[3 * i] - sa * V[3 * i + 1], y = sa * V[3 * i] + ca * V[3 * i + 1], z = V[3 * i + 2];
    sx[i] = cx + rad * (x * R[0] + y * R[1] + z * R[2]);
    sy[i] = cy - rad * (x * U[0] + y * U[1] + z * U[2]);
  }
  const n = rec.n;
  const lw = Math.max(0.15, 0.5 * Math.sqrt(1000 / n)) * (window.devicePixelRatio || 1);
  ctx.lineWidth = lw;
  ctx.strokeStyle = PAL.edge;
  const outlined = [];
  for (let k = 0; k < n; k++) {
    const x = ca * P[3 * k] - sa * P[3 * k + 1], y = sa * P[3 * k] + ca * P[3 * k + 1], z = P[3 * k + 2];
    const depth = x * D[0] + y * D[1] + z * D[2];
    if (depth <= 0.02) continue;
    const reg = rec.regions[k];
    ctx.beginPath();
    ctx.moveTo(sx[reg[0]], sy[reg[0]]);
    for (let j = 1; j < reg.length; j++) ctx.lineTo(sx[reg[j]], sy[reg[j]]);
    ctx.closePath();
    ctx.fillStyle = cellColor(rec, k);
    ctx.fill();
    if (n <= 2500) ctx.stroke();
    if (state.mode === "heat" && state.outline && rec.deg[k] !== 6) outlined.push(k);
  }
  strokeDefects(ctx, rec, outlined, (i) => [sx[i], sy[i]], 1.3 * (window.devicePixelRatio || 1));
  drawSphereOverlay(ctx, n, rad, cx, cy);
}

// Limb and the zoom box at the centre of view.
function drawSphereOverlay(ctx, n, rad, cx, cy) {
  ctx.lineWidth = 1.2 * (window.devicePixelRatio || 1);
  ctx.strokeStyle = "#222";
  ctx.beginPath(); ctx.arc(cx, cy, rad, 0, 2 * Math.PI); ctx.stroke();
  const scale = Math.sqrt(n / (4 * Math.PI));
  const z = state.zoomScale;
  const bx = rad * Math.sin(z * HALF_X / scale), by = rad * Math.sin(z * HALF_Y / scale);
  ctx.strokeStyle = BOX_COLOR;
  ctx.lineWidth = 1.2 * (window.devicePixelRatio || 1);
  ctx.strokeRect(cx - bx, cy - by, 2 * bx, 2 * by);
}

function drawZoom(card, angle) {
  const rec = card.rec, cv = card.zoom;
  const [w, h] = sizeCanvas(cv, HALF_Y / HALF_X);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  // Body-frame point at the centre of view, and its tangent frame.
  const c = rotZ(D, -angle);
  let e = [-c[1], c[0], 0];
  const en = Math.hypot(e[0], e[1]) || 1;
  e = [e[0] / en, e[1] / en, 0];
  const nn = [c[1] * e[2] - c[2] * e[1], c[2] * e[0] - c[0] * e[2], c[0] * e[1] - c[1] * e[0]];
  const n = rec.n, scale = Math.sqrt(n / (4 * Math.PI));
  const px = w / (2 * HALF_X * state.zoomScale);
  const cosr = Math.cos(1.8 * state.zoomScale * Math.hypot(HALF_X, HALF_Y) / scale);
  const V = rec.large ? rec.band_verts : rec.verts, P = rec.pts;
  const proj = i => {
    const x = V[3 * i], y = V[3 * i + 1], z = V[3 * i + 2];
    const k = Math.sqrt(2 / (1 + x * c[0] + y * c[1] + z * c[2]));
    return [w / 2 + px * scale * k * (x * e[0] + y * e[1] + z * e[2]),
            h / 2 - px * scale * k * (x * nn[0] + y * nn[1] + z * nn[2])];
  };
  ctx.lineWidth = 0.9 * (window.devicePixelRatio || 1);
  ctx.strokeStyle = PAL.zedge;
  const dots = [], outlined = [];
  const cellList = rec.large ? rec.band_cells : null;
  const nIter = rec.large ? cellList.length : n;
  const regionOf = rec.large ? (i => rec.band_regions[i]) : (k => rec.regions[k]);
  for (let i = 0; i < nIter; i++) {
    const k = rec.large ? cellList[i] : i;
    const dot = P[3 * k] * c[0] + P[3 * k + 1] * c[1] + P[3 * k + 2] * c[2];
    if (dot < cosr) continue;
    if (state.mode === "heat" && state.outline && rec.deg[k] !== 6) outlined.push(i);
    const reg = regionOf(i);
    ctx.beginPath();
    let q = proj(reg[0]);
    ctx.moveTo(q[0], q[1]);
    for (let j = 1; j < reg.length; j++) { q = proj(reg[j]); ctx.lineTo(q[0], q[1]); }
    ctx.closePath();
    ctx.fillStyle = cellColor(rec, k);
    ctx.fill();
    ctx.stroke();
    const kk = Math.sqrt(2 / (1 + dot));
    const x = P[3 * k], y = P[3 * k + 1], z = P[3 * k + 2];
    dots.push([w / 2 + px * scale * kk * (x * e[0] + y * e[1] + z * e[2]),
               h / 2 - px * scale * kk * (x * nn[0] + y * nn[1] + z * nn[2])]);
  }
  strokeDefects(ctx, rec, outlined, proj, 2.6 * (window.devicePixelRatio || 1), rec.large);
  ctx.fillStyle = "#111";
  const r = Math.max(0.8, 1.8 / Math.sqrt(state.zoomScale)) * (window.devicePixelRatio || 1);
  dots.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); });
}

// Red (5 neighbours) / blue (7) / darker (4, 8+) outlines over the heat map.
function strokeDefects(ctx, rec, cells, proj, lw, band) {
  ctx.save();
  ctx.lineWidth = lw;
  ctx.lineJoin = "round";
  for (const i of cells) {
    const k = band ? rec.band_cells[i] : i;
    const reg = band ? rec.band_regions[i] : rec.regions[k];
    ctx.beginPath();
    let q = proj(reg[0]);
    ctx.moveTo(q[0], q[1]);
    for (let j = 1; j < reg.length; j++) { q = proj(reg[j]); ctx.lineTo(q[0], q[1]); }
    ctx.closePath();
    ctx.strokeStyle = defectColor(rec.deg[k]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawAll() {
  state.cards.forEach(card => {
    if (!card.rec) return;
    drawSphere(card, state.angle);
    drawZoom(card, state.angle);
  });
}

// -------------------------------------------------------------- legend ---
function updateLegend() {
  const el = document.getElementById("legend");
  const chips = (fill) => [[5, "5 neighbours (pentagon)"], [7, "7 neighbours (heptagon)"], [4, "4"], [8, "8+"]]
    .map(([d, t]) => `<span class="chip"><span style="${fill ? `background:${DEFECT[d]}` : `border:2.5px solid ${DEFECT[d]};background:transparent`}"></span>${t}</span>`).join("");
  if (state.mode === "defect") {
    el.innerHTML = `<div class="chips">${chips(true)}<span class="chip"><span style="background:${DEFECT[6]}"></span>6 (hexagon)</span></div>`;
  } else {
    const vmax = VMAX[state.metric] * 100;
    const stops = YLORBR.map((c, i) => `${c} ${(100 * i / (YLORBR.length - 1)).toFixed(1)}%`).join(",");
    const ticks = [0, 0.25, 0.5, 0.75, 1].map(f => (f === 1 ? "\u2265" : "") + (f * vmax).toFixed(f === 0 ? 0 : 1) + "%");
    el.innerHTML = `<div class="colorbar"><div class="bar" style="background:linear-gradient(to right,${stops})"></div>` +
      `<div class="lbl">${ticks.map(t => `<span>${t}</span>`).join("")}</div>` +
      `<div style="text-align:center;margin-top:2px">${state.metric === "aniso" ? "cell anisotropy (0 for a regular hexagon)" : "perimeter excess P/P<sub>hex</sub> \u2212 1"}</div></div>` +
      (state.outline ? `<div class="chips">Outlined: ${chips(false)}</div>` : "");
  }
}

// ------------------------------------------------------------ controls ---
async function setN(tIdx) {
  state.tIdx = tIdx;
  if (state.manifest.targets[tIdx] > 10000) state.zoomScale = Math.min(state.zoomScale, 1.6);
  updateZoomLabel();
  const recs = await loadSet(tIdx);
  state.cards.forEach((card, i) => { card.rec = recs[i]; });
  updateStats();
  drawAll();
}

function setSeg(seg, attr, value) {
  let active = null;
  seg.querySelectorAll("button").forEach(b => {
    const on = b.dataset[attr] === value;
    b.classList.toggle("active", on);
    b.setAttribute("aria-checked", String(on));
    if (on) active = b;
  });
  const knob = seg.querySelector(".knob");
  knob.style.left = active.offsetLeft + "px";
  knob.style.width = active.offsetWidth + "px";
}

function setSpeed(speed) {
  if (!(speed in SPEEDS)) speed = "slow";
  state.speed = speed;
  setSeg(document.getElementById("speed"), "speed", speed);
}

function updateZoomLabel() {
  const cells = Math.round(4 * HALF_X * HALF_Y * state.zoomScale ** 2);
  document.getElementById("zoominfo").textContent =
    `Inset: about ${cells} cells around the centre of view, so it zooms in as N grows. Scroll over a sphere to resize it.`;
}

// One three-way control: perim | aniso (heat maps) or defect.
function setView(view) {
  state.mode = view === "defect" ? "defect" : "heat";
  if (view !== "defect") state.metric = view;
  setSeg(document.getElementById("view"), "view", view);
  document.getElementById("outline").disabled = view === "defect";
  document.getElementById("outline-wrap").classList.toggle("disabled", view === "defect");
  updateLegend();
  updateStats();
  drawAll();
}

async function main() {
  state.manifest = await (await fetch("data/manifest.json")).json();
  makeCards();
  const ticks = document.getElementById("nticks");
  const fmt = v => v >= 1000 ? (v / 1000) + "k" : String(v);
  ticks.innerHTML = state.manifest.targets.map(t => `<span>${fmt(t)}</span>`).join("");
  const slider = document.getElementById("nslider");
  slider.max = String(state.manifest.targets.length - 1);
  slider.addEventListener("input", () => setN(Number(slider.value)));
  document.querySelectorAll("#view button").forEach(b =>
    b.addEventListener("click", () => setView(b.dataset.view)));
  document.getElementById("outline").addEventListener("change", ev => {
    state.outline = ev.target.checked; updateLegend(); drawAll();
  });
  document.querySelectorAll("#speed button").forEach(b =>
    b.addEventListener("click", () => setSpeed(b.dataset.speed)));
  // Scroll over a sphere or its inset to change the size of the zoom window.
  const onWheel = ev => {
    ev.preventDefault();
    const maxZ = state.manifest.targets[state.tIdx] > 10000 ? 1.6 : 3;  // band data limits large N
    state.zoomScale = Math.min(maxZ, Math.max(0.4, state.zoomScale * Math.exp(ev.deltaY * 0.0015)));
    updateZoomLabel(); drawAll();
  };
  state.cards.forEach(c => { c.sphere.addEventListener("wheel", onWheel, { passive: false });
                             c.zoom.addEventListener("wheel", onWheel, { passive: false }); });
  window.addEventListener("resize", () => { setView(state.mode === "defect" ? "defect" : state.metric); setSpeed(state.speed); });
  // Optional URL parameters, e.g. ?view=defect&n=3 (legacy: mode=defect, metric=aniso)
  const qs = new URLSearchParams(location.search);
  let view = qs.get("view") || (qs.get("mode") === "defect" ? "defect" : qs.get("metric")) || "perim";
  if (!["perim", "aniso", "defect"].includes(view)) view = "perim";
  if (qs.get("outline") === "0") { state.outline = false; document.getElementById("outline").checked = false; }
  if (qs.has("n")) slider.value = qs.get("n");
  if (qs.has("speed")) state.speed = qs.get("speed");
  setView(view);
  setSpeed(state.speed);
  updateZoomLabel();
  await setN(Number(slider.value));
  let last = performance.now();
  const tick = now => {
    const dt = (now - last) / 1000;
    last = now;
    const period = SPEEDS[state.speed];
    if (period) {
      state.angle = (state.angle + 2 * Math.PI * dt / period) % (2 * Math.PI);
      drawAll();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

main();
