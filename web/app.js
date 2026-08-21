/* The scene viewer.
 *
 * Everything it draws is a point cloud or a line set, so there is one geometry
 * per layer and no scene graph to speak of. The only slightly involved part is
 * `paint()`, which decides what colour each walkable cell gets: its height, the
 * detection set of a selected vertex, or the state of a schedule step.
 */

const $ = (s) => document.querySelector(s);

/* ---------------------------------------------------------------- decoding */

function decode(b64, Type) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}

/** uint16 back to metres, relative to the scene centre. */
function dequantise(pack) {
  const q = decode(pack.q, Uint16Array);
  const out = new Float32Array(q.length);
  const [sx, sy, sz] = pack.scale, [ox, oy, oz] = pack.offset;
  for (let i = 0; i < q.length; i += 3) {
    out[i] = q[i] * sx + ox;
    out[i + 1] = q[i + 1] * sy + oy;
    out[i + 2] = q[i + 2] * sz + oz;
  }
  return out;
}

/** Run lengths back to a Uint8 mask; the first run is always "off". */
function unrle(runs, n) {
  const m = new Uint8Array(n);
  let at = 0, on = 0;
  for (const r of runs) {
    if (on) m.fill(1, at, at + r);
    at += r;
    on ^= 1;
  }
  return m;
}

/* ------------------------------------------------------------------ colour */

function ramp(t, stops) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t <= t1) {
      const f = (t - t0) / Math.max(t1 - t0, 1e-9);
      return [c0[0] + (c1[0] - c0[0]) * f,
              c0[1] + (c1[1] - c0[1]) * f,
              c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}

const SURFACE = [[0, [0.07, 0.31, 0.36]], [0.5, [0.14, 0.58, 0.58]],
                 [1, [0.89, 0.84, 0.63]]];
const CLOUD = [[0, [0.25, 0.28, 0.32]], [0.6, [0.45, 0.49, 0.54]],
               [1, [0.69, 0.71, 0.76]]];
const CLEARED = [0.14, 0.58, 0.64];
const DIRTY = [0.36, 0.18, 0.27];
const WATCHED = [0.98, 0.69, 0.29];
const SEEN_BY = [0.98, 0.55, 0.22];
// Violet and pink, not green: the surface is teal and the graph edges are
// amber, and a route has to be told apart from both at a glance.
const ROUTE = 0x9d7bff;      // every route, drawn faintly
const ROUTE_HOT = 0xff5fbf;  // the selected one
// Routes are lifted clear of the surface. The walkable cells are drawn as
// sprites nearly two cells wide, so a polyline sitting on the surface is
// hidden by the very floor it runs over; the selected one is lifted further
// again so it reads over the rest.
const ROUTE_LIFT = 0.45;
const ROUTE_LIFT_SEL = 0.75;

/* ------------------------------------------------------------------- scene */

let renderer, camera, world, raycaster;
let S = null;               // the loaded scene payload
let G = {};                 // the THREE objects of the current scene
let playing = null;

function init() {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  $("#view").appendChild(renderer.domElement);
  camera = new THREE.PerspectiveCamera(52, 1, 0.2, 6000);
  world = new THREE.Scene();
  world.background = new THREE.Color(0x0b0f14);
  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 1.6;

  const sel = $("#scene");
  for (const s of window.SCENE_INDEX) {
    const o = document.createElement("option");
    o.value = s.name;
    o.textContent = `${s.title}  (${s.mb} MB)`;
    sel.appendChild(o);
  }
  sel.onchange = () => loadScene(sel.value);

  for (const id of ["cloud", "surface", "edges", "shady", "nodes", "routes"]) {
    $("#l-" + id).onchange = applyLayers;
  }
  $("#route-from").onchange = () => { fillRouteTo(); drawRoute(); };
  $("#route-to").onchange = drawRoute;
  $("#vertex").onchange = () => { $("#show-sched").checked = false; paint(); };
  $("#step").oninput = () => { $("#vertex").value = "-1";
                               $("#show-sched").checked = true; paint(); };
  $("#show-sched").onchange = () => { $("#vertex").value = "-1"; paint(); };
  $("#play").onclick = togglePlay;
  addEventListener("keydown", (e) => { if (e.key === "r" || e.key === "R") frame(); });
  addEventListener("resize", resize);
  renderer.domElement.addEventListener("click", pick);
  orbit(renderer.domElement);

  resize();
  const want = hashState();
  sel.value = window.SCENE_INDEX.some((s) => s.name === want.scene)
    ? want.scene : window.SCENE_INDEX[0].name;
  loadScene(sel.value);
  (function loop() { requestAnimationFrame(loop); renderer.render(world, camera); })();
}

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

/* -------------------------------------------------------------- loading */

const loaded = new Set();

function loadScene(name) {
  $("#loading").classList.remove("hidden");
  const go = () => { build(window.SCENES[name]); $("#loading").classList.add("hidden"); };
  if (loaded.has(name)) { setTimeout(go, 0); return; }
  const meta = window.SCENE_INDEX.find((s) => s.name === name);
  const tag = document.createElement("script");
  tag.src = meta.file;
  tag.onload = () => { loaded.add(name); go(); };
  document.body.appendChild(tag);
}

function build(data) {
  stopPlay();
  for (const k of Object.keys(G)) {
    world.remove(G[k]);
    G[k].geometry && G[k].geometry.dispose();
    G[k].material && G[k].material.dispose();
  }
  G = {};
  S = data;

  // survey cloud
  const cloud = dequantise(data.cloud);
  G.cloud = points(cloud, colourByHeight(cloud, CLOUD), 0.32);

  // walkable surface -- its colours are rewritten by paint()
  S.surf = dequantise(data.surface);
  S.nCells = S.surf.length / 3;
  S.surfBase = colourByHeight(S.surf, SURFACE);
  G.surface = points(S.surf, S.surfBase.slice(), data.cellSize * 1.7);

  // graph
  const nodes = decode(data.nodes, Float32Array);
  const edges = decode(data.edges, Uint16Array);
  const shady = decode(data.edgeShady, Uint8Array);
  S.nodes = nodes;
  S.edges = edges;
  S.shady = shady;
  const lift = 0.6;
  for (const kind of [0, 1]) {
    const pos = [];
    for (let e = 0; e < shady.length; e++) {
      if (shady[e] !== kind) continue;
      for (const v of [edges[2 * e], edges[2 * e + 1]]) {
        pos.push(nodes[3 * v], nodes[3 * v + 1], nodes[3 * v + 2] + lift);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    const m = new THREE.LineBasicMaterial({
      color: kind ? 0x7a5c46 : 0xfab04a, transparent: true,
      opacity: kind ? 0.5 : 0.75,
    });
    G[kind ? "shady" : "regular"] = new THREE.LineSegments(g, m);
  }
  const np = new Float32Array(nodes.length);
  for (let i = 0; i < nodes.length; i += 3) {
    np[i] = nodes[i]; np[i + 1] = nodes[i + 1]; np[i + 2] = nodes[i + 2] + lift;
  }
  S.nodeDraw = np;
  G.nodes = points(np, null, 2.4, 0xffeca8);

  // the walked routes, one polyline per edge
  S.routePts = dequantise(data.routes);
  S.routeOff = decode(data.routes.offsets, Uint32Array);
  S.routeSecs = decode(data.routes.seconds, Float32Array);
  S.routeMetres = decode(data.routes.metres, Float32Array);
  G.routes = segments(routeSegments(S), ROUTE, 0.7);
  G.routeSel = segments(new Float32Array(0), ROUTE_HOT, 1.0);
  G.routeDots = points(new Float32Array(0), null, 1.6, ROUTE_HOT);

  G.held = points(new Float32Array(0), null, 4.2, 0xff6a3d);

  for (const k of Object.keys(G)) world.add(G[k]);

  // detection sets, decoded lazily -- most are never looked at
  S.dsetCache = new Map();

  fillPanel();
  applyLayers();
  const want = hashState();
  if (want.vertex !== null) $("#vertex").value = want.vertex;
  if (want.step !== null) { $("#step").value = want.step; $("#show-sched").checked = true; }
  if (want.from !== null) {
    $("#route-from").value = want.from;
    fillRouteTo();
    if (want.to !== null) $("#route-to").value = want.to;
    $("#l-routes").checked = true;
    drawRoute();
  }
  paint();
  frame();
}

/** `#scene=christ-church&vertex=12` or `&step=7` -- a linkable view. */
function hashState() {
  const q = new URLSearchParams(location.hash.replace(/^#/, ""));
  const num = (k) => (q.has(k) ? parseInt(q.get(k), 10) : null);
  return { scene: q.get("scene"), vertex: num("vertex"), step: num("step"),
           from: num("from"), to: num("to") };
}

/** Every edge index, as the default "all routes" selection. */
function allEdges(sc) {
  return Array.from({ length: sc.routeOff.length - 1 }, (_, k) => k);
}

/** The given edges' polylines, flattened into consecutive segment endpoints. */
function routeSegments(sc, which, lift) {
  const keep = which || allEdges(sc);
  const dz = lift === undefined ? ROUTE_LIFT : lift;
  const out = [];
  for (const k of keep) {
    const a = sc.routeOff[k], b = sc.routeOff[k + 1];
    for (let i = a; i + 1 < b; i++) {
      out.push(sc.routePts[3 * i], sc.routePts[3 * i + 1], sc.routePts[3 * i + 2] + dz,
               sc.routePts[3 * i + 3], sc.routePts[3 * i + 4], sc.routePts[3 * i + 5] + dz);
    }
  }
  return new Float32Array(out);
}

/** The same polylines, resampled at a fixed spacing, as drawable points. */
function routeBeads(sc, which, spacing, lift) {
  const out = [];
  for (const k of which || []) {
    const a = sc.routeOff[k], b = sc.routeOff[k + 1];
    let carry = 0;
    for (let i = a; i + 1 < b; i++) {
      const p = [sc.routePts[3 * i], sc.routePts[3 * i + 1], sc.routePts[3 * i + 2]];
      const q = [sc.routePts[3 * i + 3], sc.routePts[3 * i + 4], sc.routePts[3 * i + 5]];
      const d = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
      for (let t = carry; t < d; t += spacing) {
        const f = t / Math.max(d, 1e-9);
        out.push(p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f,
                 p[2] + (q[2] - p[2]) * f + lift);
      }
      carry = d > 0 ? (spacing - ((d - carry) % spacing)) % spacing : carry;
    }
  }
  return new Float32Array(out);
}

function segments(xyz, colour, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({
    color: colour, transparent: true, opacity }));
}

function points(xyz, colours, size, colour) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
  const opts = { size, sizeAttenuation: true };
  if (colours) { g.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3)); opts.vertexColors = true; }
  else opts.color = colour === undefined ? 0xffffff : colour;
  return new THREE.Points(g, new THREE.PointsMaterial(opts));
}

function colourByHeight(xyz, stops) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 2; i < xyz.length; i += 3) { lo = Math.min(lo, xyz[i]); hi = Math.max(hi, xyz[i]); }
  const out = new Float32Array(xyz.length);
  for (let i = 0; i < xyz.length; i += 3) {
    const c = ramp((xyz[i + 2] - lo) / Math.max(hi - lo, 1e-6), stops);
    out[i] = c[0]; out[i + 1] = c[1]; out[i + 2] = c[2];
  }
  return out;
}

/* --------------------------------------------------------------- painting */

function dset(v) {
  if (!S.dsetCache.has(v)) S.dsetCache.set(v, unrle(S.detection[v], S.nCells));
  return S.dsetCache.get(v);
}

function paint() {
  const col = G.surface.geometry.attributes.color.array;
  col.set(S.surfBase);

  const v = parseInt($("#vertex").value, 10);
  const t = parseInt($("#step").value, 10);
  const showSched = $("#show-sched").checked;
  let held = [];

  if (v >= 0) {
    const m = dset(v);
    for (let i = 0; i < S.nCells; i++) {
      if (!m[i]) continue;
      col[3 * i] = SEEN_BY[0]; col[3 * i + 1] = SEEN_BY[1]; col[3 * i + 2] = SEEN_BY[2];
    }
    held = [v];
    $("#step-label").textContent = "";
  } else if (S.schedule && showSched) {
    const dirty = unrle(S.schedule.contaminated[t], S.nCells);
    held = S.schedule.steps[t];
    const seen = new Uint8Array(S.nCells);
    for (const q of held) {
      const m = dset(q);
      for (let i = 0; i < S.nCells; i++) if (m[i]) seen[i] = 1;
    }
    for (let i = 0; i < S.nCells; i++) {
      const c = seen[i] ? WATCHED : (dirty[i] ? DIRTY : CLEARED);
      col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
    }
    $("#step-label").textContent =
      `step ${t + 1} of ${S.schedule.steps.length} — ${held.length} robots`;
  } else {
    $("#step-label").textContent = "";
  }

  G.surface.geometry.attributes.color.needsUpdate = true;

  const hp = new Float32Array(held.length * 3);
  held.forEach((q, k) => {
    hp[3 * k] = S.nodeDraw[3 * q];
    hp[3 * k + 1] = S.nodeDraw[3 * q + 1];
    hp[3 * k + 2] = S.nodeDraw[3 * q + 2];
  });
  G.held.geometry.setAttribute("position", new THREE.Float32BufferAttribute(hp, 3));
  G.held.geometry.attributes.position.needsUpdate = true;
}

/* ----------------------------------------------------------------- panel */

function fillPanel() {
  const s = S.stats;
  $("#stats").innerHTML = [
    ["walkable area", `${s.areaM2.toLocaleString()} m²`],
    ["cells", s.cells.toLocaleString()],
    ["vertices", s.vertices],
    ["edges", `${s.edges} (${s.edgesShady} shady)`],
    ["detection range", `${s.range} m`],
    ["seen by nobody", `${s.uncoverableM2} m²`],
  ].map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`).join("");

  const sel = $("#vertex");
  sel.innerHTML = '<option value="-1">— none —</option>';
  for (let v = 0; v < S.stats.vertices; v++) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = `vertex ${v}`;
    sel.appendChild(o);
  }

  const from = $("#route-from");
  from.innerHTML = '<option value="-1">— none —</option>';
  for (let v = 0; v < S.stats.vertices; v++) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = `vertex ${v}`;
    from.appendChild(o);
  }
  fillRouteTo();
  drawRoute();

  const pb = $("#playback");
  if (!S.schedule) { pb.style.display = "none"; return; }
  pb.style.display = "";
  const m = S.schedule.metrics;
  $("#sched-meta").textContent =
    `${S.schedule.key}: ${m.steps} steps, peak ${m.team_peak} robots, ` +
    `residual ${m.residual_m2} m², ${(m.makespan_s / 3600).toFixed(2)} h`;
  $("#step").max = S.schedule.steps.length - 1;
  $("#step").value = 0;
}

/** Which edges leave the chosen vertex, cheapest first. */
function edgesFrom(v) {
  const out = [];
  for (let k = 0; k < S.edges.length / 2; k++) {
    const i = S.edges[2 * k], j = S.edges[2 * k + 1];
    if (i === v || j === v) out.push({ k, other: i === v ? j : i });
  }
  out.sort((a, b) => S.routeSecs[a.k] - S.routeSecs[b.k]);
  return out;
}

function fillRouteTo() {
  const v = parseInt($("#route-from").value, 10);
  const to = $("#route-to");
  to.innerHTML = "";
  if (v < 0) {
    to.innerHTML = '<option value="-1">—</option>';
    return;
  }
  const all = document.createElement("option");
  all.value = "-1";
  all.textContent = "— every route from here —";
  to.appendChild(all);
  for (const { k, other } of edgesFrom(v)) {
    const o = document.createElement("option");
    o.value = other;                       // a vertex id, so #from=12&to=61 works
    o.textContent = `vertex ${other}  —  ${S.routeMetres[k].toFixed(0)} m, `
      + fmtTime(S.routeSecs[k]);
    to.appendChild(o);
  }
}

function fmtTime(s) {
  return s < 90 ? `${s.toFixed(0)} s` : `${(s / 60).toFixed(1)} min`;
}

function drawRoute() {
  const v = parseInt($("#route-from").value, 10);
  const w = parseInt($("#route-to").value, 10);
  const hit = v >= 0 ? edgesFrom(v).find((e) => e.other === w) : null;
  let which = [];
  let text = "";

  if (hit) {
    const k = hit.k;
    which = [k];
    const i = v, j = w;
    const straight = Math.hypot(
      S.nodes[3 * i] - S.nodes[3 * j],
      S.nodes[3 * i + 1] - S.nodes[3 * j + 1],
      S.nodes[3 * i + 2] - S.nodes[3 * j + 2]);
    const detour = S.routeMetres[k] / Math.max(straight, 1e-6);
    text = `${i} → ${j}: walks ${S.routeMetres[k].toFixed(1)} m in `
      + `${fmtTime(S.routeSecs[k])} — ${detour.toFixed(2)}× the straight line `
      + `of ${straight.toFixed(1)} m.`;
  } else if (v >= 0) {
    const es = edgesFrom(v);
    which = es.map((e) => e.k);
    const secs = es.map((e) => S.routeSecs[e.k]);
    text = es.length
      ? `${es.length} routes from vertex ${v}: `
        + `${fmtTime(Math.min(...secs))} to ${fmtTime(Math.max(...secs))}.`
      : `vertex ${v} has no graph edges.`;
  }

  $("#route-meta").textContent = text;
  G.routeSel.geometry.setAttribute("position",
    new THREE.Float32BufferAttribute(
      routeSegments(S, which, ROUTE_LIFT_SEL), 3));
  G.routeSel.geometry.attributes.position.needsUpdate = true;

  // A one-pixel line disappears over a dense surface once the whole site is in
  // frame. Beads along the path do not: they scale with the view.
  G.routeDots.geometry.setAttribute("position",
    new THREE.Float32BufferAttribute(
      routeBeads(S, which, 1.5, ROUTE_LIFT_SEL), 3));
  G.routeDots.geometry.attributes.position.needsUpdate = true;
}

function applyLayers() {
  G.cloud.visible = $("#l-cloud").checked;
  G.surface.visible = $("#l-surface").checked;
  G.regular.visible = $("#l-edges").checked;
  G.shady.visible = $("#l-edges").checked && $("#l-shady").checked;
  G.nodes.visible = $("#l-nodes").checked;
  G.routes.visible = $("#l-routes").checked;
}

function togglePlay() {
  if (playing) return stopPlay();
  $("#play").textContent = "Pause";
  $("#vertex").value = "-1";
  $("#show-sched").checked = true;
  playing = setInterval(() => {
    const el = $("#step");
    el.value = (+el.value + 1) % (+el.max + 1);
    paint();
  }, 550);
}

function stopPlay() {
  if (playing) clearInterval(playing);
  playing = null;
  $("#play").textContent = "Play";
}

/* ------------------------------------------------------------- picking */

function pick(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  const m = new THREE.Vector2(
    ((ev.clientX - r.left) / r.width) * 2 - 1,
    -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(m, camera);
  const hit = raycaster.intersectObject(G.nodes, false)[0];
  if (!hit) return;
  $("#vertex").value = hit.index;
  stopPlay();
  paint();
}

/* -------------------------------------------------------------- camera */

let dist = 200, az = 0.9, el = 0.55, target = new THREE.Vector3();

function place() {
  camera.position.set(
    target.x + dist * Math.cos(el) * Math.cos(az),
    target.y + dist * Math.cos(el) * Math.sin(az),
    target.z + dist * Math.sin(el));
  camera.up.set(0, 0, 1);
  camera.lookAt(target);
}

function frame() {
  const box = new THREE.Box3().setFromBufferAttribute(
    G.surface.geometry.attributes.position);
  const size = box.getSize(new THREE.Vector3());
  target.copy(box.getCenter(new THREE.Vector3()));
  dist = Math.max(size.x, size.y) * 0.95;
  az = 0.9; el = 0.55;
  place();
}

function orbit(el0) {
  let down = null;
  el0.addEventListener("pointerdown", (e) => {
    down = { x: e.clientX, y: e.clientY, b: e.button };
    el0.setPointerCapture(e.pointerId);
  });
  el0.addEventListener("pointerup", (e) => {
    down = null;
    el0.releasePointerCapture(e.pointerId);
  });
  el0.addEventListener("pointermove", (e) => {
    if (!down) return;
    const dx = e.clientX - down.x, dy = e.clientY - down.y;
    down.x = e.clientX; down.y = e.clientY;
    if (down.b === 0) {
      az -= dx * 0.006;
      el = Math.max(-1.45, Math.min(1.45, el + dy * 0.006));
    } else {
      const s = dist * 0.0016;
      const right = new THREE.Vector3(-Math.sin(az), Math.cos(az), 0);
      const up = new THREE.Vector3(
        -Math.cos(az) * Math.sin(el), -Math.sin(az) * Math.sin(el), Math.cos(el));
      target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
    }
    place();
  });
  el0.addEventListener("contextmenu", (e) => e.preventDefault());
  el0.addEventListener("wheel", (e) => {
    e.preventDefault();
    dist = Math.max(3, Math.min(4000, dist * Math.exp(e.deltaY * 0.0012)));
    place();
  }, { passive: false });
}

init();
