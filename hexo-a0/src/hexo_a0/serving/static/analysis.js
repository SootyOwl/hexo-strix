// Analysis screen: HDS import, SSE loader, move tree, verdicts, eval bar,
// analysis board, and the app entry point (DOMContentLoaded). Loaded last.

let hdsConvertedHtttx = "";

async function convertHds() {
  const src = document.getElementById("hds-input").value.trim();
  const statusEl = document.getElementById("hds-status");
  if (!src) { statusEl.textContent = "Enter a sandbox URL or code."; return; }
  statusEl.textContent = "Converting…";
  try {
    const resp = await fetch(URL_PREFIX + "/convert_hds", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({src}),
    });
    const body = await resp.json();
    if (!resp.ok) { statusEl.textContent = body.error || "Conversion failed"; return; }
    hdsConvertedHtttx = body.htttx;
    document.getElementById("hds-htttx").value = body.htttx;
    document.getElementById("hds-name").textContent = body.name || "";
    let note = `${body.move_count} stones`;
    if (body.offset_applied) {
      note += ` · recentered by (${body.offset_applied[0]}, ${body.offset_applied[1]})`;
    }
    if (body.colors_swapped) {
      console.log("HDS colors swapped to match P1/P2 convention");
    }
    statusEl.textContent = note;
    document.getElementById("hds-result").hidden = false;
    document.getElementById("hds-analyze-btn").disabled = false;
  } catch (e) {
    statusEl.textContent = `Network error: ${e}`;
  }
}

function openConvertedInAnalysis() {
  if (!hdsConvertedHtttx) return;
  document.getElementById("analysis-htttx").value = hdsConvertedHtttx;
  loadAnalysis();
}

async function copyHdsHtttx() {
  const el = document.getElementById("hds-status");
  try {
    await navigator.clipboard.writeText(hdsConvertedHtttx);
    el.textContent = "copied HTTTX to clipboard";
  } catch (e) {
    el.textContent = "copy failed: " + e;
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  // The path selects the screen; the hash carries the line/game. Old share links
  // were "/#c=<moves>" (analysis hash on the play path) — upgrade them in place to
  // "/analysis#c=..." so path and content agree.
  const analysisHash = location.hash.startsWith("#c=") || location.hash.startsWith("#a=");
  let view = currentPathView();
  if (view === "play" && analysisHash) {
    view = "analysis";
    try { history.replaceState(null, "", viewPath("analysis") + location.hash); } catch (_e) {}
  }

  if (view === "analysis") {
    setView("analysis");
    // Deep link into a specific line: #c=<compact> (or legacy #a=<decimal>).
    let deepMoves = null;
    if (location.hash.startsWith("#c=")) deepMoves = decodeMovesCompact(location.hash.slice(3));
    else if (location.hash.startsWith("#a=")) deepMoves = decodeMovesHash(location.hash.slice(3));
    if (deepMoves && deepMoves.length) {
      document.getElementById("analysis-htttx").value = serializeHtttx([[0, 0], ...deepMoves]);
      loadAnalysis();
    }
    return;
  }

  // Play view.
  setView("play");
  let gid = null;
  if (location.hash.startsWith("#g=")) gid = location.hash.slice(3);
  else gid = localStorage.getItem("hexo_game_id");
  if (gid) {
    try {
      const resp = await fetch(URL_PREFIX + "/state", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({game_id: gid}),
      });
      if (resp.ok) {
        const body = await resp.json();
        applyState(body.state);
        return;
      }
    } catch (_e) { /* fall through to new-game modal */ }
    localStorage.removeItem("hexo_game_id");
    location.hash = "";
  }
  setStatus("Click 'New game' to begin.");
  openModal();
});

// --- Analysis view ---
// Trajectory values are native side-to-move perspective (the model's value
// head), so they flip sign every ply. We display everything from P1's fixed
// perspective so the eval bar / per-position value reads as a single
// consistent advantage line (positive = P1 winning), like policy_viewer.
let analysisMode = false;
let analysisMoves = [];          // loaded mainline move list incl. [0,0] seed
let analysisTrajectory = null;   // /analyze_game result (drives the eval bar)
let analysisView = { x: 0, y: 0, scale: 1 };
let analysisPanning = false, analysisPanStart = { x: 0, y: 0, vx: 0, vy: 0 };
function updateAnalysisHash() {
  const line = analysisCurrent ? lineOf(analysisCurrent) : analysisMoves;
  const newHash = `#c=${encodeMovesCompact(line || [[0, 0]])}`;
  if (location.hash !== newHash) {
    try { history.replaceState(null, "", newHash); } catch (_e) { location.hash = newHash; }
  }
}

// HTTTX text uses explore.htttx.io's coordinate convention, a mirror of our
// internal axial frame: internal (q, r) <-> HTTTX (q + r, -r). Self-inverse,
// so parseHtttx and serializeHtttx apply the same map (mirrors serving/htttx.py).
function mirrorAxial(q, r) { return [q + r, -r]; }

function parseHtttx(text) {
  const re = /\[(-?\d+),(-?\d+)\]/g;
  const out = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    out.push(mirrorAxial(parseInt(m[1]), parseInt(m[2])));
  }
  return out;
}

// --- Move tree (PGN-style with side-line variations) -----------------------
// analysisTree: root node {move:null,...}. analysisMain: the loaded mainline
// spine (array of nodes). analysisCurrent: the node the board currently shows.
// Each node: {move:[q,r]|null, player, parent, children:[], result, depth}.
// `result` is the per-position analysis: a mainline node points at its
// /analyze_game trajectory entry; a side-line node carries its own /analyze
// result (fetched on demand).
let analysisTree = null;
let analysisMain = [];
let analysisCurrent = null;
let analysisCancel = null;  // AbortController for an in-flight /analyze_game

function _newNode(move, player, parent, result) {
  return {move, player, parent, children: [], result,
          depth: parent ? parent.depth + 1 : 0};
}

function lineOf(node) {
  // Full move list (incl. the [0,0] seed at depth 0) from root to `node`.
  const out = [];
  let n = node;
  while (n) { if (n.move) out.unshift(n.move); n = n.parent; }
  return [[0, 0], ...out];
}

function playerAtDepth(depth) {
  // Seed (depth 0) is P1; then P2,P2,P1,P1,P2,P2,... (2 placements/turn).
  if (depth === 0) return "P1";
  return ((depth - 1) % 4) < 2 ? "P2" : "P1";
}

// Read an SSE stream, invoking onEvent(event, data) per frame. Resolves when
// the stream ends. Honors an AbortSignal (aborting cancels server-side).
async function streamSSE(url, payload, onEvent, signal) {
  const resp = await fetch(url, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload), signal,
  });
  if (!resp.ok) {
    let msg = resp.statusText;
    try { msg = (await resp.json()).error || msg; } catch (_e) {}
    throw new Error(msg);
  }
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let sep;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      let ev = "message", data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) { try { onEvent(ev, JSON.parse(data)); } catch (_e) {} }
    }
  }
}

async function loadAnalysis() {
  const text = document.getElementById("analysis-htttx").value;
  const moves = parseHtttx(text);
  if (moves.length === 0) {
    document.getElementById("analysis-info").textContent = "No moves parsed.";
    return;
  }
  analysisMoves = [[0, 0], ...moves];
  if (analysisCancel) analysisCancel.abort();
  const myCancel = new AbortController();
  analysisCancel = myCancel;
  showProgress(true, "Starting analysis…", 0, 0, null);
  try {
    let result = null;
    await streamSSE(URL_PREFIX + "/analyze_game", {moves: analysisMoves},
      (ev, data) => {
        if (ev === "queued") {
          showProgress(true, "Queued — another analysis is running…", 0, 0, null);
        } else if (ev === "progress") {
          showProgress(true, "Analyzing…", data.done, data.total, data.eta_seconds);
        } else if (ev === "result") {
          result = data;
        } else if (ev === "error") {
          throw new Error(data.error || "analysis failed");
        }
      }, myCancel.signal);
    // A newer loadAnalysis may have started while we awaited; if so, this
    // result is stale — discard it so it can't overwrite the newer tree.
    if (analysisCancel !== myCancel) return;
    showProgress(false);
    if (!result) {
      document.getElementById("analysis-info").textContent = "Analysis returned no result.";
      return;
    }
    buildTreeFromTrajectory(result);
  } catch (e) {
    if (analysisCancel === myCancel) showProgress(false);
    if (e.name === "AbortError") return;
    if (analysisCancel === myCancel)
      document.getElementById("analysis-info").textContent = `Analysis error: ${e.message || e}`;
  }
}

function buildTreeFromTrajectory(result) {
  analysisTrajectory = result;             // mainline result (drives eval bar)
  const tr = result.trajectory || [];
  // Build the mainline spine. trajectory[i] is the position AFTER applying
  // analysisMoves[0..i]; node i's move is analysisMoves[i] (null for the seed).
  analysisTree = _newNode(null, null, null, tr[0] || null);
  analysisMain = [analysisTree];
  let parent = analysisTree;
  for (let i = 1; i < tr.length; i++) {
    const mv = analysisMoves[i] || null;
    const node = _newNode(mv, playerAtDepth(i), parent, tr[i]);
    parent.children.push(node);
    analysisMain.push(node);
    parent = node;
  }
  const slider = document.getElementById("analysis-slider");
  slider.max = Math.max(0, analysisMain.length - 1);
  setCurrent(analysisMain[analysisMain.length - 1]);
  renderEvalBar();
  renderMoveTree();
}

function setCurrent(node) {
  analysisCurrent = node;
  // Any ordinary navigation (slider, undo, board click, a plain move link)
  // dismisses a selected missed-win callout; showMissedWin re-selects it
  // AFTER calling this, once its own navigation has landed.
  missedWinSelected = null;
  const slider = document.getElementById("analysis-slider");
  // Sync the slider to the depth IF the node is on the mainline.
  if (node && analysisMain[node.depth] === node) slider.value = node.depth;
  renderNode(node);
  renderEvalBar();
  renderMoveTree();
  renderMissedWinCallout();
  updateAnalysisHash();
}

function showProgress(on, msg, done, total, eta) {
  const wrap = document.getElementById("analysis-progress");
  const bar = document.getElementById("analysis-progress-bar");
  const lbl = document.getElementById("analysis-progress-label");
  if (!wrap) return;
  // Use "block" (not "") for the on case: the element has a CSS rule
  // `display:none`, so clearing the inline style to "" would fall back to that
  // rule and keep the bar hidden during loading.
  wrap.style.display = on ? "block" : "none";
  if (!on) return;
  const pct = total > 0 ? Math.round(100 * done / total) : 0;
  if (bar) bar.style.width = pct + "%";
  let etaStr = "";
  if (eta !== null && eta !== undefined && total > 0) {
    etaStr = eta >= 60 ? ` · ETA ${Math.round(eta / 60)}m${String(Math.round(eta % 60)).padStart(2,"0")}s`
                       : ` · ETA ${Math.round(eta)}s`;
  }
  if (lbl) lbl.textContent = total > 0 ? `${msg} ${done}/${total}${etaStr}` : msg;
}

function returnToMainline() {
  if (analysisMain.length) setCurrent(analysisMain[analysisMain.length - 1]);
}

function analysisUndo() {
  // Step back to the parent of the current node.
  if (analysisCurrent && analysisCurrent.parent) setCurrent(analysisCurrent.parent);
}

function renderEvalBar() {
  const canvas = document.getElementById("analysis-eval-bar");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0a0c0b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!analysisTrajectory || !analysisTrajectory.trajectory) return;
  const tr = analysisTrajectory.trajectory;
  const boundaries = new Set(analysisTrajectory.boundary_indices || []);
  const W = canvas.width, H = canvas.height;
  const pad = 4;
  const n = tr.length;
  if (n < 1) return;
  const midY = H / 2;
  // Midline (P1/P2 even).
  ctx.strokeStyle = "#1b201f";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(W, midY); ctx.stroke();
  const xAt = (i) => n === 1 ? W / 2 : pad + (W - 2 * pad) * i / (n - 1);
  const yAt = (v) => midY - v * (H / 2 - pad);
  // Filled area: P1-orange above midline, P2-blue below — clipped per side so
  // the colours don't both paint the whole bar.
  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, W, midY); ctx.clip();
  ctx.fillStyle = "#f08a3c44";
  ctx.beginPath(); ctx.moveTo(xAt(0), midY);
  for (let i = 0; i < n; i++) { const v = p1Perspective(tr[i].value, tr[i].current_player); ctx.lineTo(xAt(i), yAt(v)); }
  ctx.lineTo(xAt(n - 1), midY); ctx.closePath(); ctx.fill();
  ctx.restore();
  ctx.save();
  ctx.beginPath(); ctx.rect(0, midY, W, midY); ctx.clip();
  ctx.fillStyle = "#3fb6d944";
  ctx.beginPath(); ctx.moveTo(xAt(0), midY);
  for (let i = 0; i < n; i++) { const v = p1Perspective(tr[i].value, tr[i].current_player); ctx.lineTo(xAt(i), yAt(v)); }
  ctx.lineTo(xAt(n - 1), midY); ctx.closePath(); ctx.fill();
  ctx.restore();
  // The eval line itself (P1 perspective — no per-ply oscillation).
  ctx.strokeStyle = "#ece6da";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xAt(i);
    const v = p1Perspective(tr[i].value, tr[i].current_player);
    const y = yAt(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // Turn-boundary dots, coloured by P1 advantage.
  for (let i = 0; i < n; i++) {
    if (!boundaries.has(i)) continue;
    const x = xAt(i);
    const v = p1Perspective(tr[i].value, tr[i].current_player);
    const y = yAt(v);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = v > 0.05 ? "#79cf9a" : v < -0.05 ? "#e25c5c" : "#f2b65a";
    ctx.fill();
  }
  // Slider position marker.
  const slider = document.getElementById("analysis-slider");
  const si = parseInt(slider.value);
  if (si >= 0 && si < n) {
    const v = p1Perspective(tr[si].value, tr[si].current_player);
    ctx.strokeStyle = "#c9a35e";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xAt(si), pad); ctx.lineTo(xAt(si), H - pad); ctx.stroke();
  }
}

// Slider scrubs the MAINLINE (pure-client, no server call — every mainline
// position already has its full /analyze_game result cached in the tree node).
function onAnalysisSlider() {
  const idx = parseInt(document.getElementById("analysis-slider").value);
  if (analysisMain[idx]) setCurrent(analysisMain[idx]);
}

function currentAnalysisIdx() {
  // The index into the CURRENT line (lineOf(analysisCurrent)) — used for the
  // last-placed-hex / quality lookups.
  return analysisCurrent ? analysisCurrent.depth : 0;
}

// Re-draw the current node (used by the heatmap toggle so flipping it on/off
// repaints the board immediately).
function rerenderCurrentAnalysis() {
  renderNode(analysisCurrent);
}

// Move-quality is computed server-side (classify_turn_quality) and shipped on
// the trajectory entry at turn boundaries as {label, icon, color}. The frontend
// just reads node.result.quality.
function qualityOf(node) {
  return node && node.result ? (node.result.quality || null) : null;
}

// A turn normally = 2 placements, ending at even depth (seed=P1@d0; P2@d1,d2;
// P1@d3,d4; ...). But a WINNING/terminal placement ends the turn early (at odd
// depth), so a terminal node is also a turn end regardless of parity.
function isTurnEnd(node) {
  if (!node) return false;
  if (node.result && node.result.terminal) return true;
  return node.depth >= 2 && node.depth % 2 === 0;
}
// Depth where the current turn STARTED (the pre-turn position). For an even
// (normal) turn end that's depth-2; for a terminal turn that ended after a
// single placement, it's depth-1.
function turnStartDepth(node) {
  const single = node.result && node.result.terminal && node.depth % 2 === 1;
  return node.depth - (single ? 1 : 2);
}

const _QUAL_ICON = {best: "★", good: "✓", mistake: "?", blunder: "✗"};
// Match server QUALITY_COLORS: gold best, green good, amber mistake, crimson blunder.
const _QUAL_COLOR = {best: "#f2c14e", good: "#79cf9a", mistake: "#e0a23a", blunder: "#e25c5c"};

// Client-side port of classify_turn_quality (analysis.py) so that turns played
// out in a SIDE LINE get the same best/good/mistake/blunder verdict the server
// computes for the mainline. preResult = the pre-turn position's /analyze
// (mover to move, has q_hat); postResult = the turn-end position's /analyze;
// turnMoves = the (up to 2) placements played this turn.
// The engine's CHOICE at a position is argmax improved_policy (the MCTS visit
// distribution = what it actually plays), NOT argmax q_hat: the value head can
// rate an under-visited move higher while the search correctly avoids it (e.g. a
// move that fails to block an open four). q_hat only SIZES the loss between two
// moves at the SAME position, where the estimates are directly comparable. This
// mirrors the server's classify_turn_quality so mainline and side lines agree.
function _bestMoveBy(res, name) {
  const arr = res && res[name];
  if (!arr || !res.legal) return null;
  let bi = -1, bp = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (res.candidate_set && !res.candidate_set[i]) continue;
    if (arr[i] > bp) { bp = arr[i]; bi = i; }
  }
  return bi >= 0 ? res.legal[bi] : null;
}
function _qOfMove(res, mv) {
  if (!res || !res.q_hat || !res.legal) return null;
  for (let i = 0; i < res.legal.length; i++)
    if (res.legal[i][0] === mv[0] && res.legal[i][1] === mv[1]) return res.q_hat[i];
  return null;
}
function labelFromLoss(matched, loss) {
  if (matched) return "best";
  if (loss >= 0.40) return "blunder";
  if (loss >= 0.15) return "mistake";
  return "good";
}

// POST a line to /analyze; returns the position result (JSON) or null.
async function analyzePosition(moves) {
  try {
    const resp = await fetch(URL_PREFIX + "/analyze", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({moves}),
    });
    return resp.ok ? await resp.json() : null;
  } catch (_e) { return null; }
}

// Client mirror of classify_turn_quality (analysis.py) — ORDER-INDEPENDENT: a
// turn is judged by its placed-SET vs the engine's best placed-set, so [A,B] and
// [B,A] reach the same board and get the same verdict. `node` is the turn-end
// node. Async because the engine's 2nd pick can lie off the played line (when the
// engine's top pick wasn't played first) and then needs an on-demand /analyze.
// Returns the same quality shape the server ships, or null if unclassifiable.
async function computeTurnQuality(node) {
  const keyOf = (m) => `${m[0]},${m[1]}`;
  const a = turnStartDepth(node);
  let start = node; while (start && start.depth > a) start = start.parent;
  if (!start || !start.result) return null;
  const seq = []; { let n = node; while (n && n.depth > a) { seq.unshift(n); n = n.parent; } }
  if (!seq.length) return null;
  const played = seq.map(s => s.move);

  const e0 = _bestMoveBy(start.result, "improved_policy");
  if (!e0) return null;
  const engineLine = [e0];
  let engineEndQ = _qOfMove(start.result, e0);
  if (played.length >= 2) {
    let e1 = null;
    if (keyOf(e0) === keyOf(played[0])) {
      // Engine's first pick == player's first move → after-position is on-line.
      const after = seq[0].result;
      e1 = after ? _bestMoveBy(after, "improved_policy") : null;
      if (e1) engineEndQ = _qOfMove(after, e1);
    } else {
      // Off-line: evaluate the position after the engine's OWN first pick.
      const res = await analyzePosition([...lineOf(start), e0]);
      if (res) { e1 = _bestMoveBy(res, "improved_policy"); if (e1) engineEndQ = _qOfMove(res, e1); }
    }
    if (e1) engineLine.push(e1);
  }
  const engineSet = new Set(engineLine.map(keyOf));
  const playedSet = new Set(played.map(keyOf));
  const matched = engineLine.length === played.length &&
                  [...playedSet].every(k => engineSet.has(k));
  // Player's achieved end-of-turn value: q of the last placement at the node
  // just before it (the same board the engine's line is scored against).
  const beforeLast = seq.length >= 2 ? seq[seq.length - 2] : start;
  const playerEndQ = beforeLast && beforeLast.result
    ? _qOfMove(beforeLast.result, played[played.length - 1]) : null;
  let loss = 0;
  if (!matched && engineEndQ != null && playerEndQ != null) loss = Math.max(0, engineEndQ - playerEndQ);
  const label = labelFromLoss(matched, loss);
  return {
    label, icon: _QUAL_ICON[label], color: _QUAL_COLOR[label],
    matched, engine_pair: engineLine, played_pair: played,
    loss, player_end_q: playerEndQ, engine_end_q: engineEndQ, turn_start_depth: a,
  };
}

// After a side-line node is analyzed, if it ends a turn, compute + attach its
// (order-independent) verdict — mirrors the server's classify_turn_quality.
async function attachSideLineVerdict(node) {
  if (!node || !node.result || node.result.quality) return;   // skip if server already set it
  if (!isTurnEnd(node)) return;
  const q = await computeTurnQuality(node);
  if (q) node.result.quality = q;
}

// Render the position for a tree node: info line, board, heatmap, quality marks.
// The verdict card is ORDER-INDEPENDENT: a turn is judged by its resulting board
// (its placed-set) vs the engine's best placed-set, so a reversed in-turn order
// that reaches the same position still "matches". The whole turn's placements are
// marked, not just the last stone.
function renderNode(node) {
  if (!node || !node.result) { return; }
  const result = node.result;
  const info = document.getElementById("analysis-info");
  const cp = result.current_player || "?";
  const v1 = p1Perspective(result.value, cp);
  const evalStr = v1 >= 0 ? `+${v1.toFixed(2)}` : v1.toFixed(2);
  const q = qualityOf(node);

  updateGauge(v1, cp);
  const fmt = (arr) => arr.map(m => `[${m[0]},${m[1]}]`).join("");

  // Turn verdict: read the (order-free) fields the server/attachSideLineVerdict
  // shipped on quality — played-set vs engine's best set, sized by end-of-turn q.
  let playedMoves = null, turnPlayer = null, startNode = null;
  if (q && isTurnEnd(node)) {
    const a = (q.turn_start_depth != null) ? q.turn_start_depth : turnStartDepth(node);
    startNode = node; while (startNode && startNode.depth > a) startNode = startNode.parent;
    turnPlayer = (startNode && startNode.result && startNode.result.current_player)
               || playerAtDepth(node.depth);
    playedMoves = q.played_pair || null;
  }

  let html;
  if (q && playedMoves) {
    info.classList.add("vc");
    info.style.setProperty("--c", q.color);
    const lost = q.label === "mistake" || q.label === "blunder";
    const pcls = turnPlayer === "P1" ? "p1c" : "p2c";
    const who = `<span class="${pcls}">${turnPlayer}</span>`;
    const sgn = (x) => (x >= 0 ? "+" : "") + Number(x).toFixed(2);
    const enginePair = q.engine_pair || [];

    let board, pickTier = "", metric = "";
    if (q.matched) {
      // Same resulting position as the engine's line. Flag a reversed order so the
      // "why isn't this a mistake?" question answers itself.
      const reordered = enginePair.length === playedMoves.length && fmt(enginePair) !== fmt(playedMoves);
      board = `${who} matched the engine's line <b class="mvc">${fmt(playedMoves)}</b>`
            + (reordered ? ` <span class="ro-note">(same position, reversed order)</span>` : "") + ".";
    } else if (enginePair.length) {
      board = `${who} played <b class="mvc">${fmt(playedMoves)}</b>. The engine `
            + `${lost ? "preferred" : "also liked"} <b class="mvc">${fmt(enginePair)}</b>`
            + (lost ? `, which reaches a stronger position` : "") + ".";
      // Explore the engine's line from the turn-start position (chips branch from
      // that node; ranked by policy — the engine's choice).
      if (startNode && startNode.result) {
        pickTier = `<div class="vc-sec"><div class="h">engine's line here</div>`
                 + `<p>${renderTopMovesHtml(startNode.result, startNode.depth)}</p></div>`;
      }
      // Metric compares the two full TURNS at their end positions (comparable q̂).
      if (q.player_end_q != null && q.engine_end_q != null)
        metric = `<div class="vc-metric"><span>your turn <b>${sgn(q.player_end_q)}</b></span><span class="sep">·</span>`
               + `<span>engine's <b>${sgn(q.engine_end_q)}</b></span><span class="sep">·</span>`
               + `<span>swing <b>${sgn(q.player_end_q - q.engine_end_q)}</b></span></div>`;
    } else {
      board = `${who} played <b class="mvc">${fmt(playedMoves)}</b>.`;
    }

    html =
      `<div class="vc-head"><div class="vc-glyph">${q.icon}</div>`
      + `<div class="vc-label">${q.label}<small>turn ${Math.ceil(node.depth / 2)} · ${turnPlayer}</small></div></div>`
      + metric
      + `<div class="vc-sec"><div class="h">on the board</div><p>${board}</p></div>`
      + pickTier
      + `<div class="vc-prov">Turns are judged by the resulting position, not the move order; the engine's choice is its policy (what it plays), not the raw value.</div>`;
  } else {
    info.classList.remove("vc");
    info.style.removeProperty("--c");
    html = `<span class="ro-pos">Position ${node.depth + 1}/${lineOf(node).length} · to move <b>${cp}</b></span>`
         + `<span class="ro-eval"> · eval (P1) <b>${evalStr}</b></span>`
         + renderTopMovesHtml(result);
  }
  info.innerHTML = html;
  // The BOARD heatmap always shows THIS position's suggested next moves.
  drawAnalysisBoard(result, result, q, playedMoves);
}

// Update the advantage gauge (needle + pole readouts) from a P1-perspective
// eval. Reveals the gauge + legend once analysis data exists.
function updateGauge(v1, cp) {
  const wrap = document.getElementById("gauge-wrap");
  const legend = document.getElementById("board-legend");
  if (!wrap) return;
  if (v1 === null || v1 === undefined || isNaN(v1)) { wrap.hidden = true; if (legend) legend.hidden = true; return; }
  wrap.hidden = false; if (legend) legend.hidden = false;
  const clamped = Math.max(-1, Math.min(1, v1));
  // Tug-of-war: P1 is the left pole, so a P1 advantage (v1 > 0) pulls the needle
  // LEFT toward P1, and a P2 advantage pulls it right. (v1 is P1-perspective.)
  document.getElementById("gauge-needle").style.left = `${((1 - clamped) / 2 * 100).toFixed(1)}%`;
  const s = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);
  const p1El = document.getElementById("gauge-v1"), p2El = document.getElementById("gauge-v2");
  p1El.textContent = s(v1);        // each pole reads positive when THAT player is ahead
  p2El.textContent = s(-v1);
  // Emphasise the leader (dim the trailing pole) so who's winning is obvious.
  p1El.style.opacity = clamped < -0.02 ? "0.4" : "1";
  p2El.style.opacity = clamped > 0.02 ? "0.4" : "1";
}

// A compact "engine likes:" row of the top candidate moves, ranked by the
// improved policy (MCTS visit-improved), with q_hat eval. When `branchDepth` is
// given, clicking a chip branches from the node at THAT depth on the current
// line (used by a verdict card to explore the engine's line from the exact
// decision point); otherwise it plays from the current node.
function renderTopMovesHtml(result, branchDepth) {
  if (!result || !result.improved_policy || !result.legal) return "";
  const ip = result.improved_policy, qh = result.q_hat, cs = result.candidate_set;
  const idx = ip.map((p, i) => i).filter(i => !cs || cs[i]);
  idx.sort((a, b) => ip[b] - ip[a]);   // rank by policy — the engine's choice
  const top = idx.slice(0, 5);
  if (!top.length) return "";
  const useDepth = branchDepth !== undefined && branchDepth !== null;
  let s = `<br><span style="font-size:11px;color:#8b8f88">Engine likes: </span>`;
  for (const i of top) {
    const mv = result.legal[i];
    const qv = qh ? (qh[i] >= 0 ? "+" : "") + qh[i].toFixed(2) : "";
    const onclick = useDepth ? `analysisBranchAtDepth(${branchDepth},${mv[0]},${mv[1]})`
                             : `analysisCellClick(${mv[0]},${mv[1]})`;
    s += `<span class="topmove" onclick="${onclick}" `
       + `title="explore this line">[${mv[0]},${mv[1]}]<span style="color:#5f635d">${qv}</span></span> `;
  }
  return s;
}

// boardResult drives the stones + cell layout (the position you're looking at).
// heatResult drives the q_hat / probs heatmap + best-move markers (== boardResult
// for normal positions; the PRE-TURN position for a turn verdict). playedMoves,
// if given, are this turn's placements to mark with the quality icon.
function drawAnalysisBoard(boardResult, heatResult, quality, playedMoves) {
  const svg = document.getElementById("analysis-board");
  if (!boardResult || !boardResult.legal) { svg.innerHTML = ""; return; }
  heatResult = heatResult || boardResult;
  const stones = boardResult.stones || [];
  const stoneMap = new Map(stones.map(s => [`${s[0][0]},${s[0][1]}`, s[1]]));
  const margin = 8;
  const seeds = stones.length ? stones.map(s => ({q:s[0][0], r:s[0][1]})) : [{q:0,r:0}];
  const cellSet = new Set();
  for (const s of seeds) {
    for (let dq = -margin; dq <= margin; dq++) {
      for (let dr = -margin; dr <= margin; dr++) {
        if ((Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2 <= margin) {
          cellSet.add(`${s.q+dq},${s.r+dr}`);
        }
      }
    }
  }
  for (const [q, r] of boardResult.legal) cellSet.add(`${q},${r}`);
  if (heatResult.legal) for (const [q, r] of heatResult.legal) cellSet.add(`${q},${r}`);

  const showHeat = document.getElementById("analysis-heatmap").checked;
  // Suggested-next-move heatmap: rank the side-to-move's MCTS-visited candidates
  // by the improved policy (how much the engine wants to play each) and shade
  // them in THAT player's colour, intensity ∝ strength; q_hat labels the top few.
  const mover = heatResult.current_player;
  const BASE = mover === "P2" ? [63, 182, 217] : [240, 138, 60];   // cyan / ember
  const RING = mover === "P2" ? "#9fe6f8" : "#ffcf9a";             // brighter tint
  const ip = heatResult.improved_policy, qh = heatResult.q_hat, cs = heatResult.candidate_set;
  const sugg = [];
  if (showHeat && ip) for (let i = 0; i < heatResult.legal.length; i++) {
    if (cs && !cs[i]) continue;                                    // MCTS-visited only
    sugg.push({ key: `${heatResult.legal[i][0]},${heatResult.legal[i][1]}`, ip: ip[i], q: qh ? qh[i] : null });
  }
  sugg.sort((a, b) => b.ip - a.ip);
  const maxIp = sugg.length ? sugg[0].ip : 0;
  const suggMap = new Map(sugg.map((s, rank) => [s.key, { ...s, rank }]));
  const topKey = sugg.length ? sugg[0].key : null;

  let body = `<g id="analysis-board-group">`;
  const labels = [];  // numeric q_hat labels for in-contention moves
  for (const key of cellSet) {
    const [q, r] = key.split(",").map(Number);
    const { x, y } = axialToPixel(q, r);
    const occ = stoneMap.get(key);
    let cls = "hex hex-empty";
    let fill = null, stroke = "";
    if (occ === "P1") { cls = "hex hex-p1"; fill = "#f08a3c"; }
    else if (occ === "P2") { cls = "hex hex-p2"; fill = "#3fb6d9"; }
    else if (showHeat && suggMap.has(key)) {
      // Suggested move for the side to move: player colour, opacity ∝ strength.
      const s = suggMap.get(key);
      const t = maxIp > 0 ? s.ip / maxIp : 0;
      fill = `rgba(${BASE[0]},${BASE[1]},${BASE[2]},${(0.14 + 0.60 * t).toFixed(3)})`;
      if (s.rank < 3 && s.q != null) labels.push({ x, y, t: (s.q >= 0 ? "+" : "") + s.q.toFixed(2) });
    }
    body += `<polygon class="${cls}" points="${hexCorners(x,y)}" data-q="${q}" data-r="${r}"`
           + (fill ? ` style="fill:${fill}${stroke}"` : "")
           + ` onclick="analysisCellClick(${q},${r})"/>`;
  }
  // Last-turn highlight (like policy_viewer): outline each side's most recent
  // turn — P1 light orange, P2 light blue. Walk the ordered move list back from
  // the current node, collecting the two latest contiguous same-player runs.
  if (analysisCurrent) {
    const line = lineOf(analysisCurrent);          // [[0,0](seed,P1), m1(P2), ...]
    const playerAt = (i) => i === 0 ? "P1" : playerAtDepth(i);
    const lastTurn = {P1: [], P2: []};
    let i = line.length - 1;
    let run = playerAt(i);
    while (i >= 0 && playerAt(i) === run) { lastTurn[run].push(line[i]); i--; }
    if (i >= 0) { run = playerAt(i); while (i >= 0 && playerAt(i) === run) { lastTurn[run].push(line[i]); i--; } }
    // Last-turn highlight: a LIGHT ring (lighter than the stone) drawn fully
    // INSIDE the cell (inset via the hexCorners scale) so it reads clearly.
    const TURN_RING = {P1: "#ffd7a8", P2: "#b3ecfb"};
    for (const player of ["P1", "P2"]) for (const mv of lastTurn[player]) {
      const p = axialToPixel(mv[0], mv[1]);
      body += `<polygon points="${hexCorners(p.x, p.y, 0.74)}" fill="none" stroke="${TURN_RING[player]}" stroke-width="2.5" stroke-opacity="0.95" pointer-events="none"/>`;
    }
  }
  // Terminal position: highlight the winning 6-in-a-row (parity with the play
  // board), so the final move shows the win rather than a bare position.
  if (boardResult.terminal && boardResult.winner) {
    body += winLineSvg(stones, boardResult.winner, (analysisTrajectory && analysisTrajectory.win_length) || 6);
  }
  // Ring the single strongest suggestion in the mover's brighter tint.
  if (showHeat && topKey) {
    const [tq, tr] = topKey.split(",").map(Number);
    const p = axialToPixel(tq, tr);
    body += `<circle cx="${p.x}" cy="${p.y}" r="${S * 0.6}" fill="none" stroke="${RING}" stroke-width="2.5" pointer-events="none"/>`;
  }
  // q_hat numeric labels on contention moves (drawn above fills).
  if (showHeat) for (const l of labels) {
    body += `<text x="${l.x}" y="${l.y + 1}" font-size="${Math.round(S * 0.34)}" fill="#f4efe5" text-anchor="middle" dominant-baseline="middle" pointer-events="none">${l.t}</text>`;
  }
  // Forcing-line (VCF) overlay: number every cell of the solved PV, attacker
  // placements filled in the winner's stone colour, defender replies muted/
  // outlined. Ownership comes from the server's per-cell replay
  // (forcing.pv_owners) — the PV's chunk lengths are NOT a fixed pairs-of-2
  // cadence (the first chunk is whatever moves_remaining_this_turn() was on
  // the solved side, and the final chunk can be a single cell that ends the
  // game), so it can't be inferred from position alone. If the server's
  // replay failed (pv_owners null), fall back to styling every cell as
  // attacker rather than guessing a cadence that's often wrong.
  if (boardResult.forcing && boardResult.forcing.pv && boardResult.forcing.pv.length) {
    const f = boardResult.forcing;
    const attackerColor = f.winner === "P2" ? "#3fb6d9" : "#f08a3c";
    // Defender's forced blocks render in the DEFENDER's own stone colour
    // (still dotted via .forcing-pv-defender), so the two sides of the line
    // are immediately distinguishable instead of colour-matching the winner.
    const defenderColor = f.winner === "P2" ? "#f08a3c" : "#3fb6d9";
    f.pv.forEach((mv, i) => {
      const p = axialToPixel(mv[0], mv[1]);
      const isAttacker = f.pv_owners ? f.pv_owners[i] === f.winner : true;
      const cls = `forcing-pv ${isAttacker ? "forcing-pv-attacker" : "forcing-pv-defender"}`;
      const labelCls = `forcing-pv-label ${isAttacker ? "forcing-pv-label-attacker" : "forcing-pv-label-defender"}`;
      const fontSize = Math.round(S * 0.42);
      if (isAttacker) {
        body += `<circle class="${cls}" cx="${p.x}" cy="${p.y}" r="${S * 0.42}" fill="${attackerColor}"/>`;
        body += `<text class="${labelCls}" x="${p.x}" y="${p.y + 1}" font-size="${fontSize}">${i + 1}</text>`;
      } else {
        body += `<circle class="${cls}" cx="${p.x}" cy="${p.y}" r="${S * 0.42}" stroke="${defenderColor}"/>`;
        body += `<text class="${labelCls}" x="${p.x}" y="${p.y + 1}" font-size="${fontSize}" fill="${defenderColor}">${i + 1}</text>`;
      }
    });
  }
  // Move-quality icon: mark EVERY placement of the verdict's turn (not just the
  // last stone), so a 2-placement blunder shows the whole turn. Falls back to
  // the single current move for side-line nodes with no turn context.
  if (quality) {
    const marks = (playedMoves && playedMoves.length) ? playedMoves
                : (analysisCurrent && analysisCurrent.move ? [analysisCurrent.move] : []);
    for (const mv of marks) {
      const p = axialToPixel(mv[0], mv[1]);
      body += `<circle cx="${p.x}" cy="${p.y}" r="${S * 0.42}" fill="${quality.color}" opacity="0.9" pointer-events="none"/>`;
      body += `<text x="${p.x}" y="${p.y + 1}" font-size="${Math.round(S * 0.5)}" font-weight="bold" fill="#0d0f0e" text-anchor="middle" dominant-baseline="middle" pointer-events="none">${quality.icon}</text>`;
    }
  }
  body += "</g>";
  svg.innerHTML = body;
  updateAnalysisTransform();
  updateForcingBanner(boardResult.forcing);
}

// One-line banner above the board reporting the both-side VCF solve, if any
// (result.forcing; see drawAnalysisBoard's PV overlay for the coordinate
// rendering). Created on first use so no static markup needs to change.
function updateForcingBanner(forcing) {
  const container = document.getElementById("analysis-board-container");
  if (!container) return;
  let banner = document.getElementById("forcing-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "forcing-banner";
    banner.className = "forcing-pv-banner";
    container.insertBefore(banner, container.firstChild);
  }
  if (!forcing) { banner.hidden = true; banner.textContent = ""; return; }
  banner.hidden = false;
  // attacker_is_mover=true is a PROVEN win (the side to move executes it).
  // attacker_is_mover=false comes from a perspective-FLIP solve ("winner
  // would win if it were their turn") — the actual side to move places
  // first and may break it, so it is a threat to answer, NOT a lost
  // position. Don't overstate it.
  const defender = forcing.winner === "P1" ? "P2" : "P1";
  banner.textContent = forcing.attacker_is_mover
    ? `${forcing.winner} has a forced win (${forcing.pv_len}-move line)`
    : `${forcing.winner} threatens a forced win — ${defender} must defend this turn`;
  // "win"/"threat" classes carry the colour (observatory.css);
  // attacker_is_mover means the win is proven, not merely threatened.
  banner.classList.toggle("win", forcing.attacker_is_mover);
  banner.classList.toggle("threat", !forcing.attacker_is_mover);
}

function updateAnalysisTransform() {
  const g = document.getElementById("analysis-board-group");
  if (!g) return;
  const c = document.getElementById("analysis-board-container");
  const cx = c.clientWidth / 2, cy = c.clientHeight / 2;
  g.setAttribute("transform",
    `translate(${cx + analysisView.x},${cy + analysisView.y}) scale(${analysisView.scale})`);
}

// Click a board hex / "engine likes" chip -> branch from the CURRENT node.
function analysisCellClick(q, r) { return branchFrom(analysisCurrent, q, r); }

// Branch from the node at `depth` on the current line — used by verdict-card
// chips to explore the engine's line from the exact decision point (walks up
// from the current node, so it needs no node ids).
function analysisBranchAtDepth(depth, q, r) {
  let n = analysisCurrent;
  while (n && n.depth > depth) n = n.parent;
  return branchFrom(n || analysisCurrent, q, r);
}

async function branchFrom(node, q, r) {
  if (analysisPanning) return;
  if (!node || !node.result || !node.result.legal) return;
  if (!node.result.legal.some(c => c[0] === q && c[1] === r)) return;
  // If this child already exists (mainline or a prior side line), just go there.
  let child = node.children.find(c => c.move && c.move[0] === q && c.move[1] === r);
  if (child) { setCurrent(child); return; }
  // New side line: add a child node, fetch its /analyze on demand.
  child = _newNode([q, r], playerAtDepth(node.depth + 1), node, null);
  node.children.push(child);
  setCurrent(child);
  const info = document.getElementById("analysis-info");
  info.textContent = "Analyzing side line…";
  try {
    const moves = lineOf(child);
    const resp = await fetch(URL_PREFIX + "/analyze", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({moves}),
    });
    const r2 = await resp.json();
    if (!resp.ok) { discardSideLine(node, child, r2.error || resp.statusText); return; }
    // /analyze returns the position AFTER the move; carry current_player + value.
    child.result = r2;
    // If this move completed a turn, judge it (best/good/mistake/blunder) just
    // like the mainline — so extended/puzzle lines get verdicts too.
    await attachSideLineVerdict(child);
    if (analysisCurrent === child) { renderNode(child); renderMoveTree(); }
  } catch (e) {
    discardSideLine(node, child, `Network error: ${e}`);
  }
}

// A failed /analyze must not leave `child` in the tree: branchFrom's
// existing-child early-exit would then forever navigate to a node that renders
// nothing (result=null) and never re-fetch — the "stuck side line". Drop the
// placeholder and step back so re-clicking the cell retries from scratch.
function discardSideLine(parent, child, message) {
  const i = parent.children.indexOf(child);
  if (i >= 0) parent.children.splice(i, 1);
  if (analysisCurrent === child) setCurrent(parent);
  else renderMoveTree();
  // After setCurrent repaints the parent's info line, surface why the side
  // line vanished.
  const info = document.getElementById("analysis-info");
  if (info) info.textContent = message;
}

// --- PGN-style move tree rendering -----------------------------------------
function _moveLink(node) {
  const active = node === analysisCurrent ? " move-active" : "";
  // missed_win is per-PLACEMENT (server-computed, analyze_game_full only —
  // side-line nodes from /analyze never carry it), so it's checked here
  // rather than once per turn: whichever of the turn's (up to 2) move links
  // squandered the win gets the badge, not necessarily the turn-end one.
  const mw = node.result && node.result.missed_win;
  const badge = mw
    ? ` <span class="missed-win-badge" onclick="showMissedWin(${node._id})" `
    + `title="A forced win existed one placement earlier">&#9889; Missed win!</span>`
    : "";
  return `<span class="move-link${active}" onclick="scrubToNodeId(${node._id})">`
       + `[${node.move[0]},${node.move[1]}]</span>${badge}`;
}

// Selected missed-win callout: the {by, at_prefix, first_move, pv, pv_len,
// pv_owners} dict from the badge that's currently pinned open, or null.
let missedWinSelected = null;

// Click a "Missed win!" badge: navigate the board to `at_prefix` (the
// position where the win still existed — Task 2's forcing-pv overlay
// renders the line there automatically via that node's own `forcing` field,
// which is exactly what `missed_win` was copied from) and pin the callout
// panel open describing it.
function showMissedWin(nodeId) {
  const node = _nodeById[nodeId];
  const mw = node && node.result && node.result.missed_win;
  if (!mw) return;
  const target = analysisMain[mw.at_prefix];
  if (target) setCurrent(target);   // clears missedWinSelected + navigates
  missedWinSelected = mw;
  renderMissedWinCallout();
}

// Compact callout panel for the selected missed-win badge: "P1 had a forced
// win in N" + the squandered PV as numbered coordinate chips, attacker
// placements emphasized and defender replies muted per `pv_owners` (falling
// back to "every cell is attacker" if the server's replay couldn't determine
// ownership — mirrors drawAnalysisBoard's forcing-pv overlay fallback).
// Created on first use, like updateForcingBanner, so no static markup needs
// to change.
function renderMissedWinCallout() {
  const movetree = document.getElementById("analysis-movetree");
  if (!movetree) return;
  let panel = document.getElementById("missed-win-callout");
  if (!panel) {
    panel = document.createElement("div");
    panel.id = "missed-win-callout";
    panel.className = "missed-win-callout";
    movetree.insertAdjacentElement("afterend", panel);
  }
  const mw = missedWinSelected;
  if (!mw) { panel.hidden = true; panel.innerHTML = ""; return; }
  panel.hidden = false;
  // Defender chips take the defender's own stone colour (matches the board
  // overlay's defender colouring); attacker chips keep the phosphor emphasis.
  const mwDefColor = mw.by === "P2" ? "var(--p1)" : "var(--p2)";
  const chips = mw.pv.map((c, i) => {
    const isAttacker = mw.pv_owners ? mw.pv_owners[i] === mw.by : true;
    const cls = `pv-chip ${isAttacker ? "pv-chip-attacker" : "pv-chip-defender"}`;
    const style = isAttacker ? "" : ` style="color:${mwDefColor}"`;
    return `<span class="${cls}"${style}>${i + 1}. [${c[0]},${c[1]}]</span>`;
  }).join("");
  panel.innerHTML = `<div class="mwc-head">${mw.by} had a forced win in ${mw.pv_len}</div>`
                   + `<div class="mwc-pv">${chips}</div>`;
}

// Assign ids so onclick can find nodes. Re-runs on every render: clears the
// registry and re-numbers the whole tree (a node's id must always be present
// in _nodeById, or move-list clicks become no-ops).
let _nodeId = 0;
const _nodeById = {};
function _idify(node) {
  node._id = _nodeId++;
  _nodeById[node._id] = node;
  for (const c of node.children) _idify(c);
}

function scrubToNodeId(id) {
  const node = _nodeById[id];
  if (node) setCurrent(node);
}

// Split a line of nodes into turns: contiguous runs of the same player (up to 2
// placements each). nodes[] excludes the root/seed. Returns [{player, nodes,
// endNode}]. The seed (P1 @ depth 0) is its own degenerate turn.
function _turnsOf(nodes) {
  const turns = [];
  let cur = null;
  for (const n of nodes) {
    const p = playerAtDepth(n.depth);
    if (!cur || cur.player !== p) { cur = {player: p, nodes: [n], endNode: n}; turns.push(cur); }
    else { cur.nodes.push(n); cur.endNode = n; }
  }
  return turns;
}

// A turn cell: its (up to 2) move links + the end-of-turn verdict icon.
function _turnCellHtml(turn) {
  if (!turn) return "";
  const q = qualityOf(turn.endNode);
  const icon = q ? ` <span class="mv-q" style="color:${q.color}" title="${q.label}">${q.icon}</span>` : "";
  return turn.nodes.map(_moveLink).join(" ") + icon;
}

// Render the move list as a chess.com-style grid: one row per ROUND (a P1 turn
// + the following P2 turn), with the round number, then each side's turn cell.
// Side lines branch as indented sub-rows under the round they diverge from.
function renderMoveTree() {
  const el = document.getElementById("analysis-movetree");
  if (!el || !analysisTree) return;
  _nodeId = 0; for (const k in _nodeById) delete _nodeById[k];
  _idify(analysisTree);

  const mainNodes = analysisMain.slice(1);  // drop the seed (depth 0)
  const turns = _turnsOf(mainNodes);
  // HeXO opens with P1's seed then P2 moves first, so the first real turn is P2.
  // Pair turns into rounds [P1?, P2?]. We lead each round with P1's turn; the
  // very first round has no P1 turn (just the seed), so it shows P2 only.
  // Simplest robust pairing: walk turns, start a new round whenever we hit a P1
  // turn (or at the start), and slot P1/P2 by the turn's player.
  const rounds = [];
  let round = null;
  for (const t of turns) {
    if (t.player === "P1") { round = {P1: t, P2: null}; rounds.push(round); }
    else { // P2
      if (!round || round.P2) { round = {P1: null, P2: t}; rounds.push(round); }
      else round.P2 = t;
    }
  }

  // Variations: a side line starts at a node whose parent's mainline child is a
  // different node. Render each as an indented block of its own mini move list.
  function variationRows(startNode, depthIndent) {
    // Build the side line's own spine (following first-child).
    const nodes = [];
    let n = startNode;
    while (n && n.move) { nodes.push(n); n = n.children[0]; }
    const vturns = _turnsOf(nodes);
    let cells = vturns.map(t => {
      const lbl = (t.player === "P1" ? "P1 " : "P2 ");
      return `<span class="var-side">${lbl}</span>${_turnCellHtml(t)}`;
    }).join(" ");
    let html = `<div class="variation" style="margin-left:${depthIndent}px">(${cells})</div>`;
    // nested variations off any node in this side line
    for (const nd of nodes) for (const c of nd.children.slice(1)) html += variationRows(c, depthIndent + 12);
    return html;
  }
  // Collect variation HTML keyed by the mainline depth they branch from, so we
  // can drop them right after the relevant round.
  const varByDepth = {};
  const addVar = (parentNode) => {
    const mainChild = analysisMain[parentNode.depth + 1];
    for (const c of parentNode.children) {
      if (c !== mainChild && c.move) {
        (varByDepth[parentNode.depth] = varByDepth[parentNode.depth] || []).push(variationRows(c, 8));
      }
    }
  };
  addVar(analysisTree);
  for (const node of mainNodes) addVar(node);

  // Emit the grid.
  let html = `<div class="mv-grid">`;
  html += `<div class="mv-h">#</div><div class="mv-h">P1</div><div class="mv-h">P2</div>`;
  rounds.forEach((rd, i) => {
    const alt = i % 2 === 1 ? " mv-row-alt" : "";
    html += `<div class="mv-num${alt}">${i + 1}.</div>`;
    html += `<div class="mv-cell${alt}">${rd.P1 ? _turnCellHtml(rd.P1) : "<span class='mv-dim'>…</span>"}</div>`;
    html += `<div class="mv-cell${alt}">${rd.P2 ? _turnCellHtml(rd.P2) : "<span class='mv-dim'>…</span>"}</div>`;
    // Variations branching from any node within this round's turns.
    let vhtml = "";
    for (const t of [rd.P1, rd.P2]) if (t) for (const n of t.nodes)
      if (varByDepth[n.depth]) vhtml += varByDepth[n.depth].join("");
    if (vhtml) html += `<div class="mv-varspan">${vhtml}</div>`;
  });
  // Variations off the seed (root) appear before round 1.
  html += `</div>`;
  let seedVar = varByDepth[0] ? `<div class="mv-grid"><div class="mv-varspan">${varByDepth[0].join("")}</div></div>` : "";
  el.innerHTML = (seedVar + html) || "<span style='color:#5f635d'>No moves.</span>";
}

function serializeHtttx(moves) {
  let body = "version[1];\n";
  if (moves.length <= 1) return body;
  const rest = moves.slice(1).map(m => mirrorAxial(m[0], m[1]));
  let turn = 1, i = 0;
  while (i < rest.length) {
    const a = rest[i];
    const b = rest[i + 1];
    if (b === undefined) {
      body += `${turn}. [${a[0]},${a[1]}];\n`;
    } else {
      body += `${turn}. [${a[0]},${a[1]}][${b[0]},${b[1]}];\n`;
    }
    turn++; i += 2;
  }
  return body;
}

// Pan/zoom for analysis board
const analysisSvg = document.getElementById("analysis-board");
analysisSvg.addEventListener("mousedown", e => {
  analysisPanning = true;
  analysisPanStart = { x: e.clientX, y: e.clientY, vx: analysisView.x, vy: analysisView.y };
  analysisSvg.classList.add("panning");
});
window.addEventListener("mousemove", e => {
  if (!analysisPanning) return;
  analysisView.x = analysisPanStart.vx + (e.clientX - analysisPanStart.x);
  analysisView.y = analysisPanStart.vy + (e.clientY - analysisPanStart.y);
  updateAnalysisTransform();
});
window.addEventListener("mouseup", () => {
  analysisPanning = false;
  analysisSvg.classList.remove("panning");
});
analysisSvg.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.0004);
  const c = document.getElementById("analysis-board-container");
  const cx = c.clientWidth / 2, cy = c.clientHeight / 2;
  const rect = c.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  analysisView.x = mx - factor * (mx - analysisView.x - cx) - cx;
  analysisView.y = my - factor * (my - analysisView.y - cy) - cy;
  analysisView.scale = Math.max(0.3, Math.min(4, analysisView.scale * factor));
  updateAnalysisTransform();
}, {passive:false});
window.addEventListener("resize", updateAnalysisTransform);
