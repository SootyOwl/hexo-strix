// Play screen: board render/interaction, new-game modal, difficulty, bot
// stats, game lifecycle. Loaded after shared.js.

let viewX = 0, viewY = 0, viewScale = 1;
let isPanning = false, panStartX = 0, panStartY = 0, panStartVX = 0, panStartVY = 0;
let wasDragging = false;
let gameState = null;
let inFlight = false;
let localBotInFlight = false;
let selectedSide = "random";
const DIFFICULTY_SIMS = (window.__HEXO_CFG__ || {}).difficultySims || {};
const DEFAULT_DIFFICULTY = (window.__HEXO_CFG__ || {}).defaultDifficulty || "";
let selectedDifficulty = DEFAULT_DIFFICULTY;
let selectedPlayModel = localStorage.getItem("hexo_play_model") || DEFAULT_MODEL_ID;
if (!STRIX_MODELS.some(model => model.id === selectedPlayModel)) selectedPlayModel = DEFAULT_MODEL_ID;

// Difficulty tier keys double as display names (shared 1:1 with the Analysis
// strength tiers). The labels map is kept for a stable UI fallback.
const DIFFICULTY_LABELS = {
  quick: "Quick", standard: "Standard", strong: "Strong", deep: "Deep",
};
const DIFFICULTY_DESCRIPTIONS = {
  quick: "Responds fastest",
  standard: "Balances speed and search",
  strong: "Searches further",
  deep: "Searches furthest",
};

function renderDifficultyRow() {
  const row = document.getElementById("diff-row");
  const lbl = document.getElementById("diff-label");
  if (!DIFFICULTY_SIMS || Object.keys(DIFFICULTY_SIMS).length <= 1) {
    row.hidden = true; lbl.hidden = true;
    return;
  }
  row.hidden = false; lbl.hidden = false;
  const buttons = [];
  for (const key of Object.keys(DIFFICULTY_SIMS)) {
    const label = DIFFICULTY_LABELS[key] || key;
    const description = DIFFICULTY_DESCRIPTIONS[key] || "Computer opponent";
    const sel = key === selectedDifficulty ? " selected" : "";
    buttons.push(
      `<button class="diff-btn${sel}" data-diff="${key}" `
      + `onclick="selectDifficulty('${key}')">${label}`
      + `<span class="sims">${description}</span></button>`
    );
  }
  row.innerHTML = buttons.join("");
}

function renderPlayModelSelect() {
  const field = document.getElementById("play-model-field");
  const select = document.getElementById("play-model");
  if (!field || !select) return;
  field.hidden = STRIX_MODELS.length <= 1;
  populateModelSelect(select, selectedPlayModel);
}

function selectPlayModel() {
  const select = document.getElementById("play-model");
  if (!select) return;
  selectedPlayModel = strixModel(select.value).id;
  localStorage.setItem("hexo_play_model", selectedPlayModel);
  loadBotStats();
}

function selectDifficulty(d) {
  selectedDifficulty = d;
  for (const btn of document.querySelectorAll(".diff-btn")) {
    btn.classList.toggle("selected", btn.dataset.diff === d);
  }
  if (lastStats) renderBotStats(lastStats);
}

function updateDifficultyBadge() {
  const el = document.getElementById("difficulty-badge");
  if (!el) return;
  const d = gameState && gameState.difficulty;
  if (!d || !DIFFICULTY_LABELS[d]) { el.hidden = true; return; }
  el.hidden = false;
  el.textContent = DIFFICULTY_LABELS[d];
}

function updateTransform() {
  const g = document.getElementById("board-group");
  if (!g) return;
  const c = document.getElementById("board-container");
  const cx = c.clientWidth / 2, cy = c.clientHeight / 2;
  g.setAttribute("transform",
    `translate(${cx + viewX},${cy + viewY}) scale(${viewScale})`);
}

function drawBoard() {
  const svg = document.getElementById("board");
  if (!gameState) {
    svg.innerHTML = "";
    return;
  }
  const stones = gameState.stones || [];
  const stoneMap = new Map(stones.map(s => [`${s[0][0]},${s[0][1]}`, s[1]]));
  const lastSet = new Set((gameState.last_moves || []).map(m => `${m[0]},${m[1]}`));
  const seeds = stones.length ? stones.map(s => ({q:s[0][0], r:s[0][1]})) : [{q:0,r:0}];
  const margin = gameState.placement_radius ?? 4;
  const cellSet = new Set();
  for (const s of seeds) {
    for (let dq = -margin; dq <= margin; dq++) {
      for (let dr = -margin; dr <= margin; dr++) {
        if (hexDist(dq, dr) <= margin) {
          cellSet.add(`${s.q+dq},${s.r+dr}`);
        }
      }
    }
  }
  let body = `<g id="board-group">`;
  for (const key of cellSet) {
    const [q, r] = key.split(",").map(Number);
    const { x, y } = axialToPixel(q, r);
    const occ = stoneMap.get(key);
    let cls = "hex hex-empty";
    if (occ === "P1") cls = "hex hex-p1";
    else if (occ === "P2") cls = "hex hex-p2";
    if (lastSet.has(key)) cls += " hex-last";
    body += `<polygon class="${cls}" points="${hexCorners(x,y)}"
              data-q="${q}" data-r="${r}"
              onclick="onCellClick(${q},${r})"/>`;
  }
  if (gameState.terminal && gameState.winner) {
    body += winLineSvg(stones, gameState.winner, gameState.win_length || 6);
  }
  body += "</g>";
  svg.innerHTML = body;
  updateTransform();
}

async function onCellClick(q, r) {
  if (wasDragging) { wasDragging = false; return; }
  if (inFlight || !gameState || gameState.terminal || !gameState.is_human_turn) return;
  inFlight = true;
  document.getElementById("board").classList.add("disabled");
  try {
    const resp = await fetch(URL_PREFIX + "/move", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({game_id: gameState.game_id, q, r, local_bot: true}),
    });
    const body = await resp.json();
    if (resp.ok) {
      applyState(body.state);
    } else {
      setStatus(`That move could not be played: ${body.error || resp.statusText}`, true);
    }
  } catch (e) {
    setStatus("We could not reach the server. Check your connection and try the move again.", true);
  } finally {
    inFlight = false;
    document.getElementById("board").classList.remove("disabled");
  }
}

function applyState(state) {
  gameState = state;
  drawBoard();
  updateUi();
  updateDifficultyBadge();
  if (!state.terminal && !state.is_human_turn) queueMicrotask(runLocalBotTurn);
}

function localBotPosition(state) {
  const cfg = window.__HEXO_CFG__ || {};
  return {
    config: {win_length: cfg.winLength, placement_radius: cfg.placementRadius, max_moves: cfg.maxMoves},
    stones: (state.stones || []).map(s => ({
      q: s[0][0], r: s[0][1], player: s[1] === "P1" ? 1 : 2,
    })),
    to_move: state.current_player === "P1" ? 1 : 2,
    moves_remaining: state.moves_remaining_this_turn,
  };
}

async function runLocalBotTurn() {
  if (localBotInFlight || !gameState || gameState.terminal || gameState.is_human_turn) return;
  localBotInFlight = true;
  inFlight = true;
  document.getElementById("board").classList.add("disabled");
  try {
    const stateAtStart = gameState;
    // `??` (not `||`) so a legitimate quick-tier value of 0 (argmax, no
    // search) isn't mistaken for a missing key and coerced up to 64.
    const sims = Number(DIFFICULTY_SIMS[stateAtStart.difficulty] ?? 64);
    const result = await localInference("bestMove", {
      position: localBotPosition(stateAtStart), sims, mActions: 16,
      difficulty: stateAtStart.difficulty,
      modelId: stateAtStart.model_id || selectedPlayModel,
    });
    if (!gameState || gameState.game_id !== stateAtStart.game_id) return;
    const resp = await fetch(URL_PREFIX + "/bot_move", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        game_id: stateAtStart.game_id, q: result.move.q, r: result.move.r,
      }),
    });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.error || resp.statusText);
    applyState(body.state);
  } catch (error) {
    // Old/locked-down browsers keep working, but only they consume server inference.
    try {
      const resp = await fetch(URL_PREFIX + "/bot_turn_fallback", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({game_id: gameState.game_id}),
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body.error || resp.statusText);
      applyState(body.state);
    } catch (fallbackError) {
      setStatus(`The computer could not choose a move: ${fallbackError.message || fallbackError}`, true);
    }
  } finally {
    localBotInFlight = false;
    inFlight = false;
    document.getElementById("board").classList.remove("disabled");
    // A two-placement bot turn needs a second locally selected move.
    if (gameState && !gameState.terminal && !gameState.is_human_turn)
      queueMicrotask(runLocalBotTurn);
  }
}

function setStatus(text, isError = false) {
  const s = document.getElementById("status");
  s.textContent = text;
  s.style.color = isError ? "#e8857f" : "#d8d2c6";
}

function updateUi() {
  const resignBtn = document.getElementById("resign-btn");
  const copyBtn = document.getElementById("copy-htttx-btn");
  const analyzeBtn = document.getElementById("analyze-game-btn");
  if (!gameState) {
    setStatus("Click 'New game' to begin.");
    resignBtn.hidden = true;
    copyBtn.hidden = true;
    analyzeBtn.hidden = true;
    return;
  }
  if (gameState.terminal) {
    resignBtn.hidden = true;
    copyBtn.hidden = !gameState.htttx;
    analyzeBtn.hidden = !gameState.htttx;
    const w = gameState.winner, rt = gameState.result_type;
    const banner = w === null
      ? `Draw (${rt})`
      : w === gameState.human_side
        ? `You won as ${w}`
        : `Bot won as ${w} (${rt})`;
    setStatus(banner);
    return;
  }
  resignBtn.hidden = false;
  copyBtn.hidden = true;
  analyzeBtn.hidden = true;
  const who = gameState.is_human_turn
    ? `Your turn (${gameState.human_side}, ${gameState.moves_remaining_this_turn} left)`
    : `Bot thinking… (${gameState.current_player})`;
  setStatus(`Move ${gameState.n_moves} — ${who}`);
}

async function analyzeThisGame() {
  if (!gameState || !gameState.game_id) return;
  try {
    const resp = await fetch(URL_PREFIX + "/game_htttx", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({game_id: gameState.game_id}),
    });
    const body = await resp.json();
    if (!resp.ok) { setStatus(body.error || "The game could not be opened for analysis.", true); return; }
    document.getElementById("analysis-htttx").value = body.htttx;
    goToView("analysis");
    analyzeWholeGame();
  } catch (e) {
    setStatus("We could not reach the server. Check your connection and try again.", true);
  }
}

// --- hexo.did.science sandbox -> HTTTX import ---------------------------------
const svgEl = document.getElementById("board");
svgEl.addEventListener("mousedown", e => {
  isPanning = true; wasDragging = false;
  panStartX = e.clientX; panStartY = e.clientY;
  panStartVX = viewX; panStartVY = viewY;
  svgEl.classList.add("panning");
});
window.addEventListener("mousemove", e => {
  if (!isPanning) return;
  viewX = panStartVX + (e.clientX - panStartX);
  viewY = panStartVY + (e.clientY - panStartY);
  if (Math.abs(e.clientX - panStartX) + Math.abs(e.clientY - panStartY) > 4) wasDragging = true;
  updateTransform();
});
window.addEventListener("mouseup", () => {
  isPanning = false;
  svgEl.classList.remove("panning");
});
svgEl.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.0004);
  const c = document.getElementById("board-container");
  const cx = c.clientWidth/2, cy = c.clientHeight/2;
  const mx = e.clientX - c.getBoundingClientRect().left;
  const my = e.clientY - c.getBoundingClientRect().top;
  viewX = mx - factor*(mx - viewX - cx) - cx;
  viewY = my - factor*(my - viewY - cy) - cy;
  viewScale = Math.max(0.3, Math.min(4, viewScale * factor));
  updateTransform();
}, {passive:false});
window.addEventListener("resize", updateTransform);

svgEl.addEventListener("touchstart", e => {
  if (e.touches.length !== 1) return;
  isPanning = true; wasDragging = false;
  panStartX = e.touches[0].clientX; panStartY = e.touches[0].clientY;
  panStartVX = viewX; panStartVY = viewY;
}, {passive: true});
window.addEventListener("touchmove", e => {
  if (!isPanning || e.touches.length !== 1) return;
  e.preventDefault();
  const t = e.touches[0];
  viewX = panStartVX + (t.clientX - panStartX);
  viewY = panStartVY + (t.clientY - panStartY);
  if (Math.abs(t.clientX - panStartX) + Math.abs(t.clientY - panStartY) > 4) wasDragging = true;
  updateTransform();
}, {passive: false});
window.addEventListener("touchend", () => { isPanning = false; });

// --- Pinch-to-zoom (touch) -------------------------------------------------
// Two-finger pinch on touch devices. The single-finger pan handler above
// already checks `e.touches.length !== 1`, so 2-finger touches fall through
// to this handler cleanly.
let _pinchStartDist = 0;
let _pinchStartScale = 1;
let _pinchStartView = null;
let _pinchMid = null;
svgEl.addEventListener("touchstart", e => {
  if (e.touches.length !== 2) return;
  e.preventDefault();
  const [t1, t2] = e.touches;
  const dx = t1.clientX - t2.clientX, dy = t1.clientY - t2.clientY;
  _pinchStartDist = Math.hypot(dx, dy);
  _pinchStartScale = viewScale;
  _pinchStartView = { x: viewX, y: viewY };
  const c = document.getElementById("board-container");
  const rect = c.getBoundingClientRect();
  _pinchMid = { x: ((t1.clientX + t2.clientX) / 2) - rect.left,
                y: ((t1.clientY + t2.clientY) / 2) - rect.top };
  svgEl.classList.add("panning");
}, {passive: false});
svgEl.addEventListener("touchmove", e => {
  if (e.touches.length !== 2 || !_pinchStartDist) return;
  e.preventDefault();
  const [t1, t2] = e.touches;
  const dx = t1.clientX - t2.clientX, dy = t1.clientY - t2.clientY;
  const dist = Math.hypot(dx, dy);
  const factor = dist / _pinchStartDist;
  const newScale = Math.max(0.3, Math.min(4, _pinchStartScale * factor));
  // Anchor the zoom at the gesture midpoint (same math as the wheel handler
  // above — keeps the world-coord under the fingers fixed as the pinch grows).
  const c = document.getElementById("board-container");
  const cx = c.clientWidth / 2, cy = c.clientHeight / 2;
  const mx = _pinchMid.x, my = _pinchMid.y;
  const scaleRatio = newScale / _pinchStartScale;
  viewScale = newScale;
  viewX = mx - scaleRatio * (mx - _pinchStartView.x - cx) - cx;
  viewY = my - scaleRatio * (my - _pinchStartView.y - cy) - cy;
  updateTransform();
}, {passive: false});
const _endPinch = () => {
  _pinchStartDist = 0;
  _pinchStartView = null;
  _pinchMid = null;
  svgEl.classList.remove("panning");
};
svgEl.addEventListener("touchend", _endPinch);
svgEl.addEventListener("touchcancel", _endPinch);

// --- Responsive hex size ---------------------------------------------------
// shared.js dispatches "hexo:hex-size-changed" when the viewport crosses a
// breakpoint (e.g. rotate-to-portrait). Redraw the board and recenter the
// view so the new hex scale doesn't sit at a stale pan offset.
window.addEventListener("hexo:hex-size-changed", () => {
  if (gameState) drawBoard();
  viewX = 0; viewY = 0; viewScale = 1;
  updateTransform();
});

document.getElementById("modal-bg").addEventListener("click", e => {
  if (e.target === e.currentTarget) closeModal();
});

function openModal() {
  document.getElementById("modal-bg").classList.add("show");
  document.getElementById("modal-name").focus();
  renderDifficultyRow();
  renderPlayModelSelect();
  loadBotStats();
}

function closeModal() {
  document.getElementById("modal-bg").classList.remove("show");
}

let lastStats = null;

async function loadBotStats() {
  try {
    const r = await fetch(`${URL_PREFIX}/stats?model=${encodeURIComponent(selectedPlayModel)}`);
    if (!r.ok) return;
    lastStats = await r.json();
    renderBotStats(lastStats);
  } catch (_e) { /* stats are decorative */ }
}

const TIER_TAGS = {
  quick: {
    hi:  "Quick search and I'm still finding the lines.",
    mid: "Quick search, close games. Want to give me more time to think?",
    lo:  "You're staying ahead of my fastest search.",
    vlo: "You're making quick work of Quick. Nicely done.",
  },
  standard: {
    hi:  "Standard gives me enough time to find plenty of lines.",
    mid: "Standard is producing close games. Best kind of fight.",
    lo:  "You're staying ahead of my Standard search.",
    vlo: "Standard isn't slowing you down. Try giving me more search time.",
  },
  strong: {
    hi:  "Strong search is finding its share of wins.",
    mid: "Strong search and we're trading punches.",
    lo:  "You're finding the gaps in my Strong search.",
    vlo: "You keep outplaying my Strong search. Nicely done.",
  },
  deep: {
    hi:  "Deep gives me the most time to search, and I'm using it.",
    mid: "At Deep, I'm holding the line. Mostly.",
    lo:  "Deep search, and you keep finding the gaps.",
    vlo: "You're outplaying my deepest search. That's rare.",
  },
};
const NEUTRAL_TAGS = {
  hi:  "Care to tip the scales?",
  mid: "I'm holding my own. For now.",
  lo:  "It could go either way today.",
  vlo: "You all keep finding new ways to hurt me.",
};

function tierTagFor(ratio, tier) {
  const t = TIER_TAGS[tier] || NEUTRAL_TAGS;
  if (ratio >= 0.75) return t.hi;
  if (ratio >= 0.55) return t.mid;
  if (ratio >= 0.4)  return t.lo;
  return t.vlo;
}

function smallNCopy(t, tierLabel, label) {
  const n = t ? t.total : 0;
  if (n === 0) {
    return `No one has tested me at <em>${tierLabel}</em> yet against ` +
           `<em>${label}</em>. Be the first to put me on the board.`;
  }
  const w = t.bot_wins, l = t.human_wins, d = t.draws;
  const drawClause = d ? `–${d}D` : "";
  return `Only ${n} ${n === 1 ? "game" : "games"} played at ` +
         `<em>${tierLabel}</em> so far (${w}W–${l}L${drawClause}). ` +
         `Too few to call &mdash; care to swing the verdict?`;
}

function tierStats(s, tier) {
  const sims = DIFFICULTY_SIMS && DIFFICULTY_SIMS[tier];
  if (sims === undefined || !s.by_sims) return null;
  return s.by_sims[String(sims)] || null;
}

const BRAG_MIN_N = 5;
const TIER_STRENGTH_CLAIM = {};

function renderBotStats(s) {
  const cur = document.getElementById("bot-stats-current");
  const at  = document.getElementById("bot-stats-alltime");
  // The "current" record is now per training step; show the step unless the
  // label already encodes it (auto labels are run@step).
  const step = (s.step !== null && s.step !== undefined) ? s.step : null;
  let label = s.model_label || "this version";
  if (step != null && !String(label).includes("@" + step)) label += ` · version ${step}`;
  const a = s.all_time || {total: 0, bot_wins: 0, human_wins: 0, draws: 0};

  const tierLabel = DIFFICULTY_LABELS[selectedDifficulty] || "this strength";
  const t = tierStats(s, selectedDifficulty);
  const tierTotal = t ? t.total : 0;

  let body;
  if (tierTotal < BRAG_MIN_N) {
    body = smallNCopy(t, tierLabel, label);
  } else {
    const decided = Math.max(t.bot_wins + t.human_wins, 1);
    const ratio = t.bot_wins / decided;
    const drawClause = t.draws ? ` (with ${t.draws} ${t.draws === 1 ? "draw" : "draws"})` : "";
    const stem = `On <em>${tierLabel}</em> as <em>${label}</em>, I've beaten ` +
                 `${t.bot_wins} ${t.bot_wins === 1 ? "human" : "humans"} and ` +
                 `fallen to ${t.human_wins}${drawClause}.`;
    body = `${stem} ${tierTagFor(ratio, selectedDifficulty)}`;
  }

  const claim = TIER_STRENGTH_CLAIM[selectedDifficulty];
  if (claim) {
    body += ` <span class="strength-claim">${claim}</span>`;
  }
  cur.innerHTML = body;

  if (a.total > 0) {
    at.textContent = `Across all versions and search-effort settings: ` +
                     `${a.bot_wins}W–${a.human_wins}L–${a.draws}D from ${a.total} games.`;
  } else {
    at.textContent = "";
  }
}
loadBotStats();

function selectSide(s) {
  selectedSide = s;
  for (const btn of document.querySelectorAll(".side-btn")) {
    btn.classList.toggle("selected", btn.dataset.side === s);
  }
}

async function startGame() {
  const name = document.getElementById("modal-name").value.trim();
  const eloRaw = document.getElementById("modal-elo").value.trim();
  const elo = eloRaw ? parseFloat(eloRaw) : null;
  try {
    const resp = await fetch(URL_PREFIX + "/new_game", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        human_side: selectedSide,
        name: name || null,
        difficulty: selectedDifficulty,
        self_reported_elo: isFinite(elo) && elo >= 0 ? elo : null,
        local_bot: true,
        model_id: selectedPlayModel,
      }),
    });
    if (!resp.ok) {
      setStatus(`The game could not start: ${resp.statusText}`, true);
      return;
    }
    const body = await resp.json();
    applyState(body.state);
    localStorage.setItem("hexo_game_id", body.game_id);
    location.hash = `#g=${body.game_id}`;
    closeModal();
  } catch (e) {
    setStatus("We could not reach the server. Check your connection and try again.", true);
  }
}

async function resign() {
  if (!gameState || gameState.terminal) return;
  if (!confirm("Resign this game? The computer will be recorded as the winner.")) return;
  inFlight = true;
  try {
    const resp = await fetch(URL_PREFIX + "/resign", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({game_id: gameState.game_id}),
    });
    const body = await resp.json();
    if (resp.ok) applyState(body.state);
    else setStatus(`The game could not be resigned: ${body.error || resp.statusText}`, true);
  } catch (e) {
    setStatus("We could not reach the server. Check your connection and try again.", true);
  } finally {
    inFlight = false;
  }
}

async function copyHtttx() {
  if (!gameState || !gameState.htttx) return;
  try {
    await navigator.clipboard.writeText(gameState.htttx);
    setStatus("Game record copied.");
  } catch (e) {
    setStatus("The game record could not be copied. Check your browser's clipboard permission.", true);
  }
}
