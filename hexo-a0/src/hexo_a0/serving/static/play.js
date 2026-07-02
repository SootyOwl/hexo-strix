// Play screen: board render/interaction, new-game modal, difficulty, bot
// stats, game lifecycle. Loaded after shared.js.

let viewX = 0, viewY = 0, viewScale = 1;
let isPanning = false, panStartX = 0, panStartY = 0, panStartVX = 0, panStartVY = 0;
let wasDragging = false;
let gameState = null;
let inFlight = false;
let selectedSide = "random";
const DIFFICULTY_SIMS = (window.__HEXO_CFG__ || {}).difficultySims || {};
const DEFAULT_DIFFICULTY = (window.__HEXO_CFG__ || {}).defaultDifficulty || "";
let selectedDifficulty = DEFAULT_DIFFICULTY;

const DIFFICULTY_LABELS = {
  casual: "Casual", easy: "Easy", standard: "Standard", strong: "Strong",
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
    const sims = DIFFICULTY_SIMS[key];
    const label = DIFFICULTY_LABELS[key] || key;
    const sel = key === selectedDifficulty ? " selected" : "";
    buttons.push(
      `<button class="diff-btn${sel}" data-diff="${key}" `
      + `onclick="selectDifficulty('${key}')">${label}`
      + `<span class="sims">${sims} sims</span></button>`
    );
  }
  row.innerHTML = buttons.join("");
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
      body: JSON.stringify({game_id: gameState.game_id, q, r}),
    });
    const body = await resp.json();
    if (resp.ok) {
      applyState(body.state);
    } else {
      setStatus(`Server: ${body.error || resp.statusText}`, true);
    }
  } catch (e) {
    setStatus(`Network error: ${e}`, true);
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
    if (!resp.ok) { setStatus(body.error || "Analysis load failed", true); return; }
    document.getElementById("analysis-htttx").value = body.htttx;
    goToView("analysis");
    loadAnalysis();
  } catch (e) {
    setStatus(`Network error: ${e}`, true);
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

document.getElementById("modal-bg").addEventListener("click", e => {
  if (e.target === e.currentTarget) closeModal();
});

function openModal() {
  document.getElementById("modal-bg").classList.add("show");
  document.getElementById("modal-name").focus();
  renderDifficultyRow();
  loadBotStats();
}

function closeModal() {
  document.getElementById("modal-bg").classList.remove("show");
}

let lastStats = null;

async function loadBotStats() {
  try {
    const r = await fetch(URL_PREFIX + "/stats");
    if (!r.ok) return;
    lastStats = await r.json();
    renderBotStats(lastStats);
  } catch (_e) { /* stats are decorative */ }
}

const TIER_TAGS = {
  casual: {
    hi:  "Casual mode and I'm still winning? Pretend I tried.",
    mid: "I'm pulling punches and you're still feeling them. Nice.",
    lo:  "Casual was meant to be the gentle one. So much for that.",
    vlo: "You're trouncing me on the easiest setting. I'll be over here.",
  },
  easy: {
    hi:  "Easy tier and I'm finding the lines. Maybe try a real fight?",
    mid: "Easy fight, but it's a fight. Want to make me work?",
    lo:  "If I'm losing on Easy, my Casual self is in trouble.",
    vlo: "This is humbling. Pick a harder tier and put me out of my misery.",
  },
  standard: {
    hi:  "Standard is my comfort zone. Care to make me sweat?",
    mid: "Standard tier and we're trading punches. Best kind of fight.",
    lo:  "I'm slipping at Standard. The next loss is going on the fridge.",
    vlo: "You keep outplaying me at Standard. I'll go read the policy paper.",
  },
  strong: {
    hi:  "Strong is where I bring everything. Brave of you to take it on.",
    mid: "At Strong, I'm holding the line. Mostly.",
    lo:  "Strong, and you keep finding the seams. Annoying.",
    vlo: "On Strong? You're playing above me. That's rare.",
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
  if (step != null && !String(label).includes("@" + step)) label += ` · step ${step}`;
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
    at.textContent = `Lifetime across every difficulty and checkpoint: ` +
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
      }),
    });
    if (!resp.ok) {
      setStatus(`Failed to start: ${resp.statusText}`, true);
      return;
    }
    const body = await resp.json();
    applyState(body.state);
    localStorage.setItem("hexo_game_id", body.game_id);
    location.hash = `#g=${body.game_id}`;
    closeModal();
  } catch (e) {
    setStatus(`Network error: ${e}`, true);
  }
}

async function resign() {
  if (!gameState || gameState.terminal) return;
  if (!confirm("Resign this game?")) return;
  inFlight = true;
  try {
    const resp = await fetch(URL_PREFIX + "/resign", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({game_id: gameState.game_id}),
    });
    const body = await resp.json();
    if (resp.ok) applyState(body.state);
    else setStatus(`Resign failed: ${body.error || resp.statusText}`, true);
  } catch (e) {
    setStatus(`Network error: ${e}`, true);
  } finally {
    inFlight = false;
  }
}

async function copyHtttx() {
  if (!gameState || !gameState.htttx) return;
  try {
    await navigator.clipboard.writeText(gameState.htttx);
    setStatus("HTTTX copied to clipboard.");
  } catch (e) {
    setStatus(`Copy failed: ${e}`, true);
  }
}
