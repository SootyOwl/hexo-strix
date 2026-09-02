// Browser-local neural inference and Gumbel MCTS. Heavy synchronous WASM work
// stays off the UI thread; terminating this worker cancels an analysis.

import {selectForcingMove} from "./play-defense.js";

let apiPromise;
const botPromises = new Map();
const ANALYSIS_CACHE_DB = "hexo-local-analysis";
const ANALYSIS_CACHE_STORE = "positions";
// Bump whenever inference, forcing-defense semantics, or the cached result
// shape changes. The version is part of every key, so old results become
// unreachable immediately and the position is fully reanalyzed.
const ANALYSIS_CACHE_VERSION = 3;
const ANALYSIS_CACHE_MAX_ENTRIES = 512;
let analysisCachePromise;

function versioned(relative) {
  const url = new URL(relative, import.meta.url);
  url.search = self.location.search;
  return url;
}

async function loadApi() {
  if (!apiPromise) {
    apiPromise = (async () => {
      const api = await import(versioned("./solver/hexo_wasm.js").href);
      await api.default({module_or_path: versioned("./solver/hexo_wasm_bg.wasm")});
      return api;
    })();
  }
  return apiPromise;
}

async function loadBot(modelUrl) {
  if (!botPromises.has(modelUrl)) {
    botPromises.set(modelUrl, (async () => {
      const [api, response] = await Promise.all([loadApi(), fetch(modelUrl)]);
      if (!response.ok) throw new Error(`model download failed: HTTP ${response.status}`);
      const weights = new Uint8Array(await response.arrayBuffer());
      return new api.StrixBot(weights);
    })().catch(error => {
      botPromises.delete(modelUrl);
      throw error;
    }));
  }
  return botPromises.get(modelUrl);
}

function openAnalysisCache() {
  if (analysisCachePromise) return analysisCachePromise;
  analysisCachePromise = new Promise((resolve, reject) => {
    if (!self.indexedDB) { resolve(null); return; }
    const request = self.indexedDB.open(ANALYSIS_CACHE_DB, 1);
    request.onupgradeneeded = () => {
      const store = request.result.createObjectStore(ANALYSIS_CACHE_STORE, {keyPath: "key"});
      store.createIndex("saved", "saved");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  }).catch(() => null);
  return analysisCachePromise;
}

function analysisCacheKey(message, position) {
  const strength = message.strength || {};
  const stones = position.stones.map(stone => [stone.player, stone.q, stone.r])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  return JSON.stringify([
    ANALYSIS_CACHE_VERSION,
    // The server derives this query token from the newest static asset mtime.
    // Any deployed worker/WASM change therefore invalidates local results even
    // if a developer forgets the explicit semantic-version bump above.
    self.location.search,
    message.modelUrl,
    position.config.win_length,
    position.config.placement_radius,
    position.config.max_moves,
    position.to_move,
    position.moves_remaining,
    Number(message.sims || 0),
    Number(message.mActions || 0),
    message.autoForcing !== false,
    Number(strength.forcingDepth || 0),
    String(strength.forcingBudget || "0"),
    stones,
  ]);
}

async function analysisCacheGet(message, position) {
  const db = await openAnalysisCache();
  if (!db) return null;
  return new Promise(resolve => {
    const request = db.transaction(ANALYSIS_CACHE_STORE, "readonly")
      .objectStore(ANALYSIS_CACHE_STORE).get(analysisCacheKey(message, position));
    request.onsuccess = () => resolve(request.result?.result || null);
    request.onerror = () => resolve(null);
  });
}

async function analysisCachePut(message, position, result) {
  const db = await openAnalysisCache();
  if (!db) return;
  await new Promise(resolve => {
    const tx = db.transaction(ANALYSIS_CACHE_STORE, "readwrite");
    const store = tx.objectStore(ANALYSIS_CACHE_STORE);
    store.put({key: analysisCacheKey(message, position), saved: Date.now(), result});
    const count = store.count();
    count.onsuccess = () => {
      let excess = count.result - ANALYSIS_CACHE_MAX_ENTRIES;
      if (excess <= 0) return;
      const cursor = store.index("saved").openKeyCursor();
      cursor.onsuccess = () => {
        if (!cursor.result || excess-- <= 0) return;
        store.delete(cursor.result.primaryKey);
        cursor.result.continue();
      };
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
    tx.onabort = () => resolve();
  });
}

function hasWin(stones, player, winLength) {
  const occupied = new Set(stones.filter(s => s.player === player).map(s => `${s.q},${s.r}`));
  for (const stone of stones) {
    if (stone.player !== player) continue;
    for (const [dq, dr] of [[1, 0], [0, 1], [1, -1]]) {
      if (occupied.has(`${stone.q-dq},${stone.r-dr}`)) continue;
      let count = 1;
      while (occupied.has(`${stone.q+dq*count},${stone.r+dr*count}`)) count++;
      if (count >= winLength) return true;
    }
  }
  return false;
}

function positionAt(moves, end, config) {
  const stones = [];
  for (let i = 0; i <= end; i++) {
    const player = i === 0 ? 1 : (Math.floor((i - 1) / 2) % 2 === 0 ? 2 : 1);
    stones.push({q: moves[i][0], r: moves[i][1], player});
  }
  const placements = Math.max(0, end);
  const toMove = Math.floor(placements / 2) % 2 === 0 ? 2 : 1;
  const movesRemaining = placements % 2 === 0 ? 2 : 1;
  return {config, stones, to_move: toMove, moves_remaining: movesRemaining};
}

function terminalEntry(position) {
  const winner = hasWin(position.stones, 1, position.config.win_length) ? "P1"
    : hasWin(position.stones, 2, position.config.win_length) ? "P2" : null;
  const draw = position.stones.length - 1 >= position.config.max_moves;
  if (!winner && !draw) return null;
  // Match serving.analysis._current_player_at: terminal GameState does not
  // expose a meaningful next mover, so report the opponent of the last mover.
  const lastMover = position.stones[position.stones.length - 1].player;
  const current = lastMover === 1 ? "P2" : "P1";
  return {
    value: winner ? (winner === current ? 1 : -1) : 0,
    current_player: current, terminal: true, winner, legal: [],
    stones: position.stones.map(s => [[s.q, s.r], s.player === 1 ? "P1" : "P2"]),
  };
}

function analysisEntry(result, position) {
  const visits = result.visit_counts || [];
  const total = visits.reduce((sum, n) => sum + n, 0);
  const searchProbs = total ? visits.map(n => n / total) : result.probs;
  return {
    value: result.value,
    current_player: position.to_move === 1 ? "P1" : "P2",
    terminal: false, winner: null,
    legal: result.legal, probs: searchProbs,
    q_hat: result.q_hat, improved_policy: result.improved_policy,
    candidate_set: result.candidate_set,
    stones: position.stones.map(s => [[s.q, s.r], s.player === 1 ? "P1" : "P2"]),
  };
}

function solverPosition(api, position) {
  return new api.Position(
    position.config.win_length, position.config.placement_radius, position.config.max_moves,
    position.to_move === 1 ? api.Player.P1 : api.Player.P2,
    position.moves_remaining,
    new Int32Array(position.stones.flatMap(s => [s.q, s.r, s.player])),
  );
}

function forcingResult(api, outcome, attackerIsMover, position) {
  if (outcome.kind !== api.SolveKind.Win) return null;
  const pv = [], pvOwners = [];
  for (const turn of outcome.pv) {
    const owner = turn.player === api.Player.P1 ? "P1" : "P2";
    for (const cell of turn.cells) {
      pv.push([cell.q, cell.r]);
      pvOwners.push(owner);
      cell.free();
    }
    turn.free();
  }
  const mover = position.to_move === 1 ? "P1" : "P2";
  const winner = attackerIsMover ? mover : (mover === "P1" ? "P2" : "P1");
  return {
    winner, attacker_is_mover: attackerIsMover,
    first_move: pv[0] || null, depth: outcome.depth || null,
    pv, line_placements: pv.length, pv_len: pv.length, pv_owners: pvOwners,
    wide: true, defense: null,
  };
}

function takeCoord(coord) {
  if (!coord) return null;
  try { return [coord.q, coord.r]; }
  finally { try { coord.free(); } catch (_error) {} }
}

function takeDefense(outcome) {
  const takePairs = pairs => pairs.map(pair => {
    try { return [takeCoord(pair.first), takeCoord(pair.second)]; }
    finally { try { pair.free(); } catch (_error) {} }
  });
  return {
    killers: outcome.killers.map(takeCoord),
    pair_anchors: takePairs(outcome.pair_anchors),
    counter_threats: takePairs(outcome.counter_threats || []),
    tactical_pairs: takePairs(outcome.tactical_pairs || []),
    unresolved: (outcome.unresolved || []).map(takeCoord),
    best_delay: takeCoord(outcome.best_delay),
    wide: true,
  };
}

function solveOpponentDefense(api, position, depthCap, nodeBudget, fallbackForcing = null) {
  let solver, solverPos, limits, outcome, threat;
  try {
    solver = new api.StrixSolver();
    solverPos = solverPosition(api, position);
    limits = new api.SolverLimits(depthCap, nodeBudget, api.SolverEngineEnum.Idtt);
    outcome = solver.solve_defense_wide(solverPos, limits);
    if (outcome.kind === api.DefenseKind.BudgetExceeded)
      return {forcing: fallbackForcing, status: "budget"};
    if (outcome.kind === api.DefenseKind.NoThreat)
      return {forcing: null, status: "none"};

    threat = outcome.threat;
    const checked = forcingResult(api, threat, false, position);
    if (!checked) return {forcing: fallbackForcing, status: "budget"};
    // A win can remain proven when reconstruction of its example line runs
    // out of budget. Keep an existing line in that case, but attach the
    // authoritative defence classification from this pass.
    const forcing = checked.pv.length || !fallbackForcing
      ? checked : {...fallbackForcing, winner: checked.winner, attacker_is_mover: false};
    forcing.defense = takeDefense(outcome);
    forcing.defense_status = "checked";
    return {forcing, status: "checked"};
  } finally {
    try { threat?.free(); } catch (_error) {}
    try { outcome?.free(); } catch (_error) {}
    try { limits?.free(); } catch (_error) {}
    try { solverPos?.free(); } catch (_error) {}
    try { solver?.free(); } catch (_error) {}
  }
}

function solveForcing(api, position, depthCap = 10, nodeBudget = 20000n, withDefense = false) {
  let solver, solverPos, limits, outcome, defenseOutcome;
  try {
    solver = new api.StrixSolver();
    solverPos = solverPosition(api, position);
    limits = new api.SolverLimits(depthCap, nodeBudget, api.SolverEngineEnum.Idtt);
    outcome = solver.solve_wide(solverPos, limits);
    let result = forcingResult(api, outcome, true, position);
    outcome.free(); outcome = null;
    if (!result) {
      if (withDefense) {
        defenseOutcome = solver.solve_defense_wide(solverPos, limits);
        if (defenseOutcome.kind === api.DefenseKind.ThreatFound) {
          outcome = defenseOutcome.threat;
          result = forcingResult(api, outcome, false, position);
          if (result) result.defense = takeDefense(defenseOutcome);
        }
      } else {
        outcome = solver.solve_threat_wide(solverPos, limits);
        result = forcingResult(api, outcome, false, position);
      }
    }
    return result;
  } catch (_error) {
    return null;
  } finally {
    try { outcome?.free(); } catch (_error) {}
    try { defenseOutcome?.free(); } catch (_error) {}
    try { limits?.free(); } catch (_error) {}
    try { solverPos?.free(); } catch (_error) {}
    try { solver?.free(); } catch (_error) {}
  }
}

function forcingSettings(message) {
  const strength = message.strength || {};
  const depth = Number.isInteger(strength.forcingDepth) ? strength.forcingDepth : 10;
  const rawBudget = String(strength.forcingBudget || "20000");
  const budget = message.autoForcing === false ? 0n
    : /^\d+$/.test(rawBudget) ? BigInt(rawBudget) : 20000n;
  return {depth, budget};
}

const QUALITY_ICON = {best: "★", winning: "◆", good: "✓", mistake: "?", blunder: "✗", forced: "◇"};
const QUALITY_COLOR = {best: "#f2c14e", winning: "#79cf9a", good: "#79cf9a",
  mistake: "#e0a23a", blunder: "#e25c5c", forced: "#aeb8b1"};

function sameMove(a, b) {
  return Boolean(a && b && a[0] === b[0] && a[1] === b[1]);
}

function bestMoveByPolicy(entry) {
  if (!entry?.improved_policy || !entry.legal) return null;
  let best = -1, probability = -Infinity;
  for (let i = 0; i < entry.improved_policy.length; i++) {
    if (entry.candidate_set && !entry.candidate_set[i]) continue;
    if (entry.improved_policy[i] > probability) {
      probability = entry.improved_policy[i];
      best = i;
    }
  }
  return best >= 0 ? entry.legal[best] : null;
}

function qOfMove(entry, move) {
  if (!entry?.q_hat || !entry.legal || !move) return null;
  const index = entry.legal.findIndex(candidate => sameMove(candidate, move));
  const value = index >= 0 ? Number(entry.q_hat[index]) : NaN;
  return Number.isFinite(value) ? value : null;
}

function forcingIsCertain(forcing) {
  if (!forcing) return false;
  if (forcing.attacker_is_mover) return true;
  const defense = forcing.defense;
  return Boolean(defense && !defense.killers?.length && !defense.pair_anchors?.length && defense.best_delay);
}

async function completeNeededDefense(api, message, settings, position, entry, winner) {
  let forcing = entry?.forcing;
  if (!forcing || forcing.winner !== winner || forcing.attacker_is_mover ||
      forcing.defense || forcing.defense_status || settings.budget <= 0n) return forcing;
  const checked = solveOpponentDefense(api, position, settings.depth, settings.budget, forcing);
  entry.forcing = checked.forcing;
  await analysisCachePut(message, position, entry);
  return entry.forcing;
}

// Classify a turn as soon as its final prefix is available. This consumes the
// same trajectory entries the game analysis is already producing, so ratings
// are ready when the worker returns rather than requiring a second UI phase.
async function classifyCompletedTurn(api, message, settings, positions, trajectory, end, getBot) {
  const terminal = Boolean(trajectory[end]?.terminal);
  if (!(terminal || (end >= 2 && end % 2 === 0))) return null;
  const startIndex = end - (terminal && end % 2 === 1 ? 1 : 2);
  const start = trajectory[startIndex];
  const finish = trajectory[end];
  if (!start || !finish) return null;
  const mover = start.current_player;
  const opponent = mover === "P1" ? "P2" : "P1";
  if (!opponent) return null;

  const played = message.moves.slice(startIndex + 1, end + 1).map(move => move.slice(0, 2));
  if (!played.length) return null;
  const first = bestMoveByPolicy(start);
  if (!first) return null;
  const engineLine = [first];
  let engineEndQ = qOfMove(start, first);
  if (played.length >= 2) {
    let afterFirst;
    if (sameMove(first, played[0])) {
      afterFirst = trajectory[startIndex + 1];
    } else {
      const offMoves = [...message.moves.slice(0, startIndex + 1), first];
      const offPosition = positionAt(offMoves, offMoves.length - 1, message.config);
      afterFirst = await analysisCacheGet(message, offPosition);
      if (!afterFirst) {
        const bot = await getBot();
        afterFirst = analysisEntry(JSON.parse(bot.best_move(
          JSON.stringify(offPosition), message.sims, message.mActions, 0n)), offPosition);
        afterFirst.forcing = settings.budget > 0n
          ? solveForcing(api, offPosition, settings.depth, settings.budget) : null;
        await analysisCachePut(message, offPosition, afterFirst);
      }
    }
    const second = bestMoveByPolicy(afterFirst);
    if (!second) return null;
    engineLine.push(second);
    engineEndQ = qOfMove(afterFirst, second);
  }

  const matched = engineLine.length === played.length &&
    played.every(move => engineLine.some(engineMove => sameMove(move, engineMove)));
  let playerEndQ = qOfMove(trajectory[end - 1] || start, played[played.length - 1]);
  let loss = !matched && engineEndQ != null && playerEndQ != null
    ? Math.max(0, engineEndQ - playerEndQ) : 0;
  let label = matched ? "best" : loss >= 0.40 ? "blunder" : loss >= 0.15 ? "mistake" : "good";

  let startForcing = start.forcing;
  if (startForcing?.winner === opponent)
    startForcing = await completeNeededDefense(
      api, message, settings, positions[startIndex], start, opponent);
  const forcedBeforeTurn = Boolean(forcingIsCertain(startForcing) && startForcing.winner === opponent);
  const forcedLoss = Boolean(
    (finish.terminal && finish.winner === opponent) ||
    (forcingIsCertain(finish.forcing) && finish.forcing.winner === opponent));
  if (forcedLoss) {
    if (forcedBeforeTurn) {
      label = "forced";
      loss = 0;
    } else {
      playerEndQ = -1;
      loss = engineEndQ == null ? 1 : Math.max(0, engineEndQ - playerEndQ);
      label = "blunder";
    }
  }

  let endForcing = finish.forcing;
  if (endForcing?.winner === mover)
    endForcing = await completeNeededDefense(api, message, settings, positions[end], finish, mover);
  const provenWin = Boolean(
    (finish.terminal && finish.winner === mover) ||
    (forcingIsCertain(endForcing) && endForcing.winner === mover));
  const winBeforeTurn = Boolean(forcingIsCertain(start.forcing) && start.forcing.winner === mover);
  if (provenWin && !matched) {
    label = "winning";
    loss = 0;
  }

  return {
    label, icon: QUALITY_ICON[label], color: QUALITY_COLOR[label], matched,
    engine_pair: engineLine, played_pair: played, loss,
    player_end_q: playerEndQ, engine_end_q: engineEndQ,
    forced_loss: forcedLoss, forced_before_turn: forcedBeforeTurn,
    proven_win: provenWin, win_before_turn: winBeforeTurn,
    turn_start_depth: startIndex,
  };
}

function annotateMissedWins(api, moves, positions, trajectory) {
  for (let i = 0; i + 1 < trajectory.length; i++) {
    const forcing = trajectory[i].forcing;
    if (!forcing?.attacker_is_mover || !moves[i + 1]) continue;
    if (forcing.first_move?.[0] === moves[i + 1][0] && forcing.first_move?.[1] === moves[i + 1][1]) continue;
    const mover = forcing.winner;
    let exempt = null;
    for (let j = i + 1; j < trajectory.length; j++) {
      const entry = trajectory[j];
      if (entry.terminal) {
        exempt = entry.winner === mover;
        break;
      }
      if (entry.current_player === mover) {
        let kept = entry.forcing;
        exempt = Boolean(kept?.attacker_is_mover && kept.winner === mover);
        if (!exempt) {
          kept = solveForcing(api, positions[j], 12, 250000n);
          if (kept?.attacker_is_mover && kept.winner === mover) {
            entry.forcing = kept;
            exempt = true;
          }
        }
        break;
      }
    }
    if (exempt !== false) continue;
    trajectory[i + 1].missed_win = {
      by: mover, at_prefix: i, first_move: forcing.first_move,
      depth: forcing.depth, pv: forcing.pv,
      line_placements: forcing.line_placements, pv_len: forcing.pv_len,
      pv_owners: forcing.pv_owners,
    };
  }
}

self.onmessage = async event => {
  const msg = event.data || {};
  const requestId = msg.requestId;
  let stage = "loading model";
  try {
    if (msg.type === "bestMove") {
      const bot = await loadBot(msg.modelUrl);
      stage = "local bot search";
      // Forcing/defense layer before MCTS, mirroring the server bot's
      // `_forcing_move`. Never breaks play: any failure falls back to the
      // plain search below.
      let forced = null;
      try {
        const api = await loadApi();
        const solver = new api.StrixSolver();
        try {
          forced = selectForcingMove({
            api, solver, bot, position: msg.position, difficulty: msg.difficulty,
          });
        } finally {
          try { solver.free(); } catch (_error) {}
        }
      } catch (_error) {
        forced = null;
      }
      if (forced) {
        const result = {
          move: {q: forced.q, r: forced.r},
          value: null, policy: [], legal: [], probs: [],
          improved_policy: [], visit_counts: [], q_hat: [],
          candidate_set: [], forcing_source: forced.source,
        };
        self.postMessage({type: "bestMove", requestId, result});
        return;
      }
      const result = JSON.parse(bot.best_move(JSON.stringify(msg.position), msg.sims, msg.mActions, 0n));
      self.postMessage({type: "bestMove", requestId, result});
      return;
    }
    if (msg.type === "analyzeDefense") {
      const api = await loadApi();
      const settings = forcingSettings(msg);
      stage = "local defence check";
      const result = settings.budget > 0n
        ? solveOpponentDefense(
          api, msg.position, settings.depth, settings.budget, msg.forcing || null)
        : {forcing: msg.forcing || null, status: "disabled"};
      const cached = await analysisCacheGet(msg, msg.position);
      if (cached) {
        cached.forcing = result.forcing;
        await analysisCachePut(msg, msg.position, cached);
      }
      self.postMessage({type: "defense", requestId, result});
      return;
    }
    if (msg.type === "analyzePosition") {
      stage = "local analysis cache";
      const cached = await analysisCacheGet(msg, msg.position);
      if (cached) {
        if (msg.previewValue) self.postMessage({type: "estimate", requestId, result: {
          value: cached.value,
          current_player: cached.current_player,
        }});
        self.postMessage({type: "position", requestId, result: cached});
        return;
      }
      const bot = await loadBot(msg.modelUrl);
      const api = await loadApi();
      stage = "local position analysis";
      let result = null;
      let preview = null;
      if (msg.previewValue) {
        stage = "local value estimate";
        result = JSON.parse(bot.best_move(JSON.stringify(msg.position), 0, msg.mActions, 0n));
        preview = {
          value: result.value,
          current_player: msg.position.to_move === 1 ? "P1" : "P2",
        };
        self.postMessage({type: "estimate", requestId, result: {
          ...preview,
        }});
      }
      const settings = forcingSettings(msg);
      stage = "local forced-win check";
      const forcing = settings.budget > 0n
        ? solveForcing(api, msg.position, settings.depth, settings.budget, true) : null;
      if (preview) self.postMessage({type: "estimate", requestId, result: {
        ...preview, forcing,
      }});
      if (!result || msg.sims > 0) {
        stage = "local position search";
        result = JSON.parse(bot.best_move(JSON.stringify(msg.position), msg.sims, msg.mActions, 0n));
      }
      const entry = analysisEntry(result, msg.position);
      entry.forcing = forcing;
      await analysisCachePut(msg, msg.position, entry);
      self.postMessage({type: "position", requestId, result: entry});
      return;
    }
    if (msg.type !== "analyzeGame") return;
    const api = await loadApi();
    let bot = null;
    stage = "local trajectory analysis";
    const trajectory = [];
    const positions = [];
    const settings = forcingSettings(msg);
    let cacheHits = 0;
    let cacheMisses = 0;
    const getBot = async () => {
      bot = bot || await loadBot(msg.modelUrl);
      return bot;
    };
    for (let i = 0; i < msg.moves.length; i++) {
      const position = positionAt(msg.moves, i, msg.config);
      positions.push(position);
      const terminal = terminalEntry(position);
      stage = `local cache prefix ${i + 1}/${msg.moves.length}`;
      let entry = terminal || await analysisCacheGet(msg, position);
      if (!terminal) entry ? cacheHits++ : cacheMisses++;
      if (!entry) {
        bot = await getBot();
        stage = `local MCTS prefix ${i + 1}/${msg.moves.length}`;
        entry = analysisEntry(
          JSON.parse(bot.best_move(JSON.stringify(position), msg.sims, msg.mActions, 0n)), position);
        stage = `local forcing prefix ${i + 1}/${msg.moves.length}`;
        entry.forcing = settings.budget > 0n
          ? solveForcing(api, position, settings.depth, settings.budget) : null;
        await analysisCachePut(msg, position, entry);
      }
      trajectory.push(entry);
      const quality = await classifyCompletedTurn(
        api, msg, settings, positions, trajectory, i, getBot);
      if (quality) entry.quality = quality;
      self.postMessage({type: "progress", requestId, done: i + 1, total: msg.moves.length,
        cacheHits, cacheMisses});
      if (terminal) break;
    }
    annotateMissedWins(api, msg.moves, positions, trajectory);
    const boundaryIndices = trajectory.map((entry, i) =>
      i === 0 || i === trajectory.length - 1 || entry.current_player !== trajectory[i - 1].current_player ? i : -1
    ).filter(i => i >= 0);
    self.postMessage({type: "game", requestId, result: {
      trajectory, boundary_indices: [...new Set(boundaryIndices)],
      evaluated_prefixes: trajectory.length, completed_prefixes: trajectory.length,
      mcts_sims: msg.sims, cancelled: false,
    }});
  } catch (error) {
    self.postMessage({type: "error", requestId,
      error: `${stage}: ${error?.message || String(error)}`});
  }
};
