// Local-play forcing/defense layer for the browser bot.
//
// Mirrors the server bot's forcing pipeline (hexo_a0/serving/game.py:
// `_forcing_move` + `_defensive_move`) so local and server play make the same
// kinds of decisions at the same priority:
//
//   1. cached pair partner      (the atomic second cell of a verified pair)
//   2. cached own-win follow-up (the second cell of a proven own-win turn)
//   3. own-win solve            (Idtt, LIVE budget)
//   4. defense solve            (Idtt, DEF budget; tight generator — the same
//                                 one server live play uses):
//                                  killers       (own-win-preferred, verified)
//                                  pair anchors  (unordered, verified)
//                                  tactical pairs (exact immediate covers)
//                                  unresolved    (budget-starved replies)
//                                  best delay
//   5. null → the caller falls back to Gumbel MCTS, exactly like the server.
//
// Honesty contract (same as the wasm/FFI surfaces): `BudgetExceeded` from the
// threat check falls through to MCTS — it is never read as "no threat", and
// unverified candidates are never silently discarded in favour of a reply that
// is proved to lose on the opponent's next turn.
//
// Budgets/deadlines mirror the server constants (LIVE 50k, DEF 20k, VERIFY
// 250k, flat 3 s deadline with 10 s on deep). wasm has no monotonic clock, so
// the deadline is enforced HERE, between probes: on expiry the server behavior
// is kept — play the policy-best remaining candidate UNVERIFIED and say so.
// Each helper frees its wasm objects; a leaked allocation per move would
// accumulate over a session.

export const FORCING_DEPTH_BY_DIFFICULTY = {quick: 2, standard: 4, strong: 8, deep: 25};

const LOCAL_LIVE_NODE_BUDGET = 50000n;
const LOCAL_DEF_NODE_BUDGET = 20000n;
const LOCAL_VERIFY_NODE_BUDGET = 250000n;
const PLAY_CACHE_MAX_ENTRIES = 64;

// Cross-call caches, keyed by the exact position JSON the play page sends.
// The server keeps these on the GameRecord; local play is stateless between
// worker calls, so the previous placement's caches live here instead.
const playPartnerCache = new Map();
const playOwnWinNextCache = new Map();

export function resetPlayDefenseCaches() {
  playPartnerCache.clear();
  playOwnWinNextCache.clear();
}

function cachePut(map, key, value) {
  if (map.size >= PLAY_CACHE_MAX_ENTRIES) {
    map.delete(map.keys().next().value);
  }
  map.set(key, value);
}

function freeQuietly(value) {
  if (!value) return;
  try { value.free(); }
  catch (_error) {}
}

function toMove(pos) {
  return pos.to_move === 1 ? "P1" : "P2";
}

function opponentName(pos) {
  return toMove(pos) === "P1" ? "P2" : "P1";
}

// The position after `cells` are placed by the current mover, keeping the
// turn mechanics: 2→1 keeps the mover, reaching 0 passes with a fresh turn.
function appliedPosition(pos, cells) {
  let stones = pos.stones.concat(cells.map(([q, r]) => ({q, r, player: pos.to_move})));
  let movesRemaining = pos.moves_remaining - cells.length;
  let toMove = pos.to_move;
  if (movesRemaining <= 0) {
    movesRemaining = 2;
    toMove = toMove === 1 ? 2 : 1;
  }
  return {...pos, stones, to_move: toMove, moves_remaining: movesRemaining};
}

// The server's `_verified_defense`: after the bot's placements, the OPPONENT
// gets a fresh turn — a deliberately conservative check.
function turnsEndedPosition(pos, cells) {
  return {
    ...pos,
    stones: pos.stones.concat(cells.map(([q, r]) => ({q, r, player: pos.to_move}))),
    to_move: pos.to_move === 1 ? 2 : 1,
    moves_remaining: 2,
  };
}

function buildPosition(api, pos) {
  const stonesFlat = [];
  for (const stone of pos.stones) stonesFlat.push(stone.q, stone.r, stone.player);
  return new api.Position(
    pos.config.win_length, pos.config.placement_radius, pos.config.max_moves,
    pos.to_move === 1 ? api.Player.P1 : api.Player.P2,
    pos.moves_remaining, new Int32Array(stonesFlat),
  );
}

function solveLimits(api, depth, budget) {
  const limits = new api.SolverLimits(depth, budget, api.SolverEngineEnum.Idtt);
  return limits;
}

// Raw-prior argmax restricted to `candidates` (server: `_policy_argmax`).
// `evaluate` returns the policy sorted descending, so the first member of the
// restricted set is the choice. Returns null when the model cannot rank them.
function policyPick(bot, position, candidates) {
  try {
    const evaluation = JSON.parse(bot.evaluate(JSON.stringify(position)));
    const allowed = new Set(candidates.map(([q, r]) => `${q},${r}`));
    for (const entry of evaluation.policy || []) {
      if (allowed.has(`${entry.q},${entry.r}`)) {
        return {q: entry.q, r: entry.r, evaluation};
      }
    }
  } catch (_error) {}
  return null;
}

function readTurnCells(pv) {
  // Flatten the chunked PV into [q, r] attacker placements, freeing as we go.
  const flat = [];
  try {
    for (const turn of pv) {
      for (const cell of turn.cells) {
        flat.push([cell.q, cell.r]);
        try { cell.free(); } catch (_error) {}
      }
      try { turn.free(); } catch (_error) {}
    }
  } catch (_error) {}
  pv.length = 0;
  return flat;
}

// Server `_own_win_after`: does this placement keep or create a forcing win?
function ownWinAfter(api, solver, pos, cell, depth, liveBudget) {
  let probePos, limits;
  try {
    const left = pos.moves_remaining - 1;
    if (left > 0) {
      probePos = buildPosition(api, appliedPosition(pos, [cell]));
    } else {
      // Last placement of the turn: the follow-up win only exists as a
      // threat the opponent must answer (flipped fresh-turn solve).
      probePos = buildPosition(api, turnsEndedPosition(pos, [cell]));
      limits = solveLimits(api, depth, liveBudget);
      const outcome = solver.solve_threat(probePos, limits);
      const win = outcome.kind === api.SolveKind.Win;
      freeQuietly(outcome);
      return win;
    }
    limits = solveLimits(api, depth, liveBudget);
    const outcome = solver.solve(probePos, limits);
    const win = outcome.kind === api.SolveKind.Win;
    freeQuietly(outcome);
    return win;
  } catch (_error) {
    return false;
  } finally {
    freeQuietly(limits);
    freeQuietly(probePos);
  }
}

// Server `_verified_defense`: after playing `cells`, the opponent's threat
// must be GONE at the escalated budget (fresh opponent turn).
function verifiedDefense(api, solver, pos, cells, depth, verifyBudget) {
  let trialPos, limits, outcome;
  try {
    trialPos = buildPosition(api, turnsEndedPosition(pos, cells));
    limits = solveLimits(api, depth, verifyBudget);
    outcome = solver.solve(trialPos, limits);
    return outcome.kind === api.SolveKind.No;
  } catch (_error) {
    return false;
  } finally {
    freeQuietly(outcome);
    freeQuietly(limits);
    freeQuietly(trialPos);
  }
}

export function selectForcingMove({api, solver, bot, position, difficulty, overrides = {}}) {
  const depth = overrides.depth
    ?? FORCING_DEPTH_BY_DIFFICULTY[difficulty ?? "standard"] ?? 4;
  const settings = {
    depth,
    liveBudget: overrides.liveBudget ?? LOCAL_LIVE_NODE_BUDGET,
    defBudget: overrides.defBudget ?? LOCAL_DEF_NODE_BUDGET,
    verifyBudget: overrides.verifyBudget ?? LOCAL_VERIFY_NODE_BUDGET,
    deadlineMs: overrides.deadlineMs ?? (difficulty === "deep" ? 10000 : 3000),
  };
  const deadline = Date.now() + settings.deadlineMs;
  const key = JSON.stringify(position);

  // 1. Cached pair partner: the verified pair is one atomic choice, and its
  // second cell preempts everything (a fresh own-win suggestion must never
  // abandon the cover halfway through the turn).
  const partner = playPartnerCache.get(key);
  if (partner) {
    playPartnerCache.delete(key);
    return {q: partner[0], r: partner[1], source: "pair-partner"};
  }

  // 2. Cached own-win follow-up: the second cell of a proven own-win turn.
  const ownWinNext = playOwnWinNextCache.get(key);
  if (ownWinNext) {
    playOwnWinNextCache.delete(key);
    return {q: ownWinNext[0], r: ownWinNext[1], source: "win-pv"};
  }

  let solverPos, limits, ownOutcome, defenseOutcome;
  try {
    solverPos = buildPosition(api, position);

    // 3. Own-win solve at the live budget.
    limits = solveLimits(api, settings.depth, settings.liveBudget);
    ownOutcome = solver.solve(solverPos, limits);
    if (ownOutcome.kind === api.SolveKind.Win) {
      const flat = readTurnCells(ownOutcome.pv);
      if (flat.length > 0) {
        const [q, r] = flat[0];
        if (flat.length > 1) {
          const afterFirst = appliedPosition(position, [flat[0]]);
          // Only cache when the next cell is still THIS turn (the first
          // chunk holds `moves_remaining` cells; the rest is the opponent's).
          if (position.moves_remaining >= 2) {
            cachePut(playOwnWinNextCache, JSON.stringify(afterFirst), flat[1]);
          }
        }
        return {q, r, source: "win"};
      }
    }
    freeQuietly(ownOutcome);
    ownOutcome = null;

    // 4. Defense at the DEF budget, tight generator (server live-play parity).
    freeQuietly(limits);
    limits = solveLimits(api, settings.depth, settings.defBudget);
    defenseOutcome = solver.solve_defense(solverPos, limits);
    if (defenseOutcome.kind === api.DefenseKind.BudgetExceeded) return null;
    if (defenseOutcome.kind === api.DefenseKind.NoThreat) return null;
    // Proven opponent threat. Monotonicity: falling through to MCTS on a
    // proven threat can never be safe, so every category below acts.

    const asCoords = (wrappers) => wrappers.map((c) => {
      const coord = [c.q, c.r];
      try { c.free(); } catch (_error) {}
      return coord;
    });
    const killers = asCoords(defenseOutcome.killers);
    const pairs = (defenseOutcome.pair_anchors || []).map((pair) => {
      const cells = [[pair.first.q, pair.first.r], [pair.second.q, pair.second.r]];
      try { pair.free(); } catch (_error) {}
      return cells;
    });
    const tacticalPairs = (defenseOutcome.tactical_pairs || []).map((pair) => {
      const cells = [[pair.first.q, pair.first.r], [pair.second.q, pair.second.r]];
      try { pair.free(); } catch (_error) {}
      return cells;
    });
    const unresolved = asCoords(defenseOutcome.unresolved || []);
    const bestDelay = defenseOutcome.best_delay
      ? [defenseOutcome.best_delay.q, defenseOutcome.best_delay.r] : null;

    // 4a. Killers: prefer placements that keep the bot's own forcing win
    // (server `_own_win_after`), then verify the pick at the escalated
    // budget, dropping budget-artifact "refutations" like the server does.
    if (killers.length > 0) {
      let remaining = killers;
      let winning = [];
      if (killers.length > 1 && Date.now() < deadline) {
        winning = killers.filter((cell) =>
          ownWinAfter(api, solver, position, cell, settings.depth, settings.liveBudget));
      }
      while (remaining.length > 0) {
        const pool = winning.length > 0 ? winning : remaining;
        const pick = policyPick(bot, position, pool)
          || {q: pool[0][0], r: pool[0][1]};
        if (Date.now() >= deadline) {
          return {q: pick.q, r: pick.r, source: "killer-unverified"};
        }
        if (verifiedDefense(api, solver, position, [[pick.q, pick.r]],
            settings.depth, settings.verifyBudget)) {
          return {q: pick.q, r: pick.r, source: "killer"};
        }
        remaining = remaining.filter(([q, r]) => q !== pick.q || r !== pick.r);
        winning = winning.filter(([q, r]) => q !== pick.q || r !== pick.r);
      }
      // Every killer was a budget artifact; a pair may still hold.
    }

    // 4b. Pair anchors: unordered within the turn (Hexo turns have no move
    // order). Policy picks either cell; the other is cached as the partner.
    while (pairs.length > 0) {
      const pool = pairs.flat();
      const pick = policyPick(bot, position, pool)
        || {q: pool[0][0], r: pool[0][1]};
      const pair = pairs.find((cells) =>
        cells.some(([q, r]) => q === pick.q && r === pick.r));
      const partnerCell = pair.find(([q, r]) => q !== pick.q || r !== pick.r);
      const partnerKey = JSON.stringify(appliedPosition(position, [[pick.q, pick.r]]));
      if (Date.now() >= deadline) {
        cachePut(playPartnerCache, partnerKey, partnerCell);
        return {q: pick.q, r: pick.r, source: "pair-unverified"};
      }
      if (verifiedDefense(api, solver, position, pair,
          settings.depth, settings.verifyBudget)) {
        cachePut(playPartnerCache, partnerKey, partnerCell);
        return {q: pick.q, r: pick.r, source: "pair"};
      }
      pairs.splice(pairs.indexOf(pair), 1);
    }

    // 4c. Tactical pairs: exact kernel covers of the immediate attack whose
    // deeper status is unresolved. Stop the current line; do not claim more.
    if (tacticalPairs.length > 0) {
      const pool = tacticalPairs.flat();
      const pick = policyPick(bot, position, pool)
        || {q: pool[0][0], r: pool[0][1]};
      const pair = tacticalPairs.find((cells) =>
        cells.some(([q, r]) => q === pick.q && r === pick.r));
      const partnerCell = pair.find(([q, r]) => q !== pick.q || r !== pick.r);
      cachePut(playPartnerCache,
        JSON.stringify(appliedPosition(position, [[pick.q, pick.r]])), partnerCell);
      return {q: pick.q, r: pick.r, source: "tactical-pair"};
    }

    // 4d. Unresolved tactical replies: only on the turn's last placement, and
    // never replaced by a proved next-turn loss.
    if (position.moves_remaining === 1 && unresolved.length > 0) {
      const pick = policyPick(bot, position, unresolved)
        || {q: unresolved[0][0], r: unresolved[0][1]};
      return {q: pick.q, r: pick.r, source: "tactical-unresolved"};
    }

    // 4e. Proved delay: nothing stops the win; take the longest ride.
    if (bestDelay) {
      return {q: bestDelay[0], r: bestDelay[1], source: "delay"};
    }
    return null;
  } finally {
    freeQuietly(defenseOutcome);
    freeQuietly(ownOutcome);
    freeQuietly(limits);
    freeQuietly(solverPos);
  }
}
