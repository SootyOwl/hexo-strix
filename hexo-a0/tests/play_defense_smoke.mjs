// Node smoke for the local-play forcing/defense layer (play-defense.js).
// Run from hexo-rs/:  node ../hexo-a0/tests/play_defense_smoke.mjs
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {dirname, join} from "node:path";
import {createRequire} from "node:module";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const api = require(join(here, "../../hexo-rs/hexo-wasm/pkg-node/hexo_wasm.js"));

const tinyWeights = new Uint8Array(readFileSync(join(
  here, "../../hexo-rs/hexo-infer/tests/fixtures/tiny.safetensors")));
const bot = new api.StrixBot(tinyWeights);
const solver = new api.StrixSolver();

const pass = [];
const ok = (name, cond) => { if (!cond) throw new Error(`FAIL: ${name}`); pass.push(name); };

const position = (toMove, remaining, stones) => ({
  config: {win_length: 6, placement_radius: 8, max_moves: 400},
  stones: stones.map(([q, r, player]) => ({q, r, player})),
  to_move: toMove, moves_remaining: remaining,
});

const replay = JSON.parse(readFileSync(join(
  here, "../../scripts/fixtures/forcing_puzzles/qietby7_17_line.json"), "utf8"));
const qietStones = replay.moves.slice(0, 31).map(([q, r, player]) =>
  [q, r, player === "P1" ? 1 : 2]);

const QIET_PAIRS = [
  [[-2, 0], [-1, 0]], [[-1, 0], [2, 0]], [[2, 0], [3, 0]],
];

import {selectForcingMove, resetPlayDefenseCaches} from "../src/hexo_a0/serving/static/play-defense.js";

// 1. Proven counter-threat: the layer must answer P2's hypothetical threat
// with one of the three verified pairs (source pair or tactical cover).
resetPlayDefenseCaches();
const qietPos = position(1, 2, qietStones);
const first = selectForcingMove({api, solver, bot, position: qietPos, difficulty: "deep"});
ok("counter-threat answered", Boolean(first));
ok("source is a pair family", ["pair", "pair-unverified", "tactical-pair"].includes(first.source));
const played = [first.q, first.r];
const pair = QIET_PAIRS.find(cells => cells.some(([q, r]) => q === played[0] && r === played[1]));
ok("move belongs to a verified pair", Boolean(pair));

// 2. Partner cache: the second placement completes the same atomic pair.
const afterFirst = {
  config: qietPos.config,
  stones: qietPos.stones.concat([{q: played[0], r: played[1], player: 1}]),
  to_move: 1, moves_remaining: 1,
};
const second = selectForcingMove({api, solver, bot, position: afterFirst, difficulty: "deep"});
ok("partner cached and played", second.source === "pair-partner");
ok("partner completes the pair", pair.some(([q, r]) => q === second.q && r === second.r));

// 3. Own-win: the tiny position has a one-turn forced win.
resetPlayDefenseCaches();
const tinyPos = position(1, 2, [[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [5, 5, 2]]);
const win = selectForcingMove({api, solver, bot, position: tinyPos, difficulty: "standard"});
ok("own-win sighted", win.source === "win");
ok("win move completes the line", [[-2, 0], [-1, 0], [4, 0]].some(([q, r]) => q === win.q && r === win.r));

// 4. Own-win PV cache: the second placement is the cached follow-up.
const afterWinCell = {
  config: tinyPos.config,
  stones: tinyPos.stones.concat([{q: win.q, r: win.r, player: 1}]),
  to_move: 1, moves_remaining: 1,
};
const winPv = selectForcingMove({api, solver, bot, position: afterWinCell, difficulty: "standard"});
ok("own-win pv cached", winPv.source === "win-pv");

// 5. Honesty: a starved threat check must NOT become a move decision.
resetPlayDefenseCaches();
const starved = selectForcingMove({api, solver, bot, position: qietPos,
  difficulty: "deep", overrides: {defBudget: 1n}});
ok("starved defense defers to MCTS (never no_threat)", starved === null);

// 6. No threat: a quiet position defers to MCTS.
resetPlayDefenseCaches();
const quiet = selectForcingMove({api, solver, bot,
  position: position(1, 2, [[0, 0, 1], [10, 10, 2]]), difficulty: "standard"});
ok("quiet position defers to MCTS", quiet === null);

console.log(`OK play-defense: ${pass.length} assertions passed`);
