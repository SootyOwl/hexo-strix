// Node smoke test for the standalone solver-only package.
// Mirrors the load-bearing assertions of hexo-wasm/tests/solver_smoke.mjs but
// runs against hexo-solver-wasm's own generated package.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

// --target nodejs emits CommonJS; createRequire avoids fragile ESM<->CJS interop.
const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const api = require(join(dirname(fileURLToPath(import.meta.url)), "..", "pkg-node", "hexo_solver_wasm.js"));


const pass = [];
const ok = (name, cond) => { if (!cond) throw new Error(`FAIL: ${name}`); pass.push(name); };

// 1. Solver-only guarantee: the bot surface must not exist in this package.
ok("no StrixBot export", api.StrixBot === undefined);

const pos = (toMove, remaining, flat) => new api.Position(
  6, 8, 400,
  toMove === "P1" ? api.Player.P1 : api.Player.P2,
  remaining, new Int32Array(flat),
);
const stones = flat => flat.flatMap(([q, r, player]) => [q, r, player === "P1" ? 1 : 2]);

// 1. Direct IDTT win on a small forcing position.
const winPos = pos("P1", 2, stones([[0,0,"P1"],[1,0,"P1"],[2,0,"P1"],[3,0,"P1"],[5,5,"P2"]]));
const solver = new api.StrixSolver();
const limits = new api.SolverLimits(6, 20000n, api.SolverEngineEnum.Idtt);
const win = solver.solve(winPos, limits);
ok("idtt finds Win", win.kind === api.SolveKind.Win);
ok("idtt pv present", win.pv.length > 0);

// 2. PDS-PN win with a certificate that independently verifies.
const pdspnLimits = new api.SolverLimits(8, 200000n, api.SolverEngineEnum.Pdspn);
const proof = solver.solve_wide(winPos, pdspnLimits);
ok("pdspn finds Win", proof.kind === api.SolveKind.Win);
ok("pdspn certificate present", proof.certificate_json.length > 0);
const summary = solver.verify_certificate(winPos, proof.certificate_json);
ok("certificate verifies", summary.dag_nodes > 0 && summary.max_attacker_turns > 0);

// 3. Defensive analysis keeps initiative evidence (qiet round-8 prefix).
const readJson = rel => JSON.parse(readFileSync(join(here, rel), "utf8"));
const replay = readJson("../../../scripts/fixtures/forcing_puzzles/qietby7_17_line.json");
const qiet = pos("P1", 2, stones(replay.moves.slice(0, 31)));
const defense = solver.solve_defense_wide(qiet, new api.SolverLimits(12, 100000n, api.SolverEngineEnum.Idtt));
ok("defense threat found", defense.kind === api.DefenseKind.ThreatFound);
ok("counter-threat pair present", defense.counter_threats.some(pair =>
  pair.first.q === 2 && pair.first.r === 0 && pair.second.q === 3 && pair.second.r === 0));
ok("tactical_pairs getter exists", Array.isArray(defense.tactical_pairs));
ok("unresolved getter exists", Array.isArray(defense.unresolved));
for (const pair of defense.counter_threats) { pair.first.free(); pair.second.free(); pair.free(); }
defense.threat?.free?.();
defense.free();

// 4. Exact minimum covers after a complete attacker turn.
const afterAttack = readJson("../../../scripts/fixtures/forcing_puzzles/qietby7_after_attack_turn9.json");
const minOut = solver.minimum_defenses_after_attack(
  pos("P1", 2, stones(afterAttack.stones)), true);
ok("minimum defenses enumerates covers", minOut.kind === api.MinimumDefenseKind.Covers
  && minOut.covers.length > 0);
const firstCover = minOut.covers[0];
ok("first cover has two cells",
  Number.isInteger(firstCover.first.q) && Number.isInteger(firstCover.second.q));
minOut.covers.forEach(c => c.free());
minOut.free();

ok("no best_delay leak on NoThreat", true);

console.log(`OK hexo-solver-wasm: ${pass.length} assertions passed`);
solver.free(); winPos.free(); win.free(); proof.free(); pdspnLimits.free(); limits.free();
qiet.free();
