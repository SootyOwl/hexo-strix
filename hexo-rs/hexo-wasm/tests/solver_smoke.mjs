// End-to-end smoke test for the StrixSolver wasm export (no model weights).
// Run against the nodejs pkg: `just wasm-test` builds pkg-node, then
//   node hexo-wasm/tests/solver_smoke.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import assert from "node:assert/strict";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const {
  StrixSolver,
  Position,
  SolverLimits,
  Player,
  SolverEngineEnum,
} = require(join(here, "..", "pkg-node", "hexo_wasm.js"));

let pass = 0;
const ok = (name, cond) => { if (!cond) throw new Error(`FAIL: ${name}`); pass++; };

// stones_flat = [q, r, player, q, r, player, ...] (player: 1=P1, 2=P2)
// P1 has (0,0),(1,0),(2,0) — 3-in-a-row, win_length 4 → forced win. P2 far away.
const winStones = [0,0,1, 1,0,1, 2,0,1, 5,5,2];
const winPos = new Position(4, 8, 300, Player.P1, 2, winStones);

const solver = new StrixSolver();

// 1. idtt finds the win.
const limits = new SolverLimits(6, 20000n, SolverEngineEnum.Idtt);
const win = solver.solve(winPos, limits);
ok("idtt win kind", win.kind === 0 /* Win */);
ok("idtt win depth >= 1", win.depth >= 1);
ok("idtt win pv non-empty", win.pv.length > 0);
ok("idtt first turn is attacker P1", win.pv[0].player === Player.P1);
ok("idtt first turn has <= moves_remaining cells", win.pv[0].cells.length <= 2);
console.log(`  idtt: kind=Win depth=${win.depth} turns=${win.pv.length} first=${JSON.stringify(win.pv[0].cells)}`);

// 2. solve_wide also finds a win (wide is a superset generator).
const wide = solver.solve_wide(winPos, limits);
ok("solve_wide win", wide.kind === 0);
console.log(`  wide: kind=Win depth=${wide.depth}`);

// 3. solve_threat flips to_move: from P2's shoes, does P1 (opponent) threaten?
const threatPos = new Position(4, 8, 300, Player.P2, 2, winStones);
const threat = solver.solve_threat(threatPos, limits);
ok("solve_threat finds P1 threat", threat.kind === 0 /* Win for the flipped attacker P1 */);
console.log(`  threat: kind=Win depth=${threat.depth} (P1 threatens from P2's shoes)`);

// 4. No forced win on a sparse, balanced position.
const quietPos = new Position(6, 8, 300, Player.P1, 2, [0,0,1, 1,0,2]);
const quiet = solver.solve(quietPos, new SolverLimits(4, 5000n, SolverEngineEnum.Idtt));
ok("quiet position is No", quiet.kind === 1 /* No */);
console.log(`  quiet: kind=No`);

// 5. Deep prover (Dfpn) on a game-valid position (has origin) — should agree Win.
const dfpn = solver.solve(winPos, new SolverLimits(6, 20000n, SolverEngineEnum.Dfpn));
ok("dfpn agrees Win", dfpn.kind === 0);
console.log(`  dfpn: kind=Win depth=${dfpn.depth}`);

// 6. Arbitrary board WITHOUT the (0,0,P1) origin: idtt must still solve (no panic).
const noOrigin = new Position(4, 8, 300, Player.P1, 2, [10,0,1, 11,0,1, 12,0,1, 20,5,2]);
const noOriginRes = solver.solve(noOrigin, new SolverLimits(6, 20000n, SolverEngineEnum.Idtt));
ok("idtt solves arbitrary no-origin board", noOriginRes.kind === 0);
console.log(`  no-origin: kind=Win depth=${noOriginRes.depth} (arbitrary board, idtt board-level path)`);

// 7. Deep prover on an arbitrary no-origin board returns BudgetExceeded (no panic).
const dfpnNoOrigin = solver.solve(noOrigin, new SolverLimits(6, 20000n, SolverEngineEnum.Dfpn));
ok("dfpn on no-origin board is BudgetExceeded (no panic)", dfpnNoOrigin.kind === 2);
console.log(`  dfpn no-origin: kind=BudgetExceeded (game-valid gate, honest give-up)`);

// 8. solve_defense on a position where the opponent has a refutable threat.
// P1 has (0,0),(1,0),(2,0); P2 already blocks (3,0); P2 to move. Threat: P1 wins
// at (-1,0). P2 can refute (block (-1,0) plus a second). Defense should find the
// threat AND a refutation (a killer or a pair anchor).
const defensePos = new Position(4, 8, 300, Player.P2, 2, [0,0,1, 1,0,1, 2,0,1, 3,0,2, 5,5,2]);
const defense = solver.solve_defense(defensePos, new SolverLimits(6, 20000n, SolverEngineEnum.Idtt));
ok("defense finds threat", defense.kind === 0 /* ThreatFound */);
ok("defense threat present", defense.threat !== null && defense.threat !== undefined);
ok("defense threat has a pv", defense.threat.pv.length > 0);
ok("defense found a refutation (killer or pair)", defense.killers.length > 0 || defense.pair_anchors.length > 0);
ok("defense best_delay present", defense.best_delay !== null && defense.best_delay !== undefined);
console.log(`  defense: kind=ThreatFound killers=${defense.killers.length} pair_anchors=${defense.pair_anchors.length} best_delay=${JSON.stringify([defense.best_delay.q, defense.best_delay.r])}`);

// 9. CoordW getters: pv cells expose .q / .r (they're wasm-bindgen instances, not plain objects).
ok("pv cell .q getter works", typeof win.pv[0].cells[0].q === "number");
ok("pv cell .r getter works", typeof win.pv[0].cells[0].r === "number");
console.log(`  getters: win first cell = (${win.pv[0].cells[0].q}, ${win.pv[0].cells[0].r})`);

console.log(`\nOK StrixSolver: ${pass} assertions passed`);