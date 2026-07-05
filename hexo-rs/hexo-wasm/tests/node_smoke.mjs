// Smoke test: the wasm build loads weights, evaluates, and searches.
// Run via: just wasm-test          (tiny fixtures, committed)
//       or just wasm-test-real     (real weights via WEIGHTS env var)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import assert from "node:assert/strict";

const here = dirname(fileURLToPath(import.meta.url));
// --target nodejs emits CommonJS; createRequire avoids fragile ESM<->CJS interop.
const require = createRequire(import.meta.url);
const { StrixBot } = require(join(here, "..", "pkg-node", "hexo_wasm.js"));

const realWeights = process.env.WEIGHTS; // set by wasm-test-real
const weightsPath = realWeights ?? join(here, "..", "..", "hexo-infer", "tests", "fixtures", "tiny.safetensors");
const weights = readFileSync(weightsPath);
const fixture = JSON.parse(readFileSync(join(here, "..", "..", "hexo-infer", "tests", "fixtures", "tiny_initial.json"), "utf8"));

const bot = new StrixBot(weights);
const info = JSON.parse(bot.model_info());
assert.equal(info.format, "hexo-safetensors-v1");
assert.ok(info.model_config, "model_info must carry model_config");
assert.ok(info.source_checkpoint, "model_info must carry source_checkpoint");

// Position config: real weights carry their training game_config (stage-specific,
// e.g. radius 6); tiny fixtures use the fixture's config.
const config = realWeights && info.game_config
  ? JSON.parse(info.game_config)
  : fixture.game_config;

// Initial HeXO position: P1 stone at origin, P2 to move with 2 placements.
const position = JSON.stringify({
  config,
  stones: [{ q: 0, r: 0, player: 1 }],
  to_move: 2,
  moves_remaining: 2,
});

// evaluate(): tiny weights must match the fixture value (wasm marshalling check);
// real weights just need a sane, normalized output.
const ev = JSON.parse(bot.evaluate(position));
if (!realWeights) {
  assert.ok(Math.abs(ev.value - fixture.expected.value) < 1e-4,
    `value ${ev.value} != fixture ${fixture.expected.value}`);
}
assert.ok(ev.value >= -1 && ev.value <= 1);
assert.ok(Math.abs(ev.policy.reduce((s, e) => s + e.p, 0) - 1.0) < 1e-5, "evaluate policy sums to 1");

// best_move(): legal, deterministic across seeds (seed is inert), normalized policy.
const t0 = performance.now();
const res = JSON.parse(bot.best_move(position, 64, 16, 42n));
const dt = performance.now() - t0;
const res2 = JSON.parse(bot.best_move(position, 64, 16, 99n));
assert.deepEqual(res.move, res2.move, "bot must be deterministic (seed inert, noise off)");
assert.ok(Math.abs(res.policy.reduce((s, e) => s + e.p, 0) - 1.0) < 1e-5, "best_move policy sums to 1");
const ps = res.policy.map((e) => e.p);
assert.ok(ps.every((p, i) => i === 0 || ps[i - 1] >= p), "policy sorted descending");
// Move legality: must appear in the returned policy's coord set (which covers all legal moves).
assert.ok(res.policy.some((e) => e.q === res.move.q && e.r === res.move.r), "move is legal");

// Validation must reject cleanly (not abort the module).
assert.throws(() => bot.best_move(JSON.stringify({ config, stones: [], to_move: 1, moves_remaining: 2 }), 8, 4, 0n),
  /origin/, "missing origin must throw a clean error");

const label = realWeights ? "REAL" : "tiny";
console.log(`OK [${label}]: move (${res.move.q},${res.move.r}) value ${ev.value.toFixed(4)} in ${dt.toFixed(0)}ms @ sims=64`);