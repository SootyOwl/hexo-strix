# hexo-wasm — strixbot in the browser

wasm-bindgen wrapper around `hexo-infer` (pure-Rust GNN forward pass, no
libtorch/Python) + the existing `gumbel_mcts` search, compiled to
`wasm32-unknown-unknown` for in-browser / WebWorker play.

## Build

```bash
just wasm-build        # web target -> hexo-rs/hexo-wasm/pkg/
just wasm-test         # nodejs target + smoke test (tiny fixtures, committed)
just wasm-test-real    # smoke test against the real exported weights (release)
just check-wasm        # compile gate for all wasm-facing crates
```

Weights come from a training checkpoint:

```bash
uv run --no-sync hexo-a0 export --checkpoint <checkpoint.pt> --out weights.safetensors
```

Measured (AMD Strix Halo, single-threaded): real rel2 model (~284k params,
1.1 MB fp32 safetensors) ≈ **1.8 s/move at sims=64** in release wasm,
≈ 1.6 s native. Lower `sims` for snappier play — the Elo curve is steep from
32→64 and flat above; sims=16–32 is a respectable fast mode. `.wasm` is
463 KB raw / 197 KB gzipped before weights.

## API

```js
import init, { StrixBot } from "./pkg/hexo_wasm.js";
await init();
const bot = new StrixBot(new Uint8Array(weightsBytes));
const info = JSON.parse(bot.model_info());
const res  = JSON.parse(bot.best_move(positionJson, 64, 16, 0n));
const ev   = JSON.parse(bot.evaluate(positionJson));
```

- `model_info()` → the safetensors metadata: `model_config`, `train_steps`,
  `source_checkpoint`, and `game_config` (the **training** game config —
  stage-specific for curriculum checkpoints, e.g. radius 6 at S3).
- `best_move(positionJson, sims, mActions, seed)` → search result.
  **The bot is fully deterministic** (Gumbel noise and Dirichlet are disabled,
  matching the production serving path): `seed` is currently **inert**,
  reserved for future variety modes. Don't wire a shuffle button to it.
- `evaluate(positionJson)` → raw network eval, no search.

## Position JSON

```json
{
  "config": {"win_length": 6, "placement_radius": 6, "max_moves": 300},
  "stones": [{"q": 0, "r": 0, "player": 1}, {"q": 1, "r": 0, "player": 2}],
  "to_move": 2,
  "moves_remaining": 2
}
```

Rules enforced by validation (clean errors, never a wasm abort):

- `config` must not exceed the model's training `game_config`
  (`win_length`/`placement_radius`) — mirror the values from `model_info()`.
- The origin stone `{"q":0,"r":0,"player":1}` is **required** (the engine
  pre-seeds P1 at origin; move accounting counts it).
- `player`/`to_move` ∈ {1,2}, `moves_remaining` ∈ {1,2}, no duplicate coords,
  stones within `placement_radius`, position not already won or drawn.
- `(to_move, moves_remaining)` consistency with the stone count is the
  CLIENT's responsibility — arbitrary sandbox positions are accepted, but
  inconsistent triples give out-of-distribution evals, not errors.

## Response JSON

`best_move`:

```json
{"move": {"q": 1, "r": -1}, "value": 0.42, "policy": [{"q":1,"r":-1,"p":0.61}, ...]}
```

- `value` ∈ [-1,1] is the **network value for the root position, from
  `to_move`'s perspective** — it flips sign with `to_move`. Do not read it as
  "P1's advantage".
- `best_move().policy` is the **search-improved** distribution;
  `evaluate().policy` is the **raw prior** softmax. Same shape, different
  numbers — not interchangeable. Sorted descending by `p`, ties broken by
  coord ascending.

## StrixSolver — standalone forced-win solver (no weights)

`StrixSolver` is a second export in the same `pkg/`: the fully-forcing (VCF)
threat-space solver from `hexo-solver`, exposed via typed `wasm-bindgen`
structs/enums (no JSON, no model weights). Use it to answer "does the side to
move have a forced win, and what's the line?" in a web UI. `idtt` (the
production iterative-deepening solver) is the default and the only engine
that returns a principal variation.

```js
import init, { StrixSolver, Position, SolverLimits, Player, SolverEngineEnum } from "./pkg/hexo_wasm.js";
await init();
const solver = new StrixSolver();
const pos = new Position(
  6,         // win_length
  8,         // placement_radius
  300,       // max_moves
  Player.P1, // to_move (the attacker / side to move)
  2,         // moves_remaining (1 or 2)
  [ /* stones: {coord: {q,r}, player: Player} ... */ ],
);
const limits = new SolverLimits(10, 20000n, SolverEngineEnum.Idtt); // depth_cap, node_budget, engine
const outcome = solver.solve(pos, limits);
```

Methods: `solve`, `solve_wide` (experimental wide-partner generator — a
superset, so a `solve` win is never lost), `solve_threat` (flips `to_move`:
the *opponent's* forcing win if left unblocked — a threat, not a proven loss),
`solve_defense` (the opponent threat plus the placements that refute it).

### Return types

`SolveOutcome` has `kind` (`SolveKind::Win` / `No` / `BudgetExceeded`),
`depth` ("winning move in N"), `nodes`, `time_ms`, and `pv` — a
turn-grouped `Vec<Turn>` where each `Turn` is `{turn, player, cells: Vec<Coord>}`.
The first turn is the attacker with `moves_remaining` cells; subsequent turns
are 2 cells each, alternating players. `Coord` is `{q, r}`. `DefenseOutcome`
adds `threat: Option<SolveOutcome>`, `killers: Vec<Coord>`,
`pair_anchors: Vec<PairAnchor>` (`{first, second}`), and `best_delay`.

### Engines

`SolverEngineEnum`: `Idtt` (default), `Pns`, `Dfpn`, `Pdspn`. `idtt` takes a
board-level path and is the only engine that returns a PV; `Pns` returns a
verdict only (empty `pv`, `depth` 0). `Dfpn`/`Pdspn`/`Pns` are deep
proof-search drivers.

### Arbitrary boards

`Position.stones` may be **any arrangement** — positions need not be
reachable in a real game. `idtt` handles fully arbitrary boards (no origin
requirement). The deep provers (`Pns`/`Dfpn`/`Pdspn`) require a **game-valid
position** (origin `{q:0,r:0,player:P1}` present, no duplicate or
contradictory `{q:0,r:0,player:P2}` stones) because their PV reconstruction
builds a `GameState`; invalid boards return `BudgetExceeded` rather than
panicking. `solve_defense` likewise requires a game-valid position (returns
an error otherwise). Pathological coordinate spreads return `BudgetExceeded`.

### Caveats

- `time_ms` is **0.0 on wasm** — `std::time::Instant` has no monotonic clock
  on `wasm32-unknown-unknown`, so timing is reported only on native. `node_budget`
  bounds the search on all targets.
- `nodes` is 0 for `idtt` (the production solver does not expose a node count);
  the deep provers report search stats internally.
- `solve_defense` returns `NoThreat` for both "no opponent threat" and
  "budget exceeded during the threat check" (`BudgetExceeded` variant is
  reserved for when the two are distinguished). `limits.engine` is inert for
  defense (always uses the forcing solver internally).


## WebWorker demo

`demo/` is a minimal module-worker integration (no framework, no build step):

```bash
just wasm-build
cp <exported>.safetensors demo/weights.safetensors   # gitignored
python3 -m http.server 8931 -d hexo-rs/hexo-wasm
# open http://localhost:8931/demo/
```

`worker.js` message protocol: `{type:"init", weightsUrl}` → `{type:"ready", info}`;
`{type:"best_move", position, sims, mActions, seed}` → `{type:"move", result, ms}`;
`{type:"evaluate", position}` → `{type:"eval", result}`; failures →
`{type:"error", error}`. Requires module-worker support (all evergreen
browsers). No special headers needed (no SharedArrayBuffer/threads). If the
server doesn't send `application/wasm`, wasm-bindgen falls back from
`instantiateStreaming` with a benign console warning.

## hexo.did.science integration notes

- Everything is self-contained: one `.wasm` + JS glue + a weights file. No
  CDN, no COOP/COEP headers.
- Coordinate mapping from HDS sandbox positions: see
  `hexo-a0/src/hexo_a0/serving/hds.py` (the reference converter — HDS `(x,y)`
  → axial `(q,r)` identity, with the htttx.io-mirror codec handled there).
- Run the bot in a worker (as in `demo/worker.js`) so search never blocks the
  UI thread; at sims=64 a move is ~2 s on desktop.

## Maintenance coupling

The forward pass exists in four implementations (eager `HeXONet`,
`ScriptableHeXONet`, the tch-rs TorchScript loader, and `hexo-infer`).
`hexo-infer` loads **only** GINE/axis/pre-norm models (jk-cat or no-JK) and
rejects everything else loudly at load time. Architecture changes must either
stay inside that envelope or extend `hexo-infer` + regenerate the parity
fixtures (`uv run --no-sync python -m hexo_a0.gen_parity_fixtures`).
