# hexo-rs

The Rust engine and search core for **HeXO** — infinite hex tic-tac-toe on an unbounded hexagonal grid.

Two players alternate placing stones (two per turn) on a hex grid, racing to form a line of *N* in a row. The board has no edges: it starts with a single P1 stone at the origin and grows outward as play continues. This workspace provides the game logic, a Gumbel AlphaZero MCTS, graph builders for GNN evaluation, and Python bindings.

It's part of the [HeXO](../README.md) monorepo and powers the training pipeline in [`hexo-a0`](../hexo-a0/README.md), but the engine crate is usable on its own.

## Game rules

- P1 opens with a stone at `(0, 0)`. P2 then moves first, two stones per turn.
- Play alternates in pairs: P2(2), P1(2), P2(2), …
- A move places a stone on any empty hex within a configurable radius of an existing stone.
- Win by forming *N* consecutive stones along any of the 3 hex axes.
- Draw if the move limit is reached with no winner.

The default configuration (`GameConfig::FULL_HEXO`) is 6-in-a-row, placement radius 8, 200-move limit.

## Crates

This is a Cargo workspace with six crates:

### `hexo-engine`

Core game logic — board, move validation, win detection, legal-move generation. No runtime dependencies beyond `std`.

- **Sparse board**: `HashMap<Coord, Player>` — grows with the game, no fixed bounds.
- **Incremental legal moves**: a cached `HashSet` updated on each move, no full recomputation.
- **Efficient win detection**: checks only the 3 axes through the last-placed stone.
- **Configurable**: win length, placement radius, move limit.
- Optional threat-feature and D6 symmetry helpers used by the search/graph layers.
- Ships the interactive `play` TUI binary (behind the `tui` feature).

### `hexo-mcts`

Gumbel AlphaZero MCTS plus the graph construction used to feed a GNN, and the PyO3 registration logic.

- **Gumbel MCTS** with Sequential Halving (Danihelka et al., 2022).
- **Batched neural-network evaluation** through a pluggable eval function.
- **Two graph builders**:
  - `graph.rs` — hex-adjacency edges (distance 1).
  - `axis_graph.rs` — axis-window edges (up to `win_length - 1` along the 3 win axes), 5-dim edge features, a global dummy node, and stop-at-opponent traversal.
- **PyO3 registration** (`python.rs`, behind the `python` feature) — the `hexo_rs` module contents. Not built directly; the `hexo-py` crate re-exports it (see below).
- **Native `self_play` binary** — generates games via MCTS with neural evaluation. Three inference paths: in-process libtorch (`--features torch`), a spawned Python inference subprocess (`--python-inference`), or a spawned native Rust server (`--inference-bin`, see `hexo-infer` below). See [Native self-play binary](#native-self-play-binary).
- **Forcing solver + deep prover re-exports** — `mcts/forcing.rs` and `prover/mod.rs` are thin `pub use hexo_solver::*;` shims; the VCF solver and the `idtt`/`dfpn`/`pdspn`/`pns`/`hybrid` deep-prover drivers live in [`hexo-solver`](#hexo-solver) (this crate keeps the PyO3 `solve_forcing`/`solve_defense` bindings and the MCTS depth-N shortcut, which resolve through the shims).

### `hexo-solver`

The fully-forcing (VCF) threat-space solver plus the deep proof-search prover drivers — pure combinatorial search with **no neural-net, MCTS, or PyO3 dependencies** (only `hexo-engine`).

- **`forcing` module** — the production iterative-deepening + persistent-TT solver (`solve`, `solve_wide`, `solve_threat`, `solve_defense`), the `SolverBoard` kernel, and the `Outcome`/`ForcingWin` types. Proves "winning move in N" (the depth convention).
- **`prover` module** — the deep-prover research drivers: `idtt` (the production solver, baseline), `dfpn` (Nagai depth-first proof-number search), `pdspn` (Winands et al. two-level PDS-PN), `pns` (basic PN search), and `hybrid` (line-guided verification), plus a `race` portfolio. Powers the `prove` binary (in `hexo-mcts`).
- **`SolverPosition` + `solve_from_position`** — an arbitrary-board loader (positions need not be reachable in a real game) with engine dispatch across `Idtt`/`Pns`/`Dfpn`/`Pdspn`. The `idtt` engine takes a board-level path and handles fully arbitrary boards; the deep provers require a game-valid position (origin `(0,0,P1)` + no duplicate/contradictory stones) and return `BudgetExceeded` otherwise.
- **Consumed by** `hexo-mcts` (re-exported for the PyO3 bindings + MCTS shortcut + `prove` binary) and `hexo-wasm` (the `StrixSolver` browser export).

### `hexo-infer`

Pure-Rust GNN forward pass — no libtorch, no Python. Loads a checkpoint from safetensors and runs the HeXONet representation network + policy/value heads (GINE conv, JK-cat, per-layer edge-attr dedupe). Compiles to `wasm32-unknown-unknown` as well as native.

- **Powers three consumers**: the browser bot (`hexo-wasm`), the native serving fast-path (via `hexo-py`'s `InferModel`), and the `hexo-infer-server` binary.
- **`hexo-infer-server` binary** — a drop-in for the Python inference subprocess, speaking the same HX04 stdin/stdout protocol, so `self_play --inference-bin <path>` can run inference with zero libtorch/Python. Best on low-radius shapes and to free the GPU for training; at high radius the GPU inference path wins (see `docs/research/2026-07-06-hexo-infer-selfplay-ab.md`).
- Includes an unused, bit-identical `forward_batch` (whole-collated-batch forward) kept as the batched-layout reference for a possible future Rust-on-GPU port — deliberately not wired, as collating is slower than the per-graph split on a bandwidth-limited CPU.

### `hexo-wasm`

`wasm-bindgen` `StrixBot` API — `hexo-infer` forward + `gumbel_mcts`, compiled for the browser/WebWorker so the bot runs client-side with no server round-trip. Ships a WebWorker demo. See [`hexo-wasm/README.md`](hexo-wasm/README.md).

### `hexo-py`

The Python extension crate that maturin actually builds into the `hexo_rs` module. Re-registers the full `hexo-mcts` module (the crate cycle — `hexo-infer` depends on `hexo-mcts` — means this glue cannot live in either) and adds the `hexo-infer`-backed native inference classes (e.g. `InferModel`). See [Building the Python extension](#building-the-python-extension).

## Using the engine (Rust)

Add `hexo-engine` as a path or git dependency, then:

```rust
use hexo_engine::{GameState, GameConfig, Player};

// Standard 6-in-a-row game
let mut game = GameState::new();

// Or configure a smaller variant
let config = GameConfig { win_length: 4, placement_radius: 4, max_moves: 100 };
let mut game = GameState::with_config(config);

// P2 moves first (P1's opening stone at the origin is placed for you)
assert_eq!(game.current_player(), Some(Player::P2));

game.apply_move((1, 0)).unwrap();
game.apply_move((0, 1)).unwrap();

// Now it's P1's turn
assert_eq!(game.current_player(), Some(Player::P1));

// Query game state
let moves = game.legal_moves();          // sorted (q, r) pairs
let count = game.legal_move_count();      // no allocation
let cached = game.legal_moves_set();      // &HashSet, incrementally maintained
let done = game.is_terminal();
let winner = game.winner();
```

`GameConfig` has exactly three fields: `win_length: u8`, `placement_radius: i32`, `max_moves: u32`. The two-stones-per-turn rule is fixed.

## Interactive TUI

Play HeXO in the terminal:

```bash
cargo run --bin play -p hexo-engine --features tui
```

Controls: click to place, `t` to toggle theme, `r` to restart, `q` to quit.

## Building the Python extension

The `hexo_rs` module is built with [maturin](https://www.maturin.rs/) from the **`hexo-py`** crate (its `pyproject.toml` sets `manifest-path = "hexo-py/Cargo.toml"` and `module-name = "hexo_rs"`). Normally you build it via the workspace `uv sync` from the [repo root](../README.md), which handles it automatically. To build just this module into an active Python 3.13+ environment:

```bash
pip install maturin
maturin develop --release
```

This compiles `hexo-py` — which pulls in `hexo-mcts` (with the `python` feature) and `hexo-infer` — and installs the `hexo_rs` module. The build settings (manifest path, features, module name) are in `hexo-py/pyproject.toml`, so plain `maturin develop` / `maturin build` work with no extra flags.

The module exports the `GameState`, `GameConfig`, and `MCTSConfig` classes, the `gumbel_mcts` search entry points, the `game_to_graph_raw` / `game_to_axis_graph_raw` graph builders (with batch variants), symmetry augmentation helpers, `batched_self_play`, and the `hexo-infer`-backed `InferModel` (pure-Rust GNN forward + native search, used by the serving fast-path). See `hexo-mcts/src/python.rs` and `hexo-py/src/lib.rs` for the full list.

```python
import hexo_rs

game = hexo_rs.GameState(hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=100))
edges, node_features, edge_features = hexo_rs.game_to_axis_graph_raw(game)
```

> **Note:** [`hexo-a0`](../hexo-a0/README.md) depends on this package under the name `hexo-rs`. The workspace `uv sync` builds this module into the shared environment automatically, so you rarely need to run `maturin` by hand.

## Native self-play binary

`hexo-mcts` ships a `self_play` binary that generates games via MCTS with neural evaluation. It has three inference paths:

- **`--features torch`**: in-process libtorch inference (requires libtorch and a C++ compiler).
- **`--python-inference`**: spawns a Python inference subprocess over a binary stdin/stdout protocol (HX04; no libtorch needed at build time).
- **`--inference-bin <path>`**: spawns the pure-Rust [`hexo-infer-server`](#hexo-infer) over the same HX04 protocol — no libtorch, no Python. Implies `--python-inference`; axis graphs only.

```bash
cargo run --bin self_play -p hexo-mcts --features torch -- --help
```

Most users drive self-play through the [`hexo-a0`](../hexo-a0/README.md) training pipeline rather than invoking this binary directly.

## Performance

Benchmarked on the default `FULL_HEXO` configuration (6-in-a-row, radius 8):

| Operation | 50 stones | 100 stones |
|---|---|---|
| `apply_move` | 5.1 µs | — |
| `clone` | 490 ns | 771 ns |
| `legal_moves` (allocates + sorts) | 21 µs | 35 µs |
| `legal_moves_set` (cached reference) | 0 ns | 0 ns |

The MCTS hot path (clone + apply_move) runs in ~5 µs. Run benchmarks with:

```bash
cargo bench -p hexo-engine
```

## Coordinates

HeXO uses axial hex coordinates `(q, r)` with 6 neighbour directions. Hex distance is `max(|dq|, |dr|, |dq + dr|)`. The 3 win axes are `(1, 0)`, `(0, 1)`, and `(1, -1)`.

## Related

- [`hexo-a0`](../hexo-a0/README.md) — the Python training pipeline (GNN + Gumbel MCTS) that consumes these bindings.
- [`hexo-wasm`](hexo-wasm/README.md) — the browser/WebWorker `StrixBot` built on `hexo-infer`.

## License

MIT — see [LICENSE](LICENSE).
