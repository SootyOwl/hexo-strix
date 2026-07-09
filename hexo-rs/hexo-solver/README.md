# hexo-solver — VCF threat-space solver + deep prover

Fully-forcing (VCF) threat-space solver for HeXO, plus the deep proof-search
prover drivers. **Pure combinatorial search — no neural-net, MCTS, or PyO3
dependencies** (only [`hexo-engine`](../hexo-engine)). Extracted from
`hexo-mcts` so the solver can ship in a browser (`hexo-wasm`'s `StrixSolver`)
and be reasoned about in isolation.

## Modules

### `forcing`

The production iterative-deepening + persistent-TT solver — "can the side to
move force a win where every attacker turn creates a threat consuming the
defender's whole turn."

- `solve` / `solve_wide` (experimental wide-partner generator — a strict
  superset, so a `solve` win is never lost) / `solve_threat` (the opponent's
  forcing win if left unblocked) / `solve_defense` (the opponent threat plus
  the placements that refute it).
- The `SolverBoard` kernel (dense grid, make/unmake, deterministic Zobrist),
  threat-line table, and the `Outcome` (`Win(ForcingWin)` / `No` /
  `BudgetExceeded`) / `ForcingWin { depth, first_move, pv }` types.
- Depth convention: `depth` counts through the completing move — **"winning
  move in N"** (a puzzle author's "mate in N" is often `depth - 1`).

### `prover`

Deep proof-search research drivers sharing one rule set through `KernelCtx`:

- `idtt` — the production solver above, the baseline.
- `dfpn` — Nagai depth-first proof-number search.
- `pdspn` — Winands et al. two-level PDS-PN.
- `pns` — basic in-memory proof-number search (shared level-2 machinery for
  PDS-PN).
- `hybrid` — line-guided verification (composer lines as spines).
- `race` — portfolio: run several drivers on OS threads, take the first
  definitive verdict.

Powers the `prove` binary (which lives in `hexo-mcts` and re-exports this
module).

### `position` — arbitrary boards + engine dispatch

`SolverPosition { win_length, placement_radius, max_moves, to_move,
moves_remaining, stones }` describes a board that **need not be reachable in a
real game**. `solve_from_position` / `solve_wide_from_position` dispatch across
`SolverEngine { Idtt, Pns, Dfpn, Pdspn }`.

- **`idtt` (default)** takes a board-level path (`SolverBoard` built directly
  from the stones) and handles fully arbitrary boards — no origin requirement.
- **`Pns` / `Dfpn` / `Pdspn`** require a **game-valid position** (origin
  `(0,0,P1)` present, no duplicate or `(0,0,P2)` stones) because their PV
  reconstruction builds a `GameState`; invalid boards return
  `Outcome::BudgetExceeded` (an honest give-up, never a panic). `Pns` returns
  a verdict only (no PV) — use `idtt` for the principal variation.
- Pathological coordinate spreads return `BudgetExceeded` (a `MAX_COORD` /
  `MAX_GRID_CELLS` guard mirrors `solve_ex`).

## Consumers

- **`hexo-mcts`** — `mcts/forcing.rs` and `prover/mod.rs` are thin
  `pub use hexo_solver::*;` re-export shims, so the existing PyO3
  `solve_forcing` / `solve_defense` bindings, the MCTS depth-N shortcut, and
  the `prove` binary keep working unchanged.
- **`hexo-wasm`** — the `StrixSolver` browser export calls
  `solve_from_position` / `solve_wide_from_position` and a position-based
  `solve_defense`.

## WASM note

`std::time::Instant::now()` traps on `wasm32-unknown-unknown`, so every
timing site reachable from the WASM path is `#[cfg(not(target_arch =
"wasm32"))]`-gated: on wasm the wall-clock deadline is skipped (the search is
bounded by `node_budget` alone) and elapsed times report `0.0`. Native
behaviour is unchanged.