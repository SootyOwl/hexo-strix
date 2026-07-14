# Two solver-soundness bugs found via A2's integration gate (2026-07-14)

Both latent on main at `83a05b5` (the `solve_defense_wide` commit); both fixed
in the two commits under the `fix-solver-soundness` bookmark. Found because
the perf-plan A2 phase's integration check refused to carry a red
`test_serving_analysis.py::test_analyze_position_pv_owners_variable_length_first_chunk`
forward.

## 1. wl<5 guard pre-emptively killed sound deep wins

`83a05b5` added to `solve_from_state`: at `win_length < 5` with no immediate
completion, return `BudgetExceeded` before searching. Motivation (correct):
the strip scan only visits windows through the player's own stones (pc ≥ 1),
so at wl=4 the pc=0 threat-band element — a bare pair played into empty space,
already a forcing double threat at wl=4 — is invisible, making a proven `No`
unsound. But the guard also discarded deep wl=4 wins whose threats all run
through existing stones (pc ≥ 1), which the generator sees fine. The
serving-analysis mid-turn fork fixture (verified win, 5-cell PV) regressed
from Win (`c381ffb`) to None.

**Fix (soundness asymmetry):** an under-generating attacker is only ever
*deflated* — detected threats are real, so the hitting-set argument behind
`Win` proofs holds; only `No` is untrustworthy. Search normally at wl<5 and
downgrade top-level `No → BudgetExceeded` (both the sound-gate exit and the
iterative-deepening `!cut` exit). Quiet positions exit as cheaply as before;
deep pc≥1 wins return. Production (wl=6) unaffected.

Regression tests: `wl4_midturn_fork_is_searched_and_won`,
`wl4_no_win_found_is_never_a_proven_no`.

## 2. Zobrist pre-mix collided in-range

`zob(coord, player) = splitmix64(q·A ^ (r·B)<<1 ^ p<<63 ^ C)` — splitmix64 is
bijective, so any input collision passes straight through, and the
wrapping-multiply input is badly non-injective over real coordinates:

- **2,732 single-key collisions within |q|,|r| ≤ 64** — e.g. cells `(-2, 63)`
  and `(2, -63)` carry the SAME Zobrist value for the same player;
- ~1.6M pair-XOR collisions within ±15 (any two stone-pairs whose XOR agree
  make two different boards hash equal).

Consequence: whole-board hash collisions occurred routinely inside single
solves; every hash-keyed table (TT, comps cache, gencache) could return
verdicts computed for unrelated positions. In debug builds this surfaced as
the "double-place corrupts Zobrist" assert (the
`wide_defense_sights_quiet_builder_threat_tight_defense_cannot` test failure);
in release builds it was silent. Affected every solver consumer since the Rust
port: self-play forcing gate, serving analysis, WASM StrixSolver, prover runs.

Diagnosis path worth remembering: comps-cache entry ≠ fresh recompute at the
same hash, while `board.hash` DID equal the XOR-fold over `board.stones` —
i.e. not state corruption but genuine cross-position collision → suspect the
per-key values, then brute-force the pre-mix for duplicates in a small grid
(2,732 hits in seconds).

**Fix:** injective pre-mix — pack q, r into disjoint 32-bit halves,
`splitmix64`, XOR a per-player salt, `splitmix64` again. Per-key injectivity
over the playable range is pinned by `zobrist_keys_are_collision_free_in_range`
(±64 grid × both players).

**Caveat:** TT behaviour changes everywhere — budget-marginal solves may
resolve differently (correctly, but differently). Any experiment pinned to
exact solver node counts pre-fix is not comparable across the fix.

**Deployment note:** the live `self_play` binary (and any long-running
inference servers) predate the fix; the venv `hexo_rs` extension was rebuilt
2026-07-14. Rebuild the self-play binary at the next training pause. Solver
win-rates / self-play solver stats recorded before the fix carry the
collision noise.
