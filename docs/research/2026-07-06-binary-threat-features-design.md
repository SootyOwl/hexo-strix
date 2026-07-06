# Binary threat features (`own_hot` / `opp_hot`) — design

**Status:** planned, not implemented. Target: the lean-d6 from-scratch experiment
family. Production (`4l-128p32v-jkcat-rel2`) is unaffected.

## Context

ShrimpNet (hexo.blueshrimp.uk/learn/network.html) encodes threats as two binary
per-cell flags — own_hot / opp_hot: "empty cells inside a six-cell single-colour
window already holding ≥ 4 stones". Ours are 4 continuous dims from
`node_threat_features` (`hexo-rs/hexo-engine/src/threat.rs:15`):
`[own_max_line, opp_max_line, own_threat_axes, opp_threat_axes]`, normalised by
`win_length` and 3.

The lean-d6 schema (`configs/gine-mini/*lean-d6*.toml`) is built on the premise
that the architecture carries the structure (exact D6 invariance by
construction) so the input encoding can be minimal. The continuous threat block
is the one remaining input doing more work than it obviously needs to — the
forced-win MCTS shortcut (`gumbel_mcts.rs` Step 3.5, Phase A depth-1 + Phase B
`forcing::solve`) already owns exact "win now" tactics, so the network only
needs a tripwire, not a magnitude.

## Hypothesis

A hard binary alarm at the ≥ `win_length - 2` threshold ("completable this
turn", since turns are two placements) carries the tactically load-bearing part
of the threat signal. The sub-threshold magnitude gradient (2-of-6 vs 3-of-6)
is learnable by the backbone from raw occupancy.

**Prediction:** lean-d6 + binary threat matches lean-d6 baseline Elo at
convergence with 2 fewer input dims. Watch-for: slower early-stage (W1/W2/S1)
value calibration from losing the sub-threshold gradient — if S1/S2 convergence
is noticeably slower than the count-based lean-d6 baseline, the continuous
magnitude was pulling weight the binary can't.

## Feature definition

New opt-in `ModelConfig` flag `threat_hot: bool = False`, only meaningful with
`threat_features = true`:

- `own_hot`  = 1.0 iff any of the 3 axes has a clean window (no opponent
  stones) through the node holding ≥ `win_length - 2` own stones; else 0.0.
- `opp_hot`  = same with roles swapped. "Own" = side to move, as today.

Equivalently: `own_hot = (own_threat_axes > 0)` — the thresholding already
exists at `threat.rs:53-58`; the binary flags are a projection of the current
outputs. Computed **uniformly over all nodes** (stones included), matching how
every other node feature works; the axis dummy node stays all-zero.

Threat block width: 4 → 2. Resulting node widths:

| schema                        | today | with `threat_hot` |
|-------------------------------|-------|-------------------|
| absolute + threat             | 12    | 10                |
| relative + threat             | 11    | 9                 |
| lean-d6 (rel, compact, -coords) | 8   | 6                 |

`threat_hot = true` with `threat_features = false` is a config error (raise at
model construction).

## Backwards compatibility

- `threat_hot` defaults False; `model_config_from_checkpoint`
  (`hexo-a0/src/hexo_a0/config.py:107`) drops unknown keys, so old checkpoints
  load and resolve to today's behaviour bit-identically, and new checkpoints
  embed the flag. jkcat-rel2 never sees a diff.
- **From-scratch only** (like the rest of the lean schema). No graft:
  `graft_input_proj_for_threat_features` (`hexo-a0/src/hexo_a0/model.py:811`)
  hardcodes the 8→12 widening and refuses relative encoding anyway; do not
  extend it. `graft_threat_features + threat_hot` is a config error.

## Changes by file (hotspots mapped 2026-07-06)

The node width flows dynamically almost everywhere (Python reshapes `(n, -1)`;
the inference wire protocol carries `node_dim` per batch and is width-agnostic
— `inference_subprocess.rs:185-188`). The hardcoded "4"s are the work:

**Rust**
1. `hexo-engine/src/threat.rs` — derive the binary pair. Cheapest correct
   form: keep `node_threat_features` untouched; the fill site thresholds
   `own_axes/opp_axes > 0`. (A dedicated `[f32; 2]` kernel that early-exits per
   axis is a later micro-opt, not needed for the ablation.)
2. `hexo-mcts/src/graph.rs:239` `fill_threat_features` — parameterise the
   threat width: `stride = base_dim + n_threat` (today hardcoded `+ 4` at
   `:251`, write at `:256`). Thread a mode (second bool `threat_hot` alongside
   `threat_features`, matching the existing plain-bool precedent) through
   `build_graph` (`:66`, `fdim` at `:75`), `game_to_graph_raw_opts` (`:271`),
   `game_to_graph_batch_opts` (`:344`).
3. `hexo-mcts/src/axis_graph.rs` — same threading: `build_axis_graph:149`
   (`fdim` at `:167`), `fill_axis_threat:498`, the `*_opts`/`*_lean`/batch
   variants (`:525/:544/:586/:602`), and the augment path
   `augment_axis_graph_all_opts:633` / `augment_axis_graph_single:666` (it
   rebuilds threat features on the transformed board at `:717`).
4. `hexo-mcts/src/python.rs` — new kwarg `threat_hot=false` on the 4
   pyfunctions: `py_game_to_graph_raw:693`, `py_game_to_graph_batch:723`,
   `py_game_to_axis_graph_raw:822`, `py_game_to_axis_graph_batch:858`.
   (`py_batched_self_play`/`py_native_self_play` take no graph flags — width
   is chosen by the Python eval callback — no change.)
5. `hexo-mcts/src/bin/self_play.rs` — new `--threat-hot` CLI flag (parse loop
   near `:1816`), thread through `build_position_graph:1293` and the game
   workers; update `node_dim:2411` (`+ if threat {4} else {0}` → width-aware);
   `subprocess_model_args:2418` emits `--threat-hot` so the Python inference
   server sizes its model to match (the `--node-dim` emission at `:2457-2459`
   already covers the width number).

**Python**
6. `hexo-a0/src/hexo_a0/config.py` — new `ModelConfig.threat_hot` field
   (Graph construction group, FROM-SCRATCH ONLY note);
   `node_feature_dim:45` → `base + (2 if hot else 4 if threat else 0)`;
   `legacy_lean_columns:65` → threat keep-columns become `[7, 8]` (relative) /
   `[8, 9]` (absolute) in hot mode.
7. `hexo-a0/src/hexo_a0/graph.py` — factories
   `graph_fn_from_model_config:203` / `graph_batch_fn_from_model_config:237`
   read and forward the flag; public wrappers gain the kwarg.
   **Fix the load-bearing width sniff in `random_augment` (`:562`)**:
   `coord_q_idx = 5 if new_x.shape[1] in (8, 12) else 4` misclassifies a
   10-dim absolute graph as relative and corrupts the coord recompute. Extend
   to `in (8, 10, 12)` with a comment tying the sets to the legacy layouts
   (absolute even {8,10,12} / relative odd {7,9,11}; lean graphs must never
   reach `random_augment` — they have no coord columns). Update the docstring
   at `:470-472`. lean-d6 runs `augment_symmetries=false` so this is
   defensive, but it must not be skipped: a non-lean ablation of `threat_hot`
   would silently train on corrupted augmentations otherwise.
8. `hexo-a0/src/hexo_a0/trainer.py:_build_selfplay_cmd` (~`:1769`) — append
   `--threat-hot` beside the existing `--threat-features`.
9. `hexo-a0/src/hexo_a0/inference_server.py` — argparse `--threat-hot`
   (near `:716`), copy from the checkpoint's embedded model_config in the
   auto-configure block (`:161-167`), include in the `node_feature_dim`
   recompute (`:176`) and `ScriptableHeXONet` construction (`:199`).
10. `hexo-a0/src/hexo_a0/scriptable_model.py` — forwards via
    `node_feature_dim`/`legacy_lean_columns` (`:40`, `:447`, `:791`); verify no
    other hardcoded threat-width assumptions.

**Threat dims and D6:** the binary flags are per-node D6 invariants exactly
like the continuous ones — `random_augment` permutes node rows and only
recomputes coord columns, so no new transform logic is needed.

## Tests

- Rust `threat.rs`: parallel binary assertions on the existing 8 fixtures
  (e.g. `three_own_in_row_wl4` → `own_hot=1` since 3 ≥ 4−2;
  `opponent_stone_blocks_window` → both 0).
- Rust `graph.rs`/`axis_graph.rs`: width analogues of
  `threat_features_widen_to_12_hex:381` / `relative_threat_features_at_7_to_11:462`
  / `axis_graph.rs:779,:867` pinning widths 10/9 and dummy-node zeros.
- Python `test_threat_features.py`: hot-mode analogues of
  `TestGraphFeatureDim` and `test_known_threat_values:61` (hand-derived
  binaries for the same fixture positions); a `random_augment` case at width
  10 proving the coord-index fix.
- `test_axis_graph_lean.py`: lean+hot width = 6
  (`test_lean_node_feature_width_is_8:78` analogue); native-vs-oracle parity
  (`test_node_features_match_oracle:84`) with hot enabled.
- Config error tests: `threat_hot` without `threat_features`; `threat_hot`
  with `graft_threat_features`.
- Regression: full existing suite unchanged (`just test`) — the 4-dim path
  must be byte-identical.

## Rollout

New config `configs/gine-mini/4l-128p32v-lean-d6-hotthreat.toml`: copy of
`4l-128p32v-lean-d6.toml` + `threat_hot = true`, run dirs rebased to
`runs/gine-mini/4l-128p32v-lean-d6-hotthreat/`, header documenting
hypothesis/prediction per house style. A/B against the lean-d6 baseline
(same seeds/stages); compare stage convergence speed (esp. S1/S2, per the
prediction) and final SPRT/corpus yardsticks.

## Verification (end-to-end, before starting the run)

1. `just build && just test` — all Rust + Python tests green.
2. Byte-identity spot check: build a mid-game position, dump features with
   `threat_hot=false` and diff against current main's output.
3. Smoke the full loop on the hotthreat config with tiny budgets (W1 stage,
   few hundred steps): Rust self-play binary → inference server (`--node-dim 6`
   handshake, wire width check at `inference_server.py:280`) → trainer →
   checkpoint save/load round-trip (embedded model_config carries
   `threat_hot`) → eval.
