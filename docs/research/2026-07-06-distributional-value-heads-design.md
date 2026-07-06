# Distributional value, horizon heads, and a train-only Q head — design

**Status:** planned, not implemented. Target: from-scratch runs (lean-d6
family and successors). Production (`4l-128p32v-jkcat-rel2`) is unaffected:
every new feature defaults off and the scalar path stays bit-identical.

## Context

The scalar tanh `ValueHead` (`hexo-a0/src/hexo_a0/model.py:400`) trained with
MSE against the final game outcome (`trainer.py:860-861`) is the network's
only value signal. It has been observed to be difficult in practice, and it
is sparse: at full radius (`max_moves = 300`) most positions are far from the
single terminal label they train against.

ShrimpNet (hexo.blueshrimp.uk/learn/network.html) demonstrates the richer
package on the same game: a 65-bin distributional value head (scalar =
softmax expectation), the same binned form at short horizons (2/6/16 turns,
"a denser training signal than the final result alone"), and train-only
auxiliary heads — including a per-cell Q head — that "exist to improve the
trunk's representation" and add zero cost at serve time.

Two codebase facts make this cheap for us and shape the design:

1. **`per_child_q` is already computed and thrown away.** Gumbel MCTS
   produces a completed-Q per legal move (`MCTSResult.per_child_q`,
   `hexo-rs/hexo-mcts/src/mcts/gumbel_mcts.rs:40-74`, filled at `:551`), but
   `self_play.rs::play_one_game` uses it only for exploration acting and
   telemetry (`:1474`, `:1522-1530`) and never copies it into `PositionData`
   (`:1087-1103`). The expensive part of a Q head — accurate per-move search
   values — is already paid for at self-play time.
2. **Within-trajectory order is NOT persisted.** HX07 records carry no move
   index and `TrainingExample` (`self_play.py:326-356`) has no order field;
   the replay buffer pickles examples individually and samples randomly.
   Horizon targets therefore CANNOT be built at train time — they must be
   computed at data-ingest time in `trainer._parse_state_examples`
   (`trainer.py:2287-2329`), where `game_data["examples"]` is still the full
   ordered position list and the winner is known. (This is exactly where the
   existing `value_target` is already derived from the final outcome.)

## Hypothesis

**Prediction:** (a) the binned value head trains more stably than scalar
MSE (categorical losses avoid the tanh-saturation / outlier-gradient
pathologies of regression); (b) horizon heads densify the value signal in
long games and speed value-head calibration at S3/S4 radii; (c) the Q head
distils per-move search knowledge into the trunk, improving policy/value
quality at zero serve cost. Each is separately ablatable via its own flag.

## Design overview — three stages, one shared mechanism

All three heads share the binned-distribution machinery and all plug into
`HeXONet._forward_batch_core` (`model.py:516-584`), which is the
torch.compile target (`cli.py:369-374`, `curriculum.py:806-819` compile this
method, dynamic=True) — new head code must stay graph-break-free: only
`index_select`/`scatter`, no data-dependent `nonzero`/`.item()`.

**Zero-serve-cost rule (by construction):** the self-play inference wire is
exactly `(policy_logits, legal_counts, values)`
(`inference_server.py:_write_forward_response:125-139`, protocol VERSION=2 in
`inference_subprocess.rs`). Horizon and Q heads are **never added to
`ScriptableHeXONet`** — they exist only in the training forward. The binned
value head IS exported, but computes its softmax expectation internally and
returns the scalar in the existing `values` slot — the wire and the Rust
reader never change.

### Stage 1 — distributional (binned) value head

- New `ModelConfig` fields: `value_bins: int = 0` (0 = scalar head, current
  behaviour, default), `value_bin_min/max` fixed at −1/+1 (not configurable
  until needed). Suggested starting point: 65 bins (ShrimpNet's choice; odd
  count puts a bin center exactly at 0 for draws).
- `BinnedValueHead`: same stone-pooled input as today (reuses `pooled` from
  `_forward_batch_core:575-580`), MLP `Linear(head_in_dim, value_hidden) →
  ReLU → Linear(value_hidden, value_bins)` — no tanh. Scalar =
  `softmax(logits) @ bin_centers` (a registered buffer). `head_in_dim` comes
  from `representation.output_dim` as for the existing heads
  (`model.py:462`).
- Loss (`loss.py`, new helper + slot at `trainer.py:860-861`): cross-entropy
  against the **two-hot projection** of the scalar target onto the two
  adjacent bins (linear interpolation, C51-style) — exact in expectation,
  differentiable-friendly, no binning bias at bin edges. Weighted by the
  existing `sw / sw_sum` sample-weight pattern (`trainer.py:847-858`).
  Keep logging the decoded-scalar MSE alongside as `loss/value_mse` so runs
  stay comparable with scalar baselines.
- MCTS/eval consumers are untouched: they see the decoded scalar in the same
  slot everywhere (`forward_batch`, `_forward_batch_core` return arity for
  `values_tensor` unchanged).
- `ScriptableHeXONet` (`scriptable_model.py`): binned variant of `value_mlp`
  (L524-529, compute at L678-696) decoding to scalar in-forward;
  `load_from_hexonet` (L701-772) gains the name mapping for the binned head —
  without it, new params are silently dropped by the `named_parameters()`
  filter at L749. `fullgraph=True` compile (`inference_server.py:236`) is
  safe: softmax+matmul only.
- Self-play plumbing: `trainer.py:1790-1811` (subprocess args) and
  `inference_server.py` argparse (L668-773) gain `--value-bins`, copied from
  the checkpoint's embedded model_config like the existing flags.

### Stage 2 — short-term value horizon heads (train-only)

- New `ModelConfig` field: `value_horizons: list[int] = []` (unit:
  **placements**, to avoid turn/placement ambiguity — a HeXO turn is two
  placements; suggested `[4, 12, 32]` ≈ ShrimpNet's 2/6/16 turns). Requires
  `value_bins > 0` (validated in `config_io._validate_config:175-228`) so all
  value heads share one bin scheme.
- Target definition (computable at ingest with zero new data from Rust):
  for a position at index `i` in a game of length `L`, horizon `k`:
  `z_k = value_target if (L − i) <= k else 0.0` — i.e. the actual outcome
  (side-to-move-relative, `draw_value` for draws, matching the existing
  convention at `self_play.rs:1245-1252`) when the game resolves within the
  horizon, neutral otherwise. This teaches "is this position about to
  resolve, and how" — the dense near-terminal signal the full-game label
  lacks. (Alternative — bootstrapped value of the position k later — needs
  stored search values and is deferred; see Open questions.)
- Ingest: computed in `_parse_state_examples` (`trainer.py:2287-2329`) while
  walking the ordered examples; stored as a new `TrainingExample` field
  `horizon_targets: list[float] | None = None`. The buffer pickles examples
  whole (`replay_buffer.py:195`) so no schema change; all read sites use
  `getattr(ex, "horizon_targets", None)` so old pickled examples in a live
  buffer keep working (same precedent as `sample_weight`). The parallel
  ingest paths (`_parse_graph_examples:2331-2373`, Python-engine
  `batched_self_play_games`) either get the same treatment or explicitly
  yield `None` (loss skipped for those examples).
- Example-tuple plumbing: extend the arity ladder in `_forward_and_loss`
  (`trainer.py:775-791`) and both tuple build sites (prefetch worker
  `:694-696`, inline path `:2852-2855`).
- Heads: one `BinnedValueHead` per horizon fed from the same `pooled`
  tensor; loss = same two-hot CE, weight `TrainingConfig.horizon_loss_weight`
  (default 0.0 = off), plumbed exactly like `pc_loss_weight`
  (`config.py:216` → `train_step:1010` → `_forward_and_loss:756`). NOT
  exported to `ScriptableHeXONet` (whitelisted in `load_from_hexonet`'s
  `real_missing` guard at L756-770, like `edge_proj`/`layer_scales`).

### Stage 3 — train-only per-move Q head (needs a data-format bump)

- Data: new **HX08** record (append-only extension of HX07,
  `self_play.rs:1189-1277`): after the policy block, per legal move
  `f32 q` (root-player-perspective completed-Q, `result.per_child_q`, same
  order/length as the `policy` block) and `u16 visits`
  (`result.visit_counts`). Visits matter because unvisited children's
  completed-Q is the v_mix fill — near-constant per position and majority of
  cells at m=16 of ~200 legal; regressing on them would drown the signal.
  `PositionData` (`:1087-1103`) gains both fields, populated at `:1543` where
  the result is in hand. Bump the record magic; keep the HX07 writer path
  gated for one release if we want old-trainer compat (probably unnecessary —
  trainer and binary ship together).
- Reader: `_parse_binary_example_hx08` mirroring `:254-316`; magic table at
  `:2069-2073`. `TrainingExample` gains `q_targets` / `q_visits`
  (None-default). Old HX03–07 files keep parsing; examples without Q simply
  skip the Q loss.
- Head: MLP over `all_legal_embeddings` (already materialised at
  `model.py:556-559` for the policy head — the Q head reuses `legal_idx` and
  costs one extra MLP over the same gather). Output: start with **scalar per
  cell + MSE masked to `visits > 0`** (weight `q_loss_weight`, default 0.0),
  not ShrimpNet's per-cell 65-bin form — a scalar target (completed-Q) wants
  a scalar regression first; upgrade to binned later if it earns it. Q
  targets are root-perspective at the root's side-to-move — the same
  perspective as the value target, so no sign gymnastics.
- Train-only: lives behind a `ModelConfig.q_head: bool = False` flag; forward
  computed only in the training path (either a trailing element of
  `_forward_batch_core`'s return — Dynamo handles growing tuple arity — or a
  separate `forward_q` method the trainer calls on the shared embeddings).
  Never in `ScriptableHeXONet`, never on the wire.

## Backwards-compatibility invariants (all three stages)

1. Defaults reproduce today bit-for-bit: `value_bins=0`, `value_horizons=[]`,
   `q_head=False`, all new loss weights 0.0. `config_io._load_section`
   (`:65-82`) drops unknown TOML keys with a warning, and
   `model_config_from_checkpoint` (`config.py:107-124`) drops unknown
   checkpoint keys — old configs and old checkpoints resolve to current
   behaviour, new checkpoints round-trip the new fields via the embedded
   `asdict(model_config)` (`trainer.py:4479`).
2. The inference wire protocol (VERSION=2) is untouched at every stage.
3. Old game files (HX03–07) and old pickled buffer examples stay readable;
   absent targets disable the corresponding loss per-example.
4. Resume: `graft_state_dict` loads with `strict=False`; a new head appears
   in `missing` → `fresh_params_introduced` → `warmup_on_arch_change` LR
   ramp (`trainer.py:4577-4590`). That makes flipping heads on mid-run
   *survivable*, but scalar→binned value is a head *replacement*, not an
   addition — the trunk loses its trained value signal at the swap. These
   features target from-scratch runs; no graft helper is planned (an
   expectation-matching binned graft is possible later if rel2 ever needs
   it).
5. `split_param_groups` (`model.py:878-924`) auto-handles new head
   biases/1-D params for weight decay — no change needed.

## Losses and logging

- `total_loss = policy_KL + value_loss(+bins) + horizon_loss_weight * Σ_k CE_k
  + q_loss_weight * Q_MSE (+ pc_loss_weight * PC)` — each term follows the
  `sw`-weighted pattern and lands in the `_forward_and_loss` return dict
  (`trainer.py:969-978`), accumulates through `train_step`/`train`, and logs
  as `loss/value_bins`, `loss/value_mse`, `loss/horizon_k{K}`, `loss/q_head`
  next to the existing scalars (`trainer.py:3107-3160`).
- Model introspection (`layer_analysis`, `trainer.py:4003-4269`): add weight
  norms for the new heads, useful during any later graft experiments.
- New TB histogram `self_play/hist/q_target` beside
  `self_play/hist/value_target` (`:3432-3433`) once HX08 lands.

## Tests

- Bin machinery: two-hot projection round-trip (project scalar → decode
  expectation == scalar, incl. exact-bin-center and ±1 edges); decoded
  scalar in [−1, 1].
- Model: `value_bins=0` forward bit-identical to current (regression);
  binned forward shape/compile smoke via the existing `_forward_batch_core`
  test pattern; `ScriptableHeXONet` parity eager-vs-scripted for the binned
  decode; `load_from_hexonet` maps binned params and whitelists
  horizon/Q heads.
- Ingest: hand-built HX07 game → horizon targets match a hand-derived table
  (terminal within k vs not, draw case, side-to-move sign); HX08
  writer/reader round-trip in Rust + Python (extend the existing format
  tests near `test_inference_server.py` / trainer reader tests); old-format
  files still parse with `q_targets=None`.
- Loss: each aux loss is exactly 0 (and absent from graphs) when its weight
  is 0 or its targets are None; Q loss masks `visits == 0`.
- End-to-end: tiny from-scratch run with all three enabled (W1 budget)
  through self-play → HX08 → ingest → train → checkpoint round-trip →
  eval, plus the full existing suite (`just test`) green with defaults.

## Rollout

Stage the flags independently so each earns its keep:

1. `value_bins = 65` alone vs lean-d6 baseline (A/B, same stages/seeds).
2. + `value_horizons = [4, 12, 32]`, `horizon_loss_weight` ~0.25 (tune).
3. + HX08 / `q_head = true`, `q_loss_weight` ~0.5 (tune) — requires the
   self-play binary and trainer to be deployed together.

Each stage gets its own config TOML with the house Hypothesis/Prediction
header, run dirs under `runs/gine-mini/<base>-vbins/`, `-vhoriz/`, `-qhead/`.

## Open questions

- Horizon target semantics: outcome-if-resolved-within-k (this doc) vs
  bootstrapped search value of the position k later. The latter is arguably
  a richer target but needs root-value persistence in the game format (could
  ride along in HX08) and defines away the clean "about to resolve" signal.
  Revisit after Stage 2 results.
- Moves-left head (ShrimpNet's 0..209-bin) — deliberately out of scope here;
  once the binned machinery exists it is a small follow-up if "prefer fast
  finishes when winning" becomes a priority.
- Whether the Q head should also learn from `per_child_prior` distillation
  (opponent-policy-style aux) — out of scope; keep the first version minimal.
