"""Configuration dataclasses for HeXO AlphaZero."""

from dataclasses import dataclass, field

# Shorthand for field with description metadata
def _f(default, desc: str, *, group: str = "", **kwargs):
    """Create a dataclass field with a description in metadata."""
    return field(default=default, metadata={"description": desc, "group": group}, **kwargs)

def _ff(factory, desc: str, *, group: str = ""):
    """Create a dataclass field with default_factory and description."""
    return field(default_factory=factory, metadata={"description": desc, "group": group})


@dataclass
class ModelConfig:
    """GNN model architecture — fixed across all curriculum stages."""
    # --- Backbone ---
    hidden_dim: int = _f(256, "Hidden dimensionality used throughout the GNN and attention layers", group="Backbone")
    num_layers: int = _f(3, "Number of GATv2Conv message-passing layers (0 = auto: win_length * 1.5)", group="Backbone")
    num_heads: int = _f(8, "Number of attention heads per layer (hidden_dim must be divisible by this)", group="Backbone")
    conv_type: str = _f("gine", "Convolution layer type: 'gatv2' (GATv2Conv, additive attention) or 'gine' (GINEConv, sum aggregation with edge injection, no attention — ~1.5x faster)", group="Backbone")
    pre_norm: bool = _f(True, "Use pre-norm residual blocks (norm before conv) vs post-norm (norm after)", group="Backbone")
    dropout: float = _f(0.0, "Dropout probability applied after each layer (0.0 = disabled)", group="Backbone")
    use_layer_scale: bool = _f(False, "Add a learnable per-channel LayerScale gate to each residual block's conv contribution. Default init=1.0 preserves no-LayerScale behaviour bit-for-bit; grafted layers init at 1e-4 (see graft_state_dict).", group="Backbone")
    use_jk: bool = _f(False, "Enable Jumping Knowledge aggregation over conv layers", group="Model")
    jk_mode: str = _f("sum", "JK aggregation mode. 'sum' (legacy): learned scalar softmax over layers (1 scalar per layer, shared across nodes/channels). 'cat': concatenate per-layer outputs (head input dim becomes L*hidden_dim). 'max': per-channel max-pool over layers. 'lstm': bi-LSTM attention over layers (PyG default). Only 'sum' is honored when use_jk=false (backwards compat with legacy field default).", group="Model")
    # --- Output heads ---
    policy_hidden: int = _f(128, "Hidden size of the policy head MLP", group="Output heads")
    value_hidden: int = _f(128, "Hidden size of the value head MLP", group="Output heads")
    value_bins: int = _f(0, "Distributional (categorical) value head: number of value bins spanning [value_bin_min, value_bin_max]. 0 = scalar tanh head (legacy, default). When >0 (>=2), the value head outputs a softmax over bins and the scalar is the softmax expectation (decode_binned_value); training uses a two-hot cross-entropy. An odd count puts a bin center exactly at the draw value. Suggested 65. FROM-SCRATCH ONLY (replaces the scalar head; no graft).", group="Output heads")
    value_bin_min: float = _f(-1.0, "Lower end of the distributional value head's bin grid (only used when value_bins>0).", group="Output heads")
    value_bin_max: float = _f(1.0, "Upper end of the distributional value head's bin grid (only used when value_bins>0).", group="Output heads")
    value_horizons: list[int] = _ff(list, "Short-horizon value heads (TRAIN-ONLY, never exported to self-play). A list of horizons in PLACEMENTS (a HeXO turn is 2 placements); each adds a binned value head trained on the position's outcome IF the game resolves within that many placements, else neutral (0). A denser value signal in long games. Suggested [4, 12, 32] (~2/6/16 turns). Requires value_bins>0 (all value heads share the bin grid). Empty = disabled. Weighted by training.horizon_loss_weight. FROM-SCRATCH ONLY.", group="Output heads")
    # --- Graph construction ---
    graph_type: str = _f("axis", "Graph topology: 'hex' (distance-1 adjacency, ~9 layers) or 'axis' (axis-window with edge features, ~3 layers)", group="Graph construction")
    prune_empty_edges: bool = _f(True, "Drop empty→empty edges in axis-window graphs (reduces graph size, only affects graph_type='axis')", group="Graph construction")
    threat_features: bool = _f(False, "Append 4 threat-encoding node features (own/opp best clean-window line count through the node, own/opp count of axes holding a win_length-2 line through the node), relative to the side to move. Node features become 12-dim; resuming an 8-dim checkpoint requires training.graft_threat_features.", group="Graph construction")
    relative_stone_encoding: bool = _f(False, "Side-to-move-relative stone encoding: node features become [own_stone, opp_stone, empty, moves_remaining, norm_q, norm_r, inv_stone_dist] (7-dim base; own = current player, the absolute to_move dim is dropped). Threat features still append after the base (11-dim with threat_features). FROM-SCRATCH RUNS ONLY: the own/opp remap is input-dependent, so existing absolute-encoding checkpoints cannot be grafted; incompatible with training.graft_threat_features.", group="Graph construction")
    # --- D6-invariant lean schema (FROM-SCRATCH RUNS ONLY; all default to the
    #     legacy schema so existing checkpoints load and evaluate unchanged) ---
    axis_relational: bool = _f(False, "Represent the 3 hex axes as an edge TYPE (edge_type in {0,1,2} + unsigned edge_dist) processed by a tied-weight AxisRelationalConv with a symmetric per-axis sum, instead of an absolute axis one-hot fed to GINE. Combined with an invariant node schema this makes the network EXACTLY D6-invariant by construction. Drops the edge src_player feature and the distance sign (both redundant). FROM-SCRATCH ONLY (edge representation + conv change).", group="Graph construction")
    axis_window: int = _f(8, "Max unsigned hop distance for the axis-relational per-hop distance embedding table (only used when axis_relational=true). Must be >= win_length - 1 across all stages; distances are clamped into [1, axis_window]. Decouples the fixed model architecture from per-stage win_length.", group="Graph construction")
    compact_stone_onehot: bool = _f(False, "Drop the redundant 'empty' stone one-hot dim (empty = neither own nor opp), shrinking the stone one-hot from 3 to 2. FROM-SCRATCH ONLY.", group="Graph construction")
    node_coords: bool = _f(True, "Include the normalised (q, r) node coordinates. These are a rotation-symmetry leak and largely redundant with the axis-edge geometry; set False for the D6-invariant lean schema. FROM-SCRATCH ONLY.", group="Graph construction")
    moves_scope: str = _f("node", "Where moves-remaining lives: 'node' (a per-node feature, legacy) or 'graph' (a single per-graph scalar injected in the model). Both are D6-invariant; 'graph' is a leaner representation. FROM-SCRATCH ONLY.", group="Graph construction")


def node_feature_dim(model_config) -> int:
    """Node-feature width implied by a ModelConfig (duck-typed; missing
    attributes default to the legacy absolute 8-dim layout).

    Valid widths: 7 (relative), 8 (absolute), 11 (relative + threat),
    12 (absolute + threat).
    """
    relative = getattr(model_config, "relative_stone_encoding", False)
    threat = getattr(model_config, "threat_features", False)
    base = 7 if relative else 8
    # Lean-schema subtractions (all default to legacy: no change).
    if getattr(model_config, "compact_stone_onehot", False):
        base -= 1  # drop the redundant 'empty' one-hot dim
    if not getattr(model_config, "node_coords", True):
        base -= 2  # drop norm_q, norm_r
    if getattr(model_config, "moves_scope", "node") == "graph":
        base -= 1  # moves-remaining becomes a per-graph scalar (injected in model)
    return base + (4 if threat else 0)


def legacy_lean_columns(model_config):
    """Legacy node-feature column indices to KEEP for the lean schema, in
    ascending (native-lean) order — the map a relational model uses to convert a
    legacy graph's node features to lean inputs on the fly (self-play / eval
    paths still emit legacy graphs). Returns ``None`` when no lean node-schema
    reduction applies (legacy == lean), so callers skip the index_select.
    """
    relative = getattr(model_config, "relative_stone_encoding", False)
    threat = getattr(model_config, "threat_features", False)
    compact = getattr(model_config, "compact_stone_onehot", False)
    coords = getattr(model_config, "node_coords", True)
    moves_graph = getattr(model_config, "moves_scope", "node") == "graph"
    if not (compact or not coords or moves_graph):
        return None  # no reduction — legacy layout already equals lean
    if relative:
        # [own, opp, empty, moves, norm_q, norm_r, inv_dist, threat*4]
        keep = [0, 1]
        if not compact:
            keep.append(2)
        if not moves_graph:
            keep.append(3)
        if coords:
            keep += [4, 5]
        keep.append(6)
        if threat:
            keep += [7, 8, 9, 10]
    else:
        # [P1, P2, empty, to_move, moves, norm_q, norm_r, inv_dist, threat*4]
        keep = [0, 1]
        if not compact:
            keep.append(2)
        keep.append(3)  # to_move retained (D6-invariant)
        if not moves_graph:
            keep.append(4)
        if coords:
            keep += [5, 6]
        keep.append(7)
        if threat:
            keep += [8, 9, 10, 11]
    return keep


def model_config_from_checkpoint(ckpt, fallback_dict=None) -> "ModelConfig":
    """Build a ModelConfig for a loaded checkpoint dict.

    Prefers the checkpoint's embedded ``model_config`` (the asdict of the
    ModelConfig the weights were trained under — authoritative for graph
    flags like ``threat_features`` / ``relative_stone_encoding``); falls back
    to ``fallback_dict`` (e.g. a TOML ``[model]`` section) when absent.
    Unknown keys are dropped so older/newer checkpoints stay loadable.
    """
    import dataclasses

    d = None
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_config"), dict):
        d = ckpt["model_config"]
    if d is None:
        d = dict(fallback_dict or {})
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in d.items() if k in known})


@dataclass
class GameConfig:
    """Game rules — overridable per curriculum stage."""
    win_length: int = _f(6, "Number of consecutive hexes needed to win (full HeXO = 6)")
    placement_radius: int = _f(8, "Maximum hex-distance from origin for new placements (full HeXO = 8)")
    max_moves: int = _f(200, "Game declared a draw after this many total placements")


@dataclass
class MCTSConfig:
    """Gumbel AlphaZero MCTS parameters (Danihelka et al. 2022).

    Each placement runs a Gumbel MCTS search that produces two outputs:
      1. The *improved policy* pi' = softmax(logits + sigma(completedQ))
         — a probability distribution over all legal placements, used as
         the training target for the policy network (Eq. 11-12).
      2. The *Gumbel-selected action* = argmax over sequential-halving
         survivors of g(a) + logits(a) + sigma(q_hat(a)), where g(a) is
         Gumbel(0) noise sampled once per search.

    During self-play, each placement is in one of two regimes:

      Exploration (first N placements): the played action is *sampled*
      from pi', so even low-probability placements occur proportionally.
      This is critical for discovering tactical moves (like blocking)
      that the current policy underweights.

      Post-exploration (remaining placements): the played action is the
      Gumbel-selected action. This is stochastic across games (fresh
      Gumbel noise each search) but biased toward the current policy via
      the logits term. Lower diversity than exploration-phase sampling.

    Both regimes record the same pi' as the training target — only the
    *played action* differs, not what the model trains on.

    The number of exploration placements is either fixed (exploration_moves)
    or adaptive (exploration_fraction * EMA game length), optionally scaling
    up when one side dominates (exploration_max).

    **Rust-only acting knobs**: ``truncate_delta`` and
    ``exploration_distribution`` are acting-time CLI flags consumed by the
    Rust self-play binary only. They are NOT fields of the Rust ``MCTSConfig``
    struct and do NOT affect eval or SPRT searches (which run their own MCTS
    directly without going through the self-play binary).
    """
    # --- Search ---
    n_simulations: int = _f(64, "Number of MCTS simulations (tree expansions) per placement. With m_actions candidates, sequential halving runs log2(m) phases distributing n simulations across survivors. n >= m * log2(m) gives full halving; n = m gives 1 sim per candidate (valid but weaker targets).", group="Search")
    m_actions: int = _f(16, "Number of candidate actions sampled via Gumbel-Top-k at the root. Should be close to the number of legal placements on small boards (e.g. m=32 for 36 legal) to avoid missing critical moves. The paper uses m = min(n, 16) for Go.", group="Search")
    c_scale: float = _f(1.0, "Q-value scaling factor in the completed-Q sigma formula: sigma(q) = (c_visit + max_visits) * c_scale * q. Paper default 1.0 works across Go, chess, and Atari.", group="Search")
    c_visit: int = _f(50, "Visit count baseline in the completed-Q sigma formula. Paper default 50; any c_visit >= 50 produces similar results.", group="Search")
    root_dirichlet_alpha: float = _f(0.0, "Dirichlet noise concentration alpha applied to root priors before Gumbel-Top-k candidate sampling. 0 disables (bit-equivalent to no-noise behaviour). AlphaZero heuristic alpha ~ 10 / branching_factor; for HeXO at full board (~200 legal moves) try ~0.05. Lower alpha = more concentrated noise; higher alpha = flatter/more uniform.", group="Search")
    root_dirichlet_fraction: float = _f(0.25, "Mix fraction epsilon for root Dirichlet noise: P(a) = (1-eps) * P_network(a) + eps * Dirichlet(alpha). Standard AlphaZero uses 0.25. Ignored when root_dirichlet_alpha == 0.", group="Search")
    # --- Exploration ---
    exploration_moves: int = _f(30, "Fixed number of opening placements in the exploration regime (1 placement = 1 stone, a full turn is 2 placements). Overridden by exploration_fraction if set.", group="Exploration")
    exploration_fraction: float = _f(0.25, "Fraction of EMA game length used as exploration placements (0 = use fixed exploration_moves). When exploration_max is also set, this becomes the base (minimum) fraction used when sides are balanced.", group="Exploration")
    exploration_max: float = _f(0.9, "Adaptive exploration ceiling. When > 0, exploration_fraction scales up toward this value as p1_decided_rate drifts from 0.5, using squared deviation for smooth center / steep edges: frac = base + (max - base) * (2 * |p1_rate - 0.5|)^2. 0 = disabled (fixed fraction).", group="Exploration")
    exploration_window: int = _f(2000, "EMA effective window (in games) for the p1 decided rate used by adaptive exploration. Larger = smoother, slower to react. alpha = 2/(window+1).", group="Exploration")
    exploration_exponent: float = _f(1.0, "Exponent for the adaptive exploration deviation curve. 1.0 = linear (responsive), 2.0 = squared (gentle center, steep edges). Applied as deviation^exponent.", group="Exploration")
    exploration_distribution: str = _f("improved_policy", "Distribution sampled during opening-exploration placements: 'improved_policy' (π', the σ-sharpened completed-Q policy — near one-hot at low sims) or 'visit_counts' (the Gumbel-AZ-paper acting distribution — the broad Sequential-Halving visit staircase). Ablation toggle; Rust self-play only.", group="Exploration")
    truncate_delta: float = _f(0.0, "Q-margin truncation gate for exploration acting (Rust self-play only). When > 0, in-window sampling zeroes candidates whose completed-Q falls more than this margin below the best candidate's — diverse-but-sound exploration. 0 = off. Validated offline at 0.25; see docs/research/2026-06-11-q-margin-truncated-exploration.md. The gate's soundness guarantee is strongest under exploration_distribution='visit_counts' (production): in legacy improved_policy mode, non-candidate prior mass is never gated and remains in the sampling distribution unchanged.", group="Exploration")


@dataclass
class TrainingConfig:
    """Gradient updates, learning rate schedule, loss functions, and replay buffer."""
    # --- Optimizer ---
    batch_size: int = _f(512, "Maximum number of examples per gradient step (may be smaller if edge_budget is set)", group="Optimizer")
    edge_budget: int = _f(500_000, "Target total edges per micro-batch for auto batch sizing (0 = use fixed batch_size)", group="Optimizer")
    grad_accumulation: bool = _f(True, "Accumulate gradients over micro-batches to maintain effective batch_size", group="Optimizer")
    lr: float = _f(2e-4, "Adam learning rate (overridable per curriculum stage)", group="Optimizer")
    weight_decay: float = _f(1e-4, "L2 regularisation coefficient via AdamW", group="Optimizer")
    layer_scale_weight_decay: float = _f(0.0, "Weight decay coefficient applied specifically to LayerScale gate parameters (only active when use_layer_scale=true). 0.0 = excluded from weight decay (default; preserves prior behavior). Non-zero values create a gentle pull toward zero, countering Adam's tendency to run LayerScale gates away from init under fresh-moments scaling. Suggested 1e-4 for graft experiments where the gate is being introduced.", group="Optimizer")
    max_grad_norm: float = _f(1.0, "Gradient clipping threshold (max L2 norm)", group="Optimizer")
    # --- LR schedule ---
    lr_schedule: str = _f("cosine", "Learning rate schedule: 'constant' or 'cosine'", group="LR schedule")
    lr_min: float = _f(2e-5, "Minimum learning rate floor for cosine decay", group="LR schedule")
    lr_warmup_steps: int = _f(400, "Number of steps for linear warmup from ~0 to lr (0 = no warmup)", group="LR schedule")
    total_train_steps: int = _f(5000, "Total cosine period in steps before reaching lr_min (0 = no limit)", group="LR schedule")
    warmup_on_arch_change: bool = _f(True, "When resuming a checkpoint that introduces fresh parameters (e.g. depth-extension graft, LayerScale added), reset the LR scheduler to step 0 instead of fast-forwarding past warmup. Adam's first-step normalization gives fresh params unit-magnitude updates regardless of init scale, so warmup is the load-bearing safety net for grafts. Set False to opt out (fast-forward through warmup as for regular resumes).", group="LR schedule")
    do_lr_warmup_on_resume: bool = _f(False, "Opt-in: on every resume (even without an arch change), apply a short linear LR ramp from `lr_min` up to the cosine schedule's LR at the resume step, then continue the cosine schedule. The ramp length is `lr_warmup_steps`. Mirrors the architecture-graft safety net (project_layerscale_graft_failure) for plain weight grafts (e.g. head-grafted JK-cat resumes) where freshly-initialised columns would otherwise get a full-magnitude Adam step at step 1 because their second-moment estimate is ~0. Default False so normal resumes are unaffected.", group="LR schedule")
    graft_heads_for_jk_cat: bool = _f(False, "On checkpoint resume only: detect heads whose first Linear has in_features=hidden_dim while the current model expects in_features=L*hidden_dim (i.e. switching jk_mode='cat' on top of an older no-JK / sum-mode checkpoint), and graft the old weights into the last hidden_dim columns of the new weight matrix (zero-init the rest, copy bias unchanged). At step 0 the forward pass is bit-identical to the no-JK model; gradients flow into the zeroed columns from step 1 onward. Requires use_jk=true and jk_mode='cat'. Default False (no implicit graft).", group="LR schedule")
    graft_threat_features: bool = _f(False, "On checkpoint load, widen an 8-feature input_proj to 12 with zero-init threat-feature columns (bit-identical forward at step 0). Requires model.threat_features=true. Pair with do_lr_warmup_on_resume=true.", group="LR schedule")
    # --- Throttle ---
    step_delay: float = _f(0.0, "Seconds to sleep between training steps (0 = disabled, frees GPU for self-play)", group="Throttle")
    target_reuse: float = _f(8.0, "Target sample reuse rate. 0 disables auto-throttle and uses fixed `step_delay`. Otherwise the trainer dynamically adjusts step_delay each reporting window to track this setpoint, computed from the closed-form inversion of the reuse formula.", group="Throttle")
    max_step_delay: float = _f(5.0, "Safety cap for the auto-throttle controller. If the computed step_delay would exceed this, the controller clamps and logs a warning that self-play is too slow to sustain the target reuse rate.", group="Throttle")
    step_delay_smoothing: float = _f(0.8, "EMA smoothing factor on the controller output. New delay = smoothing * old + (1 - smoothing) * computed. 0 = no smoothing, 0.95 = very smooth. Default 0.8 is a reasonable starting point.", group="Throttle")
    # --- Loss ---
    draw_value: float = _f(-0.3, "Value target for drawn games. 0.0 = neutral, -0.1 = slight penalty for both sides (encourages decisive play). Applied symmetrically: both players see this value for all positions in a drawn game.", group="Loss")
    pc_loss_weight: float = _f(0.0, "Weight for path consistency regularisation loss (0 = disabled)", group="Loss")
    horizon_loss_weight: float = _f(0.0, "Weight for the short-horizon value heads' aggregate cross-entropy (0 = disabled). Only active when model.value_horizons is non-empty. Suggested ~0.25.", group="Loss")
    # --- Data ---
    augment_symmetries: bool = _f(True, "Apply D6 symmetry augmentation (12x training data via hex rotations/reflections)", group="Data")
    buffer_capacity: int = _f(250_000, "Maximum number of positions stored in the replay buffer", group="Data")
    reanalyze_fraction: float = _f(0.0, "Fraction of each training batch to re-evaluate with the current model (0 = disabled)", group="Data")
    reanalyze_sims: int = _f(0, "MCTS simulations for reanalyze searches (0 = use mcts.n_simulations). Reanalyze refreshes targets, it does not act — a cheaper search (e.g. 32) is usually sufficient and cuts reanalyze GPU cost proportionally.", group="Data")
    reanalyze_m_actions: int = _f(0, "Gumbel top-K width for reanalyze searches (0 = use mcts.m_actions)", group="Data")
    reanalyze_chunk: int = _f(32, "Positions per lockstep batched-reanalyze call. Leaf evaluations fuse across the chunk each Sequential-Halving phase; ~32 keeps GPU batches near the measured batch-64 throughput knee.", group="Data")


@dataclass
class RustSelfPlayConfig:
    """Settings specific to the Rust self-play engine (batched inference server)."""
    # --- Batching ---
    max_batch: int = _f(128, "Maximum number of game positions batched into a single inference forward pass (graph-count cap; with max_batch_edges>0 this becomes a safety ceiling — set it high, e.g. 512)", group="Batching")
    max_batch_edges: int = _f(0, "Edge-targeted batching: accumulate requests until total graph edges reach this budget (~44-46k = the GPU wave-fill knee) instead of a fixed graph count, so batches stay efficient as graph size varies. 0 = off (graph-count only). Purely changes request coalescing — search/targets are byte-identical. See docs/research/2026-06-16-edge-targeted-batching.md", group="Batching")
    batch_timeout_ms: int = _f(5, "Milliseconds to wait for a full batch before running a partial one", group="Batching")
    # --- Inference ---
    padded_inference: bool = _f(True, "Use bucket-padded GNN inference (forward_graphs_padded) to stabilise the HIP/CUDA caching allocator under variable input shapes.", group="Inference")
    python_inference: bool = _f(True, "Use a Python subprocess with torch.compile for inference instead of TorchScript. ~2x faster forward pass but slower model reload (~30s).", group="Inference")
    python_inference_workers: int = _f(1, "Number of parallel Python inference subprocesses (each with its own batcher thread, round-robin request distribution). 1 = single subprocess (default; backward-compatible). 2+ hides per-batch CUDA event.synchronize() latency by overlapping forward passes on separate CUDA streams. Each worker holds an independent model copy on GPU; bump only if the GPU has memory headroom and the inference path is the measured bottleneck. Only honored when python_inference=true.", group="Inference")
    virtual_loss: float = _f(0.0, "Virtual-loss leaf-parallel MCTS within Sequential Halving phases. 0 disables (serial sims, one batch per phase iteration). >0 enables fusion (one batch per phase, with VL diversification between leaf selections). 2026-05-13 clean A/B: vl=0.5 gives +19% games/s vs serial. EFFECTIVELY BOOLEAN in Gumbel-AZ: the magnitude does NOTHING — only its sign (on/off) matters. Verified 2026-06-05 (docs/research/2026-06-05-vl-fidelity): fused-vs-serial fidelity is a step function, flat across vl 0.001..25. Why: Gumbel Eq.7 selection (argmax pi'(a) - N(a)/(1+sum_N)) diversifies via the magnitude-blind visit_count+=1, while the value-perturbation channel is washed out by min-max Q normalization (any negative injection floors the child's normalized Q to 0) + tiny sigma at low visits. Unlike PUCT, magnitude is not tunable here; just pick 0 or any positive value. Strength impact of fusion untested (SPRT-gate before adopting).", group="Inference")
    # --- Output ---
    max_file_mb: int = _f(2048, "Rotate output game files when they reach this size in MB", group="Output")
    output_filename: str = _f("games", "Base filename stem for game output files (e.g. games_0001.bin)", group="Output")
    # --- Past-self pool ---
    pool_fraction: float = _f(0.3, "Fraction of self-play games where one side is played by a randomly chosen past-self snapshot from the pool dir (0 = disabled)", group="Past-self pool")
    pool_size: int = _f(8, "Maximum number of past-self snapshots retained in the pool directory (oldest are deleted)", group="Past-self pool")
    pool_snapshot_every: int = _f(1, "Snapshot to the past-self pool every Nth model export (1 = every export). Use >1 to spread the pool across more training history when model updates are frequent.", group="Past-self pool")
    # --- Playout cap ---
    playout_cap_fraction: float = _f(0.75, "Wu (2019) playout cap randomization: fraction of placements that use a reduced simulation budget and skip writing a training example. 0.0 disables. Both this and playout_cap_divisor must be >default to activate.", group="Playout cap")
    playout_cap_divisor: int = _f(4, "Wu (2019) playout cap randomization: fast-search simulation budget = max(1, n_simulations // divisor). 1 disables. Both this and playout_cap_fraction must be >default to activate.", group="Playout cap")


@dataclass
class PythonSelfPlayConfig:
    """Settings specific to the Python self-play engine (in-process, alternating with training)."""
    games_per_write: int = _f(50, "Number of games generated per replay buffer write cycle")
    playout_cap_fraction: float = _f(0.75, "Fraction of placements that use a reduced simulation budget to speed up generation")
    playout_cap_divisor: int = _f(4, "Reduced simulation budget = n_simulations // divisor")
    use_rust_mcts: bool = _f(True, "Use the Rust MCTS backend for tree search (vs pure-Python fallback)")


@dataclass
class SelfPlayMctsOverride:
    """Optional `[self_play.mcts]` overrides of the `[mcts]` σ-transform constants
    for the data-generating search ONLY. Absent fields (None) inherit `[mcts]`.

    Decouples self-play from eval/SPRT: lowering ``c_scale`` here softens the
    distilled policy target (broadens the learned prior → better m-candidate
    coverage) without affecting eval/SPRT, which keep peaking the best move at the
    paper default. See docs/research/2026-06-04-cscale-target-reshape.md."""
    c_scale: float | None = _f(None, "Self-play-only σ scaling override; inherits [mcts].c_scale when unset", group="Search")
    c_visit: int | None = _f(None, "Self-play-only σ visit-baseline override; inherits [mcts].c_visit when unset", group="Search")


@dataclass
class ActorConfig:
    """Optional `[self_play.actor]` — run the Rust self-play binary on a SEPARATE
    GPU box (the "actor"), with training staying local (the "learner").

    OFF BY DEFAULT: when ``host`` is empty, self-play runs locally exactly as
    before (byte-identical command line). When ``host`` is set, the learner
    launches the self_play binary on ``host`` over SSH (Tailscale SSH is keyless;
    set it up once with ``tailscale up --ssh`` on the actor), rsync-pulls closed
    game files into the local batches dir (so the existing ingestion loop is
    unchanged), and rsync-pushes the self-play model on each champion export (the
    actor's periodic reload picks it up). This removes the single-GPU
    training↔self-play contention that dominates self-play throughput on a shared
    APU. See docs/research/2026-06-05-forward-pass-headroom/findings.md.

    The self_play binary is CPU-libtorch-only (no GPU linkage), so it does NOT
    need a CUDA rebuild — rsync the repo to the actor and point it at a
    version-matched ``torch==<pinned>+cuXX`` venv via ``python_bin``/``torch_lib``.
    """
    host: str = _f("", "SSH/tailnet host of the remote self-play actor. EMPTY = disabled (run self-play locally, default).", group="Actor")
    repo_dir: str = _f("", "Path to the hexo repo checkout on the actor host (the remote cwd; must mirror this repo's layout — rsync it). Required when host is set.", group="Actor")
    device: str = _f("cuda", "Device for the remote actor's self-play inference (the actor's GPU).", group="Actor")
    python_bin: str = _f("", "Path to the actor's Python (a torch==<pinned>+cuXX venv) for the inference_server. EMPTY = '<repo_dir>/.venv/bin/python'.", group="Actor")
    torch_lib: str = _f("", "Path to the actor venv's torch/lib, prepended to LD_LIBRARY_PATH so the CPU-only self_play binary binds the actor's (CUDA) libtorch. EMPTY = '<python_bin venv>/lib/.../site-packages/torch/lib' is derived at launch.", group="Actor")
    ssh_opts: str = _f("-o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o BatchMode=yes", "Extra ssh options for the control connection (ConnectTimeout + keepalive so a dead/asleep actor is detected fast; the -tt forced-tty cleanup is added automatically). On disconnect the remote self-play is auto-restarted with backoff.", group="Actor")
    games_pull_interval: float = _f(1.0, "Seconds between rsync pulls of CLOSED (rotated) game files from the actor into the local batches dir. Lower rust.max_file_mb for lower ingest latency.", group="Actor")
    rsync_bin: str = _f("rsync", "rsync executable used for games-pull and model-push (must exist on both hosts).", group="Actor")


@dataclass
class SelfPlayConfig:
    """Self-play engine selection and shared settings across both engines."""
    engine: str = _f("rust", "Which self-play engine to use: 'rust' (background binary) or 'python' (in-process)")
    device: str = _f("cuda", "Device for self-play inference ('cpu' or 'cuda')")
    workers: int = _f(16, "Number of parallel self-play game threads (0 = auto-detect CPU count)")
    precision: str = _f("bf16", "Model precision for self-play inference: 'fp32', 'fp16', or 'bf16'")
    source: str = _f("auto", "Which model produces self-play games: 'champion' (frozen SPRT-validated checkpoint, AGZ-style; self-play only updates on promotion), 'trainee' (live training model with periodic re-exports, canonical AZ / Lc0 / KataGo style), or 'auto' (deprecated; legacy derivation from convergence_mode — champion if mode is 'champion'/'sprt' else trainee). 'champion' requires convergence_mode in ('champion', 'sprt').")
    model_update_interval: int = _f(0, "Export updated model for self-play every N training steps (0 = at each checkpoint). Ignored when self-play source resolves to 'champion' (self-play updates only on champion promotion).")
    rust: RustSelfPlayConfig = _ff(RustSelfPlayConfig, "Rust engine settings")
    python: PythonSelfPlayConfig = _ff(PythonSelfPlayConfig, "Python engine settings")
    mcts: SelfPlayMctsOverride = _ff(SelfPlayMctsOverride, "Self-play-only MCTS σ-constant overrides (inherit [mcts] when unset)")
    actor: ActorConfig = _ff(ActorConfig, "Optional remote self-play actor (off by default; empty host = run self-play locally)")


@dataclass
class SealbotConfig:
    """Adaptive SealBot evaluation — plays against a C++ minimax opponent at increasing difficulty."""
    # --- Schedule ---
    games: int = _f(20, "Number of games per SealBot evaluation round (0 = disabled)", group="Schedule")
    sims: int = _f(256, "MCTS simulations per placement during SealBot eval (0 = use raw policy)", group="Schedule")
    m_actions: int = _f(16, "Candidate actions for MCTS during SealBot eval", group="Schedule")
    # --- Difficulty ladder ---
    adaptive: bool = _f(True, "Automatically promote through difficulty levels when winning consistently", group="Difficulty ladder")
    time_limit: float = _f(0.5, "SealBot's minimax time limit per placement in seconds", group="Difficulty ladder")
    levels: list[float] = _ff(lambda: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], "Difficulty ladder: list of time limits from easiest to hardest", group="Difficulty ladder")
    promote_threshold: float = _f(0.7, "Win rate required to promote to the next difficulty level", group="Difficulty ladder")
    promote_patience: int = _f(2, "Number of consecutive evals above threshold before promoting", group="Difficulty ladder")
    # --- Game overrides ---
    win_length: int = _f(6, "Override win_length for SealBot games (0 = use current stage's game config)", group="Game overrides")
    placement_radius: int = _f(8, "Override placement_radius for SealBot games (0 = use current stage's game config)", group="Game overrides")
    max_moves: int = _f(1000, "Override max_moves for SealBot games (0 = use current stage's game config)", group="Game overrides")
    # --- Parallelism ---
    workers: int = _f(1, "Number of worker processes for SealBot eval. 1 = sequential, >1 = multiprocessing.Pool with spawn context. Each worker loads its own model copy on CPU. Recommended: 4–8 for n_games >= 30.", group="Parallelism")


@dataclass
class KrakenbotConfig:
    """KrakenBot evaluation via the hosted 6-tac.com API (proof-of-concept).

    Defaults to a single game. The endpoint is operated by mteam88 (Modal
    compute on their tab); do not raise ``games`` for batched runs without
    upstream permission. Strength is fixed upstream at KRAKEN_N_SIMS=200.
    """
    games: int = _f(1, "Number of games per KrakenBot evaluation round (0 = disabled). Keep at 1 until upstream operator OKs batched runs.")
    sims: int = _f(256, "Hexo-side MCTS simulations per placement during KrakenBot eval (0 = use raw policy)")
    m_actions: int = _f(16, "Candidate actions for hexo's MCTS during KrakenBot eval")
    api_url: str = _f("https://6-tac.com/api/v1/compute/best-move", "KrakenBot API endpoint. Override for a self-hosted six-tac/Modal deployment.")
    timeout: float = _f(60.0, "Per-request timeout in seconds. Cold-start on the upstream Modal container is ~11s; warm calls ~1-2s.")


@dataclass
class CorpusEvalConfig:
    """Human-corpus eval per champion promotion — an absolute, stage-independent
    full-board yardstick (policy top-k / value calibration on the 6,902-game
    human corpus). Runs `scripts/eval_human_corpus.py` as a niced background
    subprocess after each SPRT promotion; summary scalars land in TB under
    `corpus/*`. The slope of these vs step measures transferable (radius-8)
    value extracted from the current stage, complementing in-stage Elo signals.
    Caveat: policy perplexity inverts for sharp/strong nets (NLL punishes
    confident disagreement) — read top16 + value ECE/sign_acc as the strength
    signals, perplexity as a sharpness diagnostic."""
    games: int = _f(0, "Corpus games per champion-promotion eval (0 = disabled; >= corpus size = full corpus). Sampling is seeded and fixed, so successive evals are comparable.", group="Schedule")
    device: str = _f("cpu", "Device for the corpus eval subprocess. 'cpu' keeps it off the training GPU (full corpus ~20 min niced CPU; GPU ~6 min but contends with training + self-play).", group="Schedule")


@dataclass
class EvalConfig:
    """Evaluation schedule, checkpointing, and opponent settings."""
    # --- Schedule ---
    interval: int = _f(1000, "Run evaluation every N training steps (0 = disabled)", group="Schedule")
    checkpoint_interval: int = _f(300, "Save a model checkpoint every N training steps", group="Schedule")
    model_analysis_interval: int = _f(0, "Run model introspection (layer_analysis/* and edge_features/* TB scalars) every N training steps. 0 = inherit `interval` (default; backwards-compatible). Set lower than `interval` for higher-resolution monitoring during graft experiments — the analysis itself is cheap (~100ms) so this is not cost-bound.", group="Schedule")
    # --- Games ---
    games: int = _f(50, "Number of games per evaluation round (vs random + vs previous checkpoint)", group="Games")
    sims: int = _f(256, "MCTS simulations per placement during eval games (0 = use raw policy, faster)", group="Games")
    m_actions: int = _f(16, "Candidate actions for MCTS during eval games", group="Games")
    # --- Opponent ---
    sealbot: SealbotConfig = _ff(SealbotConfig, "SealBot evaluation settings", group="Opponent")
    corpus: CorpusEvalConfig = _ff(CorpusEvalConfig, "Human-corpus eval per champion promotion", group="Opponent")


@dataclass
class RunConfig:
    """Runtime environment — devices, directories, and compilation settings."""
    device: str = _f("cuda", "PyTorch device for training ('cpu', 'cuda', 'cuda:0', etc.)")
    checkpoint_dir: str = _f("checkpoints", "Directory for saving model checkpoints")
    log_dir: str = _f("runs", "TensorBoard log directory")
    compile: bool = _f(True, "Enable torch.compile for faster training and inference")
    fp16: bool = _f(True, "Use bf16 mixed precision for training (reduces memory, may improve speed)")
    seed: int | None = _f(None, "RNG seed for reproducibility (None = non-deterministic)")


@dataclass
class CurriculumConfig:
    """Multi-stage curriculum — automatically advances through progressively harder game variants."""
    # --- Convergence detection ---
    convergence_mode: str = _f("champion", "How to detect convergence: 'champion' (vs champion checkpoint, fixed-N eval), 'sprt' (vs champion checkpoint, continuous SPRT daemon), 'checkpoint' (vs lagged checkpoint), 'random' (vs random opponent), or 'none' (manual only)", group="Convergence detection")
    patience: int = _f(10, "Random: consecutive evals above threshold to advance. Checkpoint: rolling window size for plateau detection. Champion: consecutive non-promotion evals before declaring plateau (reset on each promotion).", group="Convergence detection")
    min_steps_per_stage: int = _f(10_000, "Minimum training steps before convergence checks begin (warmup period)", group="Convergence detection")
    confirm_advance: bool = _f(True, "When convergence is detected, pause and wait for a CONFIRM file in the checkpoint dir before advancing. Training continues while waiting.", group="Convergence detection")
    resume_from_champion_on_advance: bool = _f(True, "At fresh stage transitions, override trainee weights with the previous stage's champion snapshot. Champion is the most recently SPRT-validated state; any post-promotion trainee progress is unvalidated by construction. Especially helpful at degenerate stages (e.g. small-board endgames where the trainee unlearns win-finding skills converging to drawn play). Adam moments are zeroed when this fires.", group="Convergence detection")
    # --- Mode: random ---
    win_rate_threshold: float = _f(0.85, "Win rate vs random opponent that counts as converged (mode=random)", group="Mode: random")
    # --- Mode: checkpoint ---
    improve_threshold: float = _f(0.55, "Mean win rate vs lagged checkpoint below which the model is considered plateaued (mode=checkpoint). Averaged over the last `patience` evaluations.", group="Mode: checkpoint")
    checkpoint_lag: int = _f(30, "Compare against the checkpoint from N saves ago for convergence detection", group="Mode: checkpoint")
    # --- Mode: champion ---
    champion_threshold: float = _f(0.55, "Win rate vs champion to promote trainee to new champion. If mean win rate stays below this for `patience` evals, stage is considered plateaued and auto-advances (mode=champion).", group="Mode: champion")
    # --- Mode: sprt ---
    # Continuous SPRT daemon runs a separate eval subprocess that plays games
    # between the latest trainee checkpoint and the current champion, tests
    # H0 (score = sprt_s0) vs H1 (score = sprt_s1) via Wald's SPRT, and
    # promotes on `accept_h1` / advances the stage on `reject_h1`.
    sprt_s0: float = _f(0.50, "Null hypothesis (trainee not stronger than champion) per-game score (mode=sprt).", group="Mode: sprt")
    sprt_s1: float = _f(0.55, "Alternative hypothesis (trainee stronger, worth promoting) per-game score (mode=sprt).", group="Mode: sprt")
    sprt_alpha: float = _f(0.05, "Per-round SPRT type-I error rate — probability of promoting when trainee is not actually stronger (mode=sprt). Ignored if sprt_alpha_stage > 0.", group="Mode: sprt")
    sprt_beta: float = _f(0.05, "Per-round SPRT type-II error rate — probability of failing to promote when trainee is actually stronger (mode=sprt). Ignored if sprt_beta_stage > 0.", group="Mode: sprt")
    sprt_alpha_stage: float = _f(0.0, "Desired stage-level false-promotion rate across sprt_reject_patience rounds. When > 0, overrides sprt_alpha with 1 - (1 - sprt_alpha_stage)^(1/patience). Example: sprt_alpha_stage=0.05 with patience=3 → sprt_alpha ≈ 0.017 per round (mode=sprt).", group="Mode: sprt")
    sprt_beta_stage: float = _f(0.0, "Desired stage-level false-plateau rate across sprt_reject_patience rounds. When > 0, overrides sprt_beta with sprt_beta_stage^(1/patience). Example: sprt_beta_stage=0.05 with patience=3 → sprt_beta ≈ 0.368 per round (mode=sprt).", group="Mode: sprt")
    sprt_window_size: int = _f(1000, "Sliding window of recent games for SPRT state; older games are dropped. Must exceed the ~412-game accept horizon at s1, else a stronger trainee plateaus below the accept bound forever (dead band); 1000 ≈ 2.5× that horizon. 0 = unbounded accumulation (mode=sprt).", group="Mode: sprt")
    sprt_mcts_sims: int = _f(16, "MCTS simulations per placement during SPRT eval games. 0 = raw policy (deterministic; breaks SPRT) (mode=sprt).", group="Mode: sprt")
    sprt_mcts_m_actions: int = _f(16, "Root candidate actions for SPRT eval MCTS (mode=sprt).", group="Mode: sprt")
    sprt_device: str = _f("cpu", "Device for SPRT daemon inference. 'cpu' keeps the daemon off the training GPU (mode=sprt).", group="Mode: sprt")
    sprt_eval_workers: int = _f(1, "Parallel SPRT pair-eval worker processes. 1 = inline serial play (unchanged). >1 spawns CPU workers that each play a whole P1+P2 pair concurrently, exploiting cores freed when self-play moves to a remote actor. Clamped to 1 unless sprt_device='cpu' (>1 on GPU = the occupancy-contention disaster) (mode=sprt).", group="Mode: sprt")
    sprt_poll_interval: float = _f(2.0, "Seconds between SPRT state-file polls by the trainer (mode=sprt).", group="Mode: sprt")
    sprt_reject_patience: int = _f(3, "Consecutive SPRT rounds terminating in reject_h1 (since current champion was promoted) before declaring stage plateau and advancing. Each reject resets SPRT state and starts a new round against the same champion, so this controls how many independent rejections constitute plateau. Also the bootstrap floor for the auto-calibrated warn/decl thresholds and the round count N in the sprt_alpha_stage/sprt_beta_stage derivation. <= 0 disables the reject-patience plateau detector entirely (reject counts are still tracked for the CLI display and TB; prefer this over the old 9999 idiom) (mode=sprt).", group="Mode: sprt")
    sprt_promotion_grace_steps: int = _f(1000, "Training steps after a champion promotion (or stage start) during which reject_h1 rounds do not count toward sprt_reject_patience. Gives the trainee time to pull ahead of the just-promoted champion (which IS the trainee from a moment ago) before plateau is declared. 0 = no grace period (mode=sprt).", group="Mode: sprt")
    sprt_pentanomial: bool = _f(True, "Pair consecutive P1/P2 games and use 5-bucket pair scores as the statistical unit. Cancels first-player advantage — useful for games with significant side asymmetry like HeXO. False = classic per-game trinomial SPRT (mode=sprt).", group="Mode: sprt")
    sprt_pair_variance: float = _f(0.35, "Per-pair score variance for pentanomial LLR. Pair scores are in [0,2]; independent-games upper bound is 2*variance=0.5, but actual pair variance is typically 0.30–0.40 due to P1/P2 correlation. Lower values speed up convergence; 0.35 is the chess-engine-testing default. Initial value when sprt_adaptive_pair_variance is enabled (mode=sprt).", group="Mode: sprt")
    sprt_adaptive_pair_variance: bool = _f(False, "Adaptively track pair variance via EMA of round-end empirical estimates, clamped to [floor, ceil]. Resets on champion change. Removes manual stage-tuning burden but introduces a feedback loop — if disabled, the configured sprt_pair_variance is used throughout (mode=sprt).", group="Mode: sprt")
    sprt_adaptive_pair_variance_floor: float = _f(0.20, "Minimum pair variance when adaptive tracking is on. Prevents over-confident accepts on lucky low-variance rounds (mode=sprt).", group="Mode: sprt")
    sprt_adaptive_pair_variance_ceil: float = _f(0.50, "Maximum pair variance when adaptive tracking is on. Independent-games upper bound (mode=sprt).", group="Mode: sprt")
    sprt_adaptive_pair_variance_ema_alpha: float = _f(0.30, "EMA weight on the latest empirical observation when updating pair variance. Lower = more smoothing, slower adaptation (mode=sprt).", group="Mode: sprt")
    sprt_prev_ckpt_patience: int = _f(0, "Secondary plateau detector: pool the last N eval/win_rate_vs_prev_checkpoint evals (N × eval.games games) and declare plateau when the one-sided Wilson upper bound on the pooled win rate is confidently below sprt_prev_ckpt_band_high. Catches the failure mode where SPRT's auto-calibrated decl_thr keeps rising past what's achievable while the model has actually stopped improving. 0 = disabled (SPRT-only plateau detection). Suggested 8-10 (mode=sprt).", group="Mode: sprt")
    sprt_prev_ckpt_band_low: float = _f(0.45, "Regression guard for the prev-checkpoint plateau detector: a pooled mean below this indicates regression (a separate signal, not plateau) and withholds the plateau declaration (mode=sprt).", group="Mode: sprt")
    sprt_prev_ckpt_band_high: float = _f(0.55, "Improvement threshold for the prev-checkpoint plateau detector, in win-rate-vs-lagged-checkpoint units (an Elo-velocity threshold over checkpoint_lag × eval.checkpoint_interval steps). Plateau fires only when the pooled Wilson upper bound is below this AND the freshest eval is not itself above it (recency guard: a window whose latest reading clears the band is mid-recovery — e.g. post-LR-reset — not plateaued) (mode=sprt).", group="Mode: sprt")
    sprt_promo_velocity_factor: float = _f(0.0, "Promotion-velocity plateau detector: declare stage plateau when steps-since-last-promotion exceeds factor × median(promotion intervals this stage). Each SPRT promotion is an alpha-certified improvement, so intervals are certified-Elo-velocity quanta — much higher SNR than counting variable-length reject rounds, and median-based so one slow cycle can't ratchet the threshold the way peak_history max does. 0 = disabled; the sprt/steps_since_promotion and sprt/promo_velocity_ratio TB scalars are emitted regardless, for shadow-mode calibration. Suggested 3.0 (mode=sprt).", group="Mode: sprt")
    sprt_promo_velocity_min_intervals: int = _f(3, "Completed promotion intervals required (this stage; persisted across restarts via <stage_log_dir>/detector_state.json, discarded on stage mismatch or rolled-back step counters) before the promotion-velocity detector arms (mode=sprt).", group="Mode: sprt")
    # --- Stages ---
    stages: list[dict] = _ff(list, "Per-stage override dicts from [[curriculum.stages]]", group="Stages")


@dataclass
class FullConfig:
    """Top-level container holding all configuration sub-objects."""
    model: ModelConfig = _ff(ModelConfig, "Model architecture")
    game: GameConfig = _ff(GameConfig, "Game rules")
    mcts: MCTSConfig = _ff(MCTSConfig, "MCTS search parameters")
    training: TrainingConfig = _ff(TrainingConfig, "Training parameters")
    self_play: SelfPlayConfig = _ff(SelfPlayConfig, "Self-play engine")
    eval: EvalConfig = _ff(EvalConfig, "Evaluation settings")
    run: RunConfig = _ff(RunConfig, "Runtime settings")
    curriculum: CurriculumConfig | None = _f(None, "Curriculum settings (optional)")
