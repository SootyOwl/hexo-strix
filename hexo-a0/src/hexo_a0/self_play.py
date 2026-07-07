"""Self-play game generation for HeXO AlphaZero.

Implements:
  - TrainingExample: dataclass holding graph, policy target, value target (REQ-13)
  - self_play_game: play one game via Gumbel MCTS self-play (REQ-12)
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from hexo_a0.config import FullConfig, MCTSConfig as PyMCTSConfig, node_feature_dim
from hexo_a0.gpu_memory import CacheClearGate
from hexo_a0.graph import game_to_graph, game_to_axis_graph, game_to_graph_batch, game_to_axis_graph_batch
from hexo_a0.mcts import gumbel_mcts as python_gumbel_mcts

logger = logging.getLogger(__name__)


def _rust_gumbel_mcts(
    game: object,
    network: "torch.nn.Module",
    mcts: PyMCTSConfig,
    device: "torch.device",
    *,
    use_fp16: bool = False,
    seed: int | None = None,
    graph_fn=None,
    graph_batch_fn=None,
    clear_gate: "CacheClearGate | None" = None,
) -> tuple[tuple[int, int], torch.Tensor]:
    """Run Gumbel MCTS using the Rust backend.

    ``clear_gate`` is a persistent :class:`CacheClearGate` shared across the
    moves of a game so its cadence counter is not reset every move. When None
    a fresh per-call gate is created (fine for one-off callers).
    """
    import hexo_rs
    from torch_geometric.data import Batch

    if graph_fn is None:
        graph_fn = game_to_graph
    if graph_batch_fn is None:
        graph_batch_fn = game_to_graph_batch

    mcts_config = hexo_rs.MCTSConfig(
        n_simulations=mcts.n_simulations,
        m_actions=mcts.m_actions,
        c_visit=mcts.c_visit,
        c_scale=mcts.c_scale,
        root_dirichlet_alpha=getattr(mcts, "root_dirichlet_alpha", 0.0),
        root_dirichlet_fraction=getattr(mcts, "root_dirichlet_fraction", 0.0),
    )

    _stream = torch.cuda.Stream(device) if device.type == "cuda" else None
    _use_fp16 = use_fp16 and device.type == "cuda"
    # Clear the caching allocator only every Nth forward (cadence gate) instead
    # of every call; PYTORCH_CUDA_ALLOC_CONF (set at process entry) is the real
    # cross-process OOM safety net. See hexo_a0.gpu_memory. The gate is shared
    # across moves by self_play_game so its counter spans the whole game.
    _clear_gate = clear_gate if clear_gate is not None else CacheClearGate.for_device(device)

    def eval_fn(states):
        data_list = graph_batch_fn(states) if len(states) > 1 else [graph_fn(states[0])]

        stream_ctx = torch.cuda.stream(_stream) if _stream else nullcontext()
        with stream_ctx:
            batch = Batch.from_data_list(data_list).to(device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=_use_fp16):
                policy_logits_list, values = network.forward_batch(batch)

        if _stream:
            _stream.synchronize()

        logits_list = [logits_j.tolist() for logits_j in policy_logits_list]
        values_list = [float(v.item()) for v in values]
        del batch, data_list, policy_logits_list, values
        _clear_gate.step()
        return (logits_list, values_list)

    action, policy_list = hexo_rs.gumbel_mcts(game, eval_fn, mcts_config, seed=seed)
    return action, torch.tensor(policy_list, dtype=torch.float32)


def _rust_batched_gumbel_mcts(
    states: list,  # list of hexo_rs.GameState — kept untyped, PyO3 class
    network: "torch.nn.Module",
    mcts: PyMCTSConfig,
    device: "torch.device",
    *,
    n_simulations: int | None = None,
    m_actions: int | None = None,
    seed: int | None = None,
    graph_fn=None,
    graph_batch_fn=None,
    clear_gate: "CacheClearGate | None" = None,
) -> list[torch.Tensor]:
    """Run Gumbel MCTS on many positions in lockstep (fused leaf evaluation).

    Used by trainer reanalyze: one search per state, leaf evaluations batched
    ACROSS the whole chunk per Sequential-Halving phase, so GPU batches are
    chunk-sized rather than single-search-sized. Returns one improved-policy
    tensor per state, ordered like ``state.legal_moves()`` — the same
    ordering as ``TrainingExample.policy_target``.

    ``n_simulations``/``m_actions`` override the values in ``mcts`` so
    reanalyze can use a cheaper search than acting self-play. All states
    must be non-terminal.

    Reanalyze does not need within-search fusion (``virtual_loss``) or
    E.4-inject (``forced_candidate_capture_k``); both stay at their PyO3
    defaults (0).
    """
    import hexo_rs
    from torch_geometric.data import Batch

    if graph_fn is None:
        graph_fn = game_to_graph
    if graph_batch_fn is None:
        graph_batch_fn = game_to_graph_batch

    mcts_config = hexo_rs.MCTSConfig(
        n_simulations=n_simulations if n_simulations is not None else mcts.n_simulations,
        m_actions=m_actions if m_actions is not None else mcts.m_actions,
        c_visit=mcts.c_visit,
        c_scale=mcts.c_scale,
        root_dirichlet_alpha=getattr(mcts, "root_dirichlet_alpha", 0.0),
        root_dirichlet_fraction=getattr(mcts, "root_dirichlet_fraction", 0.0),
    )
    _clear_gate = clear_gate if clear_gate is not None else CacheClearGate.for_device(device)

    def eval_fn(sts):
        data_list = graph_batch_fn(sts) if len(sts) > 1 else [graph_fn(sts[0])]
        batch = Batch.from_data_list(data_list).to(device)
        with torch.inference_mode():
            policy_logits_list, values = network.forward_batch(batch)
        logits_list = [logits_j.tolist() for logits_j in policy_logits_list]
        values_list = [float(v.item()) for v in values]
        del batch, data_list, policy_logits_list, values
        _clear_gate.step()
        return (logits_list, values_list)

    results = hexo_rs.batched_gumbel_mcts(states, eval_fn, mcts_config, seed=seed)
    return [torch.tensor(policy, dtype=torch.float32) for _action, policy in results]


def batched_self_play_games(
    network: "torch.nn.Module",
    config: FullConfig,
    game_config: object,
    device: "torch.device",
    n_games: int,
    model_config: object = None,  # deprecated, use config.model
    n_simulations_override: int | None = None,
) -> list[list[TrainingExample]]:
    """Play N games via Rust batched self-play.

    All games run sequentially in Rust sharing one eval_fn, avoiding
    Python threading overhead. Returns a list of game trajectories.
    """
    import hexo_rs
    from torch_geometric.data import Batch

    mcts = config.mcts
    sp = config.self_play
    mc = config.model
    run = config.run

    # Select graph builder based on model config graph_type
    graph_type = mc.graph_type
    prune = mc.prune_empty_edges
    threat = getattr(mc, "threat_features", False)
    rel = getattr(mc, "relative_stone_encoding", False)
    if graph_type == "axis":
        _graph_fn = lambda g: game_to_axis_graph(g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
        _graph_batch_fn = lambda gs: game_to_axis_graph_batch(gs, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
    else:
        _graph_fn = lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel)
        _graph_batch_fn = lambda gs: game_to_graph_batch(gs, threat_features=threat, relative_stones=rel)

    network.eval()

    # Set up scriptable model for raw tensor path (hex mode only)
    _use_raw = graph_type == "hex"
    _node_dim = node_feature_dim(mc)
    scriptable_net = None
    if _use_raw:
        from hexo_a0.scriptable_model import ScriptableHeXONet, load_from_hexonet
        scriptable_net = ScriptableHeXONet(
            node_features=node_feature_dim(mc),
            hidden_dim=mc.hidden_dim,
            num_layers=mc.num_layers,
            num_heads=mc.num_heads,
            policy_hidden=mc.policy_hidden,
            value_hidden=mc.value_hidden,
            graph_type="hex",
            # Distributional value head (Stage 1): the scriptable model must be
            # built with the same value_bins as HeXONet or load_from_hexonet
            # hits a value_mlp size mismatch. Horizon/Q heads are train-only
            # and never enter the scriptable inference path.
            value_bins=int(getattr(mc, "value_bins", 0) or 0),
            value_bin_min=getattr(mc, "value_bin_min", -1.0),
            value_bin_max=getattr(mc, "value_bin_max", 1.0),
        )
        load_from_hexonet(scriptable_net, network.state_dict())
        scriptable_net.to(device)
        scriptable_net.eval()

    n_sims = n_simulations_override if n_simulations_override is not None else mcts.n_simulations
    mcts_config = hexo_rs.MCTSConfig(
        n_simulations=n_sims,
        m_actions=mcts.m_actions,
        c_visit=mcts.c_visit,
        c_scale=mcts.c_scale,
        root_dirichlet_alpha=getattr(mcts, "root_dirichlet_alpha", 0.0),
        root_dirichlet_fraction=getattr(mcts, "root_dirichlet_fraction", 0.0),
    )

    _stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    _empty_ea = torch.zeros(0, dtype=torch.float32).to(device) if _use_raw else None
    _use_fp16 = sp.precision in ("fp16", "bf16") and device.type == "cuda"
    # Clear the caching allocator on a cadence (every Nth forward) rather than
    # every call. The allocator config set at process entry is the real OOM
    # safety net. See hexo_a0.gpu_memory.
    _clear_gate = CacheClearGate.for_device(device)

    def eval_fn(states):
        if _use_raw and scriptable_net is not None:
            (features, edge_src, edge_dst, legal_mask_flat, stone_mask_flat,
             batch_vec, num_graphs, legal_counts,
             legal_idx_flat, stone_idx_flat, stone_batch_flat,
             ) = hexo_rs.game_states_to_batch(
                states, threat_features=threat, relative_stones=rel,
            )

            x = torch.tensor(features, dtype=torch.float32).view(-1, _node_dim).to(device)
            edge_index = torch.stack([
                torch.tensor(edge_src, dtype=torch.long),
                torch.tensor(edge_dst, dtype=torch.long),
            ]).to(device)
            lm = torch.tensor(legal_mask_flat, dtype=torch.bool).to(device)
            sm = torch.tensor(stone_mask_flat, dtype=torch.bool).to(device)
            bv = torch.tensor(batch_vec, dtype=torch.long).to(device)
            li = torch.tensor(legal_idx_flat, dtype=torch.long).to(device)
            si = torch.tensor(stone_idx_flat, dtype=torch.long).to(device)
            sb = torch.tensor(stone_batch_flat, dtype=torch.long).to(device)

            stream_ctx = torch.cuda.stream(_stream) if _stream else nullcontext()
            with stream_ctx:
                with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=_use_fp16):
                    all_logits, lc, values = scriptable_net(
                        x, edge_index, lm, sm, bv, num_graphs, _empty_ea,
                        li, si, sb,
                    )
            if _stream:
                _stream.synchronize()

            # Split all_logits by legal_counts
            logits_list = []
            offset = 0
            for count in legal_counts:
                logits_list.append(all_logits[offset:offset + count].tolist())
                offset += count

            result = (logits_list, values.tolist())
            del x, edge_index, lm, sm, bv, all_logits, lc, values
            _clear_gate.step()
            return result
        else:
            # Axis mode: use PyG path
            data_list = _graph_batch_fn(states) if len(states) > 1 else [_graph_fn(states[0])]
            stream_ctx = torch.cuda.stream(_stream) if _stream else nullcontext()
            with stream_ctx:
                batch = Batch.from_data_list(data_list).to(device)
                with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=_use_fp16):
                    policy_logits_list, values = network.forward_batch(batch)
            if _stream:
                _stream.synchronize()
            result = (
                [lg.tolist() for lg in policy_logits_list],
                [float(v.item()) for v in values],
            )
            del batch, data_list, policy_logits_list, values
            _clear_gate.step()
            return result

    exploration_moves = mcts.exploration_moves
    seed = run.seed

    # Rust plays all N games, returns trajectories
    raw_games = hexo_rs.batched_self_play(
        game_config, eval_fn, mcts_config,
        n_games, exploration_moves,
        playout_cap_fraction=sp.python.playout_cap_fraction,
        playout_cap_divisor=sp.python.playout_cap_divisor,
        seed=seed,
    )

    # Convert to TrainingExamples
    all_examples = []
    for traj_idx, (raw_trajectory, winner) in enumerate(raw_games):
        examples = []
        for (snapshot, _action, policy, current_player) in raw_trajectory:
            if winner is None:
                value_target = config.training.draw_value
            elif winner == current_player:
                value_target = 1.0
            else:
                value_target = -1.0
            examples.append(TrainingExample(
                policy_target=torch.tensor(policy, dtype=torch.float32),
                value_target=value_target,
                game_state=snapshot,
                trajectory_id=traj_idx,
            ))
        all_examples.append(examples)

    return all_examples



# ---------------------------------------------------------------------------
# TrainingExample (REQ-13)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TrainingExample:
    """A single training example produced by self-play.

    Attributes:
        data:          PyG Data object from game_to_graph() for the position.
        policy_target: 1-D tensor of length = num legal moves.  This is the
                       improved policy pi' returned by gumbel_mcts, normalised
                       to sum to 1.  Ordering aligns with data.coords[legal_mask].
        value_target:  Scalar outcome relative to the player to move at this
                       position: +1 (win), -1 (loss), 0 (draw).
        sample_weight: Per-example loss weight in (0, 1]. 1.0 for full-cap
                       MCTS searches; 1/playout_cap_divisor for fast-cap
                       searches under PCR. Reflects the relative trust in
                       the improved-policy target as a function of the
                       search budget that produced it. Applied
                       multiplicatively in the trainer's loss.
    """

    data: Data | None = None  # reconstructed at load from game_state; not persisted for new examples
    policy_target: torch.Tensor
    value_target: float
    game_state: object | None = None  # hexo_rs.GameState, preserved for reanalyze
    trajectory_id: int | None = None
    sample_weight: float = 1.0
    # Short-horizon value targets (Stage 2), one float per model.value_horizons
    # entry, in that column order; None when horizon heads are disabled. Built
    # at ingest time (within-trajectory order is not persisted downstream).
    horizon_targets: list[float] | None = None
    # Per-move Q head data (Stage 3), one entry per legal move (policy order):
    # q_targets = MCTS completed-Q, q_visits = visit counts (0 = unsearched).
    # None unless the source data is HX08. The trainer masks the Q loss to
    # visits>0 and weights by sample_weight.
    q_targets: list[float] | None = None
    q_visits: list[int] | None = None

    def __post_init__(self):
        if self.data is None and self.game_state is None:
            raise ValueError(
                "TrainingExample requires data or game_state (a reader must be "
                "able to obtain/reconstruct a graph); both are None")


# ---------------------------------------------------------------------------
# self_play_game (REQ-12)
# ---------------------------------------------------------------------------


def self_play_game(
    network: "torch.nn.Module",
    config: FullConfig,
    game_config: object,
    device: "torch.device",
) -> list[TrainingExample]:
    """Play one full game via Gumbel MCTS self-play.

    Args:
        network:     HeXONet (set to eval mode by this function).
        config:      FullConfig with mcts, self_play, model, run sub-configs.
        game_config: hexo_rs.GameConfig describing board parameters.
        device:      Torch device for inference.

    Returns:
        list[TrainingExample] — one entry per non-terminal position visited.
    """
    import hexo_rs  # local import; hexo_rs is a native extension

    mcts = config.mcts
    sp = config.self_play
    mc = config.model
    run = config.run

    # Select graph builder based on model config graph_type
    graph_type = mc.graph_type
    prune = mc.prune_empty_edges
    threat = getattr(mc, "threat_features", False)
    rel = getattr(mc, "relative_stone_encoding", False)
    if graph_type == "axis":
        graph_fn = lambda g: game_to_axis_graph(g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
        graph_batch_fn = lambda gs: game_to_axis_graph_batch(gs, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
    else:
        graph_fn = lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel)
        graph_batch_fn = lambda gs: game_to_graph_batch(gs, threat_features=threat, relative_stones=rel)

    _use_fp16 = sp.precision in ("fp16", "bf16") and device.type == "cuda"

    # Ensure eval mode (REQ-12, network must be eval during inference)
    network.eval()

    game = hexo_rs.GameState(game_config)

    # One cache-clear gate per game, shared across all moves so its cadence
    # counter spans the whole game instead of resetting every move.
    clear_gate = CacheClearGate.for_device(device)

    # trajectory: (state_snapshot, improved_policy, current_player_at_snapshot)
    trajectory: list[tuple[object, torch.Tensor, str]] = []
    move_count = 0

    while not game.is_terminal():
        state_snapshot = game.clone()
        current_player = game.current_player()

        logger.debug("  move %d: %s to play, %d legal moves",
                      move_count, current_player, len(game.legal_moves()))

        # Run Gumbel MCTS to get the deterministic best action and improved policy
        if sp.python.use_rust_mcts:
            action, improved_policy = _rust_gumbel_mcts(
                game, network, mcts, device,
                use_fp16=_use_fp16, seed=run.seed,
                graph_fn=graph_fn, graph_batch_fn=graph_batch_fn,
                clear_gate=clear_gate,
            )
        else:
            action, improved_policy = python_gumbel_mcts(
                game, network, mcts, device,
                graph_fn=graph_fn,
            )

        # Exploration: during the first exploration_moves placements, sample an
        # action from the improved policy rather than using the deterministic
        # MCTS-selected action.  pi' already incorporates the MCTS signal, so
        # this is a principled stochastic exploration strategy.
        if move_count < mcts.exploration_moves:
            # Clamp to avoid multinomial issues with near-zero probs
            safe_policy = improved_policy.clamp(min=0.0)
            if safe_policy.sum().item() > 0:
                action_idx = int(torch.multinomial(safe_policy, 1).item())
                coords = sorted(game.legal_moves())
                action = coords[action_idx]

        trajectory.append((state_snapshot, improved_policy, current_player))
        game.apply_move(action[0], action[1])
        move_count += 1
        logger.debug("  move %d: placed at (%d, %d)", move_count, action[0], action[1])

    # Determine the winner from the terminal state
    winner = game.winner()  # "P1", "P2", or None (draw)
    logger.debug("  game finished: %d moves, winner=%s", move_count, winner)

    # Build TrainingExamples from the recorded trajectory
    examples: list[TrainingExample] = []
    for state_snapshot, improved_policy, cp in trajectory:
        if winner is None:
            value_target = 0.0
        elif winner == cp:
            value_target = 1.0
        else:
            value_target = -1.0

        examples.append(
            TrainingExample(
                policy_target=improved_policy,
                value_target=value_target,
                game_state=state_snapshot,
                trajectory_id=0,
            )
        )

    return examples
