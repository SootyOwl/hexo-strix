"""Evaluation: play the current model against baselines to measure strength.

Provides fast greedy evaluation (no MCTS) against a random opponent,
used to track training progress.
"""

from __future__ import annotations

import logging
import random

import torch
from torch_geometric.data import Batch

from hexo_a0.graph import axis_states_to_batch, game_to_graph, game_to_axis_graph

logger = logging.getLogger(__name__)


def _graph_fn_for(model_config):
    """Build the ``game -> PyG Data`` builder for a model's config (hex or axis)."""
    graph_type = model_config.graph_type if model_config else "hex"
    prune = model_config.prune_empty_edges if model_config else False
    threat = getattr(model_config, "threat_features", False) if model_config else False
    rel = getattr(model_config, "relative_stone_encoding", False) if model_config else False
    if graph_type == "axis":
        return lambda g: game_to_axis_graph(
            g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
    return lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel)


def sample_opening(model, game_config, device, k, temperature, seed,
                   model_config=None, max_resamples=20):
    """Sample a ``k``-ply opening from ``model``'s raw policy with temperature.

    Diversity source for noise-off SPRT: the first ``k`` plies are sampled (so
    paired games start from varied positions), after which play is
    deterministic. Fully determined by ``seed`` (seeds the global torch RNG per
    attempt). Guarantees the returned opening leaves the game **non-terminal** —
    if an attempt reaches a terminal state within ``k`` plies (a win, or a
    double-four instant loss) it is rejected and resampled with a derived seed.
    Raises ``RuntimeError`` if no non-terminal opening is found within
    ``max_resamples`` attempts.

    Returns a list of ``k`` ``(q, r)`` coords (replayable via ``apply_move``).
    """
    import hexo_rs

    graph_fn = _graph_fn_for(model_config)
    for attempt in range(max_resamples):
        # Derived per-attempt seed keeps the result deterministic in `seed`
        # while letting rejected attempts explore a different line.
        torch.manual_seed(seed * 1_000_003 + attempt)
        game = hexo_rs.GameState(game_config)
        opening: list[tuple[int, int]] = []
        for _ in range(k):
            if game.is_terminal():
                break
            move = _model_move(model, game, device,
                               temperature=temperature, graph_fn=graph_fn)
            game.apply_move(move[0], move[1])
            opening.append((int(move[0]), int(move[1])))
        if len(opening) == k and not game.is_terminal():
            return opening
    raise RuntimeError(
        f"sample_opening: no non-terminal {k}-ply opening after {max_resamples} "
        f"attempts (seed={seed}, temperature={temperature})"
    )


def play_eval_game(
    model: torch.nn.Module,
    game_config: object,
    device: torch.device,
    opponent: str | torch.nn.Module = "random",
    model_side: str = "P1",
    temperature: float = 0.0,
    model_config: object = None,
    opponent_config: object = None,
    mcts_sims: int = 0,
    mcts_m_actions: int = 16,
    opponent_mcts_sims: int | None = None,
    opponent_mcts_m_actions: int | None = None,
    mcts_c_visit: int = 50,
    mcts_c_scale: float = 1.0,
    opponent_mcts_c_visit: int | None = None,
    opponent_mcts_c_scale: float | None = None,
    mcts_forced_candidate_capture_k: int = 0,
    opponent_mcts_forced_candidate_capture_k: int = 0,
    mcts_virtual_loss: float = 0.0,
    opponent_mcts_virtual_loss: float = 0.0,
    opening: list[tuple[int, int]] | None = None,
    disable_gumbel_noise: bool = False,
) -> dict:
    """Play one complete evaluation game: model vs opponent.

    Two action-selection paths:
      * ``mcts_sims > 0``: run a full Gumbel MCTS search via the Rust
        backend and play ``argmax`` of the improved policy — the
        deterministic "no exploration" eval mode from the Gumbel AZ paper.
      * ``mcts_sims == 0``: raw policy logits. ``temperature == 0`` is
        greedy argmax; ``temperature > 0`` samples. Faster but weaker —
        kept as a cheap fallback.

    Args:
        model:       HeXONet in eval mode.
        game_config: hexo_rs.GameConfig.
        device:      Torch device for inference.
        opponent:    ``"random"`` for uniform random, or another nn.Module.
        model_side:  ``"P1"`` or ``"P2"`` — which side the model plays.
        mcts_sims:   If > 0, run MCTS with this many simulations per move.
        mcts_m_actions: Root candidate actions for Gumbel-Top-k.
        opponent_mcts_sims: Per-side sim override for the opponent;
            falls back to ``mcts_sims`` when ``None``.
        opponent_mcts_m_actions: Per-side Gumbel-Top-k width override for
            the opponent; falls back to ``mcts_m_actions`` when ``None``.
            Together with ``opponent_mcts_sims`` this lets one network play
            itself at two different search-config operating points.
        mcts_c_visit / mcts_c_scale: σ-transform constants for the model
            side (paper defaults 50 / 1.0). σ(q̂) = (c_visit + max N)·c_scale·q̂.
        opponent_mcts_c_visit / opponent_mcts_c_scale: per-side overrides for
            the opponent; fall back to the model-side values when ``None``.
        mcts_forced_candidate_capture_k: E.4-inject capture K for the
            ``model`` side. When > 0 the model's MCTS at mr=2 roots
            additionally batch-evals post-p₁ states and exposes the
            chosen p₁'s top-K p₂s; on the next (mr=1) move these are
            passed as ``forced_candidates`` so they always appear in
            the p₂ search's candidate set. 0 disables (current behaviour).
        opponent_mcts_forced_candidate_capture_k: same for the opponent.

    Returns:
        Dict with keys ``winner``, ``moves``, ``model_side``.
    """
    import hexo_rs

    # Select graph builder based on model_config.graph_type
    graph_type = model_config.graph_type if model_config else "hex"
    prune = model_config.prune_empty_edges if model_config else False
    threat = getattr(model_config, "threat_features", False) if model_config else False
    rel = getattr(model_config, "relative_stone_encoding", False) if model_config else False
    if graph_type == "axis":
        graph_fn = lambda g: game_to_axis_graph(g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
    else:
        graph_fn = lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel)

    # Separate graph builder for the opponent if its config differs
    opp_graph_type = opponent_config.graph_type if opponent_config else graph_type
    opp_prune = opponent_config.prune_empty_edges if opponent_config else False
    opp_threat = getattr(opponent_config, "threat_features", False) if opponent_config else False
    opp_rel = getattr(opponent_config, "relative_stone_encoding", False) if opponent_config else False
    if opp_graph_type == "axis":
        opp_graph_fn = lambda g: game_to_axis_graph(g, prune_empty_edges=opp_prune, threat_features=opp_threat, relative_stones=opp_rel)
    else:
        opp_graph_fn = lambda g: game_to_graph(g, threat_features=opp_threat, relative_stones=opp_rel)

    eval_fn = make_eval_fn(
        model, device, graph_type=graph_type, prune_empty_edges=prune,
        threat_features=threat, relative_stones=rel, graph_fn=graph_fn,
    )
    opp_eval_fn = (
        make_eval_fn(
            opponent, device, graph_type=opp_graph_type, prune_empty_edges=opp_prune,
            threat_features=opp_threat, relative_stones=opp_rel, graph_fn=opp_graph_fn,
        )
        if isinstance(opponent, torch.nn.Module) else None
    )

    def _make_cfg(sims: int, m_actions: int, c_visit: int, c_scale: float,
                  capture_k: int = 0, vl: float = 0.0):
        if sims <= 0:
            return None
        return hexo_rs.MCTSConfig(
            n_simulations=sims, m_actions=m_actions,
            c_visit=c_visit, c_scale=c_scale,
            virtual_loss=vl,
            forced_candidate_capture_k=capture_k,
            disable_gumbel_noise=disable_gumbel_noise,
        )

    mcts_config = _make_cfg(
        mcts_sims, mcts_m_actions, mcts_c_visit, mcts_c_scale,
        mcts_forced_candidate_capture_k, mcts_virtual_loss,
    )
    opp_sims = mcts_sims if opponent_mcts_sims is None else opponent_mcts_sims
    opp_m_actions = (
        mcts_m_actions if opponent_mcts_m_actions is None else opponent_mcts_m_actions
    )
    opp_c_visit = mcts_c_visit if opponent_mcts_c_visit is None else opponent_mcts_c_visit
    opp_c_scale = mcts_c_scale if opponent_mcts_c_scale is None else opponent_mcts_c_scale
    opp_mcts_config = _make_cfg(
        opp_sims, opp_m_actions, opp_c_visit, opp_c_scale,
        opponent_mcts_forced_candidate_capture_k, opponent_mcts_virtual_loss,
    )

    game = hexo_rs.GameState(game_config)
    moves = 0
    move_log: list[tuple[int, int]] = []

    # Forced opening (diversity source for noise-off SPRT): replay the given
    # plies before either engine starts choosing. These are setup, not engine
    # decisions, but count toward the move log / length.
    if opening:
        for (q, r) in opening:
            game.apply_move(q, r)
            move_log.append((int(q), int(r)))
            moves += 1

    # E.4-inject per-side state: list of p₂ coords captured at the most
    # recent mr=2 root for each side. Read at the next mr=1 root; cleared
    # after use. Tracked per-side because P1's pending capture is
    # irrelevant to P2 (different player's mid-turn state).
    pending_forced: dict[str, list[tuple[int, int]]] = {"P1": [], "P2": []}

    while not game.is_terminal():
        current_player = game.current_player()

        if current_player == model_side:
            action = _choose_move(
                model, game, device, graph_fn, temperature, mcts_config,
                pending_forced=pending_forced[current_player], eval_fn=eval_fn,
            )
            _update_pending_forced(
                pending_forced, current_player, game, mcts_config, action,
            )
        elif opponent == "random":
            action = _random_move(game)
        elif isinstance(opponent, torch.nn.Module):
            action = _choose_move(
                opponent, game, device, opp_graph_fn, temperature, opp_mcts_config,
                pending_forced=pending_forced[current_player], eval_fn=opp_eval_fn,
            )
            _update_pending_forced(
                pending_forced, current_player, game, opp_mcts_config, action,
            )
        else:
            raise ValueError(f"Unknown opponent type: {opponent!r}")

        game.apply_move(action[0], action[1])
        move_log.append((int(action[0]), int(action[1])))
        moves += 1

    winner = game.winner()
    return {"winner": winner, "moves": moves, "model_side": model_side,
            "move_log": move_log}


def _update_pending_forced(
    pending_forced: dict,
    player: str,
    game_at_root: object,
    mcts_config,
    action: tuple[int, int],
) -> None:
    """Refresh the per-side pending forced list after a MCTS move.

    Called *before* `apply_move` so we can read the just-completed root's
    `moves_remaining_this_turn()` and tell whether the search we just ran
    was at mr=2 (capture happens here) or mr=1 (consume + clear).

    No-op when MCTS isn't active (``mcts_config is None``) — pending
    state is only updated for paths that actually invoked the search.
    The actual captured coords are set by `_mcts_argmax_move` via the
    module-level `_LAST_CAPTURE` cell (a thread-local-ish stash to avoid
    threading additional return values through `_choose_move`).
    """
    if mcts_config is None:
        return
    mr = game_at_root.moves_remaining_this_turn()
    if mr == 2:
        # Just ran the p₁ search; pick up any captured p₂s and store
        # for the upcoming mr=1 (p₂) call.
        pending_forced[player] = list(_LAST_CAPTURE)
    else:
        # mr=1: the forced list (if any) was consumed by this call;
        # clear for the next turn.
        pending_forced[player] = []
    _LAST_CAPTURE.clear()


# Module-level stash for the most-recent MCTS call's
# `chosen_action_forced_candidates`. Read by `_update_pending_forced`
# immediately after `_choose_move` returns to avoid threading an extra
# return value through every call site. Cleared after each consumption.
# Single-threaded by construction (eval games are sequential).
_LAST_CAPTURE: list[tuple[int, int]] = []


def make_eval_fn(
    model: torch.nn.Module,
    device: torch.device,
    *,
    graph_type: str,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
    graph_fn,
):
    """Build a batched ``states -> (logits_list, values_list)`` fn for gumbel_mcts.

    Axis-graph models collate leaf batches in Rust and feed
    ``_forward_batch_core`` directly (one frombuffer memcpy per field,
    Rust-precomputed index tensors). Hex models keep the per-graph
    ``graph_fn`` + PyG-collation path.
    """
    if graph_type == "axis":
        def eval_fn(states):
            batch, aux = axis_states_to_batch(
                states, prune_empty_edges=prune_empty_edges,
                threat_features=threat_features, relative_stones=relative_stones,
                device=device,
            )
            with torch.inference_mode():
                logits, counts, values = model._forward_batch_core(
                    batch, legal_idx=aux.legal_idx,
                    stone_idx=aux.stone_idx, stone_batch=aux.stone_batch,
                )
            flat = logits.tolist()
            logits_list, off = [], 0
            for c in counts.tolist():
                logits_list.append(flat[off:off + c])
                off += c
            return (logits_list, values.tolist())
    else:
        def eval_fn(states):
            data_list = [graph_fn(s) for s in states]
            batch = Batch.from_data_list(data_list).to(device)
            with torch.inference_mode():
                policy_logits_list, values = model.forward_batch(batch)
            logits_list = [logits.tolist() for logits in policy_logits_list]
            values_list = [float(v.item()) for v in values]
            return (logits_list, values_list)
    return eval_fn


def _choose_move(
    model: torch.nn.Module,
    game: object,
    device: torch.device,
    graph_fn,
    temperature: float,
    mcts_config,
    pending_forced: list[tuple[int, int]] | None = None,
    eval_fn=None,
) -> tuple[int, int]:
    """Route to MCTS argmax-of-improved-policy or raw-logits selection.

    ``pending_forced``: E.4-inject forced-candidates list to pass to
    Gumbel MCTS. ``None`` or empty skips the inject (default behaviour).
    Only relevant for the MCTS branch; ignored for raw-policy moves.
    """
    if mcts_config is not None:
        return _mcts_argmax_move(
            model, game, device, graph_fn, mcts_config,
            pending_forced=pending_forced, eval_fn=eval_fn,
        )
    return _model_move(model, game, device, temperature=temperature, graph_fn=graph_fn)


def _mcts_argmax_move(
    model: torch.nn.Module,
    game: object,
    device: torch.device,
    graph_fn,
    mcts_config,
    pending_forced: list[tuple[int, int]] | None = None,
    eval_fn=None,
) -> tuple[int, int]:
    """Run Gumbel MCTS and return ``legal_moves[argmax(improved_policy)]``.

    This is the deterministic eval action per the Gumbel AZ paper: the
    search itself uses Gumbel-Top-k at the root (for candidate sampling
    during sequential halving), but the *played* move is argmax of the
    improved policy rather than the Gumbel-sampled action.

    Uses ``gumbel_mcts_with_diagnostics`` so the
    ``chosen_action_forced_candidates`` capture output is observable.
    Stashes the captured coords in module-level ``_LAST_CAPTURE`` for
    the caller's post-move bookkeeping.
    """
    import hexo_rs

    if eval_fn is None:
        # Legacy per-call construction (kept for direct callers/tests that
        # pass only a graph_fn — e.g. hex-graph paths).
        def eval_fn(states):
            data_list = [graph_fn(s) for s in states]
            batch = Batch.from_data_list(data_list).to(device)
            with torch.inference_mode():
                policy_logits_list, values = model.forward_batch(batch)
            logits_list = [logits.tolist() for logits in policy_logits_list]
            values_list = [float(v.item()) for v in values]
            return (logits_list, values_list)

    forced = pending_forced if pending_forced else None
    (
        _action, improved_policy, _visit_counts,
        _per_child_q, _per_child_prior, _candidate_indices,
        chosen_action_forced_candidates,
    ) = hexo_rs.gumbel_mcts_with_diagnostics(
        game, eval_fn, mcts_config, forced_candidates=forced,
    )
    # Stash the captured coords for `_update_pending_forced` to read.
    _LAST_CAPTURE.clear()
    _LAST_CAPTURE.extend(tuple(c) for c in chosen_action_forced_candidates)

    legal_moves = game.legal_moves()
    best_idx = max(range(len(improved_policy)), key=lambda i: improved_policy[i])
    return legal_moves[best_idx]


def _model_move(
    model: torch.nn.Module,
    game: object,
    device: torch.device,
    temperature: float = 0.0,
    graph_fn=None,
) -> tuple[int, int]:
    """Select a move using the model's policy logits.

    Args:
        temperature: 0.0 = greedy argmax (deterministic).
                     >0 = sample from softmax(logits/temperature).
        graph_fn: Graph builder function. Defaults to game_to_graph.
    """
    if graph_fn is None:
        graph_fn = game_to_graph
    data = graph_fn(game)
    batch = Batch.from_data_list([data]).to(device)

    with torch.inference_mode():
        policy_logits_list, _values = model.forward_batch(batch)

    logits = policy_logits_list[0]

    if temperature <= 0.0:
        best_idx = int(logits.argmax().item())
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        best_idx = int(torch.multinomial(probs, 1).item())

    legal_moves = game.legal_moves()
    return legal_moves[best_idx]


def _random_move(game: object) -> tuple[int, int]:
    """Pick uniformly from legal moves."""
    return random.choice(game.legal_moves())


def evaluate_vs_random(
    model: torch.nn.Module,
    game_config: object,
    device: torch.device,
    n_games: int = 20,
    model_config: object = None,
) -> dict:
    """Evaluate the model against a random opponent using the raw policy head.

    Plays half the games as P1, half as P2. Deliberately bypasses MCTS:
    this is a vibe check on the policy head itself, not on search+policy
    strength. MCTS-vs-random saturates at 100% almost immediately and
    stops being an informative signal; raw-policy-vs-random is a much
    harder bar to clear because a single policy-head gap is enough for
    random play to stumble into and punish.

    For MCTS-backed strength comparisons, use ``evaluate_vs_checkpoint``
    (both sides run MCTS) or the sealbot eval.

    Args:
        model:       HeXONet (will be set to eval mode).
        game_config: hexo_rs.GameConfig.
        device:      Torch device for inference.
        n_games:     Total number of games to play.

    Returns:
        Dict with ``win_rate``, ``wins``, ``losses``, ``draws``,
        ``mean_game_length``, plus per-side ``p1_wins``/``p2_wins`` and
        ``p1_decided_rate``.
    """
    model.eval()

    wins = 0
    losses = 0
    draws = 0
    p1_wins = 0
    p2_wins = 0
    total_moves = 0

    games_as_p1 = n_games // 2

    for i in range(n_games):
        side = "P1" if i < games_as_p1 else "P2"
        result = play_eval_game(
            model, game_config, device, opponent="random", model_side=side,
            model_config=model_config,
            mcts_sims=0,
        )
        total_moves += result["moves"]

        if result["winner"] is None:
            draws += 1
        elif result["winner"] == side:
            wins += 1
        else:
            losses += 1

        if result["winner"] == "P1":
            p1_wins += 1
        elif result["winner"] == "P2":
            p2_wins += 1

    decided = p1_wins + p2_wins
    return {
        # vs random: draws count as losses — the model should always win.
        "win_rate": wins / n_games if n_games > 0 else 0.0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "p1_decided_rate": p1_wins / decided if decided > 0 else 0.0,
        "mean_game_length": total_moves / n_games if n_games > 0 else 0.0,
    }


def evaluate_vs_checkpoint(
    model: torch.nn.Module,
    opponent_path: str,
    model_config: object,
    game_config: object,
    device: torch.device,
    n_games: int = 20,
    mcts_sims: int = 0,
    mcts_m_actions: int = 16,
) -> dict:
    """Evaluate the current model against a previous checkpoint.

    Loads the opponent from a .pt file, plays n_games (half as P1, half as P2).

    Args:
        model:        Current model (will be set to eval mode).
        opponent_path: Path to opponent checkpoint .pt file.
        model_config:  ModelConfig for constructing the opponent model.
        game_config:   hexo_rs.GameConfig.
        device:        Torch device.
        n_games:       Total games to play.

    Returns:
        Dict with ``win_rate``, ``wins``, ``losses``, ``draws``,
        ``mean_game_length``, ``opponent_path``.
    """
    from hexo_a0.model import HeXONet
    from hexo_a0.config import ModelConfig

    # Load opponent — reconstruct with graph_type from checkpoint
    opp_mc = model_config
    try:
        ckpt_peek = torch.load(opponent_path, map_location="cpu", weights_only=True)
        ckpt_mc_dict = ckpt_peek.get("model_config", {})
        if ckpt_mc_dict:
            opp_mc = ModelConfig(**ckpt_mc_dict)
    except Exception:
        pass  # Fall back to caller's model_config
    opponent = HeXONet(opp_mc).to(device)
    try:
        checkpoint = torch.load(opponent_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(opponent_path, map_location=device, weights_only=False)
    # Strip _orig_mod. prefixes from compiled-model checkpoints so they
    # load correctly into the uncompiled opponent.
    ckpt_sd = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model_state_dict"].items()}
    opponent.load_state_dict(ckpt_sd, strict=False)
    opponent.eval()
    model.eval()

    wins = 0
    losses = 0
    draws = 0
    p1_wins = 0
    p2_wins = 0
    total_moves = 0

    games_as_p1 = n_games // 2

    for i in range(n_games):
        side = "P1" if i < games_as_p1 else "P2"
        result = play_eval_game(
            model, game_config, device, opponent=opponent, model_side=side,
            model_config=model_config, opponent_config=opp_mc,
            mcts_sims=mcts_sims, mcts_m_actions=mcts_m_actions,
        )
        total_moves += result["moves"]

        if result["winner"] is None:
            draws += 1
        elif result["winner"] == side:
            wins += 1
        else:
            losses += 1

        if result["winner"] == "P1":
            p1_wins += 1
        elif result["winner"] == "P2":
            p2_wins += 1

    decided = p1_wins + p2_wins
    return {
        "win_rate": (wins + 0.5 * draws) / n_games if n_games > 0 else 0.0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "p1_decided_rate": p1_wins / decided if decided > 0 else 0.0,
        "mean_game_length": total_moves / n_games if n_games > 0 else 0.0,
        "opponent_path": opponent_path,
    }
