"""One-shot SPRT match between two arbitrary checkpoints.

Plays games between two fixed checkpoints until the SPRT terminates
(accept_h1 / reject_h1) or ``max_games`` is exhausted. Each checkpoint
carries its own ``model_config`` and is reconstructed independently —
the two sides may differ in every model hyperparameter (graph_type,
conv_type, hidden_dim, num_layers, use_jk, ...). The only shared piece
is the ``GameConfig`` (board size, win condition, max moves) supplied
via CLI args.

Cross-graph-type support: ``hexo_a0.evaluate.play_eval_game`` already
threads the per-side ``model_config``/``opponent_config`` through
``_choose_move`` and builds an independent graph for each side, so
this harness does not need its own adapter — two checkpoints with
different ``graph_type`` (e.g. "hex" vs "axis") will play correctly.

Hypotheses are framed from A's perspective:
  H0: score(A vs B) = sprt_s0  (typically 0.50, equal strength)
  H1: score(A vs B) = sprt_s1  (typically 0.55, ~35 Elo advantage)

Pentanomial pairs are formed from consecutive (A-as-P1, A-as-P2)
games against B, matching the daemon. Sides alternate by game index:
``game_idx % 2 == 0`` → A plays P1.

This is one-shot per invocation. Resumption from a prior ``--state-file``
is intentionally out of scope (the daemon handles the long-running
case); the state file here is monitoring-only.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from hexo_a0.config import ModelConfig
from hexo_a0.evaluate import play_eval_game, sample_opening
from hexo_a0.model import HeXONet
from hexo_a0.sealbot_eval import _win_rate_stats
from hexo_a0.sprt_eval import SPRTConfig, SPRTState, sample_pair_variance


# Default hex-distance from origin past which a P2 opening stone counts as
# "away" (not contesting the centre). Used by the refuted-opening detector.
_REFUTED_FAR_DEFAULT = 3


def _hexdist(q: int, r: int) -> int:
    """Axial hex distance from the origin (0, 0)."""
    return (abs(q) + abs(r) + abs(q + r)) // 2


def _percentile(sorted_vals, q):
    """Nearest-rank percentile of a pre-sorted list (q in [0,1])."""
    if not sorted_vals:
        return 0.0
    idx = int(round(q * (len(sorted_vals) - 1)))
    return float(sorted_vals[min(len(sorted_vals) - 1, max(0, idx))])


def _is_aligned(s1, s2) -> bool:
    """True iff two cells share one of the three hex win-axes.

    Axes in axial coords: same r (axis (1,0)), same q (axis (0,1)), or
    constant q+r (axis (1,-1)).
    """
    (q1, r1), (q2, r2) = s1, s2
    return r1 == r2 or q1 == q2 or (q1 + r1) == (q2 + r2)


def _is_threat_pair(s1, s2, win_length) -> bool:
    """True iff two stones form a real *preemptive* — aligned AND close enough
    to co-occur in a single ``win_length`` line (separation <= win_length-1),
    so the pair can grow into a threat. Aligned-but-too-far-apart is not a
    preemptive (it can't become one line).
    """
    (q1, r1), (q2, r2) = s1, s2
    return _is_aligned(s1, s2) and _hexdist(q1 - q2, r1 - r2) <= win_length - 1


def _p2_opening_stats(opening, far_threshold=_REFUTED_FAR_DEFAULT, win_length=6):
    """Opening-quality signal for P2's first turn.

    HeXO seeds P1 at the origin (consumes no move-count), so the first two
    sampled plies (``opening[:2]``) are P2's opening turn. Per HeXO opening
    theory the *refuted* P2 opening is **two stones both played away from the
    origin that don't form a preemptive** — they can't grow into a threat
    together, so with neither contesting the centre P1 builds an open three and
    wins. A pair is a preemptive only if it's a threat pair (aligned AND within
    win_length-1; see ``_is_threat_pair``); unaligned stones, or aligned ones
    too far apart to share a line, are not. The *fine* cases this must NOT
    flag: a "balanced colony" (one stone near the origin + one away) and a
    genuine "preemptive island" (both away but a threat pair). Direct +
    outcome-independent, unlike game length (which lags at low sims).

    Returns ``(mean_dist_of_p2_first_turn, is_refuted)``; ``is_refuted`` is True
    iff P2 played a full 2-stone turn with BOTH stones at hex-distance
    ``>= far_threshold`` from the origin AND the pair is not a preemptive.
    """
    p2 = opening[:2]
    if not p2:
        return 0.0, False
    dists = [_hexdist(q, r) for (q, r) in p2]
    mean_d = sum(dists) / len(dists)
    refuted = (
        len(p2) == 2
        and all(d >= far_threshold for d in dists)
        and not _is_threat_pair(p2[0], p2[1], win_length)
    )
    return mean_d, refuted

logger = logging.getLogger("head_to_head")


def _score_to_elo(s: float, eps: float = 1e-6) -> float:
    """Logistic Elo mapping. Clamps at eps to avoid +/-inf for 0/1 scores."""
    s = max(eps, min(1.0 - eps, s))
    return -400.0 * math.log10(1.0 / s - 1.0)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedCheckpoint:
    """A model loaded onto a device alongside its ModelConfig and metadata."""
    path: Path
    model: HeXONet
    model_config: ModelConfig
    train_steps: int | str  # "?" if absent


def _torch_load(path: Path):
    """torch.load with weights_only first, falling back to unrestricted."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(path: Path, device: torch.device) -> LoadedCheckpoint:
    """Load a HeXONet checkpoint, reconstructing its ModelConfig from the file.

    The checkpoint **must** carry a ``model_config`` dict so we can rebuild
    the architecture; cross-config matches have no shared fallback to lean on.
    """
    raw = _torch_load(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint {path} is not a dict-style state file")
    mc_dict = raw.get("model_config")
    if not mc_dict:
        raise ValueError(
            f"Checkpoint {path} has no 'model_config' — head-to-head requires "
            "each checkpoint to carry its own ModelConfig for cross-config support."
        )
    mc = ModelConfig(**mc_dict)
    model = HeXONet(mc).to(device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in raw["model_state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    train_steps = raw.get("train_steps", raw.get("iteration", "?"))
    return LoadedCheckpoint(path=path, model=model, model_config=mc, train_steps=train_steps)


def _arch_digest(mc: ModelConfig) -> str:
    """Short architecture summary for the header banner."""
    return (
        f"graph={mc.graph_type} conv={mc.conv_type} layers={mc.num_layers} "
        f"hidden={mc.hidden_dim} heads={mc.num_heads} jk={mc.use_jk}"
    )


# ---------------------------------------------------------------------------
# State file I/O
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Match loop
# ---------------------------------------------------------------------------


def run_head_to_head(
    checkpoint_a: Path,
    checkpoint_b: Path,
    *,
    win_length: int,
    radius: int,
    max_moves: int,
    mcts_sims: int = 200,
    mcts_m_actions: int = 16,
    device_str: str = "cpu",
    sprt_s0: float = 0.50,
    sprt_s1: float = 0.55,
    sprt_alpha: float = 0.01,
    sprt_beta: float = 0.05,
    window_size: int | None = 400,
    pair_variance_floor: float = 0.05,
    pair_variance_ceil: float = 0.65,
    max_games: int = 1000,
    seed: int | None = None,
    state_file: Path | None = None,
    mcts_a_forced_candidate_capture_k: int = 0,
    mcts_b_forced_candidate_capture_k: int = 0,
    mcts_a_virtual_loss: float = 0.0,
    mcts_b_virtual_loss: float = 0.0,
    opening_plies: int = 0,
    opening_temperature: float = 1.0,
    opening_generator: Literal["alternate", "a", "b", "champion"] = "alternate",
) -> dict:
    """Run a SPRT-bounded match between two checkpoints.

    Returns a summary dict with final games / wins / draws / losses /
    score / llr / decision / winner ("A", "B", or "inconclusive").

    When ``opening_plies > 0``, each pentanomial pair starts from a sampled
    ``opening_plies``-ply opening (temperature ``opening_temperature``, drawn
    from the ``opening_generator`` model's raw policy) replayed identically
    into both swapped-side games, and both engines play the remainder with
    **Gumbel noise disabled** — diversity comes from the opening, not in-tree
    noise. ``opening_plies == 0`` (default) is the legacy noise-on, no-opening
    mode. ``opening_generator``: ``"alternate"`` (B, A, B, A … per pair),
    ``"a"``, ``"b"``, or ``"champion"`` (= B).
    """
    import hexo_rs  # lazy import — same pattern as evaluate.py / sprt_daemon

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device(device_str)
    game_config = hexo_rs.GameConfig(win_length, radius, max_moves)

    ckpt_a = load_checkpoint(checkpoint_a, device)
    ckpt_b = load_checkpoint(checkpoint_b, device)

    # Pentanomial mode below assumes adaptive pair variance — initial value
    # is the chess-engine convention; we clamp to [floor, ceil] like the daemon.
    sprt_cfg = SPRTConfig(
        s0=sprt_s0, s1=sprt_s1, alpha=sprt_alpha, beta=sprt_beta,
        window_size=window_size,
        pentanomial=True,
        pair_variance=0.35,
    )
    lower, upper = sprt_cfg.bounds()

    # ---- Header ----
    logger.info("=" * 78)
    logger.info("Head-to-head match: A vs B")
    logger.info("  A: %s", ckpt_a.path)
    logger.info("     train_steps=%s  %s", ckpt_a.train_steps, _arch_digest(ckpt_a.model_config))
    logger.info("  B: %s", ckpt_b.path)
    logger.info("     train_steps=%s  %s", ckpt_b.train_steps, _arch_digest(ckpt_b.model_config))
    logger.info("Game: win_length=%d radius=%d max_moves=%d", win_length, radius, max_moves)
    logger.info(
        "MCTS: sims=%d m_actions=%d device=%s  capture_k A=%d B=%d  vl A=%.3g B=%.3g",
        mcts_sims, mcts_m_actions, device_str,
        mcts_a_forced_candidate_capture_k, mcts_b_forced_candidate_capture_k,
        mcts_a_virtual_loss, mcts_b_virtual_loss,
    )
    logger.info(
        "SPRT: s0=%.3f s1=%.3f alpha=%.3f beta=%.3f  bounds [%.3f, %.3f]",
        sprt_s0, sprt_s1, sprt_alpha, sprt_beta, lower, upper,
    )
    logger.info("Max games: %d   pentanomial=True (pair_variance clamped to [%.2f, %.2f])",
                max_games, pair_variance_floor, pair_variance_ceil)
    if state_file is not None:
        logger.info("State file: %s", state_file)
    logger.info("=" * 78)

    state = SPRTState()
    game_idx = 0

    # Paired self-generated openings (diversity source for noise-off SPRT).
    noise_off = opening_plies > 0
    base_seed = seed if seed is not None else 0

    def _generator_for_pair(pair_idx: int):
        if opening_generator == "a":
            return ckpt_a, "A"
        if opening_generator in ("b", "champion"):
            return ckpt_b, "B"
        # "alternate": B, A, B, A, … per pair
        return (ckpt_b, "B") if pair_idx % 2 == 0 else (ckpt_a, "A")

    current_opening: list[tuple[int, int]] | None = None
    current_gen_label: str | None = None
    openings_per_pair: list[list[tuple[int, int]]] = []
    per_game_opening: list[list[tuple[int, int]]] = []
    game_move_logs: list[list[tuple[int, int]]] = []
    lengths_by_gen: dict[str, list[int]] = {"A": [], "B": []}

    while game_idx < max_games:
        side_a = "P1" if game_idx % 2 == 0 else "P2"

        if noise_off and game_idx % 2 == 0:
            pair_idx = game_idx // 2
            gen_ckpt, current_gen_label = _generator_for_pair(pair_idx)
            current_opening = sample_opening(
                gen_ckpt.model, game_config, device, opening_plies,
                opening_temperature, seed=base_seed + pair_idx,
                model_config=gen_ckpt.model_config,
            )
            openings_per_pair.append(current_opening)
        result = play_eval_game(
            ckpt_a.model, game_config, device,
            opponent=ckpt_b.model, model_side=side_a,
            model_config=ckpt_a.model_config,
            opponent_config=ckpt_b.model_config,
            mcts_sims=mcts_sims, mcts_m_actions=mcts_m_actions,
            mcts_forced_candidate_capture_k=mcts_a_forced_candidate_capture_k,
            opponent_mcts_forced_candidate_capture_k=mcts_b_forced_candidate_capture_k,
            mcts_virtual_loss=mcts_a_virtual_loss,
            opponent_mcts_virtual_loss=mcts_b_virtual_loss,
            opening=current_opening if noise_off else None,
            disable_gumbel_noise=noise_off,
        )
        game_move_logs.append(result["move_log"])  # collected in both modes
        if noise_off:
            per_game_opening.append(current_opening)
            lengths_by_gen[current_gen_label].append(result["moves"])
        winner = result["winner"]
        if winner is None:
            outcome = "D"
        elif winner == side_a:
            outcome = "W"
        else:
            outcome = "L"

        # Adapt pair_variance from empirical observations before recording the
        # new outcome (clamped). The daemon does this between rounds via EMA;
        # for a one-shot match we just track the running empirical sample, which
        # is a reasonable single-round approximation.
        emp_var = sample_pair_variance(state._outcomes)
        if emp_var is not None:
            sprt_cfg.pair_variance = max(
                pair_variance_floor, min(pair_variance_ceil, emp_var),
            )

        state.record(outcome, sprt_cfg)
        game_idx += 1

        score, ci_lo, ci_hi, elo_diff = _win_rate_stats(
            state.wins, state.losses, state.draws,
        )
        elo_ci_lo = _score_to_elo(ci_lo)
        elo_ci_hi = _score_to_elo(ci_hi)

        logger.info(
            "game=%d side=%s %s  W-D-L=%d-%d-%d  score=%.3f  "
            "Elo(A-B)=%+.0f [%+.0f, %+.0f]  LLR=%.3f  %s",
            game_idx, side_a, outcome, state.wins, state.draws, state.losses,
            score, elo_diff, elo_ci_lo, elo_ci_hi, state.llr, state.decision,
        )

        if state_file is not None:
            payload = {
                "timestamp": time.time(),
                "checkpoint_a": str(ckpt_a.path),
                "checkpoint_b": str(ckpt_b.path),
                "games": game_idx,
                "wins": state.wins,
                "draws": state.draws,
                "losses": state.losses,
                "score": state.score,
                "elo_diff": elo_diff,
                "elo_ci_lo": elo_ci_lo,
                "elo_ci_hi": elo_ci_hi,
                "llr": state.llr,
                "decision": state.decision,
                "bounds": {"lower": lower, "upper": upper},
                "pair_variance": sprt_cfg.pair_variance,
                "empirical_pair_variance": sample_pair_variance(state._outcomes),
                "mcts_sims": mcts_sims,
                "mcts_m_actions": mcts_m_actions,
                "mcts_a_forced_candidate_capture_k": mcts_a_forced_candidate_capture_k,
                "mcts_b_forced_candidate_capture_k": mcts_b_forced_candidate_capture_k,
                "mcts_a_virtual_loss": mcts_a_virtual_loss,
                "mcts_b_virtual_loss": mcts_b_virtual_loss,
            }
            _atomic_write_json(state_file, payload)

        if state.decision in ("accept_h1", "reject_h1"):
            break

    # ---- Final summary ----
    if state.decision == "accept_h1":
        winner_label = "A"
        verdict = f"A is stronger than B (accept H1 at LLR={state.llr:.3f})"
    elif state.decision == "reject_h1":
        winner_label = "B"
        verdict = f"B is at least as strong as A (reject H1 at LLR={state.llr:.3f})"
    else:
        winner_label = "inconclusive"
        verdict = (
            f"Inconclusive after max_games={max_games} "
            f"(LLR={state.llr:.3f} stayed inside bounds [{lower:.3f}, {upper:.3f}])"
        )

    final_score, final_ci_lo, final_ci_hi, final_elo = _win_rate_stats(
        state.wins, state.losses, state.draws,
    )
    final_elo_lo = _score_to_elo(final_ci_lo)
    final_elo_hi = _score_to_elo(final_ci_hi)

    logger.info("=" * 78)
    logger.info("Final: %s", verdict)
    logger.info(
        "Games=%d  W-D-L=%d-%d-%d  score=%.3f  LLR=%.3f",
        game_idx, state.wins, state.draws, state.losses, state.score, state.llr,
    )
    logger.info(
        "Elo(A-B) = %+.0f   (95%% CI: [%+.0f, %+.0f]; Wilson on score [%.3f, %.3f])",
        final_elo, final_elo_lo, final_elo_hi, final_ci_lo, final_ci_hi,
    )
    logger.info("=" * 78)

    diversity_summary: dict = {}
    # Game-level diversity — reported in BOTH modes, so a legacy noise-on run
    # surfaces how repetitive its games are (relevant to long SPRT rounds).
    if game_move_logs:
        n_games = len(game_move_logs)
        lengths = sorted(len(ml) for ml in game_move_logs)
        diversity_summary["frac_unique_game"] = (
            len({tuple(ml) for ml in game_move_logs}) / n_games)
        diversity_summary["mean_game_length"] = sum(lengths) / n_games
        # Lower tail: p10/min surface short games (e.g. opening blunders).
        diversity_summary["p10_game_length"] = _percentile(lengths, 0.10)
        diversity_summary["min_game_length"] = float(lengths[0])
        logger.info(
            "Games: mode=%s  frac_unique_game=%.3f  mean_len=%.1f p10_len=%.1f min_len=%.1f",
            "openings" if noise_off else "noise-on",
            diversity_summary["frac_unique_game"], diversity_summary["mean_game_length"],
            diversity_summary["p10_game_length"], diversity_summary["min_game_length"],
        )
    # Opening-specific signals (only in noise-off opening mode).
    if noise_off and openings_per_pair:
        n_pairs = len(openings_per_pair)
        diversity_summary["frac_unique_opening"] = (
            len({frozenset(o) for o in openings_per_pair}) / n_pairs)
        diversity_summary["mean_game_length_by_generator"] = {
            g: (sum(ls) / len(ls) if ls else 0.0)
            for g, ls in lengths_by_gen.items()
        }
        diversity_summary["per_game_opening"] = per_game_opening
        p2_stats = [_p2_opening_stats(o) for o in openings_per_pair]
        diversity_summary["p2_open_mean_dist"] = sum(d for d, _ in p2_stats) / n_pairs
        diversity_summary["refuted_open_frac"] = sum(1 for _, r in p2_stats if r) / n_pairs
        logger.info(
            "Openings: plies=%d T=%.2g gen=%s  frac_unique_opening=%.3f  refuted=%.2f "
            "p2_dist=%.1f (A-gen %.1f / B-gen %.1f)",
            opening_plies, opening_temperature, opening_generator,
            diversity_summary["frac_unique_opening"],
            diversity_summary["refuted_open_frac"],
            diversity_summary["p2_open_mean_dist"],
            diversity_summary["mean_game_length_by_generator"]["A"],
            diversity_summary["mean_game_length_by_generator"]["B"],
        )
        logger.info("=" * 78)

    return {
        **diversity_summary,
        "games": game_idx,
        "wins": state.wins,
        "draws": state.draws,
        "losses": state.losses,
        "score": state.score,
        "llr": state.llr,
        "decision": state.decision,
        "winner": winner_label,
        "elo_diff": final_elo,
        "elo_ci_lo": final_elo_lo,
        "elo_ci_hi": final_elo_hi,
        "score_ci_lo": final_ci_lo,
        "score_ci_hi": final_ci_hi,
        "checkpoint_a": str(ckpt_a.path),
        "checkpoint_b": str(ckpt_b.path),
    }
