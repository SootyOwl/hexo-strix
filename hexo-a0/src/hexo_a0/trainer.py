"""Training step and Trainer class for HeXO AlphaZero."""

from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from hexo_a0.config import FullConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

import shutil
import subprocess

from hexo_a0.loss import kl_policy_loss, path_consistency_loss, binned_value_ce
from hexo_a0.model import (
    graft_heads_for_jk_cat,
    graft_input_proj_for_threat_features,
    graft_state_dict,
    split_param_groups,
)


class _ResumeWarmupScheduler:
    """LR scheduler wrapper that prepends a linear warmup on resume.

    For the first ``warmup_steps`` calls to ``step()`` (after construction)
    the LR is linearly interpolated from ``start_lr`` up to the LR the
    underlying scheduler reports at ``resume_step``. After the ramp
    completes, ``step()`` defers entirely to the underlying scheduler.

    The wrapper is deliberately stateless w.r.t. the underlying scheduler:
    it never calls ``underlying.step()`` until ramp completion, then snaps
    the param-group LR to whatever the underlying scheduler currently
    reports (i.e. the LR for ``resume_step``) and steps it from there on
    every subsequent call.

    Used by ``Trainer.load_checkpoint`` when ``do_lr_warmup_on_resume`` is
    set — the rationale mirrors ``warmup_on_arch_change`` for grafts: when
    head columns are zero-initialised on a JK-cat resume, Adam's first
    step would otherwise give them a full-magnitude update (because their
    second-moment estimate is ~0), which can rocket them to unstable
    values before the moment estimate stabilises. The ramp keeps the LR
    small while Adam's running estimates catch up to the new parameters.
    """

    def __init__(
        self,
        underlying,
        optimizer: torch.optim.Optimizer,
        start_lr: float,
        target_lr: float,
        warmup_steps: int,
    ) -> None:
        self._underlying = underlying
        self._optimizer = optimizer
        self._start_lr = float(start_lr)
        self._target_lr = float(target_lr)
        self._warmup_steps = max(1, int(warmup_steps))
        self._step_count = 0
        # Initialise the param group LR to start_lr so the very first
        # forward/backward pair uses the ramped value, not the underlying
        # scheduler's current value.
        for group in self._optimizer.param_groups:
            group["lr"] = self._start_lr

    def step(self) -> None:
        self._step_count += 1
        if self._step_count <= self._warmup_steps:
            # Linear interp from start_lr to target_lr.
            frac = self._step_count / self._warmup_steps
            lr = self._start_lr + frac * (self._target_lr - self._start_lr)
            for group in self._optimizer.param_groups:
                group["lr"] = lr
            return
        # Ramp complete: step the underlying scheduler from here on.
        self._underlying.step()

    def state_dict(self) -> dict:
        return {
            "step_count": self._step_count,
            "start_lr": self._start_lr,
            "target_lr": self._target_lr,
            "warmup_steps": self._warmup_steps,
            "underlying": self._underlying.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state.get("step_count", 0)
        self._start_lr = state.get("start_lr", self._start_lr)
        self._target_lr = state.get("target_lr", self._target_lr)
        self._warmup_steps = state.get("warmup_steps", self._warmup_steps)
        if "underlying" in state:
            self._underlying.load_state_dict(state["underlying"])


def _node_feature_names(model_config) -> list[str]:
    """Node-feature channel names, index-aligned with the graph builders'
    schema for ``model_config``'s layout (see hexo_a0.graph.game_to_graph).

    With model.relative_stone_encoding the stone one-hot is side-to-move
    relative (own/opp) and to_move is dropped (7-dim base); otherwise the
    legacy absolute layout (8-dim base). The 4 threat channels exist only
    when model.threat_features is enabled — callers slice to the model's
    actual input width.
    """
    if getattr(model_config, "relative_stone_encoding", False):
        base = [
            "own_stone", "opp_stone", "empty", "moves_remaining",
            "norm_q", "norm_r", "inv_stone_dist",
        ]
    else:
        base = [
            "p1_stone", "p2_stone", "empty", "to_move", "moves_remaining",
            "norm_q", "norm_r", "inv_stone_dist",
        ]
    return base + [
        "own_max_line", "opp_max_line", "own_threat_axes", "opp_threat_axes",
    ]


def _resync_binary(fh) -> bool:
    """Scan forward in a binary file to find the next supported magic marker.

    Looks for ``HX03`` (legacy, no per-example weight), ``HX04`` (8-dim node
    features + sample_weight), ``HX05`` (current, adds a node_dim byte), or
    ``HX06`` (adds 12B window telemetry after node_dim).
    Reads in 4KB chunks for efficiency. Positions the file handle at the
    start of the magic marker if found. Returns True if found.
    """
    buf = b""
    while True:
        chunk = fh.read(4096)
        if not chunk:
            return False
        buf = buf[-3:] + chunk  # keep last 3 bytes for cross-boundary match
        i03 = buf.find(b"HX03")
        i04 = buf.find(b"HX04")
        i05 = buf.find(b"HX05")
        i06 = buf.find(b"HX06")
        i07 = buf.find(b"HX07")
        candidates = [i for i in (i03, i04, i05, i06, i07) if i >= 0]
        if candidates:
            idx = min(candidates)
            fh.seek(fh.tell() - len(buf) + idx)
            return True
from hexo_a0.replay_buffer import ReplayBuffer


def _parse_binary_example(
    data: bytes, offset: int, has_sample_weight: bool, node_dim: int = 8
) -> tuple[dict, int]:
    """Parse one example from a binary game record.

    Returns (example_dict, new_offset).  The dict has the same keys as
    the graph-format dicts (features, edge_src, …, policy, value,
    sample_weight) so it can be passed to ``_parse_graph_examples``.

    ``has_sample_weight`` is True for HX04+ records and False for legacy
    HX03 (where ``sample_weight`` defaults to 1.0). ``node_dim`` is the
    per-node feature width from the HX05 envelope (8 for HX03/HX04, which
    predate threat features).

    Uses numpy for zero-copy reads from the buffer.
    """
    import struct
    import numpy as np

    num_nodes, num_edges, num_legal = struct.unpack_from("<HHH", data, offset)
    offset += 6
    has_edge_attr = data[offset] != 0
    offset += 1
    (value,) = struct.unpack_from("<f", data, offset)
    offset += 4
    if has_sample_weight:
        (sample_weight,) = struct.unpack_from("<f", data, offset)
        offset += 4
    else:
        sample_weight = 1.0

    # Read arrays directly as numpy, then convert to lists for dict compat.
    # (The overhead here is minimal — the big win is avoiding JSON parsing.)
    n_feat = num_nodes * node_dim
    features = np.frombuffer(data, dtype=np.float32, count=n_feat, offset=offset).tolist()
    offset += n_feat * 4

    edge_src = np.frombuffer(data, dtype=np.uint16, count=num_edges, offset=offset).astype(np.int64).tolist()
    offset += num_edges * 2
    edge_dst = np.frombuffer(data, dtype=np.uint16, count=num_edges, offset=offset).astype(np.int64).tolist()
    offset += num_edges * 2

    edge_attr = None
    if has_edge_attr:
        n_ea = num_edges * 5
        edge_attr = np.frombuffer(data, dtype=np.float32, count=n_ea, offset=offset).tolist()
        offset += n_ea * 4

    legal_mask = [bool(b) for b in data[offset : offset + num_nodes]]
    offset += num_nodes
    stone_mask = [bool(b) for b in data[offset : offset + num_nodes]]
    offset += num_nodes

    n_coords = num_nodes * 2
    coords = np.frombuffer(data, dtype=np.int16, count=n_coords, offset=offset).astype(np.int32).tolist()
    offset += n_coords * 2

    policy = np.frombuffer(data, dtype=np.float32, count=num_legal, offset=offset).astype(np.float64).tolist()
    offset += num_legal * 4

    # Validate: num_legal in header must match legal_mask True count
    legal_count = sum(legal_mask)
    if num_legal != legal_count:
        raise ValueError(
            f"Binary example corrupt: header num_legal={num_legal} but "
            f"legal_mask has {legal_count} True values "
            f"(num_nodes={num_nodes}, num_edges={num_edges}, "
            f"stones={sum(stone_mask)})"
        )

    ex: dict = {
        "features": features,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "legal_mask": legal_mask,
        "stone_mask": stone_mask,
        "coords": coords,
        "num_nodes": num_nodes,
        "policy": policy,
        "value": value,
        "sample_weight": float(sample_weight),
    }
    if edge_attr is not None:
        ex["edge_attr"] = edge_attr

    return ex, offset


_PLAYER_FROM_BYTE = {0: "P1", 1: "P2"}


def _parse_binary_example_hx07(data: bytes, offset: int) -> tuple[dict, int]:
    """Parse one HX07 board-state example from a binary game record.

    Returns ``(example_dict, new_offset)``. The dict carries board state —
    ``stones`` (list of ``((q, r), "P1"|"P2")``), ``current_player``,
    ``moves_remaining`` — plus ``policy``/``value``/``sample_weight``, so the
    trainer can reconstruct a ``GameState`` via ``GameState.from_state``.

    Per-example layout (must match ``write_example_hx07`` in self_play.rs):
    ``[2B num_stones u16][1B current_player u8 (0=P1,1=P2)][1B moves_remaining u8]
    [2B num_legal u16][4B value f32][4B sample_weight f32]
    [num_stones*(2B q i16, 2B r i16, 1B player u8)][num_legal*4B policy f32]``.
    """
    import struct
    import numpy as np

    if offset + 14 > len(data):
        raise ValueError(
            f"HX07 record truncated: header needs {offset + 14} bytes, "
            f"buffer has {len(data)}")
    num_stones, cur_byte, moves_remaining, num_legal = struct.unpack_from(
        "<HBBH", data, offset)
    offset += 6
    (value,) = struct.unpack_from("<f", data, offset)
    offset += 4
    (sample_weight,) = struct.unpack_from("<f", data, offset)
    offset += 4

    stones_needed = num_stones * 5
    if offset + stones_needed > len(data):
        raise ValueError(
            f"HX07 record truncated: stones block needs {offset + stones_needed} "
            f"bytes ({num_stones} stones), buffer has {len(data)}")
    stones: list = []
    for _ in range(num_stones):
        q, r, p = struct.unpack_from("<hhB", data, offset)
        offset += 5
        stones.append(((int(q), int(r)), _PLAYER_FROM_BYTE[p]))

    policy_needed = num_legal * 4
    if offset + policy_needed > len(data):
        raise ValueError(
            f"HX07 record truncated: policy block needs {offset + policy_needed} "
            f"bytes ({num_legal} legal moves), buffer has {len(data)}")
    policy = np.frombuffer(
        data, dtype=np.float32, count=num_legal, offset=offset
    ).astype(np.float64).tolist()
    if len(policy) != num_legal:
        raise ValueError(
            f"HX07 record truncated: policy has {len(policy)} entries, "
            f"expected {num_legal}")
    offset += num_legal * 4

    ex: dict = {
        "stones": stones,
        "current_player": _PLAYER_FROM_BYTE[cur_byte],
        "moves_remaining": int(moves_remaining),
        "num_legal": int(num_legal),
        "policy": policy,
        "value": float(value),
        "sample_weight": float(sample_weight),
    }
    return ex, offset
from hexo_a0.self_play import batched_self_play_games, TrainingExample

logger = logging.getLogger(__name__)


class _SelfPlayReport(NamedTuple):
    games: int
    mean_game_length: float
    p1_wins: int
    p2_wins: int
    examples_written: int
    game_lengths: list[int]
    policy_entropies: list[float]
    policy_top1: list[float]
    value_targets: list[float]
    window_stats: list[dict]


def _sum_window_stats(stats: list[dict]) -> tuple[int, int, float, float]:
    """Sum per-game HX06 window stats: (in_window_moves, gated_moves,
    mass_removed_sum, acted_deficit_sum)."""
    iw = sum(s["in_window_moves"] for s in stats)
    g = sum(s["gated_moves"] for s in stats)
    mr = sum(s["mass_removed_sum"] for s in stats)
    ad = sum(s["acted_deficit_sum"] for s in stats)
    return iw, g, mr, ad


def _example_histogram_stats(
    examples: list,
) -> tuple[list[float], list[float], list[float]]:
    """Per-example entropy of pi', top-1 mass, and value target."""
    entropies: list[float] = []
    top1s: list[float] = []
    values: list[float] = []
    for ex in examples:
        p = ex.policy_target
        mask = p > 0
        if mask.any():
            pm = p[mask]
            h = float(-(pm * pm.log()).sum().item())
        else:
            h = 0.0
        entropies.append(h)
        top1s.append(float(p.max().item()) if p.numel() else 0.0)
        values.append(float(ex.value_target))
    return entropies, top1s, values


def _log_normalized_histogram(
    writer, tag: str, values, bin_edges, step: int
) -> None:
    """Emit a shape-comparable histogram: bucket counts normalized to sum to 1.

    Ridge heights are fractions regardless of how many samples landed in
    this reporting window, so distributions can be compared across steps.
    """
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return
    counts, _ = np.histogram(arr, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return
    norm_counts = (counts.astype(np.float64) / float(total)).tolist()
    edges = np.asarray(bin_edges, dtype=np.float64)
    writer.add_histogram_raw(
        tag=tag,
        min=float(arr.min()),
        max=float(arr.max()),
        num=int(arr.size),
        sum=float(arr.sum()),
        sum_squares=float((arr * arr).sum()),
        bucket_limits=edges[1:].tolist(),
        bucket_counts=norm_counts,
        global_step=step,
    )


# Fixed stone-count phase buckets for the perplexity-by-move metric. Stone
# count (number of placed stones, read off the graph) is the move-number
# proxy: it needs no self-play schema change, and with two placements per turn
# it is monotonic with move number. Each entry is ``(label, lo, hi)`` with the
# half-open range ``[lo, hi)``; the last bucket is open-ended.
#
# Sized from the live S3 replay buffer (radius-8, max_moves=300): per-position
# stone counts run 1..~300, median ~39, p90 ~111. Buckets are fine in the
# opening (where the open/offensive collapse-and-coverage signal lives) and
# coarsen with depth; occupancy at an 8k buffer sample is roughly
# 6/14/19/20/28/9/4 %, so every line has enough support for a stable mean/p10.
# Retune here if the board size / max_moves change materially.
_PERPLEXITY_STONE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("000-004", 0, 5),
    ("005-014", 5, 15),
    ("015-029", 15, 30),
    ("030-049", 30, 50),
    ("050-099", 50, 100),
    ("100-149", 100, 150),
    ("150+", 150, 1_000_000),
)


def _perplexity_by_move(
    model, examples: list, device, use_fp16: bool
):
    """Per-graph perplexity of the network prior and the MCTS target, plus the
    per-graph stone count, for one batch of examples.

    Perplexity = ``exp(H)`` with ``H = -Σ p·log p`` (nats) — the effective
    number of legal moves the distribution spreads over (so it is directly
    comparable to the legal-move count and to the ~1.3-effective-actions target
    figure). Mirrors the per-graph softmax/entropy bookkeeping in
    ``_compute_losses`` but under ``no_grad`` and without any loss.

    Returns ``(prior_perplexity, target_perplexity, stone_counts)`` as three
    1-D numpy arrays of length = number of graphs in ``examples``.
    """
    import numpy as np  # noqa: F401  (kept symmetric with sibling helpers)

    data_list = [ex[0] for ex in examples]
    policy_targets = [ex[1] for ex in examples]

    batch = Batch.from_data_list(list(data_list)).to(device)
    legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
    stone_mask = (
        batch.stone_mask
        if hasattr(batch, "stone_mask") and batch.stone_mask is not None
        else ~batch.legal_mask
    )
    stone_idx = stone_mask.nonzero(as_tuple=False).squeeze(1)
    stone_batch = batch.batch[stone_idx]

    use_amp = use_fp16 and device.type == "cuda"
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            all_logits, legal_counts, _values = model._forward_batch_core(
                batch, legal_idx=legal_idx, stone_idx=stone_idx, stone_batch=stone_batch,
            )

    n = legal_counts.shape[0]
    graph_idx = torch.arange(n, device=device).repeat_interleave(legal_counts)

    # Network prior q = per-graph softmax(logits), computed in a numerically
    # stable way (shift by the per-graph max before exp).
    all_logits_f = all_logits.float()
    max_per_graph = torch.zeros(n, device=device).scatter_reduce(
        0, graph_idx, all_logits_f, reduce="amax", include_self=False,
    )
    shifted = all_logits_f - max_per_graph[graph_idx]
    sum_exp = torch.zeros(n, device=device)
    sum_exp.scatter_add_(0, graph_idx, shifted.exp())
    log_q = shifted - sum_exp[graph_idx].log()
    prior_entropy = torch.zeros(n, device=device)
    prior_entropy.scatter_add_(0, graph_idx, -(log_q.exp() * log_q))

    # MCTS target entropy H(π'). xlogy gives p·log p with the p=0 limit = 0.
    all_targets = torch.cat(list(policy_targets)).float().to(device)
    target_entropy = torch.zeros(n, device=device)
    target_entropy.scatter_add_(
        0, graph_idx, -torch.special.xlogy(all_targets, all_targets),
    )

    # Stone count per graph — the move-number proxy.
    stone_counts = torch.zeros(n, device=device)
    stone_counts.scatter_add_(
        0, stone_batch, torch.ones_like(stone_batch, dtype=torch.float32),
    )

    return (
        prior_entropy.exp().cpu().numpy(),
        target_entropy.exp().cpu().numpy(),
        stone_counts.cpu().numpy(),
    )


def _log_perplexity_by_move_scalars(
    writer, prefix: str, perplexity, stone_counts, step: int
) -> None:
    """Bucket per-graph ``perplexity`` by stone count and emit two overlaid
    charts — ``{prefix}_mean`` and ``{prefix}_p10`` — with one line per phase
    bucket. ``p10`` (the low tail) is the early-warning signal for collapse:
    perplexity → 1 means one-hot, so a bucket whose p10 hugs 1.0 has a
    sub-population that has already collapsed even if its mean looks healthy.
    Buckets with no samples this window are simply omitted (the line skips that
    step) rather than logged as a misleading zero.
    """
    import numpy as np

    mean_d: dict[str, float] = {}
    p10_d: dict[str, float] = {}
    for label, lo, hi in _PERPLEXITY_STONE_BUCKETS:
        sel = (stone_counts >= lo) & (stone_counts < hi)
        if not sel.any():
            continue
        vals = perplexity[sel]
        mean_d[label] = float(np.mean(vals))
        p10_d[label] = float(np.percentile(vals, 10))
    if mean_d:
        writer.add_scalars(f"{prefix}_mean", mean_d, step)
    if p10_d:
        writer.add_scalars(f"{prefix}_p10", p10_d, step)


def _rotate_pool_snapshots(pool_dir: Path, source_path: Path, max_size: int) -> Path | None:
    """Copy ``source_path`` into ``pool_dir`` and prune oldest snapshots
    until at most ``max_size`` ``*.pt`` files remain.

    Uses ``shutil.copy2`` (not symlink/hardlink) so pool snapshots survive
    deletion of the source checkpoint. The copied filename is the source
    file's basename plus a unique suffix derived from the source mtime to
    avoid collisions when the same export is rotated repeatedly.

    Returns the path of the new snapshot, or ``None`` if ``source_path``
    does not exist or ``max_size <= 0`` (pool effectively disabled).
    """
    if max_size <= 0:
        return None
    if not source_path.exists():
        return None
    pool_dir.mkdir(parents=True, exist_ok=True)
    stat = source_path.stat()
    # Unique-ish destination name: <stem>_<mtime_ns>.pt
    stem = source_path.stem
    suffix = source_path.suffix or ".pt"
    dest = pool_dir / f"{stem}_{stat.st_mtime_ns}{suffix}"
    if not dest.exists():
        shutil.copy2(source_path, dest)
    # Prune oldest *.pt files (sort by mtime ascending, drop excess)
    snaps = sorted(pool_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    excess = len(snaps) - max_size
    for old in snaps[:max(0, excess)]:
        try:
            old.unlink()
        except OSError:
            pass
    return dest


# Number of consecutive batched-augment fast-path failures we tolerate
# (falling back per-batch) before raising — so a real bug surfaces loudly
# rather than silently degrading to the slow path forever.
_AUGMENT_FALLBACK_MAX_STREAK = 10


def _augment_fallback_escalate(streak: int, exc: BaseException, where: str) -> int:
    """Account one failure of the batched byte-buffer augment fast path.

    Increments the consecutive-failure ``streak``, logs a warning, and RAISES
    once it reaches ``_AUGMENT_FALLBACK_MAX_STREAK`` so a persistent bug in the
    augment path surfaces loudly instead of silently downgrading to the slow
    per-Data path forever (a mere perf loss). Returns the new streak. Shared by
    the prefetch worker and the inline (reanalyze) training path so the
    escalation policy lives in exactly one place.
    """
    streak += 1
    logger.warning(
        "%s: batched augment failed (streak %d/%d); falling back to per-Data "
        "augment: %r", where, streak, _AUGMENT_FALLBACK_MAX_STREAK, exc,
    )
    if streak >= _AUGMENT_FALLBACK_MAX_STREAK:
        raise RuntimeError(
            f"{where}: batched augment failed {streak} times consecutively — "
            "persistent bug in the byte-buffer augment path (aborting instead "
            "of silently degrading to the slow path)"
        ) from exc
    return streak


def _prefetch_worker(
    buf,
    batch_size: int,
    augment_enabled: bool,
    q: "queue.Queue",
    stop_event: "threading.Event",
    device_type: str,
    graph_batch_fn,
    augment_fn=None,
) -> None:
    """Sample + augment + pin batches in the background.

    Runs CPU-only work (SQLite read, pickle.loads, symmetry augmentation,
    pinning) so the main thread only has to do GPU work + the single H2D
    copy. Pinning is skipped on CPU device.

    The worker produces a list of ``(data, policy_target, value_target,
    trajectory_id, age)`` tuples — the same shape the main loop used to
    build inline. The main loop just calls ``q.get()``. ``age`` is a float
    in [0.0, 1.0] (0=newest, 1=oldest) attached by the buffer at sample
    time; used for staleness stratification in `_forward_and_loss`.

    Fast path: when ``augment_fn`` is provided, augmentation is enabled, and
    every sampled example is game_state-bearing (``ex.data is None``), the
    whole batch is augmented in a single GIL-released Rust call and ``data``
    is a per-graph SimpleNamespace (with precomputed index tensors) rather
    than a PyG ``Data``. Otherwise the legacy per-``Data`` ``random_augment``
    path runs unchanged.
    """
    from hexo_a0.graph import random_augment

    # Consecutive failures of the batched byte-buffer augment fast path. Reset
    # to 0 on any success; after _AUGMENT_FALLBACK_MAX_STREAK consecutive
    # failures we RAISE so a persistent bug surfaces loudly instead of silently
    # downgrading to the slow per-Data path forever (a mere perf loss).
    augment_fallback_streak = 0

    while not stop_event.is_set():
        try:
            raw_batch = buf.sample(batch_size)
        except Exception:
            logger.exception("Prefetch worker: buffer.sample failed")
            return

        # Fast path: batched byte-buffer augment for an all-game_state batch.
        ns_parts = None
        ns_perms = None
        if (
            augment_enabled
            and augment_fn is not None
            and raw_batch
            and all(ex.data is None for ex in raw_batch)
        ):
            states = [ex.game_state for ex in raw_batch]
            tindices = [random.randint(1, 11) for _ in raw_batch]
            try:
                ns_parts, ns_perms = augment_fn(states, tindices)
                augment_fallback_streak = 0
            except Exception as exc:
                # Any failure (e.g. a non-axis graph_type) falls back cleanly
                # to the legacy per-Data path below; escalation is shared with
                # the inline path via _augment_fallback_escalate.
                ns_parts = None
                augment_fallback_streak = _augment_fallback_escalate(
                    augment_fallback_streak, exc, "Prefetch worker")

        recon = (
            [ex.game_state for ex in raw_batch if ex.data is None]
            if ns_parts is None
            else []
        )
        rebuilt = iter(graph_batch_fn(recon)) if recon else iter(())
        all_examples = []
        for i, ex in enumerate(raw_batch):
            if ns_parts is not None:
                data = ns_parts[i]
                policy = ex.policy_target[torch.tensor(ns_perms[i], dtype=torch.long)]
            else:
                data = ex.data if ex.data is not None else next(rebuilt)
                if augment_enabled:
                    data, policy = random_augment(data, ex.policy_target)
                else:
                    policy = ex.policy_target
            value_t = torch.tensor(ex.value_target, dtype=torch.float32)
            if device_type == "cuda":
                # Pin so the H2D copies in _forward_and_loss can actually
                # overlap (non_blocking=True only overlaps from pinned src).
                try:
                    data.x = data.x.pin_memory()
                    data.edge_index = data.edge_index.pin_memory()
                    if getattr(data, "legal_mask", None) is not None:
                        data.legal_mask = data.legal_mask.pin_memory()
                    if getattr(data, "stone_mask", None) is not None:
                        data.stone_mask = data.stone_mask.pin_memory()
                    if getattr(data, "edge_attr", None) is not None:
                        data.edge_attr = data.edge_attr.pin_memory()
                    policy = policy.pin_memory()
                    value_t = value_t.pin_memory()
                except RuntimeError:
                    # pin_memory can fail under memory pressure or if the
                    # tensor is already pinned via shared storage. Fall back
                    # silently — correctness is unaffected, only overlap is.
                    pass
            all_examples.append((
                data,
                policy,
                value_t,
                getattr(ex, "trajectory_id", None),
                getattr(ex, "_age", None),
                float(getattr(ex, "sample_weight", 1.0)),
            ))

        # Block on put with timeout so the worker can notice stop_event.
        while not stop_event.is_set():
            try:
                q.put(all_examples, timeout=0.5)
                break
            except queue.Full:
                continue


def _collate_ns_batch(ns_list, device):
    """Collate a list of single-graph SimpleNamespaces into one batch + indices.

    Each namespace comes from `augment_axis_states_to_per_graph`: a single graph
    with local edge_index, legal_mask, edge_attr and per-graph precomputed
    legal_idx/stone_idx/stone_batch. We concatenate node/edge tensors with the
    usual PyG offsets and rebase the carried index tensors to the collated
    layout — avoiding the GPU nonzero() syncs the legacy path needs.

    Returns (batch_ns, legal_idx, stone_idx, stone_batch) all on ``device``.
    """
    from types import SimpleNamespace

    xs, eis, eas, lms, batch_vec = [], [], [], [], []
    legal_idx_parts, stone_idx_parts, stone_batch_parts = [], [], []
    node_off = 0
    for gi, ns in enumerate(ns_list):
        n = ns.x.shape[0]
        xs.append(ns.x)
        eis.append(ns.edge_index + node_off)
        eas.append(ns.edge_attr)
        lms.append(ns.legal_mask)
        batch_vec.append(torch.full((n,), gi, dtype=torch.long))
        legal_idx_parts.append(ns.legal_idx + node_off)
        stone_idx_parts.append(ns.stone_idx + node_off)
        stone_batch_parts.append(torch.full((ns.stone_idx.shape[0],), gi, dtype=torch.long))
        node_off += n

    batch = SimpleNamespace(
        x=torch.cat(xs).to(device, non_blocking=True),
        edge_index=torch.cat(eis, dim=1).to(device, non_blocking=True),
        edge_attr=torch.cat(eas).to(device, non_blocking=True),
        legal_mask=torch.cat(lms).to(device, non_blocking=True),
        batch=torch.cat(batch_vec).to(device, non_blocking=True),
        num_graphs=len(ns_list),
    )
    legal_idx = torch.cat(legal_idx_parts).to(device, non_blocking=True)
    stone_idx = torch.cat(stone_idx_parts).to(device, non_blocking=True)
    stone_batch = torch.cat(stone_batch_parts).to(device, non_blocking=True)
    return batch, legal_idx, stone_idx, stone_batch


def _forward_and_loss(
    model: nn.Module,
    examples: list[tuple],
    device: torch.device,
    use_fp16: bool = False,
    loss_scale: float = 1.0,
    pc_loss_weight: float = 0.0,
) -> dict[str, "torch.Tensor | float"]:
    """Forward pass + loss computation + backward (no optimizer step).

    Computes scaled loss and calls .backward().  The caller is responsible
    for optimizer.zero_grad() before the first micro-batch and
    optimizer.step() after the last.

    Args:
        loss_scale: Multiply loss by this before backward (1/N for N
                    gradient-accumulation micro-batches).
        pc_loss_weight: Weight for path consistency loss (0 = disabled).

    Returns:
        Dict with detached GPU tensors for losses (sync-free). The caller
        is responsible for syncing them (one ``.item()`` per step) and for
        the NaN/Inf guard. ``pc_loss`` is a tensor when PC loss is active,
        otherwise the float ``0.0`` sentinel.
    """
    # Unpack — examples may have 3, 4, 5, or 6 elements:
    #   3-tuple: (data, policy, value)
    #   4-tuple: + trajectory_id
    #   5-tuple: + age (buffer position at sample time, [0, 1], 0=newest)
    #   6-tuple: + sample_weight (in (0, 1]; <1 for PCR fast-cap examples)
    ages: tuple | None = None
    sample_weights: tuple | None = None
    arity = len(examples[0])
    if arity == 6:
        data_list, policy_targets, value_targets, traj_ids, ages, sample_weights = zip(*examples)
    elif arity == 5:
        data_list, policy_targets, value_targets, traj_ids, ages = zip(*examples)
    elif arity == 4:
        data_list, policy_targets, value_targets, traj_ids = zip(*examples)
    else:
        data_list, policy_targets, value_targets = zip(*examples)
        traj_ids = None
    # non_blocking lets the pinned tensors from the prefetch worker (fix 4)
    # actually overlap with GPU compute. Falls back to a sync copy if the
    # source isn't pinned, which is harmless.
    #
    # Two collation paths:
    #  - Legacy: per-example PyG ``Data`` -> ``Batch.from_data_list`` (+ nonzero
    #    to derive the index tensors).
    #  - New byte-buffer augment path: each element is a single-graph
    #    ``SimpleNamespace`` carrying Rust-precomputed per-graph index tensors
    #    (legal_idx/stone_idx/stone_batch). We concatenate them ourselves,
    #    re-offsetting node/edge indices, and reuse the carried indices so we
    #    skip the GPU nonzero syncs entirely.
    # The collator is selected off element 0; the prefetch worker / inline path
    # build a batch all-Data or all-namespace (never mixed), so assert that
    # invariant here rather than mis-collating silently deeper down.
    first_is_data = isinstance(data_list[0], Data)
    assert all(isinstance(d, Data) == first_is_data for d in data_list), (
        "Mixed Data/namespace batch in _forward_and_loss; the augment path must "
        "produce a homogeneous batch."
    )
    if first_is_data:
        batch = Batch.from_data_list(list(data_list)).to(device, non_blocking=True)
        # Precompute index tensors to avoid nonzero() graph breaks under torch.compile
        legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
        stone_mask = batch.stone_mask if hasattr(batch, "stone_mask") and batch.stone_mask is not None else ~batch.legal_mask
        stone_idx = stone_mask.nonzero(as_tuple=False).squeeze(1)
        stone_batch = batch.batch[stone_idx]
    else:
        batch, legal_idx, stone_idx, stone_batch = _collate_ns_batch(
            data_list, device
        )

    use_amp = use_fp16 and device.type == "cuda"

    # Distributional value head (value_bins>0): the core additionally returns
    # the raw bin logits so the value loss is a two-hot cross-entropy; the
    # scalar decode still rides in ``values`` for the diagnostic MSE. Scalar
    # heads keep the byte-identical 3-tuple path.
    value_bins = int(getattr(model, "value_bins", 0) or 0)

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        if value_bins > 0:
            all_logits, legal_counts, values, value_bin_logits = model._forward_batch_core(
                batch, legal_idx=legal_idx, stone_idx=stone_idx,
                stone_batch=stone_batch, return_value_logits=True,
            )
        else:
            all_logits, legal_counts, values = model._forward_batch_core(
                batch, legal_idx=legal_idx, stone_idx=stone_idx, stone_batch=stone_batch,
            )

        # Stack on CPU first, then a single non-blocking H2D copy. Avoids
        # 2*batch_size separate small-tensor copies (ROCm is sensitive to
        # small-copy launch overhead). With fix 4 the source per-element
        # tensors are already pinned, so the stacked result is also pinned
        # (torch.stack/cat preserves pinning when all inputs are pinned).
        # On the off chance it isn't, we pin explicitly so non_blocking
        # actually overlaps.
        v_targets_cpu = torch.stack(list(value_targets))
        if device.type == "cuda" and not v_targets_cpu.is_pinned():
            v_targets_cpu = v_targets_cpu.pin_memory()
        v_targets = v_targets_cpu.to(device, non_blocking=True)

        # Per-example sample weights (HX04+). Default 1.0 when not present
        # (legacy HX03 data, manual training paths). When all weights are
        # 1.0 the weighted mean is bitwise equivalent to the plain mean
        # modulo the explicit `sum(w)` denominator (numerically very close).
        if sample_weights is not None:
            sw_cpu = torch.tensor(sample_weights, dtype=torch.float32)
            if device.type == "cuda" and not sw_cpu.is_pinned():
                sw_cpu = sw_cpu.pin_memory()
            sw = sw_cpu.to(device, non_blocking=True)
        else:
            sw = torch.ones(len(value_targets), device=device, dtype=torch.float32)
        # Guard against degenerate batches where every example happens to
        # have weight 0 (shouldn't occur in practice; PCR fast-cap weights
        # are 1/divisor > 0). Falls back to uniform to keep gradients
        # finite if it ever does.
        sw_sum = sw.sum().clamp_min(1e-8)

        # Decoded-scalar MSE — the optimized term for scalar heads, and a
        # comparable diagnostic for distributional heads.
        v_diff_sq = (values.float() - v_targets).pow(2)
        value_mse = (v_diff_sq * sw).sum() / sw_sum
        if value_bins > 0:
            # Distributional head: optimize the two-hot cross-entropy over bins.
            ce_per_ex = binned_value_ce(
                value_bin_logits.float(), v_targets, model.value_bin_centers
            )
            mean_value = (ce_per_ex * sw).sum() / sw_sum
        else:
            mean_value = value_mse

        all_targets_cpu = torch.cat(list(policy_targets))
        if all_targets_cpu.shape[0] != all_logits.shape[0]:
            # Diagnose which examples have mismatched policy/legal_mask
            mismatches = []
            for i, (d, pt) in enumerate(zip(data_list, policy_targets)):
                lm_sum = int(d.legal_mask.sum().item())
                if pt.shape[0] != lm_sum:
                    sm = getattr(d, "stone_mask", None)
                    stones = int(sm.sum().item()) if sm is not None else -1
                    mismatches.append(
                        f"  ex[{i}]: policy={pt.shape[0]} legal_mask={lm_sum} "
                        f"nodes={d.x.shape[0]} stones={stones}"
                    )
            raise ValueError(
                f"Policy target length {all_targets_cpu.shape[0]} != model logits "
                f"length {all_logits.shape[0]} (batch of {len(data_list)} examples). "
                f"Mismatched examples:\n" + "\n".join(mismatches[:20])
            )
        if device.type == "cuda" and not all_targets_cpu.is_pinned():
            all_targets_cpu = all_targets_cpu.pin_memory()
        all_targets = all_targets_cpu.to(device, non_blocking=True)
        n = legal_counts.shape[0]
        graph_idx = torch.arange(n, device=device).repeat_interleave(legal_counts)

        all_logits_f = all_logits.float()
        max_per_graph = torch.zeros(n, device=device).scatter_reduce(
            0, graph_idx, all_logits_f, reduce="amax", include_self=False,
        )
        shifted = all_logits_f - max_per_graph[graph_idx]
        exp_shifted = shifted.exp()
        sum_exp = torch.zeros(n, device=device)
        sum_exp.scatter_add_(0, graph_idx, exp_shifted)
        all_log_q = shifted - sum_exp[graph_idx].log()

        all_targets_f = all_targets.float()
        cross_entropy_per_node = -(all_targets_f * all_log_q)
        entropy_per_node = torch.special.xlogy(all_targets_f, all_targets_f)
        kl_per_graph = torch.zeros(n, device=device)
        kl_per_graph.scatter_add_(0, graph_idx, cross_entropy_per_node + entropy_per_node)
        # Weighted mean: each per-graph KL is multiplied by its sample weight,
        # normalised by the sum of weights (effective sample size). When
        # weights are all 1.0 this collapses to the plain arithmetic mean.
        mean_policy = (kl_per_graph * sw).sum() / sw_sum

        # Diagnostic-only scalars: target entropy H(π) and cross-entropy CE(π, q).
        # These do not influence the gradient (entropy is target-only; CE shares
        # the gradient of mean_policy up to the constant H term). Used to track
        # the KL floor: KL ≈ 0 means the model has matched the targets; the
        # remaining "loss budget" sits in CE = KL + H.
        ce_per_graph = torch.zeros(n, device=device)
        ce_per_graph.scatter_add_(0, graph_idx, cross_entropy_per_node)
        mean_cross_entropy = (ce_per_graph * sw).sum() / sw_sum
        target_entropy_per_graph = torch.zeros(n, device=device)
        target_entropy_per_graph.scatter_add_(0, graph_idx, -entropy_per_node)
        mean_target_entropy = (target_entropy_per_graph * sw).sum() / sw_sum

        total_loss = mean_policy + mean_value

    # Path consistency loss (PCZero)
    pc_loss_val: "torch.Tensor | float" = 0.0
    if pc_loss_weight > 0 and traj_ids is not None and traj_ids[0] is not None:
        traj_groups: dict[int, list[int]] = {}
        for i, tid in enumerate(traj_ids):
            if tid is not None:
                traj_groups.setdefault(tid, []).append(i)

        pc_losses = []
        for tid, indices in traj_groups.items():
            if len(indices) > 1:
                traj_values = values[indices]
                traj_outcome = v_targets[indices[0]]
                pc_losses.append(path_consistency_loss(traj_values, traj_outcome))
        if pc_losses:
            pc_loss = torch.stack(pc_losses).mean()
            total_loss = total_loss + pc_loss_weight * pc_loss
            pc_loss_val = pc_loss.detach()

    # NB: NaN/Inf guard moved to caller (single .item() sync per step).
    # Calling backward() on a NaN loss is harmless as long as the caller
    # then skips optimizer.step() and zeros the grads.
    (total_loss * loss_scale).backward()

    # Staleness instrumentation: stratify per-graph policy KL by sample age.
    # When ages are present, return per-quartile sums and counts so the
    # caller can accumulate across micro-batches and steps.
    staleness_kl_q_sum: torch.Tensor | None = None
    staleness_kl_q_count: torch.Tensor | None = None
    if ages is not None and all(a is not None for a in ages):
        ages_t = torch.tensor(ages, device=device, dtype=torch.float32)
        # Quartile bounds: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
        # Last bucket includes 1.0 inclusive.
        kl_detached = kl_per_graph.detach()
        q_sum = torch.zeros(4, device=device)
        q_count = torch.zeros(4, device=device)
        bounds = [0.0, 0.25, 0.5, 0.75, 1.0]
        for q in range(4):
            lo, hi = bounds[q], bounds[q + 1]
            if q < 3:
                mask = (ages_t >= lo) & (ages_t < hi)
            else:
                mask = (ages_t >= lo) & (ages_t <= hi)
            q_sum[q] = (kl_detached * mask.float()).sum()
            q_count[q] = mask.float().sum()
        staleness_kl_q_sum = q_sum
        staleness_kl_q_count = q_count

    return {
        "total_loss": total_loss.detach(),
        "policy_loss": mean_policy.detach(),
        "value_loss": mean_value.detach(),
        "value_mse": value_mse.detach(),
        "pc_loss": pc_loss_val,
        "cross_entropy": mean_cross_entropy.detach(),
        "target_entropy": mean_target_entropy.detach(),
        "staleness_kl_q_sum": staleness_kl_q_sum,
        "staleness_kl_q_count": staleness_kl_q_count,
    }


def _clip_and_step_if_finite(model, optimizer, max_grad_norm, scheduler=None):
    """Clip gradients and step the optimizer ONLY if the gradients are finite.

    ``clip_grad_norm_`` returns the *pre-clip* total norm, which is NaN/Inf iff
    any gradient is non-finite. A non-finite gradient can occur even with a
    finite loss (e.g. an unstable backward); the loss-only guard does not catch
    that, so without this check ``optimizer.step()`` would write NaN into the
    weights (and optimizer state), poisoning checkpoints. On a non-finite
    gradient we zero the grads and skip the step. Returns ``(stepped, grad_norm)``.
    """
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not bool(torch.isfinite(grad_norm)):
        optimizer.zero_grad(set_to_none=True)
        return False, float(grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return True, float(grad_norm)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: list[tuple],
    device: torch.device,
    max_grad_norm: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: object | None = None,
    use_fp16: bool = False,
    pc_loss_weight: float = 0.0,
) -> dict[str, float]:
    """Perform a single training step on a batch of examples.

    Args:
        model:          HeXONet (or any module with ``forward_batch``).
        optimizer:      PyTorch optimizer bound to model parameters.
        examples:       List of ``(data, policy_target, value_target)`` tuples
                        (optionally with a 4th element for trajectory_id)
                        where ``data`` is a PyG ``Data`` from ``game_to_graph``,
                        ``policy_target`` is a 1-D probability tensor of length
                        equal to the number of legal moves, and ``value_target``
                        is a scalar tensor.
        device:         Device to move tensors to before the forward pass.
        max_grad_norm:  Maximum gradient norm for clipping (default 1.0).
        scaler:         Optional GradScaler (unused, kept for compatibility).
        scheduler:      Optional LR scheduler — stepped after optimizer.step().
        use_fp16:       Enable bf16 autocast for mixed-precision training.

    Returns:
        A dict with keys ``"total_loss"``, ``"policy_loss"``, and
        ``"value_loss"``, each containing the corresponding float value.
    """
    if not examples:
        return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "pc_loss": 0.0}

    model.train()
    optimizer.zero_grad()
    tensor_losses = _forward_and_loss(model, examples, device, use_fp16, pc_loss_weight=pc_loss_weight)

    # Single sync per call: convert tensors to floats here. _forward_and_loss
    # returns detached GPU tensors (sync-free); the NaN guard lives at the
    # caller boundary so the hot training loop only pays one .item() per step.
    # (value_mse is intentionally NOT surfaced here — train_step's dict contract
    # is total/policy/value/pc; the distributional-head value_mse diagnostic is
    # logged from Trainer.train()'s accumulator, the production TB path.)
    losses: dict[str, float] = {
        "total_loss": float(tensor_losses["total_loss"].item()),
        "policy_loss": float(tensor_losses["policy_loss"].item()),
        "value_loss": float(tensor_losses["value_loss"].item()),
        "pc_loss": (
            float(tensor_losses["pc_loss"].item())
            if torch.is_tensor(tensor_losses["pc_loss"])
            else float(tensor_losses["pc_loss"])
        ),
    }

    if not (math.isnan(losses["total_loss"]) or math.isinf(losses["total_loss"])):
        _clip_and_step_if_finite(model, optimizer, max_grad_norm, scheduler)

    return losses


class Trainer:
    """Manages the HeXO AlphaZero training loop.

    Encapsulates self-play data generation, replay buffer management,
    gradient-update steps, and checkpoint save/load.

    Args:
        model:           HeXONet instance.
        model_config:    ModelConfig (stored for checkpoint reconstruction).
        training_config: TrainingConfig hyperparameters.
        game_config:     hexo_rs.GameConfig (PyO3; stored as plain dict in checkpoints).
        device:          Torch device for all computation.
    """

    @staticmethod
    def _make_augment_batch_fn(model_config):
        """Build the batched augment fn for axis graphs, else None.

        Returns a callable ``(states, transform_indices) -> (parts, perms)``
        bound to this model's graph flags (prune/threat/relative), so the
        prefetch worker and inline path stay flag-agnostic. Only axis graphs
        have a byte-buffer augment binding; hex configs return None and fall
        back to per-Data ``random_augment``.
        """
        if getattr(model_config, "graph_type", None) != "axis":
            return None
        from hexo_a0.graph import augment_axis_states_to_per_graph

        prune = model_config.prune_empty_edges
        threat = getattr(model_config, "threat_features", False)
        rel = getattr(model_config, "relative_stone_encoding", False)

        def _fn(states, transform_indices):
            return augment_axis_states_to_per_graph(
                states, transform_indices,
                prune_empty_edges=prune, threat_features=threat,
                relative_stones=rel,
            )

        return _fn

    def __init__(self, model, config: FullConfig, game_config, device, log_dir=None, ckpt_dir=None):
        self.model = model
        self.config = config
        self.model_config = config.model
        from hexo_a0.graph import graph_fn_from_model_config
        self.graph_fn = graph_fn_from_model_config(self.model_config)
        from hexo_a0.graph import graph_batch_fn_from_model_config
        self.graph_batch_fn = graph_batch_fn_from_model_config(self.model_config)
        # Batched byte-buffer augment fn (axis graphs only). Returns per-graph
        # namespaces + permutations; None for non-axis configs (those fall back
        # to the per-Data random_augment path).
        self.augment_batch_fn = self._make_augment_batch_fn(self.model_config)
        self.training_config = config.training  # kept for compat during migration
        self.mcts_config = config.mcts
        self.self_play_config = config.self_play
        self.eval_config = config.eval
        self.run_config = config.run
        self.game_config = game_config
        self.device = device
        self._ckpt_dir = ckpt_dir
        self._sp_pool = None
        self._pool_snapshot_counter = 0
        # Counter for throttling the stale-binary version-skew warning.
        self._stale_binary_warn_count = 0
        # Loud feedback for an invalid truncate_delta: negative / inf / nan
        # values silently disable the gate everywhere (the trainer never sends
        # the flag, so the binary's discard warning and the version-skew guard
        # both stay silent). Warn once at init so it isn't a silent no-op.
        _td = self.mcts_config.truncate_delta
        if _td != 0 and not (0 < _td and math.isfinite(_td)):
            logger.warning(
                "truncate_delta=%r is not a positive finite value; the Q-margin "
                "gate is OFF (set 0 to silence this warning)",
                _td,
            )
        elif _td > 0 and self.self_play_config.engine != "rust":
            # Config-level engine-inertness check at init. The binary-not-found
            # case can only be resolved at worker start (filesystem probe in
            # start_self_play), so that warning stays there too.
            logger.warning(
                "truncate_delta=%.4g is set but self_play.engine=%r is not 'rust' "
                "— the Q-margin gate is silently inert (Rust self-play only)",
                _td,
                self.self_play_config.engine,
            )
        self._champion_path: str | None = None  # set when convergence_mode="champion"
        self._champion_mode: bool = False  # suppresses periodic self-play model re-exports
        self._sprt_mode: bool = False  # when True, skip vs-champion fixed-N eval (SPRT daemon handles it)
        self._checkpoint_lag: int = 1  # set by curriculum.py from curriculum config
        self.optimizer = torch.optim.AdamW(
            split_param_groups(
                model,
                self.training_config.weight_decay,
                self.training_config.layer_scale_weight_decay,
            ),
            lr=self.training_config.lr,
        )
        self._init_scheduler()
        # Mixed-precision (bf16 — no GradScaler needed)
        self.scaler = None
        # Derive db_path from ckpt_dir when available (REQ-20)
        if ckpt_dir is not None:
            _db_path = str(Path(ckpt_dir) / "replay_buffer.db")
        else:
            _db_path = ":memory:"
        self.buffer = ReplayBuffer(self.training_config.buffer_capacity, db_path=_db_path)

        self.current_step_delay = self.training_config.step_delay
        self.train_steps = 0
        # When resuming from a checkpoint taken mid-chunk, the first outer
        # iteration runs a short chunk so that train_steps realigns to a
        # multiple of the normal chunk size, preserving trigger cadence.
        self._resume_short_chunk = 0
        self._sealbot_level_idx = 0
        self.stage_name = ""  # set by curriculum
        self.stage_start_step = 0  # train_steps at the start of the current stage
        self._sealbot_consecutive_above = 0
        self.writer = None
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            from hexo_a0.tb_writer import DescribedWriter
            self.writer = DescribedWriter(SummaryWriter(log_dir=log_dir))

        from hexo_a0.display import TrainingDisplay
        self.display = TrainingDisplay()

        if self.run_config.seed is not None:
            torch.manual_seed(self.run_config.seed)
            random.seed(self.run_config.seed)

    def log_event(self, tag: str, text: str) -> None:
        """Write a text event to TensorBoard at the current step.

        Tag is namespaced under ``events/`` automatically, e.g.
        ``log_event("champion", "promoted")`` writes to ``events/champion``.
        """
        if self.writer is not None:
            self.writer.add_text(f"events/{tag}", text, self.train_steps)

    def _init_scheduler(self):
        """Create the LR scheduler based on training_config."""
        tc = self.training_config
        warmup_steps = tc.lr_warmup_steps
        if tc.lr_schedule == "cosine" and tc.total_train_steps > 0:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
            schedulers = []
            milestones = []
            if warmup_steps > 0:
                schedulers.append(LinearLR(
                    self.optimizer, start_factor=1e-3, total_iters=warmup_steps,
                ))
                milestones.append(warmup_steps)
            schedulers.append(CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=tc.total_train_steps - warmup_steps,
                # Growing restart cycles (interim hardcode; TODO: config knob):
                # re-warm often early in a stage when the self-play target drifts
                # fastest, settle longer as the stage matures. Autonomous —
                # deliberately NOT tied to the (vibe-check) convergence detector.
                T_mult=2,
                eta_min=tc.lr_min,
            ))
            if len(schedulers) > 1:
                self.scheduler = SequentialLR(self.optimizer, schedulers, milestones)
            else:
                self.scheduler = schedulers[0]
        elif warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer, start_factor=1e-3, total_iters=warmup_steps,
            )
        else:
            self.scheduler = None

    # ------------------------------------------------------------------
    # Continuous self-play
    # ------------------------------------------------------------------

    def start_self_play(self) -> None:
        """Start background self-play (Rust binary only).

        When engine is not "rust", self-play is handled inline
        by train_iteration (alternating GPU self-play + training).
        """
        if self._sp_pool is not None:
            return  # already running

        use_rust = self.self_play_config.engine == "rust"
        rust_binary = self._find_self_play_binary() if use_rust else None

        if rust_binary is None:
            # Alternating mode: no background workers needed
            logger.info("Using alternating GPU self-play (no background workers)")
            if (
                use_rust  # engine != "rust" already warned at trainer init
                and self.mcts_config.truncate_delta > 0
                and math.isfinite(self.mcts_config.truncate_delta)
            ):
                logger.warning(
                    "truncate_delta=%.4g is set but the Rust self-play binary was not "
                    "found — the Q-margin gate is silently inert until it is built",
                    self.mcts_config.truncate_delta,
                )
            return

        import threading
        from concurrent.futures import ThreadPoolExecutor

        self._sp_stop = threading.Event()
        self._sp_games = 0
        # Per-game training-example counts (number of full-search positions
        # written per game). With playout cap randomization this differs
        # from the actual game move count, which is tracked separately.
        self._sp_game_examples: list[int] = []
        # Per-game actual move count from the binary record header.
        self._sp_game_move_counts: list[int] = []
        self._sp_p1_wins = 0
        self._sp_p2_wins = 0
        # Per-example stats accumulated at ingestion; drained each report.
        self._sp_policy_entropies: list[float] = []
        self._sp_policy_top1: list[float] = []
        self._sp_value_targets: list[float] = []
        # Per-game HX06 window telemetry (None for non-HX06 records).
        self._sp_window_stats: list[dict] = []
        self._sp_lock = threading.Lock()
        self._sp_pool = ThreadPoolExecutor(max_workers=1)
        self._sp_pool.submit(self._rust_self_play_worker, rust_binary)

    def _find_self_play_binary(self) -> str | None:
        """Find the compiled self_play binary."""
        import shutil
        # Check common locations
        candidates = [
            Path(__file__).parents[3] / "hexo-rs" / "target" / "release" / "self_play",
            Path("hexo-rs/target/release/self_play"),
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return shutil.which("self_play")

    def _snapshot_to_pool(self, source_path: str) -> None:
        """Copy the just-exported self-play model into the past-self pool dir,
        then prune oldest snapshots beyond pool_size.

        Pool is disabled when pool_fraction == 0; we still skip the copy in
        that case to keep existing runs byte-identical.
        """
        rust_cfg = self.self_play_config.rust
        if rust_cfg.pool_fraction <= 0.0 or self._ckpt_dir is None:
            return
        self._pool_snapshot_counter += 1
        every = max(1, rust_cfg.pool_snapshot_every)
        if self._pool_snapshot_counter % every != 0:
            return
        pool_dir = Path(self._ckpt_dir) / "self_play" / "pool"
        try:
            _rotate_pool_snapshots(pool_dir, Path(source_path), rust_cfg.pool_size)
        except Exception as e:
            logger.warning("Failed to rotate pool snapshot: %s", e)

    def _export_champion(
        self,
        state_dict_override: dict | None = None,
        train_steps_override: int | None = None,
    ) -> None:
        """Export a model as the new champion for self-play.

        Called on promotion (trainee beat champion) and at stage start.
        Updates the self-play model file so the Rust binary hot-reloads it,
        and snapshots the champion into the past-self pool.

        Defaults to exporting the live training model. When promoting via
        SPRT, pass the state_dict + train_steps of the actual SPRT-validated
        trainee checkpoint so the champion matches what was measured (the
        live model may be several hundred steps newer and unvalidated).
        """
        if self._ckpt_dir is None:
            return
        sp_dir = Path(self._ckpt_dir) / "self_play"
        sp_dir.mkdir(parents=True, exist_ok=True)
        sp_model_path = str(sp_dir / "model_selfplay.pt")

        if state_dict_override is not None:
            sd = {k.removeprefix("_orig_mod."): v for k, v in state_dict_override.items()}
            export_steps = train_steps_override if train_steps_override is not None else self.train_steps
            source_label = f"validated checkpoint (step {export_steps})"
        else:
            sd = self.model.state_dict()
            export_steps = self.train_steps
            source_label = f"live model (step {export_steps})"

        # Save champion checkpoint (for eval-vs-champion comparisons)
        champion_path = str(sp_dir / "champion.pt")
        import os, tempfile
        fd, tmp = tempfile.mkstemp(dir=str(sp_dir), suffix=".pt.tmp")
        os.close(fd)
        mc_dict = {
            f.name: getattr(self.model_config, f.name)
            for f in self.model_config.__dataclass_fields__.values()
        }
        torch.save({"model_state_dict": sd, "model_config": mc_dict,
                    "train_steps": export_steps}, tmp)
        os.replace(tmp, champion_path)
        self._champion_path = champion_path

        # Also export to the self-play model path so Rust reloads. Embed
        # model_config so the inference server can read the (lean/relational)
        # schema and build a matching model with no extra self-play CLI plumbing.
        if self.self_play_config.rust.python_inference:
            fd2, tmp2 = tempfile.mkstemp(dir=str(sp_dir), suffix=".pt.tmp")
            os.close(fd2)
            torch.save({"model_state_dict": sd, "model_config": mc_dict}, tmp2)
            os.replace(tmp2, sp_model_path)
            self._snapshot_to_pool(sp_model_path)
        else:
            self._export_torchscript(sp_model_path, state_dict_override=sd)
            precision = self.self_play_config.precision
            if precision == "fp16":
                pool_src = sp_model_path.replace(".pt", "_fp16.pt")
            elif precision == "fp32":
                pool_src = sp_model_path.replace(".pt", "_fp32.pt")
            else:
                pool_src = sp_model_path
            self._snapshot_to_pool(pool_src)

        # Mirror the fresh self-play model to the remote actor so its periodic
        # reload picks up the new champion (no-op when no actor is configured).
        actor = self.self_play_config.actor
        if actor.host:
            self._actor_push_model(actor, sp_dir)

        logger.info("Champion exported from %s: %s", source_label, champion_path)
        self.log_event("champion", f"Exported from {source_label}")

    def _export_torchscript(self, path: str, state_dict_override: dict | None = None) -> None:
        """Export current model to TorchScript for the Rust binary.

        Exports three variants:
          - path              (bf16, default for self-play)
          - path_fp32.pt      (fp32)
          - path_fp16.pt      (fp16, for GPUs with fp16 tensor cores)
        """
        from hexo_a0.scriptable_model import export_torchscript
        if state_dict_override is not None:
            sd = state_dict_override
        else:
            sd = self.model.state_dict()
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        export_torchscript(sd, self.model_config, path, bf16=True)
        fp32_path = path.replace(".pt", "_fp32.pt")
        export_torchscript(sd, self.model_config, fp32_path, bf16=False)
        fp16_path = path.replace(".pt", "_fp16.pt")
        export_torchscript(sd, self.model_config, fp16_path, fp16=True)

    # ------------------------------------------------------------------
    # Optional remote self-play actor ([self_play.actor]). Every method below
    # is a no-op unless actor.host is set, so local runs are unaffected. See
    # ActorConfig + docs/research/2026-06-05-forward-pass-headroom for the
    # actor/learner split rationale (kills the single-GPU training<->self-play
    # contention by giving each workload its own device).
    # ------------------------------------------------------------------
    def _relpath_for_remote(self, p: str, cwd: "Path") -> str:
        """Make a path relative to cwd so it resolves under the actor's repo_dir.

        Relative paths pass through. Absolute paths inside cwd are relativised;
        absolute paths outside cwd are returned unchanged with a warning (they
        must then exist at that same absolute location on the actor).
        """
        pp = Path(p)
        if not pp.is_absolute():
            return p
        try:
            return str(pp.relative_to(cwd))
        except ValueError:
            logger.warning(
                "actor: path %s is outside cwd %s; passing through unchanged "
                "(ensure it exists at that path on the actor)", p, cwd,
            )
            return p

    def _rsync_ssh_opts(self, actor) -> list[str]:
        return ["-e", f"ssh {actor.ssh_opts}"] if actor.ssh_opts else []

    def _actorize_cmd(self, cmd: list[str], actor) -> tuple[list[str], dict]:
        """Wrap a local self_play command for execution on the remote actor host.

        Rewrites the binary path / --python-bin / --device to the actor's,
        relativises path args (so they resolve under the remote repo cwd), and
        wraps everything in ``ssh -tt <host> 'cd <repo> && mkdir -p … &&
        export LD_LIBRARY_PATH=… && exec <cmd>'``. The forced tty (``-tt``) makes
        the remote process receive SIGHUP and die when the local ssh is killed,
        so there are no orphaned GPU processes left on the actor. Returns
        ``(ssh_cmd, env)``; env is a clean copy (the remote env is in the ssh
        command string, not the local process env).
        """
        import os
        import shlex
        repo_dir = actor.repo_dir or "."
        python_bin = actor.python_bin or f"{repo_dir}/.venv/bin/python"
        cwd = Path.cwd()

        rcmd = list(cmd)
        rcmd[0] = f"{repo_dir}/hexo-rs/target/release/self_play"

        def _set_opt(opt: str, value: str) -> None:
            if opt in rcmd:
                rcmd[rcmd.index(opt) + 1] = value

        _set_opt("--device", actor.device)
        _set_opt("--python-bin", python_bin)
        for opt in ("--model", "--output-dir", "--checkpoint", "--pool-dir"):
            if opt in rcmd:
                i = rcmd.index(opt) + 1
                rcmd[i] = self._relpath_for_remote(rcmd[i], cwd)

        # Dirs that must exist on the actor before the binary starts.
        need_dirs: set[str] = set()
        if "--output-dir" in rcmd:
            need_dirs.add(rcmd[rcmd.index("--output-dir") + 1])
        if "--model" in rcmd:
            need_dirs.add(str(Path(rcmd[rcmd.index("--model") + 1]).parent))
        mkdir = (
            " mkdir -p " + " ".join(shlex.quote(d) for d in sorted(need_dirs)) + " &&"
            if need_dirs else ""
        )

        # LD_LIBRARY_PATH so the CPU-only self_play binary binds the actor's
        # (CUDA) libtorch. Derived on the actor from python_bin if torch_lib unset.
        if actor.torch_lib:
            ld = f"export LD_LIBRARY_PATH={shlex.quote(actor.torch_lib)}:${{LD_LIBRARY_PATH:-}}"
        else:
            code = shlex.quote(
                'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'
            )
            ld = (
                f'export LD_LIBRARY_PATH="$({shlex.quote(python_bin)} -c {code})"'
                ":${LD_LIBRARY_PATH:-}"
            )

        # Make the rsync'd hexo_a0 package importable by the inference_server
        # (which self_play spawns via --python-bin) WITHOUT installing it on the
        # actor: the package __init__ is eager, so $PWD/hexo-a0/src on PYTHONPATH
        # is enough. $PWD is repo_dir after the cd above. So the actor venv only
        # needs torch + torch_geometric + tensorboard, not the hexo_a0 package.
        pp = 'export PYTHONPATH="$PWD/hexo-a0/src:${PYTHONPATH:-}"'

        inner = " ".join(shlex.quote(a) for a in rcmd)
        remote = f"cd {shlex.quote(repo_dir)} &&{mkdir} {ld} && {pp} && exec {inner}"
        ssh_cmd = ["ssh", "-tt", *shlex.split(actor.ssh_opts), actor.host, remote]
        return ssh_cmd, os.environ.copy()

    def _actor_push_model(self, actor, sp_dir: "Path") -> None:
        """rsync the self-play model (and past-self pool, if enabled) to the actor.

        Pushes every ``model_selfplay*.pt`` (bf16 + any precision variants) so the
        actor's ``--model`` — whichever variant it selects — is fresh, and MIRRORS
        the past-self pool dir (``--delete``) when ``pool_fraction>0`` so pooled-
        opponent games actually work on the actor. The pool is otherwise written
        only on the learner, so without this the actor's ``--pool-dir`` stays empty
        and ~``pool_fraction`` of games silently degrade to vanilla self-play. The
        actor's periodic mtime reload then picks up the model.
        """
        import glob
        import subprocess
        repo_dir = actor.repo_dir or "."
        cwd = Path.cwd()
        ssh_e = self._rsync_ssh_opts(actor)

        model_files = sorted(glob.glob(str(sp_dir / "model_selfplay*.pt")))
        if model_files:
            rel_dir = self._relpath_for_remote(str(sp_dir), cwd)
            dst = f"{actor.host}:{repo_dir}/{rel_dir}/"
            try:
                subprocess.run(
                    [actor.rsync_bin, "-a", "--mkpath", *ssh_e, *model_files, dst],
                    check=True, capture_output=True, timeout=300,
                )
            except Exception as e:
                logger.warning("actor model-push failed: %s", e)

        # Mirror the past-self pool so --pool-dir is populated on the actor.
        pool_dir = sp_dir / "pool"
        if self.self_play_config.rust.pool_fraction > 0.0 and pool_dir.is_dir():
            rel_pool = self._relpath_for_remote(str(pool_dir), cwd)
            dst = f"{actor.host}:{repo_dir}/{rel_pool}/"
            try:
                subprocess.run(
                    [actor.rsync_bin, "-a", "--delete", "--mkpath", *ssh_e,
                     str(pool_dir) + "/", dst],
                    check=True, capture_output=True, timeout=300,
                )
            except Exception as e:
                logger.warning("actor pool-push failed: %s", e)

    def _start_actor_games_pull(self, actor, sp_dir: "Path") -> None:
        import threading
        (sp_dir / "batches").mkdir(parents=True, exist_ok=True)
        threading.Thread(
            target=self._actor_games_pull_loop, args=(actor, sp_dir), daemon=True,
        ).start()

    def _actor_games_pull_loop(self, actor, sp_dir: "Path") -> None:
        """Pull CLOSED game files from the actor into the local batches dir.

        Lists all but the newest (still-active) file on the actor and rsyncs
        them with ``--remove-source-files`` (closed files are complete → safe to
        move and to bound the actor's disk). The local ingestion loop then tails
        them exactly as for local self-play. Lower ``rust.max_file_mb`` for lower
        ingest latency (files close — and thus become pullable — more often).
        """
        import shlex
        import subprocess
        repo_dir = actor.repo_dir or "."
        local_batches = sp_dir / "batches"
        rel_batches = self._relpath_for_remote(str(sp_dir / "batches"), Path.cwd())
        ssh_base = ["ssh", *shlex.split(actor.ssh_opts), actor.host]
        list_cmd = (
            f"cd {shlex.quote(repo_dir)} && "
            f"ls -t {shlex.quote(rel_batches)}/*.bin {shlex.quote(rel_batches)}/*.jsonl "
            f"2>/dev/null | tail -n +2"
        )
        backoff = actor.games_pull_interval
        while not self._sp_stop.is_set():
            ok = False
            try:
                listing = subprocess.run(
                    ssh_base + [list_cmd], capture_output=True, text=True, timeout=30,
                )
                files = [f for f in listing.stdout.splitlines() if f.strip()]
                if files:
                    srcs = [f"{actor.host}:{repo_dir}/{f}" for f in files]
                    subprocess.run(
                        [actor.rsync_bin, "-a", "--remove-source-files",
                         *self._rsync_ssh_opts(actor), *srcs, str(local_batches) + "/"],
                        check=True, capture_output=True, timeout=300,
                    )
                ok = True
            except Exception as e:
                logger.warning("actor games-pull failed (%s); backing off", e)
            # Exponential backoff while the actor is unreachable (don't spam a
            # 30s-timeout every interval); snap back to the fast cadence on success.
            backoff = actor.games_pull_interval if ok else min(max(backoff * 2, 5.0), 60.0)
            self._sp_stop.wait(backoff)

    def _rust_self_play_worker(self, binary_path: str) -> None:
        """Background worker: run Rust self-play binary, read trajectories."""
        try:
            self._rust_self_play_worker_inner(binary_path)
        except Exception:
            logger.exception("Self-play worker crashed")

    def _rust_self_play_worker_inner(self, binary_path: str) -> None:
        import json
        import subprocess
        import tempfile

        gc = self.game_config

        # Set up output directory
        sp_dir = Path(self._ckpt_dir or ".") / "self_play"
        sp_dir.mkdir(parents=True, exist_ok=True)
        base_model_path = str(sp_dir / "model_selfplay.pt")

        rust_cfg = self.self_play_config.rust

        # In champion mode, export from the champion checkpoint rather than
        # the current training model. The champion is the strongest verified
        # model and should be what self-play uses.
        if self._champion_mode and self._champion_path and Path(self._champion_path).exists():
            import os, tempfile, torch
            ckpt = torch.load(self._champion_path, map_location="cpu", weights_only=True)
            sd = ckpt["model_state_dict"]
            if rust_cfg.python_inference:
                fd, tmp = tempfile.mkstemp(dir=str(sp_dir), suffix=".pt.tmp")
                os.close(fd)
                # Preserve the champion's embedded model_config (lean/relational
                # schema) so the inference server auto-configures from the ckpt.
                torch.save({"model_state_dict": sd,
                            "model_config": ckpt.get("model_config")}, tmp)
                os.replace(tmp, base_model_path)
            else:
                from hexo_a0.scriptable_model import export_torchscript
                sd_clean = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
                export_torchscript(sd_clean, self.model_config, base_model_path, bf16=True)
                fp32_path = base_model_path.replace(".pt", "_fp32.pt")
                export_torchscript(sd_clean, self.model_config, fp32_path, bf16=False)
                fp16_path = base_model_path.replace(".pt", "_fp16.pt")
                export_torchscript(sd_clean, self.model_config, fp16_path, fp16=True)
            model_path = base_model_path
            if not rust_cfg.python_inference:
                precision = self.self_play_config.precision
                if precision == "fp16":
                    model_path = base_model_path.replace(".pt", "_fp16.pt")
                elif precision == "fp32":
                    model_path = base_model_path.replace(".pt", "_fp32.pt")
            champ_step = ckpt.get("train_steps", "?")
            logger.info("Champion mode: exported champion (step %s) to self-play model %s", champ_step, model_path)
            self.log_event("champion", f"Self-play started from champion (step {champ_step})")
            self._snapshot_to_pool(model_path)
        elif rust_cfg.python_inference:
            # Python subprocess loads raw state dict, not TorchScript
            import os, tempfile, torch
            sd = self.model.state_dict()
            mc_dict = {
                f.name: getattr(self.model_config, f.name)
                for f in self.model_config.__dataclass_fields__.values()
            }
            fd, tmp = tempfile.mkstemp(dir=str(sp_dir), suffix=".pt.tmp")
            os.close(fd)
            torch.save({"model_state_dict": sd, "model_config": mc_dict}, tmp)
            os.replace(tmp, base_model_path)
            model_path = base_model_path
            logger.info("Python inference: saved checkpoint %s", model_path)
            self._snapshot_to_pool(model_path)
        else:
            # Export all precision variants (bf16, fp32, fp16)
            self._export_torchscript(base_model_path)
            # Select the right variant for the Rust binary
            precision = self.self_play_config.precision
            if precision == "fp16":
                model_path = base_model_path.replace(".pt", "_fp16.pt")
            elif precision == "fp32":
                model_path = base_model_path.replace(".pt", "_fp32.pt")
            else:
                model_path = base_model_path
            logger.info("Rust self-play: using model %s", model_path)
            # Snapshot the precision-matched export into the past-self pool
            self._snapshot_to_pool(model_path)
        pool_dir = sp_dir / "pool"

        # Find libtorch path
        import torch
        torch_lib = str(Path(torch.__file__).parent / "lib")

        # Build command
        graph_type = self.model_config.graph_type
        sp_mcts = self.self_play_config.mcts  # [self_play.mcts] σ overrides
        cmd = [
            binary_path,
            "--model", model_path,
            "--graph-type", graph_type,
            "--games", str(self.self_play_config.python.games_per_write),
            "--device", self.self_play_config.device,
            "--win-length", str(gc.win_length),
            "--radius", str(gc.placement_radius),
            "--max-moves", str(gc.max_moves),
            "--sims", str(self.mcts_config.n_simulations),
            "--m-actions", str(self.mcts_config.m_actions),
            # Self-play σ constants: [self_play.mcts] overrides [mcts] when set,
            # so self-play can soften the distilled target (lower c_scale) while
            # eval/SPRT keep the peaked paper default. See cscale-target-reshape.
            "--c-scale", str(
                sp_mcts.c_scale if sp_mcts.c_scale is not None
                else self.mcts_config.c_scale
            ),
            "--c-visit", str(
                sp_mcts.c_visit if sp_mcts.c_visit is not None
                else self.mcts_config.c_visit
            ),
            "--exploration", str(self.mcts_config.exploration_moves),
            *(["--exploration-fraction", str(self.mcts_config.exploration_fraction)]
              if self.mcts_config.exploration_fraction > 0
              else []),
            *(["--exploration-max", str(self.mcts_config.exploration_max),
               "--exploration-window", str(self.mcts_config.exploration_window),
               "--exploration-exponent", str(self.mcts_config.exploration_exponent)]
              if self.mcts_config.exploration_max > 0
              else []),
            *(["--exploration-distribution", self.mcts_config.exploration_distribution]
              if self.mcts_config.exploration_distribution != "improved_policy"
              else []),
            *(["--truncate-delta", str(self.mcts_config.truncate_delta)]
              if 0 < self.mcts_config.truncate_delta and math.isfinite(self.mcts_config.truncate_delta)
              else []),
            *(["--draw-value", str(self.training_config.draw_value)]
              if self.training_config.draw_value != 0.0
              else []),
            "--continuous",
            "--output-dir", str(sp_dir / "batches"),
        ]
        # Pass worker count if configured (0 = auto in both Python and Rust)
        n_workers = self.self_play_config.workers
        if n_workers > 0:
            cmd.extend(["--threads", str(n_workers)])
        # Rust engine settings
        rust_cfg = self.self_play_config.rust
        cmd.extend(["--max-batch", str(rust_cfg.max_batch)])
        # Edge-targeted batching: only emit when enabled so configs that don't
        # use it produce a byte-identical command line.
        if getattr(rust_cfg, "max_batch_edges", 0) > 0:
            cmd.extend(["--max-batch-edges", str(rust_cfg.max_batch_edges)])
        cmd.extend(["--batch-timeout-ms", str(rust_cfg.batch_timeout_ms)])
        cmd.extend(["--max-file-mb", str(rust_cfg.max_file_mb)])
        cmd.extend(["--output-filename", rust_cfg.output_filename])
        # Past-self opponent pool: only pass flags when enabled, so existing
        # runs without pool config produce a byte-identical command line.
        if rust_cfg.pool_fraction > 0.0:
            cmd.extend(["--pool-dir", str(pool_dir)])
            cmd.extend(["--pool-fraction", str(rust_cfg.pool_fraction)])

        # Wu (2019) playout cap randomization: only emit flags when both
        # parameters are non-default, so existing runs are byte-identical.
        if rust_cfg.padded_inference:
            cmd.append("--padded-inference")
        if self.model_config.prune_empty_edges:
            cmd.append("--prune-empty-edges")
        if getattr(self.model_config, "threat_features", False):
            cmd.append("--threat-features")
        if getattr(self.model_config, "relative_stone_encoding", False):
            cmd.append("--relative-stones")

        if rust_cfg.playout_cap_fraction > 0.0 and rust_cfg.playout_cap_divisor > 1:
            cmd.extend(["--playout-cap-fraction", str(rust_cfg.playout_cap_fraction)])
            cmd.extend(["--playout-cap-divisor", str(rust_cfg.playout_cap_divisor)])

        if rust_cfg.virtual_loss > 0.0:
            cmd.extend(["--virtual-loss", str(rust_cfg.virtual_loss)])

        # Root Dirichlet noise (AlphaZero-style root exploration applied
        # before Gumbel-Top-k). Only emit flags when alpha > 0 so existing
        # runs produce a byte-identical command line.
        if self.mcts_config.root_dirichlet_alpha > 0.0:
            cmd.extend([
                "--root-dirichlet-alpha", str(self.mcts_config.root_dirichlet_alpha),
                "--root-dirichlet-fraction", str(self.mcts_config.root_dirichlet_fraction),
            ])

        # Python subprocess inference (torch.compile, ~2x faster forward pass)
        if rust_cfg.python_inference:
            import sys
            cmd.extend([
                "--python-inference",
                "--python-bin", sys.executable,
                "--checkpoint", model_path,
                "--model-hidden-dim", str(self.model_config.hidden_dim),
                "--model-num-layers", str(self.model_config.num_layers),
                "--model-num-heads", str(self.model_config.num_heads),
                "--model-policy-hidden", str(self.model_config.policy_hidden),
                "--model-value-hidden", str(self.model_config.value_hidden),
                "--model-conv-type", str(getattr(self.model_config, "conv_type", "gatv2")),
            ])
            # Forward JK flags so the Rust self_play binary can pass them on
            # to the Python inference subprocess. Only emit when JK is on, to
            # keep the subprocess command byte-identical for non-JK runs.
            if getattr(self.model_config, "use_jk", False):
                cmd.append("--model-use-jk")
                cmd.extend([
                    "--model-jk-mode",
                    str(getattr(self.model_config, "jk_mode", "sum")),
                ])
            # Only emit the multi-batcher flag when explicitly configured,
            # so existing configs produce a byte-identical CLI.
            if rust_cfg.python_inference_workers > 1:
                cmd.extend([
                    "--python-inference-workers",
                    str(rust_cfg.python_inference_workers),
                ])

        import os
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"

        if not rust_cfg.python_inference:
            # Preload GPU libtorch so the CUDA/HIP backend is registered
            # before the Rust binary tries to use Device::Cuda.
            # Not needed for python_inference — the subprocess handles GPU.
            for gpu_lib_name in ("libtorch_hip.so", "libtorch_cuda.so"):
                gpu_lib = Path(torch_lib) / gpu_lib_name
                if gpu_lib.exists():
                    existing = env.get("LD_PRELOAD", "")
                    env["LD_PRELOAD"] = f"{gpu_lib}:{existing}" if existing else str(gpu_lib)
                    break

        # Optional remote actor: push the model and wrap the command in ssh.
        # Training stays local. The games-pull thread is started AFTER the
        # initial_files snapshot below (not here), so every pulled closed file is
        # read from the start and never mistaken for a stale startup file to be
        # tailed-from-end. No-op when host is unset.
        actor = self.self_play_config.actor
        if actor.host:
            self._actor_push_model(actor, sp_dir)
            cmd, env = self._actorize_cmd(cmd, actor)

        logger.info("Rust self-play: %s", " ".join(cmd))
        (sp_dir / "batches").mkdir(parents=True, exist_ok=True)

        # Launch the binary — stderr goes to a log file to avoid pipe buffer deadlock
        sp_stderr_path = sp_dir / "self_play.log"
        self._sp_stderr_file = open(sp_stderr_path, "w")
        # stdin=DEVNULL matters in actor mode: the command is `ssh -tt …`, and an
        # ssh with a controlling tty on its stdin would put THIS terminal into raw
        # mode (dropping ONLCR), cascade-indenting the trainer/SPRT status lines.
        # Detaching stdin keeps ssh off the local terminal while -tt still allocates
        # the REMOTE pty for SIGHUP-on-disconnect cleanup.
        #
        # Output routing differs by mode. The pty that -tt allocates has NO separate
        # stderr channel, so ssh sends the remote process's COMBINED stdout+stderr
        # back on the local ssh's *stdout*. So in actor mode the log must capture
        # stdout (else self_play.log stays blank); locally only the binary's stderr
        # carries the [perf] log (its stdout is unused — games are written to files).
        if actor.host:
            sp_stdout = self._sp_stderr_file
            sp_stderr = subprocess.STDOUT
        else:
            sp_stdout = subprocess.DEVNULL
            sp_stderr = self._sp_stderr_file
        self._sp_process = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.DEVNULL,
            stdout=sp_stdout, stderr=sp_stderr,
        )
        self._sp_launch_t = time.time()

        # Tail all game files in batches dir and ingest games as they appear.
        # Supports .msgpack (length-prefixed binary) and .jsonl (text) files.
        # Supports multiple writers (e.g. remote self-play via SSHFS).
        batches_dir = sp_dir / "batches"
        watched_files: dict[Path, object] = {}  # path -> open file handle
        known_paths: set[Path] = set()

        # Remember which files exist at startup so we can tail them from the
        # end (avoid replaying stale data).  Files discovered later — e.g.
        # after a rotation during a long eval — are read from the beginning
        # so no data is skipped.
        initial_files: set[Path] = set()
        for pattern in ("*.bin", "*.jsonl"):
            initial_files.update(batches_dir.glob(pattern))

        # Now that the startup snapshot is captured, begin pulling games from the
        # remote actor: any file it delivers is absent from initial_files and so
        # is read from the start (never tailed-from-end / skipped).
        if actor.host:
            self._start_actor_games_pull(actor, sp_dir)

        _INGEST_BATCH = 50
        _INGEST_INTERVAL = 0.5  # seconds between ingestion bursts

        while not self._sp_stop.is_set():
            # Check if process died
            if self._sp_process.poll() is not None:
                rc = self._sp_process.returncode
                try:
                    tail = (sp_dir / "self_play.log").read_text()[-400:]
                except Exception:
                    tail = ""
                if actor.host and not self._sp_stop.is_set():
                    # Remote actor died — almost always a transient bazzite/Tailscale
                    # drop (ssh exits 255), NOT a real crash. Auto-restart with capped
                    # backoff so the run self-heals instead of silently starving the
                    # buffer (reuse -> 0). Reset backoff if it had been healthy a while.
                    if time.time() - getattr(self, "_sp_launch_t", 0) > 120:
                        self._actor_backoff_s = 0
                    backoff = min(max(getattr(self, "_actor_backoff_s", 0) * 2, 5), 60)
                    self._actor_backoff_s = backoff
                    logger.warning(
                        "Remote self-play (actor %s) exited code %d; restarting in %ds. Tail: %s",
                        actor.host, rc, backoff, tail or "(no output)",
                    )
                    if self._sp_stop.wait(backoff):
                        break
                    try:
                        self._actor_push_model(actor, sp_dir)
                        try:
                            self._sp_stderr_file.close()
                        except Exception:
                            pass
                        self._sp_stderr_file = open(sp_dir / "self_play.log", "a")
                        self._sp_process = subprocess.Popen(
                            cmd, env=env, stdin=subprocess.DEVNULL,
                            stdout=self._sp_stderr_file, stderr=subprocess.STDOUT,
                        )
                        self._sp_launch_t = time.time()
                        logger.info("Remote self-play restarted on %s", actor.host)
                    except Exception as e:
                        logger.warning("Remote self-play restart failed (%s); will retry", e)
                    continue
                # Local mode, or shutting down: terminal.
                try:
                    self._sp_stderr_file.close()
                except Exception:
                    pass
                logger.error("Rust self-play process exited with code %d: %s", rc, tail or "(no output)")
                break

            # Discover new files (.bin or .jsonl)
            for pattern in ("*.bin", "*.jsonl"):
                for p in batches_dir.glob(pattern):
                    if p not in known_paths:
                        try:
                            fh = open(p, "rb" if p.suffix == ".bin" else "r")
                            if p in initial_files:
                                fh.seek(0, 2)  # startup file: tail from end
                                logger.info("Watching file: %s (tailing from end)", p.name)
                            else:
                                logger.info("Watching file: %s (reading from start)", p.name)
                            watched_files[p] = fh
                            known_paths.add(p)
                        except OSError:
                            pass

            if not watched_files:
                time.sleep(0.1)
                continue

            # Detect truncated files (e.g. Rust binary recreated the file)
            for path, fh in list(watched_files.items()):
                pos = fh.tell()
                try:
                    file_size = path.stat().st_size
                except OSError:
                    continue
                if pos > file_size:
                    logger.info("File %s was truncated (pos %d > size %d), re-tailing from start",
                                path.name, pos, file_size)
                    fh.seek(0)

            batch: list[dict] = []
            try:
                for path, fh in list(watched_files.items()):
                    if path.suffix == ".bin":
                        self._read_binary_records(fh, batch, _INGEST_BATCH)
                    else:
                        self._read_jsonl_records(fh, batch, _INGEST_BATCH)
                    if len(batch) >= _INGEST_BATCH:
                        break

                if batch:
                    examples = self._trajectories_to_examples(batch)
                    entropies, top1s, values = _example_histogram_stats(examples)
                    self.buffer.add_many(examples)
                    with self._sp_lock:
                        self._sp_games += len(batch)
                        self._sp_policy_entropies.extend(entropies)
                        self._sp_policy_top1.extend(top1s)
                        self._sp_value_targets.extend(values)
                        for g in batch:
                            n_examples = g["length"]
                            self._sp_game_examples.append(n_examples)
                            # Fall back to example count for sources that
                            # don't carry move_count (legacy JSONL path).
                            self._sp_game_move_counts.append(
                                g.get("move_count", n_examples)
                            )
                            w = g.get("winner")
                            if w == "P1":
                                self._sp_p1_wins += 1
                            elif w == "P2":
                                self._sp_p2_wins += 1
                            ws = g.get("window_stats")
                            if ws and all(
                                math.isfinite(ws[k])
                                for k in ("mass_removed_sum", "acted_deficit_sum")
                            ):
                                self._sp_window_stats.append(ws)
                            elif ws is not None:
                                logger.warning(
                                    "dropping non-finite HX06 window stats "
                                    "(mass_removed_sum=%r, acted_deficit_sum=%r) "
                                    "— possible wire corruption",
                                    ws.get("mass_removed_sum"),
                                    ws.get("acted_deficit_sum"),
                                )
            except Exception as e:
                logger.error("Self-play ingestion error: %s", e, exc_info=True)

            # Clean up fully-consumed rotated files. A file is eligible
            # for deletion when we're at EOF and a newer file from the
            # same writer (same stem prefix) exists. Files are named
            # e.g. games_0001.bin, games_0002.bin (same writer "games"),
            # vs games_rigel_0001.bin (writer "games_rigel").
            for path in list(watched_files):
                fh = watched_files[path]
                # Check if at EOF (no new data)
                pos = fh.tell()
                peek = fh.read(1)
                if peek:
                    fh.seek(pos)
                    continue
                # Extract the writer stem: everything before the last _NNNN
                stem = path.stem  # e.g. "games_0001"
                parts = stem.rsplit("_", 1)
                if len(parts) != 2 or not parts[1].isdigit():
                    continue  # not a rotated file, skip
                writer_prefix = parts[0]  # e.g. "games" or "games_rigel"
                # Check if a newer file from the same writer exists
                newer_exists = any(
                    p != path
                    and p.stem.rsplit("_", 1)[0] == writer_prefix
                    and p.stat().st_mtime >= path.stat().st_mtime
                    for p in batches_dir.glob(f"{writer_prefix}_*{path.suffix}")
                )
                if newer_exists:
                    fh.close()
                    try:
                        path.unlink()
                        logger.info("Cleaned up consumed file: %s", path.name)
                    except OSError:
                        pass
                    del watched_files[path]
                    known_paths.discard(path)

            time.sleep(_INGEST_INTERVAL)

        for fh in watched_files.values():
            fh.close()

    _RECORD_MAGIC_HX05 = b"HX05"
    _RECORD_MAGIC_HX04 = b"HX04"
    _RECORD_MAGIC_LEGACY = b"HX03"
    _RECORD_MAGIC_HX06 = b"HX06"
    _RECORD_MAGIC_HX07 = b"HX07"

    @staticmethod
    def _read_binary_records(fh, batch: list, max_records: int) -> None:
        """Read length-prefixed binary game records.

        Supported formats:
          - ``HX06`` (Q-margin gate on): same as HX05 but magic = "HX06" and
            12B window telemetry after node_dim: [2B in_window_moves u16]
            [2B gated_moves u16][4B mass_removed_sum f32]
            [4B acted_deficit_sum f32]. record_size counts the 12B (so +12
            vs HX05). Written IFF --truncate-delta is set in self_play.
          - ``HX05`` (default when gate off; HX06 = gated variant with window
            stats): adds a 1B node_dim to the envelope. Node features are
            7/8/11/12-dim depending on --relative-stones and
            --threat-features.
          - ``HX04``: adds 4B sample_weight per example after the value
            field; node features always 8-dim. With PCR active, fast-cap
            positions are written with weight = 1/divisor instead of being
            discarded.
          - ``HX03`` (legacy): no per-example sample_weight; defaulted to 1.0
            for back-compat. Produced by pre-2026-05-02 self-play binaries.

        Record envelope: ``[4B magic][4B LE size][4B LE num_examples]
        [1B winner i8][4B LE move_count u32][1B node_dim u8 -- HX05/HX06]
        [12B window_stats -- HX06 only][examples...]``.

        ``winner`` is encoded as 1 = P1, -1 = P2, 0 = draw.

        On corruption, scans forward for the next supported magic marker.
        Legacy ``HX01`` and ``HX02`` records are unsupported and skipped via
        the resync path.
        """
        import struct

        while len(batch) < max_records:
            pos = fh.tell()
            header = fh.read(8)  # magic (4) + size (4)
            if len(header) < 8:
                fh.seek(pos)
                break

            magic = header[:4]
            is_hx07 = False
            if magic == Trainer._RECORD_MAGIC_HX07:
                # Board-state record: no node_dim byte; a 1B has_window flag
                # sits at offset 9 (body starts at 10, or 22 with window stats).
                is_hx07 = True
                has_sample_weight = True
                has_node_dim = False
                has_window_stats = False  # decided per-record via the flag byte
            elif magic == Trainer._RECORD_MAGIC_HX06:
                has_sample_weight = True
                has_node_dim = True
                has_window_stats = True
            elif magic == Trainer._RECORD_MAGIC_HX05:
                has_sample_weight = True
                has_node_dim = True
                has_window_stats = False
            elif magic == Trainer._RECORD_MAGIC_HX04:
                has_sample_weight = True
                has_node_dim = False
                has_window_stats = False
            elif magic == Trainer._RECORD_MAGIC_LEGACY:
                has_sample_weight = False
                has_node_dim = False
                has_window_stats = False
            else:
                if magic == b"HX01":
                    logger.warning(
                        "Encountered legacy HX01 record at offset %d; skipping (no winner byte)",
                        pos,
                    )
                elif magic == b"HX02":
                    logger.warning(
                        "Encountered legacy HX02 record at offset %d; skipping (no move_count)",
                        pos,
                    )
                else:
                    logger.warning(
                        "Binary format desync at offset %d, scanning for next record", pos
                    )
                fh.seek(pos + 1)
                _resync_binary(fh)
                continue

            record_size = struct.unpack("<I", header[4:8])[0]
            data = fh.read(record_size)
            if len(data) < record_size:
                fh.seek(pos)  # incomplete record, retry later
                break
            try:
                num_examples = struct.unpack_from("<I", data, 0)[0]
                winner_byte = struct.unpack_from("<b", data, 4)[0]
                if winner_byte == 1:
                    winner = "P1"
                elif winner_byte == -1:
                    winner = "P2"
                else:
                    winner = "draw"
                move_count = struct.unpack_from("<I", data, 5)[0]
                if is_hx07:
                    # has_window flag at offset 9; window stats (if present) at
                    # offset 10; body at offset 10 (no window) or 22 (window).
                    has_window = data[9] != 0
                    offset = 10
                    window_stats = None
                    if has_window:
                        if len(data) < offset + 12:
                            raise ValueError(
                                f"HX07 record truncated: window stats need "
                                f"{offset + 12} bytes, buffer has {len(data)}")
                        in_window_moves, gated_moves, mass_removed_sum, acted_deficit_sum = \
                            struct.unpack_from("<HHff", data, offset)
                        window_stats = {
                            "in_window_moves": in_window_moves,
                            "gated_moves": gated_moves,
                            "mass_removed_sum": mass_removed_sum,
                            "acted_deficit_sum": acted_deficit_sum,
                        }
                        offset += 12
                    examples = []
                    for _ in range(num_examples):
                        ex, offset = _parse_binary_example_hx07(data, offset)
                        examples.append(ex)
                    batch.append({
                        "length": num_examples,
                        "examples": examples,
                        "winner": winner,
                        "move_count": move_count,
                        "node_dim": None,
                        "window_stats": window_stats,
                        "format": "state",
                    })
                    continue
                if has_node_dim:
                    node_dim = data[9]
                    offset = 10
                else:
                    node_dim = 8
                    offset = 9
                window_stats = None
                if has_window_stats:
                    in_window_moves, gated_moves, mass_removed_sum, acted_deficit_sum = \
                        struct.unpack_from("<HHff", data, offset)
                    window_stats = {
                        "in_window_moves": in_window_moves,
                        "gated_moves": gated_moves,
                        "mass_removed_sum": mass_removed_sum,
                        "acted_deficit_sum": acted_deficit_sum,
                    }
                    offset += 12
                examples = []
                for _ in range(num_examples):
                    ex, offset = _parse_binary_example(
                        data, offset, has_sample_weight, node_dim
                    )
                    examples.append(ex)
                batch.append({
                    "length": num_examples,
                    "examples": examples,
                    "winner": winner,
                    "move_count": move_count,
                    "node_dim": node_dim,
                    "window_stats": window_stats,
                    "format": "graph",
                })
            except Exception:
                logger.warning("Skipping corrupted binary record (%d bytes)", record_size)

    @staticmethod
    def _read_jsonl_records(fh, batch: list, max_records: int) -> None:
        """Read JSONL records from a text file handle."""
        while len(batch) < max_records:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.endswith("\n"):
                fh.seek(pos)
                break
            line = line.strip()
            if not line:
                continue
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupted JSONL line (%d chars)", len(line))

    def _trajectories_to_examples(self, games: list[dict]) -> list:
        """Convert Rust self-play records to TrainingExamples.

        Records are tagged with ``"format"``: ``"state"`` (HX07 board-state
        records → reconstruct via ``GameState.from_state``) or ``"graph"``
        (legacy HX05/HX06 graph records → legacy drain). Untagged records
        (e.g. JSONL) are sniffed: ``"state"`` if the first example carries
        ``"stones"``, else ``"graph"``.
        """
        all_examples = []
        for game_data in games:
            if "examples" not in game_data:
                logger.warning("Skipping record without 'examples' (legacy 'moves' "
                               "format is no longer ingested)")
                continue
            fmt = game_data.get("format")
            if fmt is None:
                exs = game_data["examples"]
                fmt = "state" if (exs and "stones" in exs[0]) else "graph"
            if fmt == "state":
                all_examples.extend(self._parse_state_examples(game_data))
            else:
                all_examples.extend(self._parse_graph_examples(game_data))
        return all_examples

    def _parse_state_examples(self, game_data: dict) -> list:
        """Parse HX07 board-state records into lazy-reconstruction examples.

        Each example carries board state (stones + current player +
        moves_remaining) rather than a pre-built graph. We rebuild a
        ``GameState`` via ``GameState.from_state`` and store THAT; the graph is
        reconstructed at load (matching the Phase-1 graph-record path). Policy
        length must equal the reconstructed legal-move count or the example is
        skipped (logged), so a stale/corrupt record cannot poison training.
        """
        import hexo_rs

        examples, skipped = [], 0
        for ex in game_data["examples"]:
            gs = hexo_rs.GameState.from_state(
                ex["stones"], ex["current_player"], ex["moves_remaining"],
                self.game_config,
            )
            policy = torch.tensor(ex["policy"], dtype=torch.float32)
            if policy.shape[0] != len(gs.legal_moves()):
                skipped += 1
                logger.warning(
                    "HX07 example policy length %d != legal moves %d; skipping",
                    policy.shape[0], len(gs.legal_moves()))
                continue
            examples.append(TrainingExample(
                policy_target=policy,
                value_target=float(ex["value"]),
                game_state=gs,
                sample_weight=float(ex.get("sample_weight", 1.0)),
            ))
        total = len(examples) + skipped
        if total:
            rate = skipped / total
            if rate > 0.01:
                logger.error("ingestion skip rate %.1f%% (%d/%d) — possible encoding "
                             "mismatch or stale/corrupt HX07 records; investigate",
                             100 * rate, skipped, total)
            if rate > 0.25:
                raise RuntimeError(
                    f"HX07 policy-length skip rate {100*rate:.0f}% ({skipped}/{total}) "
                    "— systemic encoding mismatch; aborting ingestion")
        return examples

    def _parse_graph_examples(self, game_data: dict) -> list:
        """Parse graph-format records (production: Rust HX05 binary).

        Derives the compact game_state from each shipped graph and stores THAT;
        the graph is rebuilt at load. ~98% fewer bytes/example vs storing graphs.
        See docs/research/2026-06-17-graph-reconstruction-storage.md.
        """
        from hexo_a0.graph import (
            _raw_to_axis_data, _raw_to_data, reconstruct_game_state,
        )
        graph_type = self.model_config.graph_type
        raw_to_data = _raw_to_axis_data if graph_type == "axis" else _raw_to_data

        examples, skipped = [], 0
        for ex in game_data["examples"]:
            data = raw_to_data(ex)
            policy = torch.tensor(ex["policy"], dtype=torch.float32)
            n_legal_mask = int(data.legal_mask.sum().item())
            if policy.shape[0] != n_legal_mask:
                skipped += 1
                continue
            try:
                game_state = reconstruct_game_state(data, self.game_config)
            except ValueError:
                skipped += 1
                logger.warning("reconstruct_game_state failed on a graph example",
                               exc_info=True)
                continue
            examples.append(TrainingExample(
                policy_target=policy,
                value_target=float(ex["value"]),
                game_state=game_state,
                sample_weight=float(ex.get("sample_weight", 1.0)),
            ))
        total = len(examples) + skipped
        if total:
            rate = skipped / total
            if rate > 0.01:
                logger.error("ingestion skip rate %.1f%% (%d/%d) — possible encoding "
                             "mismatch or reconstruct_game_state bug; investigate",
                             100 * rate, skipped, total)
            if rate > 0.25:
                raise RuntimeError(
                    f"reconstruct_game_state skip rate {100*rate:.0f}% ({skipped}/{total}) "
                    "— systemic encoding mismatch; aborting ingestion")
        return examples

    def pause_self_play(self) -> None:
        """Temporarily freeze the Rust self-play subprocess (SIGSTOP)."""
        if hasattr(self, "_sp_process") and self._sp_process is not None:
            import signal
            try:
                self._sp_process.send_signal(signal.SIGSTOP)
            except OSError:
                pass

    def resume_self_play(self) -> None:
        """Resume the Rust self-play subprocess (SIGCONT)."""
        if hasattr(self, "_sp_process") and self._sp_process is not None:
            import signal
            try:
                self._sp_process.send_signal(signal.SIGCONT)
            except OSError:
                pass

    def stop_self_play(self) -> None:
        """Stop background self-play workers."""
        if self._sp_pool is None:
            return
        self._sp_stop.set()
        logger.info("Stopping self-play workers...")
        # Kill the Rust subprocess if running
        if hasattr(self, "_sp_process") and self._sp_process is not None:
            self._sp_process.terminate()
            try:
                self._sp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._sp_process.kill()
            self._sp_process = None
        self._sp_pool.shutdown(wait=True, cancel_futures=True)
        self._sp_pool = None
        logger.info("Self-play workers stopped.")

    def _collect_self_play_stats(self) -> _SelfPlayReport:
        """Drain self-play counters and per-example histogram data.

        ``mean_game_length`` is the average actual move count per game,
        read from the binary record's ``move_count`` field. With playout
        cap randomization this can exceed the per-game training-example
        count, because fast-search positions advance the game without
        producing a recorded example. ``examples_written`` is the sum of
        per-game training-example counts (the number of full-search
        positions across all drained games), used by the sample-reuse
        metric. Raw per-game and per-example lists are returned for
        histogram logging.
        """
        with self._sp_lock:
            games = self._sp_games
            example_counts = self._sp_game_examples.copy()
            move_counts = self._sp_game_move_counts.copy()
            p1_wins = self._sp_p1_wins
            p2_wins = self._sp_p2_wins
            policy_entropies = self._sp_policy_entropies.copy()
            policy_top1 = self._sp_policy_top1.copy()
            value_targets = self._sp_value_targets.copy()
            window_stats = self._sp_window_stats.copy()
            self._sp_games = 0
            self._sp_game_examples.clear()
            self._sp_game_move_counts.clear()
            self._sp_p1_wins = 0
            self._sp_p2_wins = 0
            self._sp_policy_entropies.clear()
            self._sp_policy_top1.clear()
            self._sp_value_targets.clear()
            self._sp_window_stats.clear()
        mean_len = (
            sum(move_counts) / len(move_counts) if move_counts else 0.0
        )
        examples_written = sum(example_counts)
        return _SelfPlayReport(
            games=games,
            mean_game_length=mean_len,
            p1_wins=p1_wins,
            p2_wins=p2_wins,
            examples_written=examples_written,
            game_lengths=move_counts,
            policy_entropies=policy_entropies,
            policy_top1=policy_top1,
            value_targets=value_targets,
            window_stats=window_stats,
        )

    # ------------------------------------------------------------------
    # Reanalyze
    # ------------------------------------------------------------------

    def _reanalyze_batch(self, examples: list) -> list:
        """Re-run MCTS on sampled positions using current model to refresh policy targets."""
        if self.training_config.reanalyze_fraction <= 0 or len(examples) == 0:
            return examples

        n_reanalyze = int(len(examples) * self.training_config.reanalyze_fraction)
        if n_reanalyze == 0:
            return examples

        tc = self.training_config
        chunk_size = max(1, tc.reanalyze_chunk)
        sims = tc.reanalyze_sims or None
        m_act = tc.reanalyze_m_actions or None
        if not getattr(self, "_reanalyze_logged", False):
            self._reanalyze_logged = True
            logger.info(
                "reanalyze enabled: fraction=%.2f sims=%s m_actions=%s chunk=%d",
                tc.reanalyze_fraction, sims or self.mcts_config.n_simulations,
                m_act or self.mcts_config.m_actions, chunk_size,
            )

        self.model.eval()

        threat = getattr(self.model_config, "threat_features", False)
        rel = getattr(self.model_config, "relative_stone_encoding", False)
        if self.model_config.graph_type == "axis":
            from hexo_a0.graph import game_to_axis_graph, game_to_axis_graph_batch
            prune = self.model_config.prune_empty_edges
            graph_fn = lambda g: game_to_axis_graph(g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
            graph_batch_fn = lambda gs: game_to_axis_graph_batch(gs, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
        else:
            from hexo_a0.graph import game_to_graph, game_to_graph_batch
            graph_fn = lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel)
            graph_batch_fn = lambda gs: game_to_graph_batch(gs, threat_features=threat, relative_stones=rel)

        # Sample positions to reanalyze. Every example is a candidate: those
        # missing `game_state` (legacy buffer entries pickled before
        # PyGameState had pickle support) can be reconstructed from `data`
        # using the current GameConfig — safe because the curriculum clears
        # the replay buffer on every stage transition.
        candidates = list(range(len(examples)))
        if not candidates:
            return examples

        reanalyze_indices = random.sample(candidates, min(n_reanalyze, len(candidates)))

        from hexo_a0.graph import reconstruct_game_state
        from hexo_a0.self_play import _rust_batched_gumbel_mcts

        n_failed = 0
        for start in range(0, len(reanalyze_indices), chunk_size):
            idx_chunk = reanalyze_indices[start : start + chunk_size]
            # Materialise game states; positions that cannot be reconstructed
            # or are terminal are counted as failures and skipped.
            chunk: list[tuple[int, object]] = []
            for idx in idx_chunk:
                ex = examples[idx]
                try:
                    game_state = ex.game_state
                    if game_state is None:
                        # Legacy example: rebuild from graph + current config.
                        game_state = reconstruct_game_state(ex.data, self.game_config)
                    if game_state.is_terminal():
                        n_failed += 1
                        continue
                    chunk.append((idx, game_state))
                except Exception:
                    n_failed += 1
                    logger.debug("reanalyze: state unavailable for example %d", idx, exc_info=True)
            if not chunk:
                continue
            try:
                policies = _rust_batched_gumbel_mcts(
                    [gs for _, gs in chunk], self.model, self.mcts_config, self.device,
                    n_simulations=sims, m_actions=m_act,
                    graph_fn=graph_fn, graph_batch_fn=graph_batch_fn,
                )
            except Exception:
                # Never silently: a 100%-failure mode (e.g. passing the wrong
                # config object) previously masqueraded as a working no-op
                # for weeks.
                n_failed += len(chunk)
                logger.debug("reanalyze: chunk of %d failed", len(chunk), exc_info=True)
                continue
            for (idx, game_state), new_policy in zip(chunk, policies):
                ex = examples[idx]
                n_legal = len(game_state.legal_moves())
                if new_policy.shape[0] != n_legal:
                    n_failed += 1
                    logger.debug(
                        "reanalyze: policy length %d != legal count %d for example %d",
                        new_policy.shape[0], n_legal, idx,
                    )
                    continue
                # Create updated example with new policy but same value target.
                examples[idx] = TrainingExample(
                    policy_target=new_policy,
                    value_target=ex.value_target,
                    game_state=game_state,
                    trajectory_id=ex.trajectory_id,
                    sample_weight=ex.sample_weight,
                )

        if n_failed:
            logger.warning(
                "reanalyze: %d/%d sampled positions failed (reconstruction, "
                "terminal, or MCTS error) — policy targets NOT refreshed for those",
                n_failed, len(reanalyze_indices),
            )

        self.model.train()
        return examples

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        steps: int | None = None,
        step_callback: "Callable[[int], None] | None" = None,
    ) -> dict:
        """Run N gradient steps, then report metrics and check eval/checkpoint.

        If no background self-play is running, generates games inline first
        (alternating mode).

        Args:
            steps: Number of gradient steps to run before reporting. If None
                (the default used by curriculum.py / cli.py), runs the canonical
                chunk size (100), subject to resume-alignment: the first call
                after loading a mid-chunk checkpoint instead runs only enough
                steps to realign train_steps to a multiple of the chunk size.
                Passing an explicit int bypasses alignment (tests / callers
                that want exact control).
            step_callback: Optional callable invoked from inside the inner
                step loop alongside the display-status refresh (~every 5
                steps). Receives the absolute train_steps count. Used by the
                curriculum loop to keep auxiliary display lines (e.g. SPRT)
                fresh between chunks. Exceptions are logged and swallowed so
                a misbehaving callback never aborts training.

        Returns:
            dict with keys: total_loss, policy_loss, value_loss,
            mean_game_length, buffer_size, wall_clock_seconds, etc.
        """
        start = time.time()
        tc = self.training_config
        prev_steps = self.train_steps

        # Resume alignment: if a checkpoint was loaded mid-chunk, run a
        # shorter first chunk so that subsequent chunks land on the same
        # round-number boundaries as an uninterrupted run. Only applies when
        # the caller did not pass an explicit step count.
        is_short_chunk = False
        if steps is None:
            if self._resume_short_chunk > 0:
                steps = self._resume_short_chunk
                self._resume_short_chunk = 0
                is_short_chunk = True
            else:
                steps = 100

        # --- Self-play phase (alternating mode) ------------------------
        sp_games = 0
        sp_mean_len = 0.0
        sp_elapsed = 0.0
        if self._sp_pool is None:
            # No background self-play — do GPU self-play here
            n_games = self.self_play_config.python.games_per_write
            sp_start = time.time()
            logger.info("Self-play: generating %d games...", n_games)

            self.model.eval()
            n_sims = self.mcts_config.n_simulations
            game_trajectories = batched_self_play_games(
                self.model, self.config, self.game_config, self.device,
                n_games=n_games, n_simulations_override=n_sims,
            )

            sp_elapsed = time.time() - sp_start
            total_moves = 0
            for examples in game_trajectories:
                self.buffer.add_many(examples)
                sp_games += 1
                total_moves += len(examples)
            sp_mean_len = total_moves / sp_games if sp_games > 0 else 0.0
            logger.info("Self-play: %d games in %.1fs (%.1f/s, len %.0f)", sp_games, sp_elapsed, sp_games/sp_elapsed, sp_mean_len)

        # Free GPU memory from self-play before training (alternating mode only —
        # in background mode, Rust self-play is a separate CUDA process and
        # empty_cache() forces a sync that hurts step/s).
        if self.device.type == "cuda" and self._sp_pool is None:
            torch.cuda.empty_cache()

        # --- Training phase --------------------------------------------
        self.model.train()
        losses: dict[str, float] = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "value_mse": 0.0,
            "pc_loss": 0.0,
            "cross_entropy": 0.0,
            "target_entropy": 0.0,
        }

        augment = tc.augment_symmetries
        if augment:
            from hexo_a0.graph import random_augment

        # Prefetch is incompatible with reanalyze (which needs the model on
        # the main thread). When reanalyze is off (the common case) we use
        # a single background thread to sample + augment + pin so the main
        # thread only does GPU work + the H2D copies.
        use_prefetch = tc.reanalyze_fraction == 0

        min_buffer = tc.batch_size * 2  # need enough to sample meaningful batches
        if len(self.buffer) < min_buffer:
            logger.info("Waiting for buffer (%d/%d)...", len(self.buffer), min_buffer)
            while len(self.buffer) < min_buffer:
                if self._sp_pool is not None:
                    time.sleep(0.5)
                else:
                    # Alternating mode: generate more games
                    self.model.eval()
                    n_sims = self.mcts_config.n_simulations
                    game_trajectories = batched_self_play_games(
                        self.model, self.config, self.game_config, self.device,
                        n_games=self.self_play_config.python.games_per_write,
                        n_simulations_override=n_sims,
                    )
                    for examples in game_trajectories:
                        self.buffer.add_many(examples)
                        sp_games += 1
                    self.model.train()

        if len(self.buffer) > 0:
            actual_steps = 0
            t_sample = 0.0
            t_train = 0.0
            total_batch_nodes = 0
            total_batch_edges = 0
            total_examples = 0

            edge_budget = tc.edge_budget
            grad_accum = tc.grad_accumulation and edge_budget > 0
            use_fp16 = self.run_config.fp16

            # GPU-side running accumulators — sync-free per step. We sync them
            # once every 5 steps for the display update and once at end of the
            # report block for tensorboard. Cuts ~5 .item() syncs/step.
            running_total_t = torch.zeros((), device=self.device)
            running_policy_t = torch.zeros((), device=self.device)
            running_value_t = torch.zeros((), device=self.device)
            # Decoded-scalar value MSE — same as value_loss for scalar heads,
            # a comparable diagnostic for the distributional head (which
            # optimizes the two-hot CE in value_loss instead).
            running_value_mse_t = torch.zeros((), device=self.device)
            running_pc_t = torch.zeros((), device=self.device)
            running_cross_entropy_t = torch.zeros((), device=self.device)
            running_target_entropy_t = torch.zeros((), device=self.device)
            # Staleness instrumentation: per-quartile sums and counts of the
            # per-graph policy KL, where the quartile is the example's age in
            # the buffer at sample time (Q1=newest 25%, Q4=oldest 25%).
            running_staleness_kl_q_sum = torch.zeros(4, device=self.device)
            running_staleness_kl_q_count = torch.zeros(4, device=self.device)
            # Last known display values (only refreshed every 5 steps).
            last_disp_total = 0.0
            last_disp_policy = 0.0
            last_disp_value = 0.0

            # ----- Prefetch thread setup (fix 4) -----
            # The thread is a daemon so it will die at process exit even if
            # an exception escapes the loop. We also signal stop and join
            # explicitly after the natural loop end so each Trainer.train()
            # call doesn't leak a worker on the success path.
            prefetch_q: "queue.Queue | None" = None
            prefetch_stop: "threading.Event | None" = None
            prefetch_thread: "threading.Thread | None" = None
            if use_prefetch:
                prefetch_q = queue.Queue(maxsize=2)
                prefetch_stop = threading.Event()
                prefetch_thread = threading.Thread(
                    target=_prefetch_worker,
                    args=(
                        self.buffer,
                        tc.batch_size,
                        augment,
                        prefetch_q,
                        prefetch_stop,
                        self.device.type,
                        self.graph_batch_fn,
                        self.augment_batch_fn,
                    ),
                    name="hexo-prefetch",
                    daemon=True,
                )
                prefetch_thread.start()

            for step_i in range(steps):
                # Update display every 5 steps (no TensorBoard writes).
                # We sync the running tensors here — one sync per 5 steps.
                if step_i % 5 == 0 and step_i > 0:
                    if self._sp_pool is not None:
                        with self._sp_lock:
                            sp_count = self._sp_games
                    else:
                        sp_count = sp_games
                    elapsed_so_far = time.time() - start
                    if actual_steps > 0:
                        last_disp_total = (running_total_t / actual_steps).item()
                        last_disp_policy = (running_policy_t / actual_steps).item()
                        last_disp_value = (running_value_t / actual_steps).item()
                    lr = self.optimizer.param_groups[0]["lr"]
                    gps = f"{sp_count / elapsed_so_far:.1f}" if elapsed_so_far > 0 and sp_count > 0 else "-"
                    self.display.update_status(
                        stage=self.stage_name,
                        step=self.train_steps,
                        loss=f"{last_disp_total:.4f}",
                        policy_loss=f"{last_disp_policy:.4f}",
                        value_loss=f"{last_disp_value:.4f}",
                        games=f"+{sp_count}",
                        games_per_sec=gps,
                        buf_size=f"{len(self.buffer)}",
                        steps_per_sec=f"{step_i / elapsed_so_far:.1f}" if elapsed_so_far > 0 else "-",
                        lr=f"{lr:.2e}",
                    )
                    if step_callback is not None:
                        try:
                            step_callback(self.train_steps)
                        except Exception:
                            logger.exception("train() step_callback raised; continuing")

                _t0 = time.time()
                pc_weight = tc.pc_loss_weight

                if use_prefetch:
                    # Background thread already sampled + augmented + pinned.
                    # This blocks if the prefetch thread is behind GPU compute,
                    # which is the right behavior — the buffer can't keep up.
                    all_examples = prefetch_q.get()
                else:
                    # Legacy inline path (used when reanalyze is enabled).
                    raw_batch = self.buffer.sample(tc.batch_size)
                    if tc.reanalyze_fraction > 0:
                        raw_batch = self._reanalyze_batch(raw_batch)
                    # Fast path: batched byte-buffer augment when every example
                    # is game_state-bearing (mirrors the prefetch worker).
                    ns_parts = None
                    ns_perms = None
                    if (
                        augment
                        and self.augment_batch_fn is not None
                        and raw_batch
                        and all(ex.data is None for ex in raw_batch)
                    ):
                        states = [ex.game_state for ex in raw_batch]
                        tindices = [random.randint(1, 11) for _ in raw_batch]
                        try:
                            ns_parts, ns_perms = self.augment_batch_fn(states, tindices)
                            self._inline_augment_fallback_streak = 0
                        except Exception as exc:
                            ns_parts = None
                            self._inline_augment_fallback_streak = (
                                _augment_fallback_escalate(
                                    getattr(self, "_inline_augment_fallback_streak", 0),
                                    exc, "Inline augment",
                                )
                            )
                    recon = (
                        [ex.game_state for ex in raw_batch if ex.data is None]
                        if ns_parts is None
                        else []
                    )
                    rebuilt = iter(self.graph_batch_fn(recon)) if recon else iter(())
                    all_examples = []
                    for i, ex in enumerate(raw_batch):
                        if ns_parts is not None:
                            data = ns_parts[i]
                            policy = ex.policy_target[torch.tensor(ns_perms[i], dtype=torch.long)]
                        else:
                            data = ex.data if ex.data is not None else next(rebuilt)
                            if augment:
                                data, policy = random_augment(data, ex.policy_target)
                            else:
                                policy = ex.policy_target
                        all_examples.append((
                            data,
                            policy,
                            torch.tensor(ex.value_target, dtype=torch.float32),
                            getattr(ex, 'trajectory_id', None),
                            getattr(ex, '_age', None),
                            float(getattr(ex, 'sample_weight', 1.0)),
                        ))

                # ----- Forward / backward (sync-free; returns GPU tensors) -----
                self.model.train()
                self.optimizer.zero_grad()

                if edge_budget > 0:
                    # Split into micro-batches by edge budget
                    micro_batches = []
                    current_mb = []
                    current_edges = 0
                    for ex in all_examples:
                        e = ex[0].edge_index.shape[1] if ex[0].edge_index is not None else 0
                        if current_mb and current_edges + e > edge_budget:
                            micro_batches.append(current_mb)
                            current_mb = []
                            current_edges = 0
                        current_mb.append(ex)
                        current_edges += e
                    if current_mb:
                        micro_batches.append(current_mb)

                    if grad_accum:
                        n_micro = len(micro_batches)
                        # GPU-tensor accumulators for the micro-batches.
                        step_total_t = torch.zeros((), device=self.device)
                        step_policy_t = torch.zeros((), device=self.device)
                        step_value_t = torch.zeros((), device=self.device)
                        step_value_mse_t = torch.zeros((), device=self.device)
                        step_pc_t = torch.zeros((), device=self.device)
                        step_cross_entropy_t = torch.zeros((), device=self.device)
                        step_target_entropy_t = torch.zeros((), device=self.device)
                        for mb in micro_batches:
                            mb_losses = _forward_and_loss(
                                self.model, mb, self.device, use_fp16,
                                loss_scale=1.0 / n_micro,
                                pc_loss_weight=pc_weight,
                            )
                            step_total_t = step_total_t + mb_losses["total_loss"] / n_micro
                            step_policy_t = step_policy_t + mb_losses["policy_loss"] / n_micro
                            step_value_t = step_value_t + mb_losses["value_loss"] / n_micro
                            step_value_mse_t = step_value_mse_t + mb_losses["value_mse"] / n_micro
                            step_cross_entropy_t = step_cross_entropy_t + mb_losses["cross_entropy"] / n_micro
                            step_target_entropy_t = step_target_entropy_t + mb_losses["target_entropy"] / n_micro
                            if torch.is_tensor(mb_losses["pc_loss"]):
                                step_pc_t = step_pc_t + mb_losses["pc_loss"] / n_micro
                            if mb_losses.get("staleness_kl_q_sum") is not None:
                                running_staleness_kl_q_sum = running_staleness_kl_q_sum + mb_losses["staleness_kl_q_sum"]
                                running_staleness_kl_q_count = running_staleness_kl_q_count + mb_losses["staleness_kl_q_count"]
                    else:
                        all_examples = micro_batches[0]
                        mb_losses = _forward_and_loss(
                            self.model, all_examples, self.device, use_fp16,
                            pc_loss_weight=pc_weight,
                        )
                        step_total_t = mb_losses["total_loss"]
                        step_policy_t = mb_losses["policy_loss"]
                        step_value_t = mb_losses["value_loss"]
                        step_value_mse_t = mb_losses["value_mse"]
                        step_cross_entropy_t = mb_losses["cross_entropy"]
                        step_target_entropy_t = mb_losses["target_entropy"]
                        step_pc_t = (
                            mb_losses["pc_loss"]
                            if torch.is_tensor(mb_losses["pc_loss"])
                            else torch.zeros((), device=self.device)
                        )
                        if mb_losses.get("staleness_kl_q_sum") is not None:
                            running_staleness_kl_q_sum = running_staleness_kl_q_sum + mb_losses["staleness_kl_q_sum"]
                            running_staleness_kl_q_count = running_staleness_kl_q_count + mb_losses["staleness_kl_q_count"]
                else:
                    mb_losses = _forward_and_loss(
                        self.model, all_examples, self.device, use_fp16,
                        pc_loss_weight=pc_weight,
                    )
                    step_total_t = mb_losses["total_loss"]
                    step_policy_t = mb_losses["policy_loss"]
                    step_value_t = mb_losses["value_loss"]
                    step_value_mse_t = mb_losses["value_mse"]
                    step_cross_entropy_t = mb_losses["cross_entropy"]
                    step_target_entropy_t = mb_losses["target_entropy"]
                    step_pc_t = (
                        mb_losses["pc_loss"]
                        if torch.is_tensor(mb_losses["pc_loss"])
                        else torch.zeros((), device=self.device)
                    )
                    if mb_losses.get("staleness_kl_q_sum") is not None:
                        running_staleness_kl_q_sum = running_staleness_kl_q_sum + mb_losses["staleness_kl_q_sum"]
                        running_staleness_kl_q_count = running_staleness_kl_q_count + mb_losses["staleness_kl_q_count"]

                # ----- Single sync per step: NaN/Inf guard -----
                step_total_val = step_total_t.item()
                if math.isnan(step_total_val) or math.isinf(step_total_val):
                    logger.warning(
                        "NaN/Inf loss at step %d — skipping optimizer.step()",
                        self.train_steps + 1,
                    )
                    self.optimizer.zero_grad()
                else:
                    stepped, gnorm = _clip_and_step_if_finite(
                        self.model, self.optimizer, tc.max_grad_norm, self.scheduler
                    )
                    if not stepped:
                        logger.warning(
                            "NaN/Inf GRADIENT (norm=%.3g) at step %d — skipped "
                            "optimizer.step() to protect weights",
                            gnorm, self.train_steps + 1,
                        )

                _t1 = time.time()
                t_sample += _t1 - _t0
                for ex_tuple in all_examples:
                    d = ex_tuple[0]
                    total_batch_nodes += d.x.shape[0]
                    total_batch_edges += d.edge_index.shape[1] if d.edge_index is not None else 0
                total_examples += len(all_examples)

                # GPU-side accumulation — no sync.
                running_total_t = running_total_t + step_total_t
                running_policy_t = running_policy_t + step_policy_t
                running_value_t = running_value_t + step_value_t
                running_value_mse_t = running_value_mse_t + step_value_mse_t
                running_pc_t = running_pc_t + step_pc_t
                running_cross_entropy_t = running_cross_entropy_t + step_cross_entropy_t
                running_target_entropy_t = running_target_entropy_t + step_target_entropy_t

                actual_steps += 1
                self.train_steps += 1
                if self.current_step_delay > 0:
                    time.sleep(self.current_step_delay)

            # ----- Stop prefetch thread (success path) -----
            if prefetch_thread is not None:
                assert prefetch_stop is not None and prefetch_q is not None
                prefetch_stop.set()
                # Drain the queue so a worker blocked in put() can exit.
                try:
                    while True:
                        prefetch_q.get_nowait()
                except queue.Empty:
                    pass
                prefetch_thread.join(timeout=2.0)
                if prefetch_thread.is_alive():
                    logger.warning("Prefetch thread did not stop within 2s")

            if actual_steps > 0:
                # Final sync of running tensors → floats for tensorboard /
                # report summary. One sync each at end of the report block.
                losses["total_loss"] = (running_total_t / actual_steps).item()
                losses["policy_loss"] = (running_policy_t / actual_steps).item()
                losses["value_loss"] = (running_value_t / actual_steps).item()
                losses["value_mse"] = (running_value_mse_t / actual_steps).item()
                losses["pc_loss"] = (running_pc_t / actual_steps).item()
                losses["cross_entropy"] = (running_cross_entropy_t / actual_steps).item()
                losses["target_entropy"] = (running_target_entropy_t / actual_steps).item()
                # Staleness quartile means: per-quartile mean policy KL across
                # all examples in this train() call. None if no examples had
                # ages attached (tests / non-buffer-sampled paths).
                staleness_q_means: list[float] | None = None
                if running_staleness_kl_q_count.sum().item() > 0:
                    q_means_t = running_staleness_kl_q_sum / running_staleness_kl_q_count.clamp(min=1)
                    staleness_q_means = q_means_t.tolist()
                avg_nodes = total_batch_nodes / actual_steps
                avg_edges = total_batch_edges / actual_steps
                avg_examples = total_examples / actual_steps
                logger.info(
                    f"Timing: {t_train + t_sample:.1f}s total | "
                    f"avg batch: {avg_examples:.0f} examples, "
                    f"{avg_nodes:.0f} nodes, {avg_edges:.0f} edges"
                )

        # Free GPU memory from training before next self-play cycle
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - start

        # Collect self-play stats from background workers
        sp_p1_wins = 0
        sp_p2_wins = 0
        sp_examples_written = 0
        sp_report: _SelfPlayReport | None = None
        if self._sp_pool is not None:
            sp_report = self._collect_self_play_stats()
            sp_games = sp_report.games
            sp_mean_len = sp_report.mean_game_length
            sp_p1_wins = sp_report.p1_wins
            sp_p2_wins = sp_report.p2_wins
            sp_examples_written = sp_report.examples_written

        # Compute self-play throughput
        if sp_elapsed > 0:
            games_per_sec = sp_games / sp_elapsed
        elif sp_games > 0:
            games_per_sec = sp_games / elapsed
        else:
            games_per_sec = 0.0
        train_elapsed = elapsed - sp_elapsed
        steps_per_sec = steps / train_elapsed if train_elapsed > 0 else 0.0

        avg_batch_examples = total_examples / max(actual_steps, 1)
        # Sample reuse factor: how many times each example is sampled, on
        # average, before being evicted from the FIFO buffer.
        #   reuse = (consume rate) / (write rate)
        #         = (steps_per_sec * avg_batch_examples) / (examples_written / elapsed)
        # < 2: data-starved. ~4-8: healthy AlphaZero regime. > 20: stale.
        sp_examples_per_sec = (
            sp_examples_written / sp_elapsed if sp_elapsed > 0
            else (sp_examples_written / elapsed if elapsed > 0 else 0.0)
        )
        consume_per_sec = steps_per_sec * avg_batch_examples
        sample_reuse = (
            consume_per_sec / sp_examples_per_sec
            if sp_examples_per_sec > 0 else 0.0
        )

        # Auto-throttle controller: adjust step_delay to track target_reuse.
        # Skip on a short resume-alignment chunk — the rate inputs are
        # unrepresentative and a bad estimate would persist through the EMA.
        tc_throttle = self.training_config
        if (
            not is_short_chunk
            and tc_throttle.target_reuse > 0
            and sp_examples_per_sec > 0
            and avg_batch_examples > 0
        ):
            # 1/steps_per_sec is the full step period (compute + sleep); subtract sleep to get pure compute time.
            step_period = 1.0 / max(steps_per_sec, 1e-6)
            step_compute_time = max(step_period - self.current_step_delay, 1e-6)
            desired_delay = (
                avg_batch_examples / (tc_throttle.target_reuse * sp_examples_per_sec)
                - step_compute_time
            )
            saturated = desired_delay > tc_throttle.max_step_delay
            desired_delay = max(0.0, min(desired_delay, tc_throttle.max_step_delay))
            s = tc_throttle.step_delay_smoothing
            self.current_step_delay = s * self.current_step_delay + (1.0 - s) * desired_delay
            if saturated:
                logger.warning(
                    f"auto-throttle saturated: target_reuse={tc_throttle.target_reuse}, "
                    f"actual≈{sample_reuse:.2f}, self-play too slow"
                )

        metrics = {
            **losses,
            "mean_game_length": sp_mean_len,
            "games_per_sec": games_per_sec,
            "steps_per_sec": steps_per_sec,
            "buffer_size": len(self.buffer),
            "wall_clock_seconds": elapsed,
            "avg_batch_nodes": total_batch_nodes / max(actual_steps, 1),
            "avg_batch_edges": total_batch_edges / max(actual_steps, 1),
            "avg_batch_examples": avg_batch_examples,
            "sp_examples_per_sec": sp_examples_per_sec,
            "sample_reuse": sample_reuse,
            "staleness_kl_quartiles": staleness_q_means if actual_steps > 0 else None,
        }

        if self.writer is not None:
            step = self.train_steps
            self.writer.add_scalar(
                "loss/total",
                metrics["total_loss"],
                step,
                description=(
                    "Combined Gumbel AlphaZero loss: `alpha * policy_loss + beta * value_loss`. "
                    "The number actually optimised by the optimiser."
                ),
            )
            self.writer.add_scalar(
                "loss/policy",
                metrics["policy_loss"],
                step,
                description=(
                    "KL divergence between the Gumbel MCTS improved policy pi' "
                    "(`softmax(logits + sigma(completedQ))`, paper Eq. 11) and the network's "
                    "softmax output. Eq. 12 of Danihelka et al. 2022. Lower is better. "
                    "Floor is 0 (perfect match); the remaining 'loss budget' is the target "
                    "entropy — see `loss/target_entropy`."
                ),
            )
            self.writer.add_scalar(
                "loss/cross_entropy",
                metrics["cross_entropy"],
                step,
                description=(
                    "Cross-entropy CE(pi', q) = -sum pi' * log q between the Gumbel MCTS "
                    "improved policy and the network's softmax. Equals `loss/policy + "
                    "loss/target_entropy` up to FP noise. Reported for comparability with AZ "
                    "papers that report CE instead of KL as 'policy loss'. Not optimized "
                    "directly — KL and CE share the model gradient up to a target-only constant."
                ),
            )
            self.writer.add_scalar(
                "loss/target_entropy",
                metrics["target_entropy"],
                step,
                description=(
                    "Entropy H(pi') = -sum pi' * log pi' of the Gumbel MCTS improved policy, "
                    "averaged across the batch. Sharpness of the MCTS targets; depends only on "
                    "the targets, not the network. Sets the KL floor: CE = KL + H, so if H "
                    "drifts up the network has more 'work' to do to match. Watch alongside "
                    "`loss/policy` to distinguish 'model is learning' (KL drops) from 'targets "
                    "got softer' (H rises, CE may not drop)."
                ),
            )
            self.writer.add_scalar(
                "loss/value",
                metrics["value_loss"],
                step,
                description=(
                    "The optimized value-head loss. For a scalar head this is the MSE "
                    "against the eventual game outcome (+1 / -1 / 0). For a distributional "
                    "head (value_bins>0) this is the two-hot cross-entropy over value bins; "
                    "read loss/value_mse for the comparable decoded-scalar MSE. Stages can "
                    "floor at ~0.5 (MSE) due to inherent early-game ambiguity."
                ),
            )
            self.writer.add_scalar(
                "loss/value_mse",
                metrics["value_mse"],
                step,
                description=(
                    "MSE between the value head's decoded SCALAR (tanh output, or the "
                    "softmax expectation over bins for a distributional head) and the "
                    "eventual game outcome. Equals loss/value for scalar heads; for "
                    "distributional heads it is the outcome-comparable diagnostic while "
                    "loss/value shows the optimized cross-entropy."
                ),
            )
            # Staleness instrumentation: per-quartile mean policy KL by sample
            # age in the buffer. Q1 = newest 25%, Q4 = oldest 25%. The ratio
            # Q4/Q1 is the headline single-number staleness signal: ratio ≈ 1
            # means age doesn't matter; ratio >> 1 means old buffer entries
            # disagree more with the current network than fresh ones, which
            # is exactly the post-promotion stale-target effect that motivated
            # buffer_capacity reduction. Only logged if examples carried ages
            # (buffer-sampled path; absent for synthetic test batches).
            stale_q = metrics.get("staleness_kl_quartiles")
            if stale_q is not None:
                q1, q2, q3, q4 = stale_q
                self.writer.add_scalar(
                    "staleness/policy_kl_q1_freshest",
                    q1,
                    step,
                    description=(
                        "Mean policy KL over the freshest 25% of buffer samples "
                        "(age in [0, 0.25), 0=newest). Read only in context with "
                        "Q4 — see staleness/policy_kl_ratio_old_over_fresh for the "
                        "interpreted signal."
                    ),
                )
                self.writer.add_scalar("staleness/policy_kl_q2", q2, step)
                self.writer.add_scalar("staleness/policy_kl_q3", q3, step)
                self.writer.add_scalar(
                    "staleness/policy_kl_q4_oldest",
                    q4,
                    step,
                    description=(
                        "Mean policy KL over the oldest 25% of buffer samples "
                        "(age in [0.75, 1.0]). Read only in context with Q1 — "
                        "see staleness/policy_kl_ratio_old_over_fresh for the "
                        "interpreted signal."
                    ),
                )
                if q1 > 0:
                    self.writer.add_scalar(
                        "staleness/policy_kl_ratio_old_over_fresh",
                        q4 / q1,
                        step,
                        description=(
                            "Q4/Q1 of mean policy KL stratified by buffer-sample age. "
                            "Has two regimes:\n"
                            "• **<1 (early-stage / still-learning)**: network is "
                            "gradient-descending faster than samples go stale, so old "
                            "samples (which have been trained on many times) have lower "
                            "KL than fresh ones (which it hasn't been pulled toward yet). "
                            "A healthy training signal.\n"
                            "• **≈1 (steady state)**: network drift balances buffer turnover.\n"
                            "• **>>1 (late-stage / stale)**: network is moving faster than "
                            "the buffer can refresh — old targets reflect a past network "
                            "state and KL stays high there because gradient descent can't "
                            "reduce it further without dragging the network back. "
                            "Candidate signal for enabling reanalyze or reducing "
                            "buffer_capacity.\n"
                            "The **crossover from <1 to >1 may itself be a stage-convergence "
                            "marker** — the moment training stops making meaningful progress "
                            "on the current data distribution."
                        ),
                    )
            # Window-dependent self-play metrics: skipped on a short resume-
            # alignment chunk where the sample is too small to be representative.
            if not is_short_chunk:
                self.writer.add_scalar(
                    "self_play/mean_game_length",
                    metrics["mean_game_length"],
                    step,
                    description=(
                        "Average number of moves per self-play game (NOT the number of training "
                        "examples per game - those can differ when playout cap randomization is "
                        "active). Read from the binary record's `move_count` field."
                    ),
                )
                self.writer.add_scalar(
                    "self_play/games_per_sec",
                    games_per_sec,
                    step,
                    description=(
                        "Self-play game generation rate from the Rust workers, measured over the "
                        "most recent reporting window. Drops at later curriculum stages as MCTS "
                        "gets more expensive."
                    ),
                )
                decided = sp_p1_wins + sp_p2_wins
                p1_decided_rate = (sp_p1_wins / decided) if decided > 0 else 0.0
                self.writer.add_scalar(
                    "self_play/p1_decided_rate",
                    p1_decided_rate,
                    step,
                    description=(
                        "Fraction of decided self-play games (draws excluded) won by P1. "
                        "**0.5 = balanced**, drift toward 0 or 1 indicates side bias. Detects "
                        "co-adaptation failures invisible to standard losses. See "
                        "`docs/research/2026-04-07-per-side-decided-rate-methodology.md`."
                    ),
                )
                # Compute and log the effective exploration fraction.
                # When adaptive exploration is active (exploration_max > 0), the
                # Rust binary uses: frac = base + (max - base) * (2*|p1_ema - 0.5|)^2
                # We mirror that here from the trainer-side p1_decided_rate.
                base_frac = self.mcts_config.exploration_fraction
                max_frac = self.mcts_config.exploration_max
                eff_explore_moves = None
                if base_frac > 0:
                    if max_frac > 0:
                        deviation = min(1.0, abs(2.0 * (p1_decided_rate - 0.5)))
                        exp = self.mcts_config.exploration_exponent
                        eff_frac = base_frac + (max_frac - base_frac) * (deviation ** exp)
                    else:
                        eff_frac = base_frac
                    eff_explore_moves = metrics["mean_game_length"] * eff_frac
                    self.writer.add_scalar(
                        "self_play/exploration_fraction",
                        eff_frac,
                        step,
                        description=(
                            "Effective exploration fraction. When adaptive exploration is active "
                            "(`exploration_max > 0`), scales up from `exploration_fraction` (base) "
                            "toward `exploration_max` as `p1_decided_rate` drifts from 0.5, using "
                            "squared deviation for smooth center / steep edges."
                        ),
                    )
                    self.writer.add_scalar(
                        "self_play/exploration_moves",
                        eff_explore_moves,
                        step,
                        description=(
                            "Effective number of exploration placements per game "
                            "(= mean_game_length * exploration_fraction). During these placements "
                            "the played action is sampled from the improved policy pi'; after this "
                            "the Gumbel-selected action is used."
                        ),
                    )
            self.writer.add_scalar(
                "self_play/buffer_size",
                metrics["buffer_size"],
                step,
                description=(
                    "Current number of training examples in the replay buffer. Caps at "
                    "`training.buffer_capacity`."
                ),
            )
            if sp_report is not None and not is_short_chunk and sp_report.game_lengths:
                lengths_sorted = sorted(sp_report.game_lengths)
                p10 = lengths_sorted[int(0.10 * (len(lengths_sorted) - 1))]
                self.writer.add_scalar(
                    "self_play/game_length_p10",
                    float(p10),
                    step,
                    description=(
                        "10th percentile of self-play game lengths in the reporting "
                        "window. **Blunder-collapse canary**: exploration-induced "
                        "blunders end games early and pile up in the left tail long "
                        "before the mean moves. A sustained drop with a stable mean "
                        "means sampled exploration moves are deciding games — check "
                        "`decided_in_window_rate` and the exploration settings."
                    ),
                )
                # Canary window: adaptive (fraction of mean game length) or
                # fixed (exploration_moves) — the fixed-window regime is
                # exactly the 2026-06 exploration_moves=30 failure mode, so
                # the alarm must cover it too.
                if eff_explore_moves is not None:
                    # Reuse the effective window computed for the
                    # exploration_fraction/moves scalars above (same report
                    # window, same formula as the Rust binary's EMA mirror).
                    window = eff_explore_moves
                elif self.mcts_config.exploration_moves > 0:
                    window = float(self.mcts_config.exploration_moves)
                else:
                    window = None
                if window is not None:
                    # A game "decided in the window" ended during exploration
                    # sampling or within one turn (2 placements) of it — the
                    # signature of a sampled blunder being converted.
                    in_window = sum(1 for L in sp_report.game_lengths if L <= window + 2)
                    self.writer.add_scalar(
                        "self_play/decided_in_window_rate",
                        in_window / len(sp_report.game_lengths),
                        step,
                        description=(
                            "Fraction of self-play games that ended during the effective "
                            "exploration window (+1 turn of slack). **Direct alarm for "
                            "exploration deciding games**: ≈0 in a healthy regime. The "
                            "2026-06 exploration_moves=30 failure would have pushed this "
                            "toward 0.5+. See "
                            "`docs/research/2026-06-11-q-margin-truncated-exploration.md`."
                        ),
                    )
                # Version-skew guard: if truncate_delta is set but no HX06
                # window stats arrived, the self-play binary is likely stale.
                if (
                    0 < self.mcts_config.truncate_delta
                    and math.isfinite(self.mcts_config.truncate_delta)
                    and sp_report.game_lengths
                    and not sp_report.window_stats
                ):
                    self._stale_binary_warn_count += 1
                    if self._stale_binary_warn_count % 10 == 1:
                        logger.warning(
                            "truncate_delta=%s is set but no HX06 window stats arrived in "
                            "this window — self-play binary may be stale (rebuild + redeploy, "
                            "incl. remote actors)",
                            self.mcts_config.truncate_delta,
                        )
                if sp_report.window_stats:
                    # Stats arrived: reset the skew-warning throttle so a later
                    # staleness episode warns on its first window again.
                    self._stale_binary_warn_count = 0
                    iw, gated, mass_sum, deficit_sum = _sum_window_stats(sp_report.window_stats)
                    if iw > 0:
                        self.writer.add_scalar(
                            "self_play/truncation_mass_removed",
                            mass_sum / iw, step,
                            description=(
                                "Mean fraction of in-window sampling mass zeroed by the Q-margin "
                                "gate (per in-window move). Offline baseline ~0.15 at δ=0.25, "
                                "window 30 (visit_counts mode). Only reported while the gate is "
                                "active (HX06 records). "
                                "See docs/research/2026-06-11-q-margin-truncated-exploration.md."
                            ),
                        )
                        self.writer.add_scalar(
                            "self_play/truncation_fires_rate",
                            gated / iw, step,
                            description=(
                                "Fraction of in-window moves where the Q-margin gate removed "
                                "sampling mass from the acting distribution. Gate activity "
                                "indicator; openings should stay mostly un-gated. "
                                "gated_moves counts per-move (one increment per acted move "
                                "where any mass was removed, not per candidate)."
                            ),
                        )
                        self.writer.add_scalar(
                            "self_play/sampled_deficit_mean",
                            deficit_sum / iw, step,
                            description=(
                                "Mean Q-deficit (q_max − q[acted]) of in-window acted moves. The "
                                "residual exploration-risk metric: ungated window-30 measured "
                                "0.111 offline; gated arms measured ≈ δ/4–δ/6. A sustained jump "
                                "means exploration is playing refuted moves again. "
                                "Deficits are search-measured under visit_counts (production mode); "
                                "in legacy improved_policy mode unsearched moves use the v_mix "
                                "fallback Q so deficits may be underestimated."
                            ),
                        )

            if sp_report is not None and not is_short_chunk:
                import numpy as _np
                max_moves = int(self.game_config.max_moves)
                game_len_edges = _np.linspace(0, max_moves + 1, 31)
                entropy_edges = _np.linspace(0.0, 5.0, 31)
                top1_edges = _np.linspace(0.0, 1.0, 31)
                value_edges = _np.linspace(-1.01, 1.01, 31)
                _log_normalized_histogram(
                    self.writer, "self_play/hist/game_length",
                    sp_report.game_lengths, game_len_edges, step,
                )
                _log_normalized_histogram(
                    self.writer, "self_play/hist/policy_entropy",
                    sp_report.policy_entropies, entropy_edges, step,
                )
                _log_normalized_histogram(
                    self.writer, "self_play/hist/policy_top1",
                    sp_report.policy_top1, top1_edges, step,
                )
                _log_normalized_histogram(
                    self.writer, "self_play/hist/value_target",
                    sp_report.value_targets, value_edges, step,
                )
            if not is_short_chunk:
                self._log_perplexity_by_move(step)
            if not is_short_chunk:
                self.writer.add_scalar(
                    "training/steps_per_sec",
                    steps_per_sec,
                    step,
                    description="Optimizer steps per second over the reporting window.",
                )
                self.writer.add_scalar(
                    "training/step_delay",
                    self.current_step_delay,
                    step,
                    description=(
                        "Current step delay (seconds) being applied between optimizer steps. "
                        "When `target_reuse=0` this equals the static config value. When "
                        "auto-throttle is active this is the controller's current output."
                    ),
                )
            self.writer.add_scalar(
                "training/lr",
                self.optimizer.param_groups[0]["lr"],
                step,
                description="Current learning rate from the LR scheduler.",
            )
            if not is_short_chunk:
                self.writer.add_scalar(
                    "timing/report_seconds",
                    metrics["wall_clock_seconds"],
                    step,
                    description=(
                        "Wall-clock time of the most recent reporting window (training + self-play "
                        "stats collection). Useful for sanity-checking throughput drops."
                    ),
                )
            self.writer.add_scalar(
                "graph/avg_batch_nodes",
                metrics["avg_batch_nodes"],
                step,
                description=(
                    "Average number of graph nodes per training batch. Constrained by "
                    "`edge_budget` so it varies per stage."
                ),
            )
            self.writer.add_scalar(
                "graph/avg_batch_edges",
                metrics["avg_batch_edges"],
                step,
                description=(
                    "Average number of graph edges per training batch. Approaches the "
                    "configured `edge_budget` ceiling."
                ),
            )
            self.writer.add_scalar(
                "graph/avg_batch_examples",
                metrics["avg_batch_examples"],
                step,
                description=(
                    "Average number of training examples per batch. **Shrinks at later "
                    "curriculum stages** because edge-budget batching keeps total edges "
                    "constant - denser graphs -> fewer examples per batch. This is the "
                    "implicit batch-size lever."
                ),
            )
            if not is_short_chunk:
                self.writer.add_scalar(
                    "self_play/examples_per_sec",
                    metrics["sp_examples_per_sec"],
                    step,
                    description=(
                        "Rate at which self-play writes new training examples into the buffer. "
                        "With playout cap active this is "
                        "`games_per_sec * mean(full_search_positions_per_game)`, NOT "
                        "`games_per_sec * mean_game_length`. Numerator of the staleness write_rate."
                    ),
                )
                self.writer.add_scalar(
                    "training/sample_reuse",
                    metrics["sample_reuse"],
                    step,
                    description=(
                        "Average number of times each training example is sampled before FIFO "
                        "eviction. `(steps_per_sec * avg_batch_examples) / examples_per_sec`. "
                        "**Healthy: ~4-8x. <2: data-starved. >20: stale.** Buffer capacity does "
                        "NOT affect this - only the relative rates do. See "
                        "`docs/research/2026-04-08-buffer-staleness-and-edge-budget-batching.md`."
                    ),
                )
            self.writer.flush()

        # Final summary
        summary = (
            f"Step {self.train_steps} │ "
            f"loss {metrics['total_loss']:.4f} (π {metrics['policy_loss']:.4f} v {metrics['value_loss']:.4f}) │ "
            f"games +{sp_games} ({games_per_sec:.1f}/s, len {sp_mean_len:.0f}) │ "
            f"buf {metrics['buffer_size']} │ "
            f"train {steps_per_sec:.1f} step/s │ "
            f"reuse {metrics['sample_reuse']:.1f}× │ "
            f"lr {self.optimizer.param_groups[0]['lr']:.2e} │ "
            f"{elapsed:.1f}s"
        )
        logger.info(summary)
        self.display.update_status(
            stage=self.stage_name,
            step=self.train_steps,
            loss=f"{metrics['total_loss']:.4f}",
            policy_loss=f"{metrics['policy_loss']:.4f}",
            value_loss=f"{metrics['value_loss']:.4f}",
            games=f"+{sp_games}",
            games_per_sec=f"{games_per_sec:.1f}",
            buf_size=f"{metrics['buffer_size']}",
            steps_per_sec=f"{steps_per_sec:.1f}",
            lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
        )

        # --- Checkpoint (step-based) --------------------------------------
        ckpt_every = self.eval_config.checkpoint_interval
        saved_checkpoint = (
            ckpt_every > 0
            and self._ckpt_dir is not None
            and self.train_steps // ckpt_every > prev_steps // ckpt_every
        )
        if saved_checkpoint:
            ckpt_path = Path(self._ckpt_dir) / f"checkpoint_{self.train_steps:08d}.pt"
            self.save_checkpoint(str(ckpt_path))

        # --- Self-play model update (step-based) -------------------------
        # In champion mode, self-play only updates on promotion (handled by
        # curriculum.py calling _export_champion), not on a periodic schedule.
        sp_update_every = self.self_play_config.model_update_interval
        if sp_update_every <= 0:
            update_sp = saved_checkpoint  # fall back to checkpoint interval
        else:
            update_sp = self.train_steps // sp_update_every > prev_steps // sp_update_every
        if self._champion_mode:
            update_sp = False
        if update_sp and hasattr(self, "_sp_process") and self._sp_process is not None:
            try:
                sp_dir = Path(self._ckpt_dir or ".") / "self_play"
                sp_model_path = str(sp_dir / "model_selfplay.pt")
                if self.self_play_config.rust.python_inference:
                    # Python subprocess loads raw state dict, not TorchScript.
                    # Save a checkpoint that the subprocess can torch.load().
                    # Use atomic write (tmp + rename) to avoid partial reads.
                    import os, tempfile
                    sd = self.model.state_dict()
                    mc_dict = {
                        f.name: getattr(self.model_config, f.name)
                        for f in self.model_config.__dataclass_fields__.values()
                    }
                    fd, tmp = tempfile.mkstemp(dir=str(sp_dir), suffix=".pt.tmp")
                    os.close(fd)
                    torch.save({"model_state_dict": sd, "model_config": mc_dict}, tmp)
                    os.replace(tmp, sp_model_path)
                    logger.debug("Saved checkpoint for Python inference: %s", sp_model_path)
                    self._snapshot_to_pool(sp_model_path)
                else:
                    self._export_torchscript(sp_model_path)
                    # Snapshot the precision-matched variant into the pool
                    precision = self.self_play_config.precision
                    if precision == "fp16":
                        pool_src = sp_model_path.replace(".pt", "_fp16.pt")
                    elif precision == "fp32":
                        pool_src = sp_model_path.replace(".pt", "_fp32.pt")
                    else:
                        pool_src = sp_model_path
                    self._snapshot_to_pool(pool_src)
            except Exception as e:
                logger.warning("Failed to export model for self-play: %s", e)

        # --- Evaluation phase (step-based) --------------------------------
        eval_every = self.eval_config.interval
        run_eval = (
            self.eval_config.games > 0
            and eval_every > 0
            and self.train_steps // eval_every > prev_steps // eval_every
        )

        # Model introspection (layer_analysis/* + edge_features/* TB scalars) —
        # gated independently of run_eval so it can fire at higher cadence
        # during graft experiments. 0 in config = inherit eval_every for
        # backwards compatibility with pre-2026-05-10 configs.
        analysis_every = self.eval_config.model_analysis_interval or eval_every
        run_analysis = (
            self.writer is not None
            and self.model_config.graph_type == "axis"
            and analysis_every > 0
            and self.train_steps // analysis_every > prev_steps // analysis_every
        )
        if run_analysis:
            self._log_model_analysis(step)
        if run_eval:
            from hexo_a0.evaluate import evaluate_vs_random

            self.display.show_eval_busy("evaluating...")
            eval_result = evaluate_vs_random(
                self.model, self.game_config, self.device,
                n_games=self.eval_config.games, model_config=self.model_config,
            )
            metrics["eval_win_rate"] = eval_result["win_rate"]
            metrics["eval_mean_game_length"] = eval_result["mean_game_length"]

            if self.writer is not None:
                self.writer.add_scalar(
                    "eval/raw_policy_win_rate_vs_random",
                    eval_result["win_rate"],
                    step,
                    description=(
                        "Fraction of eval games won against a uniform-random opponent using "
                        "the raw policy head (argmax of logits, no MCTS). A vibe check on the "
                        "policy itself: does the network know what to do without search "
                        "holding its hand? Much harder to saturate than MCTS-vs-random, "
                        "because a single policy-head gap is enough for random to stumble "
                        "into and exploit. Primary convergence signal for "
                        "`convergence_mode = \"random\"`."
                    ),
                )
                self.writer.add_scalar(
                    "eval/raw_policy_mean_game_length_vs_random",
                    eval_result["mean_game_length"],
                    step,
                    description=(
                        "Average moves per raw-policy-vs-random eval game. Distribution is "
                        "distinct from MCTS-backed evals and from self-play — early in "
                        "training the raw policy loses fast, so games are short; "
                        "lengthens as the policy learns to avoid immediate blunders."
                    ),
                )
                self.writer.add_scalar(
                    "eval/raw_policy_p1_decided_rate_vs_random",
                    eval_result["p1_decided_rate"],
                    step,
                    description=(
                        "Per-side decided rate of the raw policy against random. Pinned ~0.5 "
                        "once the policy head is strong enough that random can't exploit "
                        "side bias. Diagnostic for asymmetric policy-head learning."
                    ),
                )

            eval_parts = [
                f"vs random: {eval_result['win_rate']:.0%} "
                f"(W{eval_result['wins']}/L{eval_result['losses']}/D{eval_result['draws']}) "
                f"P1/P2 {eval_result['p1_wins']}/{eval_result['p2_wins']} "
                f"(P1 decided {eval_result['p1_decided_rate']:.0%})"
            ]

            # Eval vs champion (mutually exclusive with vs-prev-checkpoint).
            # Skipped in SPRT mode: the SPRT daemon plays vs-champion games
            # continuously in a subprocess, making the fixed-N duplicate.
            if (
                self._champion_mode
                and self._champion_path is not None
                and not self._sprt_mode
            ):
                from hexo_a0.evaluate import evaluate_vs_checkpoint

                vs_champ = evaluate_vs_checkpoint(
                    self.model, self._champion_path, self.model_config,
                    self.game_config, self.device,
                    n_games=self.eval_config.games,
                    mcts_sims=self.eval_config.sims,
                    mcts_m_actions=self.eval_config.m_actions,
                )
                metrics["eval_win_rate_vs_champion"] = vs_champ["win_rate"]

                if self.writer is not None:
                    self.writer.add_scalar(
                        "eval/win_rate_vs_champion",
                        vs_champ["win_rate"],
                        step,
                        description=(
                            "Fraction of evaluation games won against the current champion "
                            "checkpoint. > champion_threshold = promotion; sustained below "
                            "threshold = plateaued. Primary convergence signal for "
                            "`convergence_mode = \"champion\"`."
                        ),
                    )
                    self.writer.add_scalar(
                        "eval/p1_decided_rate_vs_champion",
                        vs_champ["p1_decided_rate"],
                        step,
                        description=(
                            "Per-side decided rate against the champion. "
                            "Distinguishes co-adaptation from genuine side bias."
                        ),
                    )

                eval_parts.append(
                    f"vs champion: {vs_champ['win_rate']:.0%} "
                    f"(W{vs_champ['wins']}/L{vs_champ['losses']}/D{vs_champ['draws']}) "
                    f"P1/P2 {vs_champ['p1_wins']}/{vs_champ['p2_wins']} "
                    f"(P1 decided {vs_champ['p1_decided_rate']:.0%})"
                )

            # Eval vs previous checkpoint (checkpoint-lag mode)
            elif self._ckpt_dir is not None:
                from hexo_a0.evaluate import evaluate_vs_checkpoint

                pts = sorted(p for p in Path(self._ckpt_dir).glob("checkpoint_*.pt") if "_buffer" not in p.name)
                lag = self._checkpoint_lag
                if len(pts) > lag:
                    opponent_path = pts[-(lag + 1)]
                    vs_ckpt = evaluate_vs_checkpoint(
                        self.model, str(opponent_path), self.model_config,
                        self.game_config, self.device,
                        n_games=self.eval_config.games,
                        mcts_sims=self.eval_config.sims,
                        mcts_m_actions=self.eval_config.m_actions,
                    )
                    metrics["eval_win_rate_vs_prev"] = vs_ckpt["win_rate"]

                    if self.writer is not None:
                        self.writer.add_scalar(
                            "eval/win_rate_vs_prev_checkpoint",
                            vs_ckpt["win_rate"],
                            step,
                            description=(
                                "Fraction of evaluation games won against the model checkpoint "
                                "from `curriculum.checkpoint_lag` saves ago. Steady > 0.55 = improving; "
                                "~= 0.5 = saturated; < 0.5 = regressing. Primary convergence "
                                "signal for `convergence_mode = \"checkpoint\"`."
                            ),
                        )
                        self.writer.add_scalar(
                            "eval/p1_decided_rate_vs_prev_checkpoint",
                            vs_ckpt["p1_decided_rate"],
                            step,
                            description=(
                                "Per-side decided rate against the prior checkpoint. "
                                "Distinguishes co-adaptation (drifts with self-play rate) from "
                                "genuine game-theoretic side bias (persists across opponents)."
                            ),
                        )

                    eval_parts.append(
                        f"vs {opponent_path.stem}: {vs_ckpt['win_rate']:.0%} "
                        f"(W{vs_ckpt['wins']}/L{vs_ckpt['losses']}/D{vs_ckpt['draws']}) "
                        f"P1/P2 {vs_ckpt['p1_wins']}/{vs_ckpt['p2_wins']} "
                        f"(P1 decided {vs_ckpt['p1_decided_rate']:.0%})"
                    )

            # Eval vs Sealbot minimax (if configured)
            sb = self.eval_config.sealbot
            sealbot_games = sb.games
            if sealbot_games > 0:
                try:
                    from hexo_a0.sealbot_eval import evaluate_vs_sealbot

                    # Adaptive difficulty: use current level from ladder
                    if sb.adaptive and sb.levels:
                        sealbot_tl = sb.levels[min(self._sealbot_level_idx, len(sb.levels) - 1)]
                    else:
                        sealbot_tl = sb.time_limit

                    # Use sealbot-specific game config if overrides are set,
                    # otherwise fall back to the current stage's config.
                    # This lets sealbot always play full HTTTX rules even
                    # during early curriculum stages with smaller boards.
                    import hexo_rs as _hr
                    sb_game_config = self.game_config
                    if sb.win_length > 0 or sb.placement_radius > 0 or sb.max_moves > 0:
                        sb_game_config = _hr.GameConfig(
                            sb.win_length or self.game_config.win_length,
                            sb.placement_radius or self.game_config.placement_radius,
                            sb.max_moves or self.game_config.max_moves,
                        )

                    vs_sb = evaluate_vs_sealbot(
                        self.model, sb_game_config, self.device,
                        n_games=sealbot_games,
                        sealbot_time_limit=sealbot_tl,
                        n_simulations=sb.sims,
                        m_actions=sb.m_actions,
                        model_config=self.model_config,
                        workers=sb.workers,
                    )
                    metrics["sealbot_win_rate"] = vs_sb["win_rate"]
                    metrics["sealbot_elo_diff"] = vs_sb["elo_diff"]

                    # Adaptive sealbot: promote to harder level if consistently winning
                    if sb.adaptive and sb.levels:
                        if vs_sb["win_rate"] >= sb.promote_threshold:
                            self._sealbot_consecutive_above += 1
                        else:
                            self._sealbot_consecutive_above = 0
                        if (self._sealbot_consecutive_above >= sb.promote_patience
                                and self._sealbot_level_idx < len(sb.levels) - 1):
                            self._sealbot_level_idx += 1
                            self._sealbot_consecutive_above = 0
                            logger.info(
                                "Sealbot promoted to level %d (time_limit=%.2fs)",
                                self._sealbot_level_idx,
                                sb.levels[self._sealbot_level_idx],
                            )
                            self.log_event("sealbot",
                                f"Promoted to level {self._sealbot_level_idx} "
                                f"(time_limit={sb.levels[self._sealbot_level_idx]:.2f}s)"
                            )

                    if self.writer is not None:
                        self.writer.add_scalar(
                            "eval/sealbot_win_rate",
                            vs_sb["win_rate"],
                            step,
                            description=(
                                "Win rate against the SealBot C++ minimax opponent at the "
                                "current difficulty level. Adaptive: promotes through "
                                "`sealbot.levels` when win rate exceeds "
                                "`sealbot.promote_threshold`."
                            ),
                        )
                        self.writer.add_scalar(
                            "eval/sealbot_p1_decided_rate",
                            vs_sb["p1_decided_rate"],
                            step,
                            description="Per-side decided rate against SealBot.",
                        )
                        if math.isfinite(vs_sb["elo_diff"]):
                            self.writer.add_scalar(
                                "eval/sealbot_elo",
                                vs_sb["elo_diff"],
                                step,
                                description=(
                                    "Elo difference vs SealBot at the current level. Computed "
                                    "from win rate via the standard Elo formula."
                                ),
                            )
                        self.writer.add_scalar(
                            "eval/sealbot_level",
                            sealbot_tl,
                            step,
                            description=(
                                "Current SealBot difficulty level index (into "
                                "`sealbot.levels`). Increases monotonically as the model gets "
                                "stronger."
                            ),
                        )

                    elo_str = f"{vs_sb['elo_diff']:+.0f}" if math.isfinite(vs_sb["elo_diff"]) else "-inf"
                    eval_parts.append(
                        f"vs Sealbot: {vs_sb['win_rate']:.0%} "
                        f"(W{vs_sb['wins']}/L{vs_sb['losses']}/D{vs_sb['draws']}) "
                        f"P1/P2 {vs_sb['p1_wins']}/{vs_sb['p2_wins']} "
                        f"(P1 decided {vs_sb['p1_decided_rate']:.0%}) "
                        f"Elo {elo_str}"
                    )
                except FileNotFoundError:
                    logger.debug("Sealbot repo not found, skipping eval")
                except (Exception, BaseException) as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    logger.warning("Sealbot eval failed: %s", e)

            if self.writer is not None:
                self.writer.flush()

            eval_summary = " │ ".join(eval_parts)
            # Log to file only (display handles terminal output; logging
            # to both caused triple-display when stderr + stdout + file
            # all captured the same line).
            logger.debug("Eval: %s", eval_summary)
            self.display.update_eval(eval_summary)

            # Drain accumulated self-play stats so the next iteration
            # doesn't report games that arrived during eval as a spike.
            if self._sp_pool is not None:
                self._collect_self_play_stats()

        return metrics

    # ------------------------------------------------------------------
    # Model introspection
    # ------------------------------------------------------------------

    def _log_perplexity_by_move(self, step: int, n_samples: int = 4096) -> None:
        """Sample a batch from the replay buffer and log network-prior and
        MCTS-target policy perplexity stratified by game phase (stone count).

        Diagnostic only — runs one ``no_grad`` forward off the training hot
        path. Both distributions are measured on the *same* sampled positions
        so the prior-vs-target gap (the search-improvement signal) is
        comparable per phase. See the entropy-collapse discussion: healthy =
        high perplexity in early/open phases, low in late/forced phases;
        collapse = the whole surface slumping toward 1.0.
        """
        if self.writer is None:
            return
        try:
            n = min(n_samples, len(self.buffer))
            if n <= 0:
                return
            raw = self.buffer.sample(n)
            examples = [
                (ex.data if ex.data is not None else self.graph_fn(ex.game_state), ex.policy_target)
                for ex in raw
            ]
            model = self.model
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod
            was_training = model.training
            model.eval()
            try:
                prior_perp, target_perp, counts = _perplexity_by_move(
                    model, examples, self.device, self.run_config.fp16,
                )
            finally:
                if was_training:
                    model.train()
            _log_perplexity_by_move_scalars(
                self.writer, "policy/perplexity_prior", prior_perp, counts, step,
            )
            _log_perplexity_by_move_scalars(
                self.writer, "policy/perplexity_target", target_perp, counts, step,
            )
        except Exception as e:  # diagnostic must never break training
            logger.warning("perplexity-by-move logging failed: %s", e)

    def _analysis_sample_graphs(self, n: int) -> list:
        """Sample on-policy graphs from the replay buffer for introspection
        metrics (edge/node gradient + ablation). Using the *training*
        distribution matters: random self-play positions sit off-distribution
        and badly understate feature importance (a random-position probe
        understated the value head's src_player reliance ~7x vs buffer
        positions). Falls back to random self-play only when the buffer is cold
        so the metrics still populate at the very start of a stage. Returns a
        list of PyG ``Data`` objects.
        """
        graphs: list = []
        if len(self.buffer) > 0:
            try:
                for ex in self.buffer.sample(min(n, len(self.buffer))):
                    d = ex.data if getattr(ex, "data", None) is not None else self.graph_fn(ex.game_state)
                    if d is not None:
                        graphs.append(d)
            except Exception as e:  # diagnostic must never break training
                logger.debug("buffer sampling for analysis failed: %s", e)
        if graphs:
            return graphs
        # Fallback: random self-play positions (buffer cold).
        import random

        import hexo_rs

        for _ in range(n * 2):
            if len(graphs) >= n:
                break
            game = hexo_rs.GameState(self.game_config)
            for _ in range(random.randint(1, min(20, self.game_config.max_moves - 1))):
                moves = game.legal_moves()
                if not moves or game.is_terminal():
                    break
                m = random.choice(moves)
                game.apply_move(m[0], m[1])
            if not game.is_terminal():
                graphs.append(self.graph_fn(game))
        return graphs

    def _log_model_analysis(self, step: int) -> None:
        """Log edge feature importance and layer utilization to TensorBoard."""
        import hexo_rs  # noqa: F401  (legacy; kept for any direct uses below)
        eval_every = self.eval_config.interval
        # `analysis_every` is the cadence at which this method is called.
        # The expensive ablation block below uses it to fire once per
        # ablation period (default `eval_every * 5`) regardless of how often
        # this method is invoked; without this, a tighter analysis cadence
        # would multiply the ablation rate.
        analysis_every = self.eval_config.model_analysis_interval or eval_every

        # Get the raw (possibly compiled) model
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        rep = model.representation
        writer = self.writer

        # --- Node feature weight norms ---
        # Per-input-channel L2 norm of input_proj. For a threat-features
        # graft the four threat columns start zero-init, so these scalars
        # are the direct observable for the net picking the features up.
        node_w = rep.input_proj.weight.detach()  # (hidden_dim, 7/8/11/12)
        node_names = _node_feature_names(self.model_config)[: node_w.shape[1]]
        node_norms = node_w.norm(dim=0)
        for i, name in enumerate(node_names):
            writer.add_scalar(
                f"node_features/weight_norm/{name}",
                node_norms[i].item(),
                step,
                description=(
                    "L2 norm of the input-projection weight column for each "
                    "node-feature channel. Grafted threat columns (own_*/opp_*) "
                    "start at exactly 0; growth tracks feature adoption."
                ),
            )

        # --- Edge feature weight norms ---
        if hasattr(rep, "edge_proj"):
            w = rep.edge_proj.weight.detach()  # (hidden_dim, 5)
            norms = w.norm(dim=0)
            names = ["q_axis", "r_axis", "diagonal", "signed_distance", "src_player"]
            for i, name in enumerate(names):
                writer.add_scalar(
                    f"edge_features/weight_norm/{name}",
                    norms[i].item(),
                    step,
                    description=(
                        "L2 norm of the learned edge-feature weight for each input channel. "
                        "Tracks how much the network is using each edge feature; weights "
                        "drifting to 0 mean a feature has been pruned."
                    ),
                )

        # --- Edge feature gradient attribution (small sample) ---
        if hasattr(rep, "edge_proj"):
            try:
                model.eval()
                total_grad = torch.zeros(rep.edge_proj.weight.shape[1], device=self.device)
                node_dim = rep.input_proj.weight.shape[1]
                total_node_grad = torch.zeros(node_dim, device=self.device)
                count = 0

                for data in self._analysis_sample_graphs(10):
                    edge_attr = data.edge_attr.to(self.device).clone().requires_grad_(True)
                    x = data.x.to(self.device).clone().requires_grad_(True)
                    policy, value = model(
                        x,
                        data.edge_index.to(self.device),
                        data.legal_mask.to(self.device),
                        data.stone_mask.to(self.device),
                        edge_attr=edge_attr,
                    )
                    (policy.sum() + value.sum()).backward()
                    if edge_attr.grad is not None:
                        total_grad += edge_attr.grad.abs().mean(dim=0)
                        count += 1
                    if x.grad is not None:
                        total_node_grad += x.grad.abs().mean(dim=0)
                    model.zero_grad()

                if count > 0:
                    mean_grad = total_grad / count
                    for i, name in enumerate(names):
                        writer.add_scalar(
                            f"edge_features/gradient/{name}",
                            mean_grad[i].item(),
                            step,
                            description=(
                                "Mean absolute gradient flowing into each edge-feature "
                                "weight. Indicates which features are still being actively "
                                "updated."
                            ),
                        )
                    mean_node_grad = total_node_grad / count
                    for i, name in enumerate(
                        _node_feature_names(self.model_config)[:node_dim]
                    ):
                        writer.add_scalar(
                            f"node_features/gradient/{name}",
                            mean_node_grad[i].item(),
                            step,
                            description=(
                                "Mean absolute gradient w.r.t. each node-feature input "
                                "channel (same sampled positions as edge_features/gradient). "
                                "How sensitive the output currently is to each feature."
                            ),
                        )
            except Exception as e:
                logger.debug("Edge gradient analysis failed: %s", e)

        # --- Per-layer contribution ---
        # Measure how much each GATv2Conv layer changes the representation
        # relative to the residual. ratio = ||conv_out|| / ||residual||
        # High ratio = layer is active. Near-zero = layer is a skip-through.
        try:
            from hexo_a0.graph import game_to_axis_graph
            import random

            model.eval()
            game = hexo_rs.GameState(self.game_config)
            n_moves = random.randint(5, min(20, self.game_config.max_moves - 1))
            for _ in range(n_moves):
                moves = game.legal_moves()
                if not moves or game.is_terminal():
                    break
                m = random.choice(moves)
                game.apply_move(m[0], m[1])

            if not game.is_terminal():
                data = game_to_axis_graph(
                    game,
                    prune_empty_edges=self.model_config.prune_empty_edges,
                    threat_features=getattr(self.model_config, "threat_features", False),
                    relative_stones=getattr(self.model_config, "relative_stone_encoding", False),
                )
                x = data.x.to(self.device)
                edge_index = data.edge_index.to(self.device)
                edge_attr = data.edge_attr.to(self.device) if data.edge_attr is not None else None

                with torch.no_grad():
                    h = rep.input_proj(x)
                    projected_edge_attr = None
                    if hasattr(rep, "edge_proj") and edge_attr is not None:
                        projected_edge_attr = rep.edge_proj(edge_attr)

                    layer_scales = getattr(rep, "layer_scales", None)
                    for i, (conv, norm) in enumerate(zip(rep.convs, rep.norms)):
                        residual = h
                        conv_out = conv(h, edge_index, edge_attr=projected_edge_attr)
                        # Measure contribution before residual addition
                        conv_norm = conv_out.norm().item()
                        res_norm = residual.norm().item()
                        # Ratio: how much the conv changes vs the residual
                        ratio = conv_norm / max(res_norm, 1e-8)
                        writer.add_scalar(
                            f"layer_analysis/conv_ratio/layer_{i}",
                            ratio,
                            step,
                            description=(
                                "Per-layer ratio `||conv_output|| / ||residual_input||`, "
                                "computed BEFORE LayerScale gating. Preserves comparability "
                                "with historical pre-LayerScale curves. Low = residual "
                                "dominates / layer is a skip-through; high = layer is "
                                "rewriting the representation. See `gated_conv_ratio` for "
                                "the LayerScale-corrected effective contribution."
                            ),
                        )
                        if layer_scales is not None:
                            gated_out = layer_scales[i].detach() * conv_out
                            gated_ratio = gated_out.norm().item() / max(res_norm, 1e-8)
                            writer.add_scalar(
                                f"layer_analysis/gated_conv_ratio/layer_{i}",
                                gated_ratio,
                                step,
                                description=(
                                    "Per-layer ratio `||layer_scale * conv_output|| / "
                                    "||residual_input||` — the layer's actual effective "
                                    "contribution to the residual stream. Compare with "
                                    "`conv_ratio` to read what the LayerScale gate is "
                                    "doing: gated < raw means the gate is suppressing "
                                    "this layer's output. For a freshly-grafted layer "
                                    "(layer_scale init 1e-4) gated_ratio starts ~1e-4 "
                                    "even when raw conv_ratio is order-1."
                                ),
                            )
                        writer.add_scalar(
                            f"layer_analysis/conv_norm/layer_{i}",
                            conv_norm,
                            step,
                            description=(
                                "L2 norm of the conv output at each layer. Should be roughly "
                                "stable across layers in a healthy network."
                            ),
                        )
                        writer.add_scalar(
                            f"layer_analysis/residual_norm/layer_{i}",
                            res_norm,
                            step,
                            description=(
                                "L2 norm of the residual stream at each layer. Tends to grow "
                                "with depth in residual networks."
                            ),
                        )

                        # Continue forward for next layer (un-gated propagation, by design,
                        # to preserve `conv_ratio` comparability with historical curves —
                        # see `gated_conv_ratio` description above for the corrected metric).
                        h = conv_out + residual
                        h = norm(h)
                        h = rep.activation(h)
        except Exception as e:
            logger.debug("Layer analysis failed: %s", e)

        # --- LayerScale norms ---
        # Logs ||layer_scale[i]|| per layer when LayerScale is enabled.
        # Watch for: (a) grafted layer's norm growing above its 1e-4 init
        # — evidence the new layer is taking on work; (b) any layer 0-2
        # norm shrinking — evidence the model has redistributed load.
        if hasattr(rep, "layer_scales"):
            for i, ls in enumerate(rep.layer_scales):
                writer.add_scalar(
                    f"layer_analysis/layer_scale_norm/layer_{i}",
                    ls.detach().norm().item(),
                    step,
                    description=(
                        "L2 norm of the per-channel LayerScale gate for "
                        "this layer. Default init 1.0 (=> sqrt(hidden_dim)). "
                        "Grafted layers init at 1e-4 — growth above noise "
                        "indicates the layer has begun contributing."
                    ),
                )

        # --- JK aggregation weights ---
        # Logs the softmax-normalised mix the policy/value heads see, per layer.
        # Uniform 1/L => network treats all layers equally; mass concentrated
        # on h_L => network is effectively bypassing JK; non-trivial spread
        # across layers is the H2 prediction (per
        # docs/research/2026-05-13-jknet-axis-multiscale-hypothesis.md).
        if hasattr(rep, "jk_weights"):
            jk_softmax = rep.jk_weights.detach().float().softmax(dim=0)
            for i, w in enumerate(jk_softmax.tolist()):
                writer.add_scalar(
                    f"jk/weight/layer_{i}",
                    w,
                    step,
                    description=(
                        "softmax(jk_weights)[layer_i] — the fraction of the "
                        "head input that comes from this conv layer's "
                        "post-residual output under JK weighted-sum "
                        "aggregation. Sums to 1 across layers. Uniform => "
                        "no learned multi-scale preference; concentrated "
                        "on the top layer => JK is bypassed; spread across "
                        "layers => H2 prediction landing."
                    ),
                )

        # --- JK 'cat' mode head attention ---
        # In cat mode, PyG's JumpingKnowledge has no learnable parameters —
        # the routing decision lives in the heads' first Linear weight
        # matrix, which has shape (out_features, L*H) instead of
        # (out_features, H). Columns [i*H : (i+1)*H] are the weights
        # multiplying h_i in concat(h_0, ..., h_{L-1}).
        #
        # This block logs the cat-mode analog of jk/weight/*: per-head,
        # per-layer column-norm and energy-share. At step 0 after a fresh
        # head-graft layers 0..L-2 have norm 0 (zero-init) and L-1 carries
        # the original pre-graft weights → share = [0,...,0,1]. Growth in
        # the lower-layer norms is direct evidence the head is recruiting
        # multi-scale features; a flat trajectory means cat-mode is being
        # bypassed the same way scalar-sum JK was in jk-long.
        if (
            getattr(self.model_config, "use_jk", False)
            and getattr(self.model_config, "jk_mode", "sum") == "cat"
        ):
            L = self.model_config.num_layers
            H = self.model_config.hidden_dim
            heads = (("policy_head", model.policy_head), ("value_head", model.value_head))
            for head_name, head in heads:
                W = head.mlp[0].weight.detach()
                if W.shape[1] != L * H:
                    # Defensive: shouldn't happen for jk_mode='cat'.
                    continue
                slice_sq = torch.tensor([
                    W[:, i * H : (i + 1) * H].pow(2).sum().item()
                    for i in range(L)
                ])
                slice_norms = slice_sq.sqrt()
                shares = slice_sq / slice_sq.sum().clamp_min(1e-12)
                for i in range(L):
                    writer.add_scalar(
                        f"jk_cat/{head_name}/layer_weight_norm/layer_{i}",
                        slice_norms[i].item(),
                        step,
                        description=(
                            "Frobenius norm of the L-slice of the head's first "
                            "Linear weight matrix corresponding to layer i in "
                            "concat(h_0,...,h_{L-1}). At step 0 after a fresh "
                            "head-graft, layers 0..L-2 have norm 0; growth "
                            "indicates the head is recruiting lower-layer "
                            "features. Layer L-1 carries the original pre-graft "
                            "weights."
                        ),
                    )
                    writer.add_scalar(
                        f"jk_cat/{head_name}/layer_weight_share/layer_{i}",
                        shares[i].item(),
                        step,
                        description=(
                            "Energy share of layer i in the head's first Linear: "
                            "||W[:, i*H:(i+1)*H]||² / Σ_j ||W[:, j*H:(j+1)*H]||². "
                            "Sums to 1 across layers. At step 0 after head-graft "
                            "= [0,...,0,1]. Concentrated on the last layer over "
                            "long training = JK-cat being bypassed; spread "
                            "across layers = the head is using multi-scale "
                            "features (the desired regime)."
                        ),
                    )

                # Per-output-channel Projector embedding. Each of the
                # head's `out_features` output channels is a point in L-dim
                # space whose coordinates are its share vector across the
                # L conv layers. Metadata carries the channel index (so
                # hover-to-identify works in TB) and the dominant-layer
                # argmax (so the "color by dominant_layer" UI control
                # gives you instant visual segregation by routing).
                #
                # At step 0 after head-graft every point sits at
                # [0,0,0,1] — a single cluster on the layer_{L-1} axis.
                # Successful multi-scale recruitment shows points
                # migrating off that corner as training proceeds. Use
                # Projector's "Custom" projection to pick any 2 of the L
                # dims directly (e.g. share_layer_0 vs share_layer_3)
                # since L is small enough that PCA isn't doing anything
                # useful here.
                try:
                    per_ch = torch.stack([
                        W[:, i * H : (i + 1) * H].norm(dim=1)
                        for i in range(L)
                    ])  # (L, out_features)
                    per_ch = per_ch / per_ch.sum(dim=0, keepdim=True).clamp_min(1e-12)
                    dominant = per_ch.argmax(dim=0)  # (out_features,)
                    metadata = [
                        [f"ch_{j}", f"dom_L{int(dominant[j])}"]
                        for j in range(per_ch.shape[1])
                    ]
                    writer.add_embedding(
                        mat=per_ch.T,  # (out_features, L)
                        metadata=metadata,
                        metadata_header=["channel", "dominant_layer"],
                        global_step=step,
                        tag=f"jk_cat/{head_name}/channel_routing",
                    )
                except Exception as e:
                    logger.debug(
                        "jk_cat projector failed for %s: %s", head_name, e,
                    )

        # --- Ablation (every 5th eval) ---
        if hasattr(rep, "edge_proj") and step % (eval_every * 5) < analysis_every:
            try:
                import torch.nn.functional as F

                model.eval()
                edge_w = rep.edge_proj.weight.shape[1]
                names = ["q_axis", "r_axis", "diagonal", "signed_distance", "src_player"][:edge_w]

                # Baselines on on-policy buffer positions (NOT random play: the
                # training distribution is where importance must be measured).
                baselines = []
                with torch.no_grad():
                    for data in self._analysis_sample_graphs(10):
                        policy, value = model(
                            data.x.to(self.device), data.edge_index.to(self.device),
                            data.legal_mask.to(self.device), data.stone_mask.to(self.device),
                            edge_attr=data.edge_attr.to(self.device),
                        )
                        baselines.append((data, F.softmax(policy, dim=0), float(value.reshape(-1)[0])))

                    n_base = max(len(baselines), 1)
                    for feat_idx in range(edge_w):
                        kl_total = 0.0
                        dv_total = 0.0  # mean |Δvalue| — VALUE-head reliance
                        for data, baseline_p, baseline_v in baselines:
                            edge_attr = data.edge_attr.to(self.device).clone()
                            edge_attr[:, feat_idx] = 0.0
                            policy, value = model(
                                data.x.to(self.device), data.edge_index.to(self.device),
                                data.legal_mask.to(self.device), data.stone_mask.to(self.device),
                                edge_attr=edge_attr,
                            )
                            kl_total += F.kl_div(
                                F.softmax(policy, dim=0).log(), baseline_p, reduction="sum"
                            ).item()
                            dv_total += abs(float(value.reshape(-1)[0]) - baseline_v)
                        writer.add_scalar(
                            f"edge_features/ablation_kl/{names[feat_idx]}",
                            kl_total / n_base,
                            step,
                            description=(
                                "KL divergence between the model's POLICY and the policy "
                                "when this edge feature is zeroed out (on-policy buffer "
                                "positions). Higher = the feature contributes more to the "
                                "policy. Causal importance proxy — POLICY head only."
                            ),
                        )
                        writer.add_scalar(
                            f"edge_features/value_ablation_l1/{names[feat_idx]}",
                            dv_total / n_base,
                            step,
                            description=(
                                "Mean |Δvalue| when this edge feature is zeroed out "
                                "(on-policy buffer positions). The VALUE-head analogue of "
                                "ablation_kl; a feature can be policy-irrelevant yet "
                                "value-load-bearing (e.g. src_player under relative "
                                "encoding), which ablation_kl alone would miss."
                            ),
                        )

            except Exception as e:
                logger.debug("Ablation analysis failed: %s", e)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Stop self-play workers, close buffer, and close the tensorboard writer."""
        self.stop_self_play()
        self.buffer.close()
        if self.writer is not None:
            self.writer.close()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, *, save_buffer: bool = False) -> None:
        """Save training state to a .pt file.

        GameConfig is serialised as a plain dict because PyO3 objects are
        not picklable.  TrainingConfig and ModelConfig are serialised with
        dataclasses.asdict() so all fields are captured automatically.

        Args:
            save_buffer: If True, also back up the replay buffer to a companion
                         ``replay_buffer.db`` file via sqlite3 backup API.
                         Skipped for periodic saves to avoid I/O overhead.
        """
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        # Strip _orig_mod. prefixes added by torch.compile so checkpoints
        # are loadable into uncompiled models (e.g. evaluate_vs_checkpoint).
        raw_sd = self.model.state_dict()
        model_sd = {k.removeprefix("_orig_mod."): v for k, v in raw_sd.items()}
        # Parameter names in the same flat order they appear when iterating
        # optimizer.param_groups[*]['params']. Used by load_checkpoint to
        # route Adam state by parameter NAME across optimizer-group changes
        # (e.g. 1-group → 2-group when LayerScale was enabled mid-run, or
        # depth extension where new layer params are absent from the ckpt).
        param_id_to_name = {
            id(p): n.removeprefix("_orig_mod.")
            for n, p in self.model.named_parameters()
        }
        param_names_in_order = [
            param_id_to_name.get(id(p), "")
            for group in self.optimizer.param_groups
            for p in group["params"]
        ]
        # Atomic write: save to a tempfile in the same directory, then
        # os.replace into place. Without this, readers (e.g. the SPRT
        # daemon) can torch.load() the file mid-write and hit
        # "failed finding central directory" because torch.save writes
        # the zip central directory last.
        import os, tempfile
        path_parent = Path(path).parent
        path_parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path_parent), suffix=".pt.tmp")
        os.close(fd)
        torch.save(
            {
                "model_state_dict": model_sd,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "param_names_in_order": param_names_in_order,
                "scheduler_state_dict": scheduler_state,
                "train_steps": self.train_steps,
                "training_config": asdict(self.training_config),
                "model_config": asdict(self.model_config),
                # PyO3 objects are not picklable — store as plain dict.
                "game_config": {
                    "win_length": self.game_config.win_length,
                    "placement_radius": self.game_config.placement_radius,
                    "max_moves": self.game_config.max_moves,
                },
                "stage_start_step": self.stage_start_step,
                "sealbot_level_idx": self._sealbot_level_idx,
                "sealbot_consecutive_above": self._sealbot_consecutive_above,
                "current_step_delay": self.current_step_delay,
            },
            tmp,
        )
        os.replace(tmp, path)  # atomic — readers see old or new, never partial
        if save_buffer:
            ckpt_dir = Path(path).parent
            # Distinct from the live buffer path (<ckpt_dir>/replay_buffer.db),
            # otherwise backup() opens a second SQLite connection to the live
            # DB and races the ingestion writer -> "database is locked".
            buffer_backup_path = ckpt_dir / "replay_buffer_backup.db"
            self.buffer.backup(buffer_backup_path)

    def load_checkpoint(self, path) -> None:
        """Load training state from a .pt file.

        The caller's training_config and game_config take precedence; only
        model weights, optimizer state, and the step counter are restored.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            # Old checkpoints may contain non-tensor objects (e.g. TrainingExample).
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Normalize checkpoint keys: strip _orig_mod. prefixes from old
        # compiled-model checkpoints, then add them back if the current
        # model is compiled (has _orig_mod. keys in its state_dict).
        ckpt_sd = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model_state_dict"].items()}
        # Head graft for jk_mode='cat': opt-in via training.graft_heads_for_jk_cat.
        # Must run BEFORE the _orig_mod. prefix is re-added (we operate on
        # un-prefixed keys) and BEFORE graft_state_dict / load_state_dict.
        heads_grafted_for_jk = False
        if getattr(self.training_config, "graft_heads_for_jk_cat", False):
            mc = self.model_config
            if mc.use_jk and mc.jk_mode == "cat":
                heads_grafted_for_jk = graft_heads_for_jk_cat(
                    self.model, ckpt_sd,
                    hidden_dim=mc.hidden_dim,
                    num_layers=mc.num_layers,
                )
                if heads_grafted_for_jk:
                    logger.warning(
                        "Head graft for jk_mode='cat': policy/value head first "
                        "Linear expanded from (out, H=%d) to (out, L*H=%d). Old "
                        "weights placed in last H columns; other columns zero-init. "
                        "At step 0 the forward output is bit-identical to the "
                        "no-JK checkpoint.",
                        mc.hidden_dim, mc.num_layers * mc.hidden_dim,
                    )
            else:
                logger.warning(
                    "training.graft_heads_for_jk_cat=true but "
                    "model.use_jk=%s, model.jk_mode=%r — skipping head graft "
                    "(should have been caught by config validation).",
                    mc.use_jk, mc.jk_mode,
                )
        # Input-proj graft for threat_features: opt-in via
        # training.graft_threat_features. Same ordering constraints as the
        # jk-cat head graft above (un-prefixed keys, before load_state_dict).
        input_proj_grafted = False
        if getattr(self.training_config, "graft_threat_features", False):
            if getattr(self.model_config, "threat_features", False):
                input_proj_grafted = graft_input_proj_for_threat_features(
                    ckpt_sd, self.model_config
                )
                if input_proj_grafted:
                    logger.warning(
                        "Input-proj graft for threat_features: input_proj expanded "
                        "(H=%d, 8) -> (H=%d, 12); old weights in columns 0..8, "
                        "threat columns zero-init. Forward is bit-identical at "
                        "step 0.",
                        self.model_config.hidden_dim, self.model_config.hidden_dim,
                    )
            else:
                logger.warning(
                    "training.graft_threat_features=true but model.threat_features "
                    "is false — skipping input-proj graft."
                )
        model_sd_keys = set(self.model.state_dict().keys())
        if any(k.startswith("_orig_mod.") for k in model_sd_keys):
            ckpt_sd = {"_orig_mod." + k: v for k, v in ckpt_sd.items()}
        missing, unexpected, grafted = graft_state_dict(self.model, ckpt_sd)
        if grafted:
            logger.warning(
                "Depth-extension graft: layers %s newly added (zero-init MLP "
                "+ LayerScale=0 if enabled). Existing layers preserved bit-exact.",
                grafted,
            )
        if missing:
            logger.warning("New parameters (random init): %s", missing)
        if unexpected:
            logger.warning("Ignored checkpoint parameters: %s", unexpected)
        # Fresh-params condition: any state-dict key that the current model
        # expects but the checkpoint did not provide. Triggers warmup-on-arch-
        # -change to prevent Adam first-step blow-up on the new parameters.
        # JK-cat head grafting also introduces zero-initialised columns in the
        # head Linears, which carry the same Adam-step risk class even though
        # the parameter NAMES are unchanged (so `missing`/`grafted` don't fire).
        fresh_params_introduced = (
            bool(missing) or bool(grafted) or heads_grafted_for_jk
            or input_proj_grafted
        )
        # Restore scheduler state if present. If the LR config changed or
        # the saved state is incompatible, recreate the optimizer+scheduler
        # from scratch — Adam moments are then routed back below by name.
        ckpt_sched = checkpoint.get("scheduler_state_dict")
        sched_restored = False
        if self.scheduler is None and ckpt_sched is None:
            sched_restored = True  # no scheduler on either side
        elif self.scheduler is not None and ckpt_sched is not None:
            ckpt_lr = checkpoint.get("optimizer_state_dict", {}).get(
                "param_groups", [{}])[0].get("lr")
            config_lr = self.training_config.lr
            if ckpt_lr is not None and abs(ckpt_lr - config_lr) > 1e-9:
                logger.debug("LR changed (%.2e -> %.2e) — reinitialising scheduler",
                             ckpt_lr, config_lr)
            else:
                try:
                    self.scheduler.load_state_dict(ckpt_sched)
                    sched_restored = True
                except Exception:
                    pass
        if not sched_restored:
            # Recreate optimizer + scheduler from scratch with correct LR.
            # This is safer than patching initial_lr which breaks LinearLR.
            logger.debug("Scheduler state mismatch — recreating optimizer + scheduler")
            self.optimizer = torch.optim.AdamW(
                split_param_groups(
                    self.model,
                    self.training_config.weight_decay,
                    self.training_config.layer_scale_weight_decay,
                ),
                lr=self.training_config.lr,
            )
            self._init_scheduler()

        # Route saved Adam state into the (possibly fresh) optimizer by
        # parameter NAME. Handles three scenarios uniformly:
        #   (a) exact-match resume — all params recovered;
        #   (b) 1-group → 2-group transition (LayerScale enabled mid-run);
        #   (c) depth-extension graft (new layer params absent from
        #       checkpoint — initialised fresh; existing layers preserved).
        # Pass the saved model state_dict's keys so legacy checkpoints
        # (no ``param_names_in_order``) can be reconstructed by filtering
        # the current model's named_parameters() down to params that
        # actually existed in the saved model.
        saved_model_param_names = {
            k.removeprefix("_orig_mod.")
            for k in checkpoint["model_state_dict"].keys()
        }
        recovery_stats = self._load_optimizer_state_by_name(
            checkpoint["optimizer_state_dict"],
            checkpoint.get("param_names_in_order"),
            saved_model_param_names=saved_model_param_names,
        )
        if recovery_stats["recovered"] == recovery_stats["total_new"] \
                and recovery_stats["skipped"] == 0:
            logger.info(
                "Optimizer state restored: all %d params (Adam moments preserved)",
                recovery_stats["total_new"],
            )
        else:
            logger.info(
                "Optimizer state: %d/%d params restored "
                "(%d fresh, %d skipped from checkpoint)",
                recovery_stats["recovered"],
                recovery_stats["total_new"],
                recovery_stats["fresh"],
                recovery_stats["skipped"],
            )
        self.train_steps = checkpoint.get("train_steps", checkpoint.get("iteration", 0) * 100)
        self.stage_start_step = checkpoint.get("stage_start_step", 0)
        # Compute the realignment short-chunk: if we resumed mid-chunk,
        # burn just enough steps on the first post-resume iteration to land
        # train_steps back on a chunk boundary. The chunk size here must
        # match the default `steps` argument of Trainer.train() (the value
        # used by curriculum.py and cli.py callers).
        _chunk_size = 100
        _rem = self.train_steps % _chunk_size
        self._resume_short_chunk = (_chunk_size - _rem) % _chunk_size

        # Fast-forward a freshly created scheduler to match resumed steps.
        # Without this, cosine LR restarts from step 0 on resume.
        # Use stage_start_step to only fast-forward by steps within the
        # current stage. Skip if no work done in this stage yet (the
        # curriculum will reset the scheduler on stage transitions anyway).
        #
        # Exception: if the checkpoint introduced fresh parameters (depth-
        # extension graft or newly-added LayerScale params) and
        # warmup_on_arch_change is True, do NOT fast-forward — the scheduler
        # starts at step 0 so warmup applies. Adam's first-step normalization
        # gives unit-magnitude updates to fresh params regardless of init
        # scale, so warmup is load-bearing for graft stability. The trainer's
        # internal step counter (self.train_steps) is preserved either way.
        # `do_lr_warmup_on_resume` takes precedence when set: we always fast-
        # forward the underlying scheduler to the resume step so the wrapper's
        # target_lr matches the cosine schedule, then wrap with a fresh ramp.
        # Falling back to `warmup_on_arch_change` keeps prior behaviour for the
        # depth-extension / LayerScale-graft use case.
        do_resume_warmup = (
            getattr(self.training_config, "do_lr_warmup_on_resume", False)
            and self.scheduler is not None
        )
        skip_ff_for_warmup = (
            (not do_resume_warmup)
            and fresh_params_introduced
            and getattr(self.training_config, "warmup_on_arch_change", True)
        )
        if not sched_restored and self.scheduler is not None:
            if skip_ff_for_warmup:
                logger.info(
                    "Fresh params detected (grafted=%s, missing=%d, head_jk_graft=%s); "
                    "applying full warmup from scheduler step 0 "
                    "(warmup_on_arch_change=True)",
                    grafted, len(missing), heads_grafted_for_jk,
                )
            else:
                steps_in_stage = max(0, self.train_steps - self.stage_start_step)
                if steps_in_stage > 0:
                    for _ in range(steps_in_stage):
                        self.scheduler.step()
                    logger.info("Fast-forwarded scheduler by %d steps", steps_in_stage)

        # Opt-in: linear LR warmup on resume that ends at the cosine schedule's
        # LR for the current step. Reuses ``lr_warmup_steps`` for the ramp
        # length (a single warmup knob keeps cold-start and resume warmups
        # consistent — there's no principled reason a resume-warmup needs a
        # different length than the cold-start one, and adding a second knob
        # invites mistuning). Skipped when the resume LR is already at/below
        # lr_min (nothing to ramp toward).
        if do_resume_warmup:
            current_lr = self.optimizer.param_groups[0]["lr"]
            start_lr = float(self.training_config.lr_min)
            ramp_steps = max(1, int(self.training_config.lr_warmup_steps))
            if current_lr <= start_lr + 1e-12:
                logger.info(
                    "do_lr_warmup_on_resume: skipping wrapper — scheduler is "
                    "already at/below lr_min=%.2e (current=%.2e).",
                    start_lr, current_lr,
                )
            else:
                self.scheduler = _ResumeWarmupScheduler(
                    self.scheduler,
                    self.optimizer,
                    start_lr=start_lr,
                    target_lr=current_lr,
                    warmup_steps=ramp_steps,
                )
                logger.info(
                    "do_lr_warmup_on_resume: linear ramp %.2e -> %.2e over "
                    "%d steps, then defer to cosine schedule.",
                    start_lr, current_lr, ramp_steps,
                )
        self._sealbot_level_idx = checkpoint.get("sealbot_level_idx", 0)
        self._sealbot_consecutive_above = checkpoint.get("sealbot_consecutive_above", 0)
        self.current_step_delay = checkpoint.get("current_step_delay", self.training_config.step_delay)
        # Restore replay buffer (REQ-22): try formats in priority order.
        ckpt_dir = Path(path).parent
        db_backup_path = ckpt_dir / "replay_buffer_backup.db"
        legacy_buffer_path = str(path).replace(".pt", "_buffer.pt")
        live_db = Path(self.buffer._db_path) if not self.buffer._is_memory else None
        # Priority 1: the live DB, opened by __init__, is the newest source of
        # truth when it has content. Overwriting it with the stage-transition
        # backup (only written on save_buffer=True) discards ingestion done
        # since the last transition.
        if live_db is not None and len(self.buffer) > 0:
            logger.info("Resumed with live replay buffer (%d examples)", len(self.buffer))
        elif db_backup_path.exists():
            # Priority 2: stage-transition backup (live DB empty or missing).
            self.buffer.set_state(db_backup_path)
            logger.info("Restored replay buffer from %s (%d examples)", db_backup_path, len(self.buffer))
        elif Path(legacy_buffer_path).exists():
            # Priority 3: Legacy companion _buffer.pt file
            buffer_data = torch.load(legacy_buffer_path, map_location="cpu", weights_only=False)
            for ex in buffer_data:
                if ex.data is not None:
                    ex.data = ex.data.cpu()
                ex.policy_target = ex.policy_target.cpu()
            self.buffer.set_state(buffer_data)
            logger.info("Restored replay buffer from legacy %s (%d examples)", legacy_buffer_path, len(self.buffer))
        elif "replay_buffer" in checkpoint:
            # Priority 4: Inline key in old-format checkpoints
            buffer_data = checkpoint["replay_buffer"]
            for ex in buffer_data:
                if ex.data is not None:
                    ex.data = ex.data.cpu()
                ex.policy_target = ex.policy_target.cpu()
            self.buffer.set_state(buffer_data)
            logger.info("Restored replay buffer from inline key (%d examples)", len(self.buffer))

    def _load_optimizer_state_by_name(
        self,
        saved_optim_state: dict,
        saved_param_names: list[str] | None,
        saved_model_param_names: set[str] | None = None,
    ) -> dict[str, int]:
        """Route saved Adam state into the current optimizer by parameter NAME.

        Handles legacy 1-group checkpoints (no ``param_names_in_order``),
        depth-extended models where some current params are absent from the
        checkpoint, and exact-match resumes uniformly.

        Returns a stats dict with keys: ``recovered`` (params with state
        carried over), ``fresh`` (current params with no saved state —
        Adam moments will lazily init to zero on first ``.step()``),
        ``skipped`` (saved state entries that did not match any current
        param name), and ``total_new`` (total params in the current
        optimizer).
        """
        saved_state: dict = saved_optim_state.get("state", {}) or {}
        saved_param_groups: list = saved_optim_state.get("param_groups", []) or []

        # Build the flat list of saved param IDs in the order they appear
        # across saved param_groups. The i-th id corresponds to the i-th
        # entry in saved_param_names (same flat ordering at save time).
        flat_saved_ids = [pid for g in saved_param_groups for pid in g.get("params", [])]
        n_saved = len(flat_saved_ids)

        # Reconstruct or use the saved name list.
        if saved_param_names is not None and len(saved_param_names) == n_saved:
            names_for_saved_positions = list(saved_param_names)
        else:
            # Legacy checkpoint: params were saved in the order produced
            # by `model.named_parameters()` at save time over the LEGACY
            # model. Filter the CURRENT model's named_parameters down to
            # names that existed in the saved model state — that gives
            # us the legacy ordering, even if the current model has
            # interleaved new params (e.g. LayerScale, grafted layers).
            current_names = [
                n.removeprefix("_orig_mod.") for n, _ in self.model.named_parameters()
            ]
            if saved_model_param_names is not None:
                filtered = [n for n in current_names if n in saved_model_param_names]
            else:
                filtered = current_names
            if n_saved > len(filtered):
                logger.warning(
                    "Saved optimizer has %d params but only %d of the current "
                    "model's params match saved model state; cannot reconstruct "
                    "legacy name list — skipping name-based recovery.",
                    n_saved, len(filtered),
                )
                total_new = sum(len(g["params"]) for g in self.optimizer.param_groups)
                return {
                    "recovered": 0,
                    "fresh": total_new,
                    "skipped": n_saved,
                    "total_new": total_new,
                }
            names_for_saved_positions = filtered[:n_saved]

        # Build name -> per-parameter state map.
        name_to_state: dict[str, dict] = {}
        for i, old_id in enumerate(flat_saved_ids):
            entry = saved_state.get(old_id)
            if entry is None:
                continue
            name = names_for_saved_positions[i]
            if name:
                name_to_state[name] = entry

        # Map current Parameter id() -> name (normalised, no _orig_mod.).
        id_to_name: dict[int, str] = {
            id(p): n.removeprefix("_orig_mod.")
            for n, p in self.model.named_parameters()
        }

        # Walk the current optimizer's param_groups in order and route
        # state. Build the new state map keyed by NEW positional id.
        new_state: dict[int, dict] = {}
        recovered = fresh = 0
        new_id = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                name = id_to_name.get(id(p), "")
                entry = name_to_state.pop(name, None) if name else None
                if entry is not None:
                    # Validate that the moment tensors match the current
                    # parameter shape. ``step`` is a 0-dim scalar regardless
                    # of param shape, so it must be exempted.
                    compatible = True
                    for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        v = entry.get(k)
                        if torch.is_tensor(v) and v.shape != p.shape:
                            compatible = False
                            logger.warning(
                                "Optimizer state shape mismatch for %r "
                                "(saved %s vs current %s) — reinitialised.",
                                name, tuple(v.shape), tuple(p.shape),
                            )
                            break
                    if compatible:
                        new_state[new_id] = entry
                        recovered += 1
                    else:
                        fresh += 1
                else:
                    fresh += 1
                new_id += 1

        skipped = len(name_to_state)  # saved entries that matched no current param

        # Construct a state_dict matching the current optimizer's shape
        # (param_groups carry the current LR, weight_decay, etc.) and
        # load. PyTorch will reconstruct internal Parameter-keyed state
        # from our positional id mapping.
        new_full = self.optimizer.state_dict()
        new_full["state"] = new_state
        self.optimizer.load_state_dict(new_full)

        return {
            "recovered": recovered,
            "fresh": fresh,
            "skipped": skipped,
            "total_new": new_id,
        }
