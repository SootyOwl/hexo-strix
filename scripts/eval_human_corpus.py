#!/usr/bin/env python
"""Evaluate a checkpoint against the human game corpus.

Replays the 6,902-game human corpus (HF timmyburn/hexo-bootstrap-corpus,
validated in docs/research/2026-06-10-human-corpus-validation.md) and measures:

  Policy (raw prior, no MCTS): perplexity / top-1 / top-16 coverage of the
  human move at every placement, bucketed by stone count x pair-average Elo.

  Value: MSE, sign accuracy, and 10-bin calibration at end-of-turn boundary
  states (the pinned value-target convention), bucketed by stone count.

Prints summary tables and appends one summary record per invocation to
data/corpus/eval_results.jsonl.

Usage:
    uv run --no-sync python scripts/eval_human_corpus.py \
        --checkpoint runs/<run>/checkpoints/checkpoint_NNNNNNNN.pt \
        --config configs/<family>/<run>.toml
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import random
import re
import tomllib
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus" / "hexo_human_corpus.jsonl"
DEFAULT_RESULTS = REPO_ROOT / "data" / "corpus" / "eval_results.jsonl"
# Pinned to a specific HF revision, NOT `main`: the corpus eval deliberately
# stays on the 6,902-game `54fae7ae…` corpus for comparability with historical
# runs (jkcat-rel2 s1–s3, lean-d6-qhead, …). HF `main` was regenerated on
# 2026-07-07 to a newer, larger 8,698-game corpus (`b2fe61eb…`); pinning the
# revision keeps `ensure_corpus` downloads on the canonical version regardless
# of `main` drifting. To adopt the new corpus, update CORPUS_REVISION (→ the
# new commit, or `main`) AND CORPUS_SHA256 together.
CORPUS_REVISION = "e685b34eddeef1dbea29fe4e13c13cf33cf26d45"
CORPUS_URL = (
    "https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus"
    f"/resolve/{CORPUS_REVISION}/hexo_human_corpus.jsonl"
)
CORPUS_SHA256 = "54fae7aebcef2a9d19d13c1946fae36c0565e21bc726c25e2e4e230cfb42a5b7"

# Buckets: (lo, hi) inclusive; hi=None means unbounded.
STONE_BUCKETS = [(1, 20), (21, 40), (41, 80), (81, 160), (161, None)]
ELO_BUCKETS = [(0, 999), (1000, 1199), (1200, None)]
TOP_K = 16  # m_actions candidate-set size; coverage probes the prior's candidate gap
N_CALIBRATION_BINS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None,
                   help="curriculum TOML with a [model] section; fallback for checkpoints "
                        "without an embedded model_config")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--max-games", type=int, default=None,
                   help="seeded random sample of games (default: all)")
    p.add_argument("--batch-size", type=int, default=128,
                   help="max positions per forward (count cap; the edge budget "
                        "is usually what triggers the flush on deep positions)")
    p.add_argument("--edge-budget", type=int, default=500_000,
                   help="flush when the estimated (upper-bound) edge count of "
                        "pending graphs exceeds this (bounds peak activation "
                        "memory; 0 = off). Mirrors [training].edge_budget — "
                        "deep r8 positions reach ~100k unpruned edges each, so "
                        "unbudgeted 1024-position batches cost tens of GiB of "
                        "transient activations")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="also dump the summary record to stdout")
    return p.parse_args()


def ensure_corpus(path: Path) -> None:
    if not path.exists():
        print(f"corpus not found at {path}, downloading from HuggingFace ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CORPUS_SHA256:
        raise SystemExit(
            f"corpus checksum mismatch at {path}:\n  got      {digest}\n  expected {CORPUS_SHA256}"
        )


def load_model(ckpt_path: str, config_path: str | None):
    """Build HeXONet from the checkpoint's embedded model_config (authoritative —
    graft scripts update it in place), falling back to the TOML's [model] section."""
    import torch

    from hexo_a0 import config as config_mod
    from hexo_a0 import model as model_mod

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc_dict = ckpt.get("model_config")
    if mc_dict is None:
        if config_path is None:
            raise SystemExit(
                f"{ckpt_path} has no embedded model_config; pass --config <curriculum.toml>"
            )
        mc_dict = tomllib.loads(Path(config_path).read_text()).get("model", {})
    mc = config_mod.model_config_from_checkpoint(ckpt, mc_dict)
    model = model_mod.HeXONet(mc)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, mc, ckpt


def make_batch_graph_fn(mc):
    """states -> list[Data], built in one rayon-parallel Rust call."""
    from hexo_a0.graph import game_to_axis_graph_batch, game_to_graph_batch

    prune = getattr(mc, "prune_empty_edges", False)
    threat = getattr(mc, "threat_features", False)
    rel = getattr(mc, "relative_stone_encoding", False)
    if mc.graph_type == "axis":
        return lambda gs: game_to_axis_graph_batch(
            gs, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
    return lambda gs: game_to_graph_batch(gs, threat_features=threat, relative_stones=rel)


def derive_step(ckpt_path: str, ckpt: dict) -> int | None:
    m = re.match(r"checkpoint_(\d+)\.pt$", Path(ckpt_path).name)
    if m:
        return int(m.group(1))
    if isinstance(ckpt.get("train_steps"), int):
        return int(ckpt["train_steps"])
    return None


def est_edges(state, stones: int) -> int:
    """Upper-bound axis-graph edge estimate for a queued position.

    Node count is exact (stones + the engine's legal-move count); each node has
    ≤ 30 axis-window neighbours (3 axes × ≤5 steps × 2 dirs). Pruned graphs
    (prune_empty_edges) come in well under this; unpruned graphs approach it.
    """
    return 30 * (stones + state.legal_move_count())


def bucket_index(buckets, x: int) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if x >= lo and (hi is None or x <= hi):
            return i
    raise ValueError(f"{x} fits no bucket in {buckets}")


def bucket_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


class PolicyAcc:
    def __init__(self):
        self.n = 0
        self.sum_nll = 0.0
        self.top1 = 0
        self.topk = 0

    def add(self, nll: float, rank: int) -> None:
        self.n += 1
        self.sum_nll += nll
        self.top1 += rank == 0
        self.topk += rank < TOP_K

    def merge(self, other: "PolicyAcc") -> None:
        self.n += other.n
        self.sum_nll += other.sum_nll
        self.top1 += other.top1
        self.topk += other.topk

    def row(self) -> dict:
        if self.n == 0:
            return {"n": 0, "perplexity": None, "top1": None, f"top{TOP_K}": None}
        return {
            "n": self.n,
            "perplexity": math.exp(self.sum_nll / self.n),
            "top1": self.top1 / self.n,
            f"top{TOP_K}": self.topk / self.n,
        }


class ValueAcc:
    def __init__(self):
        self.n = 0
        self.sum_sq = 0.0
        self.sign_ok = 0

    def add(self, pred: float, outcome: float) -> None:
        self.n += 1
        self.sum_sq += (pred - outcome) ** 2
        self.sign_ok += (pred >= 0) == (outcome > 0)

    def merge(self, other: "ValueAcc") -> None:
        self.n += other.n
        self.sum_sq += other.sum_sq
        self.sign_ok += other.sign_ok

    def row(self) -> dict:
        if self.n == 0:
            return {"n": 0, "mse": None, "sign_acc": None}
        return {"n": self.n, "mse": self.sum_sq / self.n, "sign_acc": self.sign_ok / self.n}


def iter_games(corpus_path: Path, max_games: int | None, seed: int):
    games = [json.loads(line) for line in corpus_path.open()]
    if max_games is not None and max_games < len(games):
        games = random.Random(seed).sample(games, max_games)
    return games


def replay_positions(game: dict, game_config):
    """Yield (state_clone, (q, r), value_target_or_None, stone_count) per placement.

    Skips the forced (0,0) opener (the engine pre-seeds it). value_target is set
    only at end-of-turn boundary states (start of a player's full turn) and is the
    final outcome from the side-to-move's perspective. Graph building and the
    human-move node lookup happen later, batched, in flush().
    """
    import hexo_rs

    gs = hexo_rs.GameState(game_config)
    moves = game["moves"]
    if moves[0] != [0, 0]:
        raise SystemExit(f"game {game['game_hash']}: expected forced (0,0) opener, got {moves[0]}")
    winner = "P1" if game["winner"] == 1 else "P2"

    for j, (q, r) in enumerate(moves[1:]):
        if gs.is_terminal():
            raise SystemExit(f"game {game['game_hash']}: move {j} after terminal state")
        value_target = None
        if j % 2 == 0:  # turn boundary: side-to-move has a full 2-stone turn ahead
            value_target = 1.0 if gs.current_player() == winner else -1.0
        yield gs.clone(), (q, r), value_target, j + 1  # j+1 stones incl. pre-seeded origin
        gs.apply_move(q, r)

    if not gs.is_terminal():
        raise SystemExit(f"game {game['game_hash']}: not terminal after full replay")


def main() -> None:
    args = parse_args()
    ensure_corpus(args.corpus)

    from hexo_a0.gpu_memory import CacheClearGate, configure_cuda_alloc

    # Must run before any device-touching torch call (allocator config is
    # read once at HIP/CUDA context init).
    configure_cuda_alloc()

    import torch
    from torch_geometric.data import Batch

    import hexo_rs

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Corpus graphs vary wildly in shape (late-game r8 positions are huge), so
    # the caching allocator retains every batch's high-water blocks; on
    # unified-memory APUs that squeezes the OS (GTT = system RAM) and
    # gc_threshold never fires. With edge-budgeted batches a per-flush trim
    # costs a handful of ms and keeps the footprint flat.
    cache_gate = CacheClearGate.for_device(torch.device(device), every=1)

    model, mc, ckpt = load_model(args.checkpoint, args.config)
    model = model.to(device)
    batch_graph_fn = make_batch_graph_fn(mc)
    game_config = hexo_rs.GameConfig(win_length=6, placement_radius=8, max_moves=1000)

    games = iter_games(args.corpus, args.max_games, args.seed)
    print(f"evaluating {len(games)} games on {device} "
          f"(graph_type={mc.graph_type}, threat_features={getattr(mc, 'threat_features', False)}, "
          f"relative_stone_encoding={getattr(mc, 'relative_stone_encoding', False)})")

    policy_acc = {(s, e): PolicyAcc() for s in range(len(STONE_BUCKETS))
                  for e in range(len(ELO_BUCKETS))}
    value_acc = {s: ValueAcc() for s in range(len(STONE_BUCKETS))}
    # Calibration bins over predicted v in [-1, 1]
    cal = [{"n": 0, "sum_pred": 0.0, "sum_outcome": 0.0} for _ in range(N_CALIBRATION_BINS)]

    pending: list = []  # (state_clone, (q, r), value_target, stone_bucket, elo_bucket)
    pending_edges = 0  # estimated; bounds peak activation memory per flush
    use_axis_fast_path = mc.graph_type == "axis"

    def _forward_axis_bytes(states):
        """Rust-collated flat batch (see graph.axis_states_to_batch).

        Returns (flat_legal_logits, values, seg, legal_coords), all on device. Bypasses per-graph Data objects and PyG collation.
        """
        from hexo_a0.graph import axis_states_to_batch

        batch, aux = axis_states_to_batch(
            states,
            prune_empty_edges=getattr(mc, "prune_empty_edges", False),
            threat_features=getattr(mc, "threat_features", False),
            relative_stones=getattr(mc, "relative_stone_encoding", False),
            device=device,
        )
        logits, _counts, values = model._forward_batch_core(
            batch, legal_idx=aux.legal_idx,
            stone_idx=aux.stone_idx, stone_batch=aux.stone_batch,
        )
        seg = batch.batch[aux.legal_idx]
        legal_coords = aux.coords[aux.legal_idx]
        return logits.float(), values, seg, legal_coords

    def _forward_pyg(states):
        """Per-graph Data + PyG collation fallback (hex graphs)."""
        datas = batch_graph_fn(states)
        batch = Batch.from_data_list(datas).to(device)
        logits_list, values = model.forward_batch(batch)
        logits = torch.cat(logits_list).float()  # flat over legal nodes, batch order
        counts = torch.tensor([len(l) for l in logits_list], device=device)
        n = len(states)
        seg = torch.repeat_interleave(torch.arange(n, device=device), counts)
        return logits, values, seg, batch.coords[batch.legal_mask]

    def flush() -> None:
        nonlocal pending_edges
        pending_edges = 0
        if not pending:
            return
        n = len(pending)
        states = [p[0] for p in pending]
        moves_t = torch.tensor([p[1] for p in pending], dtype=torch.int32, device=device)

        with torch.inference_mode():
            fwd = _forward_axis_bytes if use_axis_fast_path else _forward_pyg
            logits, values, seg, legal_coords = fwd(states)

            # Human-move flat index: exactly one (q, r) match within each graph's
            # legal nodes (replay-validated corpus guarantees legality).
            match = (legal_coords == moves_t[seg]).all(dim=1)
            human_flat = match.nonzero(as_tuple=False).squeeze(1)
            if human_flat.numel() != n or not torch.equal(
                seg[human_flat], torch.arange(n, device=device)
            ):
                raise SystemExit("human move lookup failed: != 1 legal match in some graph")

            # Segment log-softmax NLL + rank, all on device, synced at the end.
            seg_max = torch.full((n,), float("-inf"), device=device).scatter_reduce_(
                0, seg, logits, "amax")
            log_z = torch.zeros(n, device=device).index_add_(
                0, seg, torch.exp(logits - seg_max[seg])).log() + seg_max
            nll = (log_z - logits[human_flat]).tolist()
            rank = torch.zeros(n, dtype=torch.long, device=device).index_add_(
                0, seg, (logits > logits[human_flat][seg]).long()).tolist()
            values = values.tolist()

        for (_, _, value_target, sb, eb), nll_i, rank_i, pred in zip(pending, nll, rank, values):
            policy_acc[(sb, eb)].add(nll_i, rank_i)
            if value_target is not None:
                value_acc[sb].add(pred, value_target)
                b = min(int((pred + 1.0) / 2.0 * N_CALIBRATION_BINS), N_CALIBRATION_BINS - 1)
                cal[b]["n"] += 1
                cal[b]["sum_pred"] += pred
                cal[b]["sum_outcome"] += value_target
        pending.clear()
        cache_gate.step()

    from tqdm import tqdm

    for game in tqdm(games, desc="games", unit="game"):
        elo_bucket = bucket_index(ELO_BUCKETS, int(sum(game["elo"]) / 2))
        for state, move, value_target, stones in replay_positions(game, game_config):
            stone_bucket = bucket_index(STONE_BUCKETS, stones)
            pending.append((state, move, value_target, stone_bucket, elo_bucket))
            pending_edges += est_edges(state, stones)
            if len(pending) >= args.batch_size or (
                args.edge_budget > 0 and pending_edges >= args.edge_budget
            ):
                flush()
    flush()

    # ---- aggregate ----
    policy_overall = PolicyAcc()
    for acc in policy_acc.values():
        policy_overall.merge(acc)
    value_overall = ValueAcc()
    for acc in value_acc.values():
        value_overall.merge(acc)

    n_cal = sum(b["n"] for b in cal)
    ece = sum(
        b["n"] / n_cal * abs(b["sum_pred"] / b["n"] - b["sum_outcome"] / b["n"])
        for b in cal if b["n"] > 0
    )

    # ---- print tables ----
    s_labels = [bucket_label(*b) for b in STONE_BUCKETS]
    e_labels = [bucket_label(*b) for b in ELO_BUCKETS]

    po = policy_overall.row()
    vo = value_overall.row()
    print(f"\n== Overall ==")
    print(f"policy: n={po['n']}  perplexity={po['perplexity']:.3f}  "
          f"top1={po['top1']:.3f}  top{TOP_K}={po[f'top{TOP_K}']:.3f}")
    print(f"value:  n={vo['n']}  mse={vo['mse']:.4f}  sign_acc={vo['sign_acc']:.3f}  ece={ece:.4f}")

    def fmt(x, spec):
        return format(x, spec) if x is not None else "-"

    print(f"\n== Policy perplexity (top1 / top{TOP_K}) by stones x Elo ==")
    header = f"{'stones':>8} | " + " | ".join(f"{l:>24}" for l in e_labels + ["all"])
    print(header)
    print("-" * len(header))
    for s in range(len(STONE_BUCKETS)):
        row_all = PolicyAcc()
        cells = []
        for e in range(len(ELO_BUCKETS)):
            r = policy_acc[(s, e)].row()
            row_all.merge(policy_acc[(s, e)])
            cells.append(f"{fmt(r['perplexity'], '8.2f')} ({fmt(r['top1'], '.2f')}/{fmt(r[f'top{TOP_K}'], '.2f')})")
        r = row_all.row()
        cells.append(f"{fmt(r['perplexity'], '8.2f')} ({fmt(r['top1'], '.2f')}/{fmt(r[f'top{TOP_K}'], '.2f')})")
        print(f"{s_labels[s]:>8} | " + " | ".join(f"{c:>24}" for c in cells))

    print("\n== Value by stones ==")
    print(f"{'stones':>8} | {'n':>7} | {'mse':>7} | {'sign_acc':>8}")
    for s in range(len(STONE_BUCKETS)):
        r = value_acc[s].row()
        print(f"{s_labels[s]:>8} | {r['n']:>7} | {fmt(r['mse'], '7.4f')} | {fmt(r['sign_acc'], '8.3f')}")

    print("\n== Value calibration (10 bins over predicted v) ==")
    print(f"{'bin':>12} | {'n':>7} | {'mean_pred':>9} | {'mean_outcome':>12}")
    for i, b in enumerate(cal):
        lo = -1.0 + 2.0 * i / N_CALIBRATION_BINS
        hi = lo + 2.0 / N_CALIBRATION_BINS
        if b["n"] == 0:
            print(f"{lo:>5.1f},{hi:>5.1f} | {0:>7} | {'-':>9} | {'-':>12}")
        else:
            print(f"{lo:>5.1f},{hi:>5.1f} | {b['n']:>7} | {b['sum_pred'] / b['n']:>9.3f} | "
                  f"{b['sum_outcome'] / b['n']:>12.3f}")

    # ---- append summary record ----
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": derive_step(args.checkpoint, ckpt),
        "config": str(Path(args.config).resolve()) if args.config else None,
        "corpus_sha256": CORPUS_SHA256,
        "n_games": len(games),
        "seed": args.seed,
        "policy": po,
        "value": {**vo, "ece": ece},
        "policy_buckets": {
            f"{s_labels[s]}|{e_labels[e]}": policy_acc[(s, e)].row()
            for s in range(len(STONE_BUCKETS)) for e in range(len(ELO_BUCKETS))
        },
        "value_buckets": {s_labels[s]: value_acc[s].row() for s in range(len(STONE_BUCKETS))},
        "calibration_bins": [
            {"n": b["n"],
             "mean_pred": b["sum_pred"] / b["n"] if b["n"] else None,
             "mean_outcome": b["sum_outcome"] / b["n"] if b["n"] else None}
            for b in cal
        ],
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nappended summary to {args.results}")
    if args.json:
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
