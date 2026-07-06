"""A/B bench: axis-GINE `RepresentationNetwork` forward, dedupe-ON vs
dedupe-OFF (`DedupGINEConv`'s unique-edge-attr path vs. stock per-edge
`GINEConv`).

Times `net._forward_batch_core(batch)` directly (eager mode — this does NOT
exercise the `torch.compiler.is_dynamo_compiling()` guard added to
`RepresentationNetwork.forward`, which forces the stock path under
torch.compile regardless of this A/B) on an untrained model with a
production-like axis-graph config, for a batch of self-similar game-graph
copies. `--dedupe-force-stock` toggling is done via each conv's
`_dedup_force_stock` test/bench escape hatch.

Promoted from a throwaway script written for task 0.2 (see
docs/research/2026-07-06-infermodel-optimization-plan.md, "Phase 0").

Usage (CPU, default):
    uv run --no-sync python hexo-a0/scripts/bench_dedup_torch.py

Usage (GPU) — ONLY run when the GPU is free (this repo runs live training;
do not contend with it):
    uv run --no-sync python hexo-a0/scripts/bench_dedup_torch.py --device cuda
"""
import argparse
import time

import torch
import hexo_rs
from torch_geometric.data import Batch

from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_axis_graph
from hexo_a0.model import HeXONet


# Hand-tuned 20-move r=8 line (from the original throwaway bench) — known
# not to complete a win_length=6 run or trigger a double-four instant loss.
# (0, 0) is the engine's auto-seed (not itself an `apply_move` call).
_BASE_MOVES = [
    (0, 0), (1, 0), (0, 1), (-1, 1),
    (2, 0), (2, -1), (-2, 2), (-2, 1),
    (3, 0), (3, -1), (-3, 3), (-3, 2),
    (1, 1), (2, 1), (-1, 2), (0, 2),
    (4, 0), (4, -1), (-4, 4), (-4, 3),
]


def _extra_ring_moves(k: int):
    return [(k, 0), (k, -1), (-k, k), (-k, k - 1)]


def build_game_data(n_stones: int, radius: int, win_length: int = 6, max_moves: int = 400):
    """Build a single game graph with up to `n_stones` stones placed.

    Uses `_BASE_MOVES` (proven not to accidentally complete a win or a
    double-four instant loss for r=8) for the first 20 stones; beyond that,
    extrapolates further outward rings in the same style. This is NOT
    verified safe for every `--stones`/`--radius` combination — if the
    engine reports a terminal state (win or forced-loss) before `n_stones`
    moves are placed, this stops early and returns however many stones did
    land, with a warning, rather than crashing.
    """
    moves = list(_BASE_MOVES)
    k = 5
    while len(moves) < n_stones:
        moves.extend(_extra_ring_moves(k))
        k += 1
    moves = moves[:n_stones]

    game = hexo_rs.GameState(hexo_rs.GameConfig(win_length, radius, max_moves))
    placed = 0
    for x, y in moves[1:]:
        # Snapshot before each attempt so a move that ends the game (win or
        # forced loss) can be rolled back to the last non-terminal state
        # rather than leaving `game` in a state `game_to_axis_graph` rejects.
        last_good = game.clone()
        try:
            game.apply_move(x, y)
        except Exception as e:
            print(f"[bench_dedup_torch] apply_move({x}, {y}) failed after "
                  f"{placed} stones ({e!r}); using this shorter graph instead.")
            game = last_good
            break
        placed += 1
        if game.is_terminal():
            print(f"[bench_dedup_torch] game reached a terminal state after "
                  f"{placed} stones (< requested {n_stones}); rolling back "
                  f"to the last non-terminal graph.")
            game = last_good
            placed -= 1
            break
    data = game_to_axis_graph(
        game, prune_empty_edges=True, threat_features=True, relative_stones=True
    )
    return data, placed + 1


def time_forward(net, batch, device, n_repeats, n_warmup, force_stock):
    for conv in net.representation.convs:
        conv._dedup_force_stock = force_stock
    times = []
    with torch.no_grad():
        for _ in range(n_warmup):
            net._forward_batch_core(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
        for _ in range(n_repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            net._forward_batch_core(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def fmt(times):
    ms = sorted(t * 1000 for t in times)
    return f"best={ms[0]:.2f} ms  median={ms[len(ms) // 2]:.2f} ms  all={[f'{t:.2f}' for t in ms]} ms"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="Device to run the bench on. cuda requires the GPU "
                         "to be free (this repo runs live training).")
    p.add_argument("--batch", type=int, default=16,
                    help="Number of game-graph copies per batch.")
    p.add_argument("--stones", type=int, default=20,
                    help="Number of stones placed before snapshotting the graph.")
    p.add_argument("--radius", type=int, default=8, help="placement_radius.")
    p.add_argument("--threads", type=int, default=4,
                    help="torch.set_num_threads() value (cpu only).")
    p.add_argument("--repeats", type=int, default=5, help="Timed iterations (best-of-N reported).")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations (untimed).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    else:
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    mc = ModelConfig(
        hidden_dim=128, num_layers=4, num_heads=8, conv_type="gine",
        graph_type="axis", pre_norm=True, use_jk=True, jk_mode="cat",
        policy_hidden=128, value_hidden=32, threat_features=True,
        relative_stone_encoding=True, prune_empty_edges=True,
    )
    torch.manual_seed(0)
    net = HeXONet(mc).eval().to(device)

    data, n_stones = build_game_data(args.stones, args.radius)
    data = data.to(device)
    batch = Batch.from_data_list([data.clone() for _ in range(args.batch)])

    if device.type == "cpu":
        print(f"torch.set_num_threads({args.threads}); torch.get_num_threads()={torch.get_num_threads()}")
    print(f"device: {device}")
    print(f"game: {n_stones} stones placed, r={args.radius}, win_length=6")
    print(f"single-graph: {data.x.shape[0]} nodes / {data.edge_attr.shape[0]} edges "
          f"({torch.unique(data.edge_attr, dim=0).shape[0]} unique edge_attr rows)")
    print(f"batch: {args.batch} copies -> {batch.x.shape[0]} nodes / {batch.edge_attr.shape[0]} edges")
    print(f"config: hidden_dim={mc.hidden_dim}, num_layers={mc.num_layers}, "
          f"num_heads={mc.num_heads}, conv_type={mc.conv_type}, jk_mode={mc.jk_mode}")
    print(f"warmup={args.warmup}, repeats={args.repeats} (best-of-{args.repeats} reported)")
    print()

    off_times = time_forward(net, batch, device, args.repeats, args.warmup, force_stock=True)
    on_times = time_forward(net, batch, device, args.repeats, args.warmup, force_stock=False)

    print(f"dedupe-OFF (_dedup_force_stock=True):  {fmt(off_times)}")
    print(f"dedupe-ON  (wired path):                {fmt(on_times)}")
    best_off = min(off_times)
    best_on = min(on_times)
    speedup = best_off / best_on if best_on > 0 else float("nan")
    pct = (1 - best_on / best_off) * 100 if best_off > 0 else float("nan")
    print(f"\nbest-of-{args.repeats} speedup (OFF/ON): {speedup:.3f}x  ({pct:+.1f}% forward-time change)")


if __name__ == "__main__":
    main()
