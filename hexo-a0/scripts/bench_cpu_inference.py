"""Benchmark CPU inference for the public server deployment.

Measures:
- analyze_trajectory (raw value bar): one batched forward pass over game prefixes.
- analyze_position with MCTS at 16/32/64/128/256/512 sims on opening and midgame positions.

Run locally or on the Oracle ARM64 box:
    uv run --no-sync python hexo-a0/scripts/bench_cpu_inference.py \
        --checkpoint runs/gine-mini-4l-128p32v-jkcat/checkpoints/self_play/champion.pt \
        --device cpu
"""
import argparse
import os
import time

import hexo_rs
from hexo_a0.serving.model import load_model
from hexo_a0.serving.analysis import build_state, analyze_position, analyze_trajectory


def _game(win_length=6, placement_radius=8, max_moves=400):
    return hexo_rs.GameConfig(win_length, placement_radius, max_moves)


def opening_state():
    return build_state([(0, 0), (1, 0), (2, 0)], 6, 8, 400)


def midgame_state():
    # A plausible midgame line with ~20 stones.
    moves = [(0, 0), (1, 0), (0, 1), (-1, 1),
             (2, 0), (2, -1), (-2, 2), (-2, 1),
             (3, 0), (3, -1), (-3, 3), (-3, 2),
             (1, 1), (2, 1), (-1, 2), (0, 2)]
    return build_state(moves, 6, 8, 400)


def trajectory_game():
    return build_state([(0, 0), (1, 0), (2, 0), (-1, 1), (3, 0), (-2, 2),
                        (1, 1), (-3, 3), (2, 1), (0, 1), (3, -1), (-1, 2)], 6, 8, 400)


def fmt_seconds(t):
    if t < 1.0:
        return f"{t*1000:.1f} ms"
    return f"{t:.2f} s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sims", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    args = parser.parse_args()

    print(f"Loading {args.checkpoint} on {args.device} ...")
    t0 = time.perf_counter()
    model, mc, ckpt = load_model(args.checkpoint, args.device)
    print(f"  load: {fmt_seconds(time.perf_counter() - t0)}")
    print(f"  graph_type={mc.graph_type}, threat_features={getattr(mc, 'threat_features', False)}, "
          f"relative_stone_encoding={getattr(mc, 'relative_stone_encoding', False)}")

    print("\n--- analyze_trajectory (raw value bar) ---")
    moves = [(0, 0)] + trajectory_game().legal_moves()[:11]  # short 12-prefix game
    t0 = time.perf_counter()
    out = analyze_trajectory(model, mc, moves, 6, 8, 400)
    print(f"  {len(out['trajectory'])} prefixes: {fmt_seconds(time.perf_counter() - t0)}")

    for label, state in [("opening", opening_state()), ("midgame", midgame_state())]:
        print(f"\n--- {label} position ({len(state.legal_moves())} legal moves) ---")
        print(f"  raw analyze_position: ", end="", flush=True)
        t0 = time.perf_counter()
        analyze_position(model, mc, state, mcts_sims=0)
        print(fmt_seconds(time.perf_counter() - t0))

        for n in args.sims:
            t0 = time.perf_counter()
            analyze_position(model, mc, state, mcts_sims=n, mcts_m_actions=16)
            elapsed = time.perf_counter() - t0
            print(f"  MCTS n={n:>3}: {fmt_seconds(elapsed)}")


if __name__ == "__main__":
    main()
