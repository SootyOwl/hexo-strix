#!/usr/bin/env python3
"""Benchmark Rust native self-play for hex and axis graph types.

Exports temporary TorchScript models, runs the self-play binary, and
reports games/s for each configuration.

Usage:
    python scripts/bench_self_play.py [--games N] [--sims N]
    just bench-self-play
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path


def find_self_play_binary() -> str | None:
    candidates = [
        Path("hexo-rs/target/release/self_play"),
        Path("target/release/self_play"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return shutil.which("self_play")


def find_torch_lib() -> str:
    import torch
    return str(Path(torch.__file__).parent / "lib")


def export_model(graph_type: str, path: str, num_layers: int) -> None:
    from hexo_a0.config import ModelConfig
    from hexo_a0.model import HeXONet
    from hexo_a0.scriptable_model import export_torchscript

    mc = ModelConfig(
        hidden_dim=256,
        num_heads=8,
        num_layers=num_layers,
        policy_hidden=128,
        value_hidden=128,
        graph_type=graph_type,
    )
    net = HeXONet(mc)
    export_torchscript(net.state_dict(), asdict(mc), path, bf16=False)


def run_self_play(
    binary: str,
    model_path: str,
    graph_type: str,
    win_length: int,
    radius: int,
    max_moves: int,
    n_games: int,
    n_sims: int,
    torch_lib: str,
    threads: int = 0,
    omp_threads: int = 0,
    device: str = "cpu",
) -> tuple[float, str]:
    """Run self-play binary and return (elapsed_seconds, stderr_output)."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    # ROCm: preload libtorch_hip.so so HIP runtime initializes for Device::Cuda
    hip_lib = Path(torch_lib) / "libtorch_hip.so"
    if hip_lib.exists():
        existing = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = f"{hip_lib}:{existing}" if existing else str(hip_lib)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        cmd = [
            binary,
            "--model", model_path,
            "--graph-type", graph_type,
            "--win-length", str(win_length),
            "--radius", str(radius),
            "--max-moves", str(max_moves),
            "--games", str(n_games),
            "--sims", str(n_sims),
            "--output", output_path,
        ]
        if threads > 0:
            cmd += ["--threads", str(threads)]
        if omp_threads > 0:
            cmd += ["--omp", str(omp_threads)]
        if device != "cpu":
            cmd += ["--device", device]

        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            return elapsed, "FAILED (timeout after 600s)"
        elapsed = time.perf_counter() - t0

        stderr = result.stderr.strip()
        if result.returncode != 0:
            return elapsed, f"FAILED (rc={result.returncode}): {stderr}"
        # Detect silent failures: binary exits 0 but played 0 moves
        if "0 moves" in stderr and "0 games" not in stderr:
            return elapsed, f"FAILED (0 moves — model error?): {stderr}"
        return elapsed, stderr
    finally:
        Path(output_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Rust native self-play")
    parser.add_argument("--games", type=int, default=20, help="Games per config")
    parser.add_argument("--sims", type=int, default=16, help="MCTS simulations")
    parser.add_argument("--threads", type=int, default=0, help="Game threads (0 = auto)")
    parser.add_argument("--omp", type=int, default=0, help="OMP threads per game thread (0 = auto)")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device: cpu or cuda")
    args = parser.parse_args()

    binary = find_self_play_binary()
    if not binary:
        print("ERROR: self_play binary not found. Run: just self-play", file=sys.stderr)
        sys.exit(1)

    torch_lib = find_torch_lib()

    # max_moves capped low — random models never win, so games always draw.
    # We want to measure inference throughput, not game length.
    configs = [
        # (label, graph_type, num_layers, win_length, radius, max_moves, games_override)
        ("Hex 9L  S1 (4×r2)", "hex",  9, 4, 2, 30, None),
        ("Axis 3L S1 (4×r2)", "axis", 3, 4, 2, 30, None),
        ("Hex 9L  S3 (5×r4)", "hex",  9, 5, 4, 30, None),
        ("Axis 3L S3 (5×r4)", "axis", 3, 5, 4, 30, None),
        ("Hex 9L  S5 (6×r6)", "hex",  9, 6, 6, 30, None),
        ("Axis 3L S5 (6×r6)", "axis", 3, 6, 6, 30, None),
    ]

    # Export models
    tmpdir = tempfile.mkdtemp(prefix="hexo_bench_")
    model_paths = {}
    print("Exporting TorchScript models...")
    for label, graph_type, num_layers, *_ in configs:
        key = f"{graph_type}_{num_layers}L"
        if key not in model_paths:
            path = os.path.join(tmpdir, f"{key}.pt")
            export_model(graph_type, path, num_layers)
            model_paths[key] = path
            print(f"  {key} -> {path}")

    # Run benchmarks
    thread_info = ""
    if args.threads > 0:
        thread_info += f", {args.threads} threads"
    if args.omp > 0:
        thread_info += f", {args.omp} omp"
    if not thread_info:
        thread_info = ", threads=auto"
    print(f"\nBenchmarking ({args.games} games, {args.sims} sims{thread_info})...")
    print(f"{'Config':<22}  {'Games':>6}  {'Time (s)':>9}  {'Games/s':>8}  {'s/game':>8}")
    print("-" * 65)

    for label, graph_type, num_layers, win_length, radius, max_moves, games_override in configs:
        key = f"{graph_type}_{num_layers}L"
        model_path = model_paths[key]
        n = games_override if games_override is not None else args.games

        elapsed, output = run_self_play(
            binary, model_path, graph_type,
            win_length, radius, max_moves,
            n, args.sims, torch_lib,
            threads=args.threads, omp_threads=args.omp,
            device=args.device,
        )

        if "FAILED" in output:
            print(f"{label:<22}  {output}")
        else:
            games_s = n / elapsed if elapsed > 0 else 0
            s_game = elapsed / n if n > 0 else 0
            print(f"{label:<22}  {n:>6}  {elapsed:>9.2f}  {games_s:>8.2f}  {s_game:>8.3f}")

    print("-" * 65)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
