#!/usr/bin/env python3
"""Benchmark Rust native self-play for hex and axis graph types.

Exports temporary TorchScript models, runs the self-play binary, and
reports games/s for each configuration.

Usage:
    python scripts/bench_self_play.py [--games N] [--sims N]
    just bench-self-play

Also supports a subprocess A/B mode for comparing inference backends
(python-gpu / python-cpu / infer-native) against the production self-play
command:

    python scripts/bench_self_play.py --ab --arm python-gpu \\
        --checkpoint /path/to/model_selfplay.pt --win-length 6 --radius 8 \\
        --games 4 --json out.json
"""

import argparse
import json
import os
import re
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


# ---------------------------------------------------------------------------
# Subprocess A/B mode: compares inference backends against the exact
# production self-play command (see PRODUCTION_PYTHON_BIN / build_ab_argv).
# ---------------------------------------------------------------------------

PRODUCTION_PYTHON_BIN = "/var/home/tyto/Development/personal/hexo-strix/.venv/bin/python3"

ARMS = ("python-gpu", "python-cpu", "infer-native")

# Distinctive stderr line the pure-Rust hexo-infer-server prints right before
# READY (see hexo-rs/hexo-infer/src/server.rs log_metadata). The Python
# inference_server.py never prints this — it's the spawn-verification signal
# for the infer-native arm.
NATIVE_SPAWN_MARKER = "hexo-infer-server: loaded"

DONE_LINE_RE = re.compile(
    r"Done: (\d+) games, (\d+) moves, ([\d.]+)s \(([\d.]+)s/game, ([\d.]+) games/s\)"
)


def find_self_play_binary_ab(override: str | None) -> str:
    if override:
        p = Path(override)
        if not p.exists():
            print(f"ERROR: --binary {override} does not exist", file=sys.stderr)
            sys.exit(1)
        return str(p)
    candidates = [
        Path("hexo-rs/target-forcing/release/self_play"),
        Path("target-forcing/release/self_play"),
        Path("hexo-rs/target/release/self_play"),
        Path("target/release/self_play"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    which = shutil.which("self_play")
    if which:
        return which
    print("ERROR: self_play binary not found for --ab mode (searched target-forcing "
          "and target release dirs). Pass --binary explicitly.", file=sys.stderr)
    sys.exit(1)


def assert_checkpoint_is_a_copy(checkpoint: str) -> None:
    """Sanity-guard: refuse to run against a live checkpoint under runs/.

    Benches must operate on copies (see task's copy-first rule) — pointing at
    a checkpoint still inside runs/ risks reading a file mid-write by a live
    training/self-play process, or (worse) racing a promotion swap.
    """
    p = Path(checkpoint)
    parts = set(p.parts) | set(p.resolve().parts)
    if "runs" in parts:
        print(
            f"ERROR: --checkpoint {checkpoint} is inside runs/ — copy it out first "
            "(bench must never point at live training/self-play checkpoints).",
            file=sys.stderr,
        )
        sys.exit(1)


def build_ab_argv(
    *,
    checkpoint: str,
    games: int,
    device: str,
    win_length: int,
    radius: int,
    max_moves: int,
    sims: int,
    threads: int,
    output_path: str,
    inference_bin: str | None,
) -> list[str]:
    """Build the self_play argv mirroring the production command exactly.

    Mirrors the live production invocation minus --continuous and the
    rotating-output flags (--output-dir/--max-file-mb/--output-filename),
    which are replaced here with a single --output <tmpfile> for one-shot
    A/B runs.
    """
    argv = [
        "--model", checkpoint,
        "--graph-type", "axis",
        "--games", str(games),
        "--device", device,
        "--win-length", str(win_length),
        "--radius", str(radius),
        "--max-moves", str(max_moves),
        "--sims", str(sims),
        "--m-actions", "16",
        "--c-scale", "1.0",
        "--c-visit", "50",
        "--exploration", "15",
        "--exploration-fraction", "0.2",
        "--exploration-distribution", "visit_counts",
        "--truncate-delta", "0.25",
        "--threads", str(threads),
        "--max-batch", "512",
        "--max-batch-edges", "45000",
        "--batch-timeout-ms", "5",
        "--prune-empty-edges",
        "--threat-features",
        "--relative-stones",
        "--python-inference",
        "--python-bin", PRODUCTION_PYTHON_BIN,
        "--checkpoint", checkpoint,
        "--model-hidden-dim", "128",
        "--model-num-layers", "4",
        "--model-num-heads", "8",
        "--model-policy-hidden", "128",
        "--model-value-hidden", "32",
        "--model-conv-type", "gine",
        "--model-use-jk",
        "--model-jk-mode", "cat",
        "--python-inference-workers", "2",
        "--output", output_path,
    ]
    if inference_bin:
        argv += ["--inference-bin", inference_bin]
    return argv


def run_ab(args) -> int:
    if args.arm not in ARMS:
        print(f"ERROR: --arm must be one of {ARMS}, got {args.arm!r}", file=sys.stderr)
        return 1
    if not args.checkpoint:
        print("ERROR: --ab mode requires --checkpoint", file=sys.stderr)
        return 1
    if args.win_length is None or args.radius is None:
        print("ERROR: --ab mode requires --win-length and --radius", file=sys.stderr)
        return 1
    if args.arm == "infer-native" and not args.inference_bin:
        print("ERROR: --arm infer-native requires --inference-bin", file=sys.stderr)
        return 1

    assert_checkpoint_is_a_copy(args.checkpoint)

    binary = find_self_play_binary_ab(args.binary)
    device = {"python-gpu": "cuda", "python-cpu": "cpu", "infer-native": "cpu"}[args.arm]
    games = args.games if args.games is not None else 20
    sims = args.sims if args.sims is not None else 128
    threads = args.threads if args.threads is not None else 32
    max_moves = args.max_moves if args.max_moves is not None else 300
    inference_bin = args.inference_bin if args.arm == "infer-native" else None

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    argv = build_ab_argv(
        checkpoint=args.checkpoint,
        games=games,
        device=device,
        win_length=args.win_length,
        radius=args.radius,
        max_moves=max_moves,
        sims=sims,
        threads=threads,
        output_path=output_path,
        inference_bin=inference_bin,
    )
    cmd = [binary] + argv

    # Subprocess A/B mode: clean env, NO LD_PRELOAD. The old in-process mode's
    # ROCm libtorch_hip.so preload is only needed when self_play itself links
    # torch in-process; here self_play runs as a pure-Rust binary talking to a
    # separate python (or hexo-infer-server) subprocess over stdin/stdout, and
    # preloading ROCm HIP libs into the Rust binary crashes its HIP runtime.
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)

    print(f"[{args.arm}] {' '.join(cmd)}", file=sys.stderr)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        stderr = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        print(f"[{args.arm}] FAILED (timeout after {args.timeout}s)", file=sys.stderr)
        _write_json(args.json, {
            "arm": args.arm, "ok": False, "error": f"timeout after {args.timeout}s",
            "elapsed_wall_s": elapsed, "stderr_tail": stderr[-4000:],
        })
        return 1
    elapsed_wall = time.perf_counter() - t0

    stderr = result.stderr or ""
    stdout = result.stdout or ""
    perf_lines = [ln for ln in stderr.splitlines() if "[perf]" in ln]
    done_match = DONE_LINE_RE.search(stderr)
    spawn_verified = NATIVE_SPAWN_MARKER in stderr

    Path(output_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"[{args.arm}] FAILED (rc={result.returncode})", file=sys.stderr)
        print(stderr[-4000:], file=sys.stderr)
        _write_json(args.json, {
            "arm": args.arm, "ok": False, "rc": result.returncode,
            "elapsed_wall_s": elapsed_wall, "stderr_tail": stderr[-4000:],
            "perf_lines": perf_lines, "cmd": cmd,
        })
        return 1

    if done_match:
        reported_games = int(done_match.group(1))
        reported_moves = int(done_match.group(2))
        reported_elapsed = float(done_match.group(3))
        games_per_s = float(done_match.group(5))
    else:
        reported_games = games
        reported_moves = None
        reported_elapsed = elapsed_wall
        games_per_s = games / elapsed_wall if elapsed_wall > 0 else 0.0

    print(
        f"[{args.arm}] games={reported_games} moves={reported_moves} "
        f"elapsed_wall={elapsed_wall:.2f}s reported_elapsed={reported_elapsed:.2f}s "
        f"games/s={games_per_s:.3f} spawn_verified={spawn_verified}",
        file=sys.stderr,
    )
    if args.arm == "infer-native" and not spawn_verified:
        print(
            f"WARNING: --arm infer-native but stderr did not contain "
            f"'{NATIVE_SPAWN_MARKER}' — the native server may not have been spawned.",
            file=sys.stderr,
        )

    _write_json(args.json, {
        "arm": args.arm,
        "ok": True,
        "rc": result.returncode,
        "games": reported_games,
        "moves": reported_moves,
        "elapsed_wall_s": elapsed_wall,
        "reported_elapsed_s": reported_elapsed,
        "games_per_s": games_per_s,
        "spawn_verified": spawn_verified,
        "perf_lines": perf_lines,
        "cmd": cmd,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-6000:],
    })
    return 0


def _write_json(path: str | None, payload: dict) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Rust native self-play")
    parser.add_argument("--games", type=int, default=None, help="Games per config/run")
    parser.add_argument("--sims", type=int, default=None, help="MCTS simulations")
    parser.add_argument("--threads", type=int, default=None, help="Game threads (0 = auto)")
    parser.add_argument("--omp", type=int, default=0, help="OMP threads per game thread (0 = auto, legacy mode only)")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device: cpu or cuda (legacy mode only)")

    # --ab subprocess A/B mode
    parser.add_argument("--ab", action="store_true", help="Subprocess A/B mode (production-mirroring self_play invocation)")
    parser.add_argument("--arm", type=str, choices=list(ARMS), help="Inference backend arm for --ab mode")
    parser.add_argument("--checkpoint", type=str, help="Path to model_selfplay.pt copy (--ab mode)")
    parser.add_argument("--inference-bin", type=str, help="Path to hexo-infer-server binary (infer-native arm)")
    parser.add_argument("--win-length", type=int, default=None, help="Win length (--ab mode)")
    parser.add_argument("--radius", type=int, default=None, help="Board radius (--ab mode)")
    parser.add_argument("--max-moves", type=int, default=None, help="Max moves override (--ab mode; production default 300)")
    parser.add_argument("--binary", type=str, default=None, help="self_play binary path override (--ab mode)")
    parser.add_argument("--json", type=str, default=None, help="Write machine-readable result JSON here (--ab mode)")
    parser.add_argument("--timeout", type=int, default=600, help="Subprocess timeout in seconds (--ab mode)")

    args = parser.parse_args()

    if args.ab:
        sys.exit(run_ab(args))

    # --- Legacy in-process-torch benchmark mode (unchanged behavior) ---
    if args.games is None:
        args.games = 20
    if args.sims is None:
        args.sims = 16
    if args.threads is None:
        args.threads = 0

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
