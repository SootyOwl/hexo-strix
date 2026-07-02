# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Version Control

This is a single unified repository. It uses **Jujutsu (jj)** on a colocated git backend (`.jj/` + `.git/`), so `jj` and `git` both work; prefer `jj` for VCS operations. `hexo-rs/` and `hexo-a0/` are subdirectories of this one repo, not separate histories.

## Build & Test Commands

```bash
just sync          # uv sync — rebuilds hexo-rs Python extension via maturin
just self-play     # Build Rust native self-play binary (requires libtorch + C++ compiler)
just build         # sync + self-play
just test          # Rust + Python tests
just test-all      # Including GPU tests
just test-rust     # cargo test -p hexo-mcts only
just test-python   # pytest hexo-a0/tests/ only (skips gpu tests by default)
```

Single test: `uv run --no-sync pytest hexo-a0/tests/test_foo.py::test_name --tb=short`
Single Rust test: `cd hexo-rs && cargo test -p hexo-mcts test_name`

**Always pass `--no-sync` to `uv run`** to avoid unnecessary dependency resolution.

Training: `just train-axis` (axis-window graphs) or `just train-hex` (hex-adjacency graphs).

## Architecture

**uv workspace** with two members that communicate via PyO3 bindings:

### hexo-rs/ (Rust, built with maturin)
Two crates in a Cargo workspace:
- **hexo-engine**: Game logic — immutable game state, board, hex coordinates, win detection, symmetry. Optional TUI feature.
- **hexo-mcts**: Gumbel AlphaZero MCTS, graph builders (hex + axis), batch tensor ops, PyO3 bindings (`python.rs`), and a native `self_play` binary (feature-gated behind `torch`).

The PyO3 module `hexo_rs` exports: `PyGameState`, `PyGameConfig`, `PyMCTSConfig`, `gumbel_mcts()`, graph conversion functions (`game_to_graph_raw`, `game_to_axis_graph_raw`, batch variants), and `batched_self_play()`.

### hexo-a0/ (Python)
CLI entry point `hexo-a0` with `train` and `curriculum` subcommands (`cli.py`).
- **model.py**: `RepresentationNetwork` (GATv2Conv GNN) + `HeXONet` (policy/value heads)
- **trainer.py**: Training loop with replay buffer
- **self_play.py**: Game generation using Rust or Python MCTS
- **curriculum.py**: Multi-stage curriculum with convergence detection
- **graph.py**: Converts game states to PyG `Data` objects (wraps Rust graph builders)
- **mcts.py**: Pure Python MCTS fallback

Python dynamically imports `hexo_rs` at function level, not module level.

### Graph Types
- **hex-adjacency** (`graph_type = "hex"`): Edges between hex-adjacent nodes. Needs ~9 GATv2Conv layers.
- **axis-window** (`graph_type = "axis"`): Edges along 3 hex axes within `win_length - 1` steps, with edge features. 3 layers suffice.

Set via `graph_type` in the `[model]` section of curriculum TOML configs (`configs/`).

## Development Workflow

Use `/phase-development` for multi-phase implementation and `/adversarial-review` for review gates.

Write analysis, benchmark results, and other findings interesting for a future research paper to `docs/research/`.

## Platform Notes

- Requires Python 3.13+, Rust toolchain, uv, and a C++ compiler
- Uses ROCm PyTorch wheels on Linux (configured in root `pyproject.toml`)
- The Rust self-play binary must run WITHOUT `LD_PRELOAD` of ROCm libs (crashes HIP runtime); only Python training needs the preload
