# HeXO

**HeXO** is infinite hex tic-tac-toe — two players alternate placing stones (two per turn) on an *unbounded* hexagonal grid, racing to form a line of *N* in a row. There are no edges: the board starts with a single stone at the origin and grows outward as play continues.

This monorepo contains the game engine, an AlphaZero-style training pipeline that learns to play it, and a server for playing against a trained model.

> **Status:** a solo research project built around a single-GPU (ROCm APU) setup. The core *build → train → watch → evaluate* loop works and is what most people will want; several subsystems (SPRT promotion gating, remote self-play, the play server) are shaped around one author's hardware and workflow. Expect research-grade rough edges rather than a turnkey product.

## What's in here

| Path | What it is |
|------|-----------|
| [`hexo-rs/`](hexo-rs/README.md) | Rust workspace — the game engine (`hexo-engine`) and the Gumbel MCTS + graph builders + PyO3 bindings (`hexo-mcts`, exported as the `hexo_rs` Python module). |
| [`hexo-a0/`](hexo-a0/README.md) | Python package `hexo-a0` — self-play, GNN training, curriculum, evaluation, and the play server. Depends on `hexo_rs`. |
| `configs/` | TOML training/curriculum configs (axis-graph families, size tiers, ablations). |
| `scripts/` | Two helper scripts: human-corpus evaluation and a self-play benchmark. |
| `docker/`, `Dockerfile`, `compose.yaml` | Container packaging for the play server. |
| `justfile` | Convenience recipes (tuned to the author's ROCm + Homebrew setup). |

The two packages are wired together as a [uv](https://docs.astral.sh/uv/) workspace, so one `uv sync` builds the Rust extension (via [maturin](https://www.maturin.rs/)) and installs the Python side together.

## How it works

HeXO learns policy and value functions with a **Graph Neural Network** and searches with **Gumbel MCTS** (Danihelka et al., 2022):

- The board is encoded as a graph. The default `axis` representation connects stones along the 3 win axes (with 5-dim edge features and a global node), letting 3 message-passing layers cover the same receptive field that ~9 layers of plain hex adjacency would need.
- The network (`HeXONet`) is a representation GNN — `GINEConv` by default, `GATv2Conv` optionally — plus per-node policy and stone-pooled value heads.
- Training alternates **self-play** (Gumbel MCTS generates games), **learning** (KL policy + MSE value loss on a replay buffer), and **evaluation** (vs. random, a lagged checkpoint, and [SealBot](https://github.com/Ramora0/HexTicTacToe)), progressing through a **curriculum** of increasing board sizes.

See [`hexo-rs/README.md`](hexo-rs/README.md) for the engine/rules and [`hexo-a0/README.md`](hexo-a0/README.md) for the training details.

## Quick start

**Prerequisites:** Python 3.13+, a Rust toolchain, a C++ compiler, and `uv`.

```bash
git clone https://github.com/SootyOwl/hexo-strix
cd hexo-strix

# Sync the workspace: builds the hexo_rs Rust extension and installs hexo-a0.
# Pick the torch backend that matches your hardware:
uv sync --group cpu          # or: --group cuda  |  --group rocm
```

That's the portable path. The `justfile` wraps this and the native self-play build for the author's environment (`just sync`, `just build`, `just test`) — handy on a matching ROCm setup, but not required.

### Train

`curriculum` is the main training loop — it runs the full self-play → learn → evaluate cycle through a schedule of increasing board sizes.

```bash
# Generate an annotated curriculum config, then train small board -> full HeXO
uv run --no-sync hexo-a0 default-curriculum > curriculum.toml
uv run --no-sync hexo-a0 curriculum --config curriculum.toml --device cuda:0
```

Ready-made configs live in `configs/` (e.g. `configs/axis/curriculum.toml`). (`hexo-a0 train` runs a single fixed stage instead of a curriculum; `curriculum` is the maintained path.)

### Play against / watch a trained model

`serve` is the current way to see a model play — an HTTP play-and-analyze server with selectable difficulty tiers that records games to SQLite and exposes a token-gated `/admin` page:

```bash
uv run --no-sync hexo-a0 serve --config config.toml \
  --checkpoint <ckpt>.pt --port 8765
```

(There's also an older `hexo-a0 watch` pygame viewer, largely superseded by `serve`.)

### Play in the terminal (no model)

```bash
cargo run --bin play -p hexo-engine --features tui
```

## Testing

```bash
cargo test -p hexo-mcts                             # Rust
uv run --no-sync pytest hexo-a0/tests/ --tb=short   # Python
# or: just test
```

## Evaluation opponents

[SealBot](https://github.com/Ramora0/HexTicTacToe) (a C++ minimax engine) is the main non-trivial evaluation opponent. It is **not vendored here** — to enable SealBot evaluation, clone `Ramora0/HexTicTacToe` (tag `sealbot-v1.0`) into `sealbot/` at the repo root and build its `minimax_cpp` extension (`pip install -r requirements.txt` then `python setup.py build_ext --inplace`, which needs a C++ compiler and pybind11). Training also always evaluates against a uniform-random opponent and a lagged checkpoint, and can optionally replay against a human game corpus.

## License

MIT — see [`LICENSE`](LICENSE) (per-package `LICENSE` files also apply).
