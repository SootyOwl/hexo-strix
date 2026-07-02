# HeXO AlphaZero

A Gumbel AlphaZero training pipeline for **HeXO** — infinite hex tic-tac-toe on an unbounded hexagonal grid.

HeXO learns policy and value functions with a Graph Neural Network and searches with Gumbel MCTS (Danihelka et al., 2022). The fast game engine, MCTS, and graph construction come from the sibling Rust crate [`hexo-rs`](../hexo-rs/README.md) via PyO3 bindings; this package (`hexo-a0`) is the Python side — self-play, training, curriculum, evaluation, and a play server. It's part of the [HeXO](../README.md) monorepo.

> **Status:** this is a research codebase for a single-GPU (ROCm APU) setup. The core train / watch / evaluate loop is what most people will want; a number of subsystems (SPRT promotion gating, remote self-play, the play server) are built around one author's hardware and workflow. It should get you playing and training, but expect some rough edges.

## What the game is

Two players alternate placing stones (two per turn) on an unbounded hex grid, racing to form *N* in a row along one of the 3 hex axes. The board starts with a single P1 stone at the origin and grows outward; legal moves are empty hexes within a radius of existing stones. See [`hexo-rs`](../hexo-rs/README.md) for the full rules.

## Architecture

**Network (`HeXONet`)** — a representation GNN plus policy and value heads:

- **Representation network**: node/edge feature projection followed by stacked message-passing + LayerNorm residual blocks. The convolution is configurable:
  - `gine` (default) — `GINEConv`, sum aggregation with edge-feature injection, no attention (~1.5× faster).
  - `gatv2` — `GATv2Conv` additive attention.
- **Policy head**: per-node MLP scoring legal moves.
- **Value head**: stone-pooled aggregation with a `tanh` output.

**Graph representations** (`graph_type` in config):

| Type | Edges | Layers | Edge features | Notes |
|------|-------|--------|---------------|-------|
| `axis` (default) | Axis-window (up to `win_length - 1`) | 3 | 5-dim | Connects along the 3 win axes with stop-at-opponent traversal, plus a global dummy node |
| `hex` | Distance-1 hex adjacency | ~9 (auto: `win_length × 1.5`) | None | Each node connects to its 6 hex neighbours |

The axis graph reaches the same receptive field in 3 layers that hex adjacency needs ~9 layers for, and carries richer per-edge information.

**Node features** are 8-dim by default (stone one-hot, side-to-move, normalized centroid-relative coordinates, inverse distance to nearest stone). Two optional encodings change the width: `relative_stone_encoding` (side-to-move-relative) and `threat_features` (+4 threat dims).

**Axis edge features** (5-dim): axis one-hot (3), signed distance, and a same-colour flag.

**Search**: Gumbel AlphaZero with Sequential Halving — batched network evaluation, improved policy via `softmax(logits + σ(normalized Q))`.

## Training pipeline

1. **Self-play** — generate games with Gumbel MCTS under the current network.
2. **Training** — KL policy loss + MSE value loss on samples from a SQLite replay buffer.
3. **Evaluation** — periodic win rate vs. random, vs. a previous checkpoint, and vs. [SealBot](https://github.com/Ramora0/HexTicTacToe).
4. **Curriculum** — progress through smaller board variants up to full HeXO, with automatic convergence detection.

## Install

This package is built and installed as part of the [HeXO](../README.md) workspace, which also builds the `hexo_rs` extension it depends on. From the repository root:

```bash
uv sync --group cpu          # or: --group cuda | --group rocm
```

Requires Python **3.13+**, a Rust toolchain, and a C++ compiler. The `--group` flag selects the PyTorch backend (torch **2.11+**) for your hardware. To pull in the `pygame` viewer used by `watch`, add the `viewer` extra (e.g. `uv sync --group cpu --extra viewer`).

See the [root README](../README.md) for the full workspace build. Commands below assume you run from this `hexo-a0/` directory (or prefix with `uv run --no-sync`).

## Quick start

Everything is driven through the `hexo-a0` CLI. Generate a config template, then train:

```bash
# Print an annotated curriculum config template and save it
hexo-a0 default-curriculum > curriculum.toml

# Run the multi-stage curriculum (small board → full HeXO)
hexo-a0 curriculum --config curriculum.toml --device cuda:0

# Watch a checkpoint play a game (needs the 'viewer' extra)
hexo-a0 watch --config curriculum.toml --checkpoint runs/<name>/checkpoints/latest.pt

# Evaluate a checkpoint against SealBot
hexo-a0 eval-sealbot --config curriculum.toml --checkpoint <ckpt.pt> --games 100
```

For a single fixed-stage run instead of a curriculum, use `hexo-a0 default-config > config.toml` and `hexo-a0 train --config config.toml`.

## CLI commands

| Command | Purpose |
|---------|---------|
| `train` | Run the training loop for a single fixed configuration. |
| `curriculum` | Run automatic multi-stage curriculum training. |
| `watch` | Watch a checkpoint play a game in a pygame hex viewer. |
| `eval-sealbot` | Evaluate a checkpoint against the SealBot minimax opponent. |
| `head-to-head` | Run an SPRT-bounded match between two checkpoints (configs may differ). |
| `serve` | Run the public play server / analysis tool (see below). |
| `default-config` | Print a standalone training config template. |
| `default-curriculum` | Print an annotated curriculum config template. |

Run any command with `--help` for its full flag list.

## Configuration

Training is configured with TOML. The top-level sections are `[model]`, `[game]`, `[mcts]`, `[training]`, `[self_play]`, `[eval]`, `[run]`, and (for curricula) `[curriculum]` with `[[curriculum.stages]]` entries. `hexo-a0 default-curriculum` prints a fully annotated template — the best starting point.

```toml
[model]
hidden_dim = 256
num_layers = 3        # 0 = auto (win_length × 1.5)
num_heads = 8
graph_type = "axis"   # "axis" or "hex"
conv_type = "gine"    # "gine" or "gatv2"

[mcts]
n_simulations = 16
m_actions = 16

[training]
batch_size = 512
lr = 0.0002
lr_schedule = "cosine"
total_train_steps = 30000
edge_budget = 500000  # target edges per micro-batch (0 = fixed batch_size)
```

### Batch sizing for variable-size graphs

Axis graphs grow with board size — a small early stage has ~180 edges per graph, a large late stage several thousand. With a fixed batch size the late stages would pack far more edges per step and grind to a crawl. `edge_budget` fixes this by packing examples until a target edge count is reached, so each stage gets a similar amount of work per step. `batch_size` then acts as a ceiling. Set `edge_budget = 0` for legacy fixed-batch behaviour.

## Curriculum learning

A curriculum runs through progressively larger board variants with automatic convergence detection between stages:

```toml
[curriculum]
convergence_mode = "checkpoint"  # "random", "checkpoint", "champion", or "none"
win_rate_threshold = 0.85        # vs-random threshold
improve_threshold = 0.65         # vs previous-checkpoint threshold
patience = 10                    # consecutive plateau evals before advancing

[[curriculum.stages]]
name = "S1: 4-in-a-row, radius 2"
win_length = 4
placement_radius = 2
```

To skip to the next stage manually, drop an `ADVANCE` marker in the run's checkpoint directory:

```bash
touch runs/<name>/checkpoints/ADVANCE
```

## Evaluation

Several evaluation opponents are available:

- **vs. Random** — a uniform random opponent, the primary convergence signal during training.
- **vs. Checkpoint** — the current model against a lagged checkpoint, to detect plateaus.
- **vs. SealBot** — an adaptive difficulty ladder against the [SealBot](https://github.com/Ramora0/HexTicTacToe) C++ minimax engine, configured under `[eval.sealbot]` and runnable one-off via `eval-sealbot`.
- **vs. Human corpus** — optional replay-based evaluation against a human game corpus (`[eval.corpus]`, disabled by default).

Promotion between champions can be gated with SPRT (`convergence_mode = "champion"`); see the `sprt_*` modules and `head-to-head`.

## Play server

`hexo-a0 serve` starts an HTTP play-and-analyze server: a human can play the trained bot at selectable difficulty tiers, completed games are recorded to SQLite, and a token-gated `/admin` page exposes analysis. The web frontend (HTML/JS/CSS, fonts, and social-card assets) is served from `hexo_a0/serving/static` and `templates`. It's designed to bind `0.0.0.0` behind a reverse proxy / Tailscale Funnel with `--url-prefix`.

```bash
hexo-a0 serve --config config.toml --checkpoint <ckpt.pt> --port 8765
```

## Known limitations

- **D6 augmentation** for axis graphs is limited; check `augment_symmetries` support for your graph type.
- **KrakenBot** evaluation (`krakenbot_eval.py`) exists as a standalone proof of concept against a hosted API but is not wired into the config or CLI.
- Several subsystems (remote self-play actor, SPRT daemon, ROCm allocator tuning) assume the author's single-GPU hardware.

## Related

- [`hexo-rs`](../hexo-rs/README.md) — the Rust game engine, MCTS, and PyO3 bindings this package depends on.
- [SealBot](https://github.com/Ramora0/HexTicTacToe) — the C++ minimax evaluation opponent.

## License

MIT — see [LICENSE](LICENSE).
