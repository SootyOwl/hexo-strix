# HeXO workspace build recipes

# Default: sync Python deps + build everything
default: sync build

# Sync Python workspace (rebuilds hexo-rs Python extension via maturin).
# --all-extras pulls the `research` extra (matplotlib/plotly/tb-query) used by
# scripts/; the serving Docker image omits it to stay lean.
sync:
    uv sync --group rocm --all-packages --all-extras

# Build the Rust self-play binary (requires libtorch via PyTorch in venv + C++ compiler)
# If `c++` is missing but `g++-14` exists (homebrew), set CXX explicitly.
# Homebrew lib/include paths for linking (Fedora Atomic lacks -devel packages)
brew_prefix := `brew --prefix 2>/dev/null || echo /home/linuxbrew/.linuxbrew`

self-play:
    cd hexo-rs && \
      LIBTORCH_USE_PYTORCH=1 \
      CXX="${CXX:-$(command -v c++ || command -v g++-15 || command -v g++-14 || command -v g++ || echo c++)}" \
      LIBRARY_PATH="{{brew_prefix}}/lib:{{brew_prefix}}/lib/gcc/current:${LIBRARY_PATH:-}" \
      CPLUS_INCLUDE_PATH="{{brew_prefix}}/include:${CPLUS_INCLUDE_PATH:-}" \
      cargo build --release --features torch

# Build everything: Python extension + self-play binary
build: sync self-play

# Build the ARM64 serving image and push it to GHCR (for the Oracle box).
# Requires `docker login ghcr.io`. Emulated arm64 build, so first run is slow;
# the builder stage caches. Usage: just image-push [tag]   (default tag: arm64)
image-push tag="arm64":
    docker buildx build --platform linux/arm64 --push -t ghcr.io/sootyowl/hexo-serve:{{tag}} .

# Ship a checkpoint + deploy files to the Oracle box (Ubuntu). Usage:
#   just deploy runs/gine-mini/4l-128p32v-jkcat-rel2/checkpoints/self_play/champion.pt
# --chmod=Fa+r makes the transferred files world-readable on the box, so the
# non-root container (uid 10001) can read the checkpoint (they're 0600 at rest).
# Then on the box: cp .env.example .env, fill it in, `docker compose up -d`.
deploy checkpoint:
    rsync -avz --progress --chmod=Fa+r {{checkpoint}} .env.example compose.yaml ubuntu@oracle:/home/ubuntu/hexo-serve/

# Import an existing games DB into the Oracle deployment's /data volume.
# WAL-safe online snapshot (the source server can stay up) -> scp -> load into the
# hexo-data volume over SSH. REPLACES /data/games.sqlite on the box and briefly
# bounces the stack; the new server migrates the old schema in place on next open.
# Run AFTER the stack is deployed (.env present). Needs the sqlite3 CLI locally +
# SSH access to `oracle`.  Usage: just import-db path/to/games.sqlite
import-db db:
    #!/usr/bin/env bash
    set -euo pipefail
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    echo "-> snapshotting {{db}} (WAL-safe)..."
    sqlite3 "{{db}}" ".backup '$tmp/games-export.sqlite'"
    echo "-> copying to oracle..."
    scp "$tmp/games-export.sqlite" ubuntu@oracle:/home/ubuntu/hexo-serve/games-export.sqlite
    echo "-> loading into the hexo-data volume (brief restart)..."
    ssh ubuntu@oracle 'cd /home/ubuntu/hexo-serve && docker compose up -d && docker compose down && VOL=$(docker volume ls -q -f name=hexo-data | head -1) && docker run --rm --user root -v "$VOL":/data -v "$PWD/games-export.sqlite":/import.sqlite:ro alpine sh -c "cp /import.sqlite /data/games.sqlite && chown -R 10001:10001 /data && rm -f /data/games.sqlite-wal /data/games.sqlite-shm" && rm -f games-export.sqlite && docker compose up -d'
    echo "OK - old games history is now on the new server."

# Run tests: Rust + Python
test: test-rust test-python

# Empty on a fresh tree (before `just sync` installs torch); only used by the
# post-sync self-play/test recipes. `|| true` keeps `just` from aborting at
# parse time when torch isn't importable yet.
torch_lib_dir := `uv run --no-sync python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')" 2>/dev/null || true`

test-rust:
    cd hexo-rs && \
      LIBTORCH_USE_PYTORCH=1 \
      CXX="${CXX:-$(command -v c++ || command -v g++-15 || command -v g++-14 || command -v g++ || echo c++)}" \
      LIBRARY_PATH="{{brew_prefix}}/lib:{{brew_prefix}}/lib/gcc/current:${LIBRARY_PATH:-}" \
      CPLUS_INCLUDE_PATH="{{brew_prefix}}/include:${CPLUS_INCLUDE_PATH:-}" \
      LD_LIBRARY_PATH="{{torch_lib_dir}}:${LD_LIBRARY_PATH:-}" \
      cargo test -p hexo-mcts --features torch

test-python:
    uv run --no-sync pytest hexo-a0/tests/ --tb=short

# Run tests including GPU tests
test-all:
    uv run --no-sync pytest hexo-a0/tests/ --tb=short -o "addopts="

# Verify the engine + MCTS libs compile for the browser/WebWorker (wasm32) target.
# Self-contained CARGO_TARGET_DIR: just runs each recipe in a fresh shell, so the
# separate build dir is set here (never the shared hexo-rs/target/ the live trainer uses).
check-wasm:
    cd hexo-rs && CARGO_TARGET_DIR=$HOME/.cache/hexo-wasm-target cargo check --lib -p hexo-engine -p hexo-mcts --target wasm32-unknown-unknown

# Run the axis curriculum
train-axis:
    uv run --no-sync hexo-a0 curriculum

# Run the hex curriculum
train-hex:
    uv run --no-sync hexo-a0 curriculum

# Rsync to a remote machine (e.g. `just push rigel`)
push host:
    rsync -avz --exclude-from=.rsync-exclude . {{host}}:Development/personal/hexo/

copy-self-play host:
    scp hexo-rs/target/release/self_play rigel:/home/tyto/hexo/self_play

# Benchmark Rust native self-play (hex vs axis)
bench-self-play *ARGS:
    uv run --no-sync python scripts/bench_self_play.py {{ARGS}}
