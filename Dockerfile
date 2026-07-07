# syntax=docker/dockerfile:1.7
#
# HeXO serving image — the `hexo-a0 serve` public play/analysis server.
#
# Serving-only and TORCH-FREE: it needs just the PyO3 module (hexo_rs, built
# WITHOUT the `torch` Rust feature) — serving runs the pure-Rust
# hexo_rs.InferModel over a safetensors checkpoint, so NO Python torch, no
# torch-geometric, no libtorch, and none of the ROCm/LD_PRELOAD crash surface.
# Dropping the ~742 MB CPU torch wheel is the point. The checkpoint must be a
# `.safetensors` (torch-free code can't torch.load a `.pt` pickle); produce one
# from a champion with `hexo-a0 export --checkpoint champion.pt --out champ.safetensors`.
#
# (A torch serving fallback still exists for dev via `uv run hexo-a0 serve` on a
# .pt, but it is deliberately NOT containerized. To build a torch image anyway:
# `--build-arg TORCH_GROUP=cpu` — pulls the torch stack via the `train` extra.)
#
# Build ON the target host (native) — recommended for the arm64 box:
#   docker build -t hexo-serve .
# Or cross-build from x86 (slower; QEMU emulates the Rust build):
#   docker buildx build --platform linux/arm64 -t hexo-serve .
#
# Run (mount the safetensors checkpoint + a data volume for the SQLite files):
#   docker run -d --name hexo -p 8080:8080 \
#     -v /path/to/champ.safetensors:/models/champion.safetensors:ro,Z \
#     -v hexo-data:/data \
#     -e HEXO_ADMIN_TOKEN=changeme \
#     hexo-serve
# NOTE: the checkpoint must be readable by the container's uid (chmod a+r), and on
# SELinux hosts (Fedora/RHEL/Oracle Linux) the mount needs the ',Z' relabel shown
# above — otherwise the server aborts with "Permission denied" on the checkpoint.

########################################
# Stage 1 — builder
########################################
FROM python:3.13-slim-bookworm AS builder

# uv (Astral) — copied from the official multi-arch image (amd64 + arm64).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Rust toolchain + a C linker for the maturin/PyO3 build, in one layer. No libtorch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
         | sh -s -- -y --profile minimal --default-toolchain stable \
    && rm -rf /var/lib/apt/lists/*

# CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256: serving inference is not
# throughput-critical (unlike self-play), so favour more parallel codegen over
# the last sliver of runtime speed — noticeably faster compile on a many-core box.
ENV PATH="/root/.cargo/bin:${PATH}" \
    CARGO_HOME=/root/.cargo \
    CARGO_TARGET_DIR=/root/cargo-target \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Torch backend. Default `""` = TORCH-FREE: serving runs the pure-Rust
# hexo_rs.InferModel, so no torch is installed and the image is ~742 MB leaner.
# To build a torch serving image anyway, pass `--build-arg TORCH_GROUP=cpu`
# (or rocm/cuda) — that selects the backend group AND pulls hexo-a0's `train`
# extra (torch + torch-geometric) so the torch eval/analysis path is complete.
ARG TORCH_GROUP=""

# --- Phase 1: third-party dependencies only ---------------------------------
# Copy just the lockfile + every workspace member's pyproject (small, rarely
# change). --no-install-workspace builds NONE of our code, so this heavy layer
# is cached and only re-runs when the lock or a pyproject changes, NOT when
# Python/JS/Rust source does. With TORCH_GROUP="" this installs no torch at all;
# a non-empty group adds `--group $TORCH_GROUP --all-extras` (the torch stack
# lives in the ROOT backend group + hexo-a0's `train` extra).
#   --no-dev → no pytest
COPY pyproject.toml uv.lock ./
COPY hexo-a0/pyproject.toml hexo-a0/
COPY hexo-rs/pyproject.toml hexo-rs/
RUN --mount=type=cache,target=/root/.cache/uv \
    GRP=""; [ -n "$TORCH_GROUP" ] && GRP="--group $TORCH_GROUP --all-extras"; \
    uv sync --frozen --no-dev --no-install-workspace $GRP

# --- Phase 2: build hexo_rs (Rust) — the slow step, busted only by *.rs -------
# Split from the Python install (Phase 3) so editing hexo-a0 (app.py, app.js, …)
# does NOT re-trigger the maturin/Rust compile — only hexo-rs source invalidates
# this layer. --no-install-package hexo-a0 builds + installs hexo-rs (+ its deps)
# but skips the pure-Python member. The cargo cache mounts keep the compile
# incremental across builds (a no-*.rs change just relinks).
#
# RUSTFLAGS tunes for the deploy CPU (Oracle Ampere A1 = Neoverse N1 on arm64;
# x86-64-v3 = AVX2+FMA on any post-2015 x86). rustc does no FP contraction, so
# results are unchanged. (config.toml rustflags are wasm-target-only; no wasm here.)
ARG TARGETARCH
COPY hexo-rs/ hexo-rs/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/root/cargo-target \
    if [ "$TARGETARCH" = "arm64" ]; then \
        export RUSTFLAGS="-C target-cpu=neoverse-n1"; \
    else \
        export RUSTFLAGS="-C target-cpu=x86-64-v3"; \
    fi; \
    GRP=""; [ -n "$TORCH_GROUP" ] && GRP="--group $TORCH_GROUP --all-extras"; \
    uv sync --frozen --no-dev --no-editable --no-install-package hexo-a0 $GRP

# --- Phase 3: install hexo-a0 (pure Python) — fast, changes often -------------
# hexo_rs is already built in Phase 2; this only bakes the Python package (server,
# frontend assets) into .venv, so the common edit-serve loop skips Rust entirely.
COPY hexo-a0/ hexo-a0/
RUN --mount=type=cache,target=/root/.cache/uv \
    GRP=""; [ -n "$TORCH_GROUP" ] && GRP="--group $TORCH_GROUP --all-extras"; \
    uv sync --frozen --no-dev --no-editable $GRP

# Fail the build early if the static assets weren't packaged into the venv.
RUN test -n "$(find /app/.venv -type d -name static -path '*serving*' -print -quit)" \
    || (echo 'ERROR: serving/static not found in .venv — static assets missing' >&2; exit 1)

########################################
# Stage 2 — runtime
########################################
FROM python:3.13-slim-bookworm AS runtime

# tini: correct signal handling / zombie reaping for the long-lived server.
# No libgomp1: the torch-free default only links libgcc_s (in the base image);
# hexo_rs pulls in no OpenMP. A `--build-arg TORCH_GROUP=...` torch build should
# re-add `libgomp1` here if its torch wheel needs system OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user owning the writable mounts.
RUN useradd --create-home --uid 10001 hexo \
    && mkdir -p /data /models \
    && chown -R hexo:hexo /data /models

# Self-contained venv (its bin/python targets this image's /usr/local/bin/python3.13).
COPY --from=builder --chown=hexo:hexo /app/.venv /app/.venv
# Model-architecture fallback config (small); the checkpoint itself is mounted.
COPY --chown=hexo:hexo configs/ /app/configs/
COPY --chown=hexo:hexo docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    # ---- serving config (override with `docker run -e NAME=value`) ----
    HEXO_CONFIG=/app/configs/curriculum.toml \
    HEXO_CHECKPOINT=/models/champion.safetensors \
    HEXO_DB=/data/games.sqlite \
    HEXO_PORT=8080 \
    HEXO_BIND=0.0.0.0 \
    HEXO_MCTS_SIMS=64 \
    HEXO_MODEL_LABEL=hexo \
    HEXO_URL_PREFIX="" \
    HEXO_ADMIN_TOKEN="" \
    HEXO_DIFFICULTY_SIMS="" \
    HEXO_DEFAULT_DIFFICULTY=""

USER hexo
WORKDIR /app
VOLUME ["/data", "/models"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os,urllib.request; p=os.environ.get('HEXO_URL_PREFIX',''); urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['HEXO_PORT']}{p}/\", timeout=4).read()" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
