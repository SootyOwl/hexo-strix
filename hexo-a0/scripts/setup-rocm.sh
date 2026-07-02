#!/bin/bash
# Run inside the llama-rocm-7.2 toolbox to switch the venv to ROCm PyTorch.
# Usage: source scripts/setup-rocm.sh

set -e

echo "Installing ROCm PyTorch..."
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.1 --reinstall-package torch
uv pip install triton-rocm --index-url https://download.pytorch.org/whl/rocm7.1

echo "Rebuilding hexo-rs PyO3 bindings..."
(cd ../hexo-rs && VIRTUAL_ENV="$(cd ../hexo-a0 && pwd)/.venv" maturin develop)

echo "Done. ROCm PyTorch ready."
python -c "import torch; print(f'torch {torch.__version__}, HIP: {torch.version.hip}')"
