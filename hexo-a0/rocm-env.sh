#!/bin/bash
# Source this inside the llama-rocm-7.2 toolbox before running hexo-a0.
# Usage: source rocm-env.sh

export LD_PRELOAD="/opt/rocm-7.2.0/lib/libhsa-runtime64.so:/opt/rocm-7.2.0/lib/libamdhip64.so"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PATH="/opt/rocm-7.2.0/bin:$PATH"
