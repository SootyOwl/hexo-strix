"""analyze_position's native (torch-free) branch, exercised against real weights.

Skipped unless HEXO_REAL_WEIGHTS points at a .safetensors checkpoint:

    HEXO_REAL_WEIGHTS=/path/to/strixbot-rel2.safetensors \\
        uv run --no-sync pytest hexo-a0/tests/test_analyze_position_native.py --tb=short
"""
import json
import math
import os
import sys
import subprocess

import pytest

HEXO_REAL_WEIGHTS = os.environ.get("HEXO_REAL_WEIGHTS")
pytestmark = pytest.mark.skipif(
    not HEXO_REAL_WEIGHTS, reason="HEXO_REAL_WEIGHTS unset"
)


def _build_state():
    """A small, legal, non-terminal mid-game state matching the rel2 weights'
    embedded game config (win_length=6, placement_radius=6, max_moves=300)."""
    import hexo_rs as hr
    from hexo_a0.serving.model import load_native_model

    model, mc, meta = load_native_model(HEXO_REAL_WEIGHTS)
    gc = meta.get("game_config")
    gc = json.loads(gc) if isinstance(gc, str) else gc
    cfg = hr.GameConfig(gc["win_length"], gc["placement_radius"], gc["max_moves"])
    state = hr.GameState(cfg)  # auto-seeds P1 at (0, 0)
    state.apply_move(1, 0)
    state.apply_move(0, 1)
    assert not state.is_terminal()
    return model, mc, state


def test_analyze_position_native_raw():
    from hexo_a0.serving.analysis import analyze_position

    model, mc, state = _build_state()
    res = analyze_position(model, mc, state, mcts_sims=0)

    assert res.terminal is False
    assert isinstance(res.value, float) and math.isfinite(res.value)
    assert -1.0 <= res.value <= 1.0
    assert len(res.probs) == len(res.legal)
    assert abs(sum(res.probs) - 1.0) < 1e-6
    assert res.legal == [list(c) for c in state.legal_moves()]


def test_analyze_position_native_mcts():
    from hexo_a0.serving.analysis import analyze_position

    model, mc, state = _build_state()
    res = analyze_position(model, mc, state, mcts_sims=32)

    assert res.q_hat is not None
    assert res.candidate_set is not None
    assert len(res.q_hat) == len(res.legal)
    assert isinstance(res.value, float) and math.isfinite(res.value)


def test_analyze_position_native_no_torch():
    """Prove the native raw path never touches torch at runtime: block torch
    (and torch_geometric) imports via a sys.meta_path finder installed BEFORE
    hexo_a0.serving.analysis is imported, then run analyze_position(mcts_sims=0)
    in a subprocess and assert it completes with torch never having loaded."""
    code = (
        "import sys, json, os\n"
        "class _BlockTorch:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'torch' or name.startswith('torch.') or \\\n"
        "           name == 'torch_geometric' or name.startswith('torch_geometric.'):\n"
        "            raise ImportError(f'blocked: {name}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockTorch())\n"
        "\n"
        "import hexo_rs as hr\n"
        "from hexo_a0.serving.model import load_native_model\n"
        "from hexo_a0.serving.analysis import analyze_position\n"
        "\n"
        f"path = {HEXO_REAL_WEIGHTS!r}\n"
        "model, mc, meta = load_native_model(path)\n"
        "gc = meta.get('game_config')\n"
        "gc = json.loads(gc) if isinstance(gc, str) else gc\n"
        "cfg = hr.GameConfig(gc['win_length'], gc['placement_radius'], gc['max_moves'])\n"
        "state = hr.GameState(cfg)\n"
        "state.apply_move(1, 0)\n"
        "state.apply_move(0, 1)\n"
        "res = analyze_position(model, mc, state, mcts_sims=0)\n"
        "assert res.terminal is False\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'torch_geometric' not in sys.modules\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
