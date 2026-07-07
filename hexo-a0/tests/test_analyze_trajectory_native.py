"""Native (torch-free) parity for the trajectory analysis entry points.

`analyze_trajectory` (value-only over prefixes) and `analyze_game_full`
(Pass-1 value+policy over prefixes, then per-prefix MCTS) must run on a
`NativeModel` without importing torch, producing the same output shapes as the
torch path.
"""
import json
import os
import subprocess
import sys

import pytest

REAL = os.environ.get("HEXO_REAL_WEIGHTS")


def _load():
    from hexo_a0.serving.model import load_native_model
    model, mc, meta = load_native_model(REAL)
    gc = json.loads(meta["game_config"]) if isinstance(meta["game_config"], str) \
        else meta["game_config"]
    return model, mc, (gc["win_length"], gc["placement_radius"], gc["max_moves"])


MOVES = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]


@pytest.mark.skipif(not REAL, reason="HEXO_REAL_WEIGHTS unset")
def test_analyze_trajectory_native_shape():
    from hexo_a0.serving.analysis import analyze_trajectory
    model, mc, (wl, pr, mm) = _load()
    out = analyze_trajectory(model, mc, MOVES, wl, pr, mm)
    assert out["evaluated_prefixes"] == len(MOVES)
    assert len(out["trajectory"]) == len(MOVES)
    for entry in out["trajectory"]:
        v = entry["value"]
        assert isinstance(v, float) and -1.0 <= v <= 1.0
        if entry["terminal"]:
            assert entry["value"] in (-1.0, 0.0, 1.0)


@pytest.mark.skipif(not REAL, reason="HEXO_REAL_WEIGHTS unset")
def test_analyze_game_full_native_shape():
    from hexo_a0.serving.analysis import analyze_game_full
    model, mc, (wl, pr, mm) = _load()
    out = analyze_game_full(model, mc, MOVES, wl, pr, mm, mcts_sims=8)
    nonterm = [t for t in out["trajectory"] if not t["terminal"]]
    assert nonterm, "expected some non-terminal prefixes"
    for t in nonterm:
        assert "probs" in t and "legal" in t
        assert len(t["probs"]) == len(t["legal"])
        if t["probs"]:
            assert abs(sum(t["probs"]) - 1.0) < 1e-5


@pytest.mark.skipif(not REAL, reason="HEXO_REAL_WEIGHTS unset")
def test_trajectory_native_is_torch_free():
    """Both trajectory functions run under a hard torch import block."""
    code = f'''
import sys, json
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch.") \\
                or name == "torch_geometric" or name.startswith("torch_geometric."):
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, _Block())
from hexo_a0.serving.model import load_native_model
from hexo_a0.serving.analysis import analyze_trajectory, analyze_game_full
model, mc, meta = load_native_model({REAL!r})
gc = json.loads(meta["game_config"]) if isinstance(meta["game_config"], str) else meta["game_config"]
wl, pr, mm = gc["win_length"], gc["placement_radius"], gc["max_moves"]
moves = {MOVES!r}
analyze_trajectory(model, mc, moves, wl, pr, mm)
analyze_game_full(model, mc, moves, wl, pr, mm, mcts_sims=8)
assert "torch" not in sys.modules, "torch got imported on the native path"
print("TRAJ_TORCH_FREE_OK")
'''
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "TRAJ_TORCH_FREE_OK" in r.stdout
