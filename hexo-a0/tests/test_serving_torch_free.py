"""End-to-end parity gate: the whole serve inference path runs torch-free.

A torch-PRESENT fixture bakes a tiny GINE checkpoint to a `.safetensors` on
disk (no real weights needed). Then, in a subprocess with torch/torch_geometric
hard-blocked via a `sys.meta_path` finder, the native serve path is driven end
to end: load the safetensors, run a full `make_bot_turn_fn` bot turn, and run
`analyze_position` + `analyze_trajectory`. If any of those imports torch, the
block raises and the subprocess fails — this is the parity gate for the whole
torch-free-serving refactor.
"""
import dataclasses
import subprocess
import sys

import pytest

# Torch is present in THIS process (fixture only); the subprocess blocks it.
import torch

from hexo_a0.gen_parity_fixtures import tiny_model_and_config

GAME_CONFIG = {"win_length": 6, "placement_radius": 4, "max_moves": 300}


@pytest.fixture(scope="module")
def tiny_safetensors(tmp_path_factory):
    """Bake a tiny GINE model to a safetensors the native engine can load."""
    from hexo_a0.export import serialize_safetensors

    model, mc = tiny_model_and_config()
    mc_dict = {**dataclasses.asdict(mc), "game_config": GAME_CONFIG}
    blob, _ = serialize_safetensors(model.state_dict(), mc_dict, 123, "tiny.pt")
    path = tmp_path_factory.mktemp("st") / "tiny.safetensors"
    path.write_bytes(blob)
    return str(path)


# The subprocess program. {path} is filled in per-run. It installs a hard torch
# import block, then drives the native serve path end to end.
_CHILD = r'''
import sys, json

class _Block:
    def find_spec(self, name, path=None, target=None):
        if (name == "torch" or name.startswith("torch.")
                or name == "torch_geometric" or name.startswith("torch_geometric.")):
            raise ImportError("blocked: " + name)
        return None

sys.meta_path.insert(0, _Block())

import hexo_rs
from hexo_a0.serving.model import load_native_model
from hexo_a0.serving.game import GameRecord, make_bot_turn_fn
from hexo_a0.serving.inference import InferenceGuard
from hexo_a0.serving.analysis import analyze_position, analyze_trajectory
from datetime import datetime, timezone

PATH = {path!r}
GC = {gc!r}

model, mc, meta = load_native_model(PATH)
assert model._hexo_native is not None

guard = InferenceGuard(1)
game_kwargs = dict(**GC)
bot_fn = make_bot_turn_fn(
    model=model, model_config=mc, mcts_sims=16, m_actions=8,
    guard=guard, game_kwargs=game_kwargs,
)

# Fresh game: P1 seed at (0,0) pre-placed, P2 to move -> bot_side=P2 plays now.
state = hexo_rs.GameState(hexo_rs.GameConfig(**GC))
assert state.current_player() == "P2"
now = datetime.now(timezone.utc)
rec = GameRecord(
    game_id="t", created_at=now, last_active_at=now, state=state,
    human_side="P1", bot_side="P2", human_name=None,
)
before = len(rec.state.placed_stones())
with rec.lock:
    bot_fn(rec)
after = len(rec.state.placed_stones())
assert after > before, "bot turn placed no stones"

# Analysis path (raw forward + softmax + threat scan, then a short trajectory).
res = analyze_position(model, mc, rec.state, mcts_sims=0)
assert res.terminal is False
assert len(res.probs) == len(res.legal)
assert abs(sum(res.probs) - 1.0) < 1e-6

moves = [(0, 0), (1, 0), (0, 1), (2, 0)]
traj = analyze_trajectory(model, mc, moves, GC["win_length"],
                          GC["placement_radius"], GC["max_moves"])
assert traj["evaluated_prefixes"] == len(moves)

assert "torch" not in sys.modules, "torch was imported on the native serve path"
print("SERVE_TORCH_FREE_OK")
'''


def test_serve_path_is_torch_free(tiny_safetensors):
    code = _CHILD.format(path=tiny_safetensors, gc=GAME_CONFIG)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "SERVE_TORCH_FREE_OK" in r.stdout, r.stdout
