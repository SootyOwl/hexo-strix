"""Model loading + graph factory for the serving module."""
import os

import pytest

import hexo_rs
from hexo_a0.serving.model import load_model, make_graph_fn

CKPT = "runs/gine-mini-4l-128p32v-jkcat/checkpoints/self_play/champion.pt"
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="champion ckpt absent")


def test_load_model_returns_triple_and_caches():
    model, mc, ckpt = load_model(CKPT, "cpu")
    assert hasattr(model, "forward_batch") and hasattr(mc, "graph_type")
    assert isinstance(ckpt, dict)
    model2, _, _ = load_model(CKPT, "cpu")
    assert model2 is model  # cache returns the same instance


def test_load_model_accepts_dict_model_config_dict():
    # Regression: load_model used to be @lru_cache, which hashed every arg;
    # passing a dict model_config_dict (the normal [model]-fallback case)
    # raised TypeError: unhashable type: 'dict' and crashed `serve` at startup.
    model, mc, ckpt = load_model(CKPT, "cpu", model_config_dict={"graph_type": "axis"})
    assert hasattr(model, "forward_batch")
    assert mc.graph_type == "axis"  # embedded config wins over the fallback


def test_graph_fn_builds_graph():
    _, mc, _ = load_model(CKPT, "cpu")
    st = hexo_rs.GameState(hexo_rs.GameConfig(6, 8, 400))
    st.apply_move(1, 0)  # NOT (0,0): origin is pre-seeded
    data = make_graph_fn(mc)(st)
    assert data.x.shape[0] > 0 and data.edge_index.shape[0] == 2  # PyG [2, E]
