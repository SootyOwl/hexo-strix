"""Native hexo-infer serving backend: attach + hot-path equivalence tests.

Builds a tiny GINE checkpoint on the fly, so no real weights are needed and
nothing here depends on gitignored artifacts.
"""
import dataclasses
import json

import pytest
import torch

from hexo_a0.gen_parity_fixtures import tiny_model_and_config
from hexo_a0.serving.model import load_model
from hexo_a0.serving.analysis import _run_mcts

GAME_CONFIG = {"win_length": 6, "placement_radius": 4, "max_moves": 300}


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    model, mc = tiny_model_and_config()
    path = tmp_path_factory.mktemp("ckpt") / "tiny_champion.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": dataclasses.asdict(mc),
        "game_config": GAME_CONFIG,
        "train_steps": 123,
    }, path)
    return str(path)


def _fresh_load(tiny_checkpoint):
    # load_model caches on (path, device); the fixture path is unique per run.
    return load_model(tiny_checkpoint, "cpu")


def test_native_attach_kill_switch(tiny_checkpoint, tmp_path, monkeypatch):
    # HEXO_NATIVE_INFER=0 must skip the native attach so serving runs the
    # torch eval path — the A/B lever for deployments where the pure-Rust
    # kernel measures slower than torch (2026-07-06: it does, on x86 dev box).
    import shutil
    fresh = tmp_path / "tiny_kill_switch.pt"  # load_model caches by path
    shutil.copy(tiny_checkpoint, fresh)
    monkeypatch.setenv("HEXO_NATIVE_INFER", "0")
    model, _, _ = load_model(str(fresh), "cpu")
    assert model._hexo_native is None


def test_native_engine_attaches(tiny_checkpoint):
    model, mc, ckpt = _fresh_load(tiny_checkpoint)
    assert getattr(model, "_hexo_native", None) is not None
    meta = json.loads(model._hexo_native.metadata_json())
    assert meta["train_steps"] == "123"
    assert json.loads(meta["game_config"]) == GAME_CONFIG


def test_native_run_mcts_matches_torch(tiny_checkpoint):
    import hexo_rs as hr
    model, mc, ckpt = _fresh_load(tiny_checkpoint)
    assert model._hexo_native is not None
    state = hr.GameState(hr.GameConfig(**GAME_CONFIG))

    native_out = _run_mcts(model, mc, state, 32, 8)
    # Detach the engine to force the torch path on the same model.
    engine = model._hexo_native
    model._hexo_native = None
    try:
        torch_out = _run_mcts(model, mc, state, 32, 8)
    finally:
        model._hexo_native = engine

    n_pol, n_val, n_legal, n_visits, n_q, n_cand = native_out
    t_pol, t_val, t_legal, t_visits, t_q, t_cand = torch_out
    assert n_legal == t_legal
    assert len(n_pol) == len(t_pol) == len(n_legal)
    assert len(n_visits) == len(t_visits) == len(n_legal)
    # Same weights, deterministic search (gumbel noise off). fp32 reordering
    # between backends can flip near-tied candidate selections on this tiny
    # untrained model, so visit counts are compared by budget, not per-move.
    assert n_val == pytest.approx(t_val, abs=1e-4)
    assert n_pol == pytest.approx(t_pol, abs=1e-3)
    assert sum(n_visits) == sum(t_visits)


def test_native_forward_matches_torch_eval(tiny_checkpoint):
    """Raw forward parity through the serving objects (not just fixtures)."""
    import hexo_rs as hr
    from hexo_a0.serving.model import make_graph_fn

    model, mc, ckpt = _fresh_load(tiny_checkpoint)
    assert model._hexo_native is not None
    state = hr.GameState(hr.GameConfig(**GAME_CONFIG))

    logit_map, native_value = model._hexo_native.forward(state)
    data = make_graph_fn(mc)(state)
    with torch.no_grad():
        logits, value = model(
            data.x, data.edge_index, data.legal_mask,
            stone_mask=data.stone_mask,
            edge_attr=getattr(data, "edge_attr", None),
        )
    legal = [tuple(c) for c in data.coords[data.legal_mask].tolist()]
    torch_logits = logits[data.legal_mask] if logits.shape[0] == data.x.shape[0] else logits
    if torch_logits.dim() > 1:
        torch_logits = torch_logits.squeeze(-1)
    assert set(logit_map) == set(legal)
    for coord, t_logit in zip(legal, torch_logits.tolist()):
        assert logit_map[coord] == pytest.approx(t_logit, abs=1e-4), coord
    assert native_value == pytest.approx(float(value.item()), abs=1e-4)


def test_unsupported_arch_falls_back(tmp_path):
    """A non-GINE checkpoint must load fine with _hexo_native left as None."""
    from hexo_a0.config import ModelConfig
    from hexo_a0.model import HeXONet

    torch.manual_seed(2)
    mc = ModelConfig(
        hidden_dim=16, num_layers=2, num_heads=1, conv_type="gatv2",
        pre_norm=True, dropout=0.0, use_jk=False,
        policy_hidden=16, value_hidden=8, graph_type="axis",
    )
    model = HeXONet(mc).eval()
    path = tmp_path / "gatv2.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": dataclasses.asdict(mc),
        "train_steps": 1,
    }, path)
    loaded, _, _ = load_model(str(path), "cpu")
    assert loaded._hexo_native is None
