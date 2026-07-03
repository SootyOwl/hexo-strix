"""Tests for safetensors checkpoint export."""

import json

import pytest
import torch

from hexo_a0.config import ModelConfig
from hexo_a0.export import export_checkpoint, save_safetensors
from hexo_a0.model import HeXONet


def tiny_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=16, num_layers=2, num_heads=1, conv_type="gine",
        pre_norm=True, dropout=0.0, use_jk=True, jk_mode="cat",
        policy_hidden=16, value_hidden=8, graph_type="axis",
        prune_empty_edges=True, threat_features=True,
        relative_stone_encoding=True,
    )


@pytest.fixture()
def tiny_checkpoint(tmp_path):
    torch.manual_seed(0)
    cfg = tiny_config()
    model = HeXONet(cfg)
    import dataclasses
    ckpt = {
        "model_state_dict": {"_orig_mod." + k: v for k, v in model.state_dict().items()},
        "model_config": dataclasses.asdict(cfg),
        "train_steps": 123,
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)
    return path, model, cfg


def test_save_safetensors_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = tiny_config()
    model = HeXONet(cfg)
    import dataclasses
    out = tmp_path / "weights.safetensors"
    meta = save_safetensors(model.state_dict(), dataclasses.asdict(cfg), "0", "tiny", out)

    from safetensors import safe_open
    with safe_open(str(out), framework="pt") as f:
        file_meta = f.metadata()
        keys = set(f.keys())
        sd = model.state_dict()
        assert keys == set(sd.keys())  # _orig_mod. stripped
        for k in keys:
            t = f.get_tensor(k)
            assert t.dtype == torch.float32
            assert torch.equal(t, sd[k].float())
    assert file_meta["format"] == "hexo-safetensors-v1"
    assert json.loads(file_meta["model_config"])["hidden_dim"] == 16
    assert file_meta["train_steps"] == "0"
    assert file_meta["source_checkpoint"] == "tiny"
    assert meta["train_steps"] == "0"


def test_export_roundtrip(tiny_checkpoint, tmp_path):
    ckpt_path, model, cfg = tiny_checkpoint
    out = tmp_path / "weights.safetensors"
    meta = export_checkpoint(ckpt_path, out)

    from safetensors import safe_open
    with safe_open(str(out), framework="pt") as f:
        keys = set(f.keys())
        sd = model.state_dict()
        assert keys == set(sd.keys())  # _orig_mod. stripped
        for k in keys:
            assert torch.equal(f.get_tensor(k), sd[k].float())
    assert meta["format"] == "hexo-safetensors-v1"
    assert json.loads(meta["model_config"])["hidden_dim"] == 16
    assert meta["train_steps"] == "123"
    assert meta["source_checkpoint"] == "ckpt.pt"


def test_export_rejects_no_model_config(tmp_path):
    import dataclasses
    torch.manual_seed(0)
    model = HeXONet(tiny_config())
    ckpt = {"model_state_dict": model.state_dict()}  # no model_config
    path = tmp_path / "bare.pt"
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="model_config"):
        export_checkpoint(path, tmp_path / "out.safetensors")


def test_export_embeds_game_config(tmp_path):
    import dataclasses
    torch.manual_seed(0)
    cfg = tiny_config()
    model = HeXONet(cfg)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": dataclasses.asdict(cfg),
        "game_config": {"win_length": 6, "placement_radius": 6, "max_moves": 300},
        "train_steps": 5,
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)
    out = tmp_path / "out.safetensors"
    export_checkpoint(path, out)
    from safetensors import safe_open
    with safe_open(str(out), framework="pt") as f:
        gc = json.loads(f.metadata()["game_config"])
    assert gc["placement_radius"] == 6  # stage-specific config travels with the weights