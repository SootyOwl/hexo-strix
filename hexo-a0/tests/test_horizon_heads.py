"""Model, trainer-loss, config, and export tests for short-horizon value heads."""

import pytest

torch = pytest.importorskip("torch")
from torch_geometric.data import Batch, Data

from hexo_a0.config import ModelConfig, TrainingConfig, model_config_from_checkpoint
from hexo_a0.model import HeXONet


def _cfg(value_bins=65, horizons=None, **kw):
    base = dict(
        hidden_dim=32, num_layers=2, num_heads=2, policy_hidden=16,
        value_hidden=16, graph_type="hex", conv_type="gatv2",
    )
    base.update(kw)
    base["value_bins"] = value_bins
    base["value_horizons"] = horizons or []
    return ModelConfig(**base)


def _data(num_nodes=10, num_legal=4, nf=8):
    x = torch.randn(num_nodes, nf)
    src, dst = [], []
    for i in range(num_nodes - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    ei = torch.tensor([src, dst], dtype=torch.int64)
    lm = torch.zeros(num_nodes, dtype=torch.bool)
    lm[:num_legal] = True
    return Data(x=x, edge_index=ei, legal_mask=lm)


# --- Config validation ------------------------------------------------------

class TestConfigValidation:
    def test_horizons_require_value_bins(self):
        from hexo_a0.config_io import _validate_config
        from hexo_a0.config import FullConfig
        cfg = FullConfig()
        cfg.model = _cfg(value_bins=0, horizons=[4, 12])
        with pytest.raises(ValueError, match="value_horizons requires value_bins"):
            _validate_config(cfg)

    def test_horizon_weight_requires_horizons(self):
        from hexo_a0.config_io import _validate_config
        from hexo_a0.config import FullConfig
        cfg = FullConfig()
        cfg.model = _cfg(value_bins=65, horizons=[])
        cfg.training = TrainingConfig(horizon_loss_weight=0.25)
        with pytest.raises(ValueError, match="horizon_loss_weight>0 requires"):
            _validate_config(cfg)

    def test_negative_horizon_rejected(self):
        from hexo_a0.config_io import _validate_config
        from hexo_a0.config import FullConfig
        cfg = FullConfig()
        cfg.model = _cfg(value_bins=65, horizons=[4, 0])
        with pytest.raises(ValueError, match="positive integers"):
            _validate_config(cfg)

    def test_config_round_trip(self):
        import dataclasses
        cfg = _cfg(value_bins=65, horizons=[4, 12, 32])
        ckpt = {"model_config": dataclasses.asdict(cfg)}
        rebuilt = model_config_from_checkpoint(ckpt)
        assert rebuilt.value_horizons == [4, 12, 32]


# --- Model ------------------------------------------------------------------

class TestHorizonHeadsModel:
    def test_builds_one_head_per_horizon(self):
        net = HeXONet(_cfg(horizons=[4, 12, 32]))
        assert len(net.horizon_value_heads) == 3

    def test_no_heads_without_horizons(self):
        net = HeXONet(_cfg(horizons=[]))
        assert not hasattr(net, "horizon_value_heads")

    def test_horizons_require_value_bins_at_construction(self):
        with pytest.raises(ValueError, match="value_horizons requires value_bins"):
            HeXONet(_cfg(value_bins=0, horizons=[4]))

    def test_extras_has_horizon_logits(self):
        net = HeXONet(_cfg(horizons=[4, 12, 32]))
        batch = Batch.from_data_list([_data(), _data(8, 3)])
        out = net._forward_batch_core(batch, return_train_extras=True)
        assert len(out) == 4
        extras = out[3]
        assert "value_logits" in extras
        assert extras["horizon_logits"].shape == (2, 3, 65)  # (G, n_horizons, bins)

    def test_default_path_unaffected_three_tuple(self):
        net = HeXONet(_cfg(horizons=[4, 12]))
        batch = Batch.from_data_list([_data()])
        out = net._forward_batch_core(batch)  # no extras
        assert len(out) == 3
        assert (out[2] >= -1).all() and (out[2] <= 1).all()

    def test_extras_omits_horizon_when_disabled(self):
        net = HeXONet(_cfg(horizons=[]))
        batch = Batch.from_data_list([_data()])
        _, _, _, extras = net._forward_batch_core(batch, return_train_extras=True)
        assert "horizon_logits" not in extras
        assert "value_logits" in extras


# --- Trainer loss -----------------------------------------------------------

def _example7(num_nodes=10, num_legal=4, value=0.5, horizons_row=None):
    d = _data(num_nodes, num_legal)
    policy = torch.full((num_legal,), 1.0 / num_legal)
    # (data, policy, value, traj_id, age, sample_weight, horizon_targets)
    return (d, policy, torch.tensor(float(value)), None, None, 1.0, horizons_row)


class TestTrainerHorizonLoss:
    def test_horizon_loss_trains_heads(self):
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg(horizons=[4, 12]))
        examples = [
            _example7(value=1.0, horizons_row=[1.0, 1.0]),
            _example7(8, 3, value=-1.0, horizons_row=[0.0, -1.0]),
        ]
        out = _forward_and_loss(
            net, examples, torch.device("cpu"), horizon_loss_weight=0.25
        )
        assert torch.is_tensor(out["horizon_loss"])
        assert out["horizon_loss"].item() >= 0.0
        # horizon head params must have received gradient
        assert any(p.grad is not None for p in net.horizon_value_heads.parameters())

    def test_horizon_loss_zero_when_weight_off(self):
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg(horizons=[4, 12]))
        examples = [_example7(value=1.0, horizons_row=[1.0, 1.0])]
        out = _forward_and_loss(net, examples, torch.device("cpu"), horizon_loss_weight=0.0)
        assert out["horizon_loss"] == 0.0  # float sentinel, not optimized

    def test_missing_horizon_targets_falls_back_neutral(self):
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg(horizons=[4, 12]))
        # One example has None targets (e.g. legacy buffer entry) — must not crash.
        examples = [
            _example7(value=1.0, horizons_row=[1.0, 1.0]),
            _example7(8, 3, value=-1.0, horizons_row=None),
        ]
        out = _forward_and_loss(net, examples, torch.device("cpu"), horizon_loss_weight=0.25)
        assert torch.is_tensor(out["horizon_loss"])


# --- Export unaffected ------------------------------------------------------

class TestExportDropsHorizonHeads:
    def test_export_ignores_horizon_heads(self, tmp_path):
        from hexo_a0.scriptable_model import export_torchscript
        net = HeXONet(_cfg(horizons=[4, 12]))
        sd = {k: v for k, v in net.state_dict().items()}
        out = tmp_path / "m.pt"
        # Must not raise: horizon head params are simply not mapped into the
        # scriptable model (train-only), and value head still exports (binned).
        export_torchscript(sd, _cfg(horizons=[4, 12]), str(out))
        assert out.exists()
        scripted = torch.jit.load(str(out))
        # scripted model has no horizon heads
        names = [n for n, _ in scripted.named_parameters()]
        assert not any("horizon" in n for n in names)
