"""Model + trainer-loss tests for the distributional (binned) value head.

Covers: scalar-head backward compatibility (value_bins=0 path unchanged),
binned forward shape + decode, the return_train_extras training contract,
checkpoint config round-trip, and the trainer value-loss (two-hot CE) path.
"""

import pytest

torch = pytest.importorskip("torch")
from torch_geometric.data import Batch, Data

from hexo_a0.config import ModelConfig, model_config_from_checkpoint
from hexo_a0.model import HeXONet
from hexo_a0.loss import decode_binned_value


def _cfg(value_bins=0, **kw):
    base = dict(
        hidden_dim=32, num_layers=2, num_heads=2, policy_hidden=16,
        value_hidden=16, graph_type="hex", conv_type="gatv2",
    )
    base.update(kw)
    base["value_bins"] = value_bins
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


# --- Scalar head backward-compat -------------------------------------------

class TestScalarHeadUnchanged:
    def test_scalar_has_no_centers_buffer(self):
        net = HeXONet(_cfg(0))
        assert net.value_bins == 0
        assert not hasattr(net, "value_bin_centers")

    def test_scalar_value_head_single_output_with_tanh(self):
        net = HeXONet(_cfg(0))
        last = net.value_head.mlp[-2]  # -1 is Tanh, -2 is final Linear
        assert isinstance(last, torch.nn.Linear) and last.out_features == 1
        assert isinstance(net.value_head.mlp[-1], torch.nn.Tanh)

    def test_scalar_core_returns_three_tuple(self):
        net = HeXONet(_cfg(0))
        batch = Batch.from_data_list([_data()])
        out = net._forward_batch_core(batch)
        assert len(out) == 3
        _, _, values = out
        assert values.shape == (1,)
        assert -1.0 <= values.item() <= 1.0


# --- Binned head -----------------------------------------------------------

class TestBinnedHead:
    def test_binned_registers_centers(self):
        net = HeXONet(_cfg(65))
        assert net.value_bins == 65
        assert net.value_bin_centers.shape == (65,)
        assert torch.isclose(net.value_bin_centers[0], torch.tensor(-1.0))
        assert torch.isclose(net.value_bin_centers[-1], torch.tensor(1.0))

    def test_binned_value_head_outputs_bins_no_tanh(self):
        net = HeXONet(_cfg(65))
        last = net.value_head.mlp[-1]
        assert isinstance(last, torch.nn.Linear) and last.out_features == 65

    def test_binned_core_default_is_three_tuple_scalar(self):
        net = HeXONet(_cfg(65))
        batch = Batch.from_data_list([_data(), _data(8, 3)])
        out = net._forward_batch_core(batch)
        assert len(out) == 3
        _, _, values = out
        assert values.shape == (2,)
        assert (values >= -1.0).all() and (values <= 1.0).all()

    def test_binned_core_return_train_extras_four_tuple(self):
        net = HeXONet(_cfg(65))
        batch = Batch.from_data_list([_data(), _data(8, 3)])
        out = net._forward_batch_core(batch, return_train_extras=True)
        assert len(out) == 4
        _, _, values, extras = out
        bin_logits = extras["value_logits"]
        assert bin_logits.shape == (2, 65)
        # decoded scalar must match the softmax expectation of the returned logits
        assert torch.allclose(
            values, decode_binned_value(bin_logits, net.value_bin_centers), atol=1e-5
        )

    def test_binned_custom_range(self):
        net = HeXONet(_cfg(5, value_bin_min=0.0, value_bin_max=209.0))
        assert torch.isclose(net.value_bin_centers[0], torch.tensor(0.0))
        assert torch.isclose(net.value_bin_centers[-1], torch.tensor(209.0))

    def test_binned_gradients_flow(self):
        net = HeXONet(_cfg(9))
        batch = Batch.from_data_list([_data()])
        _, _, values, extras = net._forward_batch_core(batch, return_train_extras=True)
        extras["value_logits"].sum().backward()
        for name, p in net.value_head.named_parameters():
            assert p.grad is not None, f"no grad for {name}"


# --- Checkpoint config round-trip ------------------------------------------

class TestConfigRoundTrip:
    def test_value_bins_survives_checkpoint(self):
        import dataclasses
        cfg = _cfg(65, value_bin_min=-1.0, value_bin_max=1.0)
        ckpt = {"model_config": dataclasses.asdict(cfg)}
        rebuilt = model_config_from_checkpoint(ckpt)
        assert rebuilt.value_bins == 65

    def test_old_checkpoint_without_value_bins_defaults_scalar(self):
        # An old embedded config lacking the new keys must default to scalar.
        ckpt = {"model_config": {"hidden_dim": 32, "num_layers": 2}}
        rebuilt = model_config_from_checkpoint(ckpt)
        assert rebuilt.value_bins == 0


# --- Trainer value-loss path -----------------------------------------------

def _example(num_nodes=10, num_legal=4, nf=8, value=0.3):
    d = _data(num_nodes, num_legal, nf)
    policy = torch.full((num_legal,), 1.0 / num_legal)
    return (d, policy, torch.tensor(float(value)))


class TestTrainerValueLoss:
    def test_binned_value_loss_is_ce_and_backward(self):
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg(65))
        examples = [_example(value=1.0), _example(8, 3, value=-1.0)]
        out = _forward_and_loss(net, examples, torch.device("cpu"))
        assert out["value_loss"].item() >= 0.0          # CE >= 0
        assert "value_mse" in out
        # backward ran inside _forward_and_loss; params must have grads
        assert any(p.grad is not None for p in net.value_head.parameters())

    def test_scalar_value_loss_still_mse(self):
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg(0))
        examples = [_example(value=1.0), _example(8, 3, value=-1.0)]
        out = _forward_and_loss(net, examples, torch.device("cpu"))
        # scalar path: value_loss == value_mse (same tensor value)
        assert torch.isclose(out["value_loss"], out["value_mse"])
