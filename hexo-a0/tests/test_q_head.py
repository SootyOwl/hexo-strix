"""Model, config, trainer-loss, and export tests for the train-only Q head."""

import pytest

torch = pytest.importorskip("torch")
from torch_geometric.data import Batch, Data

from hexo_a0.config import ModelConfig, TrainingConfig
from hexo_a0.model import HeXONet


def _cfg(q_head=True, **kw):
    base = dict(
        hidden_dim=32, num_layers=2, num_heads=2, policy_hidden=16,
        value_hidden=16, graph_type="hex", conv_type="gatv2",
    )
    base.update(kw)
    base["q_head"] = q_head
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


# --- Config -----------------------------------------------------------------

class TestConfig:
    def test_q_loss_weight_requires_q_head(self):
        from hexo_a0.config_io import _validate_config
        from hexo_a0.config import FullConfig
        cfg = FullConfig()
        cfg.model = _cfg(q_head=False)
        cfg.training = TrainingConfig(q_loss_weight=0.5)
        with pytest.raises(ValueError, match="q_loss_weight>0 requires"):
            _validate_config(cfg)


# --- Model ------------------------------------------------------------------

class TestQHeadModel:
    def test_builds_q_head(self):
        net = HeXONet(_cfg(q_head=True))
        assert net.q_head_enabled
        assert hasattr(net, "q_head")

    def test_no_q_head_when_disabled(self):
        net = HeXONet(_cfg(q_head=False))
        assert not net.q_head_enabled
        assert not hasattr(net, "q_head")

    def test_extras_has_q_values_aligned_with_logits(self):
        net = HeXONet(_cfg(q_head=True))
        batch = Batch.from_data_list([_data(10, 4), _data(8, 3)])
        all_logits, legal_counts, values, extras = net._forward_batch_core(
            batch, return_train_extras=True
        )
        assert "q_values" in extras
        # q_values aligned 1:1 with the flat legal logits
        assert extras["q_values"].shape == all_logits.shape == (7,)  # 4 + 3
        assert (extras["q_values"] >= -1).all() and (extras["q_values"] <= 1).all()

    def test_default_path_no_q_cost(self):
        net = HeXONet(_cfg(q_head=True))
        batch = Batch.from_data_list([_data()])
        out = net._forward_batch_core(batch)  # inference path
        assert len(out) == 3  # q head never computed


# --- Trainer loss -----------------------------------------------------------

def _example9(num_nodes=10, num_legal=4, value=0.5, q=None, visits=None):
    d = _data(num_nodes, num_legal)
    policy = torch.full((num_legal,), 1.0 / num_legal)
    q = q if q is not None else [0.2] * num_legal
    visits = visits if visits is not None else [1] * num_legal
    # (data, policy, value, traj, age, sw, horizon_targets, q_targets, q_visits)
    return (d, policy, torch.tensor(float(value)), None, None, 1.0, None, q, visits)


class TestTrainerQLoss:
    def test_q_loss_trains_head(self):
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg(q_head=True))
        examples = [
            _example9(value=1.0, q=[0.9, 0.8, -0.1, 0.0], visits=[10, 5, 3, 1]),
            _example9(8, 3, value=-1.0, q=[-0.5, 0.2, 0.1], visits=[4, 2, 1]),
        ]
        out = _forward_and_loss(net, examples, torch.device("cpu"), q_loss_weight=0.5)
        assert torch.is_tensor(out["q_loss"])
        assert out["q_loss"].item() >= 0.0
        assert any(p.grad is not None for p in net.q_head.parameters())

    def test_q_loss_zero_when_weight_off(self):
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg(q_head=True))
        examples = [_example9(value=1.0)]
        out = _forward_and_loss(net, examples, torch.device("cpu"), q_loss_weight=0.0)
        assert out["q_loss"] == 0.0

    def test_visit_mask_excludes_unsearched_moves(self):
        # A move with visit=0 must not contribute to the Q loss: if all visited
        # moves are perfectly predicted, loss is ~0 regardless of the visit=0
        # move's target.
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg(q_head=True))
        # Force the q head to output a known constant by zeroing its final layer
        # weights and setting bias so tanh(0)=0 → predictions all 0.
        with torch.no_grad():
            net.q_head.mlp[-2].weight.zero_()
            net.q_head.mlp[-2].bias.zero_()
        # visited moves have target 0 (matches prediction 0); unvisited move has
        # a wild target that must be ignored.
        examples = [
            _example9(value=1.0, q=[0.0, 0.0, 999.0], visits=[5, 5, 0], num_legal=3),
        ]
        out = _forward_and_loss(net, examples, torch.device("cpu"), q_loss_weight=1.0)
        assert out["q_loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_none_q_targets_skipped(self):
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg(q_head=True))
        # 9-tuple with None q data (legacy example): must not crash; masked out.
        ex = (_data(10, 4), torch.full((4,), 0.25), torch.tensor(1.0),
              None, None, 1.0, None, None, None)
        out = _forward_and_loss(net, [ex], torch.device("cpu"), q_loss_weight=0.5)
        assert out["q_loss"].item() == pytest.approx(0.0, abs=1e-6)


# --- Export drops the Q head ------------------------------------------------

class TestExportDropsQHead:
    def test_export_ignores_q_head(self, tmp_path):
        from hexo_a0.scriptable_model import export_torchscript
        net = HeXONet(_cfg(q_head=True))
        sd = dict(net.state_dict())
        out = tmp_path / "m.pt"
        export_torchscript(sd, _cfg(q_head=True), str(out))
        assert out.exists()
        scripted = torch.jit.load(str(out))
        names = [n for n, _ in scripted.named_parameters()]
        assert not any("q_head" in n for n in names)
