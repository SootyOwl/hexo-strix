"""Trainer-level tests for the PCZero path-consistency grouping fix.

Verifies the two correctness bugs that were fixed:
  1. Within-trajectory temporal ordering is recovered via the buffer insertion
     id (``_sid``), so the consistency term pairs positions by game order
     instead of random batch-sample order.
  2. The terminal term uses the latest (highest-sid) position's own
     ``value_target`` instead of the earliest position's (wrong sign).

Plus the plumbing contract: when the example tuples carry ``_sid`` (10-tuple),
PC loss is a finite tensor; when they don't (legacy 9-tuple), PC loss is the
``0.0`` sentinel because the consistency term cannot be made sound without
move-parity information.
"""

import pytest

torch = pytest.importorskip("torch")
from torch_geometric.data import Data

from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet


def _cfg(**kw):
    base = dict(
        hidden_dim=32, num_layers=2, num_heads=2, policy_hidden=16,
        value_hidden=16, graph_type="hex", conv_type="gatv2",
    )
    base.update(kw)
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


def _traj_example(value, traj_id, sid, num_legal=4):
    """Build a 10-tuple example on one trajectory with a known insertion id."""
    d = _data(num_legal=num_legal)
    policy = torch.full((num_legal,), 1.0 / num_legal)
    # (data, policy, value, traj_id, age, sw, horizon, q_t, q_v, _sid)
    return (
        d, policy, torch.tensor(float(value)), traj_id, None, 1.0,
        None, None, None, sid,
    )


class TestPCZeroGrouping:
    def test_pc_loss_finite_when_sids_present(self):
        """A 10-tuple batch with sids enables a real (finite) PC loss tensor."""
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg())
        # Four positions on one trajectory, sids in temporal order.
        examples = [_traj_example(v, 1, s) for s, v in enumerate([1, -1, 1, -1])]
        out = _forward_and_loss(net, examples, torch.device("cpu"), pc_loss_weight=0.5)
        assert torch.is_tensor(out["pc_loss"])
        assert torch.isfinite(out["pc_loss"])
        # PC loss must have contributed to total_loss (backward ran).
        assert any(p.grad is not None for p in net.value_head.parameters())

    def test_pc_loss_skipped_without_sids(self):
        """Legacy 9-tuple (no _sid) → consistency term unsound → PC loss is the
        0.0 sentinel, not a tensor, and total_loss is unaffected by pc_weight."""
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg())
        d = _data()
        policy = torch.full((4,), 0.25)
        # 9-tuple (no _sid): same trajectory_id so grouping fires, but no sids.
        examples = [
            (d, policy, torch.tensor(1.0), 1, None, 1.0, None, None, None),
            (d, policy, torch.tensor(-1.0), 1, None, 1.0, None, None, None),
        ]
        out = _forward_and_loss(net, examples, torch.device("cpu"), pc_loss_weight=0.5)
        assert out["pc_loss"] == 0.0

    def test_terminal_uses_latest_position_label(self):
        """The terminal term anchors the highest-sid position to its own
        ``value_target``, not the earliest position's (the old bug). With the
        value head forced to a constant prediction ``pred`` for every position:

          consistency = mean over odd-gap (even-sid × odd-sid) pairs of
                       (pred + pred)^2 = 4·pred^2
          terminal    = (pred - latest_label)^2

        sids [0,1,2,3] with labels [+1, +1, +1, -1] → latest_label = -1, so
        terminal = (pred + 1)^2. The old earliest-label code would instead use
        +1 and give (pred - 1)^2, a different value — this test pins the fix.
        """
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg())
        with torch.no_grad():
            net.value_head.mlp[-2].weight.zero_()   # final Linear -> output = bias
            net.value_head.mlp[-2].bias.fill_(0.5)  # tanh(0.5) for every position
        examples = [_traj_example(v, 1, s) for s, v in enumerate([1, 1, 1, -1])]
        out = _forward_and_loss(net, examples, torch.device("cpu"), pc_loss_weight=1.0)
        pred = float(torch.tanh(torch.tensor(0.5)))
        expected = 4 * pred * pred + (pred + 1.0) ** 2   # correct (latest = -1)
        assert out["pc_loss"].item() == pytest.approx(expected, abs=1e-4)

    def test_single_position_trajectory_skipped(self):
        """A trajectory with only one sampled position contributes nothing
        (no consistency pairs, terminal redundant)."""
        from hexo_a0.trainer import _forward_and_loss
        net = HeXONet(_cfg())
        examples = [_traj_example(1.0, 1, 0)]  # lone position
        out = _forward_and_loss(net, examples, torch.device("cpu"), pc_loss_weight=0.5)
        assert out["pc_loss"] == 0.0