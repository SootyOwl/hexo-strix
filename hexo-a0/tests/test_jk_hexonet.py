"""Tests for Jumping Knowledge (JK) aggregation in HeXONet.

See docs/research/2026-05-13-jknet-axis-multiscale-hypothesis.md.
Variant A (scalar-weighted sum, jk_mode="sum") is the legacy
mode. Variants B (cat), C (max) and D (lstm) are now supported via PyG's
JumpingKnowledge module.
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet, split_param_groups


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic graphs, in the style of test_model.py
# ---------------------------------------------------------------------------

def _make_legal_mask(num_nodes: int, num_legal: int) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[:num_legal] = True
    return mask


def _make_pyg_data(num_nodes: int, num_legal: int, node_features: int = 8) -> Data:
    """Tiny chain graph with a legal_mask. Mirrors test_model.py."""
    x = torch.randn(num_nodes, node_features)
    src, dst = [], []
    for i in range(num_nodes - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    if not src:
        src, dst = [0], [0]
    edge_index = torch.tensor([src, dst], dtype=torch.int64)
    legal_mask = _make_legal_mask(num_nodes, num_legal)
    return Data(x=x, edge_index=edge_index, legal_mask=legal_mask)


def _small_cfg(**overrides) -> ModelConfig:
    """Small hex/gatv2 config for fast tests."""
    base = dict(
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        policy_hidden=16,
        value_hidden=16,
        graph_type="hex",
        conv_type="gatv2",
        prune_empty_edges=False,
        dropout=0.0,
        pre_norm=True,
    )
    base.update(overrides)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJKDisabledBitEquivalent:
    """use_jk=False is bit-identical to default (no new code path)."""

    def test_jk_disabled_bit_equivalent(self):
        torch.manual_seed(0)
        cfg_default = _small_cfg()  # use_jk=False by default
        torch.manual_seed(1234)
        net_default = HeXONet(cfg_default)

        torch.manual_seed(0)
        cfg_explicit = _small_cfg(use_jk=False)
        torch.manual_seed(1234)
        net_explicit = HeXONet(cfg_explicit)

        # Sanity: both networks have identical parameters under same seed
        for (n_a, p_a), (n_b, p_b) in zip(
            net_default.named_parameters(), net_explicit.named_parameters()
        ):
            assert n_a == n_b, f"parameter name mismatch: {n_a} vs {n_b}"
            assert torch.equal(p_a, p_b), f"parameter values differ for {n_a}"

        # Forward pass — bit-identical
        data = _make_pyg_data(num_nodes=10, num_legal=4, node_features=8)
        net_default.eval()
        net_explicit.eval()
        with torch.no_grad():
            p_a, v_a = net_default(data.x, data.edge_index, data.legal_mask)
            p_b, v_b = net_explicit(data.x, data.edge_index, data.legal_mask)
        assert torch.equal(p_a, p_b), "policy logits differ between default and use_jk=False"
        assert torch.equal(v_a, v_b), "value differs between default and use_jk=False"

    def test_jk_disabled_has_no_jk_weights_parameter(self):
        """When use_jk=False, jk_weights must not be in state_dict."""
        net = HeXONet(_small_cfg(use_jk=False))
        sd = net.state_dict()
        assert "representation.jk_weights" not in sd


class TestJKEnabledForwardShape:
    """use_jk=True doesn't change downstream tensor shapes."""

    def test_jk_enabled_forward_shape(self):
        cfg = _small_cfg(num_layers=3, hidden_dim=128, num_heads=8, use_jk=True)
        net = HeXONet(cfg)
        num_nodes, num_legal = 12, 5
        data = _make_pyg_data(num_nodes=num_nodes, num_legal=num_legal, node_features=8)
        net.eval()
        with torch.no_grad():
            policy, value = net(data.x, data.edge_index, data.legal_mask)
        assert policy.shape == (num_legal,), f"policy shape {policy.shape} != ({num_legal},)"
        assert value.dim() == 0, f"value should be 0-D scalar, got shape {value.shape}"

    def test_jk_weights_present_in_state_dict(self):
        cfg = _small_cfg(num_layers=3, hidden_dim=128, num_heads=8, use_jk=True)
        net = HeXONet(cfg)
        sd = net.state_dict()
        assert "representation.jk_weights" in sd
        assert sd["representation.jk_weights"].shape == (3,)

    def test_jk_weights_init_zeros(self):
        """Default init should be zeros (→ uniform 1/L after softmax)."""
        cfg = _small_cfg(num_layers=3, hidden_dim=128, num_heads=8, use_jk=True)
        net = HeXONet(cfg)
        assert torch.equal(net.representation.jk_weights, torch.zeros(3))


class TestJKGradient:
    """Gradients reach jk_weights through value-loss backprop."""

    def test_jk_weights_gradient_flows(self):
        torch.manual_seed(7)
        cfg = _small_cfg(num_layers=3, use_jk=True)
        net = HeXONet(cfg)
        data = _make_pyg_data(num_nodes=10, num_legal=4, node_features=8)

        _, value = net(data.x, data.edge_index, data.legal_mask)
        target = torch.tensor(0.5)
        loss = (value - target).pow(2)
        loss.backward()

        jk_grad = net.representation.jk_weights.grad
        assert jk_grad is not None, "jk_weights.grad is None"
        assert jk_grad.abs().sum().item() > 0.0, (
            f"jk_weights.grad is zero: {jk_grad}"
        )


class TestJKNoDecay:
    """jk_weights (1-D) lands in the no-decay AdamW param group."""

    def test_jk_weights_in_no_decay(self):
        cfg = _small_cfg(num_layers=3, use_jk=True)
        net = HeXONet(cfg)
        jk_param = net.representation.jk_weights
        jk_id = id(jk_param)

        groups = split_param_groups(net, weight_decay=1e-4)

        # Find which group contains jk_weights
        found_group: dict | None = None
        for g in groups:
            for p in g["params"]:
                if id(p) == jk_id:
                    found_group = g
                    break
            if found_group is not None:
                break

        assert found_group is not None, "jk_weights not found in any param group"
        assert found_group["weight_decay"] == 0.0, (
            f"jk_weights in group with weight_decay={found_group['weight_decay']}, "
            "expected 0.0"
        )


class TestJKInvalidMode:
    """Construction rejects unrecognised jk_mode values."""

    @pytest.mark.parametrize("bad_mode", ["concat", "channel", "", "weighted"])
    def test_jk_invalid_mode_rejected(self, bad_mode):
        cfg = _small_cfg(num_layers=3, use_jk=True, jk_mode=bad_mode)
        with pytest.raises(ValueError, match="jk_mode"):
            HeXONet(cfg)


# ---------------------------------------------------------------------------
# PyG JK modes: cat / max / lstm — added 2026-05-26 for the cat-mode fork.
# ---------------------------------------------------------------------------


class TestJKPygModes:
    """jk_mode in {cat, max, lstm} routes through PyG's JumpingKnowledge."""

    @pytest.mark.parametrize("mode", ["cat", "max", "lstm"])
    def test_pyg_mode_forward_shape(self, mode):
        torch.manual_seed(0)
        cfg = _small_cfg(num_layers=4, hidden_dim=32, use_jk=True, jk_mode=mode)
        net = HeXONet(cfg)
        num_nodes, num_legal = 12, 5
        data = _make_pyg_data(num_nodes=num_nodes, num_legal=num_legal, node_features=8)
        net.eval()
        with torch.no_grad():
            policy, value = net(data.x, data.edge_index, data.legal_mask)
        assert policy.shape == (num_legal,), f"{mode}: policy {policy.shape} != ({num_legal},)"
        assert value.dim() == 0, f"{mode}: value should be 0-D scalar"

    def test_cat_mode_head_input_dim(self):
        """cat mode: heads' first Linear has in_features = L*hidden_dim,
        but final_norm stays at hidden_dim (applied per-layer pre-cat).
        """
        H, L = 32, 4
        cfg = _small_cfg(num_layers=L, hidden_dim=H, use_jk=True, jk_mode="cat")
        net = HeXONet(cfg)
        assert net.representation.output_dim == L * H
        assert net.policy_head.mlp[0].in_features == L * H
        assert net.value_head.mlp[0].in_features == L * H
        # final_norm is still sized to H — applied to each h_i before cat,
        # so the no-JK checkpoint's final_norm weights load directly.
        assert net.representation.final_norm.normalized_shape == (H,)

    def test_max_mode_head_input_dim_unchanged(self):
        """max/lstm modes: heads stay at hidden_dim."""
        H, L = 32, 4
        cfg = _small_cfg(num_layers=L, hidden_dim=H, use_jk=True, jk_mode="max")
        net = HeXONet(cfg)
        assert net.representation.output_dim == H
        assert net.policy_head.mlp[0].in_features == H

    def test_cat_mode_has_jk_module(self):
        cfg = _small_cfg(num_layers=4, use_jk=True, jk_mode="cat")
        net = HeXONet(cfg)
        # PyG module path: representation.jk exists; no jk_weights scalar.
        assert hasattr(net.representation, "jk")
        sd = net.state_dict()
        assert "representation.jk_weights" not in sd


class TestJKCatGraft:
    """Head-graft routine preserves no-JK forward output at switch."""

    def test_graft_preserves_forward_output(self):
        """The load-bearing correctness property.

        Build a no-JK checkpoint, build a cat-mode model, graft the no-JK
        head weights into the cat-mode model's heads, and verify that the
        cat-mode model's forward output equals the no-JK model's output
        on the same input (within float tolerance).
        """
        from hexo_a0.model import graft_heads_for_jk_cat

        torch.manual_seed(123)
        H, L = 32, 4
        # 1. Build the source (no-JK) model and capture its state_dict.
        src_cfg = _small_cfg(num_layers=L, hidden_dim=H, use_jk=False)
        src_net = HeXONet(src_cfg)
        src_net.eval()
        src_sd = {k: v.clone() for k, v in src_net.state_dict().items()}

        # 2. Build a cat-mode target with the same backbone hyperparams.
        torch.manual_seed(0)  # different seed → different initial heads
        tgt_cfg = _small_cfg(num_layers=L, hidden_dim=H, use_jk=True, jk_mode="cat")
        tgt_net = HeXONet(tgt_cfg)
        tgt_net.eval()

        # 3. Graft heads into a working state_dict, then load.
        sd_to_load = {k: v.clone() for k, v in src_sd.items()}
        grafted = graft_heads_for_jk_cat(
            tgt_net, sd_to_load, hidden_dim=H, num_layers=L,
        )
        assert grafted, "graft_heads_for_jk_cat returned False — expected a graft"

        # The grafted weights should be the right shape now.
        assert sd_to_load["policy_head.mlp.0.weight"].shape == (
            tgt_cfg.policy_hidden, L * H,
        )
        # Last H columns must equal the original src weights.
        new_pw = sd_to_load["policy_head.mlp.0.weight"]
        assert torch.equal(
            new_pw[:, (L - 1) * H :], src_sd["policy_head.mlp.0.weight"],
        )
        # Other columns must be zero.
        assert torch.equal(
            new_pw[:, : (L - 1) * H],
            torch.zeros(tgt_cfg.policy_hidden, (L - 1) * H),
        )

        # 4. Load into target (strict=False; the cat-mode model has extra
        # bookkeeping like representation.jk submodules, but no weights to
        # load since PyG's JumpingKnowledge for cat has no params).
        missing, unexpected = tgt_net.load_state_dict(sd_to_load, strict=False)
        # Unexpected keys are fine; missing keys should at most be the
        # cat-specific keys (currently none — cat has no learnable params).
        # If new modules with params get added later, they must not be
        # absent here (the test will catch that).
        assert not unexpected, f"unexpected keys after load: {unexpected}"

        # 5. Compare forward outputs.
        data = _make_pyg_data(num_nodes=12, num_legal=5, node_features=8)
        with torch.no_grad():
            p_src, v_src = src_net(data.x, data.edge_index, data.legal_mask)
            p_tgt, v_tgt = tgt_net(data.x, data.edge_index, data.legal_mask)

        # Same float dtype, same math, no nondet → should be very tight.
        assert torch.allclose(p_src, p_tgt, atol=1e-6, rtol=1e-5), (
            f"policy mismatch after graft: src={p_src} tgt={p_tgt}"
        )
        assert torch.allclose(v_src, v_tgt, atol=1e-6, rtol=1e-5), (
            f"value mismatch after graft: src={v_src.item()} tgt={v_tgt.item()}"
        )

    def test_graft_missing_keys_raises(self):
        from hexo_a0.model import graft_heads_for_jk_cat

        tgt_cfg = _small_cfg(num_layers=2, hidden_dim=16, use_jk=True, jk_mode="cat")
        tgt_net = HeXONet(tgt_cfg)
        # Empty checkpoint dict — should fail loud.
        with pytest.raises(ValueError, match="required head keys missing"):
            graft_heads_for_jk_cat(
                tgt_net, {}, hidden_dim=16, num_layers=2,
            )

    def test_graft_idempotent_on_already_cat_shape(self):
        """If the checkpoint is already cat-shaped, graft returns False."""
        from hexo_a0.model import graft_heads_for_jk_cat

        H, L = 16, 2
        tgt_cfg = _small_cfg(num_layers=L, hidden_dim=H, use_jk=True, jk_mode="cat")
        tgt_net = HeXONet(tgt_cfg)
        sd = {k: v.clone() for k, v in tgt_net.state_dict().items()}
        before = sd["policy_head.mlp.0.weight"].clone()
        grafted = graft_heads_for_jk_cat(
            tgt_net, sd, hidden_dim=H, num_layers=L,
        )
        assert grafted is False
        assert torch.equal(sd["policy_head.mlp.0.weight"], before)
