"""Tests for Jumping Knowledge aggregation in ScriptableHeXONet.

Covers the Variant A (weighted-sum) JK implementation defined in
docs/research/2026-05-13-jknet-axis-multiscale-hypothesis.md.

These tests are deliberately self-contained: they construct
``ScriptableHeXONet`` directly and do not rely on parallel changes in
``HeXONet`` / ``ModelConfig``. Cross-class roundtrip (HeXONet ↔
ScriptableHeXONet with ``use_jk=True``) is covered by the integrator.
"""

from __future__ import annotations

import pytest
import torch

from hexo_a0.scriptable_model import ScriptableHeXONet, load_from_hexonet


# ---------------------------------------------------------------------------
# Synthetic-graph helpers (deliberately tiny; we don't need real game states).
# ---------------------------------------------------------------------------


def _make_inputs(
    num_nodes: int = 8,
    num_legal: int = 3,
    num_stones: int = 2,
    node_features: int = 8,
    seed: int = 0,
):
    """Build a minimal axis-graph batch (1 graph) for forward-pass tests.

    The chain edges make the graph well-defined for both GATv2 and GINE
    layers. ``edge_attr`` is the 5-dim axis edge feature.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, node_features, generator=g)

    # Bidirectional chain edges
    src = []
    dst = []
    for i in range(num_nodes - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    if not src:
        src, dst = [0], [0]
    edge_index = torch.tensor([src, dst], dtype=torch.int64)
    edge_attr = torch.randn(edge_index.shape[1], 5, generator=g)

    legal_mask = torch.zeros(num_nodes, dtype=torch.bool)
    legal_mask[:num_legal] = True
    stone_mask = torch.zeros(num_nodes, dtype=torch.bool)
    stone_mask[num_legal : num_legal + num_stones] = True
    batch = torch.zeros(num_nodes, dtype=torch.long)
    return x, edge_index, legal_mask, stone_mask, batch, edge_attr


def _make_net(use_jk: bool = False, jk_mode: str = "sum", seed: int = 1234):
    torch.manual_seed(seed)
    return ScriptableHeXONet(
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        policy_hidden=32,
        value_hidden=32,
        graph_type="axis",
        conv_type="gine",
        use_jk=use_jk,
        jk_mode=jk_mode,
    )


# ---------------------------------------------------------------------------
# 1. Forward runs with JK on.
# ---------------------------------------------------------------------------


def test_scriptable_jk_forward_runs():
    """JK forward produces finite tensors with the expected shapes."""
    net = _make_net(use_jk=True, jk_mode="sum")
    net.eval()
    x, edge_index, legal_mask, stone_mask, batch, edge_attr = _make_inputs(
        num_nodes=8, num_legal=3, num_stones=2
    )
    with torch.no_grad():
        logits, counts, values = net(
            x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr,
        )
    assert logits.shape == (3,)
    assert counts.shape == (1,)
    assert values.shape == (1,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(counts).all()
    assert torch.isfinite(values).all()
    # jk_weights parameter must exist and have the right shape.
    assert net.jk_weights.shape == (3,)


# ---------------------------------------------------------------------------
# 2. With use_jk=False, jk_weights value is irrelevant — forward is unchanged.
# ---------------------------------------------------------------------------


def test_scriptable_jk_disabled_unchanged():
    """When use_jk=False the value of jk_weights must not affect forward.

    This is the bit-identical-when-off guarantee for existing checkpoints.
    """
    net = _make_net(use_jk=False)
    net.eval()
    inputs = _make_inputs(num_nodes=8, num_legal=3, num_stones=2, seed=5)
    x, edge_index, legal_mask, stone_mask, batch, edge_attr = inputs

    with torch.no_grad():
        l0, c0, v0 = net(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)

    # Scramble the (unused) JK weights and re-run.
    with torch.no_grad():
        net.jk_weights.copy_(torch.randn_like(net.jk_weights) * 10.0)
        l1, c1, v1 = net(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)

    assert torch.equal(l0, l1), "logits drifted when jk_weights changed with use_jk=False"
    assert torch.equal(c0, c1)
    assert torch.equal(v0, v1), "values drifted when jk_weights changed with use_jk=False"


# ---------------------------------------------------------------------------
# 3. torch.jit.script compiles, and scripted output matches eager.
# ---------------------------------------------------------------------------


def test_scriptable_jk_torchscript_compiles():
    """JK-enabled model scripts cleanly and matches the eager forward."""
    net = _make_net(use_jk=True, jk_mode="sum", seed=99)
    net.eval()
    scripted = torch.jit.script(net)

    x, edge_index, legal_mask, stone_mask, batch, edge_attr = _make_inputs(
        num_nodes=10, num_legal=4, num_stones=2, seed=7
    )
    with torch.no_grad():
        le, ce, ve = net(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)
        ls, cs, vs = scripted(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)

    assert torch.allclose(le, ls, atol=1e-6), (
        f"scripted logits differ; max |diff|={(le-ls).abs().max().item()}"
    )
    assert torch.equal(ce, cs)
    assert torch.allclose(ve, vs, atol=1e-6), (
        f"scripted values differ; max |diff|={(ve-vs).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 4. load_from_hexonet maps representation.jk_weights → jk_weights.
# ---------------------------------------------------------------------------


def test_load_from_hexonet_jk_key_mapping():
    """A state-dict with representation.jk_weights loads into jk_weights."""
    net = _make_net(use_jk=True, jk_mode="sum", seed=11)

    # Take the scriptable model's own state_dict and re-prefix conv/norm/etc.
    # keys with ``representation.`` to mimic a HeXONet checkpoint layout.
    base = net.state_dict()
    representation_prefixes = (
        "input_proj.",
        "edge_proj.",
        "convs.",
        "norms.",
        "layer_scales.",
        "jk_weights",
        "final_norm.",
    )
    hex_sd: dict[str, torch.Tensor] = {}
    for k, v in base.items():
        if any(k.startswith(p) or k == p for p in representation_prefixes):
            hex_sd["representation." + k] = v.clone()
        elif k.startswith("policy_mlp."):
            hex_sd["policy_head.mlp." + k[len("policy_mlp."):]] = v.clone()
        elif k.startswith("value_mlp."):
            hex_sd["value_head.mlp." + k[len("value_mlp."):]] = v.clone()
        else:
            # Unknown key — pass through unchanged.
            hex_sd[k] = v.clone()

    # Replace the JK weights with a known value so we can assert on load.
    jk_known = torch.tensor([0.1, 0.2, 0.3])
    hex_sd["representation.jk_weights"] = jk_known.clone()

    # Fresh target net with default (zero) jk_weights.
    target = _make_net(use_jk=True, jk_mode="sum", seed=22)
    assert torch.equal(target.jk_weights.detach(), torch.zeros(3))

    load_from_hexonet(target, hex_sd)

    assert torch.allclose(target.jk_weights.detach(), jk_known, atol=0.0), (
        f"jk_weights not loaded; got {target.jk_weights.detach()}"
    )


# ---------------------------------------------------------------------------
# 5. Loading a state-dict without jk_weights leaves the default in place.
# ---------------------------------------------------------------------------


def test_load_from_hexonet_jk_key_absent_ok():
    """Non-JK checkpoint loads cleanly into a JK-allocated scriptable model."""
    net = _make_net(use_jk=True, jk_mode="sum", seed=33)

    base = net.state_dict()
    representation_prefixes = (
        "input_proj.",
        "edge_proj.",
        "convs.",
        "norms.",
        "layer_scales.",
        "final_norm.",
    )
    hex_sd: dict[str, torch.Tensor] = {}
    for k, v in base.items():
        if k == "jk_weights":
            # Intentionally drop the JK key — simulates a pre-JK checkpoint.
            continue
        if any(k.startswith(p) for p in representation_prefixes):
            hex_sd["representation." + k] = v.clone()
        elif k.startswith("policy_mlp."):
            hex_sd["policy_head.mlp." + k[len("policy_mlp."):]] = v.clone()
        elif k.startswith("value_mlp."):
            hex_sd["value_head.mlp." + k[len("value_mlp."):]] = v.clone()
        else:
            hex_sd[k] = v.clone()

    target = _make_net(use_jk=True, jk_mode="sum", seed=44)
    default_jk = target.jk_weights.detach().clone()

    # Must not raise.
    load_from_hexonet(target, hex_sd)

    # jk_weights should be unchanged from its default init (torch.zeros).
    assert torch.equal(target.jk_weights.detach(), default_jk)
    assert torch.equal(target.jk_weights.detach(), torch.zeros(3))


# ---------------------------------------------------------------------------
# 6. Unsupported jk_mode is rejected at construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_mode", ["concat", "channel", "lstm", ""])
def test_scriptable_jk_invalid_mode_rejected(bad_mode):
    """jk_mode not in {sum, cat, max} must raise ValueError.

    'lstm' is intentionally NOT supported in ScriptableHeXONet — it adds
    an LSTM module with internal state which makes the forward pass less
    TorchScript-friendly and is not currently exercised by self-play.
    """
    with pytest.raises(ValueError):
        ScriptableHeXONet(
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            policy_hidden=16,
            value_hidden=16,
            graph_type="axis",
            conv_type="gine",
            use_jk=True,
            jk_mode=bad_mode,
        )


@pytest.mark.parametrize("mode", ["cat", "max"])
def test_scriptable_jk_pyg_modes_forward_shape(mode):
    """cat / max modes run and produce well-shaped outputs."""
    net = _make_net(use_jk=True, jk_mode=mode)
    net.eval()
    x, edge_index, legal_mask, stone_mask, batch, edge_attr = _make_inputs(
        num_nodes=8, num_legal=3, num_stones=2,
    )
    with torch.no_grad():
        logits, counts, values = net(
            x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr,
        )
    assert logits.shape == (3,)
    assert counts.shape == (1,)
    assert values.shape == (1,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()


def test_scriptable_jk_cat_torchscript_compiles():
    """cat-mode model scripts cleanly and matches the eager forward."""
    net = _make_net(use_jk=True, jk_mode="cat", seed=77)
    net.eval()
    scripted = torch.jit.script(net)

    x, edge_index, legal_mask, stone_mask, batch, edge_attr = _make_inputs(
        num_nodes=10, num_legal=4, num_stones=2, seed=7,
    )
    with torch.no_grad():
        le, ce, ve = net(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)
        ls, cs, vs = scripted(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)
    assert torch.allclose(le, ls, atol=1e-6)
    assert torch.equal(ce, cs)
    assert torch.allclose(ve, vs, atol=1e-6)
