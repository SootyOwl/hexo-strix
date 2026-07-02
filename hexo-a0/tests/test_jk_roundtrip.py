"""Cross-class roundtrip tests for the JK aggregation.

Verifies that a HeXONet trained with ``use_jk=True`` can be exported to
ScriptableHeXONet via ``load_from_hexonet`` and produces bit-equivalent
forward outputs. This is the integration seam between the parallel
implementations in ``model.py`` and ``scriptable_model.py`` and is
deliberately not covered by either of the per-class test files.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_a0.scriptable_model import ScriptableHeXONet, load_from_hexonet


def _make_axis_batch(num_nodes: int = 10, num_legal: int = 4, num_stones: int = 3, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, 8, generator=g)
    src, dst = [], []
    for i in range(num_nodes - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    edge_index = torch.tensor([src, dst], dtype=torch.int64)
    edge_attr = torch.randn(edge_index.shape[1], 5, generator=g)
    legal_mask = torch.zeros(num_nodes, dtype=torch.bool)
    legal_mask[:num_legal] = True
    stone_mask = torch.zeros(num_nodes, dtype=torch.bool)
    stone_mask[num_legal : num_legal + num_stones] = True
    return x, edge_index, edge_attr, legal_mask, stone_mask


def _cfg_axis_jk(use_jk: bool) -> ModelConfig:
    return ModelConfig(
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        policy_hidden=32,
        value_hidden=32,
        graph_type="axis",
        conv_type="gine",
        use_jk=use_jk,
        jk_mode="sum",
    )


def _run_hexonet(net: HeXONet, x, edge_index, edge_attr, legal_mask, stone_mask):
    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        legal_mask=legal_mask, stone_mask=stone_mask,
    )
    batch = Batch.from_data_list([data])
    policy_list, values = net.forward_batch(batch)
    return policy_list[0], values[0]


def _run_scriptable(net: ScriptableHeXONet, x, edge_index, edge_attr, legal_mask, stone_mask):
    batch_vec = torch.zeros(x.shape[0], dtype=torch.long)
    logits, _legal_counts, values = net(
        x=x,
        edge_index=edge_index,
        legal_mask=legal_mask,
        stone_mask=stone_mask,
        batch=batch_vec,
        num_graphs=1,
        edge_attr=edge_attr,
    )
    return logits, values[0]


def test_jk_roundtrip_outputs_match():
    """HeXONet(use_jk=True) → ScriptableHeXONet via load_from_hexonet produces matching outputs."""
    torch.manual_seed(2026)
    cfg = _cfg_axis_jk(use_jk=True)
    hexonet = HeXONet(cfg).eval()

    with torch.no_grad():
        hexonet.representation.jk_weights.copy_(torch.tensor([0.4, -0.1, 0.7]))

    scriptable = ScriptableHeXONet(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        policy_hidden=cfg.policy_hidden,
        value_hidden=cfg.value_hidden,
        graph_type=cfg.graph_type,
        conv_type=cfg.conv_type,
        use_jk=cfg.use_jk,
        jk_mode=cfg.jk_mode,
    ).eval()
    load_from_hexonet(scriptable, hexonet.state_dict())

    assert torch.equal(
        scriptable.jk_weights,
        hexonet.representation.jk_weights,
    ), "jk_weights did not roundtrip through load_from_hexonet"

    x, edge_index, edge_attr, legal_mask, stone_mask = _make_axis_batch()
    with torch.no_grad():
        h_logits, h_value = _run_hexonet(hexonet, x, edge_index, edge_attr, legal_mask, stone_mask)
        s_logits, s_value = _run_scriptable(scriptable, x, edge_index, edge_attr, legal_mask, stone_mask)

    torch.testing.assert_close(h_logits, s_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h_value, s_value, atol=1e-5, rtol=1e-5)


def test_jk_roundtrip_disabled_matches_legacy_path():
    """use_jk=False on both classes still produces matching outputs (no regression)."""
    torch.manual_seed(2027)
    cfg = _cfg_axis_jk(use_jk=False)
    hexonet = HeXONet(cfg).eval()

    scriptable = ScriptableHeXONet(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        policy_hidden=cfg.policy_hidden,
        value_hidden=cfg.value_hidden,
        graph_type=cfg.graph_type,
        conv_type=cfg.conv_type,
        use_jk=False,
        jk_mode="sum",
    ).eval()
    load_from_hexonet(scriptable, hexonet.state_dict())

    x, edge_index, edge_attr, legal_mask, stone_mask = _make_axis_batch()
    with torch.no_grad():
        h_logits, h_value = _run_hexonet(hexonet, x, edge_index, edge_attr, legal_mask, stone_mask)
        s_logits, s_value = _run_scriptable(scriptable, x, edge_index, edge_attr, legal_mask, stone_mask)

    torch.testing.assert_close(h_logits, s_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h_value, s_value, atol=1e-5, rtol=1e-5)


def test_jk_roundtrip_torchscript_export():
    """A use_jk=True ScriptableHeXONet loaded from HeXONet scripts and runs."""
    torch.manual_seed(2028)
    cfg = _cfg_axis_jk(use_jk=True)
    hexonet = HeXONet(cfg).eval()

    scriptable = ScriptableHeXONet(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        policy_hidden=cfg.policy_hidden,
        value_hidden=cfg.value_hidden,
        graph_type=cfg.graph_type,
        conv_type=cfg.conv_type,
        use_jk=cfg.use_jk,
        jk_mode=cfg.jk_mode,
    ).eval()
    load_from_hexonet(scriptable, hexonet.state_dict())

    scripted = torch.jit.script(scriptable)

    x, edge_index, edge_attr, legal_mask, stone_mask = _make_axis_batch()
    batch_vec = torch.zeros(x.shape[0], dtype=torch.long)
    with torch.no_grad():
        eager_logits, _, eager_values = scriptable(
            x=x, edge_index=edge_index, legal_mask=legal_mask,
            stone_mask=stone_mask, batch=batch_vec, num_graphs=1, edge_attr=edge_attr,
        )
        scripted_logits, _, scripted_values = scripted(
            x=x, edge_index=edge_index, legal_mask=legal_mask,
            stone_mask=stone_mask, batch=batch_vec, num_graphs=1, edge_attr=edge_attr,
        )

    torch.testing.assert_close(eager_logits, scripted_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(eager_values, scripted_values, atol=1e-5, rtol=1e-5)
