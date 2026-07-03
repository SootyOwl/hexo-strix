"""Full-model tests for the axis-relational HeXO backbone (Phase B).

These exercise ``HeXONet``/``RepresentationNetwork`` with
``model.axis_relational=True``, where the 3 hex axes are an edge TYPE
partition consumed by tied-weight ``AxisRelationalConv`` layers. Because
those layers are EXACTLY invariant to relabelling of the 3 axes, the whole
network's outputs must be unchanged under any of the 6 axis-label
permutations.

All graph tensors are built BY HAND; these tests never import ``hexo_rs``.
"""

import itertools

import pytest
import torch

from hexo_a0.config import ModelConfig, node_feature_dim
from hexo_a0.model import HeXONet


def _lean_relational_config(**overrides) -> ModelConfig:
    """A lean, D6-invariant axis-relational config (small + fast)."""
    kwargs = dict(
        relative_stone_encoding=True,
        threat_features=True,
        compact_stone_onehot=True,
        node_coords=False,
        axis_relational=True,
        axis_window=8,
        num_layers=3,
        hidden_dim=32,
        use_jk=True,
        jk_mode="cat",
        graph_type="axis",
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _synthetic_axis_graph(config: ModelConfig, seed: int = 0, with_global: bool = True):
    """Build an axis-window graph by hand.

    Returns a dict with x, edge_index, edge_type, edge_dist, legal_mask,
    stone_mask, and (optionally) global_edge_index. Stones and legal-move
    nodes are disjoint.
    """
    torch.manual_seed(seed)
    n = 8
    feat_dim = node_feature_dim(config)
    x = torch.randn(n, feat_dim)

    # (src, dst, axis_type, dist) — bidirectional, all 3 axis types present,
    # varied hop distances within [1, axis_window].
    edges = [
        (0, 3, 0, 1), (3, 0, 0, 1),
        (1, 4, 1, 2), (4, 1, 1, 2),
        (2, 5, 2, 3), (5, 2, 2, 3),
        (0, 1, 0, 2), (1, 0, 0, 2),
        (3, 6, 1, 1), (6, 3, 1, 1),
        (4, 7, 2, 2), (7, 4, 2, 2),
        (2, 6, 0, 4), (6, 2, 0, 4),
        (5, 7, 1, 3), (7, 5, 1, 3),
    ]
    edge_index = torch.tensor(
        [[s for s, _, _, _ in edges], [d for _, d, _, _ in edges]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([t for _, _, t, _ in edges], dtype=torch.long)
    edge_dist = torch.tensor([w for _, _, _, w in edges], dtype=torch.long)

    # Nodes 0,1,2 are stones; 3..7 are legal (empty) candidate cells.
    stone_mask = torch.zeros(n, dtype=torch.bool)
    stone_mask[:3] = True
    legal_mask = torch.zeros(n, dtype=torch.bool)
    legal_mask[3:] = True

    out = dict(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_dist=edge_dist,
        legal_mask=legal_mask,
        stone_mask=stone_mask,
    )
    if with_global:
        out["global_edge_index"] = torch.tensor(
            [[0, 4, 7], [7, 0, 4]], dtype=torch.long
        )
    return out


def test_full_model_axis_permutation_invariance():
    """Relabelling the 3 axis types by ANY of the 6 permutations must leave
    BOTH policy_logits and value unchanged (exact by construction)."""
    config = _lean_relational_config()
    torch.manual_seed(7)
    net = HeXONet(config)
    net.eval()

    g = _synthetic_axis_graph(config, seed=1, with_global=True)

    with torch.no_grad():
        base_policy, base_value = net(
            g["x"], g["edge_index"], g["legal_mask"], g["stone_mask"],
            edge_type=g["edge_type"], edge_dist=g["edge_dist"],
            global_edge_index=g["global_edge_index"],
        )

    for perm in itertools.permutations(range(3)):
        pmap = torch.tensor(perm, dtype=torch.long)
        permuted_type = pmap[g["edge_type"]]
        with torch.no_grad():
            policy, value = net(
                g["x"], g["edge_index"], g["legal_mask"], g["stone_mask"],
                edge_type=permuted_type, edge_dist=g["edge_dist"],
                global_edge_index=g["global_edge_index"],
            )
        assert torch.allclose(policy, base_policy, atol=1e-5), (
            f"axis permutation {perm} changed policy_logits"
        )
        assert torch.allclose(value, base_value, atol=1e-5), (
            f"axis permutation {perm} changed value"
        )


def test_relational_model_is_not_degenerate():
    """Different graphs must produce different outputs (the network is not
    constant)."""
    config = _lean_relational_config()
    torch.manual_seed(11)
    net = HeXONet(config)
    net.eval()

    g_a = _synthetic_axis_graph(config, seed=1, with_global=True)

    # Same nodes/masks, DIFFERENT edge topology → different embeddings.
    feat_dim = node_feature_dim(config)
    torch.manual_seed(1)
    x = torch.randn(8, feat_dim)  # identical node features to g_a
    edges_b = [
        (0, 5, 0, 1), (5, 0, 0, 1),
        (1, 6, 1, 4), (6, 1, 1, 4),
        (2, 7, 2, 2), (7, 2, 2, 2),
        (3, 4, 0, 5), (4, 3, 0, 5),
    ]
    edge_index_b = torch.tensor(
        [[s for s, _, _, _ in edges_b], [d for _, d, _, _ in edges_b]],
        dtype=torch.long,
    )
    edge_type_b = torch.tensor([t for _, _, t, _ in edges_b], dtype=torch.long)
    edge_dist_b = torch.tensor([w for _, _, _, w in edges_b], dtype=torch.long)

    with torch.no_grad():
        policy_a, value_a = net(
            g_a["x"], g_a["edge_index"], g_a["legal_mask"], g_a["stone_mask"],
            edge_type=g_a["edge_type"], edge_dist=g_a["edge_dist"],
            global_edge_index=g_a["global_edge_index"],
        )
        policy_b, value_b = net(
            x, edge_index_b, g_a["legal_mask"], g_a["stone_mask"],
            edge_type=edge_type_b, edge_dist=edge_dist_b,
            global_edge_index=g_a["global_edge_index"],
        )

    assert not torch.allclose(policy_a, policy_b, atol=1e-4), (
        "relational model produced identical policy for different graphs"
    )
    assert torch.isfinite(policy_a).all() and torch.isfinite(value_a).all()
    assert torch.isfinite(policy_b).all() and torch.isfinite(value_b).all()


def test_relational_model_rejects_edge_attr():
    """Relational mode must not silently accept legacy edge_attr."""
    config = _lean_relational_config()
    net = HeXONet(config)
    net.eval()
    g = _synthetic_axis_graph(config, seed=1, with_global=False)
    edge_attr = torch.randn(g["edge_index"].shape[1], 5)
    with pytest.raises(ValueError):
        net(
            g["x"], g["edge_index"], g["legal_mask"], g["stone_mask"],
            edge_attr=edge_attr,
        )


def test_relational_model_requires_edge_type_and_dist():
    """Relational mode must raise when edge_type/edge_dist are missing."""
    config = _lean_relational_config()
    net = HeXONet(config)
    net.eval()
    g = _synthetic_axis_graph(config, seed=1, with_global=False)
    with pytest.raises(ValueError):
        net(g["x"], g["edge_index"], g["legal_mask"], g["stone_mask"])


def test_legacy_axis_path_intact():
    """A legacy (axis_relational=False) axis-graph HeXONet still constructs
    and runs a forward pass on an (E, 5) edge_attr, producing finite
    policy + value."""
    config = ModelConfig(
        hidden_dim=32,
        num_layers=3,
        num_heads=8,
        graph_type="axis",
        conv_type="gine",
        relative_stone_encoding=True,
        threat_features=True,
        axis_relational=False,
    )
    net = HeXONet(config)
    net.eval()

    # Legacy path must build an edge_proj (Linear(5, hidden)).
    assert hasattr(net.representation, "edge_proj")

    feat_dim = node_feature_dim(config)  # 7 (relative) + 4 (threat) = 11
    assert feat_dim == 11

    torch.manual_seed(3)
    n = 8
    x = torch.randn(n, feat_dim)
    edges = [
        (0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2),
        (0, 1), (1, 0), (3, 6), (6, 3), (4, 7), (7, 4),
    ]
    edge_index = torch.tensor(
        [[s for s, _ in edges], [d for _, d in edges]], dtype=torch.long
    )
    edge_attr = torch.randn(edge_index.shape[1], 5)
    stone_mask = torch.zeros(n, dtype=torch.bool)
    stone_mask[:3] = True
    legal_mask = torch.zeros(n, dtype=torch.bool)
    legal_mask[3:] = True

    with torch.no_grad():
        policy, value = net(
            x, edge_index, legal_mask, stone_mask, edge_attr=edge_attr
        )
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()
    assert value.dim() == 0
