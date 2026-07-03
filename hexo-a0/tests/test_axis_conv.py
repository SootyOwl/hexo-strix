"""Tests for AxisRelationalConv (Phase A).

The module encodes the 3 hex axes as an edge *type* partition with TIED
weights (not an absolute one-hot feature), so it is EXACTLY invariant to
permutation of the 3 axis labels. These tests build the graph tensors by
hand and never import ``hexo_rs``.
"""

import itertools

import torch

from hexo_a0.axis_conv import AxisRelationalConv


def _make_module(in_dim=6, out_dim=8, window=4, use_global=True, seed=0):
    torch.manual_seed(seed)
    mod = AxisRelationalConv(
        in_dim=in_dim, out_dim=out_dim, window=window, use_global=use_global
    )
    mod.eval()
    return mod


def _random_axis_graph(num_nodes, edges, seed=0):
    """edges: list of (src, dst, axis_type, dist). Returns tensors."""
    edge_index = torch.tensor([[s for s, _, _, _ in edges],
                               [d for _, d, _, _ in edges]], dtype=torch.long)
    edge_type = torch.tensor([t for _, _, t, _ in edges], dtype=torch.long)
    edge_dist = torch.tensor([w for _, _, _, w in edges], dtype=torch.long)
    return edge_index, edge_type, edge_dist


def test_axis_permutation_invariance():
    """Relabelling the 3 axis types by ANY of the 6 permutations must leave
    every node's output unchanged (tied weights + symmetric sum)."""
    mod = _make_module(use_global=False)
    torch.manual_seed(1)
    N = 6
    x = torch.randn(N, 6)
    # A graph with edges of all 3 axis types, bidirectional, varied dists.
    edges = [
        (0, 1, 0, 1), (1, 0, 0, 1),
        (1, 2, 1, 2), (2, 1, 1, 2),
        (2, 3, 2, 3), (3, 2, 2, 3),
        (0, 4, 1, 1), (4, 0, 1, 1),
        (3, 5, 0, 2), (5, 3, 0, 2),
        (4, 5, 2, 1), (5, 4, 2, 1),
    ]
    edge_index, edge_type, edge_dist = _random_axis_graph(N, edges)

    with torch.no_grad():
        base = mod(x, edge_index, edge_type, edge_dist)

    for perm in itertools.permutations(range(3)):
        pmap = torch.tensor(perm, dtype=torch.long)
        permuted_type = pmap[edge_type]
        with torch.no_grad():
            out = mod(x, edge_index, permuted_type, edge_dist)
        assert torch.allclose(base, out, atol=1e-5), (
            f"axis permutation {perm} changed the output"
        )


def test_non_degeneracy_and_line_distinction():
    """The module is not constant, and it distinguishes a same-axis
    (collinear) neighbour pair from a cross-axis neighbour pair."""
    mod = _make_module(use_global=False)
    torch.manual_seed(2)
    N = 3
    x = torch.randn(N, 6)

    # Two structurally different graphs give different outputs.
    ei_a, et_a, ed_a = _random_axis_graph(N, [(0, 1, 0, 1), (1, 0, 0, 1)])
    ei_b, et_b, ed_b = _random_axis_graph(N, [(0, 2, 1, 1), (2, 0, 1, 1)])
    with torch.no_grad():
        out_a = mod(x, ei_a, et_a, ed_a)
        out_b = mod(x, ei_b, et_b, ed_b)
    assert not torch.allclose(out_a, out_b, atol=1e-4), "module output is degenerate"

    # Node 0 RECEIVING from neighbours {1,2} (GINE flows source->target, so
    # the neighbours must be the edge sources): both same axis vs. different.
    same_axis = [(1, 0, 0, 1), (2, 0, 0, 1)]        # collinear pair
    cross_axis = [(1, 0, 0, 1), (2, 0, 1, 1)]       # two different lines
    ei_s, et_s, ed_s = _random_axis_graph(N, same_axis)
    ei_c, et_c, ed_c = _random_axis_graph(N, cross_axis)
    with torch.no_grad():
        out_s = mod(x, ei_s, et_s, ed_s)
        out_c = mod(x, ei_c, et_c, ed_c)
    assert not torch.allclose(out_s[0], out_c[0], atol=1e-5), (
        "same-axis and cross-axis neighbour pairs produced identical embeddings"
    )


def test_global_relation_isolation():
    """With a separate global/dummy relation present, permuting the 3 axis
    labels is still exactly invariant (global edges are not permuted)."""
    mod = _make_module(use_global=True)
    torch.manual_seed(3)
    N = 5
    x = torch.randn(N, 6)
    edges = [
        (0, 1, 0, 1), (1, 0, 0, 1),
        (1, 2, 1, 2), (2, 1, 1, 2),
        (2, 3, 2, 1), (3, 2, 2, 1),
        (3, 4, 0, 3), (4, 3, 0, 3),
    ]
    edge_index, edge_type, edge_dist = _random_axis_graph(N, edges)
    # Global edges: connect a few nodes, unrelated to axis partition.
    global_edge_index = torch.tensor([[0, 2, 4], [4, 0, 2]], dtype=torch.long)

    with torch.no_grad():
        base = mod(x, edge_index, edge_type, edge_dist,
                   global_edge_index=global_edge_index)

    for perm in itertools.permutations(range(3)):
        pmap = torch.tensor(perm, dtype=torch.long)
        with torch.no_grad():
            out = mod(x, edge_index, pmap[edge_type], edge_dist,
                      global_edge_index=global_edge_index)
        assert torch.allclose(base, out, atol=1e-5), (
            f"axis permutation {perm} changed output with global relation present"
        )


def test_distance_sensitivity():
    """Changing an edge's hop distance changes the output (the per-hop
    embedding table is actually used)."""
    mod = _make_module(use_global=False, window=4)
    torch.manual_seed(4)
    N = 3
    x = torch.randn(N, 6)
    edges = [(0, 1, 0, 1), (1, 0, 0, 1)]
    edge_index, edge_type, edge_dist = _random_axis_graph(N, edges)

    with torch.no_grad():
        out1 = mod(x, edge_index, edge_type, edge_dist)

    edge_dist2 = edge_dist.clone()
    edge_dist2[:] = 3  # change hop distance 1 -> 3
    with torch.no_grad():
        out2 = mod(x, edge_index, edge_type, edge_dist2)

    assert not torch.allclose(out1, out2, atol=1e-5), (
        "changing edge_dist did not change the output"
    )
