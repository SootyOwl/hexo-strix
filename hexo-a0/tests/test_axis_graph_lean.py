"""Native Rust lean axis-window builder vs the Python oracle.

The lean relational graph produced by ``game_to_axis_graph(..., lean flags...)``
MUST reproduce ``legacy_to_lean(game_to_axis_graph(..., legacy flags...))``
byte-for-byte (node features) / as-a-multiset (edges). ``legacy_to_lean`` is
imported from the end-to-end invariance test — it is the spec.
"""
from collections import Counter

import torch
import hexo_rs as hr

from hexo_a0.graph import game_to_axis_graph
from test_end_to_end_invariance import legacy_to_lean

WIN, RADIUS, MAXM = 6, 6, 300

# Real-ish positions: a list of move sequences (first move (0,0) is the seeded
# origin stone and is skipped by the builder).
POSITIONS = [
    [(0, 0), (1, 0), (0, 1), (2, 0), (-1, 1)],
    [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)],
    [(0, 0), (2, 0), (0, 2), (-2, 2), (1, 0), (0, -2), (2, -2), (3, -1)],
    [(0, 0), (1, 1), (2, 2), (-1, -1), (3, 0), (0, 3)],
]

# Legacy (relative + threat) flags.
_LEGACY = dict(prune_empty_edges=True, threat_features=True, relative_stones=True)
# Lean relational flags (add the D6-invariant lean schema on top).
_LEAN = dict(
    prune_empty_edges=True,
    threat_features=True,
    relative_stones=True,
    axis_relational=True,
    compact_stone_onehot=True,
    node_coords=False,
    moves_scope="node",
)


def _build_state(moves):
    st = hr.GameState(hr.GameConfig(WIN, RADIUS, MAXM))
    for i, (q, r) in enumerate(moves):
        if i == 0 and q == 0 and r == 0:
            continue
        st.apply_move(q, r)
    return st


def _axis_edge_multiset_from_lean(data):
    ei = data.edge_index
    return Counter(
        (int(ei[0, e]), int(ei[1, e]), int(data.edge_type[e]), int(data.edge_dist[e]))
        for e in range(ei.shape[1])
    )


def _axis_edge_multiset_from_oracle(o):
    ei = o["edge_index"]
    return Counter(
        (int(ei[0, e]), int(ei[1, e]), int(o["edge_type"][e]), int(o["edge_dist"][e]))
        for e in range(ei.shape[1])
    )


def _global_multiset(ei):
    return Counter((int(ei[0, e]), int(ei[1, e])) for e in range(ei.shape[1]))


def _cases():
    for moves in POSITIONS:
        st = _build_state(moves)
        native = game_to_axis_graph(st, **_LEAN)
        oracle = legacy_to_lean(game_to_axis_graph(st, **_LEGACY))
        yield moves, native, oracle


def test_lean_node_feature_width_is_8():
    st = _build_state(POSITIONS[0])
    native = game_to_axis_graph(st, **_LEAN)
    assert native.x.shape[1] == 8


def test_node_features_match_oracle():
    for moves, native, oracle in _cases():
        assert torch.equal(native.x, oracle["x"]), f"node features differ for {moves}"


def test_masks_and_coords_match_oracle():
    for moves, native, oracle in _cases():
        assert torch.equal(native.legal_mask, oracle["legal_mask"]), moves
        assert torch.equal(native.stone_mask, oracle["stone_mask"]), moves
        assert torch.equal(native.coords, oracle["coords"]), moves


def test_axis_edges_match_oracle():
    for moves, native, oracle in _cases():
        assert _axis_edge_multiset_from_lean(native) == _axis_edge_multiset_from_oracle(
            oracle
        ), f"axis edges differ for {moves}"


def test_global_edges_match_oracle():
    for moves, native, oracle in _cases():
        assert native.global_edge_index.numel() > 0
        assert _global_multiset(native.global_edge_index) == _global_multiset(
            oracle["global_edge_index"]
        ), f"global edges differ for {moves}"
