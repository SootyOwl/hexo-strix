"""End-to-end D6 invariance on REAL positions (no hexo_rs rebuild needed).

Bridges today's LEGACY axis graph (unchanged Rust builder) into the lean
relational format in Python, feeds it to HeXONet(axis_relational=True), and
asserts the full model's policy+value are EXACTLY invariant under every D6
transform of a real board — proving the architecture, not just the conv.

``legacy_to_lean`` is also the ORACLE for the native Rust lean builder
(Phase C): the Rust output must reproduce it.
"""
import itertools

import torch

from hexo_a0.config import ModelConfig, node_feature_dim
from hexo_a0.model import HeXONet
from hexo_a0.graph import game_to_axis_graph
import hexo_rs as hr

WIN, RADIUS, MAXM = 6, 6, 300

# D6_TRANSFORMS order 1..11 (identity omitted); matches random_augment's table.
TFN = [
    (lambda q, r: (-r, q + r)), (lambda q, r: (-q - r, q)),
    (lambda q, r: (-q, -r)),    (lambda q, r: (r, -q - r)),
    (lambda q, r: (q + r, -q)), (lambda q, r: (r, q)),
    (lambda q, r: (-q, q + r)), (lambda q, r: (-q - r, r)),
    (lambda q, r: (-r, -q)),    (lambda q, r: (q, -q - r)),
    (lambda q, r: (q + r, -r)),
]

# Legacy relative+threat node layout (11-dim):
#   [own, opp, empty, moves, norm_q, norm_r, inv_dist, threat x4]
# Lean (compact + no coords + moves as node feature) = 8-dim:
#   [own, opp, moves, inv_dist, threat x4]
_LEAN_NODE_COLS = torch.tensor([0, 1, 3, 6, 7, 8, 9, 10])


def _build_state(moves):
    st = hr.GameState(hr.GameConfig(WIN, RADIUS, MAXM))
    for i, (q, r) in enumerate(moves):
        if i == 0 and q == 0 and r == 0:
            continue
        st.apply_move(q, r)
    return st


def legacy_to_lean(data):
    """Convert a legacy axis Data (x[N,11], edge_attr[E,5]) to lean relational
    inputs: drop empty/coord node cols; axis one-hot -> edge_type; |signed_dist|
    -> edge_dist; drop src_player; route dummy (all-zero one-hot) edges to the
    global relation."""
    x, ea, ei = data.x, data.edge_attr, data.edge_index
    is_axis = ea[:, 0:3].abs().sum(dim=1) > 0.5
    edge_type = ea[:, 0:3].argmax(dim=1)[is_axis]
    edge_dist = ea[is_axis, 3].abs().round().long().clamp(min=1)
    return dict(
        x=x[:, _LEAN_NODE_COLS],
        edge_index=ei[:, is_axis],
        edge_type=edge_type,
        edge_dist=edge_dist,
        global_edge_index=ei[:, ~is_axis],
        legal_mask=data.legal_mask,
        stone_mask=data.stone_mask,
        coords=data.coords,
    )


def _lean_graph(moves):
    st = _build_state(moves)
    d = game_to_axis_graph(st, prune_empty_edges=True, threat_features=True,
                           relative_stones=True)
    return legacy_to_lean(d)


def _run(model, g):
    return model(g["x"], g["edge_index"], g["legal_mask"],
                 stone_mask=g["stone_mask"], edge_type=g["edge_type"],
                 edge_dist=g["edge_dist"], global_edge_index=g["global_edge_index"])


def test_real_position_full_d6_invariance():
    cfg = ModelConfig(
        relative_stone_encoding=True, threat_features=True,
        compact_stone_onehot=True, node_coords=False, moves_scope="node",
        axis_relational=True, num_layers=3, hidden_dim=32,
        use_jk=True, jk_mode="cat", axis_window=8,
    )
    assert node_feature_dim(cfg) == 8
    torch.manual_seed(0)
    model = HeXONet(cfg).eval()

    moves = [(0, 0), (1, 0), (0, 1), (2, 0), (-1, 1)]
    gP = _lean_graph(moves)
    assert gP["global_edge_index"].numel() > 0  # dummy relation present
    with torch.no_grad():
        polP, valP = _run(model, gP)
    legalP = gP["coords"][gP["legal_mask"]].tolist()

    for t in range(1, 12):
        tfn = TFN[t - 1]
        gG = _lean_graph([tfn(q, r) for (q, r) in moves])
        with torch.no_grad():
            polG, valG = _run(model, gG)

        assert abs(valP.item() - valG.item()) < 1e-4, f"value not invariant under t={t}"

        legalG = {tuple(c): i for i, c in enumerate(gG["coords"][gG["legal_mask"]].tolist())}
        for i, c in enumerate(legalP):
            j = legalG[tuple(tfn(c[0], c[1]))]
            assert abs(polP[i].item() - polG[j].item()) < 1e-4, (
                f"policy for move {tuple(c)} not invariant under t={t}"
            )


def _native_lean_graph(moves):
    """Lean relational inputs from the NATIVE Rust builder (Phase C), bypassing
    the Python bridge — this is the real training/serving path."""
    st = _build_state(moves)
    d = game_to_axis_graph(
        st, prune_empty_edges=True, threat_features=True, relative_stones=True,
        axis_relational=True, compact_stone_onehot=True, node_coords=False,
        moves_scope="node",
    )
    return dict(x=d.x, edge_index=d.edge_index, edge_type=d.edge_type,
                edge_dist=d.edge_dist, global_edge_index=d.global_edge_index,
                legal_mask=d.legal_mask, stone_mask=d.stone_mask, coords=d.coords)


def test_native_lean_full_d6_invariance():
    """The REAL path: native Rust lean builder -> HeXONet(axis_relational) is
    exactly D6-invariant on real positions (policy + value)."""
    cfg = ModelConfig(
        relative_stone_encoding=True, threat_features=True,
        compact_stone_onehot=True, node_coords=False, moves_scope="node",
        axis_relational=True, num_layers=3, hidden_dim=32,
        use_jk=True, jk_mode="cat", axis_window=8,
    )
    torch.manual_seed(0)
    model = HeXONet(cfg).eval()

    moves = [(0, 0), (1, 0), (0, 1), (2, 0), (-1, 1)]
    gP = _native_lean_graph(moves)
    with torch.no_grad():
        polP, valP = _run(model, gP)
    legalP = gP["coords"][gP["legal_mask"]].tolist()

    for t in range(1, 12):
        tfn = TFN[t - 1]
        gG = _native_lean_graph([tfn(q, r) for (q, r) in moves])
        with torch.no_grad():
            polG, valG = _run(model, gG)
        assert abs(valP.item() - valG.item()) < 1e-4, f"value not invariant t={t}"
        legalG = {tuple(c): i for i, c in enumerate(gG["coords"][gG["legal_mask"]].tolist())}
        for i, c in enumerate(legalP):
            j = legalG[tuple(tfn(c[0], c[1]))]
            assert abs(polP[i].item() - polG[j].item()) < 1e-4, (
                f"policy for {tuple(c)} not invariant t={t}"
            )


def test_lean_graph_matches_transform():
    """Sanity: the lean graph of g.P equals the g-transform of the lean graph of
    P — invariant node features, edge_type globally permuted, edge_dist equal."""
    moves = [(0, 0), (1, 0), (0, 1), (2, 0)]
    base = _lean_graph(moves)
    # multiset of (node feature rows) must be identical under any transform
    base_rows = sorted(map(tuple, base["x"].tolist()))
    for t in range(1, 12):
        tfn = TFN[t - 1]
        g = _lean_graph([tfn(q, r) for (q, r) in moves])
        assert sorted(map(tuple, g["x"].tolist())) == base_rows, f"node features drift t={t}"
        # edge_dist multiset invariant
        assert sorted(base["edge_dist"].tolist()) == sorted(g["edge_dist"].tolist()), t
