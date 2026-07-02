"""Comprehensive tests for game_to_graph (GameState -> PyG Data)."""

import pytest
import torch
import hexo_rs
from torch_geometric.data import Batch

from hexo_a0.graph import game_to_graph


# ---------------------------------------------------------------------------
# AC-1: Node count
# ---------------------------------------------------------------------------

class TestNodeCount:
    """AC-1a: node count == placed stones + legal moves."""

    def test_initial_game_node_count(self, initial_game):
        data = game_to_graph(initial_game)
        expected = len(initial_game.placed_stones()) + len(initial_game.legal_moves())
        assert data.x.shape[0] == expected

    def test_small_game_node_count(self, small_game):
        data = game_to_graph(small_game)
        expected = len(small_game.placed_stones()) + len(small_game.legal_moves())
        assert data.x.shape[0] == expected

    def test_node_feature_dim(self, initial_game):
        data = game_to_graph(initial_game)
        assert data.x.shape[1] == 8

    def test_node_features_float32(self, initial_game):
        data = game_to_graph(initial_game)
        assert data.x.dtype == torch.float32


# ---------------------------------------------------------------------------
# AC-1c: Terminal state raises ValueError
# ---------------------------------------------------------------------------

class TestTerminalRaises:
    """AC-1c: calling game_to_graph on a terminal state raises ValueError."""

    def test_terminal_raises(self):
        # Build a terminal game by exhausting max_moves
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=4)
        game = hexo_rs.GameState(cfg)
        # Apply moves until terminal
        while not game.is_terminal():
            moves = game.legal_moves()
            if not moves:
                break
            game.apply_move(*moves[0])
        assert game.is_terminal(), "Precondition: game must be terminal"
        with pytest.raises(ValueError):
            game_to_graph(game)


# ---------------------------------------------------------------------------
# AC-2: Node features
# ---------------------------------------------------------------------------

class TestNodeFeatures:
    """AC-2: feature encoding correctness."""

    def test_p1_stone_one_hot(self, initial_game):
        """AC-2a: P1 stone → [1, 0, 0]."""
        data = game_to_graph(initial_game)
        stones = sorted(initial_game.placed_stones())
        p1_stones = [(coord, owner) for coord, owner in stones if owner == "P1"]
        assert len(p1_stones) > 0, "Need at least one P1 stone"
        # P1 stone at origin is index 0 (stones come first, sorted)
        idx = 0
        assert data.x[idx, 0].item() == pytest.approx(1.0)
        assert data.x[idx, 1].item() == pytest.approx(0.0)
        assert data.x[idx, 2].item() == pytest.approx(0.0)

    def test_p2_stone_one_hot(self, small_game):
        """AC-2b: P2 stone → [0, 1, 0]."""
        data = game_to_graph(small_game)
        stones = sorted(small_game.placed_stones())
        # Find first P2 stone index
        p2_idx = None
        for i, (coord, owner) in enumerate(stones):
            if owner == "P2":
                p2_idx = i
                break
        assert p2_idx is not None, "Need at least one P2 stone"
        assert data.x[p2_idx, 0].item() == pytest.approx(0.0)
        assert data.x[p2_idx, 1].item() == pytest.approx(1.0)
        assert data.x[p2_idx, 2].item() == pytest.approx(0.0)

    def test_empty_cell_one_hot(self, initial_game):
        """AC-2c: empty cell → [0, 0, 1]."""
        data = game_to_graph(initial_game)
        num_stones = len(initial_game.placed_stones())
        # First empty node is at index num_stones
        idx = num_stones
        assert data.x[idx, 0].item() == pytest.approx(0.0)
        assert data.x[idx, 1].item() == pytest.approx(0.0)
        assert data.x[idx, 2].item() == pytest.approx(1.0)

    def test_current_player_p2(self, initial_game):
        """AC-2d: current player is P2 → feature[3] == -1.0."""
        assert initial_game.current_player() == "P2"
        data = game_to_graph(initial_game)
        # All nodes have same current-player value
        assert torch.all(data.x[:, 3] == -1.0)

    def test_current_player_p1(self):
        """AC-2d: current player is P1 → feature[3] == +1.0."""
        # After P2 places one stone, it's P1's turn (P2 has 2 moves remaining initially)
        # But in default game, P2 moves first with 2 moves. After 1 P2 move, P2 still moves.
        # We need a state where P1 is to move. That happens after P2 uses both moves.
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)   # P2 move 1
        game.apply_move(-1, 0)  # P2 move 2 → now P1's turn
        assert game.current_player() == "P1"
        data = game_to_graph(game)
        assert torch.all(data.x[:, 3] == 1.0)

    def test_moves_remaining_two(self, initial_game):
        """AC-2e: 2 moves remaining → feature[4] == 1.0."""
        assert initial_game.moves_remaining_this_turn() == 2
        data = game_to_graph(initial_game)
        assert torch.allclose(data.x[:, 4], torch.ones(data.x.shape[0]))

    def test_moves_remaining_one(self):
        """AC-2e: 1 move remaining → feature[4] == 0.5."""
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)  # P2 uses first of 2 moves
        assert game.moves_remaining_this_turn() == 1
        data = game_to_graph(game)
        expected = torch.full((data.x.shape[0],), 0.5)
        assert torch.allclose(data.x[:, 4], expected)

    def test_normalised_coords_stones_centred(self, initial_game):
        """AC-2f: normalised coords for STONE nodes should have mean near 0."""
        data = game_to_graph(initial_game)
        n_stones = len(initial_game.placed_stones())
        # Centroid is computed over stones, so stone coords should be centred
        mean_q = data.x[:n_stones, 5].mean().item()
        mean_r = data.x[:n_stones, 6].mean().item()
        assert abs(mean_q) < 0.01, f"stone mean_q={mean_q} not near 0"
        assert abs(mean_r) < 0.01, f"stone mean_r={mean_r} not near 0"

    def test_normalised_coords_small_game(self, small_game):
        """AC-2f: normalised stone coords centred for small_game too."""
        data = game_to_graph(small_game)
        n_stones = len(small_game.placed_stones())
        mean_q = data.x[:n_stones, 5].mean().item()
        mean_r = data.x[:n_stones, 6].mean().item()
        assert abs(mean_q) < 0.01
        assert abs(mean_r) < 0.01


# ---------------------------------------------------------------------------
# AC-2g: Proximity and neighbour features
# ---------------------------------------------------------------------------

class TestProximityFeatures:
    """Features [7:10]: inverse distance, friendly/enemy neighbour counts."""

    def test_stone_nodes_have_zero_inv_distance(self, initial_game):
        """Stones themselves get inv_distance=0 (they ARE stones)."""
        data = game_to_graph(initial_game)
        n_stones = len(initial_game.placed_stones())
        assert torch.all(data.x[:n_stones, 7] == 0.0)

    def test_adjacent_legal_moves_have_high_inv_distance(self, initial_game):
        """Legal moves at distance 1 from a stone get inv_distance=1.0."""
        data = game_to_graph(initial_game)
        n_stones = len(initial_game.placed_stones())
        legal = sorted(initial_game.legal_moves())
        stones = {c for c, _ in initial_game.placed_stones()}
        for j, (q, r) in enumerate(legal):
            idx = n_stones + j
            # Check if distance-1 from any stone
            min_d = min(
                (abs(q - sq) + abs(r - sr) + abs((q - sq) + (r - sr))) // 2
                for sq, sr in stones
            )
            if min_d == 1:
                assert data.x[idx, 7].item() == pytest.approx(1.0)

    def test_inv_distance_decreases_with_distance(self, small_game):
        """Legal moves further from stones have lower inv_distance."""
        data = game_to_graph(small_game)
        n_stones = len(small_game.placed_stones())
        inv_dists = data.x[n_stones:, 7]
        # Should have values in (0, 1] — all > 0 since all legal moves are within radius
        assert inv_dists.min().item() > 0.0
        assert inv_dists.max().item() <= 1.0

    def test_stone_mask_present(self, initial_game):
        """Graph Data includes stone_mask attribute."""
        data = game_to_graph(initial_game)
        assert hasattr(data, "stone_mask")
        n_stones = len(initial_game.placed_stones())
        assert data.stone_mask[:n_stones].all()
        assert not data.stone_mask[n_stones:].any()


# ---------------------------------------------------------------------------
# AC-3: Edges
# ---------------------------------------------------------------------------

class TestEdges:
    """AC-3a: undirected — every edge (u, v) has a corresponding (v, u)."""

    def test_edges_undirected(self, initial_game):
        data = game_to_graph(initial_game)
        edge_index = data.edge_index
        # Build set of (u, v) tuples
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for u, v in list(edges):
            assert (v, u) in edges, f"Missing reverse edge for ({u}, {v})"

    def test_edges_undirected_small_game(self, small_game):
        data = game_to_graph(small_game)
        edge_index = data.edge_index
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for u, v in list(edges):
            assert (v, u) in edges, f"Missing reverse edge for ({u}, {v})"

    def test_edge_index_dtype(self, initial_game):
        data = game_to_graph(initial_game)
        assert data.edge_index.dtype == torch.int64

    def test_edge_index_shape(self, initial_game):
        data = game_to_graph(initial_game)
        assert data.edge_index.dim() == 2
        assert data.edge_index.shape[0] == 2

    def test_self_loops_present(self, initial_game):
        """Self-loops should be pre-baked from Rust graph construction."""
        data = game_to_graph(initial_game)
        src, dst = data.edge_index
        n_nodes = data.x.shape[0]
        self_loops = (src == dst).sum().item()
        assert self_loops == n_nodes, f"Expected {n_nodes} self-loops, got {self_loops}"

    def test_adjacent_stones_connected(self, small_game):
        """Stones adjacent in hex space should be connected by an edge."""
        data = game_to_graph(small_game)
        stones = sorted(small_game.placed_stones())
        coord_to_idx = {coord: i for i, (coord, _) in enumerate(stones)}
        # (0,0) and (1,0) are hex-adjacent — small_game has both
        assert (0, 0) in coord_to_idx, "Expected stone at (0,0)"
        assert (1, 0) in coord_to_idx, "Expected stone at (1,0)"
        u = coord_to_idx[(0, 0)]
        v = coord_to_idx[(1, 0)]
        edges = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        assert (u, v) in edges
        assert (v, u) in edges


# ---------------------------------------------------------------------------
# AC-4: legal_mask and coords
# ---------------------------------------------------------------------------

class TestLegalMaskAndCoords:
    """AC-4a/4b: legal_mask and coords tensors."""

    def test_legal_mask_false_for_stones(self, initial_game):
        """AC-4a: legal_mask is False for stone nodes."""
        data = game_to_graph(initial_game)
        num_stones = len(initial_game.placed_stones())
        stone_mask = data.legal_mask[:num_stones]
        assert not torch.any(stone_mask), "Stone nodes must have legal_mask=False"

    def test_legal_mask_true_for_empties(self, initial_game):
        """AC-4a: legal_mask is True for empty/legal nodes."""
        data = game_to_graph(initial_game)
        num_stones = len(initial_game.placed_stones())
        empty_mask = data.legal_mask[num_stones:]
        assert torch.all(empty_mask), "Empty nodes must have legal_mask=True"

    def test_legal_mask_dtype(self, initial_game):
        data = game_to_graph(initial_game)
        assert data.legal_mask.dtype == torch.bool

    def test_legal_mask_count(self, small_game):
        data = game_to_graph(small_game)
        n_stones = len(small_game.placed_stones())
        n_legal = len(small_game.legal_moves())
        assert data.legal_mask.sum().item() == n_legal
        assert (~data.legal_mask).sum().item() == n_stones

    def test_coords_shape(self, initial_game):
        """AC-4b: coords shape is (N, 2)."""
        data = game_to_graph(initial_game)
        N = data.x.shape[0]
        assert data.coords.shape == (N, 2)

    def test_coords_dtype(self, initial_game):
        """AC-4b: coords dtype is int32."""
        data = game_to_graph(initial_game)
        assert data.coords.dtype == torch.int32

    def test_coords_stone_values(self, initial_game):
        """AC-4b: stone coords match placed_stones() sorted by coordinate."""
        data = game_to_graph(initial_game)
        stones = sorted(initial_game.placed_stones())
        for i, ((q, r), _) in enumerate(stones):
            assert data.coords[i, 0].item() == q, f"q mismatch at stone index {i}"
            assert data.coords[i, 1].item() == r, f"r mismatch at stone index {i}"

    def test_coords_empty_values(self, initial_game):
        """AC-4b: empty node coords match legal_moves() sorted."""
        data = game_to_graph(initial_game)
        num_stones = len(initial_game.placed_stones())
        legal = initial_game.legal_moves()  # already sorted
        for j, (q, r) in enumerate(legal):
            idx = num_stones + j
            assert data.coords[idx, 0].item() == q, f"q mismatch at legal index {j}"
            assert data.coords[idx, 1].item() == r, f"r mismatch at legal index {j}"


# ---------------------------------------------------------------------------
# AC-5: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """AC-5: two calls on same state produce identical Data objects."""

    def test_deterministic_initial(self, initial_game):
        data1 = game_to_graph(initial_game)
        data2 = game_to_graph(initial_game)
        assert torch.equal(data1.x, data2.x)
        assert torch.equal(data1.edge_index, data2.edge_index)
        assert torch.equal(data1.legal_mask, data2.legal_mask)
        assert torch.equal(data1.coords, data2.coords)

    def test_deterministic_small_game(self, small_game):
        data1 = game_to_graph(small_game)
        data2 = game_to_graph(small_game)
        assert torch.equal(data1.x, data2.x)
        assert torch.equal(data1.edge_index, data2.edge_index)
        assert torch.equal(data1.legal_mask, data2.legal_mask)
        assert torch.equal(data1.coords, data2.coords)


# ---------------------------------------------------------------------------
# Origin not in legal_moves
# ---------------------------------------------------------------------------

class TestOriginNotLegal:
    """Origin (0,0) is occupied by P1 and must NOT appear in legal_moves."""

    def test_origin_not_in_legal_moves(self, initial_game):
        legal = initial_game.legal_moves()
        assert (0, 0) not in legal, "(0,0) is occupied and must not be in legal_moves"

    def test_origin_stone_not_in_empty_nodes(self, initial_game):
        data = game_to_graph(initial_game)
        num_stones = len(initial_game.placed_stones())
        empty_coords = data.coords[num_stones:]
        for i in range(empty_coords.shape[0]):
            q, r = empty_coords[i, 0].item(), empty_coords[i, 1].item()
            assert (q, r) != (0, 0), "(0,0) should not appear as an empty node"


# ---------------------------------------------------------------------------
# CRITICAL: policy-logit-to-coordinate roundtrip
# ---------------------------------------------------------------------------

class TestPolicyRoundtrip:
    """CRITICAL: data.coords[data.legal_mask] must match game.legal_moves() in order."""

    def test_policy_roundtrip_initial(self, initial_game):
        data = game_to_graph(initial_game)
        legal_from_game = initial_game.legal_moves()  # sorted list of (q, r)
        masked_coords = data.coords[data.legal_mask]  # (n_legal, 2)
        assert masked_coords.shape[0] == len(legal_from_game)
        for i, (q, r) in enumerate(legal_from_game):
            assert masked_coords[i, 0].item() == q, f"q mismatch at legal index {i}"
            assert masked_coords[i, 1].item() == r, f"r mismatch at legal index {i}"

    def test_policy_roundtrip_small_game(self, small_game):
        data = game_to_graph(small_game)
        legal_from_game = small_game.legal_moves()
        masked_coords = data.coords[data.legal_mask]
        assert masked_coords.shape[0] == len(legal_from_game)
        for i, (q, r) in enumerate(legal_from_game):
            assert masked_coords[i, 0].item() == q, f"q mismatch at legal index {i}"
            assert masked_coords[i, 1].item() == r, f"r mismatch at legal index {i}"


# ---------------------------------------------------------------------------
# AC-11: Batching — PyG Batch construction from multiple graphs
# ---------------------------------------------------------------------------

class TestBatching:
    """AC-11: Batch.from_data_list produces correct combined graph properties."""

    def test_num_graphs(self, initial_game, small_game):
        """AC-11c: Batch.from_data_list([g1, g2]).num_graphs == 2."""
        g1 = game_to_graph(initial_game)
        g2 = game_to_graph(small_game)
        batch = Batch.from_data_list([g1, g2])
        assert batch.num_graphs == 2

    def test_total_node_count(self, initial_game, small_game):
        """AC-11a: total node count equals sum of individual graph node counts."""
        g1 = game_to_graph(initial_game)
        g2 = game_to_graph(small_game)
        expected_nodes = g1.x.shape[0] + g2.x.shape[0]
        batch = Batch.from_data_list([g1, g2])
        assert batch.x.shape[0] == expected_nodes

    def test_legal_mask_total_preserved(self, initial_game, small_game):
        """AC-11b: total True count in batched legal_mask equals sum of individual counts."""
        g1 = game_to_graph(initial_game)
        g2 = game_to_graph(small_game)
        expected_legal = g1.legal_mask.sum().item() + g2.legal_mask.sum().item()
        batch = Batch.from_data_list([g1, g2])
        assert batch.legal_mask.sum().item() == expected_legal


# ---------------------------------------------------------------------------
# graph_fn_from_model_config factory + byte-identity reconstruction round-trip
# ---------------------------------------------------------------------------

import itertools

from hexo_a0.config import ModelConfig
from hexo_a0.graph import (
    graph_fn_from_model_config, game_to_axis_graph, reconstruct_game_state,
)


def _random_axis_state(plies, radius=4):
    cfg = hexo_rs.GameConfig(6, radius, 300)
    g = hexo_rs.GameState(cfg)
    import random
    rng = random.Random(plies)
    for _ in range(plies):
        if g.is_terminal():
            break
        legal = g.legal_moves()
        if not legal:
            break
        g.apply_move(*rng.choice(legal))
    return cfg, g


def test_graph_fn_from_model_config_axis_roundtrip():
    cfg, g = _random_axis_state(6)
    mc = ModelConfig(graph_type="axis", prune_empty_edges=True,
                     threat_features=False, relative_stone_encoding=False)
    fn = graph_fn_from_model_config(mc)
    rebuilt = fn(g)
    direct = game_to_axis_graph(g, prune_empty_edges=True,
                                threat_features=False, relative_stones=False)
    assert torch.equal(rebuilt.legal_mask, direct.legal_mask)
    assert torch.equal(rebuilt.edge_index, direct.edge_index)
    assert torch.allclose(rebuilt.x, direct.x, atol=1e-5)


@pytest.mark.parametrize(
    "graph_type,threat,relative",
    list(itertools.product(["axis", "hex"], [False, True], [False, True])))
def test_reconstruct_roundtrip_fidelity(graph_type, threat, relative):
    cfg, g = _random_axis_state(30)
    assert not g.is_terminal(), "stored positions must be non-terminal (picklable)"
    mc = ModelConfig(graph_type=graph_type, prune_empty_edges=True,
                     threat_features=threat, relative_stone_encoding=relative)
    fn = graph_fn_from_model_config(mc)
    g1 = fn(g)
    state2 = reconstruct_game_state(g1, cfg)
    g2 = fn(state2)
    assert torch.equal(g2.legal_mask, g1.legal_mask), "legal_mask drift"
    assert torch.equal(g2.edge_index, g1.edge_index), "edge_index drift"
    assert torch.equal(g2.coords, g1.coords), "coords drift"
    assert torch.allclose(g2.x, g1.x, atol=1e-5), "node-feature drift"
    if getattr(g1, "edge_attr", None) is not None:
        assert torch.allclose(g2.edge_attr, g1.edge_attr, atol=1e-5), "edge_attr drift"
