"""Tests for scripts/audit_joint_q_regret.py binary-record reconstruction.

The reconstruction must use the HX05 envelope's node_dim as the feature
stride (8 absolute, 12 absolute+threat) and HARD-REFUSE relative-encoding
records (7/11): the own/opp layout drops the absolute to_move dim, so
absolute stone ownership is not recoverable.
"""
import struct
import sys
from pathlib import Path

import pytest

import hexo_rs

# scripts/ is not a package, so we add it to sys.path on import
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import audit_joint_q_regret as ajq  # noqa: E402


GC_KWARGS = dict(win_length=4, placement_radius=4, max_moves=50)


def _game(moves=((1, 0), (2, 0))):
    """Small game; origin (0,0) is pre-seeded with a P1 stone, P2 moves first."""
    gc = hexo_rs.GameConfig(**GC_KWARGS)
    g = hexo_rs.GameState(gc)
    for q, r in moves:
        g.apply_move(q, r)
    return g, gc


def _hx05_record(game, *, threat=False, relative=False, axis=False) -> bytes:
    """Serialize one HX05 record with a single example from `game` (hex graph
    by default; axis-window graph with trailing dummy node + edge_attr when
    ``axis=True``).

    Mirrors the Rust self-play writer envelope:
    [magic HX05][size u32][num_examples u32][winner i8][move_count u32]
    [node_dim u8][examples...].
    """
    if axis:
        raw = hexo_rs.game_to_axis_graph_raw(
            game, threat_features=threat, relative_stones=relative)
    else:
        raw = hexo_rs.game_to_graph_raw(
            game, threat_features=threat, relative_stones=relative)
    node_dim = (7 if relative else 8) + (4 if threat else 0)
    n = raw["num_nodes"]
    assert len(raw["features"]) == n * node_dim
    n_legal = sum(raw["legal_mask"])
    policy = [1.0 / n_legal] * n_legal
    edge_attr = raw.get("edge_attr") if axis else None

    body = struct.pack("<I", 1)                      # num_examples
    body += struct.pack("<b", 0)                     # winner = draw
    body += struct.pack("<I", game.move_count())     # move_count
    body += struct.pack("B", node_dim)               # node_dim (HX05)
    # example
    body += struct.pack("<HHH", n, len(raw["edge_src"]), n_legal)
    body += struct.pack("B", int(edge_attr is not None))  # has_edge_attr
    body += struct.pack("<f", 0.0)                   # value
    body += struct.pack("<f", 1.0)                   # sample_weight (HX04+)
    for f in raw["features"]:
        body += struct.pack("<f", f)
    for s in raw["edge_src"]:
        body += struct.pack("<H", s)
    for d in raw["edge_dst"]:
        body += struct.pack("<H", d)
    if edge_attr is not None:
        assert len(edge_attr) == len(raw["edge_src"]) * 5
        for a in edge_attr:
            body += struct.pack("<f", a)
    for b in raw["legal_mask"]:
        body += struct.pack("B", int(b))
    for b in raw["stone_mask"]:
        body += struct.pack("B", int(b))
    for c in raw["coords"]:
        body += struct.pack("<h", c)
    for p in policy:
        body += struct.pack("<f", p)
    return b"HX05" + struct.pack("<I", len(body)) + body


def _write_bin(tmp_path, record: bytes) -> Path:
    p = tmp_path / "games_0001.bin"
    p.write_bytes(record)
    return p


def _assert_state_matches(gs, game):
    assert sorted(gs.placed_stones()) == sorted(game.placed_stones())
    assert gs.current_player() == game.current_player()
    assert gs.moves_remaining_this_turn() == game.moves_remaining_this_turn()


class TestAbsoluteDecoding:
    def test_8dim_roundtrip(self, tmp_path):
        game, gc = _game()
        path = _write_bin(tmp_path, _hx05_record(game))
        positions = list(ajq.iter_positions_from_bin(path, gc))
        assert len(positions) == 1
        gs, meta = positions[0]
        _assert_state_matches(gs, game)
        assert meta["stone_count"] == len(game.placed_stones())

    def test_8dim_mid_turn_moves_remaining(self, tmp_path):
        # One P2 placement: mid-turn, moves_remaining == 1
        game, gc = _game(moves=((1, 0),))
        assert game.moves_remaining_this_turn() == 1
        path = _write_bin(tmp_path, _hx05_record(game))
        (gs, _meta), = ajq.iter_positions_from_bin(path, gc)
        _assert_state_matches(gs, game)

    def test_12dim_threat_stride(self, tmp_path):
        """Threat records (node_dim=12) decode with stride 12; the 4 threat
        dims are ignored and the state matches the 8-dim decode."""
        game, gc = _game()
        path = _write_bin(tmp_path, _hx05_record(game, threat=True))
        positions = list(ajq.iter_positions_from_bin(path, gc))
        assert len(positions) == 1
        gs, _meta = positions[0]
        _assert_state_matches(gs, game)


class TestRelativeRefusal:
    @pytest.mark.parametrize("threat,dim", [(False, 7), (True, 11)])
    def test_iter_refuses_relative_records(self, tmp_path, threat, dim):
        game, gc = _game()
        path = _write_bin(
            tmp_path, _hx05_record(game, threat=threat, relative=True))
        with pytest.raises(ValueError, match="relative"):
            list(ajq.iter_positions_from_bin(path, gc))

    @pytest.mark.parametrize("dim", [7, 11])
    def test_reconstruct_refuses_relative_node_dim(self, dim):
        _game_obj, gc = _game()
        with pytest.raises(ValueError, match="relative"):
            ajq.reconstruct_from_bin_example({}, gc, node_dim=dim)

    def test_reconstruct_refuses_unknown_node_dim(self):
        _game_obj, gc = _game()
        with pytest.raises(ValueError, match="unsupported node_dim"):
            ajq.reconstruct_from_bin_example({}, gc, node_dim=9)
