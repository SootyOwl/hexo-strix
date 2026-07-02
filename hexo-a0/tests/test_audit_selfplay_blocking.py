"""Tests for scripts/audit_selfplay_blocking.py node_dim-aware decoding.

The threat classifier must use the HX05 envelope's node_dim as the feature
stride. Absolute records (8/12) decode P1/P2 + to_move; relative records
(7/11) carry own/opp directly (side-to-move relative), which is exactly the
perspective the audit buckets by — no to_move needed. Unknown widths must
fail loudly.
"""
import sys
from pathlib import Path

import pytest

import hexo_rs

# scripts/ is not a package, so we add it to sys.path on import
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import audit_selfplay_blocking as asb  # noqa: E402
from test_audit_joint_q_regret import _hx05_record  # noqa: E402


def _threat_game(current_player="P1", moves_remaining=2):
    """5 consecutive P1 stones along the q-axis at win_length=6: a depth-1
    forced win for P1 with two completing gaps, (-1,0) and (5,0)."""
    gc = hexo_rs.GameConfig(win_length=6, placement_radius=2, max_moves=100)
    stones = [((q, 0), "P1") for q in range(5)]
    game = hexo_rs.GameState.from_state(stones, current_player, moves_remaining, gc)
    return game


def _example_at(game, *, threat=False, relative=False, axis=False):
    """Serialize `game` into one HX05 record and read it back through the
    trainer's envelope-aware reader (same path the script uses)."""
    import io

    from hexo_a0.trainer import Trainer

    record = _hx05_record(game, threat=threat, relative=relative, axis=axis)
    batch: list[dict] = []
    Trainer._read_binary_records(io.BytesIO(record), batch, 10)
    assert len(batch) == 1
    rec = batch[0]
    return rec["examples"][0], int(rec["node_dim"])


WIN_GAPS = {(-1, 0), (5, 0)}


class TestAbsoluteDecoding:
    @pytest.mark.parametrize("threat,dim", [(False, 8), (True, 12)])
    def test_own_threat_p1_to_move(self, threat, dim):
        game = _threat_game(current_player="P1")
        ex, node_dim = _example_at(game, threat=threat)
        assert node_dim == dim
        result = asb.analyse_position(ex, node_dim=node_dim)
        assert set(result["own_wins_1"]) == WIN_GAPS
        assert result["opp_wins_1"] == []
        assert result["start_of_turn"] is True
        assert set(result["policy_by_coord"]) == set(
            tuple(c) for c in game.legal_moves())

    @pytest.mark.parametrize("threat,dim", [(False, 8), (True, 12)])
    def test_opp_threat_p2_to_move(self, threat, dim):
        game = _threat_game(current_player="P2")
        ex, node_dim = _example_at(game, threat=threat)
        assert node_dim == dim
        result = asb.analyse_position(ex, node_dim=node_dim)
        assert result["own_wins_1"] == []
        assert set(result["opp_wins_1"]) == WIN_GAPS

    def test_mid_turn_not_start_of_turn(self):
        game = _threat_game(current_player="P1", moves_remaining=1)
        ex, node_dim = _example_at(game)
        result = asb.analyse_position(ex, node_dim=node_dim)
        assert result["start_of_turn"] is False

    def test_12dim_matches_8dim(self):
        game = _threat_game(current_player="P1")
        ex8, _ = _example_at(game)
        ex12, _ = _example_at(game, threat=True)
        r8 = asb.analyse_position(ex8, node_dim=8)
        r12 = asb.analyse_position(ex12, node_dim=12)
        for key in ("own_wins_1", "own_wins_2", "opp_wins_1", "opp_wins_2",
                    "policy_by_coord", "start_of_turn"):
            assert r8[key] == r12[key]


class TestRelativeDecoding:
    @pytest.mark.parametrize("threat,dim", [(False, 7), (True, 11)])
    def test_own_opp_direct_from_relative_layout(self, threat, dim):
        """P1 stones are 'own' when P1 is to move, 'opp' when P2 is — with no
        to_move dim involved."""
        game_own = _threat_game(current_player="P1")
        ex, node_dim = _example_at(game_own, threat=threat, relative=True)
        assert node_dim == dim
        r = asb.analyse_position(ex, node_dim=node_dim)
        assert set(r["own_wins_1"]) == WIN_GAPS
        assert r["opp_wins_1"] == []

        game_opp = _threat_game(current_player="P2")
        ex, node_dim = _example_at(game_opp, threat=threat, relative=True)
        r = asb.analyse_position(ex, node_dim=node_dim)
        assert r["own_wins_1"] == []
        assert set(r["opp_wins_1"]) == WIN_GAPS

    def test_relative_matches_absolute(self):
        game = _threat_game(current_player="P2")
        ex_abs, _ = _example_at(game)
        ex_rel, _ = _example_at(game, relative=True)
        r_abs = asb.analyse_position(ex_abs, node_dim=8)
        r_rel = asb.analyse_position(ex_rel, node_dim=7)
        for key in ("own_wins_1", "own_wins_2", "opp_wins_1", "opp_wins_2",
                    "policy_by_coord", "start_of_turn"):
            assert r_abs[key] == r_rel[key]

    def test_relative_start_of_turn_at_index_3(self):
        game = _threat_game(current_player="P1", moves_remaining=1)
        ex, node_dim = _example_at(game, relative=True)
        r = asb.analyse_position(ex, node_dim=node_dim)
        assert r["start_of_turn"] is False


class TestAxisDummyNode:
    """Axis graphs append a global dummy node (last, coords (0,0)). Keying
    nodes by coords let it clobber the origin stone's entry, hiding every
    win-window through the origin (the hex-graph tests above never see it)."""

    @pytest.mark.parametrize("relative", [False, True])
    def test_origin_window_survives_dummy(self, relative):
        # The 5-stone row runs THROUGH the pre-seeded origin (0,0); both
        # completing windows contain it.
        game = _threat_game(current_player="P1")
        ex, node_dim = _example_at(game, relative=relative, axis=True)
        assert "edge_attr" in ex  # really an axis record (dummy node present)
        result = asb.analyse_position(ex, node_dim=node_dim)
        assert set(result["own_wins_1"]) == WIN_GAPS
        assert result["opp_wins_1"] == []

    def test_axis_matches_hex_record(self):
        game = _threat_game(current_player="P2")
        ex_hex, _ = _example_at(game)
        ex_axis, _ = _example_at(game, axis=True)
        r_hex = asb.analyse_position(ex_hex, node_dim=8)
        r_axis = asb.analyse_position(ex_axis, node_dim=8)
        for key in ("own_wins_1", "own_wins_2", "opp_wins_1", "opp_wins_2",
                    "policy_by_coord", "start_of_turn"):
            assert r_hex[key] == r_axis[key]


class TestRefusals:
    def test_unknown_node_dim_raises(self):
        game = _threat_game()
        ex, _ = _example_at(game)
        with pytest.raises(ValueError, match="unsupported node_dim"):
            asb.analyse_position(ex, node_dim=9)

    def test_stride_mismatch_raises(self):
        """A wrong (but supported) stride for the record's actual width must
        fail loudly, never silently misparse."""
        game = _threat_game()
        ex, _ = _example_at(game, relative=True)  # 7-dim features
        with pytest.raises(ValueError, match="features length"):
            asb.analyse_position(ex, node_dim=8)
