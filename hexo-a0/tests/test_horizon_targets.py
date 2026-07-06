"""Tests for short-horizon value-target construction (Stage 2).

The horizon target for a position is the position's own side-to-move outcome
IF the game resolves within k placements of it, else 0.0 (neutral). Targets are
built at data-ingest time (order is not persisted downstream), so this pure
helper is the TDD foundation.
"""

import pytest

torch = pytest.importorskip("torch")

from hexo_a0.trainer import _compute_horizon_targets


def test_empty_horizons_returns_empty_rows():
    vals = [1.0, -1.0, 1.0]
    rows = _compute_horizon_targets(vals, [])
    assert rows == [[], [], []]


def test_last_position_always_within_any_horizon():
    # plies_to_end for the last position is 0, so every k>=0 covers it.
    vals = [0.5, -0.5, 1.0]
    rows = _compute_horizon_targets(vals, [2])
    assert rows[-1] == [1.0]  # last position's own value


def test_horizon_gate_by_plies_to_end():
    # n=5; position i has plies_to_end = 4 - i.
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = _compute_horizon_targets(vals, [1])
    # k=1 covers positions with plies_to_end <= 1  → i in {3,4}
    assert rows[0] == [0.0]
    assert rows[1] == [0.0]
    assert rows[2] == [0.0]
    assert rows[3] == [0.4]
    assert rows[4] == [0.5]


def test_multiple_horizons_are_independent_columns():
    vals = [1.0, -1.0, 1.0, -1.0]  # n=4, plies_to_end = 3,2,1,0
    rows = _compute_horizon_targets(vals, [1, 6])
    # k=1: covers plies_to_end<=1 → i in {2,3}; k=6: covers all
    assert rows[0] == [0.0, 1.0]
    assert rows[1] == [0.0, -1.0]
    assert rows[2] == [1.0, 1.0]
    assert rows[3] == [-1.0, -1.0]


def test_preserves_per_position_sign():
    # value alternates with side-to-move; horizon target keeps each row's own sign
    vals = [1.0, -1.0]
    rows = _compute_horizon_targets(vals, [6])
    assert rows == [[1.0], [-1.0]]
