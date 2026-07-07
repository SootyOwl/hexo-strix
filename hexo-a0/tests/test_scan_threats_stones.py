"""Tests for the torch-free ``_scan_threats_from_state`` twin.

Verifies it shares the axis-walk logic with the torch-backed ``_scan_threats``
(both now delegate to ``_threats_from_stone_sets``) and produces equivalent
threat detections when bucketing stones directly from
``hexo_rs.GameState.placed_stones()`` instead of a graph tensor.
"""
import hexo_rs

from hexo_a0.serving.analysis import _scan_threats_from_state


def _state_with_open_three():
    """P1 holds an open 3-in-a-row (0,0)-(1,0)-(2,0); P2 to move.

    Both extension coords, (3, 0) and (-1, 0), are unoccupied and legal.
    """
    cfg = hexo_rs.GameConfig(6, 8, 200)
    return hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P1"), ((2, 0), "P1")],
        "P2", 2, cfg,
    )


def test_scan_threats_from_state_absolute():
    state = _state_with_open_three()
    current_player = state.current_player()
    assert current_player == "P2"
    legal_coords = [list(c) for c in state.legal_moves()]
    probs = [1.0 / len(legal_coords)] * len(legal_coords)

    threats = _scan_threats_from_state(
        state, legal_coords, probs,
        current_player=current_player, relative_stones=False,
    )

    long_threats = [t for t in threats if t.length >= 3]
    assert long_threats, "expected an open 3-in-a-row threat"
    for t in long_threats:
        assert t.owner == "P1"
        assert t.start == (0, 0)
        assert t.axis == (1, 0)
        assert t.extensions
        for ext in t.extensions:
            assert ext["coord"] in legal_coords


def test_scan_threats_from_state_relative_flips():
    state = _state_with_open_three()
    current_player = state.current_player()  # "P2"
    legal_coords = [list(c) for c in state.legal_moves()]
    probs = [1.0 / len(legal_coords)] * len(legal_coords)

    abs_threats = _scan_threats_from_state(
        state, legal_coords, probs,
        current_player=current_player, relative_stones=False,
    )
    rel_threats = _scan_threats_from_state(
        state, legal_coords, probs,
        current_player=current_player, relative_stones=True,
    )

    abs_starts = {t.start for t in abs_threats}
    rel_starts = {t.start for t in rel_threats}
    assert abs_starts == rel_starts
    assert abs_starts, "expected at least one threat on this board"

    # own_abs (the criterion used to bucket a stone into the "own" label) is
    # always defined so that own_abs == own: absolute fixes both to "P1";
    # relative sets both to current_player. So a stone matching own_abs is
    # always labelled with its own true absolute identity either way — the
    # state-based scan already knows physical ownership from
    # placed_stones(), so relative_stones only changes *which* string
    # ("current_player" vs the literal "P1") is used as the own/opp
    # criterion, not the reported owner. Verify that invariant explicitly,
    # using the mapping rule from the docstring as the expected formula.
    other = "P2" if current_player == "P1" else "P1"
    abs_by_start = {t.start: t.owner for t in abs_threats}
    rel_by_start = {t.start: t.owner for t in rel_threats}
    assert abs_by_start[(0, 0)] == "P1"  # sanity: known board owner
    for start, abs_owner in abs_by_start.items():
        expected_rel_owner = current_player if abs_owner == current_player else other
        assert rel_by_start[start] == expected_rel_owner
        # On this board abs_owner ("P1") != current_player ("P2"), so
        # expected_rel_owner resolves to `other` == "P1" == abs_owner: the
        # label is provably invariant here, which we assert directly too.
        assert rel_by_start[start] == abs_owner
