"""TDD tests for the fully-forcing (VCF) solver prototype.

Run: uv run --no-sync pytest scripts/test_forcing_search_prototype.py -q
"""
import forcing_search_prototype as fsp

WL = 6  # win_length


def _line(cells, player):
    return {c: player for c in cells}


# --- Tier 1: threat kernel -------------------------------------------------

def test_completions_four_run_open_both_sides_has_three_completions():
    # __XXXX__ along the q-axis: P1 at q=0..3, empties at -2,-1,4,5, no enemies.
    board = _line([(0, 0), (1, 0), (2, 0), (3, 0)], "P1")
    comps = fsp.completions(board, "P1", WL)
    assert comps == {
        frozenset({(-2, 0), (-1, 0)}),  # left
        frozenset({(-1, 0), (4, 0)}),   # straddle (the trap)
        frozenset({(4, 0), (5, 0)}),    # right
    }


def test_min_covers_four_run_open_both_sides_has_three_covers_B2():
    comps = {
        frozenset({(-2, 0), (-1, 0)}),
        frozenset({(-1, 0), (4, 0)}),
        frozenset({(4, 0), (5, 0)}),
    }
    B, covers = fsp.min_covers(comps)
    assert B == 2
    assert set(covers) == {
        frozenset({(-2, 0), (4, 0)}),
        frozenset({(-1, 0), (5, 0)}),
        frozenset({(-1, 0), (4, 0)}),
    }
    # the naive symmetric block misses the straddle and must NOT be a cover
    assert frozenset({(-2, 0), (5, 0)}) not in set(covers)


def test_five_run_open_both_ends_unique_two_end_cover_B2():
    # _XXXXX_ : two single-cell completions force the cover to both ends.
    board = _line([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], "P1")
    B, covers = fsp.min_covers(fsp.completions(board, "P1", WL))
    assert B == 2
    assert covers == [frozenset({(-1, 0), (5, 0)})]  # unique: both ends


def test_walled_four_run_is_not_forcing_B1():
    # O_XXXX_O : enemy walls kill the left/right/straddle-pair windows, leaving one
    # straddle completion the defender blocks with a single stone.
    board = _line([(0, 0), (1, 0), (2, 0), (3, 0)], "P1")
    board[(-2, 0)] = "P2"
    board[(5, 0)] = "P2"
    B, _ = fsp.min_covers(fsp.completions(board, "P1", WL))
    assert B == 1


def test_one_gap_five_is_not_forcing_B1():
    # XXX_XX : every completion runs through the single gap → one blocking stone.
    board = _line([(0, 0), (1, 0), (2, 0), (4, 0), (5, 0)], "P1")
    B, covers = fsp.min_covers(fsp.completions(board, "P1", WL))
    assert B == 1
    assert covers == [frozenset({(3, 0)})]


# --- AND-OR search ---------------------------------------------------------

def test_mate_in_one_five_in_a_row_open():
    board = _line([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], "P1")  # _XXXXX_
    res = fsp.solve(board, "P1", win_length=WL)
    assert res.outcome == "WIN"
    assert res.depth == 1


def test_mate_in_two_double_four_fork():
    # Two 3-runs sharing (0,0) on the q- and r-axes. P1 plays (3,0)+(0,3) to make
    # a double open-four (B=4) the defender's two stones cannot cover; completes
    # next turn. No immediate completion at the root, so the minimum is 2.
    board = _line([(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)], "P1")
    res = fsp.solve(board, "P1", win_length=WL)
    assert res.outcome == "WIN"
    assert res.depth == 2


def test_no_forcing_win_from_scattered_stones():
    board = _line([(0, 0), (10, 10)], "P1")  # nothing alignable within reach
    res = fsp.solve(board, "P1", win_length=WL)
    assert res.outcome == "NO"


# --- Rendering: analysis-board #c= codec (port of static/shared.js) ---------

def test_encode_moves_compact_hand_vector():
    # moves include the (0,0) seed, which is dropped. (1,0) -> zig (2,0) -> varint
    # bytes [2,0] -> base64url 'AgA'.
    assert fsp.encode_moves_compact([(0, 0), (1, 0)]) == "AgA"


def test_encode_moves_compact_round_trips():
    moves = [(0, 0), (1, 0), (-3, 4), (127, -128), (0, 5)]
    enc = fsp.encode_moves_compact(moves)
    assert fsp.decode_moves_compact(enc) == moves[1:]  # seed dropped


def test_pns_agrees_with_search_mate_in_one():
    board = _line([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], "P1")
    assert fsp.solve_pns(board, "P1", win_length=WL).outcome == "WIN"


def test_pns_agrees_with_search_mate_in_two():
    board = _line([(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)], "P1")
    assert fsp.solve_pns(board, "P1", win_length=WL).outcome == "WIN"


def test_pns_agrees_with_search_no_forcing_win():
    board = _line([(0, 0), (10, 10)], "P1")
    assert fsp.solve_pns(board, "P1", win_length=WL).outcome == "NO"


def test_dfpn_agrees_with_search_mate_in_one():
    board = _line([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], "P1")
    assert fsp.solve_dfpn(board, "P1", win_length=WL).outcome == "WIN"


def test_dfpn_agrees_with_search_mate_in_two():
    board = _line([(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)], "P1")
    assert fsp.solve_dfpn(board, "P1", win_length=WL).outcome == "WIN"


def test_dfpn_agrees_with_search_no_forcing_win():
    board = _line([(0, 0), (10, 10)], "P1")
    assert fsp.solve_dfpn(board, "P1", win_length=WL).outcome == "NO"


def test_principal_variation_plays_out_to_a_win():
    board = _line([(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)], "P1")  # mate-in-2 fork
    res, pv = fsp.principal_variation(board, "P1", win_length=WL)
    assert res.outcome == "WIN"
    b = dict(board)
    for player, cells in pv:
        for c in cells:
            b[c] = player
    # after playing the line out, the attacker has WL in a row on some axis
    won = any(all(b.get((q + j * dq, r + j * dr)) == "P1" for j in range(WL))
              for (q, r), o in b.items() if o == "P1" for (dq, dr) in fsp.WIN_AXES)
    assert won
