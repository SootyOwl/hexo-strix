"""Pure unit tests for analysis helpers that need no checkpoint.

Covers ``end_of_turn_indices``, ``_scan_threats`` (owner labelling under both
absolute and relative stone encodings), and NaN/Inf sanitisation of
``AnalysisResult.to_json`` — all exercisable without a model or GPU.
"""
import json
from types import SimpleNamespace

import torch

from hexo_a0.serving.analysis import (
    AnalysisResult,
    _scan_threats,
    classify_turn_quality,
    end_of_turn_indices,
)


def test_end_of_turn_indices_pure():
    cps = ["P2", "P2", "P1", "P1", "P2", "P2"]
    assert end_of_turn_indices(cps) == [0, 2, 4, 5]


def test_end_of_turn_indices_single():
    assert end_of_turn_indices(["P2"]) == [0]


def test_end_of_turn_indices_no_change():
    assert end_of_turn_indices(["P1", "P1", "P1"]) == [0, 2]


def test_end_of_turn_indices_empty_is_safe():
    # Must not raise IndexError; an empty move list yields no boundaries.
    assert end_of_turn_indices([]) == []


# --- _scan_threats owner labelling ----------------------------------------

def _fake_graph(stones, legal):
    """Build a minimal stand-in for a PyG graph: .x, .stone_mask, .coords.

    stones: list of ((q,r), feature_col) where feature_col in {0,1}.
    legal:  list of (q,r) legal extensions.
    """
    coords = [list(c) for c, _ in stones] + [list(c) for c in legal]
    x = torch.zeros((len(coords), 2))
    stone_mask = torch.zeros((len(coords),), dtype=torch.bool)
    for i, (_, col) in enumerate(stones):
        x[i, col] = 1.0
        stone_mask[i] = True
    return SimpleNamespace(x=x, stone_mask=stone_mask, coords=torch.tensor(coords))


def test_scan_threats_absolute_owner_labels():
    # Three P1 stones along axis (1,0); legal extensions at both ends.
    data = _fake_graph([
        ((0, 0), 0), ((1, 0), 0), ((2, 0), 0),
    ], legal=[(-1, 0), (3, 0)])
    threats = _scan_threats(data, [(-1, 0), (3, 0)], [0.1, 0.2],
                             current_player="P2", relative_stones=False)
    assert len(threats) == 1
    assert threats[0].owner == "P1"
    assert threats[0].length == 3


def test_scan_threats_relative_owner_uses_current_player():
    # Under relative encoding col0 = side-to-move. Current player P2 -> those
    # stones belong to P2, not P1.
    data = _fake_graph([
        ((0, 0), 0), ((1, 0), 0), ((2, 0), 0),
    ], legal=[(-1, 0), (3, 0)])
    threats = _scan_threats(data, [(-1, 0), (3, 0)], [0.1, 0.2],
                             current_player="P2", relative_stones=True)
    assert len(threats) == 1
    assert threats[0].owner == "P2"

    # And col1 stones (opponent) belong to the opposite player.
    data2 = _fake_graph([
        ((0, 0), 1), ((1, 0), 1), ((2, 0), 1),
    ], legal=[(-1, 0), (3, 0)])
    threats2 = _scan_threats(data2, [(-1, 0), (3, 0)], [0.1, 0.2],
                             current_player="P2", relative_stones=True)
    assert threats2[0].owner == "P1"


def test_scan_threats_short_probs_does_not_indexerror():
    # probs shorter than legal_list (e.g. empty MCTS visit_counts) must not
    # raise IndexError; the out-of-range extension falls back to prob 0.0.
    data = _fake_graph([
        ((0, 0), 0), ((1, 0), 0), ((2, 0), 0),
    ], legal=[(-1, 0), (3, 0)])
    threats = _scan_threats(data, [(-1, 0), (3, 0)], [0.5],
                             current_player="P2", relative_stones=False)
    assert len(threats) == 1
    probs_seen = [e["prob"] for e in threats[0].extensions]
    assert 0.0 in probs_seen  # the idx>=1 extension fell back to 0.0


def test_scan_threats_no_threat_for_scattered_stones():
    data = _fake_graph([((0, 0), 0), ((5, 5), 0)], legal=[(-1, 0), (6, 5)])
    threats = _scan_threats(data, [(-1, 0), (6, 5)], [0.1, 0.2],
                             current_player="P1", relative_stones=False)
    assert threats == []


# --- AnalysisResult NaN/Inf sanitisation ----------------------------------

def test_to_json_sanitizes_nan_and_inf():
    r = AnalysisResult(
        legal=[[0, 0]],
        value=float("nan"),
        probs=[float("nan"), float("inf"), 0.5],
        q_hat=[float("-inf")],
        improved_policy=[float("nan")],
    )
    js = r.to_json()
    # allow_nan=False raises ValueError on any NaN/Infinity token, so a clean
    # pass proves the values were sanitised to finite JSON numbers.
    s = json.dumps(js, allow_nan=False)
    assert "NaN" not in s and "Infinity" not in s


# --- classify_turn_quality (order-independent, end-of-turn-gated) -------------
# Pure: no model. A turn is judged by its placed-SET vs the engine's best
# placed-set (its greedy policy line from the turn start), so intra-turn ORDER
# never matters: [A,B] and [B,A] reach the same board and get the same verdict.
# The engine's choice at each placement is argmax improved_policy (what it plays);
# q_hat sizes the loss between the two full turns' END positions. When the engine's
# top pick wasn't played first, its second pick is off the game line and an
# `eval_after_fn(prefix_index, first_move) -> mcts_json` supplies that position.

def _pos(cp, legal, ip, qh, cand=None):
    """A position BEFORE a placement, with the MCTS fields the classifier reads."""
    return {"current_player": cp, "value": 0.0, "terminal": False, "winner": None,
            "legal": legal, "improved_policy": ip, "q_hat": qh,
            "candidate_set": cand if cand is not None else [True] * len(legal)}


def _post(cp):
    """A turn-end entry (no MCTS fields; the loop stops before it)."""
    return {"current_player": cp, "value": 0.0, "terminal": False, "winner": None}


def test_classify_non_boundary_returns_none():
    traj = [_pos("P1", [[1, 0]], [1.0], [0.5]), _post("P2"), _post("P2")]
    out = classify_turn_quality(traj, [0, 2], [(0, 0), (1, 0), (2, 0)],
                                turn_end_idx=1, prev_boundary_idx=0)
    assert out is None


def test_classify_matched_policy_is_best():
    # Player plays argmax improved_policy at BOTH placements -> best.
    traj = [
        _pos("P1", [[1, 0], [2, 0]], [0.8, 0.2], [0.5, 0.3]),   # choice [1,0]
        _pos("P1", [[3, 0], [9, 9]], [0.9, 0.1], [0.4, 0.2]),   # choice [3,0]
        _post("P2"),
    ]
    q = classify_turn_quality(traj, [0, 2], [(0, 0), (1, 0), (3, 0)],
                              turn_end_idx=2, prev_boundary_idx=0)
    assert q["label"] == "best"


def test_classify_policy_choice_beats_higher_qhat():
    # The engine's policy choice [6,0] has LOWER q_hat than [2,1] (the value head
    # is blind to the threat [6,0] blocks). Playing [6,0] — what the engine plays
    # — is BEST, never a mistake.
    traj = [
        _pos("P1", [[1, 0], [2, 0]], [0.8, 0.2], [0.1, 0.0]),        # choice [1,0]
        _pos("P1", [[6, 0], [2, 1]], [0.7, 0.3], [-0.07, -0.06]),   # choice [6,0] (lower q!)
        _post("P2"),
    ]
    assert classify_turn_quality(traj, [0, 2], [(0, 0), (1, 0), (6, 0)],
                                 turn_end_idx=2, prev_boundary_idx=0)["label"] == "best"
    # Playing the higher-q_hat [2,1] leaves the policy line, but q_hat doesn't see
    # it as worse (loss <= 0) -> "good", never a false mistake.
    assert classify_turn_quality(traj, [0, 2], [(0, 0), (1, 0), (2, 1)],
                                 turn_end_idx=2, prev_boundary_idx=0)["label"] == "good"


def test_classify_divergence_loss_buckets():
    # Diverge at the 2nd placement; loss = q_hat[choice] - q_hat[played] (same
    # position) sizes the label.
    def traj_for(qh_played):
        return [
            _pos("P1", [[1, 0], [2, 0]], [0.8, 0.2], [0.1, 0.0]),         # choice [1,0]
            _pos("P1", [[3, 0], [9, 9]], [0.9, 0.1], [0.5, qh_played]),   # choice [3,0], played [9,9]
            _post("P2"),
        ]
    moves = [(0, 0), (1, 0), (9, 9)]
    assert classify_turn_quality(traj_for(0.40), [0, 2], moves, turn_end_idx=2, prev_boundary_idx=0)["label"] == "good"      # loss 0.10
    assert classify_turn_quality(traj_for(0.25), [0, 2], moves, turn_end_idx=2, prev_boundary_idx=0)["label"] == "mistake"   # loss 0.25
    assert classify_turn_quality(traj_for(0.00), [0, 2], moves, turn_end_idx=2, prev_boundary_idx=0)["label"] == "blunder"   # loss 0.50


def test_classify_reversed_order_is_best():
    # The player plays the engine's pair in REVERSED order: engine's top pick at the
    # turn start is [3,0] (played SECOND), so its second pick is off the game line.
    # eval_after_fn supplies the position after [3,0], where the engine wants [1,0]
    # (played FIRST). Same board -> "best", not a blunder for the reordering.
    traj = [
        _pos("P1", [[1, 0], [3, 0]], [0.2, 0.8], [0.0, 0.5]),   # top pick [3,0]
        _pos("P1", [[3, 0], [9, 9]], [0.9, 0.1], [0.5, 0.2]),   # after player's [1,0], wants [3,0]
        _post("P2"),
    ]
    # After the engine's own first pick [3,0], the engine wants [1,0].
    after = {"current_player": "P1", "legal": [[1, 0], [9, 9]],
             "improved_policy": [0.9, 0.1], "q_hat": [0.5, 0.2],
             "candidate_set": [True, True]}
    q = classify_turn_quality(traj, [0, 2], [(0, 0), (1, 0), (3, 0)],
                              turn_end_idx=2, prev_boundary_idx=0,
                              eval_after_fn=lambda idx, mv: after)
    assert q["label"] == "best" and q["matched"] is True
    assert sorted(map(tuple, q["engine_pair"])) == [(1, 0), (3, 0)]


def test_classify_unknown_engine_second_pick_returns_none():
    # Engine's top pick [1,0] wasn't played first and NO eval_after_fn is given, so
    # its end-of-turn board is unknown -> don't guess (return None).
    traj = [
        _pos("P1", [[1, 0], [4, -1]], [0.7, 0.3], [0.5, 0.0]),   # top pick [1,0], played [4,-1]
        _pos("P1", [[3, 0], [9, 9]], [0.9, 0.1], [0.4, 0.2]),
        _post("P2"),
    ]
    q = classify_turn_quality(traj, [0, 2], [(0, 0), (4, -1), (3, 0)],
                              turn_end_idx=2, prev_boundary_idx=0)
    assert q is None


def test_classify_p2_mover():
    # Mover = P2; q_hat is mover-perspective and the comparison is same-position,
    # so the logic is sign-agnostic.
    traj = [
        _pos("P2", [[1, 0], [2, 0]], [0.8, 0.2], [0.4, 0.3]),   # choice [1,0]
        _pos("P2", [[3, 0], [9, 9]], [0.9, 0.1], [0.5, 0.0]),   # choice [3,0], played [9,9] -> loss .5
        _post("P1"),
    ]
    q = classify_turn_quality(traj, [0, 2], [(0, 0), (1, 0), (9, 9)],
                              turn_end_idx=2, prev_boundary_idx=0)
    assert q["label"] == "blunder"


def test_classify_missing_mcts_data_returns_none():
    traj = [_post("P1"), _post("P1"), _post("P2")]
    q = classify_turn_quality(traj, [0, 2], [(0, 0), (1, 0), (2, 0)],
                              turn_end_idx=2, prev_boundary_idx=0)
    assert q is None