"""Tests for the analysis core.

Checkpoint-gated; skips gracefully if the champion checkpoint is absent.
"""
import os

import pytest

from hexo_a0.serving.model import load_model
from hexo_a0.serving.analysis import (
    analyze_position, analyze_trajectory, analyze_game_full, build_state,
    AnalysisResult,
)


CKPT = "runs/gine-mini-4l-128p32v-jkcat/checkpoints/self_play/champion.pt"
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="champion ckpt absent")


def _state():
    return build_state([(0, 0), (1, 0), (2, 0)], 6, 8, 400)


def test_build_state_replays_and_skips_seed():
    st = build_state([(0, 0), (1, 0), (2, 0)], 6, 8, 400)
    assert st.current_player() == "P1"  # P2 placed, now P1's turn
    stones = {tuple(c): p for c, p in st.placed_stones()}
    assert stones[(0, 0)] == "P1"
    assert stones[(1, 0)] == "P2"
    assert stones[(2, 0)] == "P2"


def test_raw_analyze_no_qhat():
    model, mc, _ = load_model(CKPT, "cpu")
    r = analyze_position(model, mc, _state(), mcts_sims=0)
    assert isinstance(r, AnalysisResult)
    assert -1.0 <= r.value <= 1.0
    assert len(r.legal) == len(r.probs)
    assert r.q_hat is None and r.candidate_set is None
    assert r.improved_policy is None
    assert "candidate_set" in r.to_json()


def test_mcts_qhat_and_visited_gate():
    model, mc, _ = load_model(CKPT, "cpu")
    r = analyze_position(model, mc, _state(), mcts_sims=64, mcts_m_actions=16)
    assert r.q_hat is not None and r.improved_policy is not None
    assert r.candidate_set is not None
    assert len(r.candidate_set) == len(r.legal)
    assert all(isinstance(b, bool) for b in r.candidate_set)
    assert sum(r.candidate_set) >= 1


def test_terminal_position():
    model, mc, _ = load_model(CKPT, "cpu")
    # Build a known P1 win along axis (1,0): seed (0,0) + five more P1 stones.
    # Turn order after seed: P2 places twice, then P1 places twice, repeating.
    moves = [
        (0, 0),   # P1 seed
        (0, 5), (1, 5),   # P2
        (1, 0), (2, 0),   # P1
        (2, 5), (0, 6),   # P2
        (3, 0), (4, 0),   # P1
        (1, 6), (2, 6),   # P2
        (5, 0),           # P1 wins with (0,0)-(5,0) on axis (1,0)
    ]
    st = build_state(moves, 6, 8, 400)
    assert st.is_terminal() and st.winner() == "P1"
    r = analyze_position(model, mc, st, mcts_sims=0)
    assert r.terminal
    assert r.winner == "P1"
    assert r.legal == []


def test_trajectory_one_entry_per_prefix_and_boundaries():
    model, mc, _ = load_model(CKPT, "cpu")
    out = analyze_trajectory(
        model, mc, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], 6, 8, 400
    )
    assert len(out["trajectory"]) == 5
    assert out["boundary_indices"][0] == 0
    assert out["boundary_indices"][-1] == 4
    assert out["evaluated_prefixes"] == 5


def test_analyze_position_emits_stones_matching_state():
    model, mc, _ = load_model(CKPT, "cpu")
    moves = [(0, 0), (1, 0), (2, 0), (3, 0)]  # seed + 3 plies -> 4 stones
    st = build_state(moves, 6, 8, 400)
    r = analyze_position(model, mc, st, mcts_sims=0)
    js = r.to_json()
    assert "stones" in js and js["stones"], "analysis board needs stones to render the position"
    placed = {tuple(c): p for c, p in st.placed_stones()}
    rendered = {tuple(s[0]): s[1] for s in js["stones"]}
    assert rendered == placed


def test_terminal_analysis_emits_stones():
    model, mc, _ = load_model(CKPT, "cpu")
    moves = [
        (0, 0), (0, 5), (1, 5), (1, 0), (2, 0), (2, 5), (0, 6),
        (3, 0), (4, 0), (1, 6), (2, 6), (5, 0),
    ]
    st = build_state(moves, 6, 8, 400)
    assert st.is_terminal()
    r = analyze_position(model, mc, st, mcts_sims=0)
    js = r.to_json()
    assert "stones" in js and len(js["stones"]) >= 6


def test_analyze_trajectory_keeps_non_seed_first_move():
    # No leading (0,0) seed: moves[0]=[1,0] must be applied, not silently
    # dropped. The old code unconditionally skipped moves[0] (off-by-one vs
    # build_state), so this returned 3 prefixes instead of 4.
    model, mc, _ = load_model(CKPT, "cpu")
    out = analyze_trajectory(model, mc, [(1, 0), (2, 0), (3, 0)], 6, 8, 400)
    assert out["evaluated_prefixes"] == 4  # engine seed + 3 applied moves
    assert out.get("error_at") is None


def test_analyze_trajectory_terminal_value_sign():
    # P1 completes six-in-a-row with the final (5,0). The side to move at the
    # terminal prefix is P2 (opponent of the mover), who lost → value -1.0.
    # The old parity formula was inverted and reported +1.0 here.
    model, mc, _ = load_model(CKPT, "cpu")
    moves = [(0, 0), (0, 5), (1, 5), (1, 0), (2, 0), (2, 5), (0, 6),
             (3, 0), (4, 0), (1, 6), (2, 6), (5, 0)]
    out = analyze_trajectory(model, mc, moves, 6, 8, 400)
    last = out["trajectory"][-1]
    assert last["terminal"] and last["winner"] == "P1"
    assert last["value"] == -1.0


def test_analyze_position_legal_aligns_with_state_legal_moves():
    # The MCTS branch must return `legal` in state.legal_moves() order (the
    # order probs/q_hat/candidate_set are indexed in), not graph-builder order,
    # or the frontend mis-pairs moves with their Q/prob.
    model, mc, _ = load_model(CKPT, "cpu")
    st = _state()
    r = analyze_position(model, mc, st, mcts_sims=16, mcts_m_actions=16)
    expected = [list(c) for c in st.legal_moves()]
    assert r.legal == expected
    assert len(r.q_hat) == len(expected)
    assert len(r.candidate_set) == len(expected)
    assert len(r.probs) == len(expected)


def test_analyze_trajectory_sanitizes_nan_values():
    # A NaN/Inf value-head output must be coerced to a finite number, not emitted
    # as a bare `NaN`/`Infinity` token (invalid RFC 8259 JSON that breaks the
    # browser analysis view).
    import json as _json
    import torch as _t

    class FakeModel:
        def forward_batch(self, batch):
            n = batch.num_graphs
            return [_t.zeros(1)] * n, _t.full((n,), float("nan"))

    _, mc, _ = load_model(CKPT, "cpu")
    out = analyze_trajectory(FakeModel(), mc, [(0, 0), (1, 0), (2, 0), (3, 0)], 6, 8, 400)
    s = _json.dumps(out, allow_nan=False)  # raises if any NaN/Infinity present
    assert "NaN" not in s and "Infinity" not in s
    for entry in out["trajectory"]:
        if not entry["terminal"]:
            assert entry["value"] == 0.0  # NaN coerced to 0.0


def test_trajectory_carries_current_player_for_p1_perspective_flip():
    # The frontend eval bar is drawn from P1's fixed perspective (not the
    # side-to-move perspective the value head natively emits), so it needs
    # current_player on every trajectory entry to know when to flip the sign.
    # This pins the contract: every entry has both `value` and `current_player`.
    model, mc, _ = load_model(CKPT, "cpu")
    out = analyze_trajectory(model, mc, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], 6, 8, 400)
    for e in out["trajectory"]:
        assert "value" in e and "current_player" in e
        assert e["current_player"] in ("P1", "P2")
    # The flip rule (mirrored in JS p1Perspective): P2-to-move values negate.
    def p1(v, cp):
        return -v if cp == "P2" else v
    # Two adjacent non-terminal entries from opposite sides should produce a
    # P1-perspective series that doesn't trivially oscillate in sign by rule
    # (sanity: at least the transform is defined and finite for every entry).
    assert all(-1.0 <= p1(e["value"], e["current_player"]) <= 1.0
               for e in out["trajectory"])


def test_classify_move_quality_thresholds():
    # Mirror of the JS classifyMoveQuality decision rule (without the "best"
    # branch, which needs the pre-move /analyze cache). Pins the eval-swing
    # buckets so a future threshold tweak is deliberate.
    def p1(v, cp):
        return -v if cp == "P2" else v

    def classify(pre, post, played_by):
        # moverDelta = how the move changed eval from the MOVER's perspective.
        preV, postV = p1(pre["value"], pre["current_player"]), p1(post["value"], post["current_player"])
        d = postV - preV if played_by == "P1" else preV - postV
        if d <= -0.40:
            return "blunder"
        if d <= -0.15:
            return "mistake"
        return "good"

    def entry(v, cp):
        return {"value": v, "current_player": cp}

    # NOTE: pre/post `value` is the side-to-move (value-head) perspective. The
    # mover-delta is computed in P1 perspective (p1Perspective: negate when
    # current_player=="P2"). After P1 moves it becomes P2's turn, so post has
    # current_player=="P2" and its P1-perspective value is -value. Pick pre/post
    # values so the P1-perspective delta lands in each bucket.
    # P1 moves, throws away ~0.5 (P1-persp) -> blunder (d=-0.5 <= -0.40).
    assert classify(entry(0.5, "P1"), entry(0.0, "P2"), "P1") == "blunder"
    # P1 moves, loses ~0.2 (P1-persp): pre=0.5, post P1-persp=0.3 -> post value -0.3.
    assert classify(entry(0.5, "P1"), entry(-0.3, "P2"), "P1") == "mistake"
    # P1 moves, holds: pre=0.5, post P1-persp=0.5 -> post value -0.5.
    assert classify(entry(0.5, "P1"), entry(-0.5, "P2"), "P1") == "good"
    # P2 moves and lets P1's eval jump ~0.5 -> blunder for P2.
    assert classify(entry(0.0, "P2"), entry(0.5, "P1"), "P2") == "blunder"
    # P2 moves and improves P2's lot (P1-persp drops 0.4) -> good for P2.
    #   pre P1-persp=0.2 -> value -0.2 (P2 to move); post P1-persp=-0.2 -> value -0.2 (P1 to move).
    assert classify(entry(-0.2, "P2"), entry(-0.2, "P1"), "P2") == "good"


# --- analyze_game_full (batched value+policy + looped MCTS) ----------------

_GAME = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


def test_analyze_game_full_shape():
    model, mc, _ = load_model(CKPT, "cpu")
    out = analyze_game_full(model, mc, _GAME, 6, 8, 400, mcts_sims=16, mcts_m_actions=16)
    assert out["evaluated_prefixes"] == len(out["trajectory"])
    assert out["mcts_sims"] == 16
    assert out["cancelled"] is False
    assert out["completed_prefixes"] == out["evaluated_prefixes"]
    for e in out["trajectory"]:
        assert "value" in e and "current_player" in e
        if not e["terminal"]:
            # raw policy present always; MCTS fields present after pass 2
            assert "legal" in e and "probs" in e and "stones" in e
            assert "q_hat" in e and "candidate_set" in e and "improved_policy" in e
            assert len(e["legal"]) == len(e["probs"])
            assert len(e["q_hat"]) == len(e["legal"])
            assert len(e["candidate_set"]) == len(e["legal"])


_WIN_GAME = [(0, 0), (0, 5), (1, 5), (1, 0), (2, 0), (2, 5), (0, 6),
             (3, 0), (4, 0), (1, 6), (2, 6), (5, 0)]  # P1 wins on (0,0)-(5,0)


def test_analyze_game_full_terminal_prefix_carries_stones():
    # The winning (terminal) prefix must ship stones + an (empty) legal list and
    # winner, so the analysis board renders the final position/win rather than
    # blanking (the frontend clears the SVG when boardResult.legal is missing).
    model, mc, _ = load_model(CKPT, "cpu")
    out = analyze_game_full(model, mc, _WIN_GAME, 6, 8, 400, mcts_sims=16)
    assert out["win_length"] == 6
    last = out["trajectory"][-1]
    assert last["terminal"] and last["winner"] == "P1"
    assert last["legal"] == []
    assert len(last["stones"]) >= 6


def test_analyze_game_full_progress_cb():
    model, mc, _ = load_model(CKPT, "cpu")
    calls = []
    out = analyze_game_full(model, mc, _GAME, 6, 8, 400, mcts_sims=16,
                            progress_cb=lambda done, total: calls.append((done, total)))
    total = out["evaluated_prefixes"]
    assert calls  # at least one progress callback
    assert calls[-1] == (total, total)
    assert [d for d, _ in calls] == list(range(1, total + 1))


def test_analyze_game_full_cancel_cb():
    model, mc, _ = load_model(CKPT, "cpu")
    state = {"n": 0}

    def cancel():
        state["n"] += 1
        return state["n"] > 3  # cancel after the 3rd prefix's check

    out = analyze_game_full(model, mc, _GAME, 6, 8, 400, mcts_sims=16, cancel_cb=cancel)
    assert out["cancelled"] is True
    assert out["completed_prefixes"] < out["evaluated_prefixes"]
    # Prefixes beyond the cancellation point keep raw policy but lack q_hat.
    assert any("q_hat" not in e for e in out["trajectory"] if not e["terminal"])


def test_analyze_game_full_quality_at_boundaries():
    model, mc, _ = load_model(CKPT, "cpu")
    out = analyze_game_full(model, mc, _GAME, 6, 8, 400, mcts_sims=16)
    # quality labels only appear at turn boundaries (not on every move).
    qual_idx = [i for i, e in enumerate(out["trajectory"]) if "quality" in e]
    assert all(i in out["boundary_indices"] for i in qual_idx)
    for i in qual_idx:
        assert out["trajectory"][i]["quality"]["label"] in ("best", "good", "mistake", "blunder")


def test_analyze_game_full_turn_verdict_is_order_independent():
    # The SAME turn played in either intra-turn order reaches the same board and
    # must get the SAME verdict: swapping P1's two placements (3,0)/(4,0) at the
    # turn ending index 4 leaves the end-of-turn position — and its verdict —
    # unchanged. (HeXO placements within a turn commute.)
    model, mc, _ = load_model(CKPT, "cpu")
    forward = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    swapped = [(0, 0), (1, 0), (2, 0), (4, 0), (3, 0)]
    a = analyze_game_full(model, mc, forward, 6, 8, 400, mcts_sims=32)
    b = analyze_game_full(model, mc, swapped, 6, 8, 400, mcts_sims=32)
    qa, qb = a["trajectory"][4].get("quality"), b["trajectory"][4].get("quality")
    assert qa is not None and qb is not None
    assert qa["label"] == qb["label"]
    assert qa["matched"] == qb["matched"]
    # Same resulting board -> same placed-set either way.
    assert sorted(map(tuple, qa["played_pair"])) == sorted(map(tuple, qb["played_pair"]))


def test_analyze_game_full_warm_start_skips_shared_head():
    # Extending a previously-analysed line should only run MCTS on the appended
    # tail; the shared head is copied from the warm trajectory verbatim.
    model, mc, _ = load_model(CKPT, "cpu")
    base = analyze_game_full(model, mc, _GAME, 6, 8, 400, mcts_sims=16)
    extended = _GAME + [(5, 0)]
    warm = base["trajectory"]
    calls = {"n": 0}

    def counting(st):
        calls["n"] += 1
        return analyze_position(model, mc, st, mcts_sims=16, mcts_m_actions=16)

    out = analyze_game_full(model, mc, extended, 6, 8, 400, mcts_sims=16,
                            run_mcts_fn=counting, warm_trajectory=warm)
    # MCTS runs only for positions NOT supplied (with q_hat) by the warm head.
    expected = sum(1 for i, e in enumerate(out["trajectory"])
                   if not e["terminal"]
                   and not (i < len(warm) and warm[i].get("q_hat") is not None))
    ext_nonterminal = sum(1 for e in out["trajectory"] if not e["terminal"])
    assert calls["n"] == expected
    assert calls["n"] < ext_nonterminal  # the shared head really was reused
    # Reused head positions carry exactly the warm q_hat.
    for i in range(len(warm)):
        if not out["trajectory"][i]["terminal"] and warm[i].get("q_hat") is not None:
            assert out["trajectory"][i]["q_hat"] == warm[i]["q_hat"]
