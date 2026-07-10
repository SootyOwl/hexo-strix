"""Tests for the analysis core.

Checkpoint-gated; skips gracefully if the champion checkpoint is absent.
"""
import os
import types

import pytest
import torch

from hexo_a0.serving.model import load_model
from hexo_a0.serving.analysis import (
    analyze_position, analyze_trajectory, analyze_game_full, build_state,
    classify_turn_quality, QUALITY_BLUNDER,
    AnalysisResult, ANALYSIS_DEPTH_CAP, ANALYSIS_NODE_BUDGET,
    TRAJECTORY_DEPTH_CAP, TRAJECTORY_NODE_BUDGET,
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


# --- classify_turn_quality: forced-win blunders + swing ---------------------
#
# classify_turn_quality labels a turn by the q-hat loss between the player's
# end-of-turn board and the engine's. q-hat routinely misses a forced loss the
# VCF solver catches, so a turn that hands the opponent a PROVEN forced win
# must override the loss-based label and read as "blunder" with a full -1.0
# eval (the user-facing swing then reflects the disaster, not the value head's
# optimistic q-hat). These build synthetic trajectories so the forcing/terminal
# fields can be set exactly.

def _qt_entry(cp, ip, qh, cs, lg, *, forcing=None, terminal=False, winner=None):
    e = {"current_player": cp, "improved_policy": ip, "q_hat": qh,
         "candidate_set": cs, "legal": lg}
    if forcing is not None:
        e["forcing"] = forcing
    if terminal:
        e["terminal"] = True
        e["winner"] = winner
        e["legal"] = []
        e["q_hat"] = None
    return e


def _qt_offline(qh_engine_end):
    """eval_after_fn stub: the position after the engine's first pick, with the
    engine's second pick carrying q-hat ``qh_engine_end`` (mover-perspective)."""
    return {"improved_policy": [0.1, 0.1, 0.9], "q_hat": [0.1, 0.1, qh_engine_end],
            "candidate_set": [1, 1, 1], "legal": [[5, 5], [7, 7], [8, 8]]}


# A turn where P1 misses a threat block: small q-hat loss (0.10, a "good" by
# q-hat alone) but the end-of-turn position is a PROVEN forced win for P2.
_QT_OPP_FORCED = {"winner": "P2", "attacker_is_mover": True,
                  "first_move": [5, 5], "pv": [[5, 5]], "pv_len": 1,
                  "pv_owners": ["P2"]}


def test_classify_opponent_forced_win_is_always_blunder():
    # P1's turn (prev=0, end=2). Engine ends at +0.25, player at +0.15 -> loss
    # 0.10, a "good" by q-hat. But trajectory[2] proves P2 has a forced win.
    traj = [
        _qt_entry("P1", [0.1, 0.8, 0.1], [0.1, 0.2, 0.1], [1, 1, 1],
                  [[5, 5], [6, 6], [7, 7]]),
        _qt_entry("P1", [0.5, 0.3, 0.2], [0.1, 0.15, 0.1], [1, 1, 1],
                  [[5, 5], [7, 7], [8, 8]]),
        _qt_entry("P2", [0.1, 0.1, 0.9], [0.1, 0.1, 0.25], [1, 1, 1],
                  [[5, 5], [7, 7], [8, 8]], forcing=_QT_OPP_FORCED),
    ]
    q = classify_turn_quality(traj, [0, 2], [[0, 0], [5, 5], [7, 7]],
                              turn_end_idx=2, prev_boundary_idx=0,
                              eval_after_fn=lambda a, e0: _qt_offline(0.25))
    assert q is not None
    assert q["label"] == "blunder", q
    assert q["forced_loss"] is True
    # The effective eval is -1.0 so the swing shows the full disaster, not the
    # value head's optimistic +0.15.
    assert q["player_end_q"] == -1.0
    assert q["engine_end_q"] == pytest.approx(0.25)
    # loss >= 1.0 (engine could even be slightly losing and it's still forced).
    assert q["loss"] >= 1.0
    # The displayed swing crosses the side boundary: -1.0 - 0.25 = -1.25.
    assert q["player_end_q"] - q["engine_end_q"] == pytest.approx(-1.25)


def test_classify_opponent_forced_win_overrides_match():
    # Same resulting board as the engine's line (matched) but that board is a
    # proven loss for the mover — still a blunder, never "best". One-placement
    # turn (played == engine's pick (5,5)); the end-of-turn entry at index 1 is
    # the opponent's to-move position with the proven forced win.
    traj = [
        _qt_entry("P1", [0.9, 0.1], [0.2, 0.1], [1, 1], [[5, 5], [7, 7]]),
        _qt_entry("P2", [0.1, 0.9], [0.1, 0.1], [1, 1], [[5, 5], [7, 7]],
                  forcing=_QT_OPP_FORCED),
    ]
    q = classify_turn_quality(traj, [0, 1], [[0, 0], [5, 5]],
                              turn_end_idx=1, prev_boundary_idx=0)
    assert q is not None
    assert q["matched"] is True           # same resulting board as the engine
    assert q["label"] == "blunder"        # but it's a proven loss -> blunder
    assert q["forced_loss"] is True
    assert q["player_end_q"] == -1.0


def test_classify_terminal_loss_is_blunder():
    # Mover self-inflicts an instant loss (double-four) on their own turn: the
    # end-of-turn position is terminal with the opponent winning.
    traj = [
        _qt_entry("P1", [0.9, 0.1], [0.2, 0.1], [1, 1], [[5, 5], [7, 7]]),
        _qt_entry("P2", [0.1, 0.9], [0.1, 0.1], [1, 1], [[5, 5], [7, 7]],
                  terminal=True, winner="P2"),
    ]
    q = classify_turn_quality(traj, [0, 1], [[0, 0], [5, 5]],
                              turn_end_idx=1, prev_boundary_idx=0)
    assert q is not None and q["label"] == "blunder"
    assert q["forced_loss"] is True
    assert q["player_end_q"] == -1.0


def test_classify_movers_own_forced_win_is_not_forced_blunder():
    # The MOVER had a forced win at turn-start (attacker_is_mover, winner==P1)
    # and the turn does NOT end in an opponent forced win. Squandering your own
    # win is the missed-win walk's job, not a forced blunder here.
    own_win = {"winner": "P1", "attacker_is_mover": True,
               "first_move": [6, 6], "pv": [[6, 6]], "pv_len": 1}
    traj = [
        _qt_entry("P1", [0.1, 0.8, 0.1], [0.1, 0.2, 0.1], [1, 1, 1],
                  [[5, 5], [6, 6], [7, 7]], forcing=own_win),
        _qt_entry("P1", [0.5, 0.3, 0.2], [0.1, 0.15, 0.1], [1, 1, 1],
                  [[5, 5], [7, 7], [8, 8]]),
        _qt_entry("P2", [0.1, 0.1, 0.9], [0.1, 0.1, 0.25], [1, 1, 1],
                  [[5, 5], [7, 7], [8, 8]]),  # no forcing: opponent has no win
    ]
    q = classify_turn_quality(traj, [0, 2], [[0, 0], [5, 5], [7, 7]],
                              turn_end_idx=2, prev_boundary_idx=0,
                              eval_after_fn=lambda a, e0: _qt_offline(0.25))
    assert q is not None
    assert q.get("forced_loss") is not True
    assert q["label"] != "blunder"  # loss 0.10 -> "good"; missed_win flags it elsewhere


def test_classify_swing_is_correct_across_side_switch():
    # No forcing field — pure q-hat swing. Engine favored the mover (+0.4),
    # player blundered to -0.5 (opponent favored): the eval crosses zero.
    # swing = player_end_q - engine_end_q must be -0.9 (mover-perspective),
    # correct in sign AND magnitude across the side switch.
    traj = [
        _qt_entry("P1", [0.1, 0.8, 0.1], [0.1, 0.4, 0.1], [1, 1, 1],
                  [[5, 5], [6, 6], [7, 7]]),
        _qt_entry("P1", [0.5, 0.3, 0.2], [0.1, -0.5, 0.1], [1, 1, 1],
                  [[5, 5], [7, 7], [8, 8]]),
        _qt_entry("P2", [0.1, 0.1, 0.9], [0.1, 0.1, 0.25], [1, 1, 1],
                  [[5, 5], [7, 7], [8, 8]]),
    ]
    q = classify_turn_quality(traj, [0, 2], [[0, 0], [5, 5], [7, 7]],
                              turn_end_idx=2, prev_boundary_idx=0,
                              eval_after_fn=lambda a, e0: _qt_offline(0.4))
    assert q is not None
    assert q["player_end_q"] == pytest.approx(-0.5)
    assert q["engine_end_q"] == pytest.approx(0.4)
    assert q["player_end_q"] - q["engine_end_q"] == pytest.approx(-0.9)
    assert q["label"] == "blunder"  # loss 0.9 >= QUALITY_BLUNDER
    assert q["loss"] >= QUALITY_BLUNDER


def test_analyze_game_full_forced_loss_quality_end_to_end():
    # End-to-end through analyze_game_full: P2's first turn ends at prefix index 2
    # (P1 to move). Inject a proven forced win for P1 at that position so the
    # quality loop sees the forcing field analyze_game_full put on the entry and
    # labels P2's turn a forced-loss blunder (verifying the field flows from the
    # per-prefix solve, through to_json, into classify_turn_quality, out to the
    # quality dict the frontend reads).
    model, mc, _ = load_model(CKPT, "cpu")
    injected = {"winner": "P1", "attacker_is_mover": True,
                "first_move": [3, 0], "pv": [[3, 0]], "pv_len": 1,
                "pv_owners": ["P1"]}
    counter = {"i": 0}

    def run_mcts_fn(st):
        r = analyze_position(model, mc, st, mcts_sims=16, mcts_m_actions=16)
        if counter["i"] == 2:        # P2's turn-end position (P1 to move)
            r.forcing = injected
        counter["i"] += 1
        return r

    out = analyze_game_full(model, mc, _GAME, 6, 8, 400, mcts_sims=16,
                            run_mcts_fn=run_mcts_fn)
    # P2's first turn is the first non-zero boundary; its mover is P2.
    b_p2 = next(b for b in out["boundary_indices"] if b != 0)
    assert out["trajectory"][0]["current_player"] == "P2"
    q = out["trajectory"][b_p2].get("quality")
    assert q is not None
    assert q["label"] == "blunder"
    assert q["forced_loss"] is True
    assert q["player_end_q"] == -1.0



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


# --- forcing-line display: both-side VCF solve (analyze_position) ----------
#
# These use a stubbed model/config rather than the real checkpoint (mirrors
# test_serving_game.py's live-play forcing tests): the forcing solve is pure
# hexo_rs and the display concern is independent of model quality, so a
# zero-logit stub keeps the fork constructions fast and deterministic.

class _ZeroLogitModel:
    """All-zero logits/value — analyze_position's raw-forward-pass branch
    (mcts_sims=0, the default) only needs __call__, never forward_batch."""

    def __call__(self, x, edge_index, legal_mask, stone_mask=None, edge_attr=None):
        return torch.zeros(x.shape[0]), torch.tensor(0.0)


def _forcing_fork_state(mover="P1"):
    """wl=4 fork: P1 holds an open pair, ``mover`` gets 2 free placements this
    turn (as in test_serving_game.py's test_forcing_bot_executes_own_forced_win).
    ``mover="P1"`` gives the mover itself the forced win; ``mover="P2"`` puts
    the (stoneless) opponent to move, so the forced win is on the FLIP side."""
    import hexo_rs as hr
    cfg = hr.GameConfig(4, 3, 60)
    state = hr.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P1")], mover, 2, cfg)
    return cfg, state


def _forcing_fork_state_midturn():
    """A wl=4 double-open-three fork where P1 has only 1 placement left this
    turn (``moves_remaining=1``), so the solved PV's first chunk is length 1,
    not the usual 2 — the shape review finding #1 flags as breaking the old
    pairs-of-2 parity heuristic. Stones (away from the engine's (0,0) P1 seed,
    which ``from_state`` always adds) form two open pairs crossing at (10,0);
    P1's one remaining placement fills the crossing cell, creating a double
    open three the opponent's one full turn (2 placements) can't fully block.
    Verified empirically: solve_forcing returns a 5-cell PV with owner chunks
    [1, 2, 2] (P1, P2, P2, P1, P1), not the [2, 2, ...] the old heuristic
    assumed.
    """
    import hexo_rs as hr
    cfg = hr.GameConfig(4, 3, 60)
    state = hr.GameState.from_state(
        [((9, 0), "P1"), ((11, 0), "P1"), ((10, -1), "P1"), ((10, 1), "P1")],
        "P1", 1, cfg)
    return cfg, state


def test_analyze_position_reports_movers_forced_win():
    cfg, state = _forcing_fork_state("P1")
    mc = types.SimpleNamespace(graph_type="hex")
    r = analyze_position(_ZeroLogitModel(), mc, state)
    assert r.forcing is not None
    assert r.forcing["winner"] == state.current_player() == "P1"
    assert r.forcing["attacker_is_mover"] is True
    legal = {tuple(c) for c in state.legal_moves()}
    assert tuple(r.forcing["first_move"]) in legal
    js = r.to_json()
    pv = js["forcing"]["pv"]
    assert pv and all(isinstance(m, list) and len(m) == 2 for m in pv)
    assert js["forcing"]["pv_len"] == len(pv)


def test_analyze_position_reports_opponents_threat():
    # Same stones, but P2 (with no stones of its own) is to move: P2 has no
    # win of its own, so the solve falls back to the opponent's
    # perspective-flipped THREAT (hr.solve_threat) — reported with
    # attacker_is_mover=False so the client can gate it behind the
    # default-off "Threats" toggle instead of overstating it as a win.
    cfg, state = _forcing_fork_state("P2")
    mc = types.SimpleNamespace(graph_type="hex")
    r = analyze_position(_ZeroLogitModel(), mc, state)
    assert r.forcing is not None
    assert r.forcing["winner"] == "P1"
    assert r.forcing["attacker_is_mover"] is False
    # pv_owners must reflect the FLIPPED replay (P1 moves first in the line).
    assert r.forcing["pv_owners"][0] == "P1"


def test_analyze_position_no_forcing_is_none():
    state = build_state([(0, 0), (1, 0), (2, 0)], 6, 8, 400)  # no forced win either side
    mc = types.SimpleNamespace(graph_type="hex")
    r = analyze_position(_ZeroLogitModel(), mc, state)
    assert r.forcing is None


def test_analyze_position_forcing_never_crashes_analysis(monkeypatch):
    import hexo_rs as hr
    cfg, state = _forcing_fork_state("P1")
    mc = types.SimpleNamespace(graph_type="hex")

    def _boom(*a, **k):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(hr, "solve_forcing", _boom)
    r = analyze_position(_ZeroLogitModel(), mc, state)
    assert r.forcing is None  # swallowed, not propagated
    assert r.legal  # the rest of the analysis still completed


def test_analyze_position_pv_owners_variable_length_first_chunk():
    # Review finding #1: the PV's chunk lengths are NOT a fixed pairs-of-2
    # cadence — the first chunk is however many placements the analyzed side
    # has left THIS turn (1 here, not 2), so a naive parity-of-2 heuristic
    # mislabels everything after it. pv_owners must come from a real replay.
    cfg, state = _forcing_fork_state_midturn()
    mc = types.SimpleNamespace(graph_type="hex")
    r = analyze_position(_ZeroLogitModel(), mc, state)
    assert r.forcing is not None
    js = r.to_json()["forcing"]
    pv, owners = js["pv"], js["pv_owners"]
    assert owners is not None
    assert len(owners) == len(pv)
    assert owners[0] == js["winner"]  # the analyzed (mover) side wins here

    # Hand-replay via hexo_rs directly, independent of _solve_forcing's own
    # replay, to confirm the reported owners match the engine's real turn
    # order (not just internally self-consistent).
    replay = state.clone()
    expected_owners = []
    for (q, r_) in pv:
        expected_owners.append(replay.current_player())
        replay.apply_move(q, r_)
    assert owners == expected_owners
    # This fixture's PV has a length-1 first chunk (not 2) — the input shape
    # the old pairs-of-2 heuristic mislabelled from index 1 onward.
    assert owners[0] != owners[1]


def test_analyze_game_full_forcing_present_without_mcts():
    # Review finding #3: `forcing` is independent of MCTS (computed even at
    # mcts_sims=0), but was previously only copied into the trajectory inside
    # the `q_hat is not None` gate — silently dropping it whenever MCTS
    # produced no q_hat (e.g. mcts_sims=0). It must be present regardless.
    model, mc, _ = load_model(CKPT, "cpu")
    out = analyze_game_full(model, mc, _GAME, 6, 8, 400, mcts_sims=0)
    non_terminal = [e for e in out["trajectory"] if not e["terminal"]]
    assert non_terminal  # sanity: the fixture has non-terminal prefixes
    assert all("forcing" in e for e in non_terminal)
    assert all(e.get("q_hat") is None for e in non_terminal)  # confirms sims=0 path


def test_analyze_position_defaults_to_analysis_budget(monkeypatch):
    import hexo_rs as hr
    cfg, state = _forcing_fork_state("P1")
    mc = types.SimpleNamespace(graph_type="hex")
    seen = []

    def _spy(st, depth_cap, node_budget):
        seen.append((depth_cap, node_budget))
        return None

    monkeypatch.setattr(hr, "solve_forcing", _spy)
    analyze_position(_ZeroLogitModel(), mc, state)
    assert seen and seen[0] == (ANALYSIS_DEPTH_CAP, ANALYSIS_NODE_BUDGET)


def test_solve_forcing_requests_wide_and_labels_result(monkeypatch):
    # Analysis always asks for the wide (threat + quiet-builder) generator;
    # the result records which generator actually ran.
    import hexo_rs as hr
    from hexo_a0.serving.analysis import _solve_forcing
    cfg, state = _forcing_fork_state("P1")
    seen = []

    def _spy(st, depth_cap, node_budget, wide=False):
        seen.append(wide)
        return ((9, 0), [(9, 0), (10, -1)])

    monkeypatch.setattr(hr, "solve_forcing", _spy)
    out = _solve_forcing(state)
    assert seen == [True]
    assert out is not None and out["wide"] is True


def test_solve_forcing_tight_fallback_on_old_extension(monkeypatch):
    # An extension predating the `wide` kwarg raises TypeError at call time;
    # _solve_forcing must fall back to the tight signature (same budgets) and
    # label the result wide=False instead of failing the whole analysis.
    import hexo_rs as hr
    from hexo_a0.serving.analysis import _solve_forcing
    cfg, state = _forcing_fork_state("P1")
    calls = []

    def _old_ext(st, depth_cap, node_budget):
        calls.append((depth_cap, node_budget))
        return ((9, 0), [(9, 0), (10, -1)])

    monkeypatch.setattr(hr, "solve_forcing", _old_ext)
    out = _solve_forcing(state)
    assert calls == [(ANALYSIS_DEPTH_CAP, ANALYSIS_NODE_BUDGET)]
    assert out is not None and out["wide"] is False


def test_solve_threat_tight_fallback_on_old_extension(monkeypatch):
    # The threat branch has the same old-extension fallback as the forcing
    # branch; the flipped-solve semantics (winner = opponent) must survive it.
    import hexo_rs as hr
    from hexo_a0.serving.analysis import _solve_forcing
    cfg, state = _forcing_fork_state("P2")

    monkeypatch.setattr(hr, "solve_forcing", lambda st, d, n: None)
    monkeypatch.setattr(
        hr, "solve_threat", lambda st, d, n: ((9, 0), [(9, 0), (10, -1)]))
    out = _solve_forcing(state)
    assert out is not None
    assert out["wide"] is False
    assert out["attacker_is_mover"] is False
    assert out["winner"] == "P1"  # the flipped-perspective attacker


def test_solve_forcing_internal_typeerror_is_not_misread_as_old_extension(monkeypatch):
    # A TypeError raised INSIDE a wide-capable solver (message unrelated to the
    # 'wide' kwarg) must NOT trigger the tight fallback with a wrong provenance
    # label — it propagates to _solve_forcing's crash guard and yields None.
    import hexo_rs as hr
    from hexo_a0.serving.analysis import _solve_forcing
    cfg, state = _forcing_fork_state("P1")
    calls = []

    def _buggy(st, depth_cap, node_budget, wide=False):
        calls.append(wide)
        raise TypeError("expected GraphTensor, got str")

    monkeypatch.setattr(hr, "solve_forcing", _buggy)
    monkeypatch.setattr(hr, "solve_threat", lambda *a, **k: None)
    out = _solve_forcing(state)
    assert out is None
    assert calls == [True], "must not re-call tight after an internal TypeError"


def test_solve_threat_also_requests_wide(monkeypatch):
    # The perspective-flipped threat check gets the same wide request.
    import hexo_rs as hr
    from hexo_a0.serving.analysis import _solve_forcing
    cfg, state = _forcing_fork_state("P2")
    threat_wides = []

    def _threat_spy(st, depth_cap, node_budget, wide=False):
        threat_wides.append(wide)
        return ((9, 0), [(9, 0), (10, -1)])

    monkeypatch.setattr(hr, "solve_forcing", lambda st, d, n, wide=False: None)
    monkeypatch.setattr(hr, "solve_threat", _threat_spy)
    out = _solve_forcing(state)
    assert threat_wides == [True]
    assert out is not None
    assert out["wide"] is True
    assert out["attacker_is_mover"] is False


def test_analyze_game_full_default_run_mcts_fn_uses_trajectory_budget(monkeypatch):
    # analyze_game_full's own default run_mcts_fn (no override) must use the
    # tighter TRAJECTORY_* budget, not the single-position ANALYSIS_* default
    # — it solves twice per prefix across a whole game and must stay bounded.
    import hexo_rs as hr
    model, mc, _ = load_model(CKPT, "cpu")
    seen = []

    def _spy(st, depth_cap, node_budget):
        seen.append((depth_cap, node_budget))
        return None

    monkeypatch.setattr(hr, "solve_forcing", _spy)
    analyze_game_full(model, mc, _GAME, 6, 8, 400, mcts_sims=0)
    assert seen  # forcing was solved for at least one prefix
    assert all(c == (TRAJECTORY_DEPTH_CAP, TRAJECTORY_NODE_BUDGET) for c in seen)


# --- "missed win!" flag: post-processing walk over analyze_game_full's
# trajectory (Task 3) ---------------------------------------------------
#
# The walk itself is pure trajectory-dict logic (no new solves), so these
# tests monkeypatch `_solve_forcing` directly (keyed on the EXACT stone set
# of the position it's called on, never on stone COUNT — analyze_game_full's
# off-line quality evaluator probes positions one placement past a real
# prefix, which would collide with a count-based key) rather than trying to
# hand-construct a genuinely solvable multi-turn forced win via natural play.
# This isolates the walk's branching from the (already-tested, in the
# forcing-line section above) solver-integration concern.
#
# All three walk into `analyze_game_full` over a linear, collision-free move
# sequence (no 6-in-a-row for either side, so the wl=6 game never terminates
# early): seed P1@(0,0), then P2/P1 alternate full turns of 2 placements
# along axis (1,0).
_MISSED_GAME = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
# Prefix 2 = after P2's first full turn: stones {(0,0):P1, (1,0):P2, (2,0):P2},
# P1 to move with a fresh turn (moves[3] is P1's placement from here).
_PREFIX2_STONES = [((0, 0), "P1"), ((1, 0), "P2"), ((2, 0), "P2")]


def _forcing_stub(*rules):
    """A `_solve_forcing`-shaped stub: for each `(stones, forcing_dict)` rule,
    return `forcing_dict` when called on a state whose exact placed-stone set
    matches `stones`; `None` for every other position."""
    targets = [(frozenset(stones), dict(forcing)) for stones, forcing in rules]
    def _stub(state, depth_cap, node_budget):
        current = frozenset(state.placed_stones())
        for stones, forcing in targets:
            if current == stones:
                return dict(forcing)
        return None
    return _stub


def test_analyze_game_full_missed_win_flagged(monkeypatch):
    # Step 1: P1 has a proven win at prefix 2 (first_move (10, 0)) but the
    # actual game plays (3, 0) instead — the resulting placement (trajectory
    # index 3) must carry the missed_win flag with the squandered line.
    import hexo_a0.serving.analysis as analysis_mod
    forcing = {
        "winner": "P1", "attacker_is_mover": True,
        "first_move": [10, 0], "pv": [[10, 0], [10, 1]], "pv_len": 2,
        "pv_owners": ["P1", "P1"],
    }
    monkeypatch.setattr(analysis_mod, "_solve_forcing",
                        _forcing_stub((_PREFIX2_STONES, forcing)))
    mc = types.SimpleNamespace(graph_type="hex")
    out = analyze_game_full(_ZeroLogitModel(), mc, _MISSED_GAME, 6, 8, 400, mcts_sims=0)
    traj = out["trajectory"]
    assert traj[2]["forcing"]["attacker_is_mover"] is True
    assert "missed_win" not in traj[2]  # flag lands on the squander, not the win itself
    assert traj[3]["missed_win"] == {
        "by": "P1", "at_prefix": 2, "first_move": [10, 0],
        "pv": [[10, 0], [10, 1]], "pv_len": 2, "pv_owners": ["P1", "P1"],
    }


def test_analyze_game_full_alternative_win_not_flagged(monkeypatch):
    # Step 2: P1's actually-played move (3, 0) isn't the proven first_move
    # (10, 0), but it ALSO proves a forced win for P1 (an alternative
    # winner) — not a missed win.
    import hexo_a0.serving.analysis as analysis_mod
    prefix3_stones = _PREFIX2_STONES + [((3, 0), "P1")]
    monkeypatch.setattr(analysis_mod, "_solve_forcing", _forcing_stub(
        (_PREFIX2_STONES, {"winner": "P1", "attacker_is_mover": True,
                           "first_move": [10, 0], "pv": [[10, 0], [10, 1]],
                           "pv_len": 2, "pv_owners": ["P1", "P1"]}),
        (prefix3_stones, {"winner": "P1", "attacker_is_mover": True,
                          "first_move": [9, 9], "pv": [[9, 9]],
                          "pv_len": 1, "pv_owners": ["P1"]}),
    ))
    mc = types.SimpleNamespace(graph_type="hex")
    out = analyze_game_full(_ZeroLogitModel(), mc, _MISSED_GAME, 6, 8, 400, mcts_sims=0)
    assert "missed_win" not in out["trajectory"][3]


def test_analyze_game_full_following_the_line_not_flagged(monkeypatch):
    # Step 3: the proven first_move IS what's actually played — not missed.
    import hexo_a0.serving.analysis as analysis_mod
    monkeypatch.setattr(analysis_mod, "_solve_forcing", _forcing_stub(
        (_PREFIX2_STONES, {"winner": "P1", "attacker_is_mover": True,
                           "first_move": [3, 0], "pv": [[3, 0], [4, 0]],
                           "pv_len": 2, "pv_owners": ["P1", "P1"]}),
    ))
    mc = types.SimpleNamespace(graph_type="hex")
    out = analyze_game_full(_ZeroLogitModel(), mc, _MISSED_GAME, 6, 8, 400, mcts_sims=0)
    assert "missed_win" not in out["trajectory"][3]


_PREFIX3_STONES = _PREFIX2_STONES + [((3, 0), "P1")]
# Prefix 6 = after P2's second full turn: P1 to move again with a fresh turn
# (moves[7] is P1's placement from here) — the forward-scan's landing point
# for a deviation on the turn-ENDING placement at prefix 3 (moves[4]).
_PREFIX6_STONES = _PREFIX3_STONES + [((4, 0), "P1"), ((5, 0), "P2"), ((6, 0), "P2")]


def test_analyze_game_full_turn_ending_alternative_win_not_flagged(monkeypatch):
    # Regression: the OLD rule only ever looked at trajectory[i + 1], which
    # for a deviation on the turn-ENDING placement is the OPPONENT's
    # to-move prefix — never P1's own next chance to move — so it could
    # never see that P1 kept a proven win across the opponent's reply, and
    # would falsely flag this as squandered. P1 has a proven win at prefix 3
    # (mid-turn, one placement left) but plays (4, 0) instead of the proven
    # first_move (10, 0); P1's win survives (a different forced win, still
    # proven) at prefix 6 (P1's next to-move prefix, after P2's full reply
    # turn) — the forward-scan must land there and see it, so this must NOT
    # be flagged as a missed win.
    import hexo_a0.serving.analysis as analysis_mod
    monkeypatch.setattr(analysis_mod, "_solve_forcing", _forcing_stub(
        (_PREFIX3_STONES, {"winner": "P1", "attacker_is_mover": True,
                           "first_move": [10, 0], "pv": [[10, 0], [10, 1]],
                           "pv_len": 2, "pv_owners": ["P1", "P1"]}),
        (_PREFIX6_STONES, {"winner": "P1", "attacker_is_mover": True,
                           "first_move": [11, 0], "pv": [[11, 0]],
                           "pv_len": 1, "pv_owners": ["P1"]}),
    ))
    mc = types.SimpleNamespace(graph_type="hex")
    out = analyze_game_full(_ZeroLogitModel(), mc, _MISSED_GAME, 6, 8, 400, mcts_sims=0)
    assert "missed_win" not in out["trajectory"][4]


def test_analyze_game_full_turn_ending_squandered_win_flagged(monkeypatch):
    # Same deviation as above, but P1 does NOT still hold a proven win at
    # prefix 6 (no stub there -> forcing is None) — this really was
    # squandered, and the flag must still land on the deviating placement
    # (trajectory[4], the result of the turn-ending move), not be swallowed
    # by the lookahead.
    import hexo_a0.serving.analysis as analysis_mod
    monkeypatch.setattr(analysis_mod, "_solve_forcing", _forcing_stub(
        (_PREFIX3_STONES, {"winner": "P1", "attacker_is_mover": True,
                           "first_move": [10, 0], "pv": [[10, 0], [10, 1]],
                           "pv_len": 2, "pv_owners": ["P1", "P1"]}),
    ))
    mc = types.SimpleNamespace(graph_type="hex")
    out = analyze_game_full(_ZeroLogitModel(), mc, _MISSED_GAME, 6, 8, 400, mcts_sims=0)
    assert out["trajectory"][4]["missed_win"] == {
        "by": "P1", "at_prefix": 3, "first_move": [10, 0],
        "pv": [[10, 0], [10, 1]], "pv_len": 2, "pv_owners": ["P1", "P1"],
    }


def test_analyze_game_full_budget_shy_alternative_win_not_flagged(monkeypatch):
    # Field report (chao): players get flagged "Missed win!" for playing a
    # DIFFERENT win when the alternative fails to re-prove at the tight
    # TRAJECTORY_* per-prefix budget (the solver's own line move maximizes
    # forcing pressure; an equally-winning human move can enter a wider tree).
    # The walk must escalate the exemption check to the ANALYSIS_* budget
    # before daring to flag. Stub: P1's win at prefix 2 proves at any budget;
    # the kept win at prefix 3 (the scan's landing entry) proves ONLY when
    # node_budget >= ANALYSIS_NODE_BUDGET — so the per-prefix pass records
    # forcing=None there, and only the escalated re-solve can exempt.
    import hexo_a0.serving.analysis as analysis_mod
    prefix2_win = {"winner": "P1", "attacker_is_mover": True,
                   "first_move": [10, 0], "pv": [[10, 0], [10, 1]],
                   "pv_len": 2, "pv_owners": ["P1", "P1"]}
    kept_win = {"winner": "P1", "attacker_is_mover": True,
                "first_move": [9, 9], "pv": [[9, 9]],
                "pv_len": 1, "pv_owners": ["P1"]}
    escalated_calls = []

    def _budget_shy_stub(state, depth_cap, node_budget):
        current = frozenset(state.placed_stones())
        if current == frozenset(_PREFIX2_STONES):
            return dict(prefix2_win)
        if current == frozenset(_PREFIX3_STONES):
            if node_budget >= analysis_mod.ANALYSIS_NODE_BUDGET:
                escalated_calls.append(node_budget)
                return dict(kept_win)
            return None
        return None

    monkeypatch.setattr(analysis_mod, "_solve_forcing", _budget_shy_stub)
    mc = types.SimpleNamespace(graph_type="hex")
    out = analyze_game_full(_ZeroLogitModel(), mc, _MISSED_GAME, 6, 8, 400, mcts_sims=0)
    assert escalated_calls, "walk must escalate to ANALYSIS budget before flagging"
    assert "missed_win" not in out["trajectory"][3]
    # The escalated proof is back-filled for display consistency.
    assert out["trajectory"][3]["forcing"] == kept_win


def test_analyze_game_full_warm_trajectory_never_carries_stale_missed_win(monkeypatch):
    # The walk recomputes over the FULL merged trajectory every call and must
    # never trust a `missed_win` carried in `warm_trajectory` — simulate one
    # left over from an earlier response (e.g. before a since-updated forcing
    # solve) and confirm it's stripped rather than reused.
    import hexo_a0.serving.analysis as analysis_mod
    monkeypatch.setattr(analysis_mod, "_solve_forcing", _forcing_stub(
        (_PREFIX2_STONES, {"winner": "P1", "attacker_is_mover": True,
                           "first_move": [3, 0], "pv": [[3, 0], [4, 0]],
                           "pv_len": 2, "pv_owners": ["P1", "P1"]}),
    ))  # on-line: (3, 0) IS the proven first_move, so no real missed_win
    mc = types.SimpleNamespace(graph_type="hex")
    warm = analyze_game_full(_ZeroLogitModel(), mc, _MISSED_GAME, 6, 8, 400,
                             mcts_sims=0)["trajectory"]
    warm[3]["missed_win"] = {"by": "P1", "at_prefix": 2, "first_move": [99, 99],
                             "pv": [[99, 99]], "pv_len": 1, "pv_owners": None}
    out = analyze_game_full(_ZeroLogitModel(), mc, _MISSED_GAME, 6, 8, 400,
                            mcts_sims=0, warm_trajectory=warm)
    assert "missed_win" not in out["trajectory"][3]
