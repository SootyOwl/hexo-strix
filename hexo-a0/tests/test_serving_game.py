"""Tests for GameManager and make_bot_turn_fn plumbing.

Parity target: scripts/play_server.py:310-400 (make_bot_turn_fn) and 456-663
(GameManager). Bot-turn GPU work is exercised only in the Task 1.8 manual
play-through; here we inject a no-op bot to test the manager plumbing.
"""
import random
from datetime import datetime, timezone

import hexo_rs
import pytest

from hexo_a0.serving.analysis import build_state
from hexo_a0.serving.game import (
    GameManager, GameAlreadyOverError, GameError, GameRecord,
    IllegalMoveError, NotYourTurnError, ServerBusyError, UnknownGameError,
)


def _mgr(bot_turn_fn=lambda rec: None):
    return GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=bot_turn_fn,
        recorder=None,
        mcts_sims=64,
        m_actions=16,
        checkpoint_path="x.pt",
        model_label="test",
        difficulty_sims={"standard": 64},
        default_difficulty="standard",
        idle_ttl_seconds=3600,
        max_games=10,
    )


def test_create_game_returns_record():
    rec = _mgr().create_game("P1", "alice", random.Random(0), "standard")
    assert isinstance(rec, GameRecord) and rec.human_side in ("P1", "P2")


def test_create_game_random_side():
    rec = _mgr().create_game("random", "alice", random.Random(42), "standard")
    assert rec.human_side in ("P1", "P2")


def test_create_game_uses_default_difficulty_for_unknown():
    rec = _mgr().create_game("P1", "alice", random.Random(0), "not-a-tier")
    assert rec.difficulty == "standard"


def test_get_game_unknown_returns_none():
    assert _mgr().get_game("nope") is None


def test_apply_human_move_mutates_record():
    # human="P2" so it's the human's turn right after the engine seed
    # (P1 at origin → P2 moves first). With a no-op bot, "P1" would raise
    # NotYourTurnError.
    m = _mgr()
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    q, r = rec.state.legal_moves()[0]
    rec2 = m.apply_human_move(rec.game_id, q, r)
    assert any(mv[0] == q and mv[1] == r for mv in rec2.move_log)


def test_apply_human_move_unknown_game_raises():
    with pytest.raises(UnknownGameError):
        _mgr().apply_human_move("nope", 1, 0)


def test_resign_records_terminal():
    m = _mgr()
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    rec2 = m.resign(rec.game_id)
    assert rec2.terminal_recorded
    assert rec2 is rec


def test_resign_unknown_game_raises():
    with pytest.raises(UnknownGameError):
        _mgr().resign("nope")


def test_self_reported_elo_recorded_on_record():
    m = _mgr()
    rec = m.create_game("P1", "a", random.Random(0), "standard", self_reported_elo=1500.0)
    assert rec.opp_elo == 1500.0 and rec.elo_source == "self_reported"


def test_anonymous_default_has_no_elo():
    rec = _mgr().create_game("P1", "a", random.Random(0), "standard")
    assert rec.opp_elo is None and rec.elo_source is None


def test_finished_game_exposes_htttx():
    m = _mgr()
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    q, r = rec.state.legal_moves()[0]
    m.apply_human_move(rec.game_id, q, r)
    assert m.htttx(rec.game_id).startswith("version[1];")


def test_create_game_rejects_invalid_human_side():
    # Must raise GameError (-> HTTP 400), not AssertionError / -O-stripped.
    with pytest.raises(GameError):
        _mgr().create_game("P3", "a", random.Random(0), "standard")


def test_create_game_holds_lock_during_initial_bot_turn():
    # human=P1 -> bot=P2, who is to-move after the engine seed -> create_game
    # invokes bot_turn_fn. The closure must observe rec.lock held (no
    # concurrent /state torn read).
    observed = {}

    def bot(rec):
        observed["locked"] = rec.lock.locked()

    m = _mgr(bot_turn_fn=bot)
    rec = m.create_game("P1", "a", random.Random(0), "standard")
    assert observed.get("locked") is True


def test_create_game_bot_turn_updates_last_active_at():
    moved = []

    def bot(rec):
        # Actually place a stone so state advances + last_active_at should bump.
        if rec.state.current_player() == rec.bot_side and not rec.state.is_terminal():
            q, r = rec.state.legal_moves()[0]
            rec.state.apply_move(q, r)
            rec.move_log.append((q, r, rec.bot_side))
            moved.append((q, r))

    m = _mgr(bot_turn_fn=bot)
    rec = m.create_game("P1", "a", random.Random(0), "standard")
    if moved:  # bot had a move to make
        assert rec.last_active_at > rec.created_at


def test_snapshot_returns_state_dict():
    m = _mgr()
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    snap = m.snapshot(rec, placement_radius=8, win_length=6)
    assert snap["game_id"] == rec.game_id
    assert "stones" in snap and "current_player" in snap


def test_illegal_move_raises():
    m = _mgr()
    rec = m.create_game("P2", "a", random.Random(0), "standard")  # human to move
    with pytest.raises(IllegalMoveError):
        m.apply_human_move(rec.game_id, 999, 999)  # far outside legal set


def test_not_your_turn_raises():
    # human=P1 -> bot=P2 is to move after the seed; human can't move yet.
    m = _mgr()
    rec = m.create_game("P1", "a", random.Random(0), "standard")
    q, r = rec.state.legal_moves()[0]
    with pytest.raises(NotYourTurnError):
        m.apply_human_move(rec.game_id, q, r)


def test_game_already_over_raises_after_resign():
    m = _mgr()
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    m.resign(rec.game_id)
    q, r = rec.state.legal_moves()[0]
    with pytest.raises(GameAlreadyOverError):
        m.apply_human_move(rec.game_id, q, r)


def test_record_terminal_failure_leaves_flag_false():
    # If the DB write raises (disk full / lock contention), terminal_recorded
    # must stay False so the game isn't silently lost. Old code set the flag
    # BEFORE the write, so it stayed True even after a failure.
    class BoomRecorder:
        def record_completed(self, **kw):
            raise RuntimeError("disk full")

    m = GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=lambda rec: None, recorder=BoomRecorder(),
        mcts_sims=64, m_actions=16, checkpoint_path="x.pt", model_label="test",
        difficulty_sims={"standard": 64}, default_difficulty="standard",
        idle_ttl_seconds=3600, max_games=10,
    )
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    with pytest.raises(RuntimeError):
        m._record_terminal(rec, winner="P1", result_type="win")
    assert rec.terminal_recorded is False
    # And a retry is still possible (flag didn't latch):
    with pytest.raises(RuntimeError):
        m._record_terminal(rec, winner="P1", result_type="win")


def test_resign_does_not_overwrite_terminal_unrecorded():
    # Regression guard: after a failed terminal write, terminal_recorded is
    # False but state.is_terminal() is True. A /resign must NOT sail past the
    # guard and overwrite the real result (e.g. a human win) with a bot resign.
    class SpyRecorder:
        def __init__(self):
            self.calls = []

        def record_completed(self, **kw):
            self.calls.append((kw.get("winner"), kw.get("result_type")))

    m = GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=lambda rec: None, recorder=SpyRecorder(),
        mcts_sims=64, m_actions=16, checkpoint_path="x.pt", model_label="test",
        difficulty_sims={"standard": 64}, default_difficulty="standard",
        idle_ttl_seconds=3600, max_games=10,
    )
    st = build_state([(0, 0), (0, 5), (1, 5), (1, 0), (2, 0), (2, 5), (0, 6),
                      (3, 0), (4, 0), (1, 6), (2, 6), (5, 0)], 6, 8, 400)
    assert st.is_terminal() and st.winner() == "P1"
    now = datetime.now(timezone.utc)
    rec = GameRecord(game_id="term", created_at=now, last_active_at=now, state=st,
                     human_side="P2", bot_side="P1", human_name="a", difficulty="standard")
    m._games["term"] = rec  # terminal in memory, NOT recorded

    with pytest.raises(GameAlreadyOverError):
        m.resign("term")
    assert m.recorder.calls == []  # the real result was NOT overwritten with a resign
    assert rec.terminal_recorded is False


def test_max_games_zero_rejected():
    with pytest.raises(ValueError):
        GameManager(
            game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
            bot_turn_fn=lambda rec: None, recorder=None,
            mcts_sims=64, m_actions=16, checkpoint_path="x.pt", model_label="test",
            difficulty_sims={"standard": 64}, default_difficulty="standard",
            max_games=0,
        )


def test_create_game_refuses_at_capacity_without_evicting_active():
    # Eviction-DoS mitigation: at capacity with only ACTIVE games, refuse the
    # new game (503) rather than drop an in-progress game. Idle games past the
    # grace window are still reclaimable.
    m = GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=lambda rec: None, recorder=None,
        mcts_sims=64, m_actions=16, checkpoint_path="x.pt", model_label="test",
        difficulty_sims={"standard": 64}, default_difficulty="standard",
        max_games=2, idle_grace_seconds=3600,
    )
    r1 = m.create_game("P2", "a", random.Random(0), "standard")
    r2 = m.create_game("P2", "b", random.Random(1), "standard")
    # Both games are active (just created, within grace) -> 3rd must refuse.
    with pytest.raises(ServerBusyError):
        m.create_game("P2", "c", random.Random(2), "standard")
    # The active games were NOT evicted to make room.
    assert m.get_game(r1.game_id) is not None
    assert m.get_game(r2.game_id) is not None
