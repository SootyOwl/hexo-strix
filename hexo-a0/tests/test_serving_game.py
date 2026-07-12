"""Tests for GameManager and make_bot_turn_fn plumbing.

Parity target: scripts/play_server.py:310-400 (make_bot_turn_fn) and 456-663
(GameManager). Bot-turn GPU work is exercised only in the Task 1.8 manual
play-through; here we inject a no-op bot to test the manager plumbing.
"""
import logging
import random
import time
import types
from datetime import datetime, timezone

import hexo_rs
import pytest

import hexo_a0.serving.game as game_mod
from hexo_a0.serving.analysis import build_state
from hexo_a0.serving.game import (
    GameManager, GameAlreadyOverError, GameError, GameRecord,
    IllegalMoveError, NotYourTurnError, ServerBusyError, UnknownGameError,
    make_bot_turn_fn,
)
from hexo_a0.serving.inference import InferenceGuard


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


# --------------------------------------------------------------------------
# In-progress game persistence across restarts (active_games write-through)
# --------------------------------------------------------------------------

def _playing_bot(rec):
    # Deterministic stand-in for make_bot_turn_fn: plays out the bot's whole
    # turn so persisted snapshots are human-to-move, like production.
    while (not rec.state.is_terminal()) and rec.state.current_player() == rec.bot_side:
        q, r = rec.state.legal_moves()[0]
        rec.state.apply_move(q, r)
        rec.move_log.append((q, r, rec.bot_side))


def _pmgr(tmp_path, name="g.sqlite", **over):
    """GameManager backed by a real Recorder (persistence-enabled)."""
    from hexo_a0.serving.recorder import Recorder
    recorder = Recorder(str(tmp_path / name))
    kwargs = dict(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=_playing_bot,
        recorder=recorder,
        mcts_sims=64,
        m_actions=16,
        checkpoint_path="x.pt",
        model_label="test",
        difficulty_sims={"standard": 64},
        default_difficulty="standard",
        idle_ttl_seconds=3600,
        max_games=10,
    )
    kwargs.update(over)
    return GameManager(**kwargs), recorder


def test_create_game_persists_active_row(tmp_path):
    m, recorder = _pmgr(tmp_path)
    rec = m.create_game("P1", "alice", random.Random(0), "standard")
    rows = recorder.load_active()
    assert [x["game_id"] for x in rows] == [rec.game_id]
    assert rows[0]["human_side"] == "P1" and rows[0]["difficulty"] == "standard"


def test_apply_human_move_updates_active_row(tmp_path):
    import json
    m, recorder = _pmgr(tmp_path)
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    before = len(json.loads(recorder.load_active()[0]["moves_json"]))
    q, r = rec.state.legal_moves()[0]
    m.apply_human_move(rec.game_id, q, r)
    after = len(json.loads(recorder.load_active()[0]["moves_json"]))
    assert after > before


def test_resign_deletes_active_row(tmp_path):
    m, recorder = _pmgr(tmp_path)
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    assert recorder.load_active()
    m.resign(rec.game_id)
    assert recorder.load_active() == []


def test_terminal_game_deletes_active_row(tmp_path):
    # max_moves=3: seed + the human's first two placements end the game, so the
    # terminal write-through must remove the active row (and record the game).
    m, recorder = _pmgr(tmp_path, game_kwargs={"win_length": 6, "placement_radius": 8,
                                               "max_moves": 3})
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    while not rec.state.is_terminal():
        q, r = rec.state.legal_moves()[0]
        m.apply_human_move(rec.game_id, q, r)
    assert rec.terminal_recorded
    assert recorder.load_active() == []
    assert recorder.summary()["total"] == 1


def test_restore_roundtrip(tmp_path):
    m1, recorder = _pmgr(tmp_path)
    rec = m1.create_game("P2", "alice", random.Random(0), "standard")
    q, r = rec.state.legal_moves()[0]
    m1.apply_human_move(rec.game_id, q, r)   # human move + full bot reply

    m2, _ = _pmgr(tmp_path)  # same sqlite file — fresh process stand-in
    assert m2.restore_active_games() == 1
    rec2 = m2.get_game(rec.game_id)
    assert rec2 is not None
    assert rec2.move_log == rec.move_log
    assert rec2.human_side == "P2" and rec2.human_name == "alice"
    assert rec2.state.current_player() == rec2.human_side
    # The restored game is fully playable.
    q2, r2 = rec2.state.legal_moves()[0]
    m2.apply_human_move(rec2.game_id, q2, r2)


def test_restore_skips_and_deletes_expired(tmp_path):
    m1, recorder = _pmgr(tmp_path)
    m1.create_game("P2", "a", random.Random(0), "standard")
    row = recorder.load_active()[0]
    recorder.save_active(**{**{k: row[k] for k in row if k != "moves_json"},
                            "last_active_at": "2020-01-01T00:00:00+00:00",
                            "move_log": [(0, 0, "P1")]})
    m2, _ = _pmgr(tmp_path)
    assert m2.restore_active_games() == 0
    assert recorder.load_active() == []


def test_restore_skips_and_deletes_rule_mismatch(tmp_path):
    m1, recorder = _pmgr(tmp_path)
    m1.create_game("P2", "a", random.Random(0), "standard")
    m2, _ = _pmgr(tmp_path, game_kwargs={"win_length": 5, "placement_radius": 8,
                                         "max_moves": 400})
    assert m2.restore_active_games() == 0
    assert recorder.load_active() == []


def test_restore_skips_and_deletes_unreplayable(tmp_path):
    m1, recorder = _pmgr(tmp_path)
    rec = m1.create_game("P2", "a", random.Random(0), "standard")
    row = recorder.load_active()[0]
    # Duplicate placement is illegal — replay must fail and the row be dropped.
    recorder.save_active(**{**{k: row[k] for k in row if k != "moves_json"},
                            "move_log": [(0, 0, "P1"), (1, 0, "P2"), (1, 0, "P2")]})
    m2, _ = _pmgr(tmp_path)
    assert m2.restore_active_games() == 0
    assert recorder.load_active() == []


def test_restore_skips_and_deletes_bot_to_move(tmp_path):
    # No-stuck-games invariant: nothing schedules a bot turn for a restored
    # game, so a snapshot left bot-to-move must be dropped, not restored.
    m1, recorder = _pmgr(tmp_path)
    m1.create_game("P2", "a", random.Random(0), "standard")
    row = recorder.load_active()[0]
    recorder.save_active(**{**{k: row[k] for k in row if k != "moves_json"},
                            "human_side": "P1", "bot_side": "P2",
                            "move_log": [(0, 0, "P1")]})   # P2 (bot) to move
    m2, _ = _pmgr(tmp_path)
    assert m2.restore_active_games() == 0
    assert recorder.load_active() == []


def test_restore_respects_max_games(tmp_path):
    m1, recorder = _pmgr(tmp_path)
    ids = [m1.create_game("P2", f"u{i}", random.Random(i), "standard").game_id
           for i in range(3)]
    m2, _ = _pmgr(tmp_path, max_games=2)
    assert m2.restore_active_games() == 2
    assert len(recorder.load_active()) == 2   # over-capacity rows are dropped


def test_eviction_deletes_active_row(tmp_path):
    from datetime import timedelta
    m, recorder = _pmgr(tmp_path)
    rec = m.create_game("P2", "a", random.Random(0), "standard")
    # Age the game past the TTL, then trigger lazy eviction via a new game.
    rec.last_active_at = rec.last_active_at - timedelta(seconds=7200)
    m.create_game("P2", "b", random.Random(1), "standard")
    assert m.get_game(rec.game_id) is None
    assert rec.game_id not in {x["game_id"] for x in recorder.load_active()}


def test_persistence_failure_does_not_break_play(tmp_path):
    class BoomActiveRecorder:
        def save_active(self, **kw):
            raise RuntimeError("disk full")

        def delete_active(self, game_id):
            raise RuntimeError("disk full")

    m = GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=_playing_bot, recorder=BoomActiveRecorder(),
        mcts_sims=64, m_actions=16, checkpoint_path="x.pt", model_label="test",
        difficulty_sims={"standard": 64}, default_difficulty="standard",
        idle_ttl_seconds=3600, max_games=10,
    )
    rec = m.create_game("P2", "a", random.Random(0), "standard")   # must not raise
    q, r = rec.state.legal_moves()[0]
    m.apply_human_move(rec.game_id, q, r)                          # must not raise


def test_restore_skips_and_deletes_already_completed(tmp_path):
    # A resignation doesn't change the board, so its replay looks live. If the
    # process dies between the completed-game insert and the active-row delete
    # (two separate commits), restore must drop the row by consulting the games
    # table — otherwise the player continues a finished game and its eventual
    # terminal insert collides with the UNIQUE game_id (500 + zombie).
    m1, recorder = _pmgr(tmp_path)
    rec = m1.create_game("P2", "a", random.Random(0), "standard")
    row_before = recorder.load_active()[0]
    m1.resign(rec.game_id)
    # Simulate the crash window: completed row exists, active row resurrected.
    recorder.save_active(**{**{k: row_before[k] for k in row_before if k != "moves_json"},
                            "move_log": [(0, 0, "P1")]})
    m2, _ = _pmgr(tmp_path)
    assert m2.restore_active_games() == 0
    assert recorder.load_active() == []


def test_sync_after_eviction_does_not_resurrect_row(tmp_path):
    # Race shape: /move thread fetched the record, then an eviction (triggered
    # by /new_game) dropped the game + its row; the move's write-through must
    # not re-insert a row for a game no longer in the manager.
    from datetime import timedelta
    m, recorder = _pmgr(tmp_path)
    victim = m.create_game("P2", "a", random.Random(0), "standard")
    victim.last_active_at = victim.last_active_at - timedelta(seconds=7200)
    m.create_game("P2", "b", random.Random(1), "standard")   # evicts the victim
    assert m.get_game(victim.game_id) is None
    m._sync_active(victim)   # the in-flight move's write-through, arriving late
    assert victim.game_id not in {x["game_id"] for x in recorder.load_active()}


def test_ttl_victim_row_dropped_even_when_create_refuses(tmp_path):
    # _evict's TTL pass can drop a game and then still return "at capacity"
    # (ServerBusyError). The TTL victim's active row must be deleted anyway.
    from datetime import timedelta
    m, recorder = _pmgr(tmp_path, max_games=2, idle_grace_seconds=3600)
    stale = m.create_game("P2", "a", random.Random(0), "standard")
    m.create_game("P2", "b", random.Random(1), "standard")
    stale.last_active_at = stale.last_active_at - timedelta(seconds=7200)  # past TTL
    m.create_game("P2", "c", random.Random(2), "standard")   # TTL-evicts `stale`
    # Manager now holds b + c, both within grace: next create must refuse.
    stale2 = m.get_game([g["game_id"] for g in m.active_games()][0])
    with pytest.raises(ServerBusyError):
        m.create_game("P2", "d", random.Random(3), "standard")
    assert stale.game_id not in {x["game_id"] for x in recorder.load_active()}


def test_restore_skips_and_deletes_bad_bot_side(tmp_path):
    # A row with bot_side == human_side would restore a game whose bot turn can
    # never be scheduled (permanently stuck) — drop it instead.
    m1, recorder = _pmgr(tmp_path)
    m1.create_game("P2", "a", random.Random(0), "standard")
    row = recorder.load_active()[0]
    recorder.save_active(**{**{k: row[k] for k in row if k != "moves_json"},
                            "bot_side": "P2",
                            "move_log": [(0, 0, "P1")]})
    m2, _ = _pmgr(tmp_path)
    assert m2.restore_active_games() == 0
    assert recorder.load_active() == []


# --------------------------------------------------------------------------
# Live-play forcing: PV-cached win execution + pre-emptive VCF defense.
#
# `make_bot_turn_fn` needs a real model/graph_fn to reach the raw-argmax
# fallback path, so these tests stub the model rather than injecting a no-op
# bot_turn_fn like the GameManager tests above. `mcts_sims=0` keeps every
# non-forcing fallback on the raw-argmax path (no GPU, no gumbel_mcts).
# --------------------------------------------------------------------------

class _ZeroLogitModel:
    """All-zero logits/value: raw-argmax degenerates to "first candidate" (or,
    restricted, the sole candidate) with no need to control learned weights."""

    def __call__(self, x, edge_index, legal_mask, stone_mask=None, edge_attr=None):
        import torch
        return torch.zeros(x.shape[0]), torch.tensor(0.0)

    def forward_batch(self, batch):
        # Not exercised (mcts_sims=0 in these tests); _eval_fn would recover
        # via its per-graph fallback (calling __call__) if it ever were.
        raise NotImplementedError


def _bot_turn_fn(game_kwargs, live_forcing=True, difficulty_forcing_depth=None):
    return make_bot_turn_fn(
        model=_ZeroLogitModel(),
        model_config=types.SimpleNamespace(graph_type="hex"),
        m_actions=16,
        mcts_sims=0,
        guard=InferenceGuard(),
        game_kwargs=game_kwargs,
        live_forcing=live_forcing,
        difficulty_forcing_depth=difficulty_forcing_depth,
    )


def _seq_moves(cfg, n, pick=lambda legal: legal[0]):
    """Replay `n` real, legal placements from a fresh GameState. Returns the
    (q, r) sequence, in play order. `pick` selects among `legal_moves()` at
    each step (default: the engine's first, deterministic)."""
    state = hexo_rs.GameState(cfg)
    moves = []
    for _ in range(n):
        q, r = pick(state.legal_moves())
        state.apply_move(q, r)
        moves.append((q, r))
    return moves


def test_forcing_bot_executes_own_forced_win():
    # Bot (P1) has an open pair; with 2 free placements this turn (no
    # opponent reply in between), solve_forcing finds a 2-move forced win.
    cfg_kwargs = {"win_length": 4, "placement_radius": 3, "max_moves": 60}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P1")], "P1", 2, cfg)
    # rec.difficulty defaults to DEFAULT_DIFFICULTY ("standard"), so that
    # tier's forcing depth (not a flat LIVE_DEPTH_CAP — depth is now
    # per-difficulty, see DEFAULT_DIFFICULTY_FORCING_DEPTH) is what the bot
    # actually searches at.
    standard_depth = game_mod.DEFAULT_DIFFICULTY_FORCING_DEPTH[game_mod.DEFAULT_DIFFICULTY]
    expected = hexo_rs.solve_forcing(state, standard_depth, game_mod.LIVE_NODE_BUDGET)
    assert expected is not None
    first_move, pv = expected
    assert len(pv) == 2  # both placements land in the same (bot's) turn

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-forcing-win", created_at=now, last_active_at=now, state=state,
        human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P1")],
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    assert rec.move_log[-2:] == [
        (first_move[0], first_move[1], "P1"),
        (pv[1][0], pv[1][1], "P1"),
    ]
    assert rec.state.is_terminal() and rec.state.winner() == "P1"


def test_forcing_pv_cache_replay_skips_resolve(monkeypatch):
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    # Deliberately pick the LAST-ordered legal move at each step (not the
    # first): the raw-argmax fallback's node ordering favours the lexically
    # smallest coordinate, so if the cache check were bypassed the fallback
    # would land on a different cell and this test would (correctly) fail.
    pv = _seq_moves(cfg, 6, pick=lambda legal: max(legal))

    state = hexo_rs.GameState(cfg)  # fresh, right after the (0,0) seed
    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-forcing-replay", created_at=now, last_active_at=now, state=state,
        human_side="P1", bot_side="P2", human_name="a",
        move_log=[(0, 0, "P1")],
        forcing_pv=list(pv),
        forcing_base=1,
    )

    calls = {"n": 0}

    def _counting_solve(*a, **k):
        calls["n"] += 1
        return None

    monkeypatch.setattr(hexo_rs, "solve_forcing", _counting_solve)
    _bot_turn_fn(cfg_kwargs)(rec)

    assert calls["n"] == 0  # cache hit — solver never invoked
    assert rec.move_log[-2:] == [(pv[0][0], pv[0][1], "P2"), (pv[1][0], pv[1][1], "P2")]
    assert rec.forcing_pv == list(pv)  # untouched


def test_forcing_pv_cache_invalidated_on_deviation(monkeypatch):
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    pv = _seq_moves(cfg, 6)  # bot(P2): pv[0:2] ; opp(P1): pv[2:4] ; bot(P2): pv[4:6]

    # Replay the bot's cached turn for real, then have the opponent deviate on
    # the first of their two placements (their second placement is arbitrary).
    state = hexo_rs.GameState(cfg)
    state.apply_move(*pv[0])
    state.apply_move(*pv[1])
    legal = {tuple(c) for c in state.legal_moves()}
    deviation = next(c for c in legal if c != pv[2])
    state.apply_move(*deviation)
    other = state.legal_moves()[0]
    state.apply_move(*other)
    assert state.current_player() == "P2"  # bot to move again

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-forcing-deviate", created_at=now, last_active_at=now, state=state,
        human_side="P1", bot_side="P2", human_name="a",
        move_log=[
            (0, 0, "P1"), (*pv[0], "P2"), (*pv[1], "P2"),
            (*deviation, "P1"), (*other, "P1"),
        ],
        forcing_pv=list(pv),
        forcing_base=1,
    )

    calls = {"n": 0}

    def _counting_solve(*a, **k):
        calls["n"] += 1
        return None  # avoid discovering an unrelated real win mid-test

    monkeypatch.setattr(hexo_rs, "solve_forcing", _counting_solve)
    _bot_turn_fn(cfg_kwargs)(rec)

    assert calls["n"] >= 1  # deviation forced a re-solve
    assert rec.forcing_pv is None


def test_forcing_defensive_block_plays_the_only_killer():
    # Bot (P1) has no forced win of its own; the opponent (P2), hypothetically
    # to move, has a mate-in-1 (a three blocked on one side); a single legal
    # cell on the open side kills it. The own-win check and the defensive
    # sweep now share ONE per-tier depth cap (see DEFAULT_DIFFICULTY_FORCING_DEPTH
    # / _forcing_depth_for), so this fixture pins the "standard" tier (rec's
    # default difficulty) to depth=1 — deep enough to find the mate-in-1 and
    # confirm the kill, shallow enough that the own-win check still (truly)
    # finds nothing. A deeper shared depth (verified empirically) uncovers an
    # unrelated, longer forced line through (11, 0) that isn't actually
    # blocked by it, which would spuriously turn this into a "no killer"
    # case — depth=1 is the right fixture for what this test is checking.
    forcing_depth = {"standard": 1}

    cfg_kwargs = {"win_length": 4, "placement_radius": 3, "max_moves": 60}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    stones = [((0, 0), "P1"), ((7, 0), "P1"),
              ((8, 0), "P2"), ((9, 0), "P2"), ((10, 0), "P2")]
    state = hexo_rs.GameState.from_state(stones, "P1", 2, cfg)

    # Replicate the trial re-solve loop directly to derive the expected
    # killer set, per the brief. Node budgets mirror the real call sites
    # (LIVE_NODE_BUDGET for the bot's own-win check, DEF_NODE_BUDGET for the
    # defensive sweep) rather than a hardcoded literal, so this stays correct
    # across budget re-tunes.
    assert hexo_rs.solve_forcing(
        state, 1, game_mod.LIVE_NODE_BUDGET) is None  # bot has no own VCF
    legal = {tuple(c) for c in state.legal_moves()}
    opp_win = hexo_rs.solve_forcing(
        hexo_rs.GameState.from_state(state.placed_stones(), "P2", 2, cfg),
        1, game_mod.DEF_NODE_BUDGET)
    assert opp_win is not None
    _ofm, opv = opp_win
    candidates = [c for c in {tuple(m) for m in opv} if c in legal]
    killers = []
    for c in candidates:
        trial = state.clone()
        trial.apply_move(*c)
        re_res = hexo_rs.solve_forcing(
            hexo_rs.GameState.from_state(trial.placed_stones(), "P2", 2, cfg),
            1, game_mod.DEF_NODE_BUDGET)
        if re_res is None:
            killers.append(c)
    assert killers == [(11, 0)]  # exactly one legal killing cell

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-forcing-defend", created_at=now, last_active_at=now, state=state,
        human_side="P2", bot_side="P1", human_name="a",
        move_log=[(q, r, side) for (q, r), side in stones],
    )
    before = len(rec.move_log)
    _bot_turn_fn(cfg_kwargs, difficulty_forcing_depth=forcing_depth)(rec)
    new_moves = [m[:2] for m in rec.move_log[before:]]

    assert (11, 0) in new_moves


def test_forcing_defensive_deadline_passed_to_solver(monkeypatch):
    # The DEFENSE_DEADLINE_S backstop is enforced INSIDE hr.solve_defense
    # (see forcing.rs's time-limit tests for the cutoff behaviour itself);
    # what game.py owns is the plumbing — the deadline must reach the solver
    # in milliseconds, and a returned killer must be played.
    monkeypatch.setattr(game_mod, "DEFENSE_DEADLINE_S", 0.01)

    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2")], "P1", 1, cfg)  # P1's last placement this turn
    killer = tuple(state.legal_moves()[0])

    monkeypatch.setattr(hexo_rs, "solve_forcing", lambda *a, **k: None)
    seen = {}

    def _stub_defense(_state, depth_cap, node_budget, time_limit_ms):
        seen["args"] = (depth_cap, node_budget, time_limit_ms)
        return ([killer], [], None, [killer])

    monkeypatch.setattr(hexo_rs, "solve_defense", _stub_defense)

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-forcing-deadline", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2")],
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    depth_cap, node_budget, time_limit_ms = seen["args"]
    assert time_limit_ms == 10  # 0.01 s deadline, in ms
    assert node_budget == game_mod.DEF_NODE_BUDGET
    assert depth_cap == game_mod.DEFAULT_DIFFICULTY_FORCING_DEPTH["standard"]
    assert rec.move_log[-1][:2] == killer  # the returned killer was played


def test_forcing_depth_capped_per_difficulty_tier(monkeypatch):
    # Per-difficulty forcing DEPTH cap: a forced win that only a deep-enough
    # search can prove must be executed by "strong" (depth=16) but NOT by
    # "casual" (depth=2) — casual falls through to the raw-argmax path
    # rather than executing a win it can't (at its tier's depth) see.
    # `hexo_rs.solve_forcing` is stubbed to "find" the win only when
    # depth_cap >= 6 (strictly between easy's 4 and standard's 6), recording
    # every depth_cap it's called with so the per-tier PLUMBING itself (not
    # just the pass/fail outcome) is verified.
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)

    def _rec(game_id, difficulty):
        # moves_remaining=1: P1's LAST placement this turn, so `_play`'s loop
        # exits after one placement (no second _forcing_move call to muddy
        # the recorded depths).
        state = hexo_rs.GameState.from_state(
            [((0, 0), "P1"), ((1, 0), "P2")], "P1", 1, cfg)
        now = datetime.now(timezone.utc)
        return state, GameRecord(
            game_id=game_id, created_at=now, last_active_at=now, state=state,
            human_side="P2", bot_side="P1", human_name="a",
            move_log=[(0, 0, "P1"), (1, 0, "P2")], difficulty=difficulty,
        )

    # --- casual (depth=2): the win needs depth>=6, so it's never found. ---
    state, rec = _rec("g-depth-casual", "casual")
    win_cell = tuple(state.legal_moves()[0])
    depths = []

    def _stub_casual(_state, depth_cap, _node_budget):
        depths.append(depth_cap)
        return (win_cell, [win_cell]) if depth_cap >= 6 else None

    monkeypatch.setattr(hexo_rs, "solve_forcing", _stub_casual)
    _bot_turn_fn(cfg_kwargs)(rec)

    assert depths and all(d == 2 for d in depths)  # casual's tier depth, every call
    assert rec.forcing_pv is None  # the win was never executed via forcing

    # --- strong: the same win threshold is now cleared. ---
    state, rec = _rec("g-depth-strong", "strong")
    win_cell = tuple(state.legal_moves()[0])
    depths = []

    def _stub_strong(_state, depth_cap, _node_budget):
        depths.append(depth_cap)
        return (win_cell, [win_cell]) if depth_cap >= 6 else None

    monkeypatch.setattr(hexo_rs, "solve_forcing", _stub_strong)
    _bot_turn_fn(cfg_kwargs)(rec)

    # strong's tier depth; own-win solve found it immediately
    assert depths == [game_mod.DEFAULT_DIFFICULTY_FORCING_DEPTH["strong"]]
    assert rec.move_log[-1][:2] == win_cell  # the win WAS executed


def test_forcing_defense_saves_strongloss_a_via_killer_pair():
    # Regression for the 2026-07-04 live loss (strongloss_a_line.json,
    # prefix 6): the human (P2) completed an 18-placement VCF; no SINGLE P1
    # block refutes it, but killer PAIRS exist. The old Python max-delay
    # survivor heuristic played (-3,3) — a non-anchor — and lost the game;
    # hr.solve_defense's pair pass must anchor the turn so that after the
    # bot's TWO placements the opponent's VCF is no longer provable.
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    stones = [((0, 0), "P1"), ((1, 2), "P2"), ((2, 3), "P2"), ((3, -1), "P1"),
              ((2, 1), "P1"), ((1, 4), "P2"), ((1, 3), "P2")]
    state = hexo_rs.GameState.from_state(stones, "P1", 2, cfg)
    depth = game_mod.DEFAULT_DIFFICULTY_FORCING_DEPTH["strong"]

    # Sanity: the threat is real and no single block kills it (else this
    # fixture no longer tests the pair pass).
    opp_state = hexo_rs.GameState.from_state(state.placed_stones(), "P2", 2, cfg)
    assert hexo_rs.solve_forcing(opp_state, depth, game_mod.DEF_NODE_BUDGET) is not None

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-strongloss-a", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(q, r, side) for (q, r), side in stones],
        difficulty="strong",
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    # After the bot's full turn, P2's forced win must be refuted.
    after = hexo_rs.GameState.from_state(rec.state.placed_stones(), "P2", 2, cfg)
    assert hexo_rs.solve_forcing(after, depth, game_mod.DEF_NODE_BUDGET) is None
    # And the losing move the old heuristic chose must not be the anchor.
    assert rec.move_log[len(stones)][:2] != (-3, 3)


def test_forcing_defense_pair_partner_cached_within_turn(monkeypatch):
    # Regression for the 2026-07-06 chao loss: after playing a verified pair's
    # ANCHOR, the same turn's 2nd placement must play the pair's PARTNER
    # directly from the GameRecord cache — the pair was verified jointly, and
    # only the bot's own anchor stone was added since. Re-deriving the partner
    # via a fresh solve_defense is deadline-vulnerable: on the slow production
    # host the re-solve expired mid-verification, returned killers=[] with only
    # a best_delay survivor, and the bot played the survivor and lost. The 2nd
    # stubbed call below returns exactly that deadline-expired shape; it must
    # never be consulted.
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2")], "P1", 2, cfg)  # bot's full turn ahead
    legal = [tuple(c) for c in state.legal_moves()]
    anchor, partner, survivor = legal[0], legal[1], legal[2]

    monkeypatch.setattr(hexo_rs, "solve_forcing", lambda *a, **k: None)
    calls = {"n": 0}

    def _stub_defense(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([], [(anchor, partner)], None, [anchor, partner])
        return ([], [], survivor, [])  # deadline-expired partial result

    monkeypatch.setattr(hexo_rs, "solve_defense", _stub_defense)

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-pair-cache", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2")],
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    assert [m[:2] for m in rec.move_log[-2:]] == [anchor, partner]
    assert calls["n"] == 1  # 2nd placement came from the cache, not a re-solve


def test_forcing_defense_partner_cache_not_reused_across_turns(monkeypatch):
    # The partner cache is only sound for the immediately-following placement
    # of the SAME bot turn (nothing but the bot's own anchor stone has been
    # added). Once the opponent has moved, a cached partner is stale and the
    # defense must be re-derived from scratch.
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2"), ((2, 0), "P2")], "P1", 1, cfg)
    legal = [tuple(c) for c in state.legal_moves()]
    anchor, stale_partner, killer2 = legal[0], legal[1], legal[2]

    monkeypatch.setattr(hexo_rs, "solve_forcing", lambda *a, **k: None)
    calls = {"n": 0}

    def _stub_defense(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([], [(anchor, stale_partner)], None, [anchor, stale_partner])
        return ([killer2], [], None, [killer2])

    monkeypatch.setattr(hexo_rs, "solve_defense", _stub_defense)

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-pair-stale", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2"), (2, 0, "P2")],
    )
    turn_fn = _bot_turn_fn(cfg_kwargs)
    turn_fn(rec)  # bot's LAST placement this turn: plays the anchor
    assert rec.move_log[-1][:2] == anchor

    # Human replies (two placements), then the bot's next turn begins.
    for cell in rec.state.legal_moves()[:2]:
        q, r = cell
        rec.state.apply_move(q, r)
        rec.move_log.append((q, r, "P2"))
    turn_fn(rec)

    new_moves = [m[:2] for m in rec.move_log[-2:]]
    assert stale_partner not in new_moves  # stale cache must not fire
    assert killer2 in new_moves            # defense was re-derived instead
    assert calls["n"] >= 2


def test_forcing_defense_deadline_strong_tier(monkeypatch):
    # Per-tier defense deadline: "strong" gets a longer solve_defense time
    # limit than the flat 3 s backstop (the chao loss was a deadline expiry on
    # the slow production host — see DIFFICULTY_DEFENSE_DEADLINE_S).
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2")], "P1", 1, cfg)
    killer = tuple(state.legal_moves()[0])

    monkeypatch.setattr(hexo_rs, "solve_forcing", lambda *a, **k: None)
    seen = {}

    def _stub_defense(_state, depth_cap, node_budget, time_limit_ms):
        seen["ms"] = time_limit_ms
        return ([killer], [], None, [killer])

    monkeypatch.setattr(hexo_rs, "solve_defense", _stub_defense)

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-deadline-strong", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2")], difficulty="strong",
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    expected = int(game_mod.DIFFICULTY_DEFENSE_DEADLINE_S["strong"] * 1000)
    assert seen["ms"] == expected
    assert expected > int(game_mod.DEFENSE_DEADLINE_S * 1000)


def test_forcing_defense_rejects_killer_refuted_at_escalated_budget(monkeypatch):
    # Escalated verification: solve_defense's internal candidate re-solves run
    # at the flat DEF budget where BudgetExceeded counts as refuted, so a
    # "verified" killer can actually lose (the 2026-07-06 chao position: 4 of
    # 5 verified pairs lost, exposed only at 30k-75k nodes). Before playing a
    # candidate, _defensive_move must re-check it at the escalated budget via
    # a flip-perspective solve on the trial position and skip candidates whose
    # refutation was a budget artifact. The stub "finds" the opponent win
    # whenever the bad killer is on the trial board.
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2")], "P1", 1, cfg)  # bot's last placement
    k_bad, k_good = (2, 0), (3, 0)  # policy argmax (all-zero logits) tries lowest first

    def _fake_solve_forcing(st, depth_cap, node_budget):
        if any(tuple(c) == k_bad for c, _p in st.placed_stones()):
            return ((9, 9), [(9, 9)])  # opponent win still provable past k_bad
        return None

    monkeypatch.setattr(hexo_rs, "solve_forcing", _fake_solve_forcing)
    monkeypatch.setattr(
        hexo_rs, "solve_defense",
        lambda *a, **k: ([k_bad, k_good], [], None, [k_bad, k_good]))

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-verify-killer", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2")],
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    assert rec.move_log[-1][:2] == k_good


def test_forcing_defense_rejects_pair_refuted_at_escalated_budget(monkeypatch):
    # Same as above for killer PAIRS — the actual chao failure shape: the
    # policy-preferred pair is a budget artifact and must be discarded; the
    # next pair verifies clean, its anchor is played and its partner comes
    # from the cache at the 2nd placement (still exactly one defense solve).
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2")], "P1", 2, cfg)  # bot's full turn
    bad = ((2, 0), (2, 1))
    good = ((3, 0), (3, 1))

    def _fake_solve_forcing(st, depth_cap, node_budget):
        if any(tuple(c) == bad[0] for c, _p in st.placed_stones()):
            return ((9, 9), [(9, 9)])
        return None

    monkeypatch.setattr(hexo_rs, "solve_forcing", _fake_solve_forcing)
    calls = {"n": 0}

    def _stub_defense(*a, **k):
        calls["n"] += 1
        return ([], [bad, good], None, [bad[0], bad[1]])

    monkeypatch.setattr(hexo_rs, "solve_defense", _stub_defense)

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-verify-pair", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2")],
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    assert [m[:2] for m in rec.move_log[-2:]] == [good[0], good[1]]
    assert calls["n"] == 1


def test_forcing_defense_saves_chao_game_via_pair_cache(monkeypatch):
    # End-to-end regression for the 2026-07-06 chao (P1) win over strix strong
    # (P2), decoded from the shared analysis URL. At the bot's turn after
    # chao's (4,-7),(2,-7), no single block refutes the VCF but the pair
    # ((-1,0),(2,-5)) does. Live, the bot played the anchor and then LOST the
    # partner to a defense-deadline expiry on the 2nd placement, playing the
    # max-delay survivor (0,-5) instead — after which chao's win is proven.
    # With the partner cache, the 2nd placement must come without a re-solve
    # and the full turn must refute the threat.
    chao_prefix = [
        (1, 2), (2, 3), (2, 2), (2, -3), (2, -2), (-1, 5), (3, -5), (1, 4),
        (3, -4), (-2, 4), (-1, -4), (-1, -2), (-1, -3), (-1, 3), (0, 3),
        (-3, 5), (6, -6), (0, 1), (-3, -2), (-3, 0), (-4, -1), (-2, -1),
        (-5, -1), (-2, 0), (-3, 1), (-2, -4), (4, -7), (2, -7),
    ]
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState(cfg)
    move_log = [(0, 0, "P1")]
    for q, r in chao_prefix:
        move_log.append((q, r, state.current_player()))
        state.apply_move(q, r)
    assert state.current_player() == "P2"  # strix to move, 2 placements

    depth = game_mod.DEFAULT_DIFFICULTY_FORCING_DEPTH["strong"]
    # Sanity: chao's threat is real at the bot's turn start.
    flip = hexo_rs.GameState.from_state(state.placed_stones(), "P1", 2, cfg)
    assert hexo_rs.solve_forcing(flip, depth, game_mod.DEF_NODE_BUDGET) is not None

    # Counting passthrough: on a fast dev box the deadline-vulnerable re-solve
    # happens to succeed, so the LIVE failure (deadline expiry on the slow
    # production host) can't reproduce here. What is host-independent is that
    # the 2nd placement must come from the pair cache — ONE defense solve for
    # the whole turn.
    real_solve_defense = hexo_rs.solve_defense
    calls = {"n": 0}

    def _counting_defense(*a, **k):
        calls["n"] += 1
        return real_solve_defense(*a, **k)

    monkeypatch.setattr(hexo_rs, "solve_defense", _counting_defense)

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-chao", created_at=now, last_active_at=now, state=state,
        human_side="P1", bot_side="P2", human_name="chao",
        move_log=move_log, difficulty="strong",
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    # After the bot's full turn, chao's forced win must be refuted AT THE
    # ESCALATED BUDGET, not just the flat DEF budget: of the five pairs
    # solve_defense "verifies" here at 20k, four are losing — their
    # refutations are budget artifacts exposed at 30k-75k nodes. Only
    # ((-1,0),(2,-5)) survives (proven empty to 20M). Asserting at
    # VERIFY_NODE_BUDGET pins that the bot escalation-checked its choice.
    after = hexo_rs.GameState.from_state(rec.state.placed_stones(), "P1", 2, cfg)
    assert hexo_rs.solve_forcing(after, depth, game_mod.VERIFY_NODE_BUDGET) is None
    assert calls["n"] == 1  # 2nd placement served from the pair cache


# --------------------------------------------------------------------------
# 2026-07-12 Hayes live loss (fixtures/hayes_20260712.htttx, strix as P1 at
# 512 sims): at placement 30 the bot had a deep-prover-confirmed forced win
# ((6,-5)+(6,-3): block the opponent's r=-3 completion WITH a q=6-column
# four) but the own-win solve missed it at the old 20k live budget (the win
# needs ~35k nodes / ~120 ms, at any depth cap 8..20), the defense layer took
# over, and one placement later the same budget miss let the pair cache play
# (8,-2) while (6,-1) — another verified killer — was itself a proven win.
# --------------------------------------------------------------------------

_HAYES_HTTTX = None


def _hayes_position(n):
    """State + move_log after the first `n` placements of the Hayes game
    (engine seed excluded from the count, included in the move_log)."""
    global _HAYES_HTTTX
    if _HAYES_HTTTX is None:
        import pathlib
        _HAYES_HTTTX = (pathlib.Path(__file__).parent / "fixtures"
                        / "hayes_20260712.htttx").read_text()
    from hexo_a0.serving.htttx import parse_htttx
    cfg = hexo_rs.GameConfig(win_length=6, placement_radius=8, max_moves=400)
    state = hexo_rs.GameState(cfg)
    move_log = [(0, 0, "P1")]
    for q, r in parse_htttx(_HAYES_HTTTX)[:n]:
        side = state.current_player()
        state.apply_move(q, r)
        move_log.append((q, r, side))
    return state, move_log


def test_live_budget_finds_hayes_turn16_win():
    # The turn-16 win must be visible to the own-win solve at the LIVE node
    # budget, at both the standard and strong tier depth caps (the win is 7
    # attacker-turns deep; standard's depth 8 covers it).
    state, _ = _hayes_position(30)
    assert state.current_player() == "P1"
    assert state.moves_remaining_this_turn() == 2
    for tier in ("standard", "strong"):
        depth = game_mod.DEFAULT_DIFFICULTY_FORCING_DEPTH[tier]
        res = hexo_rs.solve_forcing(state, depth, game_mod.LIVE_NODE_BUDGET)
        assert res is not None, f"turn-16 win invisible at {tier} live budget"
        assert tuple(res[0]) == (6, -5)


def test_live_budget_finds_hayes_placement31_win():
    # One placement later (the blocking half of the winning pair already
    # played), the single stone (6,-5) still wins — the live game's second
    # missed chance before the opponent extinguished the resource.
    state, _ = _hayes_position(31)
    assert state.current_player() == "P1"
    assert state.moves_remaining_this_turn() == 1
    for tier in ("standard", "strong"):
        depth = game_mod.DEFAULT_DIFFICULTY_FORCING_DEPTH[tier]
        res = hexo_rs.solve_forcing(state, depth, game_mod.LIVE_NODE_BUDGET)
        assert res is not None, f"placement-31 win invisible at {tier} live budget"
        assert tuple(res[0]) == (6, -5)


def test_forcing_bot_wins_hayes_turn16_position():
    # End to end: at the placement-30 position the bot's own-win solve must
    # preempt the defense layer and execute the proven win — the live game
    # instead played the defense pair (6,-3),(8,-2) and went on to lose.
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    state, move_log = _hayes_position(30)
    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-hayes-t16", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="hayes",
        move_log=move_log, difficulty="strong",
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    assert [m[:2] for m in rec.move_log[-2:]] == [(6, -5), (6, -3)]
    assert rec.forcing_pv is not None  # executed as a cached forced win


def test_forcing_defense_prefers_win_contributing_killer(monkeypatch):
    # Among VERIFIED killers the bot must prefer one whose placement keeps or
    # creates its own forcing win (the "block with a double threat" rule) —
    # in the Hayes game the defense layer played its policy-first killer
    # (8,-2) while (6,-1), also in the killers list, was itself a proven win.
    # Zero logits make the policy argmax pick the lexically smallest killer,
    # so without the ranking the passive one wins the tie.
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2")], "P1", 2, cfg)
    passive, winning, win_cell = (-1, 0), (3, 3), (5, 5)

    def _stub_forcing(st, _depth_cap, node_budget):
        if node_budget == game_mod.VERIFY_NODE_BUDGET:
            return None  # every killer verifies as a real block
        stones = {(int(c[0]), int(c[1])) for c, _p in st.placed_stones()}
        if winning in stones:
            # The ranking probe (and, next placement, the own-win solve) sees
            # a forcing win once the win-contributing killer is on the board.
            return (win_cell, [win_cell])
        return None  # bare position: no own win -> defense layer runs

    calls = {"defense": 0}

    def _stub_defense(*_a, **_k):
        calls["defense"] += 1
        if calls["defense"] == 1:
            return ([passive, winning], [], None, [passive, winning])
        return ([], [], None, [])

    monkeypatch.setattr(hexo_rs, "solve_forcing", _stub_forcing)
    monkeypatch.setattr(hexo_rs, "solve_defense", _stub_defense)

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-rank-killer", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2")],
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    new_moves = [m[:2] for m in rec.move_log[2:]]
    assert new_moves[0] == winning  # ranked above the policy-first killer
    assert new_moves[1] == win_cell  # ...and the own-win solve converts it


def test_forcing_defense_prefers_threat_keeping_killer_on_last_placement(monkeypatch):
    # Same preference on the turn's LAST placement, where the probe is the
    # flipped-perspective threat solve (the follow-up win only executes if
    # the opponent lets it, but keeping the initiative among proven-
    # equivalent killers is strictly better than not).
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2"), ((2, 0), "P2")], "P1", 1, cfg)
    passive, keeping = (-1, 0), (3, 3)

    monkeypatch.setattr(hexo_rs, "solve_forcing", lambda *a, **k: None)

    def _stub_threat(st, _depth_cap, _node_budget):
        stones = {(int(c[0]), int(c[1])) for c, _p in st.placed_stones()}
        return ((9, 9), [(9, 9)]) if keeping in stones else None

    monkeypatch.setattr(hexo_rs, "solve_threat", _stub_threat)
    monkeypatch.setattr(
        hexo_rs, "solve_defense",
        lambda *a, **k: ([passive, keeping], [], None, [passive, keeping]))

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-rank-threat", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2"), (2, 0, "P2")],
    )
    _bot_turn_fn(cfg_kwargs)(rec)

    assert rec.move_log[-1][:2] == keeping


def test_bot_turn_logs_placement_timing(monkeypatch, caplog):
    # Deploy observability: every bot placement emits ONE INFO line with the
    # move source and per-phase wall-clock (forcing/defense solver vs MCTS),
    # so slow turns on the production box can be attributed from docker logs
    # without a profiler.
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2")], "P1", 2, cfg)
    monkeypatch.setattr(hexo_rs, "solve_forcing", lambda *a, **k: None)
    monkeypatch.setattr(hexo_rs, "solve_defense", lambda *a, **k: None)

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-timing", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2")],
    )
    with caplog.at_level(logging.INFO, logger="hexo_a0.serving.game"):
        _bot_turn_fn(cfg_kwargs)(rec)

    lines = [r.message for r in caplog.records if "bot placement" in r.message]
    assert len(lines) == 2  # one per placement
    for line in lines:
        assert "src=argmax" in line  # mcts_sims=0 -> raw-argmax fallback
        assert "forcing_ms=" in line and "mcts_ms=" in line
        assert "game=g-timing" in line


def test_bot_turn_logs_defense_source(monkeypatch, caplog):
    # The source tag must name WHICH defense mechanism produced the move
    # (killer / pair / partner / delay) — that's the field that would have
    # shown the chao-loss deadline expiry ("delay" instead of "killer") at a
    # glance in production logs.
    cfg_kwargs = {"win_length": 6, "placement_radius": 8, "max_moves": 400}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P2")], "P1", 2, cfg)
    legal = [tuple(c) for c in state.legal_moves()]
    anchor, partner = legal[0], legal[1]

    monkeypatch.setattr(hexo_rs, "solve_forcing", lambda *a, **k: None)
    monkeypatch.setattr(
        hexo_rs, "solve_defense",
        lambda *a, **k: ([], [(anchor, partner)], None, [anchor, partner]))

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-timing-src", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P2")],
    )
    with caplog.at_level(logging.INFO, logger="hexo_a0.serving.game"):
        _bot_turn_fn(cfg_kwargs)(rec)

    lines = [r.message for r in caplog.records if "bot placement" in r.message]
    assert len(lines) == 2
    assert "src=pair" in lines[0]
    assert "src=partner" in lines[1]


def test_serve_logging_writes_rotating_file(tmp_path):
    # HEXO_LOG_FILE plumbing: the serve CLI attaches a rotating file handler
    # so logs survive container recreation (docker logs don't).
    from hexo_a0 import cli
    logf = tmp_path / "logs" / "serve.log"
    handler = cli._configure_serve_logging(str(logf))
    try:
        logging.getLogger("hexo_a0.serving.smoke").info("hello file")
        handler.flush()
        assert "hello file" in logf.read_text()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_forcing_kill_switch_never_calls_solver(monkeypatch):
    # live_forcing=False must make `_forcing_move` (and therefore
    # `hexo_rs.solve_forcing`) entirely unreachable, even on a position with a
    # known forced win — byte-identical to pre-forcing-feature MCTS/argmax
    # play. Reuses the same mate-in-2 fork as
    # test_forcing_bot_executes_own_forced_win.
    cfg_kwargs = {"win_length": 4, "placement_radius": 3, "max_moves": 60}
    cfg = hexo_rs.GameConfig(**cfg_kwargs)
    state = hexo_rs.GameState.from_state(
        [((0, 0), "P1"), ((1, 0), "P1")], "P1", 2, cfg)
    assert hexo_rs.solve_forcing(state, 4, 20_000) is not None  # sanity: a win exists

    calls = {"n": 0}

    def _counting_solve(*a, **k):
        calls["n"] += 1
        return None

    monkeypatch.setattr(hexo_rs, "solve_forcing", _counting_solve)
    monkeypatch.setattr(hexo_rs, "solve_defense", _counting_solve)

    now = datetime.now(timezone.utc)
    rec = GameRecord(
        game_id="g-forcing-kill-switch", created_at=now, last_active_at=now,
        state=state, human_side="P2", bot_side="P1", human_name="a",
        move_log=[(0, 0, "P1"), (1, 0, "P1")],
    )
    _bot_turn_fn(cfg_kwargs, live_forcing=False)(rec)

    assert calls["n"] == 0  # neither solver entry point was consulted
    assert rec.forcing_pv is None  # the PV cache was never populated either
