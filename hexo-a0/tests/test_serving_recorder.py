"""Tests for the serving SQLite recorder.

Parity target: scripts/play_server.py:106-240, adapted so Recorder creates its
schema on construction and sets busy_timeout.
"""
import sqlite3
from datetime import datetime, timezone

import pytest

from hexo_a0.serving.recorder import GAMES_MIGRATIONS, Recorder


LEGACY = {
    "id", "game_id", "created_at", "completed_at", "human_name", "human_side",
    "bot_side", "winner", "result_type", "n_moves", "mcts_sims", "win_length",
    "placement_radius", "max_moves", "checkpoint_path", "htttx", "moves_json",
    "model_label",
}


def test_schema_has_legacy_columns(tmp_path):
    db = str(tmp_path / "g.sqlite")
    Recorder(db)
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(games)")}
    assert LEGACY.issubset(cols)


def test_summary_empty_on_fresh_db(tmp_path):
    assert Recorder(str(tmp_path / "g.sqlite")).summary()["total"] == 0


def test_busy_timeout_set(tmp_path):
    db = str(tmp_path / "g.sqlite")
    r = Recorder(db)
    assert r._conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000


def test_wal_enabled(tmp_path):
    db = str(tmp_path / "g.sqlite")
    r = Recorder(db)
    journal = r._conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert journal == "wal"


def test_migrations_set_user_version(tmp_path):
    db = str(tmp_path / "g.sqlite")
    r = Recorder(db)
    assert r._conn.execute("PRAGMA user_version").fetchone()[0] == len(GAMES_MIGRATIONS)
    # re-opening runs nothing new and stays at the head version
    r2 = Recorder(db)
    assert r2._conn.execute("PRAGMA user_version").fetchone()[0] == len(GAMES_MIGRATIONS)


def _record_game(r: Recorder, game_id: str = "g1", model_label: str = "test",
                 step: int | None = None):
    now = datetime.now(timezone.utc).isoformat()
    r.record_completed(
        game_id=game_id,
        created_at=now,
        completed_at=now,
        human_name="alice",
        human_side="P2",
        bot_side="P1",
        winner="P2",
        result_type="win",
        n_moves=5,
        mcts_sims=64,
        win_length=6,
        placement_radius=8,
        max_moves=400,
        checkpoint_path="c.pt",
        model_label=model_label,
        step=step,
        move_log=[(0, 0, "P1"), (1, 0, "P2"), (2, 0, "P2"), (3, 0, "P1")],
    )


def test_record_completed_and_summary(tmp_path):
    db = str(tmp_path / "g.sqlite")
    r = Recorder(db)
    _record_game(r)
    s = r.summary()
    assert s["total"] == 1
    assert s["human_wins"] == 1
    assert s["bot_wins"] == 0
    assert s["draws"] == 0
    assert s["resigns"] == 0
    assert s["avg_moves"] == 5.0


def test_recent_games_returns_rows(tmp_path):
    db = str(tmp_path / "g.sqlite")
    r = Recorder(db)
    _record_game(r, game_id="g1")
    rows = r.recent_games(limit=10)
    assert len(rows) == 1
    assert rows[0]["game_id"] == "g1"
    assert "htttx" in rows[0]


def test_bot_brag_groups_by_sims(tmp_path):
    db = str(tmp_path / "g.sqlite")
    r = Recorder(db)
    _record_game(r, game_id="g1", model_label="ml-a")
    _record_game(r, game_id="g2", model_label="ml-a")
    brag = r.bot_brag("ml-a")
    assert brag["model_label"] == "ml-a"
    assert brag["current"]["total"] == 2
    assert brag["all_time"]["total"] == 2
    assert brag["by_sims"]["64"]["total"] == 2


def test_step_column_present(tmp_path):
    db = str(tmp_path / "g.sqlite")
    Recorder(db)
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(games)")}
    assert "step" in cols


def test_bot_brag_is_step_aware(tmp_path):
    # Same label 'strix', two checkpoints (steps 100 & 200) — the current record
    # is scoped to the running step, so retraining champion.pt doesn't merge them.
    r = Recorder(str(tmp_path / "g.sqlite"))
    _record_game(r, game_id="g1", model_label="strix", step=100)
    _record_game(r, game_id="g2", model_label="strix", step=100)
    _record_game(r, game_id="g3", model_label="strix", step=200)
    assert r.bot_brag("strix", 100)["current"]["total"] == 2
    assert r.bot_brag("strix", 200)["current"]["total"] == 1
    assert r.bot_brag("strix", 100)["by_sims"]["64"]["total"] == 2   # by_sims scoped to step too
    assert r.bot_brag("strix", 200)["all_time"]["total"] == 3        # all_time spans steps
    assert r.bot_brag("strix", 200)["step"] == 200


def test_bot_brag_step_none_falls_back_to_label(tmp_path):
    r = Recorder(str(tmp_path / "g.sqlite"))
    _record_game(r, game_id="g1", model_label="strix", step=100)
    _record_game(r, game_id="g2", model_label="strix", step=200)
    # No step given -> current spans all of this label's games (legacy behaviour).
    assert r.bot_brag("strix")["current"]["total"] == 2


def test_opens_legacy_db_without_step_column(tmp_path):
    # A games table that predates the `step` column (an older deployment) must
    # migrate cleanly — the (model_label, step) index is created AFTER the ALTER,
    # so opening must not fail with "no such column: step".
    db = str(tmp_path / "legacy.sqlite")
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE games ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT UNIQUE,"
        " created_at TEXT, completed_at TEXT, human_name TEXT, human_side TEXT,"
        " bot_side TEXT, winner TEXT, result_type TEXT, n_moves INTEGER,"
        " mcts_sims INTEGER, win_length INTEGER, placement_radius INTEGER,"
        " max_moves INTEGER, checkpoint_path TEXT, htttx TEXT, moves_json TEXT,"
        " model_label TEXT);"
        "CREATE INDEX idx_games_model_label ON games(model_label);")
    con.commit()
    con.close()
    r = Recorder(db)   # must not raise
    cols = {row[1] for row in r._conn.execute("PRAGMA table_info(games)")}
    assert "step" in cols
    _record_game(r, game_id="g1", model_label="strix", step=100)   # stats still work
    assert r.bot_brag("strix", 100)["current"]["total"] == 1


def test_model_label_default_unknown(tmp_path):
    db = str(tmp_path / "g.sqlite")
    r = Recorder(db)
    now = datetime.now(timezone.utc).isoformat()
    r.record_completed(
        game_id="g1",
        created_at=now,
        completed_at=now,
        human_name="alice",
        human_side="P2",
        bot_side="P1",
        winner="P1",
        result_type="win",
        n_moves=3,
        mcts_sims=32,
        win_length=6,
        placement_radius=8,
        max_moves=400,
        checkpoint_path="c.pt",
        move_log=[(0, 0, "P1"), (1, 0, "P2")],
    )
    ml = sqlite3.connect(db).execute(
        "SELECT model_label FROM games WHERE game_id='g1'"
    ).fetchone()[0]
    assert ml == "unknown"


def test_elo_columns_added(tmp_path):
    db = str(tmp_path / "g.sqlite")
    Recorder(db)
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(games)")}
    assert {"opp_elo", "elo_source", "opp_handle"}.issubset(cols)


def test_migration_idempotent(tmp_path):
    db = str(tmp_path / "g.sqlite")
    Recorder(db)
    Recorder(db)  # must not raise


def test_legacy_insert_still_works_post_migration(tmp_path):
    # Simulate the frozen script inserting with only legacy columns after migration.
    db = str(tmp_path / "g.sqlite")
    Recorder(db)
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO games (game_id,created_at,completed_at,human_side,bot_side,"
        "result_type,n_moves,mcts_sims,win_length,placement_radius,max_moves,"
        "checkpoint_path,htttx,moves_json) VALUES "
        "('g','t','t','P1','P2','win',5,64,6,8,400,'c.pt','version[1];','[]')"
    )
    con.commit()
    assert con.execute("SELECT opp_elo FROM games WHERE game_id='g'").fetchone()[0] is None


def test_record_completed_persists_elo(tmp_path):
    db = str(tmp_path / "g.sqlite")
    r = Recorder(db)
    now = datetime.now(timezone.utc).isoformat()
    r.record_completed(
        game_id="g1",
        created_at=now,
        completed_at=now,
        human_name="alice",
        human_side="P2",
        bot_side="P1",
        winner="P2",
        result_type="win",
        n_moves=5,
        mcts_sims=64,
        win_length=6,
        placement_radius=8,
        max_moves=400,
        checkpoint_path="c.pt",
        move_log=[(0, 0, "P1"), (1, 0, "P2")],
        opp_elo=1500.0,
        elo_source="self_reported",
    )
    row = sqlite3.connect(db).execute(
        "SELECT opp_elo, elo_source, opp_handle FROM games WHERE game_id='g1'"
    ).fetchone()
    assert row[0] == 1500.0 and row[1] == "self_reported" and row[2] is None
