"""SQLite-backed completed-game recorder.

Ported from scripts/play_server.py:80-240 with two intentional changes:

1. Schema is created on ``Recorder(db)`` construction — there is no separate
   ``init_db()`` method.
2. The connection sets ``PRAGMA busy_timeout = 5000`` to tolerate brief writer
   contention (the frozen script leaves it at the default 0 ms).

All other behavior matches the frozen script so the shared ``games.sqlite``
remains readable by either implementation.
"""
import json
import sqlite3
import threading

from hexo_a0.serving.dbmigrate import add_column, apply_migrations
from hexo_a0.serving.htttx import serialize_htttx


# Ordered, idempotent schema migrations for the games DB (see dbmigrate). Each
# upgrades from version i to i+1; append new ones, never reorder/rewrite.
def _m0_base(conn):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS games (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             game_id TEXT NOT NULL UNIQUE,
             created_at TEXT NOT NULL,
             completed_at TEXT NOT NULL,
             human_name TEXT,
             human_side TEXT NOT NULL,
             bot_side TEXT NOT NULL,
             winner TEXT,
             result_type TEXT NOT NULL,
             n_moves INTEGER NOT NULL,
             mcts_sims INTEGER NOT NULL,
             win_length INTEGER NOT NULL,
             placement_radius INTEGER NOT NULL,
             max_moves INTEGER NOT NULL,
             checkpoint_path TEXT NOT NULL,
             htttx TEXT NOT NULL,
             moves_json TEXT NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_completed ON games(completed_at)")


def _m1_model_label(conn):
    add_column(conn, "games", "model_label", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_model_label ON games(model_label)")


def _m2_opponent_elo(conn):
    add_column(conn, "games", "opp_elo", "REAL")
    add_column(conn, "games", "elo_source", "TEXT")
    add_column(conn, "games", "opp_handle", "TEXT")


def _m3_step(conn):
    add_column(conn, "games", "step", "INTEGER")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_model_step ON games(model_label, step)")


GAMES_MIGRATIONS = [_m0_base, _m1_model_label, _m2_opponent_elo, _m3_step]


class Recorder:
    """Thread-safe SQLite-backed completed-game store."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        # check_same_thread=False because handler threads share this connection.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            apply_migrations(self._conn, GAMES_MIGRATIONS)

    def record_completed(
        self,
        *,
        game_id: str,
        created_at: str,
        completed_at: str,
        human_name: str | None,
        human_side: str,
        bot_side: str,
        winner: str | None,
        result_type: str,
        n_moves: int,
        mcts_sims: int,
        win_length: int,
        placement_radius: int,
        max_moves: int,
        checkpoint_path: str,
        model_label: str = "unknown",
        step: int | None = None,
        move_log: list[tuple[int, int, str]],
        opp_elo: float | None = None,
        elo_source: str | None = None,
        opp_handle: str | None = None,
    ) -> None:
        """Persist a finished game."""
        htttx = serialize_htttx(move_log)
        moves_json = json.dumps([[q, r, p] for (q, r, p) in move_log])
        with self._lock:
            self._conn.execute(
                """INSERT INTO games (game_id, created_at, completed_at, human_name,
                       human_side, bot_side, winner, result_type, n_moves, mcts_sims,
                       win_length, placement_radius, max_moves, checkpoint_path,
                       htttx, moves_json, model_label, step, opp_elo, elo_source, opp_handle)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (game_id, created_at, completed_at, human_name, human_side, bot_side,
                 winner, result_type, n_moves, mcts_sims, win_length, placement_radius,
                 max_moves, checkpoint_path, htttx, moves_json, model_label, step,
                 opp_elo, elo_source, opp_handle),
            )
            self._conn.commit()

    def recent_games(self, limit: int = 100) -> list[dict]:
        """Most recent completed games, newest first."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT id, game_id, created_at, completed_at, human_name,
                          human_side, bot_side, winner, result_type, n_moves,
                          mcts_sims, htttx, model_label, step,
                          opp_elo, elo_source, opp_handle
                   FROM games ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def summary(self) -> dict:
        """Aggregate stats across all completed games."""
        return self._stats_where("1=1", ())

    def bot_brag(self, model_label: str, step: int | None = None) -> dict:
        """Stats for the CURRENT checkpoint (model_label + training step) plus the
        all-time aggregate. Keying on step matters when the label is held fixed
        (e.g. --model-label strix) while champion.pt is retrained: each step's
        record stays separate instead of merging across versions. When step is
        None (unavailable) the current-checkpoint filter falls back to label."""
        if step is not None:
            where, params = "model_label = ? AND step = ?", (model_label, step)
        else:
            where, params = "model_label = ?", (model_label,)
        return {
            "model_label": model_label,
            "step": step,
            "current": self._stats_where(where, params),
            "all_time": self._stats_where("1=1", ()),
            "by_sims": self._stats_by_sims(where, params),
        }

    def _stats_where(self, where_sql: str, params: tuple) -> dict:
        with self._lock:
            row = self._conn.execute(
                f"""SELECT COUNT(*),
                           SUM(CASE WHEN winner = human_side THEN 1 ELSE 0 END),
                           SUM(CASE WHEN winner = bot_side   THEN 1 ELSE 0 END),
                           SUM(CASE WHEN winner IS NULL      THEN 1 ELSE 0 END),
                           SUM(CASE WHEN result_type='resign' THEN 1 ELSE 0 END),
                           AVG(n_moves)
                    FROM games WHERE {where_sql}""",
                params,
            ).fetchone()
        return {
            "total": row[0] or 0,
            "human_wins": row[1] or 0,
            "bot_wins": row[2] or 0,
            "draws": row[3] or 0,
            "resigns": row[4] or 0,
            "avg_moves": float(row[5]) if row[5] is not None else 0.0,
        }

    def _stats_by_sims(self, where_sql: str, params: tuple) -> dict:
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT mcts_sims,
                          COUNT(*),
                          SUM(CASE WHEN winner = human_side THEN 1 ELSE 0 END),
                          SUM(CASE WHEN winner = bot_side   THEN 1 ELSE 0 END),
                          SUM(CASE WHEN winner IS NULL      THEN 1 ELSE 0 END),
                          SUM(CASE WHEN result_type='resign' THEN 1 ELSE 0 END),
                          AVG(n_moves)
                   FROM games WHERE {where_sql}
                   GROUP BY mcts_sims""",
                params,
            ).fetchall()
        return {
            str(r[0]): {
                "total": r[1] or 0,
                "human_wins": r[2] or 0,
                "bot_wins": r[3] or 0,
                "draws": r[4] or 0,
                "resigns": r[5] or 0,
                "avg_moves": float(r[6]) if r[6] is not None else 0.0,
            }
            for r in rows
        }
