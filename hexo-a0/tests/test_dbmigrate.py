"""Tests for the tiny PRAGMA user_version migration runner."""
import sqlite3

from hexo_a0.serving.dbmigrate import add_column, apply_migrations, table_columns


def test_apply_migrations_creates_and_versions():
    conn = sqlite3.connect(":memory:")
    migs = [
        lambda c: c.execute("CREATE TABLE t (a INTEGER)"),
        lambda c: add_column(c, "t", "b", "TEXT"),
    ]
    apply_migrations(conn, migs)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert "b" in table_columns(conn, "t")


def test_apply_migrations_runs_only_pending():
    conn = sqlite3.connect(":memory:")
    apply_migrations(conn, [lambda c: c.execute("CREATE TABLE t (a INTEGER)")])
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1

    def already_applied(c):
        raise AssertionError("an already-applied migration must not re-run")

    ran = {"n": 0}

    def new_one(c):
        ran["n"] += 1
        add_column(c, "t", "b", "TEXT")

    apply_migrations(conn, [already_applied, new_one])
    assert ran["n"] == 1
    assert "b" in table_columns(conn, "t")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2


def test_add_column_is_idempotent():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a INTEGER)")
    add_column(conn, "t", "b", "TEXT")
    add_column(conn, "t", "b", "TEXT")   # no "duplicate column" error
    assert "b" in table_columns(conn, "t")


def test_adopts_unversioned_db_in_place():
    # A pre-versioning DB (user_version 0) whose table already has some columns:
    # re-running the earlier steps is a no-op; only genuinely-new ones change it.
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")   # 'b' already present
    conn.commit()
    migs = [
        lambda c: c.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER)"),  # no-op
        lambda c: add_column(c, "t", "b", "TEXT"),                        # present -> skip
        lambda c: add_column(c, "t", "c", "INTEGER"),                     # new -> added
    ]
    apply_migrations(conn, migs)
    assert table_columns(conn, "t") == {"a", "b", "c"}
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
