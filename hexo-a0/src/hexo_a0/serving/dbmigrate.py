"""Tiny SQLite migration runner keyed on ``PRAGMA user_version``.

Each schema is described by an ordered list of migration callables;
``migrations[i]`` upgrades the DB from schema version ``i`` to ``i + 1``. On open
we run every migration not yet applied (index ``>= user_version``) in order, then
bump ``user_version``. This replaces the fragile "run the full CREATE script,
then a pile of ad-hoc ALTERs" pattern where a new index could reference a column
the ALTER hadn't added yet.

Migrations MUST be idempotent (use ``CREATE ... IF NOT EXISTS`` and
:func:`add_column`) so a pre-versioning database can be adopted in place: an old,
un-versioned DB reports ``user_version`` 0, so we re-run the earlier steps —
each a no-op on the already-present objects — until we reach the first step that
genuinely changes it.
"""


def table_columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def add_column(conn, table, col, dtype):
    """Idempotent ``ALTER TABLE ADD COLUMN`` (SQLite's ALTER has no IF NOT EXISTS)."""
    if col not in table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")


def apply_migrations(conn, migrations):
    """Run pending migrations in order, tracking progress in ``user_version``."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i in range(version, len(migrations)):
        migrations[i](conn)
        # PRAGMA can't be parameterised; i is a trusted loop index.
        conn.execute(f"PRAGMA user_version = {i + 1}")
    conn.commit()
