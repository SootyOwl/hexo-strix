"""SQLite-backed replay buffer with FIFO eviction and WAL mode."""

from __future__ import annotations

import logging
import pickle
import random
import shutil
import sqlite3
import threading
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS examples (
    id   INTEGER PRIMARY KEY,
    data BLOB NOT NULL
);
"""

_SQLITE_VAR_LIMIT = 500  # chunk IN(...) queries at this limit


def _example_is_finite(ex: object) -> bool:
    """Return ``False`` if the example carries a non-finite target.

    Defends the persistent buffer against a corrupt self-play network — a
    NaN ``model_selfplay.pt`` emits all-NaN policy distributions, which,
    once written, intermittently NaN the policy loss for the ~tens of
    thousands of steps it takes FIFO eviction to flush them. The trainer's
    NaN-grad guard skips such steps without corrupting weights, but
    filtering at the write boundary stops the poison at its source.

    Examples lacking the target attributes are kept (treated as finite) so
    the buffer stays type-agnostic.
    """
    pt = getattr(ex, "policy_target", None)
    if pt is not None:
        try:
            import torch

            if torch.is_tensor(pt) and not bool(torch.isfinite(pt).all()):
                return False
        except Exception:  # pragma: no cover - torch optional / unexpected type
            pass
    vt = getattr(ex, "value_target", None)
    if isinstance(vt, float):
        import math

        if not math.isfinite(vt):
            return False
    return True


def _open_conn(db_path: str, *, uri: bool = False) -> sqlite3.Connection:
    """Open a connection with WAL mode and synchronous=NORMAL."""
    conn = sqlite3.connect(db_path, check_same_thread=False, uri=uri)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


class ReplayBuffer:
    """Fixed-capacity replay buffer backed by SQLite.

    Provides FIFO eviction, O(1) ``__len__``, and O(n log N) sampling via
    contiguous application-managed IDs.
    """

    # Class-level counter for unique in-memory DB names
    _mem_counter: int = 0
    _mem_counter_lock = threading.Lock()

    def __init__(self, capacity: int, *, db_path: str = ":memory:") -> None:
        if capacity <= 0:
            raise ValueError(f"ReplayBuffer capacity must be > 0, got {capacity}")
        self._capacity = capacity
        self._db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._lock = threading.Lock()
        self._closed = False
        self._integrity_ok = True

        self._open_connections()
        self._writer.execute(_SCHEMA)
        self._writer.commit()

        self._next_id: int = 0
        self._min_id: int = 0
        self._count: int = 0
        self._refresh_bounds()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_connections(self) -> None:
        """Open writer and reader connections."""
        if self._is_memory:
            # Use shared-cache URI so both connections see the same DB
            with ReplayBuffer._mem_counter_lock:
                ReplayBuffer._mem_counter += 1
                self._mem_id = ReplayBuffer._mem_counter
            uri = f"file:hexo_replay_{self._mem_id}?mode=memory&cache=shared"
            self._writer = _open_conn(uri, uri=True)
            self._reader = _open_conn(uri, uri=True)
        else:
            self._writer = _open_conn(self._db_path)
            self._reader = _open_conn(self._db_path)

    def _refresh_bounds(self) -> None:
        """Recover ``_next_id``, ``_min_id``, ``_count`` from the DB."""
        row = self._writer.execute(
            "SELECT MIN(id), MAX(id), COUNT(*) FROM examples"
        ).fetchone()
        count = row[2]
        if count == 0:
            self._next_id = 0
            self._min_id = 0
            self._count = 0
        else:
            min_id, max_id = row[0], row[1]
            self._min_id = min_id
            self._next_id = max_id + 1
            self._count = count
            # Integrity check: IDs must be contiguous
            expected = max_id - min_id + 1
            if count != expected:
                log.warning(
                    "Replay buffer integrity check failed: "
                    "COUNT(*)=%d but max_id-min_id+1=%d. "
                    "Falling back to ORDER BY RANDOM() for sampling.",
                    count,
                    expected,
                )
                self._integrity_ok = False

    def _evict_and_insert(self, blobs: list[tuple[int, bytes]]) -> None:
        """Insert rows and evict oldest if over capacity, in one transaction.

        Must be called with ``self._lock`` held.
        """
        n_after = self._count + len(blobs)
        n_to_evict = max(0, n_after - self._capacity)

        with self._writer:  # transaction context manager
            if n_to_evict > 0:
                evict_bound = self._min_id + n_to_evict
                self._writer.execute(
                    "DELETE FROM examples WHERE id < ?", (evict_bound,)
                )
                self._min_id = evict_bound

            self._writer.executemany(
                "INSERT INTO examples (id, data) VALUES (?, ?)", blobs
            )

        self._count = self._count + len(blobs) - n_to_evict
        self._next_id = blobs[-1][0] + 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, example: object) -> None:
        """Append a single training example."""
        self.add_many([example])

    def add_many(self, examples: list) -> None:
        """Append an iterable of training examples."""
        examples = list(examples)
        if not examples:
            return

        # Drop any example carrying a non-finite policy/value target before
        # it can poison the persistent buffer (see _example_is_finite).
        finite = [ex for ex in examples if _example_is_finite(ex)]
        dropped = len(examples) - len(finite)
        if dropped:
            log.warning(
                "replay_buffer: dropped %d example(s) with non-finite "
                "targets (corrupt self-play network?)",
                dropped,
            )
        examples = finite
        if not examples:
            return

        # If more items than capacity, keep only the last `capacity` items
        if len(examples) > self._capacity:
            examples = examples[-self._capacity:]

        # Serialize OUTSIDE the lock (REQ-7)
        serialized: list[bytes] = [pickle.dumps(ex) for ex in examples]

        with self._lock:
            # Assign contiguous IDs
            start_id = self._next_id
            blobs = [
                (start_id + i, data) for i, data in enumerate(serialized)
            ]
            self._evict_and_insert(blobs)

    def sample(self, n: int) -> list:
        """Return *n* examples drawn uniformly at random.

        * ``n <= len(buffer)``: sampling **without** replacement.
        * ``n > len(buffer)``: sampling **with** replacement.

        Raises
        ------
        ValueError
            If the buffer is empty.
        """
        # Snapshot counters under lock (fast)
        with self._lock:
            count = self._count
            min_id = self._min_id
            integrity_ok = self._integrity_ok

        if count == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Generate random IDs (outside lock)
        if not integrity_ok:
            return self._sample_fallback(n)

        max_retries = 3
        for attempt in range(max_retries + 1):
            if n <= count:
                ids = random.sample(range(min_id, min_id + count), n)
            else:
                ids = random.choices(range(min_id, min_id + count), k=n)

            rows = self._fetch_by_ids(ids)

            if len(rows) == len(ids):
                # Attach a transient `_age` attribute to each example: the
                # example's relative position in the buffer at sample time.
                # 0.0 = newest, 1.0 = oldest. Used downstream for staleness
                # instrumentation (see staleness/* metrics in trainer).
                max_id = min_id + count - 1
                denom = max(1, max_id - min_id)
                out = []
                for blob, sid in zip(rows, ids):
                    ex = pickle.loads(blob)
                    ex._age = (max_id - sid) / denom
                    ex._sid = sid
                    out.append(ex)
                return out

            if attempt < max_retries:
                # Refresh snapshot for retry
                with self._lock:
                    count = self._count
                    min_id = self._min_id
                continue

            # Fall back to ORDER BY RANDOM()
            return self._sample_fallback(n)

        # Unreachable, but satisfies type checker
        return self._sample_fallback(n)  # pragma: no cover

    def _fetch_by_ids(self, ids: list[int]) -> list[bytes]:
        """Fetch blobs for the given IDs, chunking if needed (REQ-19).

        Returns blobs in the same order as ``ids`` (duplicates preserved).
        """
        # Build a mapping from id -> list of blob bytes
        unique_ids = set(ids)
        id_to_blob: dict[int, bytes] = {}

        unique_list = list(unique_ids)
        for start in range(0, len(unique_list), _SQLITE_VAR_LIMIT):
            chunk = unique_list[start : start + _SQLITE_VAR_LIMIT]
            placeholders = ",".join("?" * len(chunk))
            cursor = self._reader.execute(
                f"SELECT id, data FROM examples WHERE id IN ({placeholders})",
                chunk,
            )
            for row_id, data in cursor:
                id_to_blob[row_id] = data

        # Reconstruct in original order; missing IDs cause short return
        result = []
        for i in ids:
            blob = id_to_blob.get(i)
            if blob is not None:
                result.append(blob)
        return result

    def _sample_fallback(self, n: int) -> list:
        """Fallback sampling using ORDER BY RANDOM()."""
        with self._lock:
            count = self._count
            min_id = self._min_id
        if count == 0:
            raise ValueError("Cannot sample from empty buffer")

        max_id = min_id + count - 1
        denom = max(1, max_id - min_id)

        def _decode(rows):
            out = []
            for sid, blob in rows:
                ex = pickle.loads(blob)
                ex._age = (max_id - sid) / denom
                ex._sid = sid
                out.append(ex)
            return out

        if n <= count:
            rows = self._reader.execute(
                "SELECT id, data FROM examples ORDER BY RANDOM() LIMIT ?", (n,)
            ).fetchall()
        else:
            # With replacement: sample count items, then extend
            rows = self._reader.execute(
                "SELECT id, data FROM examples ORDER BY RANDOM() LIMIT ?",
                (count,),
            ).fetchall()
            items = _decode(rows)
            return random.choices(items, k=n)

        return _decode(rows)

    def clear(self) -> None:
        """Remove all examples and reset the buffer."""
        with self._lock:
            if self._is_memory:
                self._writer.execute("DELETE FROM examples;")
                self._writer.commit()
            else:
                # Close connections, delete DB files, reopen
                self._writer.close()
                self._reader.close()
                db = Path(self._db_path)
                for suffix in ("", "-wal", "-shm"):
                    p = Path(str(db) + suffix)
                    if p.exists():
                        p.unlink()
                self._writer = _open_conn(self._db_path)
                self._reader = _open_conn(self._db_path)
                self._writer.execute(_SCHEMA)
                self._writer.commit()

            self._next_id = 0
            self._min_id = 0
            self._count = 0
            self._integrity_ok = True

    def close(self) -> None:
        """Checkpoint WAL and close both connections."""
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        try:
            self._writer.close()
        except Exception:
            pass
        try:
            self._reader.close()
        except Exception:
            pass

    def backup(self, dest: str | Path) -> None:
        """Create an atomic backup of the DB to *dest*."""
        dest = str(dest)
        dst_conn = sqlite3.connect(dest)
        try:
            self._reader.backup(dst_conn)
        finally:
            dst_conn.close()

    def get_state(self) -> Path:
        """Return the DB file path (REQ-15)."""
        return Path(self._db_path)

    def set_state(self, source: Path | list) -> None:
        """Restore buffer from a Path (DB file) or a list of examples.

        REQ-16a: If Path, replace DB file and reopen.
        REQ-16b: If list, clear and bulk-insert.
        """
        if isinstance(source, Path):
            if self._is_memory:
                # Can't replace file for in-memory DB; load examples via list
                src_conn = sqlite3.connect(str(source))
                rows = src_conn.execute(
                    "SELECT data FROM examples ORDER BY id"
                ).fetchall()
                src_conn.close()
                examples = [pickle.loads(r[0]) for r in rows]
                self.clear()
                if examples:
                    self.add_many(examples)
            else:
                db = Path(self._db_path)
                if source.resolve() == db.resolve():
                    # Source is the live buffer — just refresh bounds
                    self._refresh_bounds()
                    return
                with self._lock:
                    self._writer.close()
                    self._reader.close()
                    # Delete WAL/SHM of existing DB
                    for suffix in ("-wal", "-shm"):
                        p = Path(str(db) + suffix)
                        if p.exists():
                            p.unlink()
                    # Replace DB file
                    shutil.copy2(str(source), str(db))
                    # Reopen
                    self._writer = _open_conn(self._db_path)
                    self._reader = _open_conn(self._db_path)
                    self._integrity_ok = True
                    self._refresh_bounds()
        elif isinstance(source, list):
            self.clear()
            if source:
                self.add_many(source)
        else:
            raise TypeError(f"set_state expects Path or list, got {type(source)}")

    def __len__(self) -> int:
        return self._count

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
