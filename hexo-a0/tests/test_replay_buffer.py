"""Tests for SQLite-backed ReplayBuffer."""

import pickle
import sqlite3
import threading

import pytest
import torch
from torch_geometric.data import Data

from hexo_a0.replay_buffer import ReplayBuffer
from hexo_a0.self_play import TrainingExample


def _make_example(label: int = 0) -> TrainingExample:
    """Create a minimal TrainingExample for testing."""
    data = Data(x=torch.randn(3, 8), edge_index=torch.zeros(2, 0, dtype=torch.long))
    return TrainingExample(
        data=data,
        policy_target=torch.tensor([0.5, 0.5]),
        value_target=float(label),
    )


def _values(examples: list) -> list[float]:
    """Extract value_targets from a list of TrainingExamples."""
    return [ex.value_target for ex in examples]


# ---------------------------------------------------------------------------
# Basic add / len
# ---------------------------------------------------------------------------


def test_add_increases_length():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    buf.add(_make_example(1))
    assert len(buf) == 1
    buf.add(_make_example(2))
    assert len(buf) == 2


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=-1)


# ---------------------------------------------------------------------------
# FIFO eviction (REQ-5)
# ---------------------------------------------------------------------------


def test_capacity_enforced_oldest_evicted():
    """Adding capacity+1 items should evict the first one."""
    buf = ReplayBuffer(capacity=3)
    for i in range(4):
        buf.add(_make_example(i))
    assert len(buf) == 3
    sampled = buf.sample(3)
    values = sorted(_values(sampled))
    assert values == [1.0, 2.0, 3.0], "Oldest item (0) should have been evicted"


def test_fifo_order_with_add_many():
    """add_many [1,2,3] with capacity=2 keeps [2,3]."""
    buf = ReplayBuffer(capacity=2)
    buf.add_many([_make_example(i) for i in [1, 2, 3]])
    assert len(buf) == 2
    sampled = buf.sample(2)
    values = sorted(_values(sampled))
    assert values == [2.0, 3.0]


def test_add_many_more_than_capacity(tmp_path):
    """REQ-5d: add_many with more items than capacity keeps only last capacity."""
    buf = ReplayBuffer(capacity=3)
    buf.add_many([_make_example(i) for i in range(10)])
    assert len(buf) == 3
    sampled = buf.sample(3)
    values = sorted(_values(sampled))
    assert values == [7.0, 8.0, 9.0]


# ---------------------------------------------------------------------------
# sample() (REQ-6)
# ---------------------------------------------------------------------------


def test_sample_returns_n_items_when_n_lte_len():
    buf = ReplayBuffer(capacity=10)
    buf.add_many([_make_example(i) for i in range(10)])
    result = buf.sample(5)
    assert len(result) == 5
    for item in result:
        assert isinstance(item, TrainingExample)


def test_sample_no_duplicates_when_n_lte_len():
    """Without-replacement sampling should not repeat items."""
    buf = ReplayBuffer(capacity=10)
    buf.add_many([_make_example(i) for i in range(10)])
    result = buf.sample(10)
    values = _values(result)
    assert len(values) == len(set(values))


def test_sample_with_replacement_when_n_gt_len():
    """When n > len, sample returns exactly n items (with replacement)."""
    buf = ReplayBuffer(capacity=3)
    buf.add_many([_make_example(i) for i in [10, 20, 30]])
    result = buf.sample(9)
    assert len(result) == 9
    for item in result:
        assert isinstance(item, TrainingExample)


def test_sample_empty_buffer_raises():
    buf = ReplayBuffer(capacity=5)
    with pytest.raises(ValueError):
        buf.sample(1)


# ---------------------------------------------------------------------------
# add_many
# ---------------------------------------------------------------------------


def test_add_many_adds_multiple_examples():
    buf = ReplayBuffer(capacity=10)
    buf.add_many([_make_example(i) for i in [1, 2, 3]])
    assert len(buf) == 3
    sampled = buf.sample(3)
    values = sorted(_values(sampled))
    assert values == [1.0, 2.0, 3.0]


def test_add_many_empty_list():
    buf = ReplayBuffer(capacity=10)
    buf.add_many([])
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# Non-finite target filtering (corrupt self-play network guard)
# ---------------------------------------------------------------------------


def _nan_policy_example(label: int = 0) -> TrainingExample:
    ex = _make_example(label)
    ex.policy_target = torch.tensor([float("nan"), 0.5])
    return ex


def _inf_value_example(label: int = 0) -> TrainingExample:
    ex = _make_example(label)
    ex.value_target = float("inf")
    return ex


def test_add_many_drops_nan_policy_targets():
    """A NaN self-play network must not poison the persistent buffer."""
    buf = ReplayBuffer(capacity=10)
    buf.add_many([_make_example(1), _nan_policy_example(2), _make_example(3)])
    assert len(buf) == 2
    assert sorted(_values(buf.sample(2))) == [1.0, 3.0]


def test_add_many_drops_non_finite_value_targets():
    buf = ReplayBuffer(capacity=10)
    buf.add_many([_make_example(1), _inf_value_example(2)])
    assert len(buf) == 1
    assert _values(buf.sample(1)) == [1.0]


def test_add_many_all_non_finite_is_noop():
    buf = ReplayBuffer(capacity=10)
    buf.add_many([_nan_policy_example(1), _inf_value_example(2)])
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# get_state / set_state (REQ-15, REQ-16)
# ---------------------------------------------------------------------------


def test_get_state_returns_path():
    buf = ReplayBuffer(capacity=5)
    state = buf.get_state()
    from pathlib import Path
    assert isinstance(state, Path)


def test_set_state_from_list():
    """REQ-16b: set_state(list) clears and bulk-inserts."""
    buf = ReplayBuffer(capacity=5)
    buf.add_many([_make_example(i) for i in range(3)])
    examples = [_make_example(i) for i in [10, 20, 30]]
    buf.set_state(examples)
    assert len(buf) == 3
    sampled = buf.sample(3)
    values = sorted(_values(sampled))
    assert values == [10.0, 20.0, 30.0]


def test_set_state_from_path(tmp_path):
    """REQ-16a: set_state(Path) restores from a DB file."""
    db1 = tmp_path / "source.db"
    buf1 = ReplayBuffer(capacity=5, db_path=str(db1))
    buf1.add_many([_make_example(i) for i in range(3)])
    buf1.close()

    db2 = tmp_path / "dest.db"
    buf2 = ReplayBuffer(capacity=5, db_path=str(db2))
    buf2.add(_make_example(99))
    buf2.set_state(db1)
    assert len(buf2) == 3
    sampled = buf2.sample(3)
    values = sorted(_values(sampled))
    assert values == [0.0, 1.0, 2.0]
    buf2.close()


# ---------------------------------------------------------------------------
# clear() (REQ-9)
# ---------------------------------------------------------------------------


def test_clear_empties_buffer():
    buf = ReplayBuffer(capacity=10)
    buf.add_many([_make_example(i) for i in range(5)])
    assert len(buf) == 5
    buf.clear()
    assert len(buf) == 0


def test_clear_allows_new_additions():
    buf = ReplayBuffer(capacity=10)
    buf.add_many([_make_example(i) for i in range(5)])
    buf.clear()
    buf.add(_make_example(99))
    assert len(buf) == 1
    sampled = buf.sample(1)
    assert sampled[0].value_target == 99.0


def test_clear_then_fifo_eviction_works():
    buf = ReplayBuffer(capacity=3)
    buf.add_many([_make_example(i) for i in range(3)])
    buf.clear()
    buf.add_many([_make_example(i) for i in range(4)])
    assert len(buf) == 3
    sampled = buf.sample(3)
    values = sorted(_values(sampled))
    assert 0.0 not in values
    assert values == [1.0, 2.0, 3.0]


def test_clear_file_backed(tmp_path):
    """REQ-9: clear on file-backed DB deletes WAL/SHM files."""
    db_path = tmp_path / "test.db"
    buf = ReplayBuffer(capacity=10, db_path=str(db_path))
    buf.add_many([_make_example(i) for i in range(5)])
    buf.clear()
    assert len(buf) == 0
    # DB file should be small (just schema, no data)
    assert db_path.stat().st_size < 8192
    # Can add new data
    buf.add(_make_example(42))
    assert len(buf) == 1
    buf.close()


def test_clear_memory_db():
    """REQ-9: clear on :memory: DB works."""
    buf = ReplayBuffer(capacity=10)
    buf.add_many([_make_example(i) for i in range(5)])
    buf.clear()
    assert len(buf) == 0
    buf.add(_make_example(1))
    assert len(buf) == 1


# ---------------------------------------------------------------------------
# Persistence (REQ-3)
# ---------------------------------------------------------------------------


def test_persistence_across_close_reopen(tmp_path):
    """REQ-3: Data persists across close/reopen."""
    db_path = tmp_path / "persist.db"
    buf = ReplayBuffer(capacity=10, db_path=str(db_path))
    buf.add_many([_make_example(i) for i in range(5)])
    buf.close()

    buf2 = ReplayBuffer(capacity=10, db_path=str(db_path))
    assert len(buf2) == 5
    sampled = buf2.sample(5)
    values = sorted(_values(sampled))
    assert values == [0.0, 1.0, 2.0, 3.0, 4.0]
    buf2.close()


def test_wal_mode_active(tmp_path):
    """REQ-3: WAL journal mode is active."""
    db_path = tmp_path / "wal.db"
    buf = ReplayBuffer(capacity=10, db_path=str(db_path))
    row = buf._writer.execute("PRAGMA journal_mode;").fetchone()
    assert row[0] == "wal"
    buf.close()


# ---------------------------------------------------------------------------
# close() (REQ-10)
# ---------------------------------------------------------------------------


def test_close_idempotent():
    """REQ-10c: Calling close() twice is safe."""
    buf = ReplayBuffer(capacity=10)
    buf.add(_make_example(1))
    buf.close()
    buf.close()  # Should not raise


# ---------------------------------------------------------------------------
# backup() (REQ-11)
# ---------------------------------------------------------------------------


def test_backup_creates_valid_copy(tmp_path):
    """REQ-11: backup() creates a valid SQLite DB copy."""
    db_path = tmp_path / "main.db"
    buf = ReplayBuffer(capacity=10, db_path=str(db_path))
    buf.add_many([_make_example(i) for i in range(5)])

    backup_path = tmp_path / "backup.db"
    buf.backup(backup_path)

    # Open backup and verify contents
    conn = sqlite3.connect(str(backup_path))
    count = conn.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
    assert count == 5
    conn.close()

    # Also verify via ReplayBuffer
    buf2 = ReplayBuffer(capacity=10, db_path=str(backup_path))
    assert len(buf2) == 5
    buf2.close()
    buf.close()


# ---------------------------------------------------------------------------
# Contiguous ID integrity (REQ-4, REQ-14)
# ---------------------------------------------------------------------------


def test_contiguous_ids_after_operations(tmp_path):
    """REQ-4b: COUNT(*) == max_id - min_id + 1 after add/evict."""
    db_path = tmp_path / "ids.db"
    buf = ReplayBuffer(capacity=5, db_path=str(db_path))

    # Add 10 items (causes evictions)
    for i in range(10):
        buf.add(_make_example(i))

    row = buf._writer.execute(
        "SELECT MIN(id), MAX(id), COUNT(*) FROM examples"
    ).fetchone()
    min_id, max_id, count = row
    assert count == 5
    assert count == max_id - min_id + 1
    buf.close()


def test_integrity_check_non_contiguous(tmp_path):
    """REQ-14: Non-contiguous IDs set fallback flag."""
    db_path = tmp_path / "broken.db"
    # Create a DB with non-contiguous IDs manually
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        "CREATE TABLE examples (id INTEGER PRIMARY KEY, data BLOB NOT NULL);"
    )
    # Insert non-contiguous IDs: 1, 3, 5
    for i in [1, 3, 5]:
        conn.execute(
            "INSERT INTO examples (id, data) VALUES (?, ?)",
            (i, pickle.dumps(_make_example(i))),
        )
    conn.commit()
    conn.close()

    buf = ReplayBuffer(capacity=10, db_path=str(db_path))
    assert not buf._integrity_ok
    # Sampling should still work via fallback
    result = buf.sample(2)
    assert len(result) == 2
    buf.close()


# ---------------------------------------------------------------------------
# ID recovery on reopen (REQ-4c)
# ---------------------------------------------------------------------------


def test_id_recovery_on_reopen(tmp_path):
    """REQ-4c: IDs continue from where they left off after reopen."""
    db_path = tmp_path / "recover.db"
    buf = ReplayBuffer(capacity=10, db_path=str(db_path))
    buf.add_many([_make_example(i) for i in range(5)])
    assert buf._next_id == 5
    buf.close()

    buf2 = ReplayBuffer(capacity=10, db_path=str(db_path))
    assert buf2._next_id == 5
    assert buf2._min_id == 0
    assert buf2._count == 5
    buf2.add(_make_example(99))
    assert buf2._next_id == 6
    assert len(buf2) == 6
    buf2.close()


# ---------------------------------------------------------------------------
# Concurrency (REQ-8)
# ---------------------------------------------------------------------------


def test_concurrent_writer_reader(tmp_path):
    """Concurrent writer + reader threads, many operations."""
    db_path = tmp_path / "concurrent.db"
    buf = ReplayBuffer(capacity=100, db_path=str(db_path))
    errors: list[str] = []

    def writer():
        try:
            for i in range(200):
                buf.add(_make_example(i))
        except Exception as e:
            errors.append(f"writer: {e}")

    def reader():
        try:
            for _ in range(200):
                if len(buf) > 0:
                    result = buf.sample(min(5, len(buf)))
                    assert len(result) > 0
        except Exception as e:
            errors.append(f"reader: {e}")

    t_write = threading.Thread(target=writer)
    t_read = threading.Thread(target=reader)
    t_write.start()
    t_read.start()
    t_write.join(timeout=30)
    t_read.join(timeout=30)

    assert not errors, f"Concurrent errors: {errors}"
    assert len(buf) == 100
    buf.close()


# ---------------------------------------------------------------------------
# __del__ (REQ-13)
# ---------------------------------------------------------------------------


def test_del_calls_close():
    """REQ-13: __del__ closes without raising."""
    buf = ReplayBuffer(capacity=5)
    buf.add(_make_example(1))
    buf.__del__()
    assert buf._closed


# ---------------------------------------------------------------------------
# Large IN() chunking (REQ-19)
# ---------------------------------------------------------------------------


def test_large_sample_chunks_query():
    """REQ-19: Sampling more than 500 items works (chunked queries)."""
    buf = ReplayBuffer(capacity=1000)
    buf.add_many([_make_example(i) for i in range(800)])
    result = buf.sample(700)
    assert len(result) == 700
