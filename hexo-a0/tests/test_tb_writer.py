"""Tests for DescribedWriter wrapper."""

from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter

from hexo_a0.tb_writer import DescribedWriter


def test_described_writer_caches_tag(tmp_path):
    inner = SummaryWriter(log_dir=str(tmp_path))
    writer = DescribedWriter(inner)

    # First write with description should mark the tag as described.
    writer.add_scalar("foo/bar", 0.5, 0, description="A **test** metric.")
    assert "foo/bar" in writer._described

    # Second write (with or without description) should not raise.
    writer.add_scalar("foo/bar", 0.75, 1)
    writer.add_scalar("foo/bar", 0.9, 2, description="Ignored second time.")

    # A plain add_scalar for a new tag without description should work and
    # must not be recorded as described.
    writer.add_scalar("baz/qux", 1.0, 0)
    assert "baz/qux" not in writer._described

    # __getattr__ delegation: flush/close should forward to the inner writer.
    writer.flush()
    writer.close()


def test_described_writer_roundtrip(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    inner = SummaryWriter(log_dir=str(tmp_path))
    writer = DescribedWriter(inner)

    writer.add_scalar("test/foo", 1.0, 0, description="this is the foo metric")
    writer.add_scalar("test/foo", 2.0, 1)
    writer.add_scalar("test/foo", 3.0, 2)
    writer.flush()
    writer.close()

    ea = EventAccumulator(str(tmp_path))
    ea.Reload()

    metadata = ea.SummaryMetadata("test/foo")
    assert metadata.summary_description == "this is the foo metric"
    assert metadata.plugin_data.plugin_name == "scalars"
    assert len(ea.Scalars("test/foo")) >= 2
