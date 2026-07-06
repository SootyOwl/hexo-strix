"""Champion promotion writes a safetensors twin for Rust evaluators."""
import json
import os

from safetensors import safe_open


def test_export_champion_writes_safetensors_twin(tmp_path, tiny_trainer):
    trainer = tiny_trainer(tmp_path)
    trainer._export_champion()
    st = tmp_path / "self_play" / "model_selfplay.safetensors"
    assert st.exists()
    with safe_open(st, framework="pt") as f:
        meta = f.metadata()
    assert meta["format"] == "hexo-safetensors-v1"
    assert "hidden_dim" in json.loads(meta["model_config"])


def test_safetensors_export_failure_does_not_break_promotion(tmp_path, tiny_trainer, monkeypatch):
    import hexo_a0.trainer as trainer_mod
    trainer = tiny_trainer(tmp_path)
    monkeypatch.setattr(trainer_mod, "save_safetensors",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")),
                        raising=False)
    trainer._export_champion()  # must not raise
    assert (tmp_path / "self_play" / "champion.pt").exists()


def test_safetensors_twin_lands_before_pt_replace(tmp_path, tiny_trainer, monkeypatch):
    """The .safetensors twin's os.replace must land before the .pt's.

    A Rust reloader polls the .pt mtime then reads the .safetensors sibling;
    if the .pt's os.replace happens first, there's a window where the poller
    sees a fresh .pt but a missing/stale twin. Pin the ordering directly by
    recording every os.replace call the trainer module makes.

    trainer.py imports ``os`` locally (function-scoped, not module-level), so
    every call resolves the same singleton ``sys.modules["os"]`` object —
    patching ``os.replace`` directly (rather than an attribute on the trainer
    module, which has none) is "as referenced by the trainer module".
    """
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def recording_replace(src, dst, *a, **k):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", recording_replace)

    trainer = tiny_trainer(tmp_path)
    trainer._export_champion()

    def index_of_dst_suffix(suffix):
        for i, (_src, dst) in enumerate(calls):
            if dst.endswith(suffix):
                return i
        raise AssertionError(f"no os.replace call landed on a {suffix!r} path: {calls}")

    st_idx = index_of_dst_suffix("model_selfplay.safetensors")
    pt_idx = index_of_dst_suffix("model_selfplay.pt")
    assert st_idx < pt_idx, (
        f"safetensors twin replace (index {st_idx}) must land before the "
        f".pt replace (index {pt_idx}); calls={calls}"
    )


def test_pool_snapshot_writes_safetensors_twin(tmp_path, tiny_trainer):
    """Pool snapshots (pool/<name>.pt) also gain a .safetensors sibling."""
    from dataclasses import replace

    trainer = tiny_trainer(tmp_path)
    trainer.self_play_config = replace(
        trainer.self_play_config,
        rust=replace(trainer.self_play_config.rust, pool_fraction=1.0, pool_snapshot_every=1, pool_size=5),
    )
    trainer._export_champion()

    pool_dir = tmp_path / "self_play" / "pool"
    pt_files = sorted(pool_dir.glob("*.pt"))
    assert pt_files, "expected at least one pool .pt snapshot"
    for pt in pt_files:
        assert pt.with_suffix(".safetensors").exists()


def test_pool_rotation_prunes_orphaned_safetensors_twins(tmp_path, tiny_trainer):
    """Rotating past the pool cap must not leave orphaned .safetensors twins.

    Each pool snapshot gets a .safetensors sibling (see the test above). The
    prune loop in ``_rotate_pool_snapshots`` globs and unlinks only ``*.pt``
    files once the pool exceeds ``pool_size`` — so without also unlinking the
    twin, every rotated-away snapshot leaks its .safetensors file forever.
    """
    from dataclasses import replace

    trainer = tiny_trainer(tmp_path)
    trainer.self_play_config = replace(
        trainer.self_play_config,
        rust=replace(trainer.self_play_config.rust, pool_fraction=1.0, pool_snapshot_every=1, pool_size=2),
    )

    # Export enough times to blow well past pool_size=2 so pruning triggers
    # repeatedly. Each export re-writes self_play/model_selfplay.pt, whose
    # mtime_ns is used as the pool destination's unique suffix — real wall
    # time elapses across each export (torch.save + safetensors write), so
    # successive dest names are distinct without needing to fake mtimes.
    for _ in range(6):
        trainer._export_champion()

    pool_dir = tmp_path / "self_play" / "pool"
    pt_files = sorted(pool_dir.glob("*.pt"))
    st_files = sorted(pool_dir.glob("*.safetensors"))
    assert len(pt_files) == 2, f"expected pool pruned to pool_size=2, got {pt_files}"

    pt_stems = {pt.stem for pt in pt_files}
    orphans = [st for st in st_files if st.stem not in pt_stems]
    assert not orphans, f"orphaned .safetensors twins with no .pt sibling: {orphans}"
