"""Tests for the HX07 board-state self-play record format.

HX07 ships board state (placed stones + current player + moves_remaining +
legal-move count + policy + value) instead of pre-built graphs. The Python
trainer parses these and reconstructs ``game_state`` via
``hexo_rs.GameState.from_state(...)``, producing lazy-reconstruction
``TrainingExample(game_state=..., data=None)`` examples. Legacy HX05/HX06
graph records must STILL ingest (legacy drain).
"""

import io
import struct

import pytest
import torch

import hexo_rs
from hexo_a0.config import FullConfig, ModelConfig, TrainingConfig
from hexo_a0.model import HeXONet
from hexo_a0.trainer import Trainer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def make_trainer():
    """Return a builder for a minimal CPU Trainer.

    Mirrors ``test_jsonl_graph_format._build_trainer``: exposes
    ``trainer.game_config`` and ``trainer.model_config``.
    """
    def _make(graph_type="axis", threat_features=False,
              relative_stone_encoding=False):
        conv = "gatv2" if graph_type == "hex" else "gine"
        mc = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1,
                         graph_type=graph_type, conv_type=conv,
                         prune_empty_edges=False,
                         threat_features=threat_features,
                         relative_stone_encoding=relative_stone_encoding)
        cfg = FullConfig(model=mc, training=TrainingConfig(
            batch_size=8, buffer_capacity=100, edge_budget=0,
            grad_accumulation=False, lr_schedule="constant", lr_warmup_steps=0,
            total_train_steps=0, target_reuse=0.0, draw_value=0.0,
            augment_symmetries=False,
        ))
        gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        model = HeXONet(mc)
        return Trainer(model, cfg, gc, torch.device("cpu"))
    return _make


_PLAYER_BYTE = {"P1": 0, "P2": 1}


def _write_hx07_record(path, stones, current_player, moves_remaining, policy,
                       value, *, sample_weight=1.0, winner="P1", move_count=6,
                       window_stats=None):
    """Hand-build ONE HX07 record (single example) and write it to ``path``.

    Byte layout must match ``write_game_binary_hx07`` in
    ``hexo-rs/hexo-mcts/src/bin/self_play.rs``:

    Envelope: ``[4B "HX07"][4B LE size][4B num_examples u32][1B winner i8]
    [4B move_count u32][1B has_window u8][opt 12B window stats]``. ``size``
    counts everything after the size field.

    Example: ``[2B num_stones u16][1B current_player u8][1B moves_remaining u8]
    [2B num_legal u16][4B value f32][4B sample_weight f32]
    [num_stones*(2B q i16, 2B r i16, 1B player u8)][num_legal*4B policy f32]``.
    """
    num_stones = len(stones)
    num_legal = len(policy)
    cur_byte = _PLAYER_BYTE[current_player]

    example = (
        struct.pack("<H", num_stones)
        + struct.pack("B", cur_byte)
        + struct.pack("B", moves_remaining)
        + struct.pack("<H", num_legal)
        + struct.pack("<f", value)
        + struct.pack("<f", sample_weight)
    )
    for (q, r), p in stones:
        example += struct.pack("<hh", q, r) + struct.pack("B", _PLAYER_BYTE[p])
    for p in policy:
        example += struct.pack("<f", p)

    winner_byte = 1 if winner == "P1" else (-1 if winner == "P2" else 0)
    has_window = window_stats is not None
    body = (
        struct.pack("<I", 1)  # num_examples
        + struct.pack("<b", winner_byte)
        + struct.pack("<I", move_count)
        + struct.pack("B", int(has_window))
    )
    if has_window:
        body += struct.pack("<HHff", *window_stats)
    body += example

    record = b"HX07" + struct.pack("<I", len(body)) + body
    with open(path, "wb") as fh:
        fh.write(record)
    return record


def _play_short_game(gc, n=6):
    g = hexo_rs.GameState(gc)
    for _ in range(n):
        if g.is_terminal():
            break
        g.apply_move(*g.legal_moves()[0])
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_hx07_record_builds_game_state(make_trainer, tmp_path):
    trainer = make_trainer(graph_type="axis")
    gc = trainer.game_config
    g = _play_short_game(gc, n=6)
    stones = g.placed_stones()
    cur = g.current_player()
    mr = g.moves_remaining_this_turn()
    legal = g.legal_moves()
    policy = [1.0 / len(legal)] * len(legal)
    path = tmp_path / "games_0001.bin"
    _write_hx07_record(path, stones, cur, mr, policy, value=1.0)

    batch = []
    with open(path, "rb") as fh:
        trainer._read_binary_records(fh, batch, 10)
    assert batch and batch[0]["format"] == "state"

    examples = trainer._parse_state_examples(batch[0])
    assert examples and all(e.game_state is not None and e.data is None
                            for e in examples)

    from hexo_a0.graph import graph_fn_from_model_config
    gfn = graph_fn_from_model_config(trainer.model_config)
    for e in examples:
        d = gfn(e.game_state)
        assert e.policy_target.shape[0] == int(d.legal_mask.sum())


def test_hx07_dispatch_via_trajectories(make_trainer, tmp_path):
    """_trajectories_to_examples dispatches HX07 batches to state parsing."""
    trainer = make_trainer(graph_type="axis")
    gc = trainer.game_config
    g = _play_short_game(gc, n=4)
    legal = g.legal_moves()
    policy = [1.0 / len(legal)] * len(legal)
    path = tmp_path / "games_0002.bin"
    _write_hx07_record(path, g.placed_stones(), g.current_player(),
                       g.moves_remaining_this_turn(), policy, value=-1.0)
    batch = []
    with open(path, "rb") as fh:
        trainer._read_binary_records(fh, batch, 10)
    examples = trainer._trajectories_to_examples(batch)
    assert examples
    assert all(e.game_state is not None and e.data is None for e in examples)
    assert examples[0].value_target == -1.0


def test_hx07_window_stats_offset(make_trainer, tmp_path):
    """has_window=1 with 12B telemetry parses (body starts at offset 22)."""
    trainer = make_trainer(graph_type="axis")
    gc = trainer.game_config
    g = _play_short_game(gc, n=6)
    legal = g.legal_moves()
    policy = [1.0 / len(legal)] * len(legal)
    path = tmp_path / "games_0003.bin"
    _write_hx07_record(path, g.placed_stones(), g.current_player(),
                       g.moves_remaining_this_turn(), policy, value=1.0,
                       window_stats=(3, 1, 0.5, 0.25))
    batch = []
    with open(path, "rb") as fh:
        trainer._read_binary_records(fh, batch, 10)
    assert batch and batch[0]["format"] == "state"
    assert batch[0]["window_stats"] == {
        "in_window_moves": 3, "gated_moves": 1,
        "mass_removed_sum": 0.5, "acted_deficit_sum": 0.25,
    }
    examples = trainer._parse_state_examples(batch[0])
    assert examples and examples[0].game_state is not None


def test_hx07_and_hx05_coexist(make_trainer, tmp_path):
    """A directory with BOTH an HX07 and a legacy HX05 file ingests both."""
    from test_jsonl_graph_format import TestBinaryReader

    trainer = make_trainer(graph_type="axis")
    gc = trainer.game_config

    # HX07 file.
    g = _play_short_game(gc, n=6)
    legal = g.legal_moves()
    policy = [1.0 / len(legal)] * len(legal)
    p07 = tmp_path / "games_hx07.bin"
    _write_hx07_record(p07, g.placed_stones(), g.current_player(),
                       g.moves_remaining_this_turn(), policy, value=1.0)

    # Legacy HX05 file (reuse the existing test suite helper).
    hx05_record = TestBinaryReader()._make_binary_record(
        n_moves=2, graph_type="axis", magic=b"HX05", node_dim=8)
    p05 = tmp_path / "games_hx05.bin"
    p05.write_bytes(hx05_record)

    batch = []
    for p in (p07, p05):
        with open(p, "rb") as fh:
            trainer._read_binary_records(fh, batch, 10)

    assert {b["format"] for b in batch} == {"state", "graph"}

    examples = trainer._trajectories_to_examples(batch)
    assert examples
    assert all(e.game_state is not None and e.data is None for e in examples)


# ---------------------------------------------------------------------------
# FIX T1: encoding coverage — HX07 ships board STATE, so threat/relative flags
# only affect the reconstructed graph. The reconstructed legal-move count (and
# hence the expected policy length) must match across all 4 encoding combos.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("threat_features,relative_stone_encoding", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_hx07_roundtrip_all_encodings(make_trainer, tmp_path,
                                      threat_features, relative_stone_encoding):
    trainer = make_trainer(graph_type="axis",
                           threat_features=threat_features,
                           relative_stone_encoding=relative_stone_encoding)
    gc = trainer.game_config
    g = _play_short_game(gc, n=8)
    legal = g.legal_moves()
    policy = [1.0 / len(legal)] * len(legal)
    path = tmp_path / "games_enc.bin"
    _write_hx07_record(path, g.placed_stones(), g.current_player(),
                       g.moves_remaining_this_turn(), policy, value=1.0)

    batch = []
    with open(path, "rb") as fh:
        trainer._read_binary_records(fh, batch, 10)
    examples = trainer._parse_state_examples(batch[0])
    assert examples

    from hexo_a0.graph import graph_fn_from_model_config
    gfn = graph_fn_from_model_config(trainer.model_config)
    for e in examples:
        d = gfn(e.game_state)
        assert e.policy_target.shape[0] == int(d.legal_mask.sum())


# ---------------------------------------------------------------------------
# FIX W1: aggregate skip-rate guard in _parse_state_examples. A batch where
# most examples have policy-length != legal-count must raise loudly rather
# than silently returning a near-empty list.
# ---------------------------------------------------------------------------

def test_parse_state_examples_skip_rate_guard(make_trainer):
    trainer = make_trainer(graph_type="axis")
    gc = trainer.game_config
    g = _play_short_game(gc, n=6)
    stones = g.placed_stones()
    cur = g.current_player()
    mr = g.moves_remaining_this_turn()
    legal = g.legal_moves()
    good_policy = [1.0 / len(legal)] * len(legal)
    bad_policy = [0.5, 0.5]  # deliberately wrong length

    def mk(policy):
        return {"stones": stones, "current_player": cur, "moves_remaining": mr,
                "policy": policy, "value": 1.0, "sample_weight": 1.0}

    # 1 good + 9 bad -> 90% skip rate -> must raise.
    game_data = {"examples": [mk(good_policy)] + [mk(bad_policy)] * 9}
    with pytest.raises(RuntimeError, match="skip rate"):
        trainer._parse_state_examples(game_data)


# ---------------------------------------------------------------------------
# FIX W4: bounds checks in _parse_binary_example_hx07 — a truncated record
# (policy bytes cut short) must raise a clear ValueError, not silently produce
# a short policy.
# ---------------------------------------------------------------------------

def test_parse_hx07_truncated_policy_raises(make_trainer, tmp_path):
    trainer = make_trainer(graph_type="axis")
    gc = trainer.game_config
    g = _play_short_game(gc, n=6)
    legal = g.legal_moves()
    policy = [1.0 / len(legal)] * len(legal)
    path = tmp_path / "games_trunc.bin"
    record = _write_hx07_record(path, g.placed_stones(), g.current_player(),
                                g.moves_remaining_this_turn(), policy, value=1.0)
    # Chop the last 8 bytes (two float32 policy entries) off the record body.
    truncated = record[:-8]

    # The per-example parser must raise a clear ValueError rather than silently
    # returning a short policy. Body starts after the 8B envelope; the single
    # example starts at body offset 10 (no window stats).
    from hexo_a0.trainer import _parse_binary_example_hx07
    body = truncated[8:]
    with pytest.raises(ValueError, match="HX07 record truncated"):
        _parse_binary_example_hx07(body, 10)
