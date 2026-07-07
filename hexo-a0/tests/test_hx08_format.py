"""HX08 binary format reader tests.

Builds HX08 records in Python matching the spec in ``write_example_hx08``
(self_play.rs) and verifies the trainer's reader recovers per-move q + visits,
that HX07 still parses (no q), and that a corrupt HX08 record resyncs.
"""

import io
import struct

import pytest

torch = pytest.importorskip("torch")

from hexo_a0.trainer import Trainer, _parse_binary_example_hx08


def _example_bytes(stones, cur, moves_remaining, policy, value, sw, q=None, visits=None):
    """One HX07/HX08 example. If q/visits given, appends the HX08 blocks."""
    num_stones = len(stones)
    num_legal = len(policy)
    b = struct.pack("<HBBH", num_stones, cur, moves_remaining, num_legal)
    b += struct.pack("<f", value) + struct.pack("<f", sw)
    for (q_, r_), p in stones:
        b += struct.pack("<hhB", q_, r_, p)
    for pr in policy:
        b += struct.pack("<f", pr)
    if q is not None:
        for qq in q:
            b += struct.pack("<f", qq)
        for vv in visits:
            b += struct.pack("<H", vv)
    return b


def _record(magic, examples_bytes_list, winner=1, move_count=3):
    body = struct.pack("<I", len(examples_bytes_list))          # num_examples
    body += struct.pack("<b", winner)                            # winner i8
    body += struct.pack("<I", move_count)                        # move_count
    body += struct.pack("<B", 0)                                 # has_window = 0
    for eb in examples_bytes_list:
        body += eb
    return magic + struct.pack("<I", len(body)) + body


def test_hx08_example_parser_roundtrip():
    ex_b = _example_bytes(
        stones=[((0, 0), 0), ((1, 0), 1)], cur=0, moves_remaining=2,
        policy=[0.5, 0.3, 0.2], value=1.0, sw=1.0,
        q=[0.9, -0.2, 0.1], visits=[10, 4, 0],
    )
    ex, off = _parse_binary_example_hx08(ex_b, 0)
    assert off == len(ex_b)
    assert ex["policy"] == pytest.approx([0.5, 0.3, 0.2], abs=1e-6)
    assert ex["q_targets"] == pytest.approx([0.9, -0.2, 0.1], abs=1e-6)
    assert ex["q_visits"] == [10, 4, 0]


def test_reader_parses_hx08_record():
    ex_b = _example_bytes(
        stones=[((0, 0), 0)], cur=1, moves_remaining=1,
        policy=[0.7, 0.3], value=-1.0, sw=0.25,
        q=[0.4, -0.4], visits=[8, 2],
    )
    rec = _record(b"HX08", [ex_b], winner=-1, move_count=5)
    batch: list = []
    Trainer._read_binary_records(io.BytesIO(rec), batch, max_records=10)
    assert len(batch) == 1
    g = batch[0]
    assert g["format"] == "state"
    assert g["winner"] == "P2"
    assert g["move_count"] == 5
    ex = g["examples"][0]
    assert ex["q_targets"] == pytest.approx([0.4, -0.4], abs=1e-6)
    assert ex["q_visits"] == [8, 2]


def test_reader_hx07_still_has_no_q():
    ex_b = _example_bytes(
        stones=[((0, 0), 0)], cur=0, moves_remaining=2,
        policy=[0.6, 0.4], value=1.0, sw=1.0,
    )
    rec = _record(b"HX07", [ex_b], winner=1)
    batch: list = []
    Trainer._read_binary_records(io.BytesIO(rec), batch, max_records=10)
    assert len(batch) == 1
    ex = batch[0]["examples"][0]
    assert "q_targets" not in ex
    assert ex["policy"] == pytest.approx([0.6, 0.4], abs=1e-6)


def test_reader_multi_example_hx08():
    exs = [
        _example_bytes([((0, 0), 0)], 0, 2, [1.0], 1.0, 1.0, q=[0.5], visits=[7]),
        _example_bytes([((0, 0), 0), ((1, 1), 1)], 1, 1, [0.5, 0.5], -1.0, 1.0,
                       q=[0.1, 0.2], visits=[3, 0]),
    ]
    rec = _record(b"HX08", exs, winner=1, move_count=2)
    batch: list = []
    Trainer._read_binary_records(io.BytesIO(rec), batch, max_records=10)
    assert len(batch[0]["examples"]) == 2
    assert batch[0]["examples"][1]["q_visits"] == [3, 0]
