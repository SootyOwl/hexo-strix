"""HTTTX codec parity with the frozen scripts/play_server.py."""
import importlib.util
import pathlib

from hexo_a0.serving.htttx import serialize_htttx, parse_htttx


def _frozen():
    """Load the frozen scripts/play_server.py as a module (parity reference)."""
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "play_server.py"
    spec = importlib.util.spec_from_file_location("frozen_play_server", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_serialize_skips_seed_and_pairs_turns():
    log = [(0, 0, "P1"), (1, 0, "P2"), (2, 0, "P2"), (3, 0, "P1")]
    assert serialize_htttx(log) == "version[1];\n1. [1,0][2,0];\n2. [3,0];\n"


def test_serialize_seed_only():
    assert serialize_htttx([(0, 0, "P1")]) == "version[1];\n"


def test_parse_roundtrip():
    text = "version[1];\n1. [1,0][2,0];\n2. [3,0];\n"
    assert parse_htttx(text) == [(1, 0), (2, 0), (3, 0)]


def test_parity_with_frozen_script():
    f = _frozen()
    log = [(0, 0, "P1"), (1, 0, "P2"), (-2, 3, "P2"), (3, -1, "P1")]
    assert serialize_htttx(log) == f.serialize_htttx(log)
    txt = f.serialize_htttx(log)
    assert parse_htttx(txt) == f.parse_htttx(txt)
