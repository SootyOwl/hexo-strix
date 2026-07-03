"""HTTTX move-log codec.

HTTTX text follows explore.htttx.io's coordinate convention, which is a
mirror of our internal axial frame: internal (q, r) <-> HTTTX (q + r, -r).
The map is an involution, so serialize and parse apply the same transform.
"""
from hexo_a0.serving.htttx import serialize_htttx, parse_htttx


def test_serialize_matches_htttx_site_convention():
    # Cross-checked visually on explore.htttx.io: our internal stones
    # (2,-2)(0,-3) are the position htttx.io writes as "1. [0,2][-3,3];".
    log = [(0, 0, "P1"), (2, -2, "P2"), (0, -3, "P2")]
    assert serialize_htttx(log) == "version[1];\n1. [0,2][-3,3];\n"


def test_parse_maps_htttx_convention_to_internal():
    text = "version[1];\n1. [0,2][-3,3];\n"
    assert parse_htttx(text) == [(2, -2), (0, -3)]


def test_serialize_skips_seed_and_pairs_turns():
    # r == 0 coords are fixed points of the mirror, so the text is unchanged.
    log = [(0, 0, "P1"), (1, 0, "P2"), (2, 0, "P2"), (3, 0, "P1")]
    assert serialize_htttx(log) == "version[1];\n1. [1,0][2,0];\n2. [3,0];\n"


def test_serialize_seed_only():
    assert serialize_htttx([(0, 0, "P1")]) == "version[1];\n"


def test_roundtrip_is_identity():
    log = [(0, 0, "P1"), (2, -2, "P2"), (0, -3, "P2"), (-1, 4, "P1")]
    assert parse_htttx(serialize_htttx(log)) == [(q, r) for q, r, _p in log[1:]]
