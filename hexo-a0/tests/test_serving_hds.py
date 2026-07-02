"""HDS sandbox -> HTTTX converter (serving/hds.py + /convert_hds route)."""
import copy
import json
import pathlib
import urllib.error

import pytest

from hexo_a0.serving.hds import (
    HdsBadResponse,
    HdsUnreachable,
    IllegalPosition,
    InvalidSandboxCode,
    SandboxNotFound,
    extract_sandbox_code,
    hds_to_htttx,
    sandbox_to_move_log,
)

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "hds_sandbox_5knldz6.json"

_GOLDEN_HTTTX = (
    "version[1];\n"
    "1. [-1,1][5,0];\n"
    "2. [0,-1][0,-2];\n"
    "3. [0,1][1,-2];\n"
    "4. [0,-3][-1,-1];\n"
    "5. [0,-5][1,-3];\n"
    "6. [-6,1][-7,1];\n"
    "7. [-8,1][-6,0];\n"
    "8. [-2,-1][1,-1];\n"
    "9. [2,-1][-4,-1];\n"
    "10. [-1,-2][-3,0];\n"
    "11. [2,-5][-4,1];\n"
)


def _fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


def _fetch_returning(response: dict):
    """An injectable fetch(url, timeout) that returns a canned response dict."""
    def _fetch(url, timeout):  # noqa: ARG001
        return response
    return _fetch


# --------------------------------------------------------------------------
# extract_sandbox_code
# --------------------------------------------------------------------------

@pytest.mark.parametrize("src", [
    "https://hexo.did.science/sandbox/5knldz6",
    "https://hexo.did.science/sandbox/5knldz6/",
    "https://hexo.did.science/sandbox/5knldz6?foo=bar",
    "https://hexo.did.science/sandbox/5knldz6#frag",
    "5knldz6",
    "  5KNLDZ6  ",
])
def test_extract_sandbox_code_accepts_valid(src):
    assert extract_sandbox_code(src) == "5knldz6"


@pytest.mark.parametrize("src", [
    "",
    "abc",                # too short
    "5knldz67",           # too long
    "5knld-6",            # illegal char
    "https://hexo.did.science/sandbox/",      # no code
    "https://hexo.did.science/sandbox/toolong1",
])
def test_extract_sandbox_code_rejects_invalid(src):
    with pytest.raises(InvalidSandboxCode):
        extract_sandbox_code(src)


# --------------------------------------------------------------------------
# sandbox_to_move_log
# --------------------------------------------------------------------------

def test_move_log_identity_happy_path():
    move_log, meta = sandbox_to_move_log(_fixture()["gamePosition"])
    assert move_log[0] == (0, 0, "P1")
    assert len(move_log) == 23
    assert meta["move_count"] == 23
    assert meta["offset_applied"] is None
    assert meta["current_turn"] == "P1"
    assert meta["placements_remaining"] == 2


def test_move_log_centers_non_origin_position():
    gp = _fixture()["gamePosition"]
    for cell in gp["cells"]:
        cell["x"] += 3
        cell["y"] -= 2
    move_log, meta = sandbox_to_move_log(gp)
    assert move_log[0] == (0, 0, "P1")
    assert meta["offset_applied"] == (-3, 2)
    from hexo_a0.serving.htttx import serialize_htttx
    assert serialize_htttx(move_log) == _GOLDEN_HTTTX


def test_move_log_trailing_single_placement():
    from hexo_a0.serving.htttx import serialize_htttx
    gp = {
        "cells": [
            {"moveId": 1, "player": "player-1", "x": 0, "y": 0},
            {"moveId": 2, "player": "player-2", "x": 1, "y": 0},
            {"moveId": 3, "player": "player-2", "x": 2, "y": 0},
            {"moveId": 4, "player": "player-1", "x": 3, "y": 0},
        ],
        "currentTurnPlayer": "player-1",
        "placementsRemaining": 1,
    }
    move_log, meta = sandbox_to_move_log(gp)
    assert serialize_htttx(move_log) == "version[1];\n1. [1,0][2,0];\n2. [3,0];\n"
    assert meta["move_count"] == 4
    assert meta["placements_remaining"] == 1
    assert meta["offset_applied"] is None


def _gp(cells):
    return {"cells": cells, "currentTurnPlayer": "player-1", "placementsRemaining": 2}


def test_move_log_rejects_non_contiguous_move_ids():
    with pytest.raises(IllegalPosition):
        sandbox_to_move_log(_gp([
            {"moveId": 1, "player": "player-1", "x": 0, "y": 0},
            {"moveId": 2, "player": "player-2", "x": 1, "y": 0},
            {"moveId": 4, "player": "player-2", "x": 2, "y": 0},
        ]))


def test_move_log_rejects_three_in_a_row_same_player():
    with pytest.raises(IllegalPosition):
        sandbox_to_move_log(_gp([
            {"moveId": 1, "player": "player-1", "x": 0, "y": 0},
            {"moveId": 2, "player": "player-2", "x": 1, "y": 0},
            {"moveId": 3, "player": "player-2", "x": 2, "y": 0},
            {"moveId": 4, "player": "player-2", "x": 3, "y": 0},
        ]))


def test_move_log_rejects_duplicate_coordinate():
    with pytest.raises(IllegalPosition):
        sandbox_to_move_log(_gp([
            {"moveId": 1, "player": "player-1", "x": 0, "y": 0},
            {"moveId": 2, "player": "player-2", "x": 1, "y": 0},
            {"moveId": 3, "player": "player-2", "x": 1, "y": 0},
        ]))


# --------------------------------------------------------------------------
# hds_to_htttx (golden)
# --------------------------------------------------------------------------

def test_hds_to_htttx_golden():
    result = hds_to_htttx("5knldz6", fetch=_fetch_returning(_fixture()))
    assert result["htttx"] == _GOLDEN_HTTTX
    assert result["name"] == "finneon7 vs ryandileepjohn - Replay Move 12/37 fixed"
    assert result["move_count"] == 23
    assert result["code"] == "5knldz6"
    assert result["offset_applied"] is None
    assert result["current_turn"] == "P1"
    assert result["placements_remaining"] == 2
    assert result["colors_swapped"] is False


def test_hds_to_htttx_accepts_full_url():
    result = hds_to_htttx(
        "https://hexo.did.science/sandbox/5knldz6",
        fetch=_fetch_returning(_fixture()),
    )
    assert result["htttx"] == _GOLDEN_HTTTX


def test_hds_to_htttx_bad_shape_raises():
    with pytest.raises(HdsBadResponse):
        hds_to_htttx("5knldz6", fetch=_fetch_returning({"id": "5knldz6"}))


# --------------------------------------------------------------------------
# handle_convert_hds (route helper)
# --------------------------------------------------------------------------

def test_handle_convert_hds_success():
    from hexo_a0.serving.app import handle_convert_hds
    status, body = handle_convert_hds({"src": "5knldz6"}, fetch=_fetch_returning(_fixture()))
    assert status == 200
    assert body["htttx"] == _GOLDEN_HTTTX
    assert body["name"] == "finneon7 vs ryandileepjohn - Replay Move 12/37 fixed"
    assert body["code"] == "5knldz6"
    assert body["move_count"] == 23
    assert body["current_turn"] == "P1"
    assert body["placements_remaining"] == 2
    assert body["offset_applied"] is None
    assert body["colors_swapped"] is False


def test_handle_convert_hds_invalid_code():
    from hexo_a0.serving.app import handle_convert_hds
    status, body = handle_convert_hds({"src": "not-a-code"}, fetch=_fetch_returning(_fixture()))
    assert status == 400
    assert "error" in body


def test_handle_convert_hds_not_found():
    from hexo_a0.serving.app import handle_convert_hds

    def _raise(url, timeout):  # noqa: ARG001
        raise SandboxNotFound("nope")

    status, body = handle_convert_hds({"src": "5knldz6"}, fetch=_raise)
    assert status == 404


def test_handle_convert_hds_illegal_position():
    from hexo_a0.serving.app import handle_convert_hds
    bad = {"id": "5knldz6", "name": "x", "gamePosition": _gp([
        {"moveId": 1, "player": "player-1", "x": 0, "y": 0},
        {"moveId": 2, "player": "player-2", "x": 1, "y": 0},
        {"moveId": 4, "player": "player-2", "x": 2, "y": 0},
    ])}
    status, body = handle_convert_hds({"src": "5knldz6"}, fetch=_fetch_returning(bad))
    assert status == 422


def test_handle_convert_hds_unreachable():
    from hexo_a0.serving.app import handle_convert_hds

    def _raise(url, timeout):  # noqa: ARG001
        raise HdsUnreachable("down")

    status, body = handle_convert_hds({"src": "5knldz6"}, fetch=_raise)
    assert status == 502


# --------------------------------------------------------------------------
# Adversarial-review hardening: stricter legality + robustness
# --------------------------------------------------------------------------

def test_move_log_swaps_colors_when_player_2_opens():
    # HDS lets either colour open. HTTTX/HeXO always treat the opener as P1, so
    # a player-2 opener is relabelled (colour swap) rather than rejected.
    move_log, meta = sandbox_to_move_log(_gp([
        {"moveId": 1, "player": "player-2", "x": 0, "y": 0},
        {"moveId": 2, "player": "player-1", "x": 1, "y": 0},
        {"moveId": 3, "player": "player-1", "x": 2, "y": 0},
        {"moveId": 4, "player": "player-2", "x": 3, "y": 0},
        {"moveId": 5, "player": "player-2", "x": 4, "y": 0},
    ]))
    assert move_log[0] == (0, 0, "P1")  # opener (HDS player-2) -> P1
    assert [m[2] for m in move_log] == ["P1", "P2", "P2", "P1", "P1"]
    assert meta["colors_swapped"] is True
    # _gp sets currentTurnPlayer="player-1", which under the swap is P2.
    assert meta["current_turn"] == "P2"


def test_move_log_player_1_opener_not_swapped():
    move_log, meta = sandbox_to_move_log(_gp([
        {"moveId": 1, "player": "player-1", "x": 0, "y": 0},
        {"moveId": 2, "player": "player-2", "x": 1, "y": 0},
        {"moveId": 3, "player": "player-2", "x": 2, "y": 0},
    ]))
    assert move_log[0] == (0, 0, "P1")
    assert [m[2] for m in move_log] == ["P1", "P2", "P2"]
    assert meta["colors_swapped"] is False


def test_move_log_rejects_illegal_even_when_player_2_opens():
    # The swap must not weaken alternating-pairs validation.
    with pytest.raises(IllegalPosition):
        sandbox_to_move_log(_gp([
            {"moveId": 1, "player": "player-2", "x": 0, "y": 0},
            {"moveId": 2, "player": "player-1", "x": 1, "y": 0},
            {"moveId": 3, "player": "player-1", "x": 2, "y": 0},
            {"moveId": 4, "player": "player-1", "x": 3, "y": 0},  # 3rd P1 in a row
        ]))


@pytest.mark.parametrize("bad_cell", [
    {"moveId": 1, "player": "player-1", "x": "0", "y": 0},   # str x
    {"moveId": 1, "player": "player-1", "x": 0, "y": None},  # None y
    {"moveId": "1", "player": "player-1", "x": 0, "y": 0},   # str moveId
    {"moveId": 1.5, "player": "player-1", "x": 0, "y": 0},   # float moveId
])
def test_move_log_rejects_non_integer_fields(bad_cell):
    with pytest.raises(HdsBadResponse):
        sandbox_to_move_log(_gp([bad_cell]))


def test_move_log_rejects_unknown_player():
    with pytest.raises(IllegalPosition):
        sandbox_to_move_log(_gp([
            {"moveId": 1, "player": "player-3", "x": 0, "y": 0},
        ]))


def test_move_log_rejects_non_dict_cell():
    with pytest.raises(HdsBadResponse):
        sandbox_to_move_log(_gp([1, 2, 3]))


def test_move_log_rejects_missing_cell_field():
    with pytest.raises(HdsBadResponse):
        sandbox_to_move_log(_gp([{"moveId": 1, "player": "player-1"}]))


def test_move_log_rejects_empty_cells():
    with pytest.raises(IllegalPosition):
        sandbox_to_move_log(_gp([]))


def test_handle_convert_hds_non_dict_payload():
    from hexo_a0.serving.app import handle_convert_hds
    for payload in (None, [], "x", 5):
        status, body = handle_convert_hds(payload, fetch=_fetch_returning(_fixture()))
        assert status == 400
        assert "error" in body


def test_handle_convert_hds_non_string_src():
    from hexo_a0.serving.app import handle_convert_hds
    status, body = handle_convert_hds({"src": 123}, fetch=_fetch_returning(_fixture()))
    assert status == 400
    assert "error" in body


def test_handle_convert_hds_bad_response_502():
    from hexo_a0.serving.app import handle_convert_hds
    status, body = handle_convert_hds(
        {"src": "5knldz6"}, fetch=_fetch_returning({"gamePosition": "not-a-dict"}))
    assert status == 502
    assert "error" in body


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self, n=-1):
        return self._body if (n is None or n < 0) else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_default_fetch_caps_response_size(monkeypatch):
    from hexo_a0.serving import hds
    oversized = b'{"x":"' + b"a" * (hds._MAX_RESPONSE_BYTES + 100) + b'"}'

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeResp(oversized)

    monkeypatch.setattr(hds.urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(HdsBadResponse):
        hds.fetch_sandbox_position("5knldz6")


def test_default_fetch_non_404_http_error_is_bad_response(monkeypatch):
    from hexo_a0.serving import hds

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError("https://hexo.did.science", 500, "boom", {}, None)

    monkeypatch.setattr(hds.urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(HdsBadResponse):
        hds.fetch_sandbox_position("5knldz6")


def test_move_log_rejects_unhashable_player():
    # An unhashable player value must not crash the dict-membership test with a
    # bare TypeError; it must surface as a domain error.
    with pytest.raises(IllegalPosition):
        sandbox_to_move_log(_gp([
            {"moveId": 1, "player": ["player-1"], "x": 0, "y": 0},
        ]))


@pytest.mark.parametrize("field", ["x", "y", "moveId"])
def test_move_log_rejects_bool_fields(field):
    cell = {"moveId": 1, "player": "player-1", "x": 0, "y": 0}
    cell[field] = True  # bool is an int subclass; must still be rejected
    with pytest.raises(HdsBadResponse):
        sandbox_to_move_log(_gp([cell]))


def test_move_log_unhashable_current_turn_player_is_none():
    # A malformed (unhashable) currentTurnPlayer must not crash; metadata is
    # lenient, so it degrades to None rather than raising TypeError.
    gp = {
        "cells": [
            {"moveId": 1, "player": "player-1", "x": 0, "y": 0},
            {"moveId": 2, "player": "player-2", "x": 1, "y": 0},
            {"moveId": 3, "player": "player-2", "x": 2, "y": 0},
        ],
        "currentTurnPlayer": ["player-1"],
        "placementsRemaining": 2,
    }
    _move_log, meta = sandbox_to_move_log(gp)
    assert meta["current_turn"] is None


def test_handle_convert_hds_unhashable_player_is_422():
    from hexo_a0.serving.app import handle_convert_hds
    bad = {"gamePosition": _gp([
        {"moveId": 1, "player": ["player-1"], "x": 0, "y": 0},
    ])}
    status, body = handle_convert_hds({"src": "5knldz6"}, fetch=_fetch_returning(bad))
    assert status == 422
    assert "error" in body


def test_default_fetch_sends_identifying_user_agent(monkeypatch):
    # hexo.did.science is behind Cloudflare, which 403s the default
    # "Python-urllib/x.y" User-Agent (error 1010). The outbound request must
    # carry a non-default UA; we send an honest one that identifies this tool.
    from hexo_a0.serving import hds
    captured = {}

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["ua"] = req.get_header("User-agent")
        return _FakeResp(b'{"gamePosition":{"cells":[]}}')

    monkeypatch.setattr(hds.urllib.request, "urlopen", _fake_urlopen)
    hds.fetch_sandbox_position("5knldz6")
    ua = captured["ua"]
    assert ua and "python-urllib" not in ua.lower()
    assert "hexo-a0" in ua  # honest, self-identifying UA


def test_default_fetch_url_error_is_unreachable(monkeypatch):
    from hexo_a0.serving import hds

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(hds.urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(HdsUnreachable):
        hds.fetch_sandbox_position("5knldz6")
