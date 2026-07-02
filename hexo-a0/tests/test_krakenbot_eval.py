"""Pure-function tests for the KrakenBot adapter — no network calls."""

import json

import pytest

from hexo_a0.krakenbot_eval import (
    _axial_to_cube,
    _cube_to_axial,
    build_turns_json,
    kraken_pair_move,
    turns_to_htttx,
)


class TestCubeAxialConversion:
    @pytest.mark.parametrize("q,r", [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (5, -3), (-7, 4), (8, -8),
    ])
    def test_roundtrip(self, q, r):
        cube = _axial_to_cube(q, r)
        assert cube["x"] + cube["y"] + cube["z"] == 0
        assert _cube_to_axial(cube) == (q, r)

    def test_six_tac_response_decodes_correctly(self):
        """Real response from POST /api/v1/compute/best-move with empty turns."""
        stone_a = {"x": -1, "y": 1, "z": 0}
        stone_b = {"x": 1, "y": 0, "z": -1}
        assert _cube_to_axial(stone_a) == (-1, 0)
        assert _cube_to_axial(stone_b) == (1, -1)

    def test_axial_to_cube_origin(self):
        assert _axial_to_cube(0, 0) == {"x": 0, "y": 0, "z": 0}

    def test_cube_invariant_holds_after_conversion(self):
        for q in range(-4, 5):
            for r in range(-4, 5):
                c = _axial_to_cube(q, r)
                assert c["x"] + c["y"] + c["z"] == 0


class TestBuildTurnsJson:
    def test_empty_history(self):
        out = build_turns_json([])
        assert json.loads(out) == {"turns": []}

    def test_single_turn(self):
        out = build_turns_json([[(1, 0), (-1, 0)]])
        parsed = json.loads(out)
        assert parsed == {
            "turns": [
                {"stones": [
                    {"x": 1, "y": -1, "z": 0},
                    {"x": -1, "y": 1, "z": 0},
                ]},
            ]
        }

    def test_alternating_turns(self):
        history = [
            [(1, 0), (-1, 0)],     # P1's first 2-stone turn
            [(0, 1), (0, -1)],     # P2's first 2-stone turn
            [(2, -1), (-2, 1)],    # P1's second
        ]
        parsed = json.loads(build_turns_json(history))
        assert len(parsed["turns"]) == 3
        # Spot-check that each turn is a 2-element list of cubes
        for turn in parsed["turns"]:
            assert len(turn["stones"]) == 2
            for cube in turn["stones"]:
                assert set(cube.keys()) == {"x", "y", "z"}
                assert cube["x"] + cube["y"] + cube["z"] == 0

    def test_partial_turn_preserved(self):
        # A turn that ended mid-pair (e.g., game-over after one stone) is still
        # serialized with whatever stones were played — the caller must decide
        # whether this is recoverable.
        parsed = json.loads(build_turns_json([[(0, 1)]]))
        assert parsed == {
            "turns": [{"stones": [{"x": 0, "y": -1, "z": 1}]}],
        }


class TestTurnsToHtttx:
    def test_empty(self):
        assert turns_to_htttx([]) == "version[1];\n"

    def test_single_turn(self):
        out = turns_to_htttx([[(-1, 1), (0, -1)]])
        assert out == "version[1];\n1. [-1,1][0,-1];\n"

    def test_multiple_turns(self):
        out = turns_to_htttx([
            [(-1, 1), (0, -1)],
            [(7, 1), (6, 0)],
            [(-1, -1), (-1, 0)],
        ])
        assert out == (
            "version[1];\n"
            "1. [-1,1][0,-1];\n"
            "2. [7,1][6,0];\n"
            "3. [-1,-1][-1,0];\n"
        )

    def test_partial_turn(self):
        out = turns_to_htttx([[(1, 0)]])
        assert out == "version[1];\n1. [1,0];\n"


class TestKrakenPairMoveStub:
    """Verify request/response plumbing without hitting the live endpoint."""

    def test_posts_expected_payload_and_parses_response(self, monkeypatch):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "stones": [
                        {"x": 1, "y": -1, "z": 0},
                        {"x": -1, "y": 1, "z": 0},
                    ],
                    "modelVersion": "kraken_v1",
                }

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

        import hexo_a0.krakenbot_eval as ke
        monkeypatch.setattr(ke.requests, "post", fake_post)

        out = kraken_pair_move(
            [[(2, 0), (-2, 0)]],
            api_url="http://example/api/v1/compute/best-move",
            timeout=3.0,
        )
        assert out == [(1, 0), (-1, 0)]
        assert captured["url"] == "http://example/api/v1/compute/best-move"
        assert captured["timeout"] == 3.0
        assert captured["json"]["config"] == {"botName": "kraken"}
        assert json.loads(captured["json"]["position"]["turnsJson"]) == {
            "turns": [{"stones": [
                {"x": 2, "y": -2, "z": 0},
                {"x": -2, "y": 2, "z": 0},
            ]}]
        }

    def test_http_error_propagates(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                raise RuntimeError("HTTP 500")

            def json(self):
                raise AssertionError("should not be called")

        import hexo_a0.krakenbot_eval as ke
        monkeypatch.setattr(ke.requests, "post", lambda *a, **k: FakeResponse())

        with pytest.raises(RuntimeError, match="HTTP 500"):
            kraken_pair_move([])
