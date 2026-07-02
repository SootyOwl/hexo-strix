"""Tests for the sprt_search_variant.py CLI wrapper (opening-book knobs)."""

import sys
from pathlib import Path

# scripts/ is not a package, so add it to sys.path on import.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sprt_search_variant  # noqa: E402


class TestOpeningCliParsing:
    def test_opening_flags_parse(self):
        ns = sprt_search_variant.parse_args([
            "--checkpoint", "a.pt",
            "--opening-plies", "4",
            "--opening-temperature", "1.25",
            "--opening-generator", "alternate",
        ])
        assert ns.opening_plies == 4
        assert ns.opening_temperature == 1.25
        assert ns.opening_generator == "alternate"

    def test_opening_defaults_to_legacy_mode(self):
        ns = sprt_search_variant.parse_args(["--checkpoint", "a.pt"])
        assert ns.opening_plies == 0  # 0 = legacy noise-on, no opening
        assert ns.opening_generator == "alternate"


class TestOpeningCliPassThrough:
    def test_main_forwards_opening_args(self, tmp_path, monkeypatch, capsys):
        ckpt = tmp_path / "a.pt"
        ckpt.write_bytes(b"stub")

        captured = {}

        def fake_run_head_to_head(**kwargs):
            captured.update(kwargs)
            return {
                "games": 0, "wins": 0, "draws": 0, "losses": 0,
                "score": 0.5, "score_ci_lo": 0.0, "score_ci_hi": 1.0,
                "elo_diff": 0.0, "elo_ci_lo": 0.0, "elo_ci_hi": 0.0,
                "llr": 0.0, "decision": "continue", "winner": "inconclusive",
            }

        monkeypatch.setattr(sprt_search_variant, "run_head_to_head", fake_run_head_to_head)

        sprt_search_variant.main([
            "--checkpoint", str(ckpt),
            "--opening-plies", "5",
            "--opening-temperature", "0.8",
            "--opening-generator", "b",
        ])

        assert captured["opening_plies"] == 5
        assert captured["opening_temperature"] == 0.8
        assert captured["opening_generator"] == "b"
