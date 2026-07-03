"""The fixture generator produces schema-valid, self-consistent fixtures."""

import json

from hexo_a0.gen_parity_fixtures import (
    generate_fixtures, tiny_model_and_config, tiny_nojk_model_and_config,
)

TINY_GAME = {"win_length": 6, "placement_radius": 4, "max_moves": 300}


def _check_opts(fx, expected):
    assert fx["graph_opts"] == expected


def test_generate_tiny_jkcat_fixtures(tmp_path):
    model, mc = tiny_model_and_config()
    paths = generate_fixtures(model, mc, prefix="tiny", out_dir=tmp_path, game_config=TINY_GAME)
    assert len(paths) == 4
    for p in paths:
        fx = json.loads(p.read_text())
        _check_opts(fx, {"prune_empty_edges": True, "threat_features": True, "relative_stones": True})
        assert len(fx["expected"]["logits"]) > 0
        assert -1.0 <= fx["expected"]["value"] <= 1.0
        coords = [tuple(c) for c, _ in fx["expected"]["logits"]]
        assert len(coords) == len(set(coords))  # every legal move exactly once


def test_generate_tiny_nojk_fixtures(tmp_path):
    model, mc = tiny_nojk_model_and_config()
    paths = generate_fixtures(model, mc, prefix="tiny_nojk", out_dir=tmp_path, game_config=TINY_GAME)
    assert len(paths) == 4
    for p in paths:
        fx = json.loads(p.read_text())
        _check_opts(fx, {"prune_empty_edges": False, "threat_features": False, "relative_stones": False})


def test_midturn_fixture_is_mid_turn(tmp_path):
    """midturn_9 must land moves_remaining==1 (the mr==1 feature encoding)."""
    model, mc = tiny_model_and_config()
    generate_fixtures(model, mc, "tiny", tmp_path, TINY_GAME)  # raises in _play_moves if not


def test_fixture_determinism(tmp_path):
    a = generate_fixtures(*tiny_model_and_config(), prefix="tiny", out_dir=tmp_path / "a", game_config=TINY_GAME)
    b = generate_fixtures(*tiny_model_and_config(), prefix="tiny", out_dir=tmp_path / "b", game_config=TINY_GAME)
    for pa, pb in zip(a, b):
        assert pa.read_text() == pb.read_text()