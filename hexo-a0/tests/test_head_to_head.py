"""Tests for the ``head-to-head`` subcommand and module."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_a0.head_to_head import (
    _arch_digest,
    load_checkpoint,
    run_head_to_head,
)


# Two distinct tiny configs — purposefully differ in hidden_dim and num_layers
# so we exercise the cross-config code path (each side reconstructs its own
# architecture from the checkpoint's model_config).
MC_A = ModelConfig(
    hidden_dim=16, num_layers=2, num_heads=2,
    policy_hidden=8, value_hidden=8,
    graph_type="hex", conv_type="gatv2",
    prune_empty_edges=False, dropout=0.0, pre_norm=True,
)
MC_B = ModelConfig(
    hidden_dim=8, num_layers=1, num_heads=2,
    policy_hidden=8, value_hidden=8,
    graph_type="hex", conv_type="gatv2",
    prune_empty_edges=False, dropout=0.0, pre_norm=True,
)

DEVICE = torch.device("cpu")


def _save_tiny_checkpoint(path: Path, mc: ModelConfig, train_steps: int = 42) -> None:
    model = HeXONet(mc).to(DEVICE)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": dataclasses.asdict(mc),
            "train_steps": train_steps,
        },
        str(path),
    )


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def test_load_reconstructs_model_config(self, tmp_path):
        p = tmp_path / "a.pt"
        _save_tiny_checkpoint(p, MC_A, train_steps=123)
        loaded = load_checkpoint(p, DEVICE)
        assert loaded.model_config.hidden_dim == MC_A.hidden_dim
        assert loaded.model_config.num_layers == MC_A.num_layers
        assert loaded.train_steps == 123
        assert isinstance(loaded.model, HeXONet)

    def test_missing_model_config_raises(self, tmp_path):
        p = tmp_path / "bare.pt"
        model = HeXONet(MC_A).to(DEVICE)
        torch.save({"model_state_dict": model.state_dict()}, str(p))
        with pytest.raises(ValueError, match="model_config"):
            load_checkpoint(p, DEVICE)


# ---------------------------------------------------------------------------
# Run-loop smoke
# ---------------------------------------------------------------------------


class TestHeadToHeadSmoke:
    def test_runs_to_max_games(self, tmp_path):
        """Smoke test: two tiny different-architecture models, bounded by max_games.

        Use very tight bounds-resistant settings so it almost certainly hits
        max_games rather than SPRT-terminating, and skip MCTS so the test is
        quick (raw policy is fine here — we're testing the harness, not strength).
        """
        a = tmp_path / "a.pt"
        b = tmp_path / "b.pt"
        _save_tiny_checkpoint(a, MC_A)
        _save_tiny_checkpoint(b, MC_B)

        summary = run_head_to_head(
            checkpoint_a=a,
            checkpoint_b=b,
            win_length=4, radius=2, max_moves=20,
            mcts_sims=0,  # raw policy → fast
            mcts_m_actions=4,
            device_str="cpu",
            sprt_s0=0.50, sprt_s1=0.55,
            sprt_alpha=0.001, sprt_beta=0.001,  # very wide bounds → likely hit max_games
            window_size=400,
            max_games=4,
            seed=0,
        )

        assert summary["games"] == 4
        assert summary["wins"] + summary["draws"] + summary["losses"] == 4
        assert summary["decision"] in ("continue", "accept_h1", "reject_h1")
        assert summary["winner"] in ("A", "B", "inconclusive")
        assert summary["checkpoint_a"] == str(a)
        assert summary["checkpoint_b"] == str(b)


class TestHeadToHeadStateFile:
    def test_state_file_written_each_game(self, tmp_path):
        a = tmp_path / "a.pt"
        b = tmp_path / "b.pt"
        sf = tmp_path / "state.json"
        _save_tiny_checkpoint(a, MC_A)
        _save_tiny_checkpoint(b, MC_B)

        summary = run_head_to_head(
            checkpoint_a=a,
            checkpoint_b=b,
            win_length=4, radius=2, max_moves=20,
            mcts_sims=0,
            mcts_m_actions=4,
            device_str="cpu",
            max_games=3,
            seed=0,
            state_file=sf,
        )

        assert sf.exists()
        data = json.loads(sf.read_text())
        # Expected keys per the docstring contract
        for key in (
            "games", "wins", "draws", "losses",
            "llr", "score", "decision", "timestamp",
            "checkpoint_a", "checkpoint_b", "bounds",
        ):
            assert key in data, f"missing key {key!r} in state file"
        assert data["games"] == summary["games"]
        assert data["decision"] == summary["decision"]


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


class TestCliRegistration:
    def test_subparser_parses_required_args(self):
        """argparse can construct the parser and accept the documented surface."""
        # Importing the cli module builds the parser lazily inside main(); we
        # re-invoke the parser-building logic by calling main with --help would
        # exit, so instead just verify argparse-level acceptance by patching
        # the dispatcher to a no-op.
        from hexo_a0 import cli as cli_mod

        called = {}

        def fake_dispatch(args):
            called["args"] = args
            return 0

        original = cli_mod._run_head_to_head
        cli_mod._run_head_to_head = fake_dispatch
        try:
            rc = cli_mod.main([
                "head-to-head",
                "--checkpoint-a", "a.pt",
                "--checkpoint-b", "b.pt",
                "--win-length", "4",
                "--radius", "2",
                "--max-moves", "100",
            ])
        finally:
            cli_mod._run_head_to_head = original

        assert rc == 0
        ns = called["args"]
        assert ns.command == "head-to-head"
        assert ns.checkpoint_a == "a.pt"
        assert ns.checkpoint_b == "b.pt"
        assert ns.win_length == 4
        assert ns.radius == 2
        assert ns.max_moves == 100
        # Defaults
        assert ns.mcts_sims == 200
        assert ns.mcts_m_actions == 16
        assert ns.device == "cpu"
        assert ns.sprt_s0 == pytest.approx(0.50)
        assert ns.sprt_s1 == pytest.approx(0.55)
        assert ns.max_games == 1000
        assert ns.state_file is None


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def test_arch_digest_includes_key_fields():
    s = _arch_digest(MC_A)
    assert "graph=hex" in s
    assert "conv=gatv2" in s
    assert "layers=2" in s
    assert "hidden=16" in s
    assert "heads=2" in s
    assert "jk=False" in s


# ---------------------------------------------------------------------------
# Paired self-generated openings (noise-off SPRT)
# ---------------------------------------------------------------------------


class TestHeadToHeadOpenings:
    def test_paired_openings_and_diversity_metrics(self, tmp_path):
        a = tmp_path / "a.pt"
        b = tmp_path / "b.pt"
        _save_tiny_checkpoint(a, MC_A)
        _save_tiny_checkpoint(b, MC_B)

        summary = run_head_to_head(
            checkpoint_a=a,
            checkpoint_b=b,
            win_length=4, radius=2, max_moves=20,
            mcts_sims=8, mcts_m_actions=4,
            device_str="cpu",
            max_games=4, seed=0,
            opening_plies=3, opening_temperature=1.0, opening_generator="alternate",
        )

        # Diversity instrumentation present and sane.
        assert 0.0 < summary["frac_unique_opening"] <= 1.0
        assert 0.0 < summary["frac_unique_game"] <= 1.0
        assert summary["mean_game_length"] > 0
        # Lower-tail length: p10/min catch opening blunders the mean smears over.
        assert summary["min_game_length"] >= 1
        assert summary["min_game_length"] <= summary["p10_game_length"] <= summary["mean_game_length"]
        # Direct opening-quality signals.
        assert 0.0 <= summary["refuted_open_frac"] <= 1.0
        assert summary["p2_open_mean_dist"] >= 0.0

        # Both games of a pentanomial pair start from the *same* opening.
        per_game = summary["per_game_opening"]
        assert len(per_game) == 4
        assert all(len(o) == 3 for o in per_game)   # opening_plies
        assert per_game[0] == per_game[1]            # pair 0
        assert per_game[2] == per_game[3]            # pair 1

    def test_opening_plies_zero_is_legacy_mode(self, tmp_path):
        """opening_plies=0 ⇒ no opening, no diversity metrics (legacy noise-on)."""
        a = tmp_path / "a.pt"
        b = tmp_path / "b.pt"
        _save_tiny_checkpoint(a, MC_A)
        _save_tiny_checkpoint(b, MC_B)

        summary = run_head_to_head(
            checkpoint_a=a,
            checkpoint_b=b,
            win_length=4, radius=2, max_moves=20,
            mcts_sims=0, mcts_m_actions=4,
            device_str="cpu",
            max_games=2, seed=0,
        )
        assert summary["games"] == 2
        # Game uniqueness IS reported even in legacy noise-on mode...
        assert 0.0 < summary["frac_unique_game"] <= 1.0
        assert summary["mean_game_length"] > 0
        # ...but opening-only signals are absent.
        assert summary.get("per_game_opening") in (None, [])
        assert "frac_unique_opening" not in summary
        assert "refuted_open_frac" not in summary


# ---------------------------------------------------------------------------
# Refuted-opening metric: P2 plays two UNALIGNED stones both away from origin.
# (Aligned-but-far = "preemptive island" = fine; one-near = "balanced" = fine.)
# ---------------------------------------------------------------------------


class TestRefutedOpeningMetric:
    def test_hexdist(self):
        from hexo_a0.head_to_head import _hexdist
        assert _hexdist(0, 0) == 0
        assert _hexdist(1, 0) == 1
        assert _hexdist(3, -3) == 3      # (3 + 3 + 0) / 2
        assert _hexdist(2, 2) == 4       # (2 + 2 + 4) / 2

    def test_aligned_on_each_hex_axis(self):
        from hexo_a0.head_to_head import _is_aligned
        assert _is_aligned((2, 0), (5, 0)) is True     # same r  (axis 1,0)
        assert _is_aligned((0, 2), (0, 5)) is True     # same q  (axis 0,1)
        assert _is_aligned((3, 0), (0, 3)) is True     # q+r=3   (axis 1,-1)
        assert _is_aligned((3, 0), (0, 5)) is False    # no shared axis

    def test_threat_pair_requires_alignment_and_closeness(self):
        from hexo_a0.head_to_head import _is_threat_pair
        # win_length=6 -> a threat pair must be aligned AND <= 5 apart.
        assert _is_threat_pair((3, 0), (6, 0), 6) is True    # aligned, 3 apart
        assert _is_threat_pair((8, 0), (0, 8), 6) is False   # aligned but 8 apart
        assert _is_threat_pair((4, 0), (0, 5), 6) is False   # unaligned

    def test_refuted_when_two_unaligned_far_stones(self):
        from hexo_a0.head_to_head import _p2_opening_stats
        # (4,0) & (0,5): unaligned, both >= far_threshold (3) from origin.
        _, refuted = _p2_opening_stats([(4, 0), (0, 5), (1, 0), (0, 1)], 3, 6)
        assert refuted is True

    def test_refuted_when_far_aligned_but_too_far_apart(self):
        from hexo_a0.head_to_head import _p2_opening_stats
        # (8,0) & (0,8) share q+r=8 but are 8 apart -> cannot form a win_length
        # line -> NOT a real preemptive -> refuted.
        _, refuted = _p2_opening_stats([(8, 0), (0, 8)], 3, 6)
        assert refuted is True

    def test_not_refuted_close_aligned_preemptive(self):
        from hexo_a0.head_to_head import _p2_opening_stats
        # (3,0) & (6,0): both away, aligned, 3 apart (<=5) -> real preemptive.
        _, refuted = _p2_opening_stats([(3, 0), (6, 0)], 3, 6)
        assert refuted is False

    def test_not_refuted_when_one_stone_near_origin(self):
        from hexo_a0.head_to_head import _p2_opening_stats
        # Unaligned, but (1,0) contests origin -> balanced colony, NOT refuted.
        mean_d, refuted = _p2_opening_stats([(1, 0), (0, 5)], 3, 6)
        assert refuted is False
        assert mean_d == 3.0  # (1 + 5) / 2
