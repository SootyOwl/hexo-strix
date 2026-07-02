"""Tests for the perplexity-by-move diagnostic (trainer.py).

Covers the per-graph perplexity computation (`_perplexity_by_move`) and the
stone-count bucketing / scalar emission (`_log_perplexity_by_move_scalars`).
"""

import numpy as np
import torch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_graph
from hexo_a0.model import HeXONet
from hexo_a0.trainer import (
    _PERPLEXITY_STONE_BUCKETS,
    _log_perplexity_by_move_scalars,
    _perplexity_by_move,
)

TINY_CONFIG = ModelConfig(
    hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex", conv_type="gatv2"
)


def _uniform_example(game_config=None):
    """A training example with a UNIFORM policy target over its legal moves.

    Perplexity of a uniform distribution over k outcomes is exactly k, which
    gives the target side an exact correctness anchor independent of the model.
    """
    if game_config is None:
        game_config = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    game = hexo_rs.GameState(game_config)
    data = game_to_graph(game)
    num_legal = int(data.legal_mask.sum().item())
    policy_target = torch.full((num_legal,), 1.0 / num_legal)
    return data, policy_target, num_legal


def test_target_perplexity_of_uniform_equals_legal_count():
    """exp(H) of a uniform target over k legal moves must equal k."""
    device = torch.device("cpu")
    model = HeXONet(TINY_CONFIG).eval()
    examples = [_uniform_example() for _ in range(4)]
    num_legals = [k for *_, k in examples]

    prior_perp, target_perp, counts = _perplexity_by_move(
        model, examples, device, use_fp16=False
    )

    assert target_perp.shape == (4,)
    np.testing.assert_allclose(target_perp, num_legals, rtol=1e-4)
    # Stone count must match each graph's own stone_mask (a fresh HeXO game
    # carries a seed stone, so this is not necessarily zero).
    expected_counts = [int(ex[0].stone_mask.sum().item()) for ex in examples]
    np.testing.assert_array_equal(counts, expected_counts)


def test_prior_perplexity_within_bounds():
    """Prior perplexity is bounded in [1, num_legal] for every example."""
    device = torch.device("cpu")
    model = HeXONet(TINY_CONFIG).eval()
    examples = [_uniform_example() for _ in range(4)]
    num_legals = np.array([k for *_, k in examples], dtype=np.float64)

    prior_perp, _target_perp, _counts = _perplexity_by_move(
        model, examples, device, use_fp16=False
    )

    assert np.all(prior_perp >= 1.0 - 1e-4)
    assert np.all(prior_perp <= num_legals + 1e-4)


class _FakeWriter:
    def __init__(self):
        self.scalars: dict[str, tuple[dict, int]] = {}

    def add_scalars(self, tag, d, step):
        self.scalars[tag] = (dict(d), step)


def test_bucketing_means_and_p10_and_empty_skip():
    """Buckets group by stone count; mean/p10 are correct; empty buckets skip."""
    writer = _FakeWriter()
    # stone counts land in buckets 000-004 (0,4) and 015-029 (20,28); the
    # other five buckets are empty and must be omitted.
    stone_counts = np.array([0, 4, 20, 28], dtype=np.float64)
    perplexity = np.array([2.0, 4.0, 10.0, 20.0], dtype=np.float64)

    _log_perplexity_by_move_scalars(writer, "policy/perplexity_prior", perplexity, stone_counts, step=7)

    mean_d, mean_step = writer.scalars["policy/perplexity_prior_mean"]
    p10_d, _ = writer.scalars["policy/perplexity_prior_p10"]
    assert mean_step == 7
    # 000-004 contains {2, 4} -> mean 3; 015-029 contains {10, 20} -> mean 15.
    assert mean_d["000-004"] == 3.0
    assert mean_d["015-029"] == 15.0
    assert set(mean_d) == {"000-004", "015-029"}  # empty buckets omitted
    assert set(p10_d) == {"000-004", "015-029"}
    # p10 of {10, 20} = 11.0 (numpy linear interpolation).
    assert p10_d["015-029"] == np.percentile([10.0, 20.0], 10)


def test_bucket_edges_are_contiguous_and_cover_zero():
    """Sanity: buckets are half-open, contiguous, and the first starts at 0."""
    assert _PERPLEXITY_STONE_BUCKETS[0][1] == 0
    for (_, _lo, hi), (_, nxt_lo, _) in zip(
        _PERPLEXITY_STONE_BUCKETS, _PERPLEXITY_STONE_BUCKETS[1:]
    ):
        assert hi == nxt_lo
