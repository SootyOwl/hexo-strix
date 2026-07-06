"""Unit tests for the distributional (binned) value helpers in loss.py.

These are pure tensor functions (no PyG, no torch.compile), so they are the
TDD foundation for the binned value head (Stage 1 of the value-head upgrades).
"""

import pytest

torch = pytest.importorskip("torch")

from hexo_a0.loss import (
    value_bin_centers,
    project_to_two_hot,
    decode_binned_value,
    binned_value_ce,
)


def test_centers_span_and_count():
    c = value_bin_centers(65)
    assert c.shape == (65,)
    assert torch.isclose(c[0], torch.tensor(-1.0))
    assert torch.isclose(c[-1], torch.tensor(1.0))
    # odd bin count puts a center exactly at 0 (draw)
    assert (c.abs() < 1e-6).any()


def test_centers_custom_range():
    c = value_bin_centers(5, vmin=0.0, vmax=209.0)
    assert torch.isclose(c[0], torch.tensor(0.0))
    assert torch.isclose(c[-1], torch.tensor(209.0))


def test_two_hot_on_center_is_one_hot():
    c = value_bin_centers(5)  # [-1, -0.5, 0, 0.5, 1]
    d = project_to_two_hot(torch.tensor([0.0]), c)
    assert d.shape == (1, 5)
    assert torch.isclose(d.sum(), torch.tensor(1.0))
    assert torch.isclose(d[0, 2], torch.tensor(1.0))


def test_two_hot_expectation_roundtrip():
    c = value_bin_centers(65)
    t = torch.tensor([-0.9, -0.3, 0.0, 0.42, 0.99])
    d = project_to_two_hot(t, c)
    # The two-hot distribution's expectation exactly reconstructs the target.
    assert torch.allclose((d * c).sum(-1), t, atol=1e-5)
    assert torch.allclose(d.sum(-1), torch.ones_like(t))


def test_two_hot_adjacent_bins_only():
    c = value_bin_centers(5)  # step 0.5
    d = project_to_two_hot(torch.tensor([0.25]), c)  # midway between idx 2 and 3
    nz = (d[0] > 0).nonzero().flatten().tolist()
    assert nz == [2, 3]
    assert torch.isclose(d[0, 2], torch.tensor(0.5))
    assert torch.isclose(d[0, 3], torch.tensor(0.5))


def test_two_hot_clamps_out_of_range():
    c = value_bin_centers(5)
    d = project_to_two_hot(torch.tensor([2.0, -2.0]), c)
    assert torch.isclose(d[0, -1], torch.tensor(1.0))
    assert torch.isclose(d[1, 0], torch.tensor(1.0))


def test_decode_matches_softmax_expectation():
    c = value_bin_centers(65)
    logits = torch.randn(4, 65)
    v = decode_binned_value(logits, c)
    expect = (torch.softmax(logits, -1) * c).sum(-1)
    assert torch.allclose(v, expect)
    assert (v >= -1).all() and (v <= 1).all()


def test_ce_lower_at_target_than_uniform():
    c = value_bin_centers(9)
    t = torch.tensor([0.3])
    d = project_to_two_hot(t, c)
    good = binned_value_ce(torch.log(d + 1e-9), t, c)  # logits peaked at target bins
    bad = binned_value_ce(torch.zeros(1, 9), t, c)      # uniform logits
    assert (good < bad).all()


def test_ce_is_per_example_shape():
    c = value_bin_centers(9)
    logits = torch.randn(7, 9)
    t = torch.rand(7) * 2 - 1
    ce = binned_value_ce(logits, t, c)
    assert ce.shape == (7,)
    assert (ce >= 0).all()


def test_ce_gradient_flows_to_logits():
    c = value_bin_centers(9)
    logits = torch.randn(3, 9, requires_grad=True)
    t = torch.tensor([-0.5, 0.0, 0.5])
    binned_value_ce(logits, t, c).mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
