"""Tests for the AlphaZero loss function (AC-10a, AC-10b, AC-10c) and KL policy loss."""

import math

import pytest
import torch
import torch.nn.functional as F

from hexo_a0.loss import alphazero_loss, kl_policy_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_inputs(n: int = 4):
    """Return simple tensors with gradients enabled."""
    logits = torch.zeros(n, requires_grad=True)
    target = torch.full((n,), 1.0 / n)
    value = torch.tensor(0.0, requires_grad=True)
    value_target = torch.tensor(1.0)
    return logits, target, value, value_target


# ---------------------------------------------------------------------------
# AC-10a: scalar, positive, finite
# ---------------------------------------------------------------------------

class TestLossShape:
    def test_total_loss_is_scalar(self):
        logits, target, value, vt = _uniform_inputs()
        total, _ = alphazero_loss(logits, target, value, vt)
        assert total.shape == (), f"Expected scalar, got shape {total.shape}"

    def test_total_loss_is_positive(self):
        logits, target, value, vt = _uniform_inputs()
        total, _ = alphazero_loss(logits, target, value, vt)
        assert total.item() > 0.0

    def test_total_loss_is_finite(self):
        logits, target, value, vt = _uniform_inputs()
        total, _ = alphazero_loss(logits, target, value, vt)
        assert math.isfinite(total.item())

    def test_components_dict_keys(self):
        logits, target, value, vt = _uniform_inputs()
        _, components = alphazero_loss(logits, target, value, vt)
        assert set(components.keys()) == {"policy_loss", "value_loss"}

    def test_component_values_are_scalars(self):
        logits, target, value, vt = _uniform_inputs()
        _, components = alphazero_loss(logits, target, value, vt)
        for key, tensor in components.items():
            assert tensor.shape == (), f"{key} should be scalar, got {tensor.shape}"

    def test_component_values_are_finite(self):
        logits, target, value, vt = _uniform_inputs()
        _, components = alphazero_loss(logits, target, value, vt)
        for key, tensor in components.items():
            assert math.isfinite(tensor.item()), f"{key} is not finite"

    def test_total_equals_sum_of_components(self):
        logits, target, value, vt = _uniform_inputs()
        total, components = alphazero_loss(logits, target, value, vt)
        expected = components["policy_loss"] + components["value_loss"]
        assert torch.isclose(total, expected), (
            f"total {total.item():.6f} != sum of components {expected.item():.6f}"
        )


# ---------------------------------------------------------------------------
# AC-10b: gradients flow to both policy_logits and value
# ---------------------------------------------------------------------------

class TestGradients:
    def test_gradient_flows_to_policy_logits(self):
        logits, target, value, vt = _uniform_inputs()
        total, _ = alphazero_loss(logits, target, value, vt)
        total.backward()
        assert logits.grad is not None, "No gradient on policy_logits"
        assert logits.grad.shape == logits.shape

    def test_gradient_flows_to_value(self):
        logits, target, value, vt = _uniform_inputs()
        total, _ = alphazero_loss(logits, target, value, vt)
        total.backward()
        assert value.grad is not None, "No gradient on value"

    def test_policy_gradient_is_nonzero(self):
        # Non-uniform target guarantees nonzero policy gradient.
        logits = torch.zeros(4, requires_grad=True)
        target = torch.tensor([0.5, 0.3, 0.1, 0.1])
        value = torch.tensor(0.0, requires_grad=True)
        total, _ = alphazero_loss(logits, target, value, torch.tensor(1.0))
        total.backward()
        assert logits.grad.abs().sum().item() > 0.0

    def test_value_gradient_is_nonzero_when_not_at_target(self):
        logits, target, value, vt = _uniform_inputs()
        # value=0, value_target=1 → MSE gradient should be nonzero
        total, _ = alphazero_loss(logits, target, value, vt)
        total.backward()
        assert value.grad.item() != 0.0


# ---------------------------------------------------------------------------
# AC-10c: near-perfect predictions → near-zero loss
# ---------------------------------------------------------------------------

class TestNearPerfect:
    def test_near_perfect_policy_gives_low_policy_loss(self):
        n = 5
        target_idx = 2
        # Very large logit for the target move, small elsewhere.
        logits = torch.full((n,), -10.0, requires_grad=True)
        with torch.no_grad():
            logits_data = logits.clone()
            logits_data[target_idx] = 10.0
        logits = logits_data.requires_grad_(True)

        target = torch.zeros(n)
        target[target_idx] = 1.0  # one-hot

        value = torch.tensor(1.0, requires_grad=True)
        value_target = torch.tensor(1.0)

        total, components = alphazero_loss(logits, target, value, value_target)
        assert components["policy_loss"].item() < 0.01, (
            f"Policy loss should be near zero, got {components['policy_loss'].item():.6f}"
        )

    def test_near_perfect_value_gives_low_value_loss(self):
        n = 4
        logits = torch.zeros(n, requires_grad=True)
        target = torch.full((n,), 0.25)

        value = torch.tensor(0.9999, requires_grad=True)
        value_target = torch.tensor(1.0)

        total, components = alphazero_loss(logits, target, value, value_target)
        assert components["value_loss"].item() < 1e-6, (
            f"Value loss should be near zero, got {components['value_loss'].item():.8f}"
        )

    def test_near_perfect_both_gives_near_zero_total(self):
        n = 5
        target_idx = 3
        logits = torch.full((n,), -10.0)
        logits[target_idx] = 10.0
        logits = logits.requires_grad_(True)

        target = torch.zeros(n)
        target[target_idx] = 1.0

        value = torch.tensor(1.0, requires_grad=True)
        value_target = torch.tensor(1.0)

        total, _ = alphazero_loss(logits, target, value, value_target)
        assert total.item() < 0.01, (
            f"Total loss should be near zero, got {total.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Alpha / beta weighting
# ---------------------------------------------------------------------------

class TestWeighting:
    def test_alpha_scales_policy_loss(self):
        logits, target, value, vt = _uniform_inputs()
        total1, _ = alphazero_loss(logits, target, value, vt, alpha=1.0, beta=0.0)
        total2, _ = alphazero_loss(logits, target, value, vt, alpha=2.0, beta=0.0)
        assert torch.isclose(total2, 2.0 * total1), (
            f"alpha=2 should double policy-only loss: {total1.item()} vs {total2.item()}"
        )

    def test_beta_scales_value_loss(self):
        logits, target, value, vt = _uniform_inputs()
        total1, _ = alphazero_loss(logits, target, value, vt, alpha=0.0, beta=1.0)
        total2, _ = alphazero_loss(logits, target, value, vt, alpha=0.0, beta=3.0)
        assert torch.isclose(total2, 3.0 * total1), (
            f"beta=3 should triple value-only loss: {total1.item()} vs {total2.item()}"
        )

    def test_different_alpha_beta_change_total(self):
        logits, target, value, vt = _uniform_inputs()
        total_default, _ = alphazero_loss(logits, target, value, vt)
        total_custom, _ = alphazero_loss(logits, target, value, vt, alpha=0.5, beta=2.0)
        assert not torch.isclose(total_default, total_custom), (
            "Different alpha/beta should produce different total loss"
        )

    def test_total_is_weighted_sum_of_components(self):
        logits, target, value, vt = _uniform_inputs()
        alpha, beta = 0.7, 1.3
        total, components = alphazero_loss(logits, target, value, vt, alpha=alpha, beta=beta)
        expected = alpha * components["policy_loss"] + beta * components["value_loss"]
        assert torch.isclose(total, expected), (
            f"total {total.item():.6f} != weighted sum {expected.item():.6f}"
        )


# ---------------------------------------------------------------------------
# kl_policy_loss tests
# ---------------------------------------------------------------------------

class TestKLPolicyLoss:
    def test_kl_is_non_negative(self):
        """KL divergence must always be >= 0."""
        logits = torch.tensor([1.0, 2.0, 0.5, -1.0])
        target = torch.tensor([0.4, 0.3, 0.2, 0.1])
        loss = kl_policy_loss(logits, target)
        assert loss.item() >= 0.0, f"KL should be >= 0, got {loss.item()}"

    def test_kl_is_zero_when_target_equals_softmax(self):
        """KL(p || p) = 0 when network exactly predicts the target distribution."""
        logits = torch.tensor([1.0, 2.0, 0.5, -1.0])
        # target == softmax(logits) → KL should be 0
        target = F.softmax(logits, dim=-1).detach()
        loss = kl_policy_loss(logits, target)
        assert abs(loss.item()) < 1e-5, (
            f"KL should be ~0 when target == softmax(logits), got {loss.item():.8f}"
        )

    def test_kl_matches_hand_computed_value(self):
        """Verify KL against a hand-computed reference."""
        # p = [0.5, 0.5], q = softmax([0, 1]) = [e^0, e^1] / (1+e) = [1/(1+e), e/(1+e)]
        logits = torch.tensor([0.0, 1.0])
        p = torch.tensor([0.5, 0.5])

        log_q = F.log_softmax(logits, dim=-1)
        # KL(p||q) = sum(p * log(p)) - sum(p * log(q))
        expected_kl = (p * (p.log() - log_q)).sum().item()

        loss = kl_policy_loss(logits, p)
        assert abs(loss.item() - expected_kl) < 1e-5, (
            f"KL {loss.item():.8f} != expected {expected_kl:.8f}"
        )

    def test_no_nan_when_target_has_zeros(self):
        """xlogy(0, 0) = 0 — the entropy term must not produce NaN."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # Sparse target: only one non-zero entry
        target = torch.tensor([0.0, 0.0, 0.0, 1.0])
        loss = kl_policy_loss(logits, target)
        assert not torch.isnan(loss), "KL loss should not be NaN with zero target entries"
        assert loss.item() >= 0.0

    def test_gradient_flows_to_logits(self):
        """Backpropagation should produce non-None, finite gradients on policy_logits."""
        logits = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
        target = torch.tensor([0.5, 0.3, 0.2])
        loss = kl_policy_loss(logits, target)
        loss.backward()
        assert logits.grad is not None, "No gradient on policy_logits"
        assert not torch.isnan(logits.grad).any(), "NaN in gradient"

    def test_kl_gradient_equals_cross_entropy_gradient(self):
        """KL(p||q) and CE(p, q) share the same gradient w.r.t. logits (entropy is const)."""
        logits_kl = torch.tensor([1.0, 2.0, 0.5, -1.0], requires_grad=True)
        logits_ce = logits_kl.detach().clone().requires_grad_(True)
        target = torch.tensor([0.4, 0.3, 0.2, 0.1])

        # KL backward
        kl_loss = kl_policy_loss(logits_kl, target)
        kl_loss.backward()

        # Cross-entropy backward (entropy term constant w.r.t. logits)
        ce_loss = -(target * F.log_softmax(logits_ce, dim=-1)).sum()
        ce_loss.backward()

        assert torch.allclose(logits_kl.grad, logits_ce.grad, atol=1e-6), (
            f"KL and CE gradients differ:\n  KL grad: {logits_kl.grad}\n  CE grad: {logits_ce.grad}"
        )


# ---------------------------------------------------------------------------
# Path consistency loss (PCZero)
# ---------------------------------------------------------------------------


class TestPathConsistencyLoss:
    def test_zero_when_consistent(self):
        """Perfectly alternating values matching outcome → near-zero loss."""
        from hexo_a0.loss import path_consistency_loss
        values = torch.tensor([1.0, -1.0, 1.0])
        outcome = torch.tensor(1.0)
        loss = path_consistency_loss(values, outcome)
        assert loss.item() < 0.01

    def test_nonzero_when_inconsistent(self):
        """Inconsistent predictions → nonzero loss."""
        from hexo_a0.loss import path_consistency_loss
        values = torch.tensor([0.9, -0.5, 0.8])
        outcome = torch.tensor(1.0)
        loss = path_consistency_loss(values, outcome)
        assert loss.item() > 0.1

    def test_single_position(self):
        """Single position → zero loss."""
        from hexo_a0.loss import path_consistency_loss
        values = torch.tensor([0.5])
        outcome = torch.tensor(1.0)
        loss = path_consistency_loss(values, outcome)
        assert loss.item() == 0.0
