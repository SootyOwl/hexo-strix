"""A finite loss with a NaN/Inf gradient must NOT update the weights.

Regression for the 2026-06-02 corruption: the train-step guard only checked the
loss, so a finite-loss/NaN-gradient sailed through clip_grad_norm_ +
optimizer.step() and wrote NaN into the weights (and Adam state), poisoning
checkpoints. The guard must also skip on a non-finite gradient.
"""
from __future__ import annotations

import torch

import hexo_a0.trainer as trainer_mod
from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_a0.trainer import train_step


def _tiny_model():
    cfg = ModelConfig(
        hidden_dim=8, num_layers=1, num_heads=1, policy_hidden=8, value_hidden=8,
        graph_type="hex", conv_type="gatv2", prune_empty_edges=False,
        dropout=0.0, pre_norm=True,
    )
    return HeXONet(cfg)


def test_nan_gradient_does_not_corrupt_weights(monkeypatch):
    model = _tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)  # lr=1 so any step is visible
    p = next(model.parameters())
    w0 = p.detach().clone()

    def fake_forward_and_loss(m, examples, device, use_fp16=False, loss_scale=1.0,
                              pc_loss_weight=0.0):
        # FINITE loss but a NaN gradient — mimics an unstable backward.
        for q in m.parameters():
            q.grad = torch.zeros_like(q)
        first = next(m.parameters())
        first.grad = torch.full_like(first, float("nan"))
        z = torch.zeros(())
        return {"total_loss": z, "policy_loss": z, "value_loss": z, "pc_loss": 0.0}

    monkeypatch.setattr(trainer_mod, "_forward_and_loss", fake_forward_and_loss)
    train_step(model, opt, [("dummy", None, None)], torch.device("cpu"))

    assert torch.isfinite(p.detach()).all(), "NaN gradient corrupted the weights"
    assert torch.equal(p.detach(), w0), "optimizer stepped despite a NaN gradient"


def test_finite_gradient_still_steps(monkeypatch):
    """Sanity: a finite loss + finite gradient must still update weights."""
    model = _tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    p = next(model.parameters())
    w0 = p.detach().clone()

    def fake_forward_and_loss(m, examples, device, use_fp16=False, loss_scale=1.0,
                              pc_loss_weight=0.0):
        first = next(m.parameters())
        first.grad = torch.ones_like(first)  # finite, nonzero
        for q in list(m.parameters())[1:]:
            q.grad = torch.zeros_like(q)
        z = torch.zeros(())
        return {"total_loss": z, "policy_loss": z, "value_loss": z, "pc_loss": 0.0}

    monkeypatch.setattr(trainer_mod, "_forward_and_loss", fake_forward_and_loss)
    train_step(model, opt, [("dummy", None, None)], torch.device("cpu"))

    assert not torch.equal(p.detach(), w0), "optimizer failed to step on a finite gradient"
    assert torch.isfinite(p.detach()).all()
