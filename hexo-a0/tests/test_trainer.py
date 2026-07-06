"""Tests for train_step in trainer.py (AC-15a, AC-15b, AC-15d, AC-15e, AC-16a, AC-16b)."""

import math

import pytest
import torch
import torch.nn.functional as F

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_graph
from hexo_a0.model import HeXONet


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_example(game_config=None):
    """Create a synthetic training example from a real game state."""
    if game_config is None:
        game_config = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    game = hexo_rs.GameState(game_config)
    data = game_to_graph(game)
    num_legal = data.legal_mask.sum().item()
    # Uniform policy target
    policy_target = torch.full((int(num_legal),), 1.0 / num_legal)
    value_target = torch.tensor(0.0)
    return (data, policy_target, value_target)


TINY_CONFIG = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex", conv_type="gatv2")


@pytest.fixture
def model():
    return HeXONet(TINY_CONFIG)


@pytest.fixture
def examples():
    return [make_example() for _ in range(3)]


# ---------------------------------------------------------------------------
# AC-15a: Loss decreases over multiple steps (overfitting test)
# ---------------------------------------------------------------------------

class TestLossDecreases:
    def test_loss_decreases_over_20_steps(self, model):
        """Loss on fixed data should decrease after repeated optimizer steps."""
        from hexo_a0.trainer import train_step

        torch.manual_seed(42)
        device = torch.device("cpu")
        fixed_examples = [make_example() for _ in range(2)]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for _ in range(20):
            metrics = train_step(model, optimizer, fixed_examples, device)
            losses.append(metrics["total_loss"])

        assert losses[0] > losses[-1], (
            f"Loss should decrease over 20 steps: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# AC-15b: Gradients reach all model parameters
# ---------------------------------------------------------------------------

class TestGradients:
    def test_all_params_have_grad_after_one_step(self, model, examples):
        """After one train_step, every parameter should have a non-None gradient."""
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_step(model, optimizer, examples, device)

        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient after train_step"
            )


# ---------------------------------------------------------------------------
# AC-15d: Variable-length policy targets
# ---------------------------------------------------------------------------

class TestVariableLengthPolicy:
    def test_handles_different_legal_move_counts(self, model):
        """Batch with examples from different positions (different num_legal) should work."""
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")

        # Create examples with different game configs → different board sizes → different legal counts
        config_a = hexo_rs.GameConfig(win_length=3, placement_radius=3, max_moves=30)
        config_b = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        mixed_examples = [
            make_example(config_a),
            make_example(config_b),
            make_example(config_a),
        ]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = train_step(model, optimizer, mixed_examples, device)

        assert math.isfinite(metrics["total_loss"]), "Loss should be finite"
        assert metrics["total_loss"] > 0.0, "Loss should be positive"


# ---------------------------------------------------------------------------
# AC-15e: Gradient norm is clipped
# ---------------------------------------------------------------------------

class TestGradientClipping:
    def test_grad_norm_is_clipped(self, model, examples):
        """With max_grad_norm=0.01, the actual gradient norm should be <= 0.01 + epsilon."""
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_grad_norm = 0.01

        # train_step clips then steps. We verify by inspecting param grads after the call.
        train_step(model, optimizer, examples, device, max_grad_norm=max_grad_norm)

        # Compute total grad norm from grads stored in params after step
        total_norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
        )
        assert total_norm.item() <= max_grad_norm + 1e-5, (
            f"Grad norm {total_norm.item():.6f} exceeds max_grad_norm {max_grad_norm}"
        )


# ---------------------------------------------------------------------------
# AC-16b: Returns loss components dict
# ---------------------------------------------------------------------------

class TestReturnDict:
    def test_returns_dict_with_required_keys(self, model, examples):
        """train_step should return dict with policy_loss, value_loss, total_loss."""
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = train_step(model, optimizer, examples, device)

        assert set(metrics.keys()) == {"policy_loss", "value_loss", "total_loss", "pc_loss"}, (
            f"Unexpected keys: {set(metrics.keys())}"
        )

    def test_all_loss_values_are_finite(self, model, examples):
        """All returned loss values should be finite floats."""
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = train_step(model, optimizer, examples, device)

        for key, value in metrics.items():
            assert math.isfinite(value), f"{key}={value} is not finite"

    def test_total_equals_policy_plus_value(self, model, examples):
        """total_loss should equal policy_loss + value_loss."""
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = train_step(model, optimizer, examples, device)

        expected = metrics["policy_loss"] + metrics["value_loss"]
        assert abs(metrics["total_loss"] - expected) < 1e-5, (
            f"total_loss {metrics['total_loss']:.6f} != "
            f"policy_loss + value_loss {expected:.6f}"
        )


# ---------------------------------------------------------------------------
# AC-16a: Batch loss is mean of per-graph losses
# ---------------------------------------------------------------------------

class TestBatchMeanLoss:
    def test_batch_loss_equals_mean_of_individual_losses(self, model, examples):
        """The reported losses should equal the mean of per-graph losses computed manually."""
        from hexo_a0.trainer import train_step
        from hexo_a0.loss import kl_policy_loss
        from torch_geometric.data import Batch

        device = torch.device("cpu")
        model.eval()  # Deterministic (no dropout)

        data_list, policy_targets, value_targets = zip(*examples)

        batch = Batch.from_data_list(list(data_list)).to(device)

        # Run train_step with lr=0 so weights don't change
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
        metrics = train_step(model, optimizer, examples, device)

        # After train_step model is in train mode; switch to eval and recompute
        model.eval()
        policy_logits_list, values = model.forward_batch(batch)

        manual_policy = 0.0
        manual_value = 0.0
        for i, (logits_i, pi_i, v_i) in enumerate(zip(policy_logits_list, policy_targets, value_targets)):
            manual_policy += kl_policy_loss(logits_i, pi_i.to(device)).item()
            manual_value += F.mse_loss(values[i], v_i.to(device)).item()
        n = len(policy_logits_list)
        manual_policy /= n
        manual_value /= n
        manual_total = manual_policy + manual_value

        assert abs(metrics["total_loss"] - manual_total) < 1e-4, (
            f"Batch total_loss {metrics['total_loss']:.6f} != manual mean {manual_total:.6f}"
        )
        assert abs(metrics["policy_loss"] - manual_policy) < 1e-4, (
            f"Batch policy_loss {metrics['policy_loss']:.6f} != manual mean {manual_policy:.6f}"
        )


# ---------------------------------------------------------------------------
# Trainer class tests (AC-17a/b/c/d, AC-18a/b/c)
# ---------------------------------------------------------------------------

import tempfile
import os
import copy


import hexo_rs
from hexo_a0.config import (
    EvalConfig,
    FullConfig,
    MCTSConfig,
    ModelConfig,
    PythonSelfPlayConfig,
    RunConfig,
    SealbotConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from hexo_a0.model import HeXONet


from conftest import _TINY_GAME_CONFIG as TINY_GAME_CONFIG, _TINY_MODEL_CONFIG as TINY_MODEL_CONFIG, _TINY_FULL_CONFIG as TINY_FULL_CONFIG, _TINY_TRAINING_CONFIG as TINY_TRAINING_CONFIG


@pytest.fixture
def tiny_trainer():
    """A Trainer instance built from tiny configs with buffer pre-filled."""
    from hexo_a0.trainer import Trainer

    model = HeXONet(TINY_MODEL_CONFIG)
    device = torch.device("cpu")
    trainer = Trainer(
        model=model,
        config=TINY_FULL_CONFIG,
        game_config=TINY_GAME_CONFIG,
        device=device,
    )
    _fill_buffer(trainer)
    return trainer


def _fill_buffer(trainer, n=10):
    """Populate the buffer with synthetic examples (no self-play needed)."""
    from hexo_a0.self_play import TrainingExample
    from hexo_a0.graph import game_to_graph
    import hexo_rs

    game = hexo_rs.GameState(trainer.game_config)
    data = game_to_graph(game)
    n_legal = data.legal_mask.sum().item()
    for _ in range(n):
        trainer.buffer.add(TrainingExample(
            data=data,
            policy_target=torch.ones(n_legal) / n_legal,
            value_target=0.0,
        ))


class TestTrainIteration:
    """AC-17: train trains on buffer data."""

    def test_buffer_has_data(self, tiny_trainer):
        """AC-17a: buffer has examples (filled by fixture)."""
        assert len(tiny_trainer.buffer) > 0, "Buffer should have examples after self-play"

    def test_model_params_change_after_iteration(self, tiny_trainer):
        """AC-17b: at least some model parameters change after one iteration."""
        params_before = {
            name: param.detach().clone()
            for name, param in tiny_trainer.model.named_parameters()
        }
        tiny_trainer.train()
        params_after = dict(tiny_trainer.model.named_parameters())

        changed = any(
            not torch.allclose(params_before[name], params_after[name])
            for name in params_before
        )
        assert changed, "At least some model parameters should change after one iteration"

    def test_multiple_train_calls_count_steps(self, tiny_trainer):
        """Train steps increment correctly across multiple train() calls."""
        tiny_trainer.train()  # 100 steps default
        tiny_trainer.train()  # 100 more
        assert tiny_trainer.train_steps == 200, (
            f"Expected 200 steps, got {tiny_trainer.train_steps}"
        )

    def test_returns_metrics_dict(self, tiny_trainer):
        """AC-17d: train returns dict with required keys."""
        metrics = tiny_trainer.train()
        required_keys = {"total_loss", "policy_loss", "value_loss", "mean_game_length", "buffer_size"}
        assert required_keys.issubset(set(metrics.keys())), (
            f"Missing keys: {required_keys - set(metrics.keys())}"
        )

    def test_prefetch_thread_shuts_down_cleanly(self, tiny_trainer):
        """Fix 4: the background prefetch worker must not leak across train() calls.

        Each Trainer.train() call starts a daemon prefetch thread (when
        reanalyze is off, the common case). The success path should signal
        stop, drain the queue, and join — so no "hexo-prefetch" threads are
        alive after train() returns.
        """
        import threading

        def count_prefetch_threads() -> int:
            return sum(1 for t in threading.enumerate() if t.name == "hexo-prefetch")

        before = count_prefetch_threads()
        tiny_trainer.train()
        after_first = count_prefetch_threads()
        tiny_trainer.train()
        after_second = count_prefetch_threads()

        assert after_first == before, (
            f"Prefetch thread leaked after first train(): "
            f"{before} → {after_first}"
        )
        assert after_second == before, (
            f"Prefetch thread leaked after second train(): "
            f"{before} → {after_second}"
        )


class TestCheckpointing:
    """AC-18: save_checkpoint / load_checkpoint round-trip."""

    def test_checkpoint_file_exists(self, tiny_trainer):
        """AC-18b: a single .pt file is created by save_checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            tiny_trainer.train()
            tiny_trainer.save_checkpoint(path)
            assert os.path.isfile(path), f"Checkpoint file not found at {path}"

    def test_model_weights_restored(self, tiny_trainer):
        """AC-18a: model weights match after save + load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            tiny_trainer.train()
            tiny_trainer.save_checkpoint(path)

            # New trainer with fresh (untrained) model
            from hexo_a0.trainer import Trainer

            fresh_model = HeXONet(TINY_MODEL_CONFIG)
            device = torch.device("cpu")
            new_trainer = Trainer(
                model=fresh_model,
                config=TINY_FULL_CONFIG,
                game_config=TINY_GAME_CONFIG,
                device=device,
            )
            new_trainer.load_checkpoint(path)

            for name, param in tiny_trainer.model.named_parameters():
                loaded_param = dict(new_trainer.model.named_parameters())[name]
                assert torch.allclose(param, loaded_param), (
                    f"Parameter '{name}' does not match after checkpoint round-trip"
                )

    def test_iteration_count_restored(self, tiny_trainer):
        """AC-18a: iteration count is restored after load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            tiny_trainer.train()
            saved_iteration = tiny_trainer.train_steps
            tiny_trainer.save_checkpoint(path)

            from hexo_a0.trainer import Trainer

            fresh_model = HeXONet(TINY_MODEL_CONFIG)
            new_trainer = Trainer(
                model=fresh_model,
                config=TINY_FULL_CONFIG,
                game_config=TINY_GAME_CONFIG,
                device=torch.device("cpu"),
            )
            new_trainer.load_checkpoint(path)

            assert new_trainer.train_steps == saved_iteration, (
                f"Expected iteration={saved_iteration}, got {new_trainer.train_steps}"
            )

    def test_optimizer_state_restored(self, tiny_trainer):
        """AC-18c: optimizer state_dict matches after save + load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            tiny_trainer.train()
            tiny_trainer.save_checkpoint(path)

            from hexo_a0.trainer import Trainer

            fresh_model = HeXONet(TINY_MODEL_CONFIG)
            new_trainer = Trainer(
                model=fresh_model,
                config=TINY_FULL_CONFIG,
                game_config=TINY_GAME_CONFIG,
                device=torch.device("cpu"),
            )
            new_trainer.load_checkpoint(path)

            orig_state = tiny_trainer.optimizer.state_dict()
            loaded_state = new_trainer.optimizer.state_dict()

            # Compare param_groups (weight_decay, betas, etc.)
            # LR and initial_lr may differ because load_checkpoint resets
            # them to the configured value.
            for orig_pg, loaded_pg in zip(orig_state["param_groups"], loaded_state["param_groups"]):
                for key in orig_pg:
                    if key in ("lr", "initial_lr"):
                        continue
                    assert orig_pg[key] == loaded_pg[key], (
                        f"Optimizer param_groups[{key}] mismatch: {orig_pg[key]} != {loaded_pg[key]}"
                    )
            # Compare state tensors
            for key in orig_state["state"]:
                for tensor_key in orig_state["state"][key]:
                    orig_val = orig_state["state"][key][tensor_key]
                    loaded_val = loaded_state["state"][key][tensor_key]
                    if isinstance(orig_val, torch.Tensor):
                        assert torch.allclose(orig_val, loaded_val), (
                            f"Optimizer state[{key}][{tensor_key}] does not match"
                        )
                    else:
                        assert orig_val == loaded_val, (
                            f"Optimizer state[{key}][{tensor_key}]: {orig_val} != {loaded_val}"
                        )

    def test_game_config_stored_as_plain_dict(self, tiny_trainer):
        """AC-18a: game_config in checkpoint is a plain dict with int values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            tiny_trainer.train()
            tiny_trainer.save_checkpoint(path)

            raw = torch.load(path, map_location="cpu", weights_only=False)
            game_cfg = raw["game_config"]

            assert isinstance(game_cfg, dict), "game_config should be a plain dict"
            assert "win_length" in game_cfg
            assert "placement_radius" in game_cfg
            assert "max_moves" in game_cfg
            assert isinstance(game_cfg["win_length"], int)
            assert isinstance(game_cfg["placement_radius"], int)
            assert isinstance(game_cfg["max_moves"], int)


# ---------------------------------------------------------------------------
# Integration test: full training step with precomputed index tensors
# ---------------------------------------------------------------------------

class TestPrecomputedIndicesIntegration:
    """Integration test for the index-tensor training path (Tasks 1-7)."""

    def test_forward_and_loss_with_precomputed_indices_produces_valid_gradients(self):
        """Full path: build batch → precompute indices → forward → loss → backward.

        _forward_and_loss internally precomputes legal_idx / stone_idx from the
        batched graph and passes them to model._forward_batch_core.  This test
        verifies that gradients flow correctly through that path on CPU.
        """
        from hexo_a0.trainer import _forward_and_loss

        device = torch.device("cpu")
        model = HeXONet(TINY_CONFIG)
        model.train()

        examples = [make_example() for _ in range(3)]

        # Zero out gradients before the call (mirroring what train_step does)
        for p in model.parameters():
            p.grad = None

        losses = _forward_and_loss(model, examples, device)

        # _forward_and_loss calls .backward() internally; check gradients exist
        params_with_grad = [
            (name, p)
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        assert len(params_with_grad) > 0, (
            "No parameters received gradients after _forward_and_loss"
        )

        # At least one gradient should be nonzero
        nonzero_grads = [
            name
            for name, p in params_with_grad
            if p.grad.abs().max().item() > 0.0
        ]
        assert len(nonzero_grads) > 0, (
            "All gradients are zero after _forward_and_loss — backward may not have run"
        )

        # Loss values should be finite and positive
        total_loss = float(losses["total_loss"].item())
        policy_loss = float(losses["policy_loss"].item())
        value_loss = float(losses["value_loss"].item())

        assert math.isfinite(total_loss), f"total_loss is not finite: {total_loss}"
        assert math.isfinite(policy_loss), f"policy_loss is not finite: {policy_loss}"
        assert math.isfinite(value_loss), f"value_loss is not finite: {value_loss}"
        assert total_loss > 0.0, f"total_loss should be positive, got {total_loss}"


class TestCheckpointingResume:
    """Resume-related checkpoint tests."""

    def test_resume_short_chunk_mid_stream(self, tiny_trainer, tmp_path):
        """Mid-stream resume sets _resume_short_chunk to realign to chunk boundary."""
        from hexo_a0.trainer import Trainer

        ckpt_path = str(tmp_path / "ckpt.pt")
        tiny_trainer.train()
        tiny_trainer.save_checkpoint(ckpt_path)

        # Rewrite train_steps to a mid-chunk value (8098 % 100 == 98).
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw["train_steps"] = 8098
        torch.save(raw, ckpt_path)

        fresh = Trainer(
            model=HeXONet(TINY_MODEL_CONFIG),
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        fresh.load_checkpoint(ckpt_path)
        assert fresh.train_steps == 8098
        assert fresh._resume_short_chunk == 2, (
            f"Expected short chunk of 2 to realign 8098 -> 8100, got {fresh._resume_short_chunk}"
        )

    def test_resume_short_chunk_aligned(self, tiny_trainer, tmp_path):
        """Aligned resume leaves _resume_short_chunk at zero."""
        from hexo_a0.trainer import Trainer

        ckpt_path = str(tmp_path / "ckpt.pt")
        tiny_trainer.train()
        tiny_trainer.save_checkpoint(ckpt_path)

        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw["train_steps"] = 8100
        torch.save(raw, ckpt_path)

        fresh = Trainer(
            model=HeXONet(TINY_MODEL_CONFIG),
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        fresh.load_checkpoint(ckpt_path)
        assert fresh.train_steps == 8100
        assert fresh._resume_short_chunk == 0


class TestResumeWarmupScheduler:
    """Unit tests for `_ResumeWarmupScheduler` (2026-05-26 JK-cat work)."""

    def test_linear_ramp_then_defer(self):
        """LR ramps linearly start→target over warmup_steps, then defers."""
        from hexo_a0.trainer import _ResumeWarmupScheduler

        # Build a trivial optimizer with one param group, plus a fake
        # underlying scheduler that we can probe.
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)

        class _Stub:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1
                # Simulate a schedule that pushes LR to 0.5 once stepped.
                for g in opt.param_groups:
                    g["lr"] = 0.5

            def state_dict(self):
                return {"steps": self.steps}

            def load_state_dict(self, s):
                self.steps = s.get("steps", 0)

        underlying = _Stub()
        sched = _ResumeWarmupScheduler(
            underlying, opt,
            start_lr=0.0, target_lr=1.0, warmup_steps=4,
        )
        # Initial LR set to start_lr.
        assert opt.param_groups[0]["lr"] == 0.0
        # Step 1..4 should ramp 0.25, 0.5, 0.75, 1.0.
        expected = [0.25, 0.5, 0.75, 1.0]
        for e in expected:
            sched.step()
            assert abs(opt.param_groups[0]["lr"] - e) < 1e-9, (
                f"expected lr={e}, got {opt.param_groups[0]['lr']}"
            )
        # Underlying must not have been stepped during the ramp.
        assert underlying.steps == 0
        # Step 5 hands off to underlying which sets LR=0.5.
        sched.step()
        assert underlying.steps == 1
        assert abs(opt.param_groups[0]["lr"] - 0.5) < 1e-9

    def test_state_dict_roundtrip(self):
        """Save and restore preserves the ramp position."""
        from hexo_a0.trainer import _ResumeWarmupScheduler

        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)

        class _Stub:
            def state_dict(self):
                return {}

            def load_state_dict(self, _):
                pass

            def step(self):
                pass

        sched1 = _ResumeWarmupScheduler(
            _Stub(), opt, start_lr=1e-6, target_lr=1e-3, warmup_steps=10,
        )
        sched1.step()
        sched1.step()
        state = sched1.state_dict()
        # Build a fresh wrapper and load state.
        sched2 = _ResumeWarmupScheduler(
            _Stub(), opt, start_lr=1e-6, target_lr=1e-3, warmup_steps=10,
        )
        sched2.load_state_dict(state)
        # Both should now be at the same ramp position (next step should
        # yield the same LR).
        sched1.step()
        lr1 = opt.param_groups[0]["lr"]
        sched2.step()
        lr2 = opt.param_groups[0]["lr"]
        assert abs(lr1 - lr2) < 1e-12


# ---------------------------------------------------------------------------
# train_step: empty examples returns zero-loss dict (line 48)
# ---------------------------------------------------------------------------

class TestTrainStepEmpty:
    def test_empty_examples_returns_zero_losses(self, model):
        """train_step with empty list should return all-zero losses without touching model."""
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        params_before = {n: p.clone() for n, p in model.named_parameters()}

        metrics = train_step(model, optimizer, [], device)

        assert metrics == {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "pc_loss": 0.0}
        # Model weights should be unchanged
        for name, param in model.named_parameters():
            assert torch.equal(param, params_before[name]), (
                f"Parameter '{name}' changed on empty train_step"
            )


# ---------------------------------------------------------------------------
# NaN/Inf loss guard: optimizer.step() is NOT called (lines 79-80)
# ---------------------------------------------------------------------------

class TestNaNLossGuard:
    def test_nan_loss_skips_optimizer_step(self, model, examples):
        """When loss is NaN, optimizer.step() should not be called."""
        from unittest.mock import patch
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Inject NaN into model output by patching _forward_batch_core
        orig_core = model._forward_batch_core

        def nan_core(batch, **kwargs):
            all_logits, legal_counts, values = orig_core(batch, **kwargs)
            return all_logits * float("nan"), legal_counts, values

        orig_step = optimizer.step
        step_called = False

        def track_step(*a, **kw):
            nonlocal step_called
            step_called = True
            return orig_step(*a, **kw)

        optimizer.step = track_step
        with patch.object(model, "_forward_batch_core", nan_core):
            metrics = train_step(model, optimizer, examples, device)

        assert not step_called, "optimizer.step() should NOT be called when loss is NaN"
        assert math.isnan(metrics["total_loss"]), "total_loss should be NaN"

    def test_train_step_bf16_autocast(self, model, examples):
        """train_step with use_fp16=True should not crash on dtype mismatches.

        Regression: scatter_reduce requires self.dtype == src.dtype, which
        fails if zeros are float32 but logits are bfloat16 under autocast.
        """
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_step(model, optimizer, examples, device, use_fp16=True)
        assert metrics["total_loss"] >= 0.0 or math.isnan(metrics["total_loss"])
        assert "policy_loss" in metrics
        assert "value_loss" in metrics


# ---------------------------------------------------------------------------
# Cosine LR scheduler creation (lines 125-126)
# ---------------------------------------------------------------------------

class TestCosineLRScheduler:
    def test_cosine_scheduler_created(self):
        """Trainer with lr_schedule='cosine' and total_train_steps>0 creates a scheduler."""
        from hexo_a0.trainer import Trainer

        from dataclasses import replace
        full = replace(TINY_FULL_CONFIG,
            training=TrainingConfig(
                batch_size=4, buffer_capacity=100,
                lr_schedule="cosine", total_train_steps=100, lr_min=1e-6,
                lr_warmup_steps=0, edge_budget=0, grad_accumulation=False,
                target_reuse=0.0, augment_symmetries=False,
            ),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        assert trainer.scheduler is not None, "Scheduler should be created for cosine schedule"

        # Verify it's a CosineAnnealingWarmRestarts
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        assert isinstance(trainer.scheduler, CosineAnnealingWarmRestarts)

    def test_constant_scheduler_is_none(self):
        """Trainer with default lr_schedule='constant' should have no scheduler."""
        from hexo_a0.trainer import Trainer

        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        assert trainer.scheduler is None


# ---------------------------------------------------------------------------
# Multi-threaded self-play (lines 164-182)
# ---------------------------------------------------------------------------

class TestMultiThreadedSelfPlay:
    def test_multi_worker_self_play(self):
        """train with self_play_workers > 1 produces buffer entries."""
        from hexo_a0.trainer import Trainer

        from dataclasses import replace
        full = replace(TINY_FULL_CONFIG,
            self_play=SelfPlayConfig(workers=2, python=PythonSelfPlayConfig(games_per_write=4)),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        _fill_buffer(trainer)
        metrics = trainer.train()
        assert len(trainer.buffer) > 0, "Buffer should have examples after self-play"


# ---------------------------------------------------------------------------
# Evaluation phase (lines 267-312)
# ---------------------------------------------------------------------------

class TestEvaluationPhase:
    def test_eval_vs_random_runs(self):
        """train triggers eval when eval_interval divides iteration."""
        from hexo_a0.trainer import Trainer

        from dataclasses import replace
        full = replace(TINY_FULL_CONFIG,
            eval=EvalConfig(interval=1, games=2, sims=0, sealbot=SealbotConfig(games=0)),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        _fill_buffer(trainer)
        metrics = trainer.train()
        assert "eval_win_rate" in metrics, "eval_win_rate should be in metrics when eval runs"
        assert "eval_mean_game_length" in metrics
        assert 0.0 <= metrics["eval_win_rate"] <= 1.0

    def test_eval_vs_checkpoint(self, tmp_path):
        """train evaluates vs a previous checkpoint when ckpt_dir has checkpoints."""
        from hexo_a0.trainer import Trainer

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        from dataclasses import replace
        full = replace(TINY_FULL_CONFIG,
            eval=EvalConfig(interval=1, games=2, sims=0, sealbot=SealbotConfig(games=0)),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(ckpt_dir),
        )

        # Save two fake checkpoints so len(pts) > lag=1
        for i in range(2):
            ckpt_path = ckpt_dir / f"checkpoint_{i:04d}.pt"
            trainer.save_checkpoint(str(ckpt_path))

        _fill_buffer(trainer)
        metrics = trainer.train()
        assert "eval_win_rate_vs_prev" in metrics, (
            "eval_win_rate_vs_prev should appear when checkpoint dir has enough checkpoints"
        )
        assert 0.0 <= metrics["eval_win_rate_vs_prev"] <= 1.0


# ---------------------------------------------------------------------------
# Trainer.close() (lines 322-323)
# ---------------------------------------------------------------------------

class TestTrainerClose:
    def test_close_without_writer(self):
        """close() on trainer with no writer should not raise."""
        from hexo_a0.trainer import Trainer

        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        assert trainer.writer is None
        trainer.close()  # Should not raise

    def test_close_with_writer(self, tmp_path):
        """close() should call writer.close() when a writer exists."""
        from hexo_a0.trainer import Trainer
        from unittest.mock import MagicMock

        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        mock_writer = MagicMock()
        trainer.writer = mock_writer
        trainer.close()
        mock_writer.close.assert_called_once()


# ---------------------------------------------------------------------------
# load_checkpoint with architecture mismatch (lines 367-378, 382)
# ---------------------------------------------------------------------------

class TestLoadCheckpointMismatch:
    def test_mismatched_architecture_resets_optimizer(self, tmp_path):
        """Loading a checkpoint into a model with different num_layers resets optimizer state."""
        from hexo_a0.trainer import Trainer

        from dataclasses import replace
        # Create and save a checkpoint with a 1-layer model
        original_mc = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1,
                                  policy_hidden=16, value_hidden=16,
                                  graph_type="hex", conv_type="gatv2")
        original_model = HeXONet(original_mc)
        original_full = replace(TINY_FULL_CONFIG, model=original_mc)
        original_trainer = Trainer(
            model=original_model,
            config=original_full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        _fill_buffer(original_trainer)
        original_trainer.train()  # Generate optimizer state
        ckpt_path = str(tmp_path / "ckpt.pt")
        original_trainer.save_checkpoint(ckpt_path)

        # Load into a model with MORE layers — same hidden_dim so shared params
        # match in shape, but extra layers create missing keys
        different_mc = ModelConfig(hidden_dim=16, num_layers=3, num_heads=1,
                                   policy_hidden=16, value_hidden=16,
                                   graph_type="hex", conv_type="gatv2")
        different_model = HeXONet(different_mc)
        different_full = replace(TINY_FULL_CONFIG, model=different_mc)
        new_trainer = Trainer(
            model=different_model,
            config=different_full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )

        # Should not raise despite architecture mismatch (strict=False)
        new_trainer.load_checkpoint(ckpt_path)

        # Iteration should still be restored
        assert new_trainer.train_steps == original_trainer.train_steps

        # The shared layer-0 conv params are present in both checkpoints,
        # so their Adam state should have been preserved by the name-based
        # recovery in load_checkpoint. The added layers' params are fresh.
        n_state_entries = len(new_trainer.optimizer.state)
        n_total_params = sum(len(g["params"]) for g in new_trainer.optimizer.param_groups)
        assert 0 < n_state_entries < n_total_params, (
            f"Expected partial state recovery (some preserved, some fresh) but got "
            f"{n_state_entries}/{n_total_params} entries"
        )

    def test_compiled_checkpoint_loads_into_uncompiled_model(self, tmp_path):
        """Checkpoint saved from compiled model loads correctly into uncompiled model."""
        from hexo_a0.trainer import Trainer

        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        _fill_buffer(trainer)
        trainer.train()

        # Simulate what torch.compile does: prefix all keys with _orig_mod.
        ckpt_path = str(tmp_path / "compiled_ckpt.pt")
        trainer.save_checkpoint(ckpt_path)

        # Verify saved checkpoint has clean keys (no _orig_mod. prefix)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        for key in raw["model_state_dict"]:
            assert not key.startswith("_orig_mod."), (
                f"Saved checkpoint should not have _orig_mod. prefix: {key}"
            )

        # Load into fresh uncompiled model — should work
        fresh_model = HeXONet(TINY_MODEL_CONFIG)
        fresh_trainer = Trainer(
            model=fresh_model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        fresh_trainer.load_checkpoint(ckpt_path)

        # Weights must match
        for name, param in trainer.model.named_parameters():
            loaded = dict(fresh_trainer.model.named_parameters())[name]
            assert torch.allclose(param, loaded), (
                f"Parameter '{name}' mismatch after compiled→uncompiled load"
            )

    def test_orig_mod_checkpoint_loads_correctly(self, tmp_path):
        """Old checkpoint with _orig_mod. prefixed keys loads into uncompiled model."""
        from hexo_a0.trainer import Trainer

        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        _fill_buffer(trainer)
        trainer.train()

        # Save a checkpoint, then manually add _orig_mod. prefixes to simulate
        # an old checkpoint saved from a compiled model (before the fix)
        ckpt_path = str(tmp_path / "old_compiled_ckpt.pt")
        trainer.save_checkpoint(ckpt_path)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        raw["model_state_dict"] = {
            f"_orig_mod.{k}": v for k, v in raw["model_state_dict"].items()
        }
        torch.save(raw, ckpt_path)

        # Load into fresh uncompiled model
        fresh_model = HeXONet(TINY_MODEL_CONFIG)
        fresh_trainer = Trainer(
            model=fresh_model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        fresh_trainer.load_checkpoint(ckpt_path)

        # Weights must match (optimizer should also be restored since arch matches)
        for name, param in trainer.model.named_parameters():
            loaded = dict(fresh_trainer.model.named_parameters())[name]
            assert torch.allclose(param, loaded), (
                f"Parameter '{name}' mismatch loading _orig_mod. checkpoint"
            )
        assert fresh_trainer.train_steps == trainer.train_steps

    def test_buffer_restored_from_companion_db(self, tmp_path):
        """load_checkpoint restores the replay buffer from the companion replay_buffer.db file."""
        from hexo_a0.trainer import Trainer

        ckpt_dir = tmp_path / "ckpt_dir"
        ckpt_dir.mkdir()
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(ckpt_dir),
        )
        _fill_buffer(trainer)
        buffer_size_before = len(trainer.buffer)
        assert buffer_size_before > 0

        # Save checkpoint to a subdirectory (like real training does)
        save_dir = ckpt_dir / "iter_001"
        save_dir.mkdir()
        ckpt_path = str(save_dir / "ckpt.pt")
        trainer.save_checkpoint(ckpt_path, save_buffer=True)

        # Verify companion DB file exists (distinct from the live buffer
        # path to avoid the live writer racing the backup connection).
        db_path = save_dir / "replay_buffer_backup.db"
        assert db_path.exists(), "Buffer companion DB should exist"

        # Load into fresh trainer with a separate ckpt_dir
        fresh_dir = tmp_path / "fresh"
        fresh_dir.mkdir()
        fresh_model = HeXONet(TINY_MODEL_CONFIG)
        fresh_trainer = Trainer(
            model=fresh_model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(fresh_dir),
        )
        assert len(fresh_trainer.buffer) == 0
        fresh_trainer.load_checkpoint(ckpt_path)
        assert len(fresh_trainer.buffer) == buffer_size_before, (
            f"Buffer should have {buffer_size_before} examples, got {len(fresh_trainer.buffer)}"
        )


# ---------------------------------------------------------------------------
# Backwards-compatible optimizer state recovery (1-group → 2-group transition,
# depth-extension graft, exact-match round-trip).
# ---------------------------------------------------------------------------


class TestOptimizerStateBackcompat:
    """Adam moments must survive: legacy resume, LayerScale graft, depth graft."""

    def _build_trainer(self, mc, tmp_path):
        """Create a trainer with the given ModelConfig + a few optimizer steps run."""
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        model = HeXONet(mc)
        full_cfg = replace(TINY_FULL_CONFIG, model=mc)
        trainer = Trainer(
            model=model,
            config=full_cfg,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        _fill_buffer(trainer)
        # Two train() chunks → 200 steps → optimizer state populated
        # (exp_avg / exp_avg_sq tensors created for every parameter).
        trainer.train()
        trainer.train()
        return trainer

    @staticmethod
    def _legacy_mc():
        return ModelConfig(
            hidden_dim=16, num_layers=1, num_heads=1,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
            use_layer_scale=False,
        )

    @staticmethod
    def _ls_same_depth_mc():
        # Same architecture as legacy + LayerScale enabled.
        return ModelConfig(
            hidden_dim=16, num_layers=1, num_heads=1,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
            use_layer_scale=True,
        )

    @staticmethod
    def _ls_extended_mc():
        # LayerScale on + extra layers grafted at the end.
        return ModelConfig(
            hidden_dim=16, num_layers=3, num_heads=1,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
            use_layer_scale=True,
        )

    def _make_legacy_single_group_checkpoint(self, ckpt_path, tmp_path):
        """Build a checkpoint that authentically mimics pre-LayerScale code:
        a single-group AdamW over ``model.parameters()`` and no
        ``param_names_in_order`` metadata. Returns the source trainer so
        callers can read its post-train Adam state for verification."""
        from dataclasses import asdict, replace
        from hexo_a0.trainer import Trainer

        mc = self._legacy_mc()
        model = HeXONet(mc)
        full_cfg = replace(TINY_FULL_CONFIG, model=mc)
        trainer = Trainer(
            model=model,
            config=full_cfg,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )

        # Swap the new 2-group optimizer for a legacy single-group AdamW
        # over model.parameters(). Same Parameter objects, so subsequent
        # training populates state we can verify.
        trainer.optimizer = torch.optim.AdamW(
            list(model.parameters()),
            lr=trainer.training_config.lr,
            weight_decay=trainer.training_config.weight_decay,
        )
        trainer._init_scheduler()

        _fill_buffer(trainer)
        trainer.train()
        trainer.train()

        # Save manually so we can omit param_names_in_order, exactly as
        # pre-fork code would have done.
        raw_sd = trainer.model.state_dict()
        model_sd = {k.removeprefix("_orig_mod."): v for k, v in raw_sd.items()}
        torch.save({
            "model_state_dict": model_sd,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict()
                if trainer.scheduler is not None else None,
            "train_steps": trainer.train_steps,
            "training_config": asdict(trainer.training_config),
            "model_config": asdict(trainer.model_config),
            "game_config": {
                "win_length": trainer.game_config.win_length,
                "placement_radius": trainer.game_config.placement_radius,
                "max_moves": trainer.game_config.max_moves,
            },
            "stage_start_step": trainer.stage_start_step,
            "sealbot_level_idx": trainer._sealbot_level_idx,
            "sealbot_consecutive_above": trainer._sealbot_consecutive_above,
            "current_step_delay": trainer.current_step_delay,
        }, ckpt_path)
        return trainer

    @staticmethod
    def _state_by_name(trainer):
        """Build {param_name: state_entry} for the live optimizer."""
        param_to_name = {id(p): n for n, p in trainer.model.named_parameters()}
        flat = [
            p for group in trainer.optimizer.param_groups
            for p in group["params"]
        ]
        out = {}
        for p in flat:
            entry = trainer.optimizer.state.get(p)
            if entry:
                out[param_to_name.get(id(p), "?")] = entry
        return out

    def test_case_a_legacy_single_group_to_same_arch(self, tmp_path):
        """Legacy 1-group ckpt → new 2-group code, same arch: ALL Adam state preserved."""
        ckpt_path = str(tmp_path / "legacy.pt")
        original = self._make_legacy_single_group_checkpoint(ckpt_path, tmp_path)
        original_state = self._state_by_name(original)
        assert original_state, "fixture: original optimizer should have state"

        from dataclasses import replace
        from hexo_a0.trainer import Trainer
        new_model = HeXONet(self._legacy_mc())
        new_trainer = Trainer(
            model=new_model,
            config=replace(TINY_FULL_CONFIG, model=self._legacy_mc()),
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        new_trainer.load_checkpoint(ckpt_path)
        new_state = self._state_by_name(new_trainer)

        # Every name from the original should be preserved.
        for name, orig_entry in original_state.items():
            assert name in new_state, f"missing recovered state for {name!r}"
            for k in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(new_state[name][k], orig_entry[k]), (
                    f"{name}/{k} not preserved bit-exactly"
                )

    def test_case_b_legacy_to_layerscale_same_depth(self, tmp_path):
        """Legacy ckpt → same-depth model with LayerScale: conv state preserved,
        layer_scale params start fresh."""
        ckpt_path = str(tmp_path / "legacy.pt")
        original = self._make_legacy_single_group_checkpoint(ckpt_path, tmp_path)
        original_state = self._state_by_name(original)

        from dataclasses import replace
        from hexo_a0.trainer import Trainer
        new_mc = self._ls_same_depth_mc()
        new_model = HeXONet(new_mc)
        new_trainer = Trainer(
            model=new_model,
            config=replace(TINY_FULL_CONFIG, model=new_mc),
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        new_trainer.load_checkpoint(ckpt_path)
        new_state = self._state_by_name(new_trainer)

        # Original conv/policy/value params must be preserved.
        for name, orig_entry in original_state.items():
            assert name in new_state, f"missing recovered state for {name!r}"
            assert torch.equal(
                new_state[name]["exp_avg"], orig_entry["exp_avg"]
            )

        # layer_scale params must be ABSENT from the optimizer state
        # (they are initialised lazily on first .step()).
        ls_names = [
            n for n, _ in new_trainer.model.named_parameters()
            if "layer_scales" in n
        ]
        assert ls_names, "fixture: LayerScale model should expose layer_scales params"
        for n in ls_names:
            assert n not in new_state, (
                f"layer_scale {n!r} should not have saved Adam state on first resume"
            )

    def test_case_c_legacy_to_layerscale_depth_extended(self, tmp_path):
        """Legacy 1-layer ckpt → 3-layer + LayerScale: layer-0 state preserved;
        layers 1/2 + all layer_scales start fresh."""
        ckpt_path = str(tmp_path / "legacy.pt")
        original = self._make_legacy_single_group_checkpoint(ckpt_path, tmp_path)
        original_state = self._state_by_name(original)

        from dataclasses import replace
        from hexo_a0.trainer import Trainer
        new_mc = self._ls_extended_mc()
        new_model = HeXONet(new_mc)
        new_trainer = Trainer(
            model=new_model,
            config=replace(TINY_FULL_CONFIG, model=new_mc),
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        new_trainer.load_checkpoint(ckpt_path)
        new_state = self._state_by_name(new_trainer)

        # Layer-0 conv + heads + input_proj: preserved.
        for name, orig_entry in original_state.items():
            assert name in new_state, f"missing recovered state for {name!r}"
            assert torch.equal(
                new_state[name]["exp_avg"], orig_entry["exp_avg"]
            ), f"{name}/exp_avg not preserved"

        # New-layer conv params (convs.1, convs.2): absent from saved state.
        new_layer_names = [
            n for n, _ in new_trainer.model.named_parameters()
            if (".convs.1." in n or ".convs.2." in n)
        ]
        assert new_layer_names, "fixture: 3-layer model should expose layers 1 and 2"
        for n in new_layer_names:
            assert n not in new_state, (
                f"new-layer param {n!r} should be fresh after graft"
            )

        # All layer_scales params: absent.
        for n, _ in new_trainer.model.named_parameters():
            if "layer_scales" in n:
                assert n not in new_state, (
                    f"layer_scale {n!r} should be fresh after graft"
                )

    def test_case_d_new_format_round_trip(self, tmp_path):
        """New-format (2-group, with param_names_in_order) ckpt → same arch:
        bit-exact Adam state round-trip."""
        ckpt_path = str(tmp_path / "newformat.pt")
        new_mc = self._ls_extended_mc()
        original = self._build_trainer(new_mc, tmp_path)
        original.save_checkpoint(ckpt_path)
        original_state = self._state_by_name(original)
        assert original_state, "fixture: optimizer state should be populated"

        # Confirm the saved checkpoint has the new metadata.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "param_names_in_order" in ckpt
        assert len(ckpt["param_names_in_order"]) > 0

        from dataclasses import replace
        from hexo_a0.trainer import Trainer
        fresh_model = HeXONet(new_mc)
        fresh_trainer = Trainer(
            model=fresh_model,
            config=replace(TINY_FULL_CONFIG, model=new_mc),
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        fresh_trainer.load_checkpoint(ckpt_path)
        fresh_state = self._state_by_name(fresh_trainer)

        # Every original param state must round-trip bit-exactly.
        assert set(fresh_state.keys()) == set(original_state.keys()), (
            "param name set must match after round-trip"
        )
        for name, orig_entry in original_state.items():
            for k in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(fresh_state[name][k], orig_entry[k]), (
                    f"{name}/{k} not bit-exact after round-trip"
                )


class TestArchChangeWarmup:
    """Resume after architecture change must reset the LR scheduler so
    warmup applies; resume *without* arch change must fast-forward as before.
    """

    @staticmethod
    def _legacy_mc():
        return ModelConfig(
            hidden_dim=16, num_layers=1, num_heads=1,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
            use_layer_scale=False,
        )

    @staticmethod
    def _ls_extended_mc():
        return ModelConfig(
            hidden_dim=16, num_layers=3, num_heads=1,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
            use_layer_scale=True,
        )

    @staticmethod
    def _cosine_training_config():
        # Cosine + warmup so we can observe LR at scheduler-step-0 vs
        # post-fast-forward. Warmup window (50 steps) is well below the
        # 200 training steps the legacy fixture produces, so without arch
        # change fast-forward lands us comfortably past warmup.
        return TrainingConfig(
            batch_size=8, buffer_capacity=1000,
            edge_budget=0, grad_accumulation=False,
            lr=2e-4, weight_decay=1e-4, max_grad_norm=1.0,
            lr_schedule="cosine", lr_warmup_steps=50, total_train_steps=1000,
            lr_min=2e-5,
            step_delay=0.0, target_reuse=0.0, draw_value=0.0,
            augment_symmetries=False, reanalyze_fraction=0.0, pc_loss_weight=0.0,
        )

    def _make_legacy_cosine_checkpoint(self, ckpt_path):
        """Legacy single-group checkpoint with cosine+warmup, ~200 steps in."""
        from dataclasses import asdict, replace
        from hexo_a0.trainer import Trainer

        mc = self._legacy_mc()
        full_cfg = replace(
            TINY_FULL_CONFIG, model=mc, training=self._cosine_training_config(),
        )
        trainer = Trainer(
            model=HeXONet(mc), config=full_cfg, game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        # Replace the 2-group optimizer with a legacy single-group AdamW
        # to mimic pre-fork checkpoints exactly.
        trainer.optimizer = torch.optim.AdamW(
            list(trainer.model.parameters()),
            lr=trainer.training_config.lr,
            weight_decay=trainer.training_config.weight_decay,
        )
        trainer._init_scheduler()

        _fill_buffer(trainer)
        trainer.train()  # 100 steps
        trainer.train()  # 200 total

        raw_sd = trainer.model.state_dict()
        model_sd = {k.removeprefix("_orig_mod."): v for k, v in raw_sd.items()}
        torch.save({
            "model_state_dict": model_sd,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "train_steps": trainer.train_steps,
            "training_config": asdict(trainer.training_config),
            "model_config": asdict(trainer.model_config),
            "game_config": {
                "win_length": trainer.game_config.win_length,
                "placement_radius": trainer.game_config.placement_radius,
                "max_moves": trainer.game_config.max_moves,
            },
            "stage_start_step": trainer.stage_start_step,
            "sealbot_level_idx": trainer._sealbot_level_idx,
            "sealbot_consecutive_above": trainer._sealbot_consecutive_above,
            "current_step_delay": trainer.current_step_delay,
        }, ckpt_path)
        return trainer

    def test_resume_with_arch_change_applies_fresh_warmup(self, tmp_path, caplog):
        """1-layer legacy ckpt -> 3-layer + LayerScale: scheduler resets to
        step 0 so warmup applies; train_steps counter is preserved."""
        import logging
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        ckpt_path = str(tmp_path / "legacy.pt")
        legacy = self._make_legacy_cosine_checkpoint(ckpt_path)
        legacy_steps = legacy.train_steps
        assert legacy_steps >= 100, "fixture: must be well past warmup"

        new_mc = self._ls_extended_mc()
        full_cfg = replace(
            TINY_FULL_CONFIG, model=new_mc, training=self._cosine_training_config(),
        )
        new_trainer = Trainer(
            model=HeXONet(new_mc), config=full_cfg,
            game_config=TINY_GAME_CONFIG, device=torch.device("cpu"),
        )

        with caplog.at_level(logging.INFO, logger="hexo_a0.trainer"):
            new_trainer.load_checkpoint(ckpt_path)

        # train_steps counter preserved.
        assert new_trainer.train_steps == legacy_steps, (
            f"train_steps should be preserved on resume: "
            f"got {new_trainer.train_steps}, expected {legacy_steps}"
        )

        # Scheduler is at step 0 of warmup → LR is in the
        # ramp-up region (LinearLR start_factor=1e-3 → ~lr*1e-3 ≈ 2e-7).
        # Definitely << peak lr=2e-4. Use 1e-5 as a safe ceiling.
        current_lr = new_trainer.optimizer.param_groups[0]["lr"]
        assert current_lr < 1e-5, (
            f"warmup should be active (lr near 0); got lr={current_lr:.3e}"
        )

        # Confirm the explanatory log line was emitted.
        msgs = [r.getMessage() for r in caplog.records]
        assert any("Fresh params detected" in m for m in msgs), (
            f"expected 'Fresh params detected' log; got: {msgs}"
        )
        assert not any("Fast-forwarded scheduler" in m for m in msgs), (
            f"fast-forward should be skipped on arch change; got: {msgs}"
        )

    def test_resume_no_arch_change_preserves_fast_forward(self, tmp_path, caplog):
        """Same-arch resume (production scenario): scheduler fast-forwards
        past warmup; the 'Fresh params detected' log must NOT appear."""
        import logging
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        ckpt_path = str(tmp_path / "legacy.pt")
        legacy = self._make_legacy_cosine_checkpoint(ckpt_path)
        legacy_steps = legacy.train_steps

        # Resume into the SAME arch (1-layer, no LayerScale): no fresh params.
        same_mc = self._legacy_mc()
        full_cfg = replace(
            TINY_FULL_CONFIG, model=same_mc, training=self._cosine_training_config(),
        )
        new_trainer = Trainer(
            model=HeXONet(same_mc), config=full_cfg,
            game_config=TINY_GAME_CONFIG, device=torch.device("cpu"),
        )

        with caplog.at_level(logging.INFO, logger="hexo_a0.trainer"):
            new_trainer.load_checkpoint(ckpt_path)

        # train_steps preserved either way.
        assert new_trainer.train_steps == legacy_steps

        # LR is past warmup (legacy_steps >= 100, warmup window = 50). It
        # may sit at peak lr=2e-4 or further along the cosine curve, but
        # MUST be substantially above the warmup-step-0 value.
        current_lr = new_trainer.optimizer.param_groups[0]["lr"]
        assert current_lr > 1e-4, (
            f"fast-forward should have advanced past warmup; got lr={current_lr:.3e}"
        )

        # The arch-change branch must not have triggered.
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("Fresh params detected" in m for m in msgs), (
            f"no arch change → no 'Fresh params detected' log; got: {msgs}"
        )

    def test_warmup_on_arch_change_disabled(self, tmp_path, caplog):
        """Opt-out: warmup_on_arch_change=False -> arch change still
        fast-forwards (legacy behavior)."""
        import logging
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        ckpt_path = str(tmp_path / "legacy.pt")
        legacy = self._make_legacy_cosine_checkpoint(ckpt_path)

        new_mc = self._ls_extended_mc()
        tc = self._cosine_training_config()
        # Override the new flag.
        tc = replace(tc, warmup_on_arch_change=False)
        full_cfg = replace(TINY_FULL_CONFIG, model=new_mc, training=tc)
        new_trainer = Trainer(
            model=HeXONet(new_mc), config=full_cfg,
            game_config=TINY_GAME_CONFIG, device=torch.device("cpu"),
        )

        with caplog.at_level(logging.INFO, logger="hexo_a0.trainer"):
            new_trainer.load_checkpoint(ckpt_path)

        current_lr = new_trainer.optimizer.param_groups[0]["lr"]
        assert current_lr > 1e-4, (
            f"opt-out: should fast-forward through warmup; got lr={current_lr:.3e}"
        )
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("Fresh params detected" in m for m in msgs)


# ---------------------------------------------------------------------------
# LayerScale weight decay (opt-in 3rd param group)
# ---------------------------------------------------------------------------

class TestLayerScaleWeightDecay:
    """Verify the opt-in LayerScale weight-decay param group + cross-config
    optimizer-state recovery via the name-routed loader."""

    @staticmethod
    def _ls_mc():
        return ModelConfig(
            hidden_dim=16, num_layers=2, num_heads=1,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
            use_layer_scale=True,
        )

    def test_layer_scale_decay_creates_third_param_group(self):
        """When layer_scale_weight_decay > 0, layer_scales get their own
        param group with that exact coefficient — not the main weight_decay,
        not 0.0."""
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        mc = self._ls_mc()
        full = replace(
            TINY_FULL_CONFIG,
            model=mc,
            training=replace(
                TINY_FULL_CONFIG.training,
                weight_decay=1e-3,
                layer_scale_weight_decay=1e-4,
            ),
        )
        trainer = Trainer(
            model=HeXONet(mc),
            config=full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        groups = trainer.optimizer.param_groups
        assert len(groups) == 3, f"expected 3 param groups, got {len(groups)}"

        # Identify the layer_scale group: contains exactly the layer_scales params.
        ls_param_ids = {
            id(p) for n, p in trainer.model.named_parameters()
            if "layer_scales" in n
        }
        assert ls_param_ids, "fixture: model should expose layer_scales params"

        ls_group = None
        for g in groups:
            ids = {id(p) for p in g["params"]}
            if ids == ls_param_ids:
                ls_group = g
                break
        assert ls_group is not None, "no param group exclusively contains layer_scales"
        assert ls_group["weight_decay"] == 1e-4, (
            f"layer_scale group weight_decay {ls_group['weight_decay']!r} != 1e-4"
        )

        # The other two groups must NOT carry the layer_scale coefficient.
        for g in groups:
            if g is ls_group:
                continue
            assert g["weight_decay"] in (0.0, 1e-3), (
                f"non-layerscale group has unexpected weight_decay {g['weight_decay']!r}"
            )

    def test_layer_scale_decay_default_keeps_two_groups(self):
        """Default (layer_scale_weight_decay=0.0) preserves prior 2-group
        structure: layer_scales live in the no-decay group exactly as before."""
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        mc = self._ls_mc()
        full = replace(TINY_FULL_CONFIG, model=mc)  # default layer_scale_weight_decay=0
        trainer = Trainer(
            model=HeXONet(mc),
            config=full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        groups = trainer.optimizer.param_groups
        assert len(groups) == 2, f"expected 2 param groups (default), got {len(groups)}"

    def test_three_group_to_two_group_resume_preserves_state(self, tmp_path):
        """Save with layer_scale_weight_decay=1e-4 (3 groups, populated state),
        resume with =0.0 (2 groups). Name-routed loader must preserve all
        Adam state including layer_scales."""
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        mc = self._ls_mc()
        # --- Save side: 3 groups ---
        full_save = replace(
            TINY_FULL_CONFIG,
            model=mc,
            training=replace(
                TINY_FULL_CONFIG.training, layer_scale_weight_decay=1e-4
            ),
        )
        save_trainer = Trainer(
            model=HeXONet(mc),
            config=full_save,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        assert len(save_trainer.optimizer.param_groups) == 3
        _fill_buffer(save_trainer)
        save_trainer.train()
        save_trainer.train()

        ckpt_path = str(tmp_path / "three_group.pt")
        save_trainer.save_checkpoint(ckpt_path)

        param_to_name = {
            id(p): n for n, p in save_trainer.model.named_parameters()
        }
        saved_state = {}
        for g in save_trainer.optimizer.param_groups:
            for p in g["params"]:
                e = save_trainer.optimizer.state.get(p)
                if e:
                    saved_state[param_to_name[id(p)]] = e
        assert any("layer_scales" in n for n in saved_state), (
            "fixture: layer_scales must have populated Adam state pre-save"
        )

        # --- Load side: 2 groups (default) ---
        full_load = replace(TINY_FULL_CONFIG, model=mc)
        load_trainer = Trainer(
            model=HeXONet(mc),
            config=full_load,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        assert len(load_trainer.optimizer.param_groups) == 2
        load_trainer.load_checkpoint(ckpt_path)

        load_p2n = {
            id(p): n for n, p in load_trainer.model.named_parameters()
        }
        loaded_state = {}
        for g in load_trainer.optimizer.param_groups:
            for p in g["params"]:
                e = load_trainer.optimizer.state.get(p)
                if e:
                    loaded_state[load_p2n[id(p)]] = e

        # Every saved param's Adam state must round-trip bit-exactly across
        # the 3-group → 2-group transition.
        for name, orig in saved_state.items():
            assert name in loaded_state, (
                f"name-routed loader dropped {name!r} on 3->2 group transition"
            )
            for k in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(loaded_state[name][k], orig[k]), (
                    f"{name}/{k} not preserved across 3->2 group transition"
                )


# ---------------------------------------------------------------------------
# LR warmup (Task 5: linear warmup support)
# ---------------------------------------------------------------------------

class TestLRWarmup:
    def test_warmup_scheduler_created(self):
        """Trainer with lr_warmup_steps > 0 creates a scheduler."""
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        full = replace(TINY_FULL_CONFIG,
            training=TrainingConfig(batch_size=4, buffer_capacity=100, lr_warmup_steps=10),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model, config=full, game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        assert trainer.scheduler is not None

    def test_warmup_lr_starts_near_zero(self):
        """During warmup, LR should start near zero."""
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        full = replace(TINY_FULL_CONFIG,
            training=TrainingConfig(batch_size=4, buffer_capacity=100, lr=0.001, lr_warmup_steps=100),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model, config=full, game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        assert initial_lr < 0.0002, f"Initial LR {initial_lr} should be near zero during warmup"

    def test_warmup_plus_cosine(self):
        """Warmup + cosine creates a SequentialLR."""
        from dataclasses import replace
        from torch.optim.lr_scheduler import SequentialLR
        from hexo_a0.trainer import Trainer

        full = replace(TINY_FULL_CONFIG,
            training=TrainingConfig(
                batch_size=4, buffer_capacity=100,
                lr=0.001, lr_warmup_steps=10,
                lr_schedule="cosine", total_train_steps=100, lr_min=1e-6,
            ),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model, config=full, game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        assert isinstance(trainer.scheduler, SequentialLR)


# ---------------------------------------------------------------------------
# Augmented training integration (Task 7)
# ---------------------------------------------------------------------------

class TestAugmentedTraining:
    def test_training_with_augmentation_runs(self):
        """Full train with augment_symmetries=True completes."""
        from hexo_a0.trainer import Trainer

        from dataclasses import replace
        full = replace(TINY_FULL_CONFIG,
            training=TrainingConfig(batch_size=8, buffer_capacity=10000, augment_symmetries=True),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model, config=full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
        )
        _fill_buffer(trainer)
        metrics = trainer.train()
        assert metrics["total_loss"] > 0


# ---------------------------------------------------------------------------
# REQ-27: Legacy checkpoint migration (_buffer.pt → SQLite)
# ---------------------------------------------------------------------------

class TestLegacyBufferMigration:
    """AC-27: Trainer migrates legacy _buffer.pt into SQLite on load_checkpoint."""

    def _make_legacy_buffer_pt(self, path: str, n: int) -> list:
        """Create a legacy companion _buffer.pt file with n synthetic examples."""
        from hexo_a0.self_play import TrainingExample
        from hexo_a0.graph import game_to_graph

        game = hexo_rs.GameState(TINY_GAME_CONFIG)
        data = game_to_graph(game)
        n_legal = int(data.legal_mask.sum().item())
        examples = [
            TrainingExample(
                data=data,
                policy_target=torch.ones(n_legal) / n_legal,
                value_target=0.0,
            )
            for _ in range(n)
        ]
        torch.save(examples, path)
        return examples

    def _save_legacy_checkpoint(self, tmp_path, n_legacy: int) -> tuple:
        """Save a checkpoint + legacy _buffer.pt in a dedicated directory.

        Returns (ckpt_path, n_legacy) where ckpt_path has no replay_buffer.db
        sibling (simulating a pre-SQLite checkpoint).
        """
        from hexo_a0.trainer import Trainer

        # Trainer uses in-memory buffer (no ckpt_dir) so no DB file is created
        # on disk next to the checkpoint we'll save.
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            # No ckpt_dir → in-memory buffer, no replay_buffer.db on disk
        )
        _fill_buffer(trainer)
        trainer.train()

        # Save checkpoint to its own subdirectory
        save_dir = tmp_path / "legacy_ckpt"
        save_dir.mkdir()
        ckpt_path = str(save_dir / "checkpoint_00000001.pt")
        trainer.save_checkpoint(ckpt_path, save_buffer=False)

        # Verify no replay_buffer.db next to the checkpoint
        assert not (save_dir / "replay_buffer.db").exists()

        # Create the legacy _buffer.pt companion
        legacy_path = ckpt_path.replace(".pt", "_buffer.pt")
        self._make_legacy_buffer_pt(legacy_path, n_legacy)

        return ckpt_path, n_legacy

    def test_legacy_buffer_migrated_on_load(self, tmp_path):
        """AC-27a: load_checkpoint migrates _buffer.pt into buffer (correct count)."""
        from hexo_a0.trainer import Trainer

        n_legacy = 7
        ckpt_path, _ = self._save_legacy_checkpoint(tmp_path, n_legacy)

        # Load into a fresh trainer — must migrate legacy buffer
        fresh_dir = tmp_path / "fresh"
        fresh_dir.mkdir()
        fresh_model = HeXONet(TINY_MODEL_CONFIG)
        fresh_trainer = Trainer(
            model=fresh_model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(fresh_dir),
        )
        assert len(fresh_trainer.buffer) == 0
        fresh_trainer.load_checkpoint(ckpt_path)

        # AC-27a: buffer count matches the legacy file
        assert len(fresh_trainer.buffer) == n_legacy, (
            f"Expected {n_legacy} migrated examples, got {len(fresh_trainer.buffer)}"
        )

    def test_training_continues_after_migration(self, tmp_path):
        """AC-27b: training runs normally after legacy buffer migration."""
        from hexo_a0.trainer import Trainer

        ckpt_path, _ = self._save_legacy_checkpoint(tmp_path, 10)

        fresh_dir = tmp_path / "fresh"
        fresh_dir.mkdir()
        fresh_model = HeXONet(TINY_MODEL_CONFIG)
        fresh_trainer = Trainer(
            model=fresh_model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(fresh_dir),
        )
        fresh_trainer.load_checkpoint(ckpt_path)

        # AC-27b: training continues without error
        metrics = fresh_trainer.train()
        assert "total_loss" in metrics
        assert metrics["total_loss"] >= 0.0

    def test_next_save_writes_db_not_legacy(self, tmp_path):
        """AC-27c: checkpoint saved after migration writes replay_buffer.db, not _buffer.pt."""
        from hexo_a0.trainer import Trainer

        ckpt_path, _ = self._save_legacy_checkpoint(tmp_path, 5)

        fresh_dir = tmp_path / "fresh"
        fresh_dir.mkdir()
        fresh_model = HeXONet(TINY_MODEL_CONFIG)
        fresh_trainer = Trainer(
            model=fresh_model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(fresh_dir),
        )
        fresh_trainer.load_checkpoint(ckpt_path)

        # Save new checkpoint
        new_ckpt = str(fresh_dir / "checkpoint_00000002.pt")
        fresh_trainer.save_checkpoint(new_ckpt, save_buffer=True)

        # AC-27c: replay_buffer.db exists in the new checkpoint's directory
        new_db = fresh_dir / "replay_buffer.db"
        assert new_db.exists(), "New checkpoint should write replay_buffer.db"

        # AC-27c: no _buffer.pt in the new checkpoint dir
        legacy_in_fresh = fresh_dir / "checkpoint_00000002_buffer.pt"
        assert not legacy_in_fresh.exists(), (
            "New checkpoint should NOT write legacy _buffer.pt"
        )


# ---------------------------------------------------------------------------
# HX03 binary record parsing (winner byte + move_count)
# ---------------------------------------------------------------------------


def _make_hx03_record(winner_byte: int, move_count: int = 1) -> bytes:
    """Build a minimal HX03 record with one example (0 nodes / 0 edges)."""
    import struct

    # Example payload: header u16 num_nodes, u16 num_edges, u16 num_legal,
    # u8 has_edge_attr, f32 value. Then zero bytes for arrays since all
    # counts are zero.
    example = struct.pack("<HHHBf", 0, 0, 0, 0, 0.5)
    body = (
        struct.pack("<I", 1)                 # num_examples
        + struct.pack("<b", winner_byte)     # winner i8
        + struct.pack("<I", move_count)      # move_count u32
        + example
    )
    record = b"HX03" + struct.pack("<I", len(body)) + body
    return record


@pytest.mark.parametrize("winner_byte,expected", [(1, "P1"), (-1, "P2"), (0, "draw")])
def test_hx03_binary_parser_surfaces_winner(winner_byte, expected, tmp_path):
    """The binary reader should surface a 'winner' field on each game dict."""
    from hexo_a0.trainer import Trainer

    blob = _make_hx03_record(winner_byte)
    p = tmp_path / "games_0001.bin"
    p.write_bytes(blob)

    fh = open(p, "rb")
    batch: list = []
    Trainer._read_binary_records(fh, batch, max_records=10)
    fh.close()

    assert len(batch) == 1
    assert batch[0]["winner"] == expected
    assert batch[0]["length"] == 1
    assert len(batch[0]["examples"]) == 1


def test_hx03_binary_parser_surfaces_move_count(tmp_path):
    """``move_count`` should round-trip via the binary parser, distinct from
    ``length`` (= num_examples) when playout cap is active."""
    from hexo_a0.trainer import Trainer

    # Simulate playout cap: 10 moves played but only 1 full-search example
    # recorded.
    blob = _make_hx03_record(winner_byte=1, move_count=10)
    p = tmp_path / "games_0001.bin"
    p.write_bytes(blob)

    fh = open(p, "rb")
    batch: list = []
    Trainer._read_binary_records(fh, batch, max_records=10)
    fh.close()

    assert len(batch) == 1
    assert batch[0]["length"] == 1, "num_examples"
    assert batch[0]["move_count"] == 10, "actual move count from header"


def test_collect_self_play_stats_uses_move_count_for_mean_length(tmp_path):
    """``mean_game_length`` must be computed from per-game move counts, not
    from per-game example counts (which differ when playout cap is active)."""
    import threading
    from hexo_a0.trainer import Trainer

    stub = Trainer.__new__(Trainer)
    stub._sp_lock = threading.Lock()
    stub._sp_games = 3
    stub._sp_game_examples = [10, 12, 15]    # 37 examples written
    stub._sp_game_move_counts = [40, 50, 60] # mean game length = 50.0
    stub._sp_p1_wins = 2
    stub._sp_p2_wins = 1
    stub._sp_policy_entropies = [0.1, 0.2]
    stub._sp_policy_top1 = [0.9, 0.8]
    stub._sp_value_targets = [1.0, -1.0]
    ws = {"in_window_moves": 7, "gated_moves": 3,
          "mass_removed_sum": 0.5, "acted_deficit_sum": 0.25}
    stub._sp_window_stats = [ws]

    report = Trainer._collect_self_play_stats(stub)
    assert report.games == 3
    assert report.mean_game_length == 50.0
    assert report.p1_wins == 2
    assert report.p2_wins == 1
    assert report.examples_written == 37
    assert report.game_lengths == [40, 50, 60]
    assert report.policy_entropies == [0.1, 0.2]
    assert report.policy_top1 == [0.9, 0.8]
    assert report.value_targets == [1.0, -1.0]
    assert report.window_stats == [ws]
    # Counters reset
    assert stub._sp_games == 0
    assert stub._sp_p1_wins == 0
    assert stub._sp_p2_wins == 0
    assert stub._sp_game_examples == []
    assert stub._sp_game_move_counts == []
    assert stub._sp_policy_entropies == []
    assert stub._sp_policy_top1 == []
    assert stub._sp_value_targets == []
    assert stub._sp_window_stats == []


# ---------------------------------------------------------------------------
# Past-self opponent pool: snapshot rotation
# ---------------------------------------------------------------------------


def test_load_checkpoint_keeps_fresh_live_buffer(tmp_path):
    """A stale ``replay_buffer_backup.db`` must not overwrite a fresher live DB.

    Regression: mid-stage Ctrl+C saves the checkpoint with ``save_buffer=False``
    so the backup stays frozen at the previous stage-transition. On resume,
    ``load_checkpoint`` used to unconditionally prefer the backup and clobber
    the live DB via ``set_state``, erasing any ingestion done since the last
    transition.
    """
    from hexo_a0.trainer import Trainer

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    device = torch.device("cpu")

    # Run 1: fill buffer with 5 examples, write stage-transition backup.
    t1 = Trainer(
        model=HeXONet(TINY_MODEL_CONFIG),
        config=TINY_FULL_CONFIG,
        game_config=TINY_GAME_CONFIG,
        device=device,
        ckpt_dir=str(ckpt_dir),
    )
    _fill_buffer(t1, n=5)
    ckpt_path = ckpt_dir / "checkpoint_00000000.pt"
    t1.save_checkpoint(str(ckpt_path), save_buffer=True)
    assert (ckpt_dir / "replay_buffer_backup.db").exists()

    # Grow the live DB past the backup, then interrupt-save (save_buffer=False).
    _fill_buffer(t1, n=7)
    assert len(t1.buffer) == 12
    t1.save_checkpoint(str(ckpt_path), save_buffer=False)
    t1.close()

    # Run 2: new Trainer against the same ckpt_dir.
    t2 = Trainer(
        model=HeXONet(TINY_MODEL_CONFIG),
        config=TINY_FULL_CONFIG,
        game_config=TINY_GAME_CONFIG,
        device=device,
        ckpt_dir=str(ckpt_dir),
    )
    try:
        assert len(t2.buffer) == 12, "live DB should open with the 12 prior rows"
        t2.load_checkpoint(str(ckpt_path))
        assert len(t2.buffer) == 12, (
            f"load_checkpoint clobbered the live DB with the stale backup: "
            f"got {len(t2.buffer)}, expected 12"
        )
    finally:
        t2.close()


class TestPoolRotation:
    def test_rotate_keeps_max_size_default(self, tmp_path):
        from hexo_a0.trainer import _rotate_pool_snapshots

        pool = tmp_path / "pool"
        # Write 12 source snapshots in order; each call rotates one in.
        for i in range(12):
            src = tmp_path / f"src_{i}.pt"
            src.write_bytes(f"snapshot-{i}".encode())
            # Force monotonic mtimes so prune order is well-defined
            import os
            t = 1_000_000 + i
            os.utime(src, (t, t))
            _rotate_pool_snapshots(pool, src, max_size=8)

        remaining = sorted(pool.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        assert len(remaining) == 8
        # Oldest 4 (src_0..src_3) must be gone, newest 8 (src_4..src_11) kept
        contents = {p.read_bytes() for p in remaining}
        for i in range(4, 12):
            assert f"snapshot-{i}".encode() in contents
        for i in range(0, 4):
            assert f"snapshot-{i}".encode() not in contents

    def test_rotate_custom_k(self, tmp_path):
        from hexo_a0.trainer import _rotate_pool_snapshots
        import os

        pool = tmp_path / "pool"
        for i in range(6):
            src = tmp_path / f"s{i}.pt"
            src.write_bytes(b"x" * (i + 1))
            t = 2_000_000 + i
            os.utime(src, (t, t))
            _rotate_pool_snapshots(pool, src, max_size=3)
        assert len(list(pool.glob("*.pt"))) == 3

    def test_rotate_disabled_when_max_zero(self, tmp_path):
        from hexo_a0.trainer import _rotate_pool_snapshots

        pool = tmp_path / "pool"
        src = tmp_path / "src.pt"
        src.write_bytes(b"data")
        result = _rotate_pool_snapshots(pool, src, max_size=0)
        assert result is None
        assert not pool.exists() or not list(pool.glob("*.pt"))

    def test_rotate_missing_source(self, tmp_path):
        from hexo_a0.trainer import _rotate_pool_snapshots

        pool = tmp_path / "pool"
        result = _rotate_pool_snapshots(pool, tmp_path / "nope.pt", max_size=8)
        assert result is None


# ---------------------------------------------------------------------------
# Past-self opponent pool: config plumbing
# ---------------------------------------------------------------------------

class TestPoolConfigPlumbing:
    def test_defaults(self):
        from hexo_a0.config import RustSelfPlayConfig

        cfg = RustSelfPlayConfig()
        assert cfg.pool_fraction == 0.3
        assert cfg.pool_size == 8

    def test_toml_round_trip(self, tmp_path):
        from hexo_a0.config import FullConfig
        from hexo_a0.config_io import load_config, save_config

        cfg = FullConfig()
        cfg.self_play.rust.pool_fraction = 0.25
        cfg.self_play.rust.pool_size = 5
        path = tmp_path / "cfg.toml"
        save_config(path, cfg)
        loaded = load_config(path)
        assert loaded.self_play.rust.pool_fraction == 0.25
        assert loaded.self_play.rust.pool_size == 5

    def test_snapshot_every_default(self):
        from hexo_a0.config import RustSelfPlayConfig

        assert RustSelfPlayConfig().pool_snapshot_every == 1


class TestPoolSnapshotEvery:
    """`pool_snapshot_every` controls how often _snapshot_to_pool actually copies."""

    def _make_trainer_stub(self, tmp_path, every: int):
        from hexo_a0.config import RustSelfPlayConfig
        from hexo_a0.trainer import Trainer

        stub = Trainer.__new__(Trainer)
        stub._ckpt_dir = str(tmp_path)
        stub._pool_snapshot_counter = 0

        class _SP:
            pass

        sp = _SP()
        sp.rust = RustSelfPlayConfig(pool_fraction=0.5, pool_size=8, pool_snapshot_every=every)
        stub.self_play_config = sp
        return stub

    def _bump_mtime(self, src, i):
        import os
        # rotation names destination files by source mtime; vary it per call
        # so successive snapshots don't collide on the same filename.
        os.utime(src, (1_000_000 + i, 1_000_000 + i))

    def test_every_one_snapshots_each_call(self, tmp_path):
        from hexo_a0.trainer import Trainer

        src = tmp_path / "model.pt"
        src.write_bytes(b"x")
        pool_dir = tmp_path / "self_play" / "pool"

        stub = self._make_trainer_stub(tmp_path, every=1)
        for i in range(3):
            self._bump_mtime(src, i)
            Trainer._snapshot_to_pool(stub, str(src))

        assert pool_dir.exists()
        assert len(list(pool_dir.glob("*.pt"))) == 3

    def test_every_n_skips_intermediate_calls(self, tmp_path):
        from hexo_a0.trainer import Trainer

        src = tmp_path / "model.pt"
        src.write_bytes(b"x")
        pool_dir = tmp_path / "self_play" / "pool"

        stub = self._make_trainer_stub(tmp_path, every=5)
        for i in range(12):
            self._bump_mtime(src, i)
            Trainer._snapshot_to_pool(stub, str(src))

        # calls 5 and 10 fire; 12 calls -> 2 snapshots
        assert len(list(pool_dir.glob("*.pt"))) == 2


# ---------------------------------------------------------------------------
# Past-self opponent pool: subprocess args
# ---------------------------------------------------------------------------

class TestPoolSubprocessArgs:
    def _build_cmd(self, pool_fraction: float) -> list[str]:
        """Reproduce the trainer's cmd-construction logic in isolation.

        We do not invoke the real Trainer here because instantiating it
        requires a model + optimizer; instead we mirror the additive
        block we added so a regression in flag-passing is caught.
        """
        from hexo_a0.config import RustSelfPlayConfig
        rust_cfg = RustSelfPlayConfig(pool_fraction=pool_fraction, pool_size=4)
        pool_dir = "/tmp/run/self_play/pool"
        cmd: list[str] = ["self_play", "--model", "/tmp/m.pt"]
        cmd.extend(["--max-batch", str(rust_cfg.max_batch)])
        if rust_cfg.pool_fraction > 0.0:
            cmd.extend(["--pool-dir", pool_dir])
            cmd.extend(["--pool-fraction", str(rust_cfg.pool_fraction)])
        return cmd

    def test_pool_flags_present_when_enabled(self):
        cmd = self._build_cmd(pool_fraction=0.3)
        assert "--pool-dir" in cmd
        assert "--pool-fraction" in cmd
        idx = cmd.index("--pool-fraction")
        assert cmd[idx + 1] == "0.3"

    def test_pool_flags_absent_when_disabled(self):
        cmd = self._build_cmd(pool_fraction=0.0)
        assert "--pool-dir" not in cmd
        assert "--pool-fraction" not in cmd

    def test_playout_cap_defaults(self):
        from hexo_a0.config import RustSelfPlayConfig

        cfg = RustSelfPlayConfig()
        assert cfg.playout_cap_fraction == 0.75
        assert cfg.playout_cap_divisor == 4

    def test_playout_cap_toml_round_trip(self, tmp_path):
        from hexo_a0.config import FullConfig
        from hexo_a0.config_io import load_config, save_config

        cfg = FullConfig()
        cfg.self_play.rust.playout_cap_fraction = 0.75
        cfg.self_play.rust.playout_cap_divisor = 4
        path = tmp_path / "cfg.toml"
        save_config(path, cfg)
        loaded = load_config(path)
        assert loaded.self_play.rust.playout_cap_fraction == 0.75
        assert loaded.self_play.rust.playout_cap_divisor == 4

    def _build_pcap_cmd(self, fraction: float, divisor: int) -> list[str]:
        from hexo_a0.config import RustSelfPlayConfig
        rust_cfg = RustSelfPlayConfig(
            playout_cap_fraction=fraction, playout_cap_divisor=divisor
        )
        cmd: list[str] = ["self_play", "--model", "/tmp/m.pt"]
        if rust_cfg.playout_cap_fraction > 0.0 and rust_cfg.playout_cap_divisor > 1:
            cmd.extend(["--playout-cap-fraction", str(rust_cfg.playout_cap_fraction)])
            cmd.extend(["--playout-cap-divisor", str(rust_cfg.playout_cap_divisor)])
        return cmd

    def test_playout_cap_subprocess_args_emitted(self):
        cmd = self._build_pcap_cmd(fraction=0.75, divisor=4)
        assert "--playout-cap-fraction" in cmd
        assert "--playout-cap-divisor" in cmd
        assert cmd[cmd.index("--playout-cap-fraction") + 1] == "0.75"
        assert cmd[cmd.index("--playout-cap-divisor") + 1] == "4"

    def test_playout_cap_subprocess_args_omitted_default(self):
        cmd = self._build_pcap_cmd(fraction=0.0, divisor=1)
        assert "--playout-cap-fraction" not in cmd
        assert "--playout-cap-divisor" not in cmd

    def test_playout_cap_subprocess_args_omitted_fraction_zero(self):
        cmd = self._build_pcap_cmd(fraction=0.0, divisor=4)
        assert "--playout-cap-fraction" not in cmd

    def test_playout_cap_subprocess_args_omitted_divisor_one(self):
        cmd = self._build_pcap_cmd(fraction=0.75, divisor=1)
        assert "--playout-cap-fraction" not in cmd

    def test_trainer_source_emits_playout_cap_flags(self):
        import inspect
        from hexo_a0 import trainer

        src = inspect.getsource(trainer.Trainer._rust_self_play_worker_inner)
        assert "rust_cfg.playout_cap_fraction > 0.0 and rust_cfg.playout_cap_divisor > 1" in src
        assert "--playout-cap-fraction" in src
        assert "--playout-cap-divisor" in src

    def test_resume_short_chunk_alignment_arithmetic(self):
        """Pure-logic check for resume realignment: burning a short first
        chunk restores train_steps to a chunk boundary.
        """
        chunk_size = 100
        cases = [
            (8098, 2),
            (8100, 0),
            (0, 0),
            (8199, 1),
            (8101, 99),
        ]
        for train_steps, expected_short in cases:
            rem = train_steps % chunk_size
            short = (chunk_size - rem) % chunk_size
            assert short == expected_short, (
                f"train_steps={train_steps}: {short} != {expected_short}"
            )
            assert (train_steps + short) % chunk_size == 0

    def test_trainer_source_has_resume_alignment_logic(self):
        """Trainer.load_checkpoint must compute _resume_short_chunk, and
        Trainer.train must consume it on the first post-resume iteration."""
        import inspect
        from hexo_a0 import trainer

        cls_src = inspect.getsource(trainer.Trainer)
        assert "_resume_short_chunk = 0" in cls_src
        lc_src = inspect.getsource(trainer.Trainer.load_checkpoint)
        assert "_resume_short_chunk" in lc_src
        tr_src = inspect.getsource(trainer.Trainer.train)
        assert "self._resume_short_chunk > 0" in tr_src

    def test_trainer_cmd_excludes_pool_when_disabled(self):
        """Smoke-test the actual trainer source: when pool_fraction == 0
        the literal flag strings must not appear before the conditional."""
        import inspect
        from hexo_a0 import trainer

        src = inspect.getsource(trainer.Trainer._rust_self_play_worker_inner)
        # The flags should only appear inside a guarded block.
        assert "rust_cfg.pool_fraction > 0.0" in src
        assert "--pool-dir" in src
        assert "--pool-fraction" in src


# ---------------------------------------------------------------------------
# Rust HX04 inference-server binary knob: [self_play.rust] inference_bin
# ---------------------------------------------------------------------------

class TestInferenceBinConfigPlumbing:
    def test_defaults(self):
        from hexo_a0.config import RustSelfPlayConfig

        cfg = RustSelfPlayConfig()
        assert cfg.inference_bin is None

    def test_toml_round_trip(self, tmp_path):
        from hexo_a0.config import FullConfig
        from hexo_a0.config_io import load_config, save_config

        cfg = FullConfig()
        cfg.self_play.rust.inference_bin = "/opt/bin/hx04_infer_server"
        path = tmp_path / "cfg.toml"
        save_config(path, cfg)
        loaded = load_config(path)
        assert loaded.self_play.rust.inference_bin == "/opt/bin/hx04_infer_server"


class TestInferenceBinSubprocessArgs:
    def _build_cmd(self, inference_bin: str | None) -> list[str]:
        """Reproduce the trainer's cmd-construction logic in isolation,
        mirroring TestPoolSubprocessArgs / max_batch_edges."""
        from hexo_a0.config import RustSelfPlayConfig
        rust_cfg = RustSelfPlayConfig(inference_bin=inference_bin)
        cmd: list[str] = ["self_play", "--model", "/tmp/m.pt"]
        cmd.extend(["--max-batch", str(rust_cfg.max_batch)])
        if getattr(rust_cfg, "inference_bin", None):
            cmd.extend(["--inference-bin", rust_cfg.inference_bin])
        return cmd

    def test_inference_bin_flag_present_when_set(self):
        cmd = self._build_cmd(inference_bin="/opt/bin/hx04_infer_server")
        assert "--inference-bin" in cmd
        idx = cmd.index("--inference-bin")
        assert cmd[idx + 1] == "/opt/bin/hx04_infer_server"

    def test_inference_bin_flag_absent_when_none(self):
        cmd = self._build_cmd(inference_bin=None)
        assert "--inference-bin" not in cmd

    def test_trainer_source_emits_inference_bin_flag(self):
        import inspect
        from hexo_a0 import trainer

        src = inspect.getsource(trainer.Trainer._rust_self_play_worker_inner)
        assert "rust_cfg.inference_bin" in src
        assert "--inference-bin" in src


# ===================================================================
# File watcher seek behavior (regression for new-file data loss bug)
# ===================================================================

class TestFileWatcherSeekBehavior:
    """Regression: new game files discovered after startup must be read from
    the beginning, not seeked to end.  Files that existed at startup should
    still be tailed from the end to avoid replaying stale data.

    These tests verify the seek-vs-start decision logic that lives in
    Trainer._rust_self_play_worker's ingest loop.
    """

    def test_initial_files_seeked_to_end(self, tmp_path):
        """Files present at startup should be seeked to end (tail mode)."""
        # Create a file before "startup"
        f = tmp_path / "games_0001.bin"
        f.write_bytes(b"x" * 1000)

        # Simulate startup: snapshot existing files
        initial_files = set(tmp_path.glob("*.bin"))
        assert f in initial_files

        # Open and apply the seek logic
        fh = open(f, "rb")
        if f in initial_files:
            fh.seek(0, 2)  # seek to end
        assert fh.tell() == 1000
        fh.close()

    def test_new_files_read_from_start(self, tmp_path):
        """Files discovered after startup should be read from position 0."""
        # Startup: no files exist
        initial_files = set(tmp_path.glob("*.bin"))
        assert len(initial_files) == 0

        # A new file appears (e.g. after rotation during a long eval)
        f = tmp_path / "games_0002.bin"
        f.write_bytes(b"x" * 5000)

        # Open and apply the seek logic
        fh = open(f, "rb")
        if f in initial_files:
            fh.seek(0, 2)  # would seek to end — but this shouldn't fire
        # New file: handle should be at position 0
        assert fh.tell() == 0
        assert len(fh.read()) == 5000
        fh.close()

    def test_source_uses_initial_files_guard(self):
        """Trainer._rust_self_play_worker must use initial_files to decide
        whether to seek to end, preventing the data-loss regression."""
        import inspect
        from hexo_a0 import trainer

        src = inspect.getsource(trainer.Trainer._rust_self_play_worker_inner)
        assert "initial_files" in src
        assert "p in initial_files" in src


# ---------------------------------------------------------------------------
# Champion mode (eval-vs-champion, export, model update suppression)
# ---------------------------------------------------------------------------

class TestChampionMode:
    def test_eval_vs_champion(self, tmp_path):
        """When _champion_mode is set and champion exists, eval produces
        eval_win_rate_vs_champion instead of eval_win_rate_vs_prev."""
        from dataclasses import replace
        from hexo_a0.trainer import Trainer

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        full = replace(TINY_FULL_CONFIG,
            eval=EvalConfig(interval=1, games=2, sims=0, sealbot=SealbotConfig(games=0)),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(ckpt_dir),
        )
        trainer._champion_mode = True
        trainer._export_champion()

        # Also save a regular checkpoint so we can confirm vs_prev is NOT produced
        trainer.save_checkpoint(str(ckpt_dir / "checkpoint_0001.pt"))
        trainer.save_checkpoint(str(ckpt_dir / "checkpoint_0002.pt"))

        _fill_buffer(trainer)
        metrics = trainer.train()

        assert "eval_win_rate_vs_champion" in metrics
        assert 0.0 <= metrics["eval_win_rate_vs_champion"] <= 1.0
        # vs_prev should NOT be produced in champion mode
        assert "eval_win_rate_vs_prev" not in metrics

    def test_export_champion_creates_files(self, tmp_path):
        """_export_champion creates champion.pt and model_selfplay.pt."""
        from hexo_a0.trainer import Trainer

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(ckpt_dir),
        )
        trainer._export_champion()

        sp_dir = ckpt_dir / "self_play"
        assert (sp_dir / "champion.pt").exists()
        assert (sp_dir / "model_selfplay.pt").exists()
        assert trainer._champion_path == str(sp_dir / "champion.pt")

    def test_champion_checkpoint_loadable(self, tmp_path):
        """Champion checkpoint can be loaded back and contains model_config."""
        from hexo_a0.trainer import Trainer

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=TINY_FULL_CONFIG,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(ckpt_dir),
        )
        trainer._export_champion()

        ckpt = torch.load(trainer._champion_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in ckpt
        assert "model_config" in ckpt
        assert ckpt["model_config"]["hidden_dim"] == 16

    def test_champion_mode_suppresses_model_update(self, tmp_path):
        """In champion mode, periodic self-play model re-exports are suppressed."""
        from dataclasses import replace
        from hexo_a0.trainer import Trainer
        from unittest.mock import patch

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        full = replace(TINY_FULL_CONFIG,
            eval=EvalConfig(interval=1, games=2, sims=0, checkpoint_interval=1, sealbot=SealbotConfig(games=0)),
            self_play=SelfPlayConfig(
                model_update_interval=1,
                python=PythonSelfPlayConfig(games_per_write=2),
            ),
        )
        model = HeXONet(TINY_MODEL_CONFIG)
        trainer = Trainer(
            model=model,
            config=full,
            game_config=TINY_GAME_CONFIG,
            device=torch.device("cpu"),
            ckpt_dir=str(ckpt_dir),
        )
        trainer._champion_mode = True
        # Simulate a running subprocess
        from unittest.mock import MagicMock
        trainer._sp_process = MagicMock()

        _fill_buffer(trainer)
        with patch.object(trainer, "_export_torchscript") as mock_export:
            trainer.train()
            mock_export.assert_not_called()


# ---------------------------------------------------------------------------
# _sum_window_stats — pure aggregation helper
# ---------------------------------------------------------------------------

class TestSumWindowStats:
    def test_sum_window_stats(self):
        from hexo_a0.trainer import _sum_window_stats
        stats = [
            {"in_window_moves": 7, "gated_moves": 3, "mass_removed_sum": 0.5, "acted_deficit_sum": 0.25},
            {"in_window_moves": 5, "gated_moves": 0, "mass_removed_sum": 0.0, "acted_deficit_sum": 0.125},
        ]
        assert _sum_window_stats(stats) == (12, 3, pytest.approx(0.5), pytest.approx(0.375))

    def test_sum_window_stats_empty(self):
        from hexo_a0.trainer import _sum_window_stats
        assert _sum_window_stats([]) == (0, 0, 0.0, 0.0)

    def test_sum_window_stats_single(self):
        from hexo_a0.trainer import _sum_window_stats
        stats = [{"in_window_moves": 10, "gated_moves": 4, "mass_removed_sum": 1.5, "acted_deficit_sum": 0.6}]
        assert _sum_window_stats(stats) == (10, 4, pytest.approx(1.5), pytest.approx(0.6))


class TestEdgeAblationMetrics:
    """edge_features/* introspection: value-head ablation alongside the policy
    ablation_kl, both measured on on-policy replay-buffer positions."""

    class _Rec:
        def __init__(self):
            self.tags = []

        def add_scalar(self, tag, *a, **k):
            self.tags.append(tag)

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return lambda *a, **k: None

    def _axis_trainer(self):
        from hexo_a0.trainer import Trainer

        mc = ModelConfig(
            hidden_dim=16, num_layers=1, num_heads=2,
            policy_hidden=8, value_hidden=8, graph_type="axis",
            conv_type="gatv2", threat_features=False, relative_stone_encoding=False,
        )
        cfg = FullConfig(
            model=mc,
            mcts=MCTSConfig(n_simulations=2, m_actions=2, exploration_moves=2),
            training=TrainingConfig(total_train_steps=0, lr_warmup_steps=0),
            self_play=SelfPlayConfig(
                device="cpu", workers=1, precision="fp32",
                python=PythonSelfPlayConfig(games_per_write=2),
            ),
            eval=EvalConfig(interval=1, games=0, sims=0, checkpoint_interval=0),
        )
        t = Trainer(HeXONet(mc), cfg, TINY_GAME_CONFIG, torch.device("cpu"))
        return t, mc

    @staticmethod
    def _fill_axis(trainer, mc, n=12):
        import random

        from hexo_a0.graph import game_to_axis_graph
        from hexo_a0.self_play import TrainingExample

        for _ in range(n):
            g = hexo_rs.GameState(trainer.game_config)
            for _ in range(random.randint(1, 6)):
                mv = g.legal_moves()
                if not mv or g.is_terminal():
                    break
                m = random.choice(mv)
                g.apply_move(m[0], m[1])
            if g.is_terminal():
                continue
            d = game_to_axis_graph(
                g, prune_empty_edges=mc.prune_empty_edges,
                threat_features=mc.threat_features,
                relative_stones=mc.relative_stone_encoding,
            )
            nl = int(d.legal_mask.sum())
            trainer.buffer.add(TrainingExample(
                data=d, policy_target=torch.ones(nl) / nl, value_target=0.0,
            ))

    def test_value_ablation_logged_and_buffer_sampled(self):
        t, mc = self._axis_trainer()
        try:
            self._fill_axis(t, mc)
            # Spy on buffer.sample to prove on-policy (not random-fallback) sampling.
            orig = t.buffer.sample
            calls = {"n": 0}
            t.buffer.sample = lambda n: (calls.__setitem__("n", calls["n"] + 1), orig(n))[1]
            t.writer = self._Rec()
            t._log_model_analysis(step=5)  # eval interval=1 → ablation gate fires
        finally:
            tags = getattr(t.writer, "tags", [])
            t.close()
        va = [x for x in tags if x.startswith("edge_features/value_ablation_l1/")]
        kl = [x for x in tags if x.startswith("edge_features/ablation_kl/")]
        # 5-wide absolute axis model → all 5 edge channels, both heads.
        assert len(va) == 5, va
        assert len(kl) == 5, kl
        assert "edge_features/value_ablation_l1/src_player" in va
        assert "edge_features/ablation_kl/src_player" in kl
        assert calls["n"] > 0  # buffer was sampled, not the random fallback
