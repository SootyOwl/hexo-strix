"""End-to-end integration tests for HeXO AlphaZero training pipeline."""

from __future__ import annotations

import math

import pytest
import torch

import hexo_rs
from hexo_a0 import (
    ModelConfig,
    HeXONet,
    ReplayBuffer,
    Trainer,
    train_step,
    self_play_game,
    TrainingExample,
    gumbel_mcts,
    kl_policy_loss,
)
from hexo_a0.config import FullConfig, MCTSConfig, TrainingConfig, SelfPlayConfig, PythonSelfPlayConfig, EvalConfig


# ---------------------------------------------------------------------------
# Test 1: Full training loop (2 iterations)
# ---------------------------------------------------------------------------


def test_full_training_loop():
    """Run 2 training iterations end-to-end with tiny config."""
    game_config = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=30)
    model_config = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex", conv_type="gatv2")
    config = FullConfig(
        model=model_config,
        mcts=MCTSConfig(n_simulations=2, m_actions=4, exploration_moves=5),
        training=TrainingConfig(batch_size=8, buffer_capacity=500, edge_budget=0, grad_accumulation=False, lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0, target_reuse=0.0, augment_symmetries=False),
        self_play=SelfPlayConfig(device="cpu", workers=1, precision="fp32", python=PythonSelfPlayConfig(games_per_write=2)),
        eval=EvalConfig(interval=0, games=0, sims=0, checkpoint_interval=0),
    )
    device = torch.device("cpu")
    model = HeXONet(model_config).to(device)

    trainer = Trainer(model, config, game_config, device)
    # Fill buffer with synthetic data
    from hexo_a0.graph import game_to_graph
    game = hexo_rs.GameState(game_config)
    data = game_to_graph(game)
    n_legal = data.legal_mask.sum().item()
    for _ in range(20):
        trainer.buffer.add(TrainingExample(
            data=data,
            policy_target=torch.ones(n_legal) / n_legal,
            value_target=0.0,
        ))

    # Iteration 1
    metrics1 = trainer.train(steps=3)
    assert math.isfinite(metrics1["total_loss"])
    assert math.isfinite(metrics1["policy_loss"])
    assert math.isfinite(metrics1["value_loss"])
    assert metrics1["buffer_size"] > 0

    # Iteration 2
    metrics2 = trainer.train(steps=3)
    assert math.isfinite(metrics2["total_loss"])
    assert trainer.train_steps == 6  # 2 calls * 3 steps each


# ---------------------------------------------------------------------------
# Test 2: Checkpoint round-trip across iterations
# ---------------------------------------------------------------------------


def test_checkpoint_resume(tmp_path):
    """Save after iter 1, load into new trainer, continue for iter 2."""
    game_config = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=30)
    model_config = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex", conv_type="gatv2")
    config = FullConfig(
        model=model_config,
        mcts=MCTSConfig(n_simulations=2, m_actions=4, exploration_moves=5),
        training=TrainingConfig(batch_size=8, buffer_capacity=500, edge_budget=0, grad_accumulation=False, lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0, target_reuse=0.0, augment_symmetries=False),
        self_play=SelfPlayConfig(device="cpu", workers=1, precision="fp32", python=PythonSelfPlayConfig(games_per_write=1)),
        eval=EvalConfig(interval=0, games=0, sims=0, checkpoint_interval=0),
    )
    device = torch.device("cpu")

    # Train 1 iteration and save
    model1 = HeXONet(model_config).to(device)
    trainer1 = Trainer(model1, config, game_config, device)
    from hexo_a0.graph import game_to_graph
    game = hexo_rs.GameState(game_config)
    data = game_to_graph(game)
    n_legal = data.legal_mask.sum().item()
    for _ in range(20):
        trainer1.buffer.add(TrainingExample(
            data=data,
            policy_target=torch.ones(n_legal) / n_legal,
            value_target=0.0,
        ))
    trainer1.train(steps=3)
    ckpt_path = tmp_path / "checkpoint.pt"
    trainer1.save_checkpoint(str(ckpt_path), save_buffer=True)

    # Load into fresh trainer and continue
    model2 = HeXONet(model_config).to(device)
    trainer2 = Trainer(model2, config, game_config, device)
    trainer2.load_checkpoint(str(ckpt_path))
    assert trainer2.train_steps == 3  # restored from checkpoint

    # Run more steps — should work without error (buffer restored)
    metrics = trainer2.train(steps=3)
    assert trainer2.train_steps == 6

    assert math.isfinite(metrics["total_loss"])


# ---------------------------------------------------------------------------
# Test 4: Per-head diagnostic TensorBoard tags
# ---------------------------------------------------------------------------


def _train_with_writer(model_config, training_config, examples, steps, tmp_path):
    """Run a few train() steps with a real TensorBoard writer and return the
    set of scalar tags written."""
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from hexo_a0.tb_writer import DescribedWriter

    game_config = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=30)
    config = FullConfig(
        model=model_config,
        mcts=MCTSConfig(n_simulations=2, m_actions=4, exploration_moves=5),
        training=training_config,
        self_play=SelfPlayConfig(device="cpu", workers=1, precision="fp32",
                                  python=PythonSelfPlayConfig(games_per_write=1)),
        eval=EvalConfig(interval=0, games=0, sims=0, checkpoint_interval=0),
    )
    device = torch.device("cpu")
    trainer = Trainer(HeXONet(model_config).to(device), config, game_config, device)
    for ex in examples:
        trainer.buffer.add(ex)
    log_dir = tmp_path / "tb"
    trainer.writer = DescribedWriter(SummaryWriter(log_dir=str(log_dir)))
    trainer.train(steps=steps)
    trainer.writer.flush()
    trainer.writer.close()

    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    return set(ea.Tags()["scalars"])


def test_per_head_diagnostic_tags_emitted(tmp_path):
    """All three value-head upgrades emit their diagnostic TensorBoard tags."""
    from hexo_a0.config import TrainingConfig
    from hexo_a0.self_play import TrainingExample
    from hexo_a0.graph import game_to_graph

    model_config = ModelConfig(
        hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex",
        conv_type="gatv2", value_bins=9, value_horizons=[4], q_head=True,
    )
    training = TrainingConfig(
        batch_size=8, buffer_capacity=500, edge_budget=0, grad_accumulation=False,
        lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0,
        target_reuse=0.0, augment_symmetries=False,
        horizon_loss_weight=0.25, q_loss_weight=0.5,
    )
    game = hexo_rs.GameState(hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=30))
    data = game_to_graph(game)
    n_legal = int(data.legal_mask.sum().item())
    examples = [
        TrainingExample(
            data=data, policy_target=torch.ones(n_legal) / n_legal,
            value_target=1.0, horizon_targets=[1.0],
            q_targets=[0.5] * n_legal, q_visits=[1] * n_legal,
        )
        for _ in range(20)
    ]
    tags = _train_with_writer(model_config, training, examples, 3, tmp_path)
    # Horizon head (per-horizon loss + coverage)
    assert "loss/horizon_h4" in tags
    assert "horizon/coverage_h4" in tags
    # Q head (MAE, coverage, correlation)
    assert "loss/q_mae" in tags
    assert "q/coverage" in tags
    assert "q/corr" in tags
    # Distributional value head (entropy + sign accuracy)
    assert "value/pred_entropy" in tags
    assert "value/sign_accuracy" in tags


def test_legacy_config_emits_no_new_head_tags(tmp_path):
    """A scalar-head, no-horizon, no-Q config must not emit the new tags —
    legacy runs keep their TensorBoard tag set unchanged."""
    from hexo_a0.config import TrainingConfig
    from hexo_a0.self_play import TrainingExample
    from hexo_a0.graph import game_to_graph

    model_config = ModelConfig(
        hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex",
        conv_type="gatv2",  # value_bins=0, no horizons, no q_head
    )
    training = TrainingConfig(
        batch_size=8, buffer_capacity=500, edge_budget=0, grad_accumulation=False,
        lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0,
        target_reuse=0.0, augment_symmetries=False,
    )
    game = hexo_rs.GameState(hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=30))
    data = game_to_graph(game)
    n_legal = int(data.legal_mask.sum().item())
    examples = [
        TrainingExample(data=data, policy_target=torch.ones(n_legal) / n_legal,
                        value_target=0.0)
        for _ in range(20)
    ]
    tags = _train_with_writer(model_config, training, examples, 3, tmp_path)
    new_tags = {
        "loss/horizon_h4", "horizon/coverage_h4",
        "loss/q_mae", "q/coverage", "q/corr",
        "value/pred_entropy", "value/sign_accuracy",
    }
    assert not (tags & new_tags), f"legacy config leaked new tags: {tags & new_tags}"


# ---------------------------------------------------------------------------
# Test 3: Ordering consistency
# ---------------------------------------------------------------------------


def test_ordering_consistency():
    """Verify coords align across game_to_graph, evaluate_state, and gumbel_mcts."""
    from hexo_a0.graph import game_to_graph
    from hexo_a0.mcts import evaluate_state, gumbel_mcts as _gumbel_mcts

    game_config = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    model_config = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex", conv_type="gatv2")
    mcts_config = MCTSConfig(n_simulations=2, m_actions=4)
    device = torch.device("cpu")
    model = HeXONet(model_config).to(device)
    model.eval()

    game = hexo_rs.GameState(game_config)

    # Get coords from game_to_graph
    data = game_to_graph(game)
    graph_coords = [(c[0].item(), c[1].item()) for c in data.coords[data.legal_mask]]

    # Get coords from evaluate_state
    _, _, eval_coords = evaluate_state(model, game, device)

    # Get improved policy from gumbel_mcts
    action, policy = _gumbel_mcts(game, model, mcts_config, device)

    # All three should agree on coordinate ordering
    assert graph_coords == eval_coords, "graph vs evaluate_state coords mismatch"
    assert len(policy) == len(graph_coords), "policy length vs coords mismatch"
    assert action in [tuple(c) for c in graph_coords], "action not in legal coords"
