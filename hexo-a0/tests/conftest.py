"""Shared fixtures for HeXO AlphaZero tests."""

import pytest
import hexo_rs
from hexo_a0.config import (
    EvalConfig,
    FullConfig,
    MCTSConfig,
    ModelConfig,
    PythonSelfPlayConfig,
    RunConfig,
    SelfPlayConfig,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Tiny configs for fast tests — explicit values so production defaults
# never leak into tests.
# ---------------------------------------------------------------------------

_TINY_GAME_CONFIG = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=20)

_TINY_MODEL_CONFIG = ModelConfig(
    hidden_dim=16, num_layers=1, num_heads=1,
    policy_hidden=16, value_hidden=16,
    graph_type="hex", conv_type="gatv2",
    prune_empty_edges=False, dropout=0.0, pre_norm=True,
)

_TINY_TRAINING_CONFIG = TrainingConfig(
    batch_size=8, buffer_capacity=1000,
    edge_budget=0, grad_accumulation=False,
    lr=2e-4, weight_decay=1e-4, max_grad_norm=1.0,
    lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0,
    step_delay=0.0, target_reuse=0.0, draw_value=0.0,
    augment_symmetries=False, reanalyze_fraction=0.0, pc_loss_weight=0.0,
)

_TINY_FULL_CONFIG = FullConfig(
    model=_TINY_MODEL_CONFIG,
    training=_TINY_TRAINING_CONFIG,
    mcts=MCTSConfig(
        n_simulations=2, m_actions=4,
        exploration_moves=5, exploration_fraction=0.0, exploration_max=0.0,
    ),
    self_play=SelfPlayConfig(
        device="cpu", workers=1, precision="fp32",
        python=PythonSelfPlayConfig(games_per_write=2),
    ),
    eval=EvalConfig(interval=0, checkpoint_interval=0, games=0, sims=0),
    run=RunConfig(device="cpu", fp16=False, compile=False),
)


@pytest.fixture
def tiny_game_config():
    return _TINY_GAME_CONFIG


@pytest.fixture
def tiny_model_config():
    return _TINY_MODEL_CONFIG


@pytest.fixture
def tiny_training_config():
    return _TINY_TRAINING_CONFIG


@pytest.fixture
def tiny_full_config():
    return _TINY_FULL_CONFIG


@pytest.fixture
def initial_game():
    """Default GameState: P1 stone at origin, P2 to move, 2 moves remaining."""
    return hexo_rs.GameState()


@pytest.fixture
def small_game():
    """Small game with ~5 stones played, P2 to move, 2 moves remaining.

    Sequence:
      P2 places at (1, 0) and (-1, 0)
      P1 places at (0, 1) and (0, -1)
    Resulting in 5 stones: origin P1 + 2 P2 + 2 extra P1.
    """
    cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    game = hexo_rs.GameState(cfg)
    game.apply_move(1, 0)    # P2 move 1
    game.apply_move(-1, 0)   # P2 move 2
    game.apply_move(0, 1)    # P1 move 1
    game.apply_move(0, -1)   # P1 move 2
    return game  # P2 to move, 2 remaining, 5 stones
