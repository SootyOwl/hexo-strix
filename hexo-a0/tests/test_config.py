"""Tests for config dataclasses."""

import pytest
from hexo_a0.config import (
    ModelConfig,
    GameConfig,
    MCTSConfig,
    TrainingConfig,
    SelfPlayConfig,
    RustSelfPlayConfig,
    PythonSelfPlayConfig,
    EvalConfig,
    SealbotConfig,
    RunConfig,
    CurriculumConfig,
    FullConfig,
)


class TestModelConfig:
    def test_defaults(self):
        c = ModelConfig()
        assert c.hidden_dim == 256
        assert c.num_layers == 3
        assert c.num_heads == 8
        assert c.conv_type == "gine"
        assert c.policy_hidden == 128
        assert c.value_hidden == 128
        assert c.dropout == 0.0
        assert c.graph_type == "axis"
        assert c.prune_empty_edges is True
        assert c.pre_norm is True

    def test_custom(self):
        c = ModelConfig(hidden_dim=64, num_layers=6, graph_type="hex", pre_norm=False)
        assert c.hidden_dim == 64
        assert c.num_layers == 6
        assert c.graph_type == "hex"
        assert c.pre_norm is False


class TestGameConfig:
    def test_defaults(self):
        c = GameConfig()
        assert c.win_length == 6
        assert c.placement_radius == 8
        assert c.max_moves == 200


class TestMCTSConfig:
    def test_defaults(self):
        c = MCTSConfig()
        assert c.n_simulations == 64
        assert c.m_actions == 16
        assert c.c_scale == 1.0
        assert c.c_visit == 50
        assert c.exploration_moves == 30
        assert c.exploration_fraction == 0.25
        assert c.exploration_max == 0.9
        assert c.exploration_window == 2000

    def test_truncate_delta_default_off(self):
        c = MCTSConfig()
        assert c.truncate_delta == 0.0


class TestTrainingConfig:
    def test_defaults(self):
        c = TrainingConfig()
        assert c.batch_size == 512
        assert c.lr == 2e-4
        assert c.weight_decay == 1e-4
        assert c.max_grad_norm == 1.0
        assert c.lr_schedule == "cosine"
        assert c.lr_min == 2e-5
        assert c.lr_warmup_steps == 400
        assert c.total_train_steps == 5000
        assert c.edge_budget == 500_000
        assert c.grad_accumulation is True
        assert c.target_reuse == 8.0
        assert c.draw_value == -0.3
        assert c.augment_symmetries is True
        assert c.buffer_capacity == 250_000
        assert c.reanalyze_fraction == 0.0
        assert c.pc_loss_weight == 0.0

    def test_no_legacy_fields(self):
        c = TrainingConfig()
        assert not hasattr(c, "games_per_iteration")
        assert not hasattr(c, "train_steps_per_iteration")
        assert not hasattr(c, "n_simulations")
        assert not hasattr(c, "exploration_moves")
        assert not hasattr(c, "use_fp16")
        assert not hasattr(c, "self_play_device")


class TestSelfPlayConfig:
    def test_defaults(self):
        c = SelfPlayConfig()
        assert c.engine == "rust"
        assert c.device == "cuda"
        assert c.workers == 16
        assert c.precision == "bf16"

    def test_nested_defaults(self):
        c = SelfPlayConfig()
        assert c.rust.max_batch == 128
        assert c.rust.batch_timeout_ms == 5
        assert c.rust.python_inference is True
        assert c.rust.pool_fraction == 0.3
        assert c.rust.playout_cap_fraction == 0.75
        assert c.rust.playout_cap_divisor == 4
        assert c.rust.max_file_mb == 2048
        assert c.rust.output_filename == "games"
        assert c.rust.inference_bin is None
        assert c.python.games_per_write == 50
        assert c.python.playout_cap_fraction == 0.75
        assert c.python.playout_cap_divisor == 4
        assert c.python.use_rust_mcts is True


class TestEvalConfig:
    def test_defaults(self):
        c = EvalConfig()
        assert c.interval == 1000
        assert c.checkpoint_interval == 300
        assert c.games == 50
        assert c.sims == 256
        assert c.m_actions == 16

    def test_sealbot_defaults(self):
        c = EvalConfig()
        assert c.sealbot.games == 20
        assert c.sealbot.adaptive is True
        assert c.sealbot.sims == 256
        assert c.sealbot.m_actions == 16
        assert c.sealbot.time_limit == 0.5
        assert c.sealbot.levels == [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        assert c.sealbot.promote_threshold == 0.7
        assert c.sealbot.promote_patience == 2
        assert c.sealbot.win_length == 6
        assert c.sealbot.placement_radius == 8
        assert c.sealbot.max_moves == 1000


class TestRunConfig:
    def test_defaults(self):
        c = RunConfig()
        assert c.device == "cuda"
        assert c.checkpoint_dir == "checkpoints"
        assert c.log_dir == "runs"
        assert c.compile is True
        assert c.fp16 is True
        assert c.seed is None


class TestCurriculumConfig:
    def test_defaults(self):
        c = CurriculumConfig()
        assert c.convergence_mode == "champion"
        assert c.win_rate_threshold == 0.85
        assert c.improve_threshold == 0.55
        assert c.champion_threshold == 0.55
        assert c.patience == 10
        assert c.checkpoint_lag == 30
        assert c.min_steps_per_stage == 10_000
        assert c.confirm_advance is True


class TestFullConfig:
    def test_defaults(self):
        c = FullConfig()
        assert isinstance(c.model, ModelConfig)
        assert isinstance(c.game, GameConfig)
        assert isinstance(c.mcts, MCTSConfig)
        assert isinstance(c.training, TrainingConfig)
        assert isinstance(c.self_play, SelfPlayConfig)
        assert isinstance(c.eval, EvalConfig)
        assert isinstance(c.run, RunConfig)
        assert c.curriculum is None

    def test_independence(self):
        assert not issubclass(TrainingConfig, ModelConfig)
        assert not issubclass(ModelConfig, TrainingConfig)
