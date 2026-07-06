"""Tests for TOML config loading (config_io module)."""

import logging
import tempfile
from pathlib import Path

import pytest

from hexo_a0.config import (
    FullConfig,
    ModelConfig,
    GameConfig,
    MCTSConfig,
    TrainingConfig,
    SelfPlayConfig,
    EvalConfig,
    RunConfig,
)
from hexo_a0.config_io import load_config, save_config, default_config_toml


def _write_tmp(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    return Path(f.name)


class TestLoadEmpty:
    def test_empty_toml(self):
        cfg = load_config(_write_tmp(""))
        assert isinstance(cfg, FullConfig)
        assert cfg.model == ModelConfig()
        assert cfg.game == GameConfig()
        assert cfg.mcts == MCTSConfig()
        assert cfg.training == TrainingConfig()
        assert cfg.self_play == SelfPlayConfig()
        assert cfg.eval == EvalConfig()
        assert cfg.run == RunConfig()
        assert cfg.curriculum is None


class TestLoadSections:
    def test_model_section(self):
        cfg = load_config(_write_tmp("""
[model]
hidden_dim = 256
num_layers = 3
graph_type = "axis"
pre_norm = false
"""))
        assert cfg.model.hidden_dim == 256
        assert cfg.model.num_layers == 3
        assert cfg.model.graph_type == "axis"
        assert cfg.model.pre_norm is False
        assert cfg.model.num_heads == 8  # default preserved

    def test_game_section(self):
        cfg = load_config(_write_tmp("""
[game]
win_length = 4
placement_radius = 2
max_moves = 80
"""))
        assert cfg.game.win_length == 4
        assert cfg.game.placement_radius == 2
        assert cfg.game.max_moves == 80

    def test_mcts_section(self):
        cfg = load_config(_write_tmp("""
[mcts]
n_simulations = 32
m_actions = 8
exploration_moves = 10
"""))
        assert cfg.mcts.n_simulations == 32
        assert cfg.mcts.m_actions == 8
        assert cfg.mcts.exploration_moves == 10

    def test_mcts_truncate_delta_toml_parse(self):
        cfg = load_config(_write_tmp("""
[mcts]
truncate_delta = 0.25
"""))
        assert cfg.mcts.truncate_delta == pytest.approx(0.25)

    def test_training_section(self):
        cfg = load_config(_write_tmp("""
[training]
batch_size = 512
lr = 0.001
edge_budget = 500000
buffer_capacity = 200000
augment_symmetries = true
"""))
        assert cfg.training.batch_size == 512
        assert cfg.training.lr == pytest.approx(0.001)
        assert cfg.training.edge_budget == 500_000
        assert cfg.training.buffer_capacity == 200_000
        assert cfg.training.augment_symmetries is True

    def test_self_play_section(self):
        cfg = load_config(_write_tmp("""
[self_play]
engine = "python"
device = "cuda"
workers = 16
precision = "bf16"
"""))
        assert cfg.self_play.engine == "python"
        assert cfg.self_play.device == "cuda"
        assert cfg.self_play.workers == 16
        assert cfg.self_play.precision == "bf16"

    def test_self_play_rust_nested(self):
        cfg = load_config(_write_tmp("""
[self_play.rust]
max_batch = 128
max_file_mb = 512
"""))
        assert cfg.self_play.rust.max_batch == 128
        assert cfg.self_play.rust.max_file_mb == 512
        assert cfg.self_play.rust.batch_timeout_ms == 5  # default
        assert cfg.self_play.rust.inference_bin is None  # default

    def test_self_play_rust_inference_bin(self):
        cfg = load_config(_write_tmp("""
[self_play.rust]
inference_bin = "/path/to/hx04_infer_server"
"""))
        assert cfg.self_play.rust.inference_bin == "/path/to/hx04_infer_server"

    def test_self_play_python_nested(self):
        cfg = load_config(_write_tmp("""
[self_play.python]
games_per_write = 100
use_rust_mcts = false
"""))
        assert cfg.self_play.python.games_per_write == 100
        assert cfg.self_play.python.use_rust_mcts is False

    def test_eval_section(self):
        cfg = load_config(_write_tmp("""
[eval]
interval = 1000
checkpoint_interval = 300
games = 50
sims = 32
"""))
        assert cfg.eval.interval == 1000
        assert cfg.eval.checkpoint_interval == 300
        assert cfg.eval.games == 50
        assert cfg.eval.sims == 32

    def test_eval_sealbot_nested(self):
        cfg = load_config(_write_tmp("""
[eval.sealbot]
games = 20
adaptive = true
sims = 64
"""))
        assert cfg.eval.sealbot.games == 20
        assert cfg.eval.sealbot.adaptive is True
        assert cfg.eval.sealbot.sims == 64

    def test_run_section(self):
        cfg = load_config(_write_tmp("""
[run]
device = "cuda"
checkpoint_dir = "runs/test/checkpoints"
compile = false
fp16 = true
seed = 42
"""))
        assert cfg.run.device == "cuda"
        assert cfg.run.checkpoint_dir == "runs/test/checkpoints"
        assert cfg.run.compile is False
        assert cfg.run.fp16 is True
        assert cfg.run.seed == 42


class TestUnknownKeys:
    def test_unknown_key_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = load_config(_write_tmp("""
[mcts]
n_simulations = 32
totally_bogus_key = 999
"""))
        assert cfg.mcts.n_simulations == 32
        assert "totally_bogus_key" in caplog.text

    def test_unknown_section_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = load_config(_write_tmp("""
[unknown_section]
foo = 42
"""))
        assert "unknown_section" in caplog.text


class TestAutoNumLayers:
    def test_auto_num_layers(self):
        cfg = load_config(_write_tmp("""
[model]
num_layers = 0
[game]
win_length = 4
"""))
        assert cfg.model.num_layers == 6  # int(4 * 1.5)


class TestRoundTrip:
    def test_round_trip(self):
        original = FullConfig(
            model=ModelConfig(hidden_dim=256, graph_type="axis", pre_norm=False),
            game=GameConfig(win_length=4, placement_radius=2, max_moves=80),
            mcts=MCTSConfig(n_simulations=32, exploration_moves=10),
            training=TrainingConfig(batch_size=512, lr=0.001),
            self_play=SelfPlayConfig(engine="rust", device="cuda", precision="fp16"),
            eval=EvalConfig(interval=1000),
            run=RunConfig(device="cuda", compile=False, fp16=True),
        )

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            path = Path(f.name)
        save_config(path, original)
        loaded = load_config(path)

        assert loaded.model.hidden_dim == 256
        assert loaded.model.graph_type == "axis"
        assert loaded.model.pre_norm is False
        assert loaded.game.win_length == 4
        assert loaded.mcts.n_simulations == 32
        assert loaded.mcts.exploration_moves == 10
        assert loaded.training.batch_size == 512
        assert loaded.training.lr == pytest.approx(0.001)
        assert loaded.self_play.engine == "rust"
        assert loaded.self_play.device == "cuda"
        assert loaded.self_play.precision == "fp16"
        assert loaded.eval.interval == 1000
        assert loaded.run.device == "cuda"
        assert loaded.run.compile is False
        assert loaded.run.fp16 is True


class TestDefaultConfigToml:
    def test_valid_toml(self):
        import tomllib
        toml_str = default_config_toml()
        data = tomllib.loads(toml_str)
        assert "model" in data
        assert "mcts" in data
        assert "training" in data
        assert "self_play" in data
        assert "eval" in data
        assert "run" in data


class TestDefaultCurriculumToml:
    def test_valid_toml(self):
        import tomllib
        from hexo_a0.config_io import default_curriculum_toml
        toml_str = default_curriculum_toml()
        data = tomllib.loads(toml_str)
        assert "curriculum" in data
        assert "model" in data
        assert "curriculum" in data and "stages" in data["curriculum"]


class TestRealConfigs:
    def test_real_configs_load_without_warnings(self, caplog):
        """Every real config file under configs/ (recursing into subfolders) loads without unknown-key warnings."""
        configs_dir = Path(__file__).parents[2] / "configs"
        toml_files = sorted(configs_dir.rglob("*.toml"))
        assert len(toml_files) > 0, "No config files found"
        for path in toml_files:
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                cfg = load_config(path)
            assert isinstance(cfg, FullConfig), f"{path.name} didn't return FullConfig"
            warnings = [r for r in caplog.records if "Unknown" in r.message]
            assert not warnings, f"{path.name} has unknown keys: {[r.message for r in warnings]}"

    def test_default_toml_loads_without_warnings(self, caplog):
        """The root default.toml loads without unknown-key warnings."""
        default_path = Path(__file__).parents[2] / "default.toml"
        if not default_path.exists():
            pytest.skip("default.toml not found")
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            cfg = load_config(default_path)
        assert isinstance(cfg, FullConfig)
        warnings = [r for r in caplog.records if "Unknown" in r.message]
        assert not warnings, f"default.toml has unknown keys: {[r.message for r in warnings]}"


class TestJKConfigValidation:
    """Cross-section validation for JK + head-graft flags (2026-05-26)."""

    @pytest.mark.parametrize("mode", ["cat", "max", "lstm"])
    def test_pyg_mode_requires_use_jk(self, mode):
        toml = f"""
[model]
use_jk = false
jk_mode = "{mode}"
"""
        with pytest.raises(ValueError, match="requires use_jk=true"):
            load_config(_write_tmp(toml))

    def test_jk_mode_unknown_rejected(self):
        toml = """
[model]
use_jk = true
jk_mode = "concat"
"""
        with pytest.raises(ValueError, match="jk_mode='concat'"):
            load_config(_write_tmp(toml))

    @pytest.mark.parametrize("mode", ["sum", "cat", "max", "lstm"])
    def test_jk_mode_with_use_jk_accepted(self, mode):
        toml = f"""
[model]
use_jk = true
jk_mode = "{mode}"
"""
        cfg = load_config(_write_tmp(toml))
        assert cfg.model.jk_mode == mode

    def test_graft_heads_requires_cat_mode(self):
        toml = """
[model]
use_jk = true
jk_mode = "sum"

[training]
graft_heads_for_jk_cat = true
"""
        with pytest.raises(ValueError, match="graft_heads_for_jk_cat=true requires"):
            load_config(_write_tmp(toml))

    def test_graft_heads_requires_use_jk(self):
        toml = """
[model]
use_jk = false
jk_mode = "sum"

[training]
graft_heads_for_jk_cat = true
"""
        with pytest.raises(ValueError, match="graft_heads_for_jk_cat=true requires"):
            load_config(_write_tmp(toml))

    def test_graft_heads_with_cat_accepted(self):
        toml = """
[model]
use_jk = true
jk_mode = "cat"

[training]
graft_heads_for_jk_cat = true
do_lr_warmup_on_resume = true
"""
        cfg = load_config(_write_tmp(toml))
        assert cfg.training.graft_heads_for_jk_cat is True
        assert cfg.training.do_lr_warmup_on_resume is True


class TestRelativeStoneEncodingValidation:
    """relative_stone_encoding is from-scratch-only: incompatible with the
    threat-features input graft (the own/opp remap is input-dependent)."""

    def test_relative_with_graft_rejected(self):
        toml = """
[model]
threat_features = true
relative_stone_encoding = true

[training]
graft_threat_features = true
"""
        with pytest.raises(ValueError, match="relative_stone_encoding"):
            load_config(_write_tmp(toml))

    def test_relative_without_graft_accepted(self):
        toml = """
[model]
threat_features = true
relative_stone_encoding = true
"""
        cfg = load_config(_write_tmp(toml))
        assert cfg.model.relative_stone_encoding is True
        assert cfg.model.threat_features is True

    def test_graft_without_relative_accepted(self):
        toml = """
[model]
threat_features = true

[training]
graft_threat_features = true
"""
        cfg = load_config(_write_tmp(toml))
        assert cfg.model.relative_stone_encoding is False
        assert cfg.training.graft_threat_features is True


class TestExports:
    def test_exported(self):
        import hexo_a0
        assert hasattr(hexo_a0, "load_config")
        assert hasattr(hexo_a0, "save_config")
        assert hasattr(hexo_a0, "ModelConfig")
        assert hasattr(hexo_a0, "FullConfig")
