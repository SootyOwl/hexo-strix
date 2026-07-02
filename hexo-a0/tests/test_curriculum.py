"""Tests for hexo_a0.curriculum — stage config building from FullConfig."""

from __future__ import annotations

import textwrap

import pytest

from hexo_a0.config import (
    CurriculumConfig,
    FullConfig,
    GameConfig,
    MCTSConfig,
    ModelConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from hexo_a0.config_io import load_config
from hexo_a0.curriculum import (
    CorpusEvalRunner,
    _build_stage_config,
    _corpus_record_scalars,
    _load_detector_state,
    _prev_ckpt_plateau_fired,
    _promo_velocity_fired,
    _resolve_self_play_source,
    _save_detector_state,
    _wilson_upper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_toml(tmp_path, content: str) -> str:
    """Write a TOML string to a temp file and return its path."""
    p = tmp_path / "curriculum.toml"
    p.write_text(textwrap.dedent(content))
    return str(p)


# A minimal but complete curriculum TOML used by several tests.
MINIMAL_TOML = """\
[curriculum]
convergence_mode = "checkpoint"
win_rate_threshold = 0.90
improve_threshold = 0.60
patience = 5
checkpoint_lag = 3

[model]
hidden_dim = 64
num_layers = 0
num_heads = 4
policy_hidden = 32
value_hidden = 32
dropout = 0.1

[game]
win_length = 6
placement_radius = 8
max_moves = 200

[mcts]
n_simulations = 16
m_actions = 8
c_scale = 2.0
c_visit = 100

[training]
batch_size = 32
lr = 0.001
weight_decay = 0.01
max_grad_norm = 2.0

[eval]
interval = 500
games = 10

[self_play]
workers = 2

[run]
device = "cpu"
checkpoint_dir = "runs/test"

[[curriculum.stages]]
name = "Stage A"
win_length = 3
placement_radius = 2
max_moves = 60
lr = 0.0005
exploration_moves = 6
log_dir = "runs/test/s1"

[[curriculum.stages]]
name = "Stage B"
win_length = 4
placement_radius = 3
max_moves = 100
lr = 0.0003
exploration_moves = 10
buffer_capacity = 80000
log_dir = "runs/test/s2"
"""


# ===================================================================
# load_config tests (curriculum sections)
# ===================================================================

class TestLoadCurriculumConfig:
    """Tests for load_config with curriculum TOML files."""

    def test_returns_full_config(self, tmp_path):
        path = _write_toml(tmp_path, MINIMAL_TOML)
        cfg = load_config(path)
        assert isinstance(cfg, FullConfig)

    def test_curriculum_not_none(self, tmp_path):
        path = _write_toml(tmp_path, MINIMAL_TOML)
        cfg = load_config(path)
        assert cfg.curriculum is not None

    def test_stages_loaded(self, tmp_path):
        path = _write_toml(tmp_path, MINIMAL_TOML)
        cfg = load_config(path)
        assert len(cfg.curriculum.stages) == 2
        assert cfg.curriculum.stages[0]["name"] == "Stage A"
        assert cfg.curriculum.stages[1]["name"] == "Stage B"

    def test_curriculum_thresholds_and_patience(self, tmp_path):
        path = _write_toml(tmp_path, MINIMAL_TOML)
        cfg = load_config(path)
        assert cfg.curriculum.convergence_mode == "checkpoint"
        assert cfg.curriculum.win_rate_threshold == 0.90
        assert cfg.curriculum.improve_threshold == 0.60
        assert cfg.curriculum.patience == 5
        assert cfg.curriculum.checkpoint_lag == 3

    def test_shared_sections_preserved(self, tmp_path):
        path = _write_toml(tmp_path, MINIMAL_TOML)
        cfg = load_config(path)
        assert cfg.model.hidden_dim == 64
        assert cfg.mcts.n_simulations == 16
        assert cfg.training.batch_size == 32
        assert cfg.self_play.workers == 2
        assert cfg.run.device == "cpu"

    def test_default_values_for_missing_keys(self, tmp_path):
        """When [curriculum] section exists but keys are absent, defaults apply."""
        toml = """\
        [curriculum]

        [model]
        hidden_dim = 64
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.curriculum is not None
        assert cfg.curriculum.convergence_mode == "champion"
        assert cfg.curriculum.win_rate_threshold == 0.85
        assert cfg.curriculum.improve_threshold == 0.55
        assert cfg.curriculum.champion_threshold == 0.55
        assert cfg.curriculum.patience == 10
        assert cfg.curriculum.checkpoint_lag == 30

    def test_no_curriculum_section(self, tmp_path):
        """No [curriculum] section at all -> curriculum is None."""
        toml = """\
        [model]
        hidden_dim = 64
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.curriculum is None


# ===================================================================
# _build_stage_config tests
# ===================================================================

class TestBuildStageConfig:
    """Tests for _build_stage_config."""

    @pytest.fixture()
    def base_cfg(self, tmp_path):
        path = _write_toml(tmp_path, MINIMAL_TOML)
        return load_config(path)

    def test_returns_full_config(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        assert isinstance(result, FullConfig)

    def test_game_config_from_stage(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        assert result.game.win_length == 3
        assert result.game.placement_radius == 2
        assert result.game.max_moves == 60

    def test_auto_num_layers_from_base(self, base_cfg):
        """num_layers=0 in TOML => auto-computed from base game.win_length by load_config.

        Model architecture is fixed across curriculum stages, so num_layers
        is computed once from the base [game] section (win_length=6 => 9).
        """
        stage = base_cfg.curriculum.stages[0]  # win_length=3 (stage override)
        result = _build_stage_config(base_cfg, stage)
        # Auto-computed from base game.win_length=6: int(6 * 1.5) = 9
        assert result.model.num_layers == 9

    def test_auto_num_layers_same_across_stages(self, base_cfg):
        """All stages share the same auto-computed num_layers from base config."""
        stage_a = _build_stage_config(base_cfg, base_cfg.curriculum.stages[0])
        stage_b = _build_stage_config(base_cfg, base_cfg.curriculum.stages[1])
        assert stage_a.model.num_layers == stage_b.model.num_layers

    def test_explicit_num_layers_not_overridden(self, tmp_path):
        """When num_layers is set to a non-zero value, auto-calc is skipped."""
        toml = """\
        [model]
        hidden_dim = 64
        num_layers = 10
        [game]
        win_length = 6
        [curriculum]
        [[curriculum.stages]]
        win_length = 3
        placement_radius = 2
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        stage = cfg.curriculum.stages[0]
        result = _build_stage_config(cfg, stage)
        assert result.model.num_layers == 10

    def test_shared_mcts_preserved(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        assert result.mcts.n_simulations == 16
        assert result.mcts.m_actions == 8
        assert result.mcts.c_scale == 2.0
        assert result.mcts.c_visit == 100

    def test_shared_training_preserved(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        assert result.training.batch_size == 32
        assert result.training.weight_decay == 0.01
        assert result.training.max_grad_norm == 2.0

    def test_shared_eval_preserved(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        assert result.eval.interval == 500
        assert result.eval.games == 10

    def test_shared_self_play_preserved(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        assert result.self_play.workers == 2

    def test_stage_lr_overrides_shared(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        # Shared lr = 0.001, stage overrides to 0.0005
        assert result.training.lr == 0.0005

    def test_stage_exploration_moves_override(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        assert result.mcts.exploration_moves == 6

    def test_stage_buffer_capacity_override(self, base_cfg):
        """Per-stage buffer_capacity overrides shared training.buffer_capacity."""
        stage = base_cfg.curriculum.stages[1]  # has buffer_capacity = 80000
        result = _build_stage_config(base_cfg, stage)
        assert result.training.buffer_capacity == 80000

    def test_model_config_shared_fields(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        assert result.model.hidden_dim == 64
        assert result.model.num_heads == 4
        assert result.model.policy_hidden == 32
        assert result.model.value_hidden == 32
        assert result.model.dropout == 0.1

    def test_log_dir_override(self, base_cfg):
        stage = base_cfg.curriculum.stages[0]
        result = _build_stage_config(base_cfg, stage)
        assert result.run.log_dir == "runs/test/s1"

    def test_does_not_mutate_base(self, base_cfg):
        """Building a stage config should not mutate the base config."""
        original_lr = base_cfg.training.lr
        stage = base_cfg.curriculum.stages[0]
        _build_stage_config(base_cfg, stage)
        assert base_cfg.training.lr == original_lr

    def test_sealbot_prefix_override(self, base_cfg):
        """sealbot_games in stage overrides eval.sealbot.games."""
        stage = {"win_length": 4, "placement_radius": 2, "sealbot_games": 30}
        result = _build_stage_config(base_cfg, stage)
        assert result.eval.sealbot.games == 30

    def test_sealbot_multiple_prefix_overrides(self, base_cfg):
        """Multiple sealbot_ prefixed keys override nested fields."""
        stage = {
            "win_length": 4, "placement_radius": 2,
            "sealbot_games": 20, "sealbot_adaptive": True, "sealbot_sims": 128,
        }
        result = _build_stage_config(base_cfg, stage)
        assert result.eval.sealbot.games == 20
        assert result.eval.sealbot.adaptive is True
        assert result.eval.sealbot.sims == 128

    def test_rust_prefix_override(self, base_cfg):
        """rust_ prefixed keys override self_play.rust fields."""
        stage = {"win_length": 4, "placement_radius": 2, "rust_max_batch": 128}
        result = _build_stage_config(base_cfg, stage)
        assert result.self_play.rust.max_batch == 128

    def test_python_prefix_override(self, base_cfg):
        """python_ prefixed keys override self_play.python fields."""
        stage = {"win_length": 4, "placement_radius": 2, "python_games_per_write": 100}
        result = _build_stage_config(base_cfg, stage)
        assert result.self_play.python.games_per_write == 100

    def test_prefix_override_does_not_mutate_base(self, base_cfg):
        """Prefixed overrides don't mutate the base config's nested objects."""
        original_games = base_cfg.eval.sealbot.games
        stage = {"win_length": 4, "placement_radius": 2, "sealbot_games": 99}
        _build_stage_config(base_cfg, stage)
        assert base_cfg.eval.sealbot.games == original_games

    # --- Full-prefix override tests (regression for tree-based resolver) ---

    def test_full_prefix_eval_sealbot_games(self, base_cfg):
        """eval_sealbot_games routes to eval.sealbot.games."""
        stage = {"eval_sealbot_games": 42}
        result = _build_stage_config(base_cfg, stage)
        assert result.eval.sealbot.games == 42

    def test_full_prefix_eval_games(self, base_cfg):
        """eval_games routes to eval.games."""
        stage = {"eval_games": 77}
        result = _build_stage_config(base_cfg, stage)
        assert result.eval.games == 77

    def test_full_prefix_game_win_length(self, base_cfg):
        """game_win_length routes to game.win_length."""
        stage = {"game_win_length": 5, "game_placement_radius": 3}
        result = _build_stage_config(base_cfg, stage)
        assert result.game.win_length == 5
        assert result.game.placement_radius == 3

    def test_full_prefix_mcts_n_simulations(self, base_cfg):
        """mcts_n_simulations routes to mcts.n_simulations."""
        stage = {"mcts_n_simulations": 256}
        result = _build_stage_config(base_cfg, stage)
        assert result.mcts.n_simulations == 256

    def test_full_prefix_self_play_rust(self, base_cfg):
        """self_play_rust_max_batch routes to self_play.rust.max_batch."""
        stage = {"self_play_rust_max_batch": 128}
        result = _build_stage_config(base_cfg, stage)
        assert result.self_play.rust.max_batch == 128

    def test_legacy_shorthand_matches_full_prefix(self, base_cfg):
        """Legacy sealbot_games and full eval_sealbot_games produce identical results."""
        legacy = _build_stage_config(base_cfg, {"sealbot_games": 50})
        full = _build_stage_config(base_cfg, {"eval_sealbot_games": 50})
        assert legacy.eval.sealbot.games == 50
        assert full.eval.sealbot.games == 50

    def test_full_prefix_overrides_bare_key(self, base_cfg):
        """When both bare and prefixed keys are present, prefixed wins."""
        stage = {"n_simulations": 32, "mcts_n_simulations": 256}
        result = _build_stage_config(base_cfg, stage)
        assert result.mcts.n_simulations == 256

    def test_run_log_dir_prefix(self, base_cfg):
        """run_log_dir routes to run.log_dir (regression: curriculum loop reads
        stage_cfg.run.log_dir, so the prefix must work)."""
        stage = {"run_log_dir": "runs/custom/stage1"}
        result = _build_stage_config(base_cfg, stage)
        assert result.run.log_dir == "runs/custom/stage1"

    def test_bare_log_dir_still_works(self, base_cfg):
        """Bare log_dir still routes to run.log_dir for backwards compat."""
        stage = {"log_dir": "runs/legacy/path"}
        result = _build_stage_config(base_cfg, stage)
        assert result.run.log_dir == "runs/legacy/path"


# ===================================================================
# Stage-override re-validation (cross-section invariants)
# ===================================================================

class TestStageOverrideValidation:
    """Per-stage overrides merge AFTER load_config's validation pass, so the
    merged stage config must be re-validated: invalid combinations the base
    config never saw (e.g. graft_threat_features as a stage override on a
    relative-encoding model) must raise instead of being silently merged."""

    RELATIVE_TOML = """\
    [model]
    hidden_dim = 32
    num_layers = 2
    num_heads = 4
    relative_stone_encoding = true
    threat_features = true

    [game]
    win_length = 4
    placement_radius = 2

    [curriculum]

    [[curriculum.stages]]
    win_length = 4
    placement_radius = 2
    """

    def test_graft_override_on_relative_model_raises(self, tmp_path):
        path = _write_toml(tmp_path, self.RELATIVE_TOML)
        cfg = load_config(path)
        stage = dict(cfg.curriculum.stages[0])
        stage["training_graft_threat_features"] = True
        with pytest.raises(ValueError, match="graft_threat_features"):
            _build_stage_config(cfg, stage)

    def test_valid_stage_override_still_passes(self, tmp_path):
        path = _write_toml(tmp_path, self.RELATIVE_TOML)
        cfg = load_config(path)
        stage = dict(cfg.curriculum.stages[0])
        result = _build_stage_config(cfg, stage)
        assert result.model.relative_stone_encoding is True


# ===================================================================
# Stage-level convergence-control overrides (regression test)
# ===================================================================

class TestStageConvergenceOverrides:
    """Per-stage overrides for curriculum-control fields must take precedence
    over the curriculum-level defaults during the convergence loop.
    """

    def test_stage_overrides_resolve_with_precedence(self, tmp_path):
        toml = """\
        [curriculum]
        convergence_mode = "random"
        win_rate_threshold = 0.85
        improve_threshold = 0.55
        patience = 3
        checkpoint_lag = 1
        min_steps_per_stage = 0

        [model]
        hidden_dim = 16

        [game]
        win_length = 3
        placement_radius = 2

        [[curriculum.stages]]
        name = "S1"
        win_length = 3
        placement_radius = 2
        min_steps_per_stage = 10000
        patience = 5
        checkpoint_lag = 10
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        stage = cfg.curriculum.stages[0]
        stage_cfg = _build_stage_config(cfg, stage)

        # The same resolution logic used inside run_curriculum's stage loop:
        convergence_mode = stage.get("convergence_mode", cfg.curriculum.convergence_mode)
        win_rate_threshold = stage.get("win_rate_threshold", cfg.curriculum.win_rate_threshold)
        improve_threshold = stage.get("improve_threshold", cfg.curriculum.improve_threshold)
        patience = stage.get("patience", cfg.curriculum.patience)
        checkpoint_lag = stage.get("checkpoint_lag", cfg.curriculum.checkpoint_lag)
        min_steps_per_stage = stage.get("min_steps_per_stage", cfg.curriculum.min_steps_per_stage)

        assert min_steps_per_stage == 10000  # stage override, not 0
        assert patience == 5                  # stage override, not 3
        assert checkpoint_lag == 10           # stage override, not 1
        # Non-overridden fields fall back to curriculum-level defaults
        assert convergence_mode == "random"
        assert win_rate_threshold == 0.85
        assert improve_threshold == 0.55


# ===================================================================
# Champion mode config tests
# ===================================================================

class TestChampionModeConfig:
    """Tests for champion mode configuration parsing and stage overrides."""

    def test_champion_threshold_default(self):
        cfg = CurriculumConfig()
        assert cfg.champion_threshold == 0.55

    def test_champion_threshold_from_toml(self, tmp_path):
        toml = """\
        [curriculum]
        convergence_mode = "champion"
        champion_threshold = 0.60
        patience = 5

        [model]
        hidden_dim = 16
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.curriculum.convergence_mode == "champion"
        assert cfg.curriculum.champion_threshold == 0.60

    def test_champion_threshold_stage_override(self, tmp_path):
        toml = """\
        [curriculum]
        convergence_mode = "champion"
        champion_threshold = 0.55
        patience = 5

        [model]
        hidden_dim = 16

        [[curriculum.stages]]
        name = "S1"
        win_length = 3
        placement_radius = 2
        champion_threshold = 0.65
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        stage = cfg.curriculum.stages[0]
        # Stage override should take precedence (resolved in curriculum.py loop)
        champion_threshold = stage.get("champion_threshold", cfg.curriculum.champion_threshold)
        assert champion_threshold == 0.65

    def test_champion_mode_with_random_warmup_stages(self, tmp_path):
        """Champion mode at curriculum level with random warmup stages."""
        toml = """\
        [curriculum]
        convergence_mode = "champion"
        champion_threshold = 0.55

        [model]
        hidden_dim = 16

        [[curriculum.stages]]
        name = "Warmup"
        win_length = 3
        placement_radius = 2
        convergence_mode = "random"
        win_rate_threshold = 0.90

        [[curriculum.stages]]
        name = "Main"
        win_length = 6
        placement_radius = 8
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        warmup = cfg.curriculum.stages[0]
        main = cfg.curriculum.stages[1]
        # Warmup stage overrides to "random"
        assert warmup.get("convergence_mode", cfg.curriculum.convergence_mode) == "random"
        # Main stage inherits "champion" from curriculum level
        assert main.get("convergence_mode", cfg.curriculum.convergence_mode) == "champion"


# ===================================================================
# Self-play source / convergence_mode decoupling tests
# ===================================================================

class TestResolveSelfPlaySource:
    """Tests for `_resolve_self_play_source` — splits self-play data
    source from curriculum convergence detection.

    Background: HeXO originally coupled the two (`convergence_mode in
    ('champion', 'sprt')` forced champion-as-self-play-source). Modern
    AZ practice (Silver 2018, Lc0, KataGo) keeps them separate so the
    live trainee can continuously generate games while a separate
    SPRT/eval pipeline judges promotions. See memory
    `project_champion_mode_history` for context.
    """

    def test_default_source_is_auto(self):
        """The shipped default is 'auto' to preserve legacy behavior."""
        cfg = SelfPlayConfig()
        assert cfg.source == "auto"

    def test_auto_with_sprt_resolves_to_champion(self):
        cfg = SelfPlayConfig(source="auto")
        with pytest.warns(DeprecationWarning, match="self_play.source defaulted to 'champion'"):
            assert _resolve_self_play_source(cfg, "sprt") == "champion"

    def test_auto_with_champion_resolves_to_champion(self):
        cfg = SelfPlayConfig(source="auto")
        with pytest.warns(DeprecationWarning, match="self_play.source defaulted to 'champion'"):
            assert _resolve_self_play_source(cfg, "champion") == "champion"

    def test_auto_with_random_resolves_to_trainee(self):
        cfg = SelfPlayConfig(source="auto")
        with pytest.warns(DeprecationWarning, match="self_play.source defaulted to 'trainee'"):
            assert _resolve_self_play_source(cfg, "random") == "trainee"

    def test_auto_with_checkpoint_resolves_to_trainee(self):
        cfg = SelfPlayConfig(source="auto")
        with pytest.warns(DeprecationWarning, match="self_play.source defaulted to 'trainee'"):
            assert _resolve_self_play_source(cfg, "checkpoint") == "trainee"

    def test_auto_with_none_resolves_to_trainee(self):
        cfg = SelfPlayConfig(source="auto")
        with pytest.warns(DeprecationWarning, match="self_play.source defaulted to 'trainee'"):
            assert _resolve_self_play_source(cfg, "none") == "trainee"

    def test_explicit_champion_with_sprt_no_warning(self, recwarn):
        cfg = SelfPlayConfig(source="champion")
        assert _resolve_self_play_source(cfg, "sprt") == "champion"
        deprecation_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
        assert deprecation_warnings == []

    def test_explicit_champion_with_champion_mode_no_warning(self, recwarn):
        cfg = SelfPlayConfig(source="champion")
        assert _resolve_self_play_source(cfg, "champion") == "champion"
        deprecation_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
        assert deprecation_warnings == []

    def test_explicit_trainee_with_sprt_no_warning(self, recwarn):
        cfg = SelfPlayConfig(source="trainee")
        assert _resolve_self_play_source(cfg, "sprt") == "trainee"
        deprecation_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
        assert deprecation_warnings == []

    def test_explicit_trainee_with_random_no_warning(self, recwarn):
        cfg = SelfPlayConfig(source="trainee")
        assert _resolve_self_play_source(cfg, "random") == "trainee"
        deprecation_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
        assert deprecation_warnings == []

    def test_explicit_trainee_with_champion_no_warning(self, recwarn):
        """Trainee source is always valid — independent of convergence_mode."""
        cfg = SelfPlayConfig(source="trainee")
        assert _resolve_self_play_source(cfg, "champion") == "trainee"
        deprecation_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
        assert deprecation_warnings == []

    def test_explicit_champion_with_random_raises(self):
        cfg = SelfPlayConfig(source="champion")
        with pytest.raises(ValueError, match="requires convergence_mode"):
            _resolve_self_play_source(cfg, "random")

    def test_explicit_champion_with_checkpoint_raises(self):
        cfg = SelfPlayConfig(source="champion")
        with pytest.raises(ValueError, match="requires convergence_mode"):
            _resolve_self_play_source(cfg, "checkpoint")

    def test_explicit_champion_with_none_raises(self):
        cfg = SelfPlayConfig(source="champion")
        with pytest.raises(ValueError, match="requires convergence_mode"):
            _resolve_self_play_source(cfg, "none")

    def test_unknown_source_raises(self):
        cfg = SelfPlayConfig(source="bogus")
        with pytest.raises(ValueError, match="must be one of"):
            _resolve_self_play_source(cfg, "sprt")


# ===================================================================
# Prev-checkpoint plateau detector (SPRT secondary signal)
# ===================================================================

class TestWilsonUpper:
    """`_wilson_upper` — one-sided Wilson score upper bound used by the
    pooled prev-checkpoint plateau detector."""

    def test_known_value_half_at_800(self):
        # p̂=0.5, n=800, z=1.645 → ≈ 0.529 (hand-computed).
        u = _wilson_upper(0.5, 800)
        assert abs(u - 0.5290) < 1e-3

    def test_tightens_with_more_evidence(self):
        assert _wilson_upper(0.5, 800) < _wilson_upper(0.5, 80) < _wilson_upper(0.5, 8)

    def test_degenerate_p_one_is_exactly_one(self):
        assert _wilson_upper(1.0, 10) == pytest.approx(1.0)

    def test_p_zero_has_positive_upper(self):
        assert 0.0 < _wilson_upper(0.0, 100) < 0.05


class TestPrevCkptPlateauDetector:
    """Unit tests for `_prev_ckpt_plateau_fired` — POOLED statistic over the
    rolling window of prev-checkpoint win rates: fires when the one-sided
    Wilson upper bound on the pooled win rate is confidently below
    `band_high` (and the pooled mean is not a regression below `band_low`).
    Replaces the old all-N-in-band AND-gate, which reset on a single noisy
    excursion and discarded ~95% of the games' evidence.
    """

    def test_disabled_when_patience_is_zero(self):
        window = [0.50] * 100
        assert not _prev_ckpt_plateau_fired(
            window, patience=0, band_low=0.45, band_high=0.55, games_per_eval=100)

    def test_fires_when_flat_with_enough_evidence(self):
        # 8 evals × 100 games at exactly 0.5: pooled upper ≈ 0.529 < 0.55.
        window = [0.50] * 8
        assert _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.45, band_high=0.55, games_per_eval=100)

    def test_does_not_fire_when_window_not_yet_full(self):
        window = [0.50] * 5
        assert not _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.45, band_high=0.55, games_per_eval=100)

    def test_single_excursion_no_longer_resets(self):
        # One 0.62 eval among 0.50s: pooled mean 0.515, upper ≈ 0.544 < 0.55
        # — still fires. Under the old AND-gate this excursion reset the
        # whole window; pooling absorbs it as the noise it is. (Only a
        # TRAILING excursion defers, via the recency guard — see
        # test_recovery_shape_withheld_by_recency_guard.)
        window = [0.50] * 5 + [0.62] + [0.50] * 2
        assert _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.45, band_high=0.55, games_per_eval=100)

    def test_trailing_excursion_defers_one_eval(self):
        # The same excursion in the freshest slot is indistinguishable from
        # recovery-onset — withheld now, fires next eval if it was noise.
        window = [0.50] * 7 + [0.62]
        assert not _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.45, band_high=0.55, games_per_eval=100)

    def test_does_not_fire_while_still_improving(self):
        # Sustained 0.58 vs the lagged checkpoint = real improvement;
        # pooled upper ≈ 0.608 > 0.55.
        window = [0.58] * 8
        assert not _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.45, band_high=0.55, games_per_eval=100)

    def test_regression_below_band_low_does_not_fire(self):
        # Regression is a separate alarm, not a plateau — keep training.
        window = [0.30] * 8
        assert not _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.45, band_high=0.55, games_per_eval=100)

    def test_insufficient_games_per_eval_withholds_fire(self):
        # Same flat 0.50 window but only 10 games/eval (n=80): upper ≈ 0.59
        # — not enough evidence to call it confidently below 0.55.
        window = [0.50] * 8
        assert not _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.45, band_high=0.55, games_per_eval=10)

    def test_negative_patience_treated_as_disabled(self):
        window = [0.50] * 10
        assert not _prev_ckpt_plateau_fired(
            window, patience=-1, band_low=0.45, band_high=0.55, games_per_eval=100)

    def test_nonpositive_games_per_eval_does_not_crash(self):
        window = [0.50] * 8
        # Degenerate config: treated as 1 game/eval → far too little evidence.
        assert not _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.45, band_high=0.55, games_per_eval=0)

    def test_band_low_boundary_inclusive(self):
        # Pooled mean exactly at band_low still counts (>=), and with enough
        # evidence the upper bound is below band_high.
        window = [0.45] * 8
        assert _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.45, band_high=0.55, games_per_eval=100)

    def test_recovery_shape_withheld_by_recency_guard(self):
        # The live 2026-06-12 false positive: post-LR-reset dip drags the
        # pooled mean into firing range while the freshest eval shows the
        # trainee improving again (> band_high) — mid-recovery, not plateau.
        window = [0.675, 0.51, 0.59, 0.70, 0.43, 0.50, 0.48, 0.65]
        assert not _prev_ckpt_plateau_fired(
            window, patience=8, band_low=0.40, band_high=0.60, games_per_eval=100)
        # Same pooled mean but with the recovery at the OLD end and the
        # freshest reading flat: a genuine settling-down — fires.
        settled = [0.65, 0.675, 0.70, 0.59, 0.51, 0.50, 0.48, 0.43]
        assert _prev_ckpt_plateau_fired(
            settled, patience=8, band_low=0.40, band_high=0.60, games_per_eval=100)


# ===================================================================
# Promotion-velocity plateau detector (SPRT primary signal)
# ===================================================================

class TestPromoVelocityDetector:
    """`_promo_velocity_fired` — declares plateau when the current
    promotion drought exceeds `factor` × the stage's median promotion
    interval. Each SPRT promotion is an α-certified improvement event, so
    intervals between them are certified-Elo-velocity quanta — far less
    noisy than counting variable-length reject rounds."""

    def test_disabled_when_factor_is_zero(self):
        assert not _promo_velocity_fired(
            [3000, 4000, 5000], steps_since_promotion=10**9,
            factor=0.0, min_intervals=3)

    def test_disarmed_below_min_intervals(self):
        assert not _promo_velocity_fired(
            [3000, 4000], steps_since_promotion=10**9,
            factor=3.0, min_intervals=3)

    def test_fires_beyond_factor_times_median(self):
        # median = 4000, factor 3 → threshold 12000 (strictly greater).
        intervals = [3000, 4000, 5000]
        assert not _promo_velocity_fired(
            intervals, steps_since_promotion=12000, factor=3.0, min_intervals=3)
        assert _promo_velocity_fired(
            intervals, steps_since_promotion=12001, factor=3.0, min_intervals=3)

    def test_median_robust_to_one_slow_cycle(self):
        # One historical 50k-step drought must not inflate the threshold the
        # way max-based calibration (the reject-patience ratchet) does.
        intervals = [3000, 4000, 50000]
        assert _promo_velocity_fired(
            intervals, steps_since_promotion=12001, factor=3.0, min_intervals=3)

    def test_min_intervals_floor_of_one(self):
        # min_intervals <= 0 is clamped to 1 — never fire on zero history.
        assert not _promo_velocity_fired(
            [], steps_since_promotion=10**9, factor=3.0, min_intervals=0)
        assert _promo_velocity_fired(
            [4000], steps_since_promotion=12001, factor=3.0, min_intervals=0)


# ===================================================================
# Detector state persistence (survives trainer restarts)
# ===================================================================

class TestDetectorStatePersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "detector_state.json"
        _save_detector_state(
            path, stage_idx=2,
            promo_intervals=[3000, 4500],
            velocity_prev_promo_step=57050,
            prev_ckpt_window=[0.52, 0.49, 0.55],
            last_promotion_step=57340,
        )
        state = _load_detector_state(path, stage_idx=2, current_step=62700)
        assert state["promo_intervals"] == [3000, 4500]
        assert state["velocity_prev_promo_step"] == 57050
        assert state["prev_ckpt_window"] == [0.52, 0.49, 0.55]
        assert state["last_promotion_step"] == 57340

    def test_missing_file_is_empty(self, tmp_path):
        state = _load_detector_state(
            tmp_path / "nope.json", stage_idx=0, current_step=100)
        assert state["promo_intervals"] == []
        assert state["velocity_prev_promo_step"] is None
        assert state["prev_ckpt_window"] == []
        assert state["last_promotion_step"] is None

    def test_future_last_promotion_event_discards(self, tmp_path):
        # Same rolled-back-run guard as velocity_prev_promo_step, but for
        # the promotion EVENT anchor (drives sprt/steps_since_promotion).
        path = tmp_path / "detector_state.json"
        _save_detector_state(
            path, stage_idx=0, promo_intervals=[3000],
            velocity_prev_promo_step=50000, prev_ckpt_window=[0.5],
            last_promotion_step=70000,
        )
        state = _load_detector_state(path, stage_idx=0, current_step=62000)
        assert state["last_promotion_step"] is None
        assert state["promo_intervals"] == []

    def test_stage_mismatch_discards(self, tmp_path):
        # A stale file from a previous stage occupying the same log dir
        # (or a copied run dir) must not seed the new stage's detectors.
        path = tmp_path / "detector_state.json"
        _save_detector_state(
            path, stage_idx=1, promo_intervals=[3000],
            velocity_prev_promo_step=50000, prev_ckpt_window=[0.5],
        )
        state = _load_detector_state(path, stage_idx=2, current_step=60000)
        assert state["promo_intervals"] == []
        assert state["velocity_prev_promo_step"] is None
        assert state["prev_ckpt_window"] == []

    def test_future_promotion_step_discards(self, tmp_path):
        # Rolled-back run: recorded last promotion is AHEAD of the resumed
        # step counter — the whole record is untrustworthy.
        path = tmp_path / "detector_state.json"
        _save_detector_state(
            path, stage_idx=0, promo_intervals=[3000],
            velocity_prev_promo_step=70000, prev_ckpt_window=[0.5],
        )
        state = _load_detector_state(path, stage_idx=0, current_step=62000)
        assert state["promo_intervals"] == []
        assert state["velocity_prev_promo_step"] is None
        assert state["prev_ckpt_window"] == []

    def test_corrupt_file_is_empty(self, tmp_path):
        path = tmp_path / "detector_state.json"
        path.write_text("{not json")
        state = _load_detector_state(path, stage_idx=0, current_step=100)
        assert state["promo_intervals"] == []

    def test_save_is_atomic_no_partial_on_existing(self, tmp_path):
        # Saving over an existing file goes through a tmp+rename, so a
        # reader never sees a partial file (best-effort check: tmp file
        # cleaned up, content complete).
        path = tmp_path / "detector_state.json"
        _save_detector_state(path, stage_idx=0, promo_intervals=[1],
                             velocity_prev_promo_step=10, prev_ckpt_window=[])
        _save_detector_state(path, stage_idx=0, promo_intervals=[1, 2],
                             velocity_prev_promo_step=20, prev_ckpt_window=[0.5])
        import json as _json
        data = _json.loads(path.read_text())
        assert data["promo_intervals"] == [1, 2]
        assert not list(tmp_path.glob("*.tmp"))


# ===================================================================
# Corpus eval runner (per-champion human-corpus yardstick)
# ===================================================================

class TestCorpusRecordScalars:
    def test_extracts_tb_scalars(self):
        record = {
            "policy": {"n": 100, "perplexity": 71.8, "top1": 0.19, "top16": 0.643},
            "value": {"n": 50, "mse": 1.188, "sign_acc": 0.547, "ece": 0.409},
        }
        scalars = _corpus_record_scalars(record)
        assert scalars == {
            "eval/corpus_policy_perplexity": 71.8,
            "eval/corpus_policy_top1": 0.19,
            "eval/corpus_policy_top16": 0.643,
            "eval/corpus_value_mse": 1.188,
            "eval/corpus_value_sign_acc": 0.547,
            "eval/corpus_value_ece": 0.409,
        }

    def test_skips_none_metrics(self):
        # Empty-bucket records carry None — never emit a None to TB.
        record = {
            "policy": {"n": 0, "perplexity": None, "top1": None, "top16": None},
            "value": {"n": 0, "mse": None, "sign_acc": None, "ece": 0.0},
        }
        scalars = _corpus_record_scalars(record)
        assert scalars == {"eval/corpus_value_ece": 0.0}


class TestCorpusEvalRunner:
    @staticmethod
    def _stub_script(tmp_path, body: str):
        """A stand-in for eval_human_corpus.py with the same CLI surface."""
        script = tmp_path / "stub_eval.py"
        script.write_text(textwrap.dedent(body))
        return script

    def _make_runner(self, tmp_path, body: str, games: int = 100):
        return CorpusEvalRunner(
            script_path=self._stub_script(tmp_path, body),
            results_path=tmp_path / "results.jsonl",
            games=games,
            device="cpu",
            log_path=tmp_path / "corpus_eval.log",
        )

    SUCCESS_BODY = """\
        import argparse, json
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint"); p.add_argument("--device")
        p.add_argument("--results"); p.add_argument("--max-games", type=int)
        a = p.parse_args()
        rec = {"checkpoint": a.checkpoint, "n_games": a.max_games,
               "policy": {"perplexity": 50.0, "top1": 0.2, "top16": 0.6},
               "value": {"mse": 1.0, "sign_acc": 0.55, "ece": 0.2}}
        with open(a.results, "a") as f:
            f.write(json.dumps(rec) + "\\n")
    """

    def test_disabled_when_games_zero(self, tmp_path):
        runner = self._make_runner(tmp_path, self.SUCCESS_BODY, games=0)
        assert not runner.enabled
        assert not runner.maybe_launch("ckpt.pt", step=100)

    def test_launch_poll_roundtrip(self, tmp_path):
        runner = self._make_runner(tmp_path, self.SUCCESS_BODY)
        assert runner.maybe_launch("some_ckpt.pt", step=1234)
        deadline = __import__("time").time() + 30
        result = None
        while result is None and __import__("time").time() < deadline:
            result = runner.poll()
        assert result is not None
        step, scalars = result
        assert step == 1234
        assert scalars["eval/corpus_policy_top16"] == 0.6
        assert scalars["eval/corpus_value_ece"] == 0.2
        # A second poll must not re-emit the same record.
        assert runner.poll() is None

    def test_skips_launch_while_running(self, tmp_path):
        body = "import time\ntime.sleep(60)\n"
        runner = self._make_runner(tmp_path, body)
        assert runner.maybe_launch("a.pt", step=1)
        assert not runner.maybe_launch("b.pt", step=2)  # still running → skip
        runner.shutdown()
        assert runner.poll() is None  # terminated runs emit nothing

    def test_failed_run_emits_nothing(self, tmp_path):
        runner = self._make_runner(tmp_path, "import sys\nsys.exit(3)\n")
        assert runner.maybe_launch("a.pt", step=7)
        deadline = __import__("time").time() + 30
        while runner._proc is not None and __import__("time").time() < deadline:
            if runner.poll() is not None:
                raise AssertionError("failed run must not produce scalars")
        assert runner.poll() is None

    def test_missing_script_disables(self, tmp_path):
        runner = CorpusEvalRunner(
            script_path=tmp_path / "does_not_exist.py",
            results_path=tmp_path / "results.jsonl",
            games=100, device="cpu", log_path=None,
        )
        assert not runner.enabled
        assert not runner.maybe_launch("ckpt.pt", step=1)

    def test_failed_launch_is_contained(self, tmp_path):
        # Telemetry-only feature: a launch failure (here: log path is a
        # directory, so open("ab") raises) must return False, leak no open
        # handle, and leave the runner ready for the next promotion.
        script = self._stub_script(tmp_path, self.SUCCESS_BODY)
        bad_log = tmp_path / "log_is_a_dir"
        bad_log.mkdir()
        runner = CorpusEvalRunner(
            script_path=script,
            results_path=tmp_path / "results.jsonl",
            games=10, device="cpu", log_path=bad_log,
        )
        assert not runner.maybe_launch("a.pt", step=5)
        assert runner._log_handle is None
        assert not runner.running
        # Recovers: a later launch with the same runner still works after
        # the obstruction is gone.
        bad_log.rmdir()
        assert runner.maybe_launch("a.pt", step=6)
        runner.shutdown()

    def test_stage_overrides_resolve_for_new_knobs(self, tmp_path):
        # Per-stage overrides on the new SPRT secondary-detector fields
        # must take precedence over curriculum-level defaults, matching
        # the resolution pattern used in `run_curriculum`'s stage loop.
        toml = """\
        [curriculum]
        convergence_mode = "sprt"
        sprt_prev_ckpt_patience = 5
        sprt_prev_ckpt_band_low = 0.45
        sprt_prev_ckpt_band_high = 0.55

        [model]
        hidden_dim = 16

        [game]
        win_length = 3
        placement_radius = 2

        [[curriculum.stages]]
        name = "S1"
        win_length = 3
        placement_radius = 2
        sprt_prev_ckpt_patience = 12
        sprt_prev_ckpt_band_high = 0.53
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        stage = cfg.curriculum.stages[0]

        sprt_prev_ckpt_patience = stage.get(
            "sprt_prev_ckpt_patience", cfg.curriculum.sprt_prev_ckpt_patience
        )
        sprt_prev_ckpt_band_low = stage.get(
            "sprt_prev_ckpt_band_low", cfg.curriculum.sprt_prev_ckpt_band_low
        )
        sprt_prev_ckpt_band_high = stage.get(
            "sprt_prev_ckpt_band_high", cfg.curriculum.sprt_prev_ckpt_band_high
        )

        assert sprt_prev_ckpt_patience == 12      # stage override
        assert sprt_prev_ckpt_band_high == 0.53   # stage override
        assert sprt_prev_ckpt_band_low == 0.45    # falls back to curriculum default

    def test_velocity_and_corpus_knobs_load(self, tmp_path):
        # New detector knobs resolve through [curriculum] / [eval.corpus],
        # with safe defaults (both detectors off) when absent.
        toml = """\
        [curriculum]
        convergence_mode = "sprt"
        sprt_promo_velocity_factor = 2.5

        [eval.corpus]
        games = 500
        device = "cpu"

        [[curriculum.stages]]
        name = "S1"
        win_length = 3
        placement_radius = 2
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.curriculum.sprt_promo_velocity_factor == 2.5
        assert cfg.curriculum.sprt_promo_velocity_min_intervals == 3  # default
        assert cfg.eval.corpus.games == 500
        assert cfg.eval.corpus.device == "cpu"
        # Defaults: both off.
        from hexo_a0.config import CorpusEvalConfig, CurriculumConfig
        assert CurriculumConfig().sprt_promo_velocity_factor == 0.0
        assert CorpusEvalConfig().games == 0

    def test_corpus_per_stage_override_resolves(self, tmp_path):
        # eval_corpus_games as a [[curriculum.stages]] key must route through
        # _CONFIG_TREE to cfg.eval.corpus.games (the sealbot analog works;
        # silently dropping this override was a review finding).
        toml = """\
        [curriculum]
        convergence_mode = "sprt"

        [eval.corpus]
        games = 2000

        [[curriculum.stages]]
        name = "S1"
        win_length = 3
        placement_radius = 2
        eval_corpus_games = 50
        eval_games = 33
        """
        path = _write_toml(tmp_path, toml)
        cfg = load_config(path)
        stage_cfg = _build_stage_config(cfg, cfg.curriculum.stages[0])
        assert stage_cfg.eval.games == 33
        assert stage_cfg.eval.corpus.games == 50
