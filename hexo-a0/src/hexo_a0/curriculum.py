"""Automatic curriculum trainer for HeXO AlphaZero.

Runs through a sequence of stage configs, advancing when the model's
win rate vs random exceeds a threshold for a patience window.

Curriculum TOML format:
  - Shared sections ([model], [mcts], [training], [self_play], [eval], [run])
    define defaults that apply to all stages.
  - [[curriculum.stages]] entries override per-stage values (game params, lr, etc.).
  - The model section is fixed across stages (architecture doesn't change).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import statistics
import warnings
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Literal

import torch

from hexo_a0.config import (
    CorpusEvalConfig,
    EvalConfig,
    FullConfig,
    GameConfig,
    MCTSConfig,
    ModelConfig,
    PythonSelfPlayConfig,
    RunConfig,
    RustSelfPlayConfig,
    SealbotConfig,
    SelfPlayConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


_CHAMPION_REQUIRED_MODES = ("champion", "sprt")


def _resolve_self_play_source(
    self_play_cfg: SelfPlayConfig,
    convergence_mode: str,
) -> Literal["champion", "trainee"]:
    """Resolve the self-play data source from config + curriculum convergence mode.

    The self-play source and the convergence-detection mechanism are
    independent concerns: convergence_mode determines *how* the trainee
    is judged stronger than the champion, while self_play.source
    determines *which* network actually generates games for the replay
    buffer. AGZ (Silver 2017) coupled them; AZ (Silver 2018) and modern
    AZ-likes (Lc0, KataGo) keep them separate. See memory
    `project_champion_mode_history` for background.

    Resolution rules:
      - "champion": validated only if convergence_mode supports a
        champion file (i.e. is "champion" or "sprt"). ValueError otherwise.
      - "trainee": always valid; works with any convergence_mode.
      - "auto": legacy derivation — "champion" iff convergence_mode is
        "champion" or "sprt", else "trainee". Emits a DeprecationWarning.
    """
    source = self_play_cfg.source
    if source == "trainee":
        return "trainee"
    if source == "champion":
        if convergence_mode not in _CHAMPION_REQUIRED_MODES:
            raise ValueError(
                f"self_play.source='champion' requires convergence_mode in "
                f"{_CHAMPION_REQUIRED_MODES!r}; got convergence_mode={convergence_mode!r}. "
                "Either change convergence_mode or use self_play.source='trainee'."
            )
        return "champion"
    if source == "auto":
        resolved: Literal["champion", "trainee"] = (
            "champion" if convergence_mode in _CHAMPION_REQUIRED_MODES else "trainee"
        )
        warnings.warn(
            f"self_play.source defaulted to {resolved!r} based on "
            f"convergence_mode={convergence_mode!r}. This auto-derivation "
            "will be removed in a future release; the future default will "
            "be 'trainee'. Set self_play.source explicitly to preserve "
            "current behavior or opt into the new default.",
            DeprecationWarning,
            stacklevel=2,
        )
        return resolved
    raise ValueError(
        f"self_play.source must be one of 'champion', 'trainee', 'auto'; got {source!r}."
    )


# Config tree for stage override routing.  Each node is
#   (dataclass_type, {child_attr: (child_type, grandchildren)})
#
# Stage keys are resolved by prefix-walking this tree:
#   eval_sealbot_games  ->  cfg.eval.sealbot.games
#   eval_games          ->  cfg.eval.games
#   game_win_length     ->  cfg.game.win_length
#   rust_max_batch      ->  cfg.self_play.rust.max_batch  (legacy shorthand)
#
# Bare keys (win_length, n_simulations) still work for backwards compat.
_CONFIG_TREE: dict[str, tuple[type, dict]] = {
    "game": (GameConfig, {}),
    "mcts": (MCTSConfig, {}),
    "training": (TrainingConfig, {}),
    "eval": (EvalConfig, {
        "sealbot": (SealbotConfig, {}),
        "corpus": (CorpusEvalConfig, {}),
    }),
    "self_play": (SelfPlayConfig, {
        "rust": (RustSelfPlayConfig, {}),
        "python": (PythonSelfPlayConfig, {}),
    }),
    "run": (RunConfig, {}),
}

# Legacy shorthand prefixes that skip intermediate levels:
#   sealbot_games  ->  eval_sealbot_games
#   rust_max_batch ->  self_play_rust_max_batch
_LEGACY_PREFIXES: dict[str, str] = {
    "sealbot_": "eval_sealbot_",
    "rust_": "self_play_rust_",
    "python_": "self_play_python_",
}


def _resolve_overrides(
    stage: dict, tree: dict[str, tuple[type, dict]],
) -> dict[tuple[str, ...], dict[str, object]]:
    """Walk the config tree to resolve each stage key to a config path.

    Returns ``{path: {field: value}}`` where path is a tuple of attr
    names, e.g. ``("eval", "sealbot")`` for ``eval_sealbot_games``.
    """
    result: dict[tuple[str, ...], dict[str, object]] = {}

    # Normalise legacy prefixes first
    normalised: dict[str, object] = {}
    for k, v in stage.items():
        for old, new in _LEGACY_PREFIXES.items():
            if k.startswith(old) and not k.startswith(new):
                k = new + k[len(old):]
                break
        normalised[k] = v

    def _try_resolve(key: str, value: object, node: dict, path: tuple[str, ...]) -> bool:
        """Try to match key against node's children (prefix) or fields (leaf)."""
        for attr, (cls, children) in node.items():
            prefix = attr + "_"
            if key.startswith(prefix):
                rest = key[len(prefix):]
                # Try children first (deeper match wins)
                if children and _try_resolve(rest, value, children, path + (attr,)):
                    return True
                # Leaf match against this node's fields
                fields = {f.name for f in dc_fields(cls)}
                if rest in fields:
                    result.setdefault(path + (attr,), {})[rest] = value
                    return True
        return False

    for key, value in normalised.items():
        if _try_resolve(key, value, tree, ()):
            continue
        # Bare key fallback: scan all top-level configs
        for attr, (cls, _children) in tree.items():
            fields = {f.name for f in dc_fields(cls)}
            if key in fields:
                result.setdefault((attr,), {})[key] = value
                break

    return result


#: One-sided 95% confidence for the pooled plateau test.
_WILSON_Z = 1.645


def _wilson_upper(p_hat: float, n: int, z: float = _WILSON_Z) -> float:
    """One-sided Wilson score upper bound for a binomial proportion."""
    if n <= 0:
        return 1.0
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p_hat + z2 / (2.0 * n)
    spread = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return (centre + spread) / denom


def _prev_ckpt_plateau_fired(
    window,
    patience: int,
    band_low: float,
    band_high: float,
    games_per_eval: int,
) -> bool:
    """Return True if the pooled prev-checkpoint plateau condition is met.

    Pools the rolling window (``patience`` evals × ``games_per_eval`` games,
    equal weight per eval) and fires when the one-sided Wilson upper bound
    on the pooled win rate is below ``band_high`` — i.e. the trainee is
    *confidently* not improving at the band_high rate over the checkpoint
    lag. A pooled mean below ``band_low`` is a regression (a separate
    signal, not plateau) and withholds the declaration. Pooling replaces
    the old all-N-in-band AND-gate, which reset on a single noisy excursion
    and used only a fraction of the games' evidence. ``patience=0``
    disables.

    Draws enter eval win rates as fractional scores, and the window pools
    evals against successive (drifting) lagged checkpoints, so the binomial
    model is an approximation — mildly anti-conservative under fast opponent
    drift. At HeXO's ~0% draw rate and one-checkpoint-per-eval drift the
    distinction is moot.
    """
    if patience <= 0:
        return False
    if len(window) < patience:
        return False
    pooled = sum(window) / len(window)
    if pooled < band_low:
        return False
    # Recency guard: a window whose FRESHEST eval clears band_high is mid-
    # recovery, not plateaued (e.g. the post-LR-reset V shape: the dip drags
    # the pooled mean down while the latest evals show the trainee improving
    # again). Observed live 2026-06-12: window
    # [.675,.51,.59,.70,.43,.50,.48,.65] pooled to 0.567 and fired while the
    # trainee was beating its pre-reset peak 65-35 at the newest eval.
    if window[-1] > band_high:
        return False
    n = len(window) * max(int(games_per_eval), 1)
    return _wilson_upper(pooled, n) < band_high


def _promo_velocity_fired(
    intervals,
    steps_since_promotion: int,
    factor: float,
    min_intervals: int,
) -> bool:
    """Return True when the promotion drought exceeds ``factor`` × the
    median promotion interval observed this stage.

    Each SPRT promotion is an alpha-certified >= s1 improvement over a
    ratcheting champion, so the intervals between promotions are certified
    Elo-velocity quanta. The median (not max) keeps one historically slow
    cycle from ratcheting the threshold the way peak_history-max
    calibration does. ``factor=0`` disables; fewer than ``min_intervals``
    completed intervals (min 1) leaves the detector disarmed.
    """
    if factor <= 0:
        return False
    if len(intervals) < max(min_intervals, 1):
        return False
    return steps_since_promotion > factor * statistics.median(intervals)


def _load_detector_state(path: Path, stage_idx: int, current_step: int) -> dict:
    """Load persisted plateau-detector state, or an empty state.

    Lives in the stage log dir (next to sprt_state.json) so each stage's
    file is naturally separate. The record is discarded wholesale when it
    can't be trusted: wrong stage_idx (stale/copied run dir) or a recorded
    last-promotion step ahead of the resumed trainer (rolled-back run).
    Never raises — detector state is reconstructible, resume must not be.
    """
    empty = {
        "promo_intervals": [],
        "velocity_prev_promo_step": None,
        "prev_ckpt_window": [],
        "last_promotion_step": None,
    }
    try:
        data = json.loads(Path(path).read_text())
        if data.get("stage_idx") != stage_idx:
            return empty
        prev = data.get("velocity_prev_promo_step")
        if prev is not None and int(prev) > current_step:
            return empty
        last_promo = data.get("last_promotion_step")
        if last_promo is not None and int(last_promo) > current_step:
            return empty
        return {
            "promo_intervals": [int(x) for x in data.get("promo_intervals", [])],
            "velocity_prev_promo_step": int(prev) if prev is not None else None,
            "prev_ckpt_window": [float(x) for x in data.get("prev_ckpt_window", [])],
            "last_promotion_step": int(last_promo) if last_promo is not None else None,
        }
    except Exception:
        return empty


def _save_detector_state(
    path: Path,
    stage_idx: int,
    promo_intervals: list,
    velocity_prev_promo_step: int | None,
    prev_ckpt_window,
    last_promotion_step: int | None = None,
) -> None:
    """Persist plateau-detector state (atomic tmp+rename; never raises).

    ``last_promotion_step`` is the promotion EVENT step (trainer step when
    accept_h1 fired) — more precise than champion.pt's train_steps (the
    validated checkpoint's step, which lags by the validation delay), so
    sprt/steps_since_promotion resumes without a discontinuity.
    """
    try:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "stage_idx": stage_idx,
            "promo_intervals": list(promo_intervals),
            "velocity_prev_promo_step": velocity_prev_promo_step,
            "prev_ckpt_window": list(prev_ckpt_window),
            "last_promotion_step": last_promotion_step,
        }))
        tmp.replace(path)
    except Exception as e:
        logger.warning("detector state save failed (%s): %s", path, e)


#: TB info-card descriptions for the corpus scalars (rendered by
#: DescribedWriter; remembered per tag after the first described write).
_CORPUS_SCALAR_DESCRIPTIONS = {
    "eval/corpus_policy_top16": (
        "Fraction of human moves inside the raw prior's top-16 on the "
        "fixed-seed human-corpus sample — literal m_actions=16 candidate "
        "coverage for Gumbel search. Primary corpus strength signal; its "
        "slope across promotions measures transferable full-board gains "
        "from the current stage."
    ),
    "eval/corpus_policy_top1": (
        "Fraction of corpus positions where the raw prior's argmax equals "
        "the human move. Strength proxy while below human level; expected "
        "to decline once the bot prefers better moves than humans played."
    ),
    "eval/corpus_policy_perplexity": (
        "exp(mean NLL) of human moves under the raw prior. CAUTION: "
        "NLL punishes confident disagreement, so sharper/stronger nets "
        "score WORSE (champion-mini ~100 vs a weaker net's ~72). Read as "
        "a sharpness diagnostic, not a strength signal — use top16 + the "
        "value metrics for strength."
    ),
    "eval/corpus_value_mse": (
        "Value head MSE vs final outcome (±1) at end-of-turn states in "
        "human games. Lower = better; carries irreducible noise from "
        "human blunders after the measured position."
    ),
    "eval/corpus_value_sign_acc": (
        "P(value prediction has the winning side's sign) at end-of-turn "
        "states. 0.5 = coin flip; practical ceiling ~0.65-0.75 (humans "
        "throw won games) — strongest bot measured 0.594."
    ),
    "eval/corpus_value_ece": (
        "Expected calibration error: |mean predicted v − mean outcome| "
        "weighted over 10 prediction bins; 0 = perfectly calibrated. "
        "Cleanest full-board value-head health signal — OOD radius "
        "inflates it (S1-only ~0.41 vs r6-trained ~0.16)."
    ),
}


def _corpus_record_scalars(record: dict) -> dict[str, float]:
    """TB scalars from one eval_human_corpus.py summary record.

    None-valued metrics (empty buckets) are skipped, never emitted.
    """
    po = record.get("policy", {})
    vo = record.get("value", {})
    raw = {
        "eval/corpus_policy_perplexity": po.get("perplexity"),
        "eval/corpus_policy_top1": po.get("top1"),
        "eval/corpus_policy_top16": po.get("top16"),
        "eval/corpus_value_mse": vo.get("mse"),
        "eval/corpus_value_sign_acc": vo.get("sign_acc"),
        "eval/corpus_value_ece": vo.get("ece"),
    }
    return {k: float(v) for k, v in raw.items() if v is not None}


class CorpusEvalRunner:
    """Runs the human-corpus eval against each freshly promoted champion.

    Spawns ``scripts/eval_human_corpus.py`` as a niced background subprocess
    (CPU by default — see ``[eval.corpus]``), polls non-blockingly from the
    curriculum loop, and yields TB-ready scalars when a run completes. At
    most one run at a time: a promotion landing mid-run is skipped (the
    next one gets evaluated), keeping the cadence self-limiting.
    """

    def __init__(
        self,
        script_path: Path,
        results_path: Path,
        games: int,
        device: str,
        log_path: Path | None,
    ):
        self.script_path = Path(script_path)
        self.results_path = Path(results_path)
        self.games = int(games)
        self.device = device
        self.log_path = Path(log_path) if log_path is not None else None
        self._proc = None
        self._step: int | None = None
        self._log_handle = None
        if self.games > 0 and not self.script_path.exists():
            logger.warning(
                "corpus eval enabled (games=%d) but script missing at %s — disabled",
                self.games, self.script_path,
            )

    @property
    def enabled(self) -> bool:
        return self.games > 0 and self.script_path.exists()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def maybe_launch(self, checkpoint_path: str, step: int) -> bool:
        """Start an eval of ``checkpoint_path`` unless disabled or busy."""
        if not self.enabled:
            return False
        if self._proc is not None:
            # Running, or finished but not yet harvested by poll() — either
            # way launching now would clobber state (a finished run's scalars
            # would be silently dropped). Skip; the cadence is self-limiting.
            logger.info(
                "corpus eval for step %s still pending; skipping launch for step %d",
                self._step, step,
            )
            return False
        import subprocess
        import sys

        # `nice` prefix instead of a preexec_fn: keeps Popen on the
        # posix_spawn fast path (no fork of the multi-GiB trainer, no
        # thread-safety caveat) while still deprioritising the eval.
        cmd = [
            "nice", "-n", "10",
            sys.executable, str(self.script_path),
            "--checkpoint", str(checkpoint_path),
            "--device", self.device,
            "--results", str(self.results_path),
            "--max-games", str(self.games),
        ]
        # Telemetry-only feature: a failed launch must never take the
        # curriculum loop down with it.
        try:
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_handle = self.log_path.open("ab")
                out = self._log_handle
            else:
                out = subprocess.DEVNULL
            self._proc = subprocess.Popen(
                cmd, stdout=out, stderr=subprocess.STDOUT,
            )
        except Exception as e:
            logger.warning("corpus eval launch for step %d failed: %s", step, e)
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            self._proc = None
            return False
        self._step = step
        logger.info(
            "corpus eval launched for step %d (%d games on %s, pid %d)",
            step, self.games, self.device, self._proc.pid,
        )
        return True

    def poll(self) -> tuple[int, dict[str, float]] | None:
        """Non-blocking: (step, scalars) once when a run completes, else None."""
        if self._proc is None or self._proc.poll() is None:
            return None
        rc = self._proc.returncode
        step = self._step
        self._proc = None
        self._step = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if rc != 0:
            logger.warning(
                "corpus eval for step %s exited %d (see %s)", step, rc,
                self.log_path or "discarded output",
            )
            return None
        try:
            with self.results_path.open() as f:
                last = None
                for line in f:
                    if line.strip():
                        last = line
            record = json.loads(last)
            scalars = _corpus_record_scalars(record)
        except Exception as e:
            # Telemetry-only: malformed output must never reach the
            # curriculum loop as an exception.
            logger.warning("corpus eval for step %s: result parse failed (%s)", step, e)
            return None
        if not scalars:
            logger.warning("corpus eval for step %s produced no scalars", step)
            return None
        return (int(step), scalars)

    def shutdown(self) -> None:
        """Terminate an in-flight run (stage advance / curriculum exit)."""
        if self.running:
            logger.info("corpus eval for step %s terminated (stage shutdown)", self._step)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)  # reap; no zombie
                except Exception:
                    pass
        self._proc = None
        self._step = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def _build_stage_config(base: FullConfig, stage: dict) -> FullConfig:
    """Build a FullConfig for a stage by merging overrides onto base.

    Keys can use any prefix depth::

        eval_sealbot_games = 5     # cfg.eval.sealbot.games
        eval_games = 20            # cfg.eval.games
        game_win_length = 6        # cfg.game.win_length
        win_length = 6             # bare fallback (backwards compat)
    """
    cfg = dataclasses.replace(base)

    for path, overrides in _resolve_overrides(stage, _CONFIG_TREE).items():
        # Walk cfg to the target object
        obj = cfg
        for attr in path:
            obj = getattr(obj, attr)
        updated = dataclasses.replace(obj, **overrides)
        # Rebuild the chain from leaf back to root.
        # e.g. path=("eval","sealbot"): update sealbot, then eval.
        for attr in reversed(path[1:]):
            parent = cfg
            for p in path[:path.index(attr)]:
                parent = getattr(parent, p)
            updated = dataclasses.replace(parent, **{attr: updated})
        if path:
            setattr(cfg, path[0], updated)

    # Auto num_layers: 0 means int(win_length * 1.5)
    if cfg.model.num_layers == 0:
        cfg.model = dataclasses.replace(
            cfg.model, num_layers=int(cfg.game.win_length * 1.5)
        )

    # Re-validate cross-section invariants: per-stage overrides can produce
    # combinations the base config never saw (e.g. graft_threat_features=true
    # as a stage override on a relative_stone_encoding model).
    from hexo_a0.config_io import _validate_config
    _validate_config(cfg)

    return cfg


def run_curriculum(
    curriculum_path: str | Path,
    device_override: str | None = None,
    no_compile: bool = False,
    verbose: bool = False,
) -> None:
    """Run the full automatic curriculum."""
    import hexo_rs
    import torch

    from hexo_a0.config_io import load_config
    from hexo_a0.display import TrainingDisplay
    from hexo_a0.model import (
        HeXONet,
        graft_heads_for_jk_cat,
        graft_input_proj_for_threat_features,
    )
    from hexo_a0.trainer import Trainer

    display = TrainingDisplay()

    cfg = load_config(curriculum_path)

    if cfg.curriculum is None:
        logger.error("No [curriculum] section found in config")
        return

    stages = cfg.curriculum.stages
    if not stages:
        logger.error("No stages defined in curriculum config")
        return

    # CLI overrides
    if device_override:
        cfg.run = dataclasses.replace(cfg.run, device=device_override)
    if no_compile:
        cfg.run = dataclasses.replace(cfg.run, compile=False)

    device = torch.device(cfg.run.device)
    ckpt_dir = Path(cfg.run.checkpoint_dir)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume: check which stage we were on
    state_file = ckpt_dir / "curriculum_state.json"
    start_stage = 0
    if state_file.exists():
        state = json.loads(state_file.read_text())
        start_stage = state.get("completed_stages", 0)
        if start_stage > 0:
            logger.info("Resuming curriculum from stage %d/%d", start_stage + 1, len(stages))

    model = None

    for stage_idx, stage in enumerate(stages):
        if stage_idx < start_stage:
            logger.info("Skipping completed stage %d/%d: %s",
                        stage_idx + 1, len(stages), stage.get("name", ""))
            continue
        stage_name = stage.get("name", f"Stage {stage_idx + 1}")
        stage_cfg = _build_stage_config(cfg, stage)

        # Resolve curriculum-control fields with per-stage precedence:
        # stage dict overrides cfg.curriculum.*.
        convergence_mode = stage.get("convergence_mode", cfg.curriculum.convergence_mode)
        win_rate_threshold = stage.get("win_rate_threshold", cfg.curriculum.win_rate_threshold)
        improve_threshold = stage.get("improve_threshold", cfg.curriculum.improve_threshold)
        champion_threshold = stage.get("champion_threshold", cfg.curriculum.champion_threshold)
        patience = stage.get("patience", cfg.curriculum.patience)
        checkpoint_lag = stage.get("checkpoint_lag", cfg.curriculum.checkpoint_lag)
        min_steps_per_stage = stage.get("min_steps_per_stage", cfg.curriculum.min_steps_per_stage)
        confirm_advance = stage.get("confirm_advance", cfg.curriculum.confirm_advance)
        # SPRT mode parameters (ignored unless convergence_mode == "sprt")
        sprt_s0 = stage.get("sprt_s0", cfg.curriculum.sprt_s0)
        sprt_s1 = stage.get("sprt_s1", cfg.curriculum.sprt_s1)
        sprt_alpha = stage.get("sprt_alpha", cfg.curriculum.sprt_alpha)
        sprt_beta = stage.get("sprt_beta", cfg.curriculum.sprt_beta)
        sprt_window_size = stage.get("sprt_window_size", cfg.curriculum.sprt_window_size)
        sprt_mcts_sims = stage.get("sprt_mcts_sims", cfg.curriculum.sprt_mcts_sims)
        sprt_mcts_m_actions = stage.get("sprt_mcts_m_actions", cfg.curriculum.sprt_mcts_m_actions)
        sprt_device = stage.get("sprt_device", cfg.curriculum.sprt_device)
        sprt_eval_workers = stage.get("sprt_eval_workers", cfg.curriculum.sprt_eval_workers)
        sprt_poll_interval = stage.get("sprt_poll_interval", cfg.curriculum.sprt_poll_interval)
        sprt_reject_patience = stage.get("sprt_reject_patience", cfg.curriculum.sprt_reject_patience)
        sprt_promotion_grace_steps = stage.get("sprt_promotion_grace_steps", cfg.curriculum.sprt_promotion_grace_steps)
        sprt_pentanomial = stage.get("sprt_pentanomial", cfg.curriculum.sprt_pentanomial)
        sprt_pair_variance = stage.get("sprt_pair_variance", cfg.curriculum.sprt_pair_variance)
        sprt_adaptive_pair_variance = stage.get("sprt_adaptive_pair_variance", cfg.curriculum.sprt_adaptive_pair_variance)
        sprt_adaptive_pv_floor = stage.get("sprt_adaptive_pair_variance_floor", cfg.curriculum.sprt_adaptive_pair_variance_floor)
        sprt_adaptive_pv_ceil = stage.get("sprt_adaptive_pair_variance_ceil", cfg.curriculum.sprt_adaptive_pair_variance_ceil)
        sprt_adaptive_pv_ema_alpha = stage.get("sprt_adaptive_pair_variance_ema_alpha", cfg.curriculum.sprt_adaptive_pair_variance_ema_alpha)
        sprt_prev_ckpt_patience = stage.get("sprt_prev_ckpt_patience", cfg.curriculum.sprt_prev_ckpt_patience)
        sprt_prev_ckpt_band_low = stage.get("sprt_prev_ckpt_band_low", cfg.curriculum.sprt_prev_ckpt_band_low)
        sprt_prev_ckpt_band_high = stage.get("sprt_prev_ckpt_band_high", cfg.curriculum.sprt_prev_ckpt_band_high)
        sprt_promo_velocity_factor = stage.get("sprt_promo_velocity_factor", cfg.curriculum.sprt_promo_velocity_factor)
        sprt_promo_velocity_min_intervals = stage.get("sprt_promo_velocity_min_intervals", cfg.curriculum.sprt_promo_velocity_min_intervals)
        # Stage-level α/β override per-round values when > 0. Derivation:
        # under H0, all N rounds reject with prob (1-α)^N, so P(any false
        # accept) = 1 - (1-α)^N = α_stage → α = 1 - (1-α_stage)^(1/N).
        # Under H1, all N rounds reject with prob β^N = β_stage → β = β_stage^(1/N).
        sprt_alpha_stage = stage.get("sprt_alpha_stage", cfg.curriculum.sprt_alpha_stage)
        sprt_beta_stage = stage.get("sprt_beta_stage", cfg.curriculum.sprt_beta_stage)
        if sprt_alpha_stage > 0 and sprt_reject_patience > 0:
            derived_alpha = 1.0 - (1.0 - sprt_alpha_stage) ** (1.0 / sprt_reject_patience)
            logger.info(
                "SPRT: sprt_alpha_stage=%.4f with patience=%d → sprt_alpha=%.4f per round",
                sprt_alpha_stage, sprt_reject_patience, derived_alpha,
            )
            sprt_alpha = derived_alpha
        if sprt_beta_stage > 0 and sprt_reject_patience > 0:
            derived_beta = sprt_beta_stage ** (1.0 / sprt_reject_patience)
            logger.info(
                "SPRT: sprt_beta_stage=%.4f with patience=%d → sprt_beta=%.4f per round",
                sprt_beta_stage, sprt_reject_patience, derived_beta,
            )
            sprt_beta = derived_beta

        game_config = hexo_rs.GameConfig(
            win_length=stage_cfg.game.win_length,
            placement_radius=stage_cfg.game.placement_radius,
            max_moves=stage_cfg.game.max_moves,
        )

        display.print_stage_header(
            stage_num=stage_idx + 1,
            total_stages=len(stages),
            name=stage_name,
            win_length=stage_cfg.game.win_length,
            radius=stage_cfg.game.placement_radius,
            max_moves=stage_cfg.game.max_moves,
            lr=stage_cfg.training.lr,
        )

        # Create model for this stage, then load weights from previous
        # stage's model or from a checkpoint.
        new_model = HeXONet(stage_cfg.model).to(device)
        pts = sorted(p for p in ckpt_dir.glob("checkpoint_*.pt") if "_buffer" not in p.name)

        # If the previous stage recorded a champion, prefer it over the
        # latest training checkpoint (it's the strongest verified model).
        champion_ckpt = None
        if state_file.exists():
            state = json.loads(state_file.read_text())
            champion_ckpt = state.get("champion_checkpoint")
            if champion_ckpt and not Path(champion_ckpt).exists():
                champion_ckpt = None

        # Pre-compute the graft predicate once — both load paths share it.
        # Trainer.load_checkpoint runs its own graft later but the curriculum
        # loads the model first, so without this hook we hit a shape mismatch
        # in load_state_dict (Linear in_features H vs L*H) before the trainer
        # is ever constructed. The graft fn is idempotent on already-cat
        # checkpoints, so re-running it is safe.
        _do_jk_cat_graft = (
            getattr(stage_cfg.training, "graft_heads_for_jk_cat", False)
            and stage_cfg.model.use_jk
            and stage_cfg.model.jk_mode == "cat"
        )
        # Same first-load problem for the threat-features input graft: the
        # curriculum loads weights before Trainer.load_checkpoint can run its
        # graft hook. Idempotent on already-12-dim checkpoints. Never fires
        # under relative_stone_encoding (config validation rejects the combo;
        # this keeps the predicate safe regardless).
        _do_threat_graft = (
            getattr(stage_cfg.training, "graft_threat_features", False)
            and getattr(stage_cfg.model, "threat_features", False)
            and not getattr(stage_cfg.model, "relative_stone_encoding", False)
        )

        if champion_ckpt:
            ckpt_sd = torch.load(champion_ckpt, map_location=device, weights_only=True)["model_state_dict"]
            ckpt_sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt_sd.items()}
            if _do_jk_cat_graft:
                graft_heads_for_jk_cat(
                    new_model, ckpt_sd,
                    hidden_dim=stage_cfg.model.hidden_dim,
                    num_layers=stage_cfg.model.num_layers,
                )
            if _do_threat_graft:
                graft_input_proj_for_threat_features(ckpt_sd, stage_cfg.model)
            missing, unexpected = new_model.load_state_dict(ckpt_sd, strict=False)
            if missing:
                logger.info("New parameters (random init): %s", missing)
            if unexpected:
                logger.info("Dropped parameters from checkpoint: %s", unexpected)
            logger.info("Loaded weights from champion: %s", champion_ckpt)
        elif pts:
            # Resuming: load checkpoint directly
            ckpt_sd = torch.load(str(pts[-1]), map_location=device, weights_only=True)["model_state_dict"]
            ckpt_sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt_sd.items()}
            if _do_jk_cat_graft:
                graft_heads_for_jk_cat(
                    new_model, ckpt_sd,
                    hidden_dim=stage_cfg.model.hidden_dim,
                    num_layers=stage_cfg.model.num_layers,
                )
            if _do_threat_graft:
                graft_input_proj_for_threat_features(ckpt_sd, stage_cfg.model)
            missing, unexpected = new_model.load_state_dict(ckpt_sd, strict=False)
            if missing:
                logger.info("New parameters (random init): %s", missing)
            if unexpected:
                logger.info("Dropped parameters from checkpoint: %s", unexpected)
            logger.info("Loaded weights from %s", pts[-1].name)
        elif model is not None:
            # Fresh stage transition: transfer weights from previous stage's model
            missing, unexpected = new_model.load_state_dict(model.state_dict(), strict=False)
            if missing:
                logger.info("New parameters (random init): %s", missing)
            if unexpected:
                logger.info("Dropped parameters: %s", unexpected)
            logger.info("Model adapted for stage: %s", stage_name)
        else:
            logger.info("Created model: %d params", sum(p.numel() for p in new_model.parameters()))

        model = new_model

        # Isolate this process's inductor cache from the self-play inference
        # subprocess's (a shared TORCHINDUCTOR_CACHE_DIR cross-process cache race
        # on the one APU is the leading suspect for the 2026-06-02 compile NaN).
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_hexo/train")

        if stage_cfg.run.compile and hasattr(torch, "compile"):
            try:
                # Compile the actual training entry point. The trainer calls
                # model._forward_batch_core directly, so wrapping the module
                # (torch.compile only intercepts __call__/forward) was a no-op.
                # fullgraph is off: the PyG Batch arg graph-breaks on attribute
                # access. A compile-induced NaN grad can't corrupt checkpoints —
                # _clip_and_step_if_finite skips the step; a persistent NaN would
                # stall (visible), recoverable by setting run.compile=false.
                model._forward_batch_core = torch.compile(
                    model._forward_batch_core, dynamic=True
                )
            except Exception as e:
                # Don't silently swallow a compile failure — on a bleeding-edge
                # stack (torch 2.12 / ROCm / free-threaded cp313t) it would leave
                # the model eager with no indication why.
                logger.warning(
                    "torch.compile failed for stage %d, using eager: %s",
                    stage_idx + 1, e,
                )

        log_dir = stage_cfg.run.log_dir or f"runs/curriculum/s{stage_idx + 1}"
        trainer = Trainer(
            model, stage_cfg, game_config, device,
            log_dir=log_dir, ckpt_dir=str(ckpt_dir),
        )
        trainer.display = display
        trainer.stage_name = stage_name.split(":")[0] if ":" in stage_name else stage_name
        trainer._checkpoint_lag = checkpoint_lag
        display.setup_file_logging(str(Path(log_dir) / "train.log"))

        if pts:
            trainer.load_checkpoint(str(pts[-1]))
            logger.info("Resumed from %s (step %d)", pts[-1].name, trainer.train_steps)

        # Track whether the champion-resume override just synced the trainee
        # to the champion. If so, treat the sync as a fresh promotion event
        # for the purposes of the SPRT grace window (full grace ahead);
        # otherwise (mid-stage resume), grace must be measured from the
        # actual last promotion time recorded in the champion file.
        just_loaded_champion_at_fresh_transition = False

        # Clear stale buffer from the previous stage (wrong board size).
        # Only clear on a genuine stage transition, not mid-stage resume.
        # We detect this by checking if the state file's last_checkpoint
        # matches the checkpoint we just loaded — if so, this is a fresh
        # entry into a new stage, not a resume of in-progress work.
        if stage_idx > 0 and state_file.exists():
            state = json.loads(state_file.read_text())
            last_ckpt = state.get("last_checkpoint")
            if last_ckpt is not None and pts and pts[-1].name == last_ckpt:
                # Fresh stage transition: reset training state
                if len(trainer.buffer) > 0:
                    old_size = len(trainer.buffer)
                    trainer.buffer.clear()
                    logger.info("Cleared replay buffer (%d stale examples from previous stage)", old_size)
                # Reinitialize LR scheduler so cosine restarts for the new stage.
                # Keep train_steps/iteration intact to avoid checkpoint name collisions.
                trainer.stage_start_step = trainer.train_steps
                trainer._init_scheduler()
                trainer._sealbot_level_idx = 0
                trainer._sealbot_consecutive_above = 0
                logger.info("Reset LR scheduler and sealbot level for new stage (step %d, stage_start=%d)",
                            trainer.train_steps, trainer.stage_start_step)

                # Override trainee weights with the previous stage's champion
                # snapshot. The trainee may have regressed past the champion
                # in the final stretch of the previous stage — this is
                # especially common in degenerate small-board endgames where
                # both sides converge to forced draws and the network unlearns
                # win-finding skills (S1 6-in-a-row radius=2 is the canonical
                # example). The champion is the most recently SPRT-validated
                # state, and any post-promotion training is by construction
                # unvalidated. Starting the new stage from the validated
                # state and letting the trainee re-improve under the new
                # game shape is strictly safer.
                champion_path = state.get("champion_checkpoint")
                if (
                    cfg.curriculum.resume_from_champion_on_advance
                    and champion_path
                    and Path(champion_path).exists()
                ):
                    ckpt = torch.load(champion_path, map_location=device, weights_only=True)
                    ckpt_sd = ckpt["model_state_dict"]
                    ckpt_sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt_sd.items()}
                    model_sd_keys = set(trainer.model.state_dict().keys())
                    if any(k.startswith("_orig_mod.") for k in model_sd_keys):
                        ckpt_sd = {"_orig_mod." + k: v for k, v in ckpt_sd.items()}
                    trainer.model.load_state_dict(ckpt_sd, strict=False)
                    # Adam moments correspond to the old trainee weights —
                    # zero them so optimization starts fresh on the champion.
                    for group in trainer.optimizer.param_groups:
                        for p in group["params"]:
                            if p in trainer.optimizer.state:
                                opt_state = trainer.optimizer.state[p]
                                if "exp_avg" in opt_state:
                                    opt_state["exp_avg"].zero_()
                                if "exp_avg_sq" in opt_state:
                                    opt_state["exp_avg_sq"].zero_()
                                if "step" in opt_state:
                                    opt_state["step"] = (
                                        torch.zeros_like(opt_state["step"])
                                        if torch.is_tensor(opt_state["step"])
                                        else 0
                                    )
                    champ_step = ckpt.get("train_steps", "?")
                    logger.info(
                        "Loaded champion weights for new stage from %s (champion step %s, overriding trainee at step %d)",
                        Path(champion_path).name, champ_step, trainer.train_steps,
                    )
                    just_loaded_champion_at_fresh_transition = True

        # Resolve the self-play data source. When the source is 'champion'
        # (either explicit or legacy-auto-derived from convergence_mode),
        # tell the trainer to suppress periodic self-play model re-exports;
        # self-play only updates on champion promotion. SPRT uses the same
        # champion-file plumbing but a different decision mechanism. With
        # source='trainee', self-play follows the live model regardless of
        # convergence_mode (canonical AZ / Lc0 / KataGo style).
        self_play_source = _resolve_self_play_source(
            stage_cfg.self_play, convergence_mode
        )
        trainer._champion_mode = self_play_source == "champion"
        trainer._sprt_mode = convergence_mode == "sprt"
        champion_promotions = 0
        if state_file.exists():
            champion_promotions = json.loads(state_file.read_text()).get("champion_promotions", 0)

        # Train until convergence
        from collections import deque
        consecutive_above = 0
        recent_win_rates: deque[float] = deque(maxlen=patience)
        # SPRT secondary plateau detector: rolling window of prev-checkpoint
        # win rates, pooled into one Wilson test (see _prev_ckpt_plateau_fired).
        # Sized to the configured patience (or 1 to keep the deque
        # constructible when the detector is off).
        prev_ckpt_window: deque[float] = deque(maxlen=max(sprt_prev_ckpt_patience, 1))
        # Promotion-velocity detector state: completed intervals between
        # post-warmup promotions THIS stage. Persisted across restarts via
        # <stage_log_dir>/detector_state.json (restored in the sprt setup
        # below); `velocity_prev_promo_step` tracks only real observed
        # promotions, so the stage-start→first-promotion gap (inflated by
        # warmup) never pollutes the median.
        promo_intervals: list[int] = []
        velocity_prev_promo_step: int | None = None
        _velocity_drought_logged = False
        # Once-per-declaration throttle for the CONVERGED/CONFIRM banner
        # (latched detectors would otherwise re-announce every loop chunk).
        _converged_announced = False
        stage_start_step = trainer.stage_start_step
        converged = False
        last_sprt_games = -1  # track SPRT state progression to dedupe TB writes
        # Reject count baseline captured when the trainee enters its first
        # "active" SPRT period — past warmup AND past the post-promotion
        # grace window. Rejects accumulated during warmup or grace (trainee
        # still adapting, can't reasonably out-play a freshly-promoted self)
        # do not count toward the patience gate.
        sprt_reject_baseline: int | None = None
        # Plateau detector: dual-threshold auto-calibrated from the per-stage
        # peak_history of reject_count at each prior promotion.
        #   warn_thr = max(floor, ⌈μ + σ⌉)  — informational; logs anomalously slow cycle
        #   decl_thr = max(floor, max + 1)  — gates the curriculum CONVERGED flag
        # `floor` is the user-set sprt_reject_patience, which doubles as the
        # bootstrap value when peak_history is empty. `_warned_this_cycle`
        # suppresses log spam — warn fires once per cycle then stays quiet.
        _warned_this_cycle = False
        _last_peak_history_len = 0
        # Step at which the trainee was promoted (or stage started). Used
        # to compute the post-promotion grace window before reject patience
        # starts counting. Read from the champion file if available so that
        # mid-stage process restarts don't spuriously reset the grace window;
        # if the champion-resume override just fired (trainee was synced to
        # champion at fresh stage transition), reset to current step instead
        # so the sync gets a full fresh grace.
        sprt_last_promotion_step = trainer.train_steps
        champion_file_for_init = ckpt_dir / "self_play" / "champion.pt"
        if (
            not just_loaded_champion_at_fresh_transition
            and champion_file_for_init.exists()
            # In both champion mode AND sprt mode, champion.pt is written by
            # _export_champion with the promoted checkpoint's train_steps, so
            # its step number IS the (slightly conservative — validated-ckpt
            # lag) true last-promotion anchor regardless of self-play source.
            # Without this, every restart of a trainee-source sprt run
            # re-anchored steps_since_promotion to process start.
            and (trainer._champion_mode or convergence_mode == "sprt")
        ):
            try:
                _champ_init_ckpt = torch.load(
                    str(champion_file_for_init), map_location="cpu", weights_only=True
                )
                _ts = _champ_init_ckpt.get("train_steps")
                # Guard _ts <= current: a rolled-back trainer (champion ahead
                # of the resumed step counter) must not produce a negative
                # gap / permanent grace.
                if isinstance(_ts, int) and _ts <= trainer.train_steps:
                    sprt_last_promotion_step = _ts
                    logger.info(
                        "SPRT grace anchored to champion's last-promotion step %d (current trainer step %d)",
                        _ts, trainer.train_steps,
                    )
            except Exception:
                pass
        trainer.start_self_play()
        trainer.log_event("stage", f"Started: {stage_name} (convergence={convergence_mode})")

        # Export the current model as the initial champion *after*
        # start_self_play so the self-play directory exists. Required when
        # either (a) champion mode (self-play opponent IS the champion) or
        # (b) SPRT mode (SPRT needs a baseline opponent for promotion eval,
        # regardless of self-play source). Skip if a champion already
        # exists (e.g. manually placed or from a previous run that was
        # interrupted mid-stage).
        if trainer._champion_mode or convergence_mode == "sprt":
            champion_file = ckpt_dir / "self_play" / "champion.pt"
            if champion_file.exists():
                trainer._champion_path = str(champion_file)
                champ_ckpt = torch.load(str(champion_file), map_location="cpu", weights_only=True)
                champ_step = champ_ckpt.get("train_steps", "?")
                logger.info("Using existing champion (step %s): %s", champ_step, champion_file)
            else:
                trainer._export_champion()
                logger.info("Initial champion exported for stage: %s", stage_name)

        # SPRT mode: launch the continuous eval daemon after champion setup.
        sprt_watcher = None
        detector_state_file = None
        if convergence_mode == "sprt":
            from hexo_a0.sprt_watcher import SPRTWatcher
            # Stage log dir if provided, otherwise checkpoint dir as fallback
            stage_log_dir = Path(stage.get("run_log_dir", cfg.run.log_dir))
            stage_log_dir.mkdir(parents=True, exist_ok=True)
            # Plateau-detector state survives trainer restarts (the velocity
            # median would otherwise need min_intervals fresh promotions to
            # re-arm, and the pooled prev-ckpt window ~16k steps to refill).
            detector_state_file = stage_log_dir / "detector_state.json"
            _ds = _load_detector_state(
                detector_state_file, stage_idx, trainer.train_steps
            )
            if (
                _ds["promo_intervals"]
                or _ds["prev_ckpt_window"]
                or _ds["velocity_prev_promo_step"] is not None
                or _ds["last_promotion_step"] is not None
            ):
                promo_intervals.extend(_ds["promo_intervals"])
                velocity_prev_promo_step = _ds["velocity_prev_promo_step"]
                prev_ckpt_window.extend(_ds["prev_ckpt_window"])
                # Promotion EVENT step beats the champion.pt anchor set
                # above (checkpoint step lags the event by the validation
                # delay → steps_since_promotion would jump on restart).
                if _ds["last_promotion_step"] is not None:
                    sprt_last_promotion_step = _ds["last_promotion_step"]
                logger.info(
                    "Detector state restored: %d promotion interval(s), "
                    "prev promotion at step %s, last promotion event at "
                    "step %s, %d prev-ckpt eval(s)",
                    len(promo_intervals), velocity_prev_promo_step,
                    _ds["last_promotion_step"], len(prev_ckpt_window),
                )
            sprt_watcher = SPRTWatcher(
                config_path=Path(curriculum_path),
                trainee_dir=ckpt_dir,
                champion_path=ckpt_dir / "self_play" / "champion.pt",
                state_file=stage_log_dir / "sprt_state.json",
                stage_index=stage_idx,
                s0=sprt_s0, s1=sprt_s1,
                alpha=sprt_alpha, beta=sprt_beta,
                window_size=sprt_window_size,
                mcts_sims=sprt_mcts_sims,
                mcts_m_actions=sprt_mcts_m_actions,
                device=sprt_device,
                eval_workers=sprt_eval_workers,
                poll_interval=sprt_poll_interval,
                pentanomial=sprt_pentanomial,
                pair_variance=sprt_pair_variance,
                adaptive_pair_variance=sprt_adaptive_pair_variance,
                adaptive_pair_variance_floor=sprt_adaptive_pv_floor,
                adaptive_pair_variance_ceil=sprt_adaptive_pv_ceil,
                adaptive_pair_variance_ema_alpha=sprt_adaptive_pv_ema_alpha,
            )
            sprt_watcher.start()
            logger.info("SPRT daemon launched; state file at %s", sprt_watcher.state_file)

        # Human-corpus yardstick: eval each promoted champion in a background
        # subprocess; scalars land in TB under corpus/* at the promotion step.
        corpus_runner = None
        if convergence_mode == "sprt" and stage_cfg.eval.corpus.games > 0:
            corpus_runner = CorpusEvalRunner(
                script_path=Path(__file__).resolve().parents[3]
                / "scripts" / "eval_human_corpus.py",
                results_path=stage_log_dir / "corpus_eval_results.jsonl",
                games=stage_cfg.eval.corpus.games,
                device=stage_cfg.eval.corpus.device,
                log_path=stage_log_dir / "corpus_eval.log",
            )
            if not corpus_runner.enabled:
                corpus_runner = None

        # Display refresh callback fired from inside trainer.train()'s inner
        # loop (~every 5 gradient steps) so the SPRT line keeps pace with the
        # daemon instead of only updating once per ~100-step chunk. Promotion
        # and plateau decisions stay on the chunk boundary; this only re-renders.
        def _refresh_sprt_display(_step: int) -> None:
            if sprt_watcher is None:
                return
            dec = sprt_watcher.check()
            if dec is None:
                return
            sprt_upper = math.log((1.0 - sprt_beta) / sprt_alpha)
            steps_since_promo = trainer.train_steps - sprt_last_promotion_step
            if trainer.train_steps - stage_start_step <= min_steps_per_stage:
                reject_status = "warmup"
                display_reject_count = dec.reject_count
            elif steps_since_promo < sprt_promotion_grace_steps:
                reject_status = f"grace {steps_since_promo}/{sprt_promotion_grace_steps}"
                display_reject_count = dec.reject_count
            else:
                reject_status = "active"
                if sprt_reject_baseline is not None:
                    display_reject_count = dec.reject_count - sprt_reject_baseline
                else:
                    display_reject_count = 0
            display.update_sprt(
                score=dec.score, llr=dec.llr, games=dec.games,
                decision=dec.decision,
                s0=sprt_s0, s1=sprt_s1, upper=sprt_upper,
                reject_count=display_reject_count,
                reject_patience=sprt_reject_patience,
                reject_status=reject_status,
            )

        display.update_status(
            stage=trainer.stage_name, step=trainer.train_steps,
            loss="-", policy_loss="-", value_loss="-",
            games="0", games_per_sec="-", buf_size=str(len(trainer.buffer)),
            steps_per_sec="-", lr=f"{stage_cfg.training.lr:.2e}",
        )

        try:
            while not converged:
                # Wait for buffer to have data before first training step.
                # In alternating mode (no background workers), train()
                # handles self-play inline, so skip this wait.
                if len(trainer.buffer) == 0 and trainer._sp_pool is not None:
                    import time as _time
                    logger.info("Waiting for initial self-play data...")
                    while len(trainer.buffer) == 0:
                        _time.sleep(0.1)
                metrics = trainer.train(
                    step_callback=_refresh_sprt_display if sprt_watcher is not None else None,
                )
                stage_steps_done = trainer.train_steps - stage_start_step

                # Manual advance: touch a sentinel file to skip to next stage
                advance_file = ckpt_dir / "ADVANCE"
                if advance_file.exists():
                    advance_file.unlink()
                    logger.info("MANUAL ADVANCE requested — skipping to next stage")
                    trainer.log_event("stage", "Manual advance requested")
                    ckpt_path = ckpt_dir / f"checkpoint_{trainer.train_steps:08d}.pt"
                    trainer.save_checkpoint(str(ckpt_path), save_buffer=True)
                    state_file.write_text(json.dumps({
                        "completed_stages": stage_idx + 1,
                        "stage_name": stage_name,
                        "train_steps": trainer.train_steps,
                        "last_checkpoint": ckpt_path.name,
                        "champion_checkpoint": trainer._champion_path,
                        "champion_promotions": champion_promotions,
                    }))
                    converged = True
                    break

                # Check convergence / champion promotion.
                # Champion promotions are allowed during warmup (they
                # strengthen self-play without advancing the stage).
                # Only auto-advance is gated by min_steps_per_stage.
                should_advance = False
                warmup = stage_steps_done <= min_steps_per_stage

                if convergence_mode == "champion":
                    vs_champ = metrics.get("eval_win_rate_vs_champion")
                    if vs_champ is not None:
                        if vs_champ >= champion_threshold:
                            # Promotion: trainee beats champion
                            champion_promotions += 1
                            logger.info(
                                "CHAMPION PROMOTED (#%d): win rate %.2f >= %.2f — exporting new champion",
                                champion_promotions, vs_champ, champion_threshold,
                            )
                            trainer._export_champion()
                            recent_win_rates.clear()
                            msg = (
                                f"Champion promoted (#{champion_promotions} at step {trainer.train_steps}, "
                                f"win rate {vs_champ:.0%} >= {champion_threshold:.0%})"
                            )
                            display.print_event(msg)
                            trainer.log_event("champion", msg)
                        elif not warmup:
                            recent_win_rates.append(vs_champ)
                            mean_wr = sum(recent_win_rates) / len(recent_win_rates)
                            full_window = len(recent_win_rates) >= patience
                            display.update_convergence(
                                mean_wr, champion_threshold, len(recent_win_rates), full_window,
                            )
                            if full_window:
                                if mean_wr < champion_threshold:
                                    logger.info(
                                        "Mean win rate vs champion %.2f < %.2f over last %d evals — plateaued",
                                        mean_wr, champion_threshold, len(recent_win_rates),
                                    )
                                    should_advance = True
                                else:
                                    logger.info(
                                        "Mean win rate vs champion %.2f >= %.2f over last %d evals — still improving",
                                        mean_wr, champion_threshold, len(recent_win_rates),
                                    )
                            else:
                                logger.info(
                                    "Win rate vs champion %.2f (collecting %d/%d evals for convergence)",
                                    vs_champ, len(recent_win_rates), patience,
                                )

                elif convergence_mode == "sprt" and sprt_watcher is not None:
                    dec = sprt_watcher.check()
                    if dec is not None:
                        sprt_upper = math.log((1.0 - sprt_beta) / sprt_alpha)
                        # Compute reject status for display: warmup → rejects
                        # ignored entirely; grace → rejects ignored until
                        # grace expires; active → rejects count toward patience.
                        steps_since_promo = trainer.train_steps - sprt_last_promotion_step
                        if warmup:
                            reject_status = "warmup"
                            display_reject_count = dec.reject_count
                        elif steps_since_promo < sprt_promotion_grace_steps:
                            reject_status = (
                                f"grace {steps_since_promo}/{sprt_promotion_grace_steps}"
                            )
                            display_reject_count = dec.reject_count
                        else:
                            reject_status = "active"
                            # Use post-baseline count if baseline has been
                            # captured, else the raw count (first active poll).
                            if sprt_reject_baseline is not None:
                                display_reject_count = dec.reject_count - sprt_reject_baseline
                            else:
                                display_reject_count = 0
                        display.update_sprt(
                            score=dec.score, llr=dec.llr, games=dec.games,
                            decision=dec.decision,
                            s0=sprt_s0, s1=sprt_s1, upper=sprt_upper,
                            reject_count=display_reject_count,
                            reject_patience=sprt_reject_patience,
                            reject_status=reject_status,
                        )
                        # TB: log only when SPRT state advanced (game count
                        # changed — increase on new games, decrease=0 on
                        # post-promotion reset).
                        if dec.games != last_sprt_games and trainer.writer is not None:
                            step = trainer.train_steps
                            trainer.writer.add_scalar("sprt/score", dec.score, step)
                            trainer.writer.add_scalar("sprt/llr", dec.llr, step)
                            trainer.writer.add_scalar("sprt/games", dec.games, step)
                            if dec.games > 0:
                                trainer.writer.add_scalar("sprt/win_rate", dec.wins / dec.games, step)
                                trainer.writer.add_scalar("sprt/draw_rate", dec.draws / dec.games, step)
                                trainer.writer.add_scalar("sprt/loss_rate", dec.losses / dec.games, step)
                            trainer.writer.add_scalar("sprt/reject_count", dec.reject_count, step)
                            trainer.writer.add_scalar("sprt/pairs", dec.pairs, step)
                            if dec.empirical_pair_variance is not None:
                                trainer.writer.add_scalar(
                                    "sprt/empirical_pair_variance",
                                    dec.empirical_pair_variance, step,
                                )
                            last_sprt_games = dec.games
                        if dec.decision == "accept_h1":
                            # Promote trainee → champion. Daemon will detect
                            # the champion.pt mtime change and reset its
                            # state automatically on the next poll.
                            #
                            # Promote the SPRT-validated checkpoint (the one
                            # the daemon was actually evaluating), not the
                            # live training model — the live model may be
                            # several hundred steps newer and unvalidated,
                            # and may have drifted away from the strength
                            # SPRT measured.
                            champion_promotions += 1
                            sd_override = None
                            ts_override = None
                            promoted_step_label = trainer.train_steps
                            if dec.trainee_path:
                                try:
                                    ckpt = torch.load(
                                        dec.trainee_path, map_location="cpu",
                                        weights_only=False,
                                    )
                                    sd_override = ckpt["model_state_dict"]
                                    ts_override = ckpt.get("train_steps")
                                    if ts_override is not None:
                                        promoted_step_label = ts_override
                                except Exception as e:
                                    logger.warning(
                                        "Failed to load validated trainee %s "
                                        "(%s); falling back to live model export",
                                        dec.trainee_path, e,
                                    )
                            else:
                                logger.warning(
                                    "SPRT decision missing trainee_path; "
                                    "falling back to live model export",
                                )
                            logger.info(
                                "CHAMPION PROMOTED (#%d) via SPRT: "
                                "score=%.3f LLR=%.3f games=%d (s0=%.2f s1=%.2f) "
                                "from %s",
                                champion_promotions, dec.score, dec.llr, dec.games,
                                sprt_s0, sprt_s1,
                                dec.trainee_path if sd_override is not None
                                else "live model",
                            )
                            trainer._export_champion(
                                state_dict_override=sd_override,
                                train_steps_override=ts_override,
                            )
                            # Daemon will reset reject_count to 0 on next
                            # poll (champion.pt mtime changed). Reset our
                            # baseline tracker too so the next active-period
                            # capture starts fresh against the new champion,
                            # and restart the post-promotion grace clock.
                            sprt_reject_baseline = None
                            sprt_last_promotion_step = trainer.train_steps
                            # Promotion-velocity bookkeeping: record only
                            # intervals strictly between POST-warmup
                            # promotions. Warmup-era promotions (against a
                            # weak initial champion, typically fast) neither
                            # append nor anchor, so no recorded interval can
                            # span warmup and deflate the median.
                            if not warmup:
                                if velocity_prev_promo_step is not None:
                                    promo_intervals.append(
                                        trainer.train_steps - velocity_prev_promo_step
                                    )
                                velocity_prev_promo_step = trainer.train_steps
                            if detector_state_file is not None:
                                _save_detector_state(
                                    detector_state_file, stage_idx,
                                    promo_intervals, velocity_prev_promo_step,
                                    prev_ckpt_window,
                                    last_promotion_step=sprt_last_promotion_step,
                                )
                            # Corpus yardstick: eval the validated checkpoint
                            # that was actually promoted (stable file), not
                            # champion.pt (overwritten by the next promotion).
                            if corpus_runner is not None:
                                _corpus_ckpt = dec.trainee_path or str(
                                    ckpt_dir / "self_play" / "champion.pt"
                                )
                                _corpus_step = (
                                    promoted_step_label
                                    if isinstance(promoted_step_label, int)
                                    else trainer.train_steps
                                )
                                corpus_runner.maybe_launch(_corpus_ckpt, _corpus_step)
                            msg = (
                                f"SPRT promoted (#{champion_promotions} from step "
                                f"{promoted_step_label}, score={dec.score:.3f} "
                                f"over {dec.games} games)"
                            )
                            display.print_event(msg)
                            trainer.log_event("champion", msg)
                        # Stage plateau gate: rejects only count when trainee
                        # is in an "active" period — past stage warmup AND past
                        # the post-promotion grace window. During warmup or
                        # grace, the trainee is still adapting (to a fresh
                        # champion or to having no training history yet) and
                        # rejects there are expected, not a plateau signal.
                        steps_since_promotion = trainer.train_steps - sprt_last_promotion_step
                        in_grace = steps_since_promotion < sprt_promotion_grace_steps
                        active = (not warmup) and (not in_grace)
                        if active:
                            if sprt_reject_baseline is None:
                                sprt_reject_baseline = dec.reject_count
                                if dec.reject_count > 0:
                                    logger.info(
                                        "SPRT active (post-warmup, post-grace) with "
                                        "%d reject(s) already accumulated; setting "
                                        "patience baseline at %d",
                                        dec.reject_count, dec.reject_count,
                                    )
                            post_baseline_rejects = dec.reject_count - sprt_reject_baseline

                            # Auto-calibrated dual-threshold detector.
                            # sprt_reject_patience <= 0 disables it entirely
                            # (there is no other safe "off" value: the floor
                            # would otherwise bottom out at 0 and decl_thr=0
                            # fires instantly on an empty peak_history). The
                            # reject counter itself stays tracked for the CLI
                            # display and TB regardless.
                            # peak_history may grow between iterations as new
                            # promotions land; reset _warned_this_cycle when it
                            # does so the next cycle's first warn re-fires.
                            reject_detector_on = sprt_reject_patience > 0
                            ph = dec.peak_history
                            if len(ph) != _last_peak_history_len:
                                _warned_this_cycle = False
                                _last_peak_history_len = len(ph)

                            floor = sprt_reject_patience
                            if ph:
                                ph_max = max(ph)
                                ph_mean = statistics.mean(ph)
                                ph_std = statistics.stdev(ph) if len(ph) >= 2 else 0.0
                                warn_thr = max(floor, math.ceil(ph_mean + ph_std))
                                decl_thr = max(floor, ph_max + 1)
                            else:
                                ph_max = ph_mean = ph_std = 0.0
                                warn_thr = decl_thr = floor

                            if reject_detector_on:
                                trainer.writer.add_scalar("sprt/warn_thr", warn_thr, step)
                                trainer.writer.add_scalar("sprt/decl_thr", decl_thr, step)
                            trainer.writer.add_scalar("sprt/peak_history_len", len(ph), step)
                            trainer.writer.add_scalar("sprt/peak_history_max", ph_max, step)
                            trainer.writer.add_scalar("sprt/peak_history_mean", ph_mean, step)

                            if (
                                reject_detector_on
                                and post_baseline_rejects >= warn_thr
                                and not _warned_this_cycle
                            ):
                                logger.warning(
                                    "SPRT cycle anomalously slow: post-baseline "
                                    "rejects=%d ≥ warn_thr=%d (μ=%.1f σ=%.1f over %d "
                                    "prior cycles, max=%.0f). Continuing training; "
                                    "decl_thr=%d would mark plateau.",
                                    post_baseline_rejects, warn_thr,
                                    ph_mean, ph_std, len(ph), ph_max, decl_thr,
                                )
                                _warned_this_cycle = True

                            if reject_detector_on and post_baseline_rejects >= decl_thr:
                                logger.info(
                                    "SPRT PLATEAU: %d active reject_h1 rounds "
                                    "against current champion (decl_thr=%d, "
                                    "auto-calibrated from peak_history=%s, "
                                    "floor=%d, baseline=%d) — marking stage converged",
                                    post_baseline_rejects, decl_thr, ph,
                                    sprt_reject_patience, sprt_reject_baseline,
                                )
                                should_advance = True

                    # Secondary plateau detector: prev-checkpoint win rate
                    # stuck inside the "no improvement" band over N consecutive
                    # evals. Catches the failure mode where SPRT's
                    # auto-calibrated decl_thr keeps rising past what's
                    # achievable while the model has actually stopped
                    # improving. Gated on the same "active" condition as the
                    # SPRT plateau check (post-warmup, post-promotion grace)
                    # so it doesn't fire while the trainee is still adapting.
                    # `in_grace` from the SPRT block above isn't in scope here
                    # (it lives inside `if dec is not None:`), so compute it
                    # fresh from the loop-level promotion-step tracker.
                    _steps_since_promo = trainer.train_steps - sprt_last_promotion_step
                    _in_grace = _steps_since_promo < sprt_promotion_grace_steps
                    if (
                        sprt_prev_ckpt_patience > 0
                        and not warmup
                        and not _in_grace
                    ):
                        vs_prev = metrics.get("eval_win_rate_vs_prev")
                        if vs_prev is not None:
                            prev_ckpt_window.append(vs_prev)
                            if detector_state_file is not None:
                                _save_detector_state(
                                    detector_state_file, stage_idx,
                                    promo_intervals, velocity_prev_promo_step,
                                    prev_ckpt_window,
                                    last_promotion_step=sprt_last_promotion_step,
                                )
                            window_mean = sum(prev_ckpt_window) / len(prev_ckpt_window)
                            pooled_n = len(prev_ckpt_window) * max(
                                stage_cfg.eval.games, 1
                            )
                            if trainer.writer is not None:
                                trainer.writer.add_scalar(
                                    "sprt/prev_ckpt_window_mean",
                                    window_mean,
                                    trainer.train_steps,
                                )
                                trainer.writer.add_scalar(
                                    "sprt/prev_ckpt_pooled_upper",
                                    _wilson_upper(window_mean, pooled_n),
                                    trainer.train_steps,
                                    description=(
                                        "One-sided 95% Wilson upper bound on "
                                        "the pooled win rate vs the lagged "
                                        "checkpoint (last "
                                        "sprt_prev_ckpt_patience evals x "
                                        "eval.games games). Reads as 'we are "
                                        "confident improvement is below this "
                                        "rate'; the plateau detector fires "
                                        "when it drops under "
                                        "sprt_prev_ckpt_band_high."
                                    ),
                                )
                            if _prev_ckpt_plateau_fired(
                                prev_ckpt_window,
                                sprt_prev_ckpt_patience,
                                sprt_prev_ckpt_band_low,
                                sprt_prev_ckpt_band_high,
                                stage_cfg.eval.games,
                            ):
                                logger.info(
                                    "PREV-CKPT PLATEAU: pooled win rate %.3f "
                                    "over %d evals (%d games), Wilson upper "
                                    "%.3f < %.2f — marking stage converged",
                                    window_mean,
                                    len(prev_ckpt_window),
                                    pooled_n,
                                    _wilson_upper(window_mean, pooled_n),
                                    sprt_prev_ckpt_band_high,
                                )
                                should_advance = True

                    # Promotion-velocity plateau detector + shadow telemetry.
                    # Gap is measured from the last promotion (or the stage/
                    # resume anchor); the median uses only completed intervals
                    # between promotions observed this stage.
                    _gap = trainer.train_steps - sprt_last_promotion_step
                    if trainer.writer is not None:
                        trainer.writer.add_scalar(
                            "sprt/steps_since_promotion", _gap, trainer.train_steps,
                            description=(
                                "Training steps since the last champion "
                                "promotion (anchored to champion.pt's step on "
                                "restart). Numerator of the promotion-velocity "
                                "plateau detector; sustained growth = drought."
                            ),
                        )
                        if promo_intervals:
                            _med = statistics.median(promo_intervals)
                            trainer.writer.add_scalar(
                                "sprt/promo_interval_median", _med,
                                trainer.train_steps,
                                description=(
                                    "Median steps between post-warmup "
                                    "promotions this stage (this process; "
                                    "restarts clear the history and the "
                                    "detector re-arms after "
                                    "sprt_promo_velocity_min_intervals fresh "
                                    "ones). Each promotion is an "
                                    "alpha-certified >= s1 improvement, so "
                                    "this is the stage's certified-Elo-"
                                    "velocity unit."
                                ),
                            )
                            trainer.writer.add_scalar(
                                "sprt/promo_velocity_ratio", _gap / _med,
                                trainer.train_steps,
                                description=(
                                    "steps_since_promotion / median promotion "
                                    "interval. ~1 = normal cadence; plateau "
                                    "is declared when it exceeds "
                                    "sprt_promo_velocity_factor (gated by "
                                    "confirm_advance). Caution: LR resets "
                                    "cause benign droughts — check loss "
                                    "curves before believing a spike."
                                ),
                            )
                    if not warmup and not _in_grace and _promo_velocity_fired(
                        promo_intervals, _gap,
                        sprt_promo_velocity_factor,
                        sprt_promo_velocity_min_intervals,
                    ):
                        # The gap only grows until the next promotion, so this
                        # latches; log once per drought to avoid spamming the
                        # event log every loop chunk while CONFIRM is pending.
                        if not _velocity_drought_logged:
                            logger.info(
                                "PROMO-VELOCITY PLATEAU: %d steps since last "
                                "promotion > %.1f x median interval %.0f "
                                "(%d intervals this stage) — marking stage "
                                "converged",
                                _gap, sprt_promo_velocity_factor,
                                statistics.median(promo_intervals),
                                len(promo_intervals),
                            )
                            _velocity_drought_logged = True
                        should_advance = True
                    else:
                        _velocity_drought_logged = False

                    # Harvest a finished corpus eval, if any (non-blocking).
                    if corpus_runner is not None:
                        _corpus_done = corpus_runner.poll()
                        if _corpus_done is not None:
                            _c_step, _c_scalars = _corpus_done
                            if trainer.writer is not None:
                                for _tag, _val in _c_scalars.items():
                                    trainer.writer.add_scalar(
                                        _tag, _val, _c_step,
                                        description=_CORPUS_SCALAR_DESCRIPTIONS.get(_tag),
                                    )
                            logger.info(
                                "corpus eval (step %d): top16=%.3f ece=%.3f "
                                "sign_acc=%.3f perplexity=%.1f",
                                _c_step,
                                _c_scalars.get("eval/corpus_policy_top16", float("nan")),
                                _c_scalars.get("eval/corpus_value_ece", float("nan")),
                                _c_scalars.get("eval/corpus_value_sign_acc", float("nan")),
                                _c_scalars.get("eval/corpus_policy_perplexity", float("nan")),
                            )

                elif warmup:
                    continue

                elif convergence_mode == "random":
                    win_rate = metrics.get("eval_win_rate")
                    if win_rate is not None:
                        if win_rate >= win_rate_threshold:
                            consecutive_above += 1
                            logger.info(
                                "Win rate vs random %.2f >= %.2f (%d/%d)",
                                win_rate, win_rate_threshold, consecutive_above, patience,
                            )
                        else:
                            consecutive_above = 0
                        should_advance = consecutive_above >= patience

                elif convergence_mode == "checkpoint":
                    vs_prev = metrics.get("eval_win_rate_vs_prev")
                    if vs_prev is not None:
                        recent_win_rates.append(vs_prev)
                        mean_wr = sum(recent_win_rates) / len(recent_win_rates)
                        full_window = len(recent_win_rates) >= patience
                        display.update_convergence(
                            mean_wr, improve_threshold, len(recent_win_rates), full_window,
                        )
                        if full_window:
                            if mean_wr < improve_threshold:
                                logger.info(
                                    "Mean win rate vs prev checkpoint %.2f < %.2f over last %d evals — plateaued",
                                    mean_wr, improve_threshold, len(recent_win_rates),
                                )
                                should_advance = True
                            else:
                                logger.info(
                                    "Mean win rate vs prev checkpoint %.2f >= %.2f over last %d evals — still improving",
                                    mean_wr, improve_threshold, len(recent_win_rates),
                                )
                        else:
                            logger.info(
                                "Win rate vs prev checkpoint %.2f (collecting %d/%d evals)",
                                vs_prev, len(recent_win_rates), patience,
                            )

                if not should_advance:
                    # Detector backed off (e.g. pooled window drifted out of
                    # band) — re-arm the announcement for the next declaration.
                    _converged_announced = False
                else:
                    # Detectors can latch (e.g. a promotion drought only
                    # grows), re-declaring every loop chunk while CONFIRM is
                    # pending — announce once, then wait quietly.
                    if not _converged_announced:
                        logger.info(
                            "STAGE CONVERGED: %s (%s mode, patience=%d)",
                            stage_name, convergence_mode, patience,
                        )
                        trainer.log_event("stage",
                            f"Converged: {stage_name} ({convergence_mode} mode, patience={patience})"
                        )
                    if confirm_advance:
                        confirm_file = ckpt_dir / "CONFIRM"
                        if not confirm_file.exists():
                            msg = f"CONVERGED — touch {confirm_file} to advance (training continues)"
                            if not _converged_announced:
                                logger.info(msg)
                                display.print_event(msg)
                            should_advance = False  # keep training
                            _converged_announced = True
                        else:
                            confirm_file.unlink()
                            logger.info("CONFIRM file found — advancing to next stage")
                            display.print_event("CONFIRM received — advancing to next stage")

                if should_advance:
                    ckpt_path = ckpt_dir / f"checkpoint_{trainer.train_steps:08d}.pt"
                    trainer.save_checkpoint(str(ckpt_path), save_buffer=True)
                    state_file.write_text(json.dumps({
                        "completed_stages": stage_idx + 1,
                        "stage_name": stage_name,
                        "train_steps": trainer.train_steps,
                        "last_checkpoint": ckpt_path.name,
                        "champion_checkpoint": trainer._champion_path,
                        "champion_promotions": champion_promotions,
                    }))
                    converged = True

        except KeyboardInterrupt:
            display.print_event("Stopping self-play workers...")
            trainer.stop_self_play()
            display.print_event("Stopping self-play workers... done!")
            display.print_event("Saving checkpoint...")
            ckpt_path = ckpt_dir / f"checkpoint_{trainer.train_steps:08d}.pt"
            trainer.save_checkpoint(str(ckpt_path), save_buffer=False)
            display.print_event(f"Saving checkpoint... done! ({ckpt_path.name}, step {trainer.train_steps})")
            display.close()
            return
        finally:
            if sprt_watcher is not None:
                sprt_watcher.stop()
            if corpus_runner is not None:
                corpus_runner.shutdown()
            trainer.stop_self_play()
            # Release the buffer's SQLite connections (and tensorboard writer)
            # before the next stage's Trainer opens the same replay_buffer.db.
            # Without this, stale file locks from the previous stage cause
            # "database is locked" when the new stage calls buffer.clear().
            trainer.close()

        # Unwrap torch.compile for weight transfer
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

    display.close()
    logger.info("CURRICULUM COMPLETE — all %d stages converged!", len(stages))
    state_file.write_text(json.dumps({
        "completed_stages": len(stages),
        "stage_name": "ALL COMPLETE",
        "train_steps": trainer.train_steps,
    }))
