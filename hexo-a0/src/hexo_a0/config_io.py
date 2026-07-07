"""TOML configuration loading and saving for HeXO AlphaZero training runs."""

import dataclasses
import logging
import math
import tomllib
from dataclasses import fields
from pathlib import Path

from hexo_a0.config import (
    CurriculumConfig,
    EvalConfig,
    FullConfig,
    ActorConfig,
    CorpusEvalConfig,
    GameConfig,
    MCTSConfig,
    ModelConfig,
    PythonSelfPlayConfig,
    RunConfig,
    RustSelfPlayConfig,
    SealbotConfig,
    SelfPlayConfig,
    SelfPlayMctsOverride,
    TrainingConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section name -> dataclass class mapping
# ---------------------------------------------------------------------------

_SECTION_MAP: dict[str, type] = {
    "model": ModelConfig,
    "game": GameConfig,
    "mcts": MCTSConfig,
    "training": TrainingConfig,
    "self_play": SelfPlayConfig,
    "eval": EvalConfig,
    "run": RunConfig,
    "curriculum": CurriculumConfig,
}

# Nested sub-tables: parent_section -> {sub_key: (cls, field_name)}
_NESTED_MAP: dict[str, dict[str, tuple[type, str]]] = {
    "self_play": {
        "rust": (RustSelfPlayConfig, "rust"),
        "python": (PythonSelfPlayConfig, "python"),
        "mcts": (SelfPlayMctsOverride, "mcts"),
        "actor": (ActorConfig, "actor"),
    },
    "eval": {
        "sealbot": (SealbotConfig, "sealbot"),
        "corpus": (CorpusEvalConfig, "corpus"),
    },
}


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------


def _load_section(cls: type, data: dict, section_name: str):
    """Load a TOML section dict into a dataclass instance.

    Introspects fields, warns on unknown keys, skips nested dataclass sub-keys.
    """
    field_names = {f.name for f in fields(cls)}
    nested_keys = set(_NESTED_MAP.get(section_name, {}).keys())
    kwargs: dict = {}
    for key, value in data.items():
        if key in nested_keys:
            continue
        if key not in field_names:
            logger.warning(
                "Unknown key %r in [%s] section (ignored)", key, section_name
            )
            continue
        kwargs[key] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> FullConfig:
    """Load training configuration from a TOML file.

    Returns a FullConfig object. Missing sections/keys use dataclass defaults.
    Unknown TOML keys trigger a warning.
    """
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    # Warn on unknown top-level sections
    known_sections = set(_SECTION_MAP.keys())
    # Allow top-level 'seed' key (legacy compat)
    known_top_keys = known_sections | {"seed"}
    for key in data:
        if key not in known_top_keys:
            logger.warning("Unknown top-level section %r (ignored)", key)

    # Load each section
    sections: dict = {}
    for section_name, cls in _SECTION_MAP.items():
        if section_name == "curriculum":
            continue  # handled separately
        section_data = data.get(section_name, {})
        # Load nested sub-tables first
        nested_map = _NESTED_MAP.get(section_name, {})
        nested_kwargs: dict = {}
        for sub_key, (sub_cls, field_name) in nested_map.items():
            if sub_key in section_data:
                nested_kwargs[field_name] = _load_section(
                    sub_cls,
                    section_data[sub_key],
                    f"{section_name}.{sub_key}",
                )
        obj = _load_section(cls, section_data, section_name)
        # Apply nested overrides
        if nested_kwargs:
            obj = dataclasses.replace(obj, **nested_kwargs)
        sections[section_name] = obj

    # Legacy compat: top-level 'seed' maps to run.seed
    if "seed" in data and "run" in sections:
        run_section_data = data.get("run", {})
        if "seed" not in run_section_data:
            sections["run"] = dataclasses.replace(
                sections["run"], seed=data["seed"]
            )

    # Auto num_layers: 0 means int(win_length * 1.5)
    model: ModelConfig = sections.get("model", ModelConfig())
    game: GameConfig = sections.get("game", GameConfig())
    if model.num_layers == 0:
        sections["model"] = dataclasses.replace(
            model, num_layers=int(game.win_length * 1.5)
        )

    # Curriculum section (optional)
    curriculum = None
    if "curriculum" in data:
        curriculum = _load_section(
            CurriculumConfig, data["curriculum"], "curriculum"
        )

    full = FullConfig(
        model=sections.get("model", ModelConfig()),
        game=game,
        mcts=sections.get("mcts", MCTSConfig()),
        training=sections.get("training", TrainingConfig()),
        self_play=sections.get("self_play", SelfPlayConfig()),
        eval=sections.get("eval", EvalConfig()),
        run=sections.get("run", RunConfig()),
        curriculum=curriculum,
    )
    _validate_config(full)
    return full


# ---------------------------------------------------------------------------
# Cross-section validation
# ---------------------------------------------------------------------------


_VALID_JK_MODES = ("sum", "cat", "max", "lstm")


def _validate_config(cfg: FullConfig) -> None:
    """Validate cross-section invariants on the parsed config.

    Raises ValueError with an informative message on bad combinations.
    Currently checks:
      - jk_mode is one of the recognised values.
      - jk_mode in {cat, max, lstm} requires use_jk=true.
      - graft_heads_for_jk_cat=true requires use_jk=true and jk_mode='cat'.
      - relative_stone_encoding is incompatible with graft_threat_features.
    """
    mc = cfg.model
    if mc.jk_mode not in _VALID_JK_MODES:
        raise ValueError(
            f"[model] jk_mode={mc.jk_mode!r} is not one of {_VALID_JK_MODES!r}."
        )
    if mc.jk_mode in ("cat", "max", "lstm") and not mc.use_jk:
        raise ValueError(
            f"[model] jk_mode={mc.jk_mode!r} requires use_jk=true; "
            f"got use_jk=false. Set use_jk=true or revert jk_mode to 'sum'."
        )
    tc = cfg.training
    if getattr(tc, "graft_heads_for_jk_cat", False):
        if not mc.use_jk or mc.jk_mode != "cat":
            raise ValueError(
                "[training] graft_heads_for_jk_cat=true requires "
                "[model] use_jk=true and jk_mode='cat'; got "
                f"use_jk={mc.use_jk}, jk_mode={mc.jk_mode!r}."
            )
    # Distributional value head invariants.
    vbins = getattr(mc, "value_bins", 0)
    if vbins != 0 and vbins < 2:
        raise ValueError(
            f"[model] value_bins={vbins} is invalid: use 0 (scalar head) or "
            f">=2 (distributional head)."
        )
    if vbins > 0 and getattr(mc, "value_bin_min", -1.0) >= getattr(
        mc, "value_bin_max", 1.0
    ):
        raise ValueError(
            f"[model] value_bin_min ({mc.value_bin_min}) must be < value_bin_max "
            f"({mc.value_bin_max}) when value_bins>0."
        )

    # Short-horizon value heads (train-only).
    horizons = getattr(mc, "value_horizons", []) or []
    if horizons:
        if vbins <= 0:
            raise ValueError(
                "[model] value_horizons requires value_bins>0 (the horizon heads "
                "share the distributional value head's bin grid)."
            )
        if any((not isinstance(k, int)) or k < 1 for k in horizons):
            raise ValueError(
                f"[model] value_horizons must be positive integers (placements); "
                f"got {horizons}."
            )
    if getattr(tc, "horizon_loss_weight", 0.0) > 0.0 and not horizons:
        raise ValueError(
            "[training] horizon_loss_weight>0 requires a non-empty "
            "[model] value_horizons."
        )

    # Train-only per-move Q head.
    if getattr(tc, "q_loss_weight", 0.0) > 0.0 and not getattr(mc, "q_head", False):
        raise ValueError(
            "[training] q_loss_weight>0 requires [model] q_head=true."
        )

    if getattr(tc, "graft_threat_features", False) and getattr(
        mc, "relative_stone_encoding", False
    ):
        raise ValueError(
            "[training] graft_threat_features=true is incompatible with "
            "[model] relative_stone_encoding=true: the own/opp stone remap is "
            "input-dependent, so absolute-encoding checkpoints cannot be "
            "grafted into a relative-encoding model. relative_stone_encoding "
            "is a from-scratch-run-only toggle — disable one of the two."
        )

    # Remote self-play actor invariants.
    actor = getattr(cfg.self_play, "actor", None)
    if actor is not None and actor.host:
        if not actor.repo_dir:
            raise ValueError(
                "[self_play.actor] host is set but repo_dir is empty. "
                "Set repo_dir to the hexo repo path on the actor host."
            )
        if not cfg.self_play.rust.python_inference:
            raise ValueError(
                "[self_play.actor] requires [self_play.rust] python_inference=true: "
                "the self_play binary is CPU-libtorch-only and cannot run GPU "
                "inference itself on the actor; GPU work must go through the Python "
                "inference_server (which uses the actor's CUDA torch venv)."
            )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def _fmt(value: object) -> str | None:
    """Format a Python value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Cannot serialize non-finite float to TOML: {value!r}"
            )
        return repr(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        items = ", ".join(_fmt(v) or "null" for v in value)
        return f"[{items}]"
    if value is None:
        return None
    return str(value)


def _write_section(
    lines: list[str], header: str, obj: object, skip_fields: set[str] | None = None
) -> None:
    """Write a [header] section from a dataclass instance.

    Emits the dataclass docstring as a comment above the header, and
    field ``metadata["description"]`` as inline TOML comments.
    """
    skip = skip_fields or set()
    # Emit dataclass docstring as section comment (multi-line, preserving
    # relative indentation via textwrap.dedent).
    doc = type(obj).__doc__
    if doc:
        import textwrap
        dedented = textwrap.dedent(doc).strip()
        for doc_line in dedented.splitlines():
            if doc_line.strip():
                lines.append(f"# {doc_line}")
            else:
                lines.append("#")
    lines.append(f"[{header}]")
    current_group = ""
    for f in fields(type(obj)):
        if f.name in skip:
            continue
        value = getattr(obj, f.name)
        formatted = _fmt(value)
        if formatted is not None:
            # Emit group separator comment when group changes
            group = (f.metadata or {}).get("group", "")
            if group and group != current_group:
                lines.append(f"# {group}")
            current_group = group
            desc = f.metadata.get("description", "") if f.metadata else ""
            if desc:
                kv = f"{f.name} = {formatted}"
                lines.append(f"{kv:<40s} # {desc}")
            else:
                lines.append(f"{f.name} = {formatted}")
    lines.append("")


def save_config(path: str | Path, config: FullConfig) -> None:
    """Save a FullConfig to a TOML file."""
    path = Path(path)
    lines: list[str] = []

    # Top-level sections (skip nested dataclass fields, handle them separately)
    _write_section(lines, "model", config.model)
    _write_section(lines, "game", config.game)
    _write_section(lines, "mcts", config.mcts)
    _write_section(lines, "training", config.training)

    # self_play: skip nested 'rust' and 'python' fields
    _write_section(lines, "self_play", config.self_play, skip_fields={"rust", "python", "mcts", "actor"})
    _write_section(lines, "self_play.rust", config.self_play.rust)
    _write_section(lines, "self_play.python", config.self_play.python)
    _write_section(lines, "self_play.mcts", config.self_play.mcts)
    _write_section(lines, "self_play.actor", config.self_play.actor)

    # eval: skip nested 'sealbot' field
    _write_section(lines, "eval", config.eval, skip_fields={"sealbot", "corpus"})
    _write_section(lines, "eval.sealbot", config.eval.sealbot)
    _write_section(lines, "eval.corpus", config.eval.corpus)

    _write_section(lines, "run", config.run)

    if config.curriculum is not None:
        _write_section(
            lines, "curriculum", config.curriculum, skip_fields={"stages"}
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def default_config_toml() -> str:
    """Return the default configuration as a valid TOML string."""
    config = FullConfig()
    lines: list[str] = []

    _write_section(lines, "model", config.model)
    _write_section(lines, "game", config.game)
    _write_section(lines, "mcts", config.mcts)
    _write_section(lines, "training", config.training)
    _write_section(lines, "self_play", config.self_play, skip_fields={"rust", "python", "mcts", "actor"})
    _write_section(lines, "self_play.rust", config.self_play.rust)
    _write_section(lines, "self_play.python", config.self_play.python)
    _write_section(lines, "self_play.mcts", config.self_play.mcts)
    _write_section(lines, "self_play.actor", config.self_play.actor)
    _write_section(lines, "eval", config.eval, skip_fields={"sealbot", "corpus"})
    _write_section(lines, "eval.sealbot", config.eval.sealbot)
    _write_section(lines, "eval.corpus", config.eval.corpus)
    _write_section(lines, "run", config.run)

    return "\n".join(lines)


def default_curriculum_toml() -> str:
    """Return an annotated curriculum TOML template.

    Generates all sections from dataclass defaults, then appends
    example curriculum stages.
    """
    config = FullConfig(
        model=ModelConfig(hidden_dim=256, num_layers=3, num_heads=8,
                          policy_hidden=128, value_hidden=128, graph_type="axis"),
        training=TrainingConfig(batch_size=512, lr=0.0002, lr_schedule="cosine",
                                lr_min=0.00002, lr_warmup_steps=400,
                                total_train_steps=30000, edge_budget=500_000,
                                augment_symmetries=True),
        self_play=SelfPlayConfig(engine="rust", device="cuda", workers=16, precision="fp16"),
        eval=EvalConfig(interval=1000,
                        checkpoint_interval=300, games=50,
                        sealbot=SealbotConfig(games=20, adaptive=True)),
        run=RunConfig(device="cuda", checkpoint_dir="runs/my-curriculum/checkpoints",
                      fp16=True),
        curriculum=CurriculumConfig(
            convergence_mode="checkpoint", improve_threshold=0.65,
            patience=10, checkpoint_lag=30, min_steps_per_stage=10000,
        ),
    )

    lines: list[str] = [
        "# HeXO AlphaZero — Curriculum Configuration Template",
        "#",
        "# Multi-stage curriculum learning: trains on progressively harder game",
        "# variants, automatically advancing when convergence is detected.",
        "#",
        "# Usage:",
        "#   hexo-a0 curriculum --config this_file.toml",
        "#",
        "# Structure:",
        "#   - Shared sections set defaults that apply to ALL stages.",
        "#   - [[curriculum.stages]] entries override per-stage values.",
        "#   - The [model] section is fixed across stages.",
        "",
    ]

    _write_section(lines, "curriculum", config.curriculum, skip_fields={"stages"})
    _write_section(lines, "model", config.model)
    _write_section(lines, "game", config.game)
    _write_section(lines, "mcts", config.mcts)
    _write_section(lines, "training", config.training)
    _write_section(lines, "self_play", config.self_play, skip_fields={"rust", "python", "mcts", "actor"})
    _write_section(lines, "self_play.rust", config.self_play.rust)
    _write_section(lines, "self_play.python", config.self_play.python)
    _write_section(lines, "self_play.mcts", config.self_play.mcts)
    _write_section(lines, "self_play.actor", config.self_play.actor)
    _write_section(lines, "eval", config.eval, skip_fields={"sealbot", "corpus"})
    _write_section(lines, "eval.sealbot", config.eval.sealbot)
    _write_section(lines, "eval.corpus", config.eval.corpus)
    _write_section(lines, "run", config.run)

    # Example stages
    lines.extend([
        "# --- Per-stage overrides ---",
        "# Each stage can override fields from [game], [mcts], [training],",
        "# [eval], [self_play], and [run] (e.g., lr, log_dir).",
        "",
        "[[curriculum.stages]]",
        'name = "S1: 4-in-a-row, radius 2"',
        "win_length = 4",
        "placement_radius = 2",
        "max_moves = 80",
        "lr = 0.0002",
        'log_dir = "runs/my-curriculum/s1"',
        "",
        "[[curriculum.stages]]",
        'name = "S2: 5-in-a-row, radius 4"',
        "win_length = 5",
        "placement_radius = 4",
        "max_moves = 160",
        "lr = 0.0002",
        'log_dir = "runs/my-curriculum/s2"',
        "",
        "[[curriculum.stages]]",
        'name = "S3: 6-in-a-row, radius 6"',
        "win_length = 6",
        "placement_radius = 6",
        "max_moves = 200",
        "lr = 0.0002",
        'log_dir = "runs/my-curriculum/s3"',
        "",
        "[[curriculum.stages]]",
        'name = "S4: 6-in-a-row, radius 8 (FULL HEXO)"',
        "win_length = 6",
        "placement_radius = 8",
        "max_moves = 1000",
        "lr = 0.0002",
        'log_dir = "runs/my-curriculum/s4"',
    ])

    return "\n".join(lines)
