"""HeXO AlphaZero — GNN architecture for infinite hex tic-tac-toe.

Public names are imported **lazily** (PEP 562 ``__getattr__``). ``from hexo_a0
import Trainer`` still works, but ``import hexo_a0`` — and importing a light
submodule such as ``hexo_a0.scriptable_model`` / ``hexo_a0.inference_server`` —
no longer *eagerly* pulls ``torch_geometric`` / ``tensorboard`` via ``graph`` /
``model`` / ``trainer``. This keeps the inference server **standalone**: it (and
its runtime deps ``scriptable_model`` + ``gpu_memory``) need only ``torch``, so a
remote self-play actor venv can be torch-only. See ``[self_play.actor]``.

Eager-import behaviour is preserved on first *access* of any exported name, and
arbitrary submodules remain accessible as attributes (lazy submodule fallback).
"""

import importlib
from typing import TYPE_CHECKING

# Public name -> submodule that defines it. Imported on first attribute access.
_EXPORTS: dict[str, str] = {
    # Config (light; kept lazy for uniformity)
    "CurriculumConfig": "hexo_a0.config",
    "EvalConfig": "hexo_a0.config",
    "FullConfig": "hexo_a0.config",
    "GameConfig": "hexo_a0.config",
    "MCTSConfig": "hexo_a0.config",
    "ModelConfig": "hexo_a0.config",
    "PythonSelfPlayConfig": "hexo_a0.config",
    "RunConfig": "hexo_a0.config",
    "RustSelfPlayConfig": "hexo_a0.config",
    "SealbotConfig": "hexo_a0.config",
    "SelfPlayConfig": "hexo_a0.config",
    "TrainingConfig": "hexo_a0.config",
    "load_config": "hexo_a0.config_io",
    "save_config": "hexo_a0.config_io",
    # Model & graph (these pull torch_geometric)
    "game_to_graph": "hexo_a0.graph",
    "game_to_axis_graph": "hexo_a0.graph",
    "HeXONet": "hexo_a0.model",
    "alphazero_loss": "hexo_a0.loss",
    "kl_policy_loss": "hexo_a0.loss",
    # Training (Trainer pulls tensorboard)
    "ReplayBuffer": "hexo_a0.replay_buffer",
    "Trainer": "hexo_a0.trainer",
    "train_step": "hexo_a0.trainer",
    "self_play_game": "hexo_a0.self_play",
    "TrainingExample": "hexo_a0.self_play",
    "gumbel_mcts": "hexo_a0.mcts",
    # Evaluation
    "evaluate_vs_random": "hexo_a0.evaluate",
    "play_eval_game": "hexo_a0.evaluate",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """PEP 562 lazy attribute access.

    Resolves an exported public name by importing its defining submodule on
    demand, then caches it in ``globals()`` so later accesses skip this hook.
    Falls back to importing ``hexo_a0.<name>`` as a submodule so attribute-style
    submodule access (e.g. ``hexo_a0.config``) keeps working without eagerly
    loading everything at package import.
    """
    module = _EXPORTS.get(name)
    if module is not None:
        obj = getattr(importlib.import_module(module), name)
        globals()[name] = obj
        return obj
    try:
        sub = importlib.import_module(f"{__name__}.{name}")
    except ImportError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = sub
    return sub


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:
    # Static-analysis / IDE resolution of the lazily-exported names.
    from hexo_a0.config import (
        CurriculumConfig,
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
    from hexo_a0.config_io import load_config, save_config
    from hexo_a0.evaluate import evaluate_vs_random, play_eval_game
    from hexo_a0.graph import game_to_axis_graph, game_to_graph
    from hexo_a0.loss import alphazero_loss, kl_policy_loss
    from hexo_a0.mcts import gumbel_mcts
    from hexo_a0.model import HeXONet
    from hexo_a0.replay_buffer import ReplayBuffer
    from hexo_a0.self_play import TrainingExample, self_play_game
    from hexo_a0.trainer import Trainer, train_step
