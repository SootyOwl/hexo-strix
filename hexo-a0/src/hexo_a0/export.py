"""Export a training checkpoint's weights to safetensors for hexo-infer / wasm.

The safetensors file is the single artifact the pure-Rust ``hexo-infer`` crate loads
(no libtorch, no Python). ``save_safetensors`` is the ONE producer of the
``hexo-safetensors-v1`` format — both ``export_checkpoint`` (the ``hexo-a0 export``
CLI) and the parity-fixture generator call it, so the format contract has a single
owner and can't drift between the two.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

FORMAT_TAG = "hexo-safetensors-v1"


def save_safetensors(
    state_dict: dict,
    model_config: dict,
    train_steps,
    source_name: str,
    out_path: Path,
) -> dict:
    """Write model weights + metadata to a safetensors file (the format producer).

    Tensor names are the state_dict keys with any torch.compile ``_orig_mod.`` prefix
    stripped; all tensors are cast to contiguous fp32. Returns the metadata dict that
    was embedded (also retrievable by hexo-infer via the safetensors header).

    ``train_steps`` may be an int or "?" (stringified). ``source_name`` is recorded as
    ``source_checkpoint`` so hexo-infer's real-parity gate can assert the loaded weights
    match the checkpoint the fixtures were minted from. ``model_config`` and (if
    present) ``game_config`` are JSON-encoded into metadata.
    """
    from safetensors.torch import save_file

    sd = {k.removeprefix("_orig_mod."): v.float().contiguous() for k, v in state_dict.items()}
    metadata = {
        "format": FORMAT_TAG,
        "model_config": json.dumps(model_config),
        "train_steps": str(train_steps),
        "source_checkpoint": source_name,
    }
    if isinstance(model_config.get("game_config"), dict):
        metadata["game_config"] = json.dumps(model_config["game_config"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out_path), metadata=metadata)
    logger.info("Exported %d tensors to %s", len(sd), out_path)
    return metadata


def export_checkpoint(ckpt_path: Path, out_path: Path) -> dict:
    """Export a .pt training checkpoint's weights to safetensors.

    Loads the checkpoint (weights_only-first with an unrestricted fallback, matching
    head_to_head.py's defensive loader — some real checkpoints contain objects
    weights_only=True rejects), validates it carries an embedded ``model_config``
    dict (only champion/training checkpoints written by trainer.py are exportable),
    and delegates to :func:`save_safetensors`.
    """
    import torch

    ckpt_path = Path(ckpt_path)
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        logger.warning("weights_only load failed for %s; retrying unrestricted", ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    model_config = ckpt.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(
            f"{ckpt_path} has no embedded model_config dict; only champion/"
            "training checkpoints written by trainer.py are exportable."
        )
    train_steps = ckpt.get("train_steps", ckpt.get("iteration", "?"))
    # game_config travels with the checkpoint (stage-specific for curriculum runs);
    # expose it in metadata so hexo-infer can cross-check the client's position config.
    if isinstance(ckpt.get("game_config"), dict):
        model_config = {**model_config, "game_config": ckpt["game_config"]}
    return save_safetensors(
        ckpt["model_state_dict"],
        model_config,
        train_steps,
        ckpt_path.name,
        out_path,
    )