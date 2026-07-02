"""Model loading + graph-builder factory (ported from scripts/play_server.py:1632
`load_model` and scripts/policy_viewer.py:58-67 `_make_graph_fn`).

`load_model` is LRU-cached and is the SINGLE place that resolves the
checkpoint-embedded model config — so the node-feature stride
(threat_features / relative_stone_encoding) is decided in exactly one location.
"""
import functools

# Lazy module refs (heavy imports deferred to first call, mirroring the scripts).
_torch = None
_hexo_rs = None
_model_mod = None
_graph_mod = None
_config_mod = None


def _ensure_imports():
    global _torch, _hexo_rs, _model_mod, _graph_mod, _config_mod
    if _torch is None:
        import torch
        import hexo_rs as hr
        from hexo_a0 import model as mm, graph as gm, config as cm
        _torch = torch
        _hexo_rs = hr
        _model_mod = mm
        _graph_mod = gm
        _config_mod = cm


# Manual cache keyed only on (checkpoint, device). We deliberately do NOT use
# @functools.lru_cache here: model_config_dict is a dict (the TOML [model]
# section) and lru_cache hashes every argument, which would raise
# ``TypeError: unhashable type: 'dict'`` at startup. The fallback dict only
# matters when a checkpoint lacks an embedded model_config, and the first load
# fixes the resolved ModelConfig, so caching on (checkpoint, device) is correct.
_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def load_model(checkpoint: str, device: str = "cpu", model_config_dict=None):
    """Load a HeXONet from a checkpoint path.

    Returns ``(model, model_config, ckpt)``. ``weights_only=True`` deserialises
    only safe types (tensors + primitive containers), closing the pickle RCE
    vector — project checkpoints store ``model_config`` as a plain dict and
    ``model_state_dict`` as tensors, so this loads them unchanged.

    ``model_config_dict`` (e.g. a TOML ``[model]`` section) is the fallback when
    the checkpoint carries no embedded ``model_config``, mirroring the frozen
    ``scripts/play_server.py`` loader; an embedded config always wins.

    Only champion/promoted checkpoints are supported: ``weights_only=True``
    deserialises tensors + primitive containers and will refuse a full training
    checkpoint whose pickle graph contains non-primitive objects (optimizer
    state, etc.) — it fails loudly at startup rather than silently. Project
    champion checkpoints store only ``model_state_dict`` (tensors) +
    ``model_config`` (dict) + ``train_steps`` (int), which load unchanged.
    """
    key = (str(checkpoint), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    _ensure_imports()
    ckpt = _torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model_state_dict"), dict):
        raise ValueError(
            f"checkpoint {checkpoint!r} is not a valid HeXONet checkpoint "
            f"(expected a dict with a dict 'model_state_dict'); got {type(ckpt).__name__}")
    mc = _config_mod.model_config_from_checkpoint(ckpt, model_config_dict)
    model = _model_mod.HeXONet(mc).to(device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        import logging
        logging.getLogger(__name__).warning(
            "checkpoint %s loaded with strict=False: missing=%d unexpected=%d "
            "(architecture/model_config mismatch?)",
            checkpoint, len(result.missing_keys), len(result.unexpected_keys),
        )
    model.eval()
    out = (model, mc, ckpt)
    _MODEL_CACHE[key] = out
    return out


def make_graph_fn(mc):
    """Per-state graph builder threading all graph flags from the model config."""
    _ensure_imports()
    prune = getattr(mc, "prune_empty_edges", False)
    threat = getattr(mc, "threat_features", False)
    rel = getattr(mc, "relative_stone_encoding", False)
    if mc.graph_type == "axis":
        return lambda g: _graph_mod.game_to_axis_graph(
            g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
    return lambda g: _graph_mod.game_to_graph(
        g, threat_features=threat, relative_stones=rel)
