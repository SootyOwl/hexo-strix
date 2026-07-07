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
    _attach_native(model, ckpt, checkpoint)
    out = (model, mc, ckpt)
    _MODEL_CACHE[key] = out
    return out


def _attach_native(model, ckpt, checkpoint_path):
    """Attach a hexo-infer native engine as ``model._hexo_native`` (best-effort).

    The checkpoint is re-serialized to safetensors in memory (nothing written
    next to the checkpoint, which may be a read-only mount) and loaded by the
    pure-Rust engine. Hot paths (bot moves, ``_run_mcts``) use it when present;
    on any failure — unsupported architecture (hexo-infer only loads
    GINE/axis/pre-norm models), missing embedded model_config, or an older
    ``hexo_rs`` build without ``InferModel`` — the attribute stays ``None`` and
    serving falls back to the torch eval path unchanged.
    """
    import logging
    import os

    model._hexo_native = None
    if os.environ.get("HEXO_NATIVE_INFER", "").lower() in ("0", "false", "off"):
        # A/B lever: the pure-Rust kernel measured SLOWER than torch for
        # serving MCTS on the x86 dev box (2026-07-06: 1.0s vs 0.67s @128
        # sims, 4.3s vs 3.5s @512 on champion.pt) — its optimizations target
        # the WASM build. Disable to compare on the deployment host.
        logging.getLogger(__name__).info(
            "native hexo-infer engine disabled via HEXO_NATIVE_INFER")
        return
    try:
        from hexo_a0.export import serialize_safetensors

        mc_dict = ckpt.get("model_config")
        if not isinstance(mc_dict, dict):
            raise ValueError("checkpoint has no embedded model_config dict")
        if isinstance(ckpt.get("game_config"), dict):
            mc_dict = {**mc_dict, "game_config": ckpt["game_config"]}
        blob, _ = serialize_safetensors(
            ckpt["model_state_dict"], mc_dict,
            ckpt.get("train_steps", ckpt.get("iteration", "?")),
            os.path.basename(str(checkpoint_path)),
        )
        model._hexo_native = _hexo_rs.InferModel(blob)
        logging.getLogger(__name__).info(
            "native hexo-infer engine attached for %s", checkpoint_path)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "native search engine unavailable (%s); torch eval path in use", e)


def load_native_model(checkpoint: str):
    """Load a safetensors checkpoint into the pure-Rust engine — no torch.

    Returns ``(model, model_config, meta)`` with the same arity as
    ``load_model``. ``model`` is a ``NativeModel`` shim carrying
    ``._hexo_native``; ``model_config`` is resolved from the safetensors
    ``__metadata__`` (authoritative for graph flags); ``meta`` is the parsed
    metadata dict. Builds NO ``HeXONet`` and imports no torch.
    """
    import json
    import hexo_rs
    from hexo_a0.config import model_config_from_checkpoint
    from hexo_a0.serving.nativeutil import NativeModel

    with open(checkpoint, "rb") as f:
        blob = f.read()
    infer = hexo_rs.InferModel(blob)
    meta = json.loads(infer.metadata_json())
    mc_json = meta.get("model_config")
    mc_dict = json.loads(mc_json) if isinstance(mc_json, str) else mc_json
    if not isinstance(mc_dict, dict):
        raise ValueError(
            f"safetensors {checkpoint!r} has no usable model_config metadata")
    mc = model_config_from_checkpoint({"model_config": mc_dict})
    return NativeModel(infer), mc, meta


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
