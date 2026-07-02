"""GPU caching-allocator management for shared-device self-play training.

Two responsibilities, both aimed at the same problem: the training process and
one-or-more inference subprocesses share a single ROCm/HIP GPU, each with its
own PyTorch caching allocator. A freed block hoarded by one process is invisible
to its co-tenant, so under memory pressure a co-tenant OOMs even though the GPU
has free memory in aggregate.

Historically this was papered over with a ``torch.cuda.empty_cache()`` on EVERY
MCTS leaf-eval forward pass — correct, but it forces a device sync + allocator
churn on the hottest path in self-play.

This module replaces that with:

1. ``configure_cuda_alloc()`` — sets ``PYTORCH_CUDA_ALLOC_CONF`` so the
   allocator auto-releases cached blocks under pressure
   (``garbage_collection_threshold``) and fragments less
   (``max_split_size_mb``), optionally enabling ``expandable_segments`` when the
   installed build supports it. This is the *root-cause* fix and the real safety
   net.

   CRITICAL: ``PYTORCH_CUDA_ALLOC_CONF`` is read once, when the HIP/CUDA context
   is first initialised. Mutating ``os.environ`` after that is a silent no-op.
   Callers MUST invoke ``configure_cuda_alloc()`` at process entry, before any
   ``.to(cuda)`` / ``torch.cuda.*`` device-touching call.

2. ``CacheClearGate`` — a call-counter gate so the hot eval path clears the
   cache only every Nth invocation instead of every forward, bounding
   steady-state hoarding without paying the per-forward sync cost.

Both are driven by environment knobs so the change stays self-contained and does
not need to thread new fields through the (separately-owned) config dataclasses.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

# --- Allocator config knobs ------------------------------------------------

#: Standard env var the PyTorch allocator itself reads.
_PYTORCH_ENV = "PYTORCH_CUDA_ALLOC_CONF"
#: Our override knob: if set, becomes the value we write into _PYTORCH_ENV
#: (unless the user already exported _PYTORCH_ENV directly, which always wins).
_HEXO_ENV = "HEXO_CUDA_ALLOC_CONF"
#: Explicit opt-in/out for expandable_segments (defers to platform probe if unset).
_EXPANDABLE_ENV = "HEXO_EXPANDABLE_SEGMENTS"

#: Default allocator config. garbage_collection_threshold lets the allocator
#: proactively release cached blocks once usage crosses 80% (so a co-tenant can
#: grab them); max_split_size_mb caps block splitting to curb fragmentation.
DEFAULT_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:128"

# --- Cache-clear cadence knobs ---------------------------------------------

#: Env knob for the clear cadence (clear every Nth eval invocation).
_CADENCE_ENV = "HEXO_CACHE_CLEAR_EVERY"
#: Default cadence: clear roughly once per ~128 forwards. Much less frequent
#: than the old per-forward clear, but still bounds steady-state hoarding.
DEFAULT_CACHE_CLEAR_EVERY = 128


def supports_expandable_segments() -> bool:
    """Best-effort probe: should we AUTO-enable ``expandable_segments:True``?

    On CUDA builds expandable_segments is stable since ~torch 2.1. On ROCm/HIP
    its availability is driver/GPU-specific and CANNOT be cleanly pre-detected:
    where unavailable the HIP allocator silently ignores the key and emits a
    ``UserWarning`` ("expandable_segments not supported on this platform") at
    first device touch. So we never auto-enable it on HIP — a user whose HIP
    build does support it can opt in explicitly via ``HEXO_EXPANDABLE_SEGMENTS``
    (or by putting the key in ``HEXO_CUDA_ALLOC_CONF``). Returning False just
    falls back to the still-effective gc-threshold + split-size config.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch always present in practice
        return False

    # ROCm/HIP build: do not auto-enable (see docstring). Opt-in only.
    if getattr(getattr(torch, "version", None), "hip", None) is not None:
        return False

    # CUDA build: expandable_segments landed ~torch 2.1.
    try:
        major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        return False
    return (major, minor) >= (2, 1)


def _expandable_opt_in() -> bool | None:
    """Explicit override via ``HEXO_EXPANDABLE_SEGMENTS`` (truthy/falsey).

    Returns None when unset (defer to the platform probe), else the parsed bool.
    Lets a user force expandable_segments on a HIP build that supports it, or
    force it off on CUDA.
    """
    raw = os.environ.get(_EXPANDABLE_ENV)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _merge_expandable(conf: str) -> str:
    """Append ``expandable_segments:True`` unless the key is already present."""
    if "expandable_segments" in conf:
        return conf
    if not conf:
        return "expandable_segments:True"
    return f"{conf},expandable_segments:True"


def configure_cuda_alloc(supports_expandable: bool | None = None) -> str:
    """Set ``PYTORCH_CUDA_ALLOC_CONF`` for the allocator, before context init.

    Precedence:
      1. An externally-exported ``PYTORCH_CUDA_ALLOC_CONF`` always wins and is
         returned untouched (user override).
      2. Otherwise ``HEXO_CUDA_ALLOC_CONF`` if set, else the module default.
      3. ``expandable_segments:True`` is appended only when enabled and not
         already specified. Enablement = explicit ``HEXO_EXPANDABLE_SEGMENTS``
         opt-in if set, otherwise the platform probe (never auto-on for HIP,
         where an unsupported key emits a noisy UserWarning at device init).

    Returns the effective config string (also written into ``os.environ`` so the
    allocator picks it up at first device touch).

    ``supports_expandable`` is injectable for testing; when ``None`` it is
    resolved from :func:`_expandable_opt_in` then :func:`supports_expandable_segments`.
    """
    existing = os.environ.get(_PYTORCH_ENV)
    if existing is not None and existing.strip():
        # User exported it directly — do not clobber.
        logger.debug("%s already set to %r; leaving as-is", _PYTORCH_ENV, existing)
        return existing

    # Kill-switch: HEXO_CUDA_ALLOC_CONF set to a disable sentinel leaves
    # PYTORCH_CUDA_ALLOC_CONF untouched (PyTorch's built-in default) — the exact
    # pre-change baseline, for A/B-ing whether our allocator config regressed perf.
    hexo_raw = os.environ.get(_HEXO_ENV)
    if hexo_raw is not None and hexo_raw.strip().lower() in ("", "off", "none", "default"):
        logger.info("%s=%r -> leaving %s unset (PyTorch default)", _HEXO_ENV, hexo_raw, _PYTORCH_ENV)
        return ""

    conf = hexo_raw or DEFAULT_CUDA_ALLOC_CONF

    if supports_expandable is None:
        override = _expandable_opt_in()
        supports_expandable = override if override is not None else supports_expandable_segments()
    if supports_expandable:
        conf = _merge_expandable(conf)

    os.environ[_PYTORCH_ENV] = conf
    logger.info("Set %s=%s", _PYTORCH_ENV, conf)
    return conf


def cache_clear_cadence() -> int:
    """Resolve the clear cadence from ``HEXO_CACHE_CLEAR_EVERY``.

    Falls back to :data:`DEFAULT_CACHE_CLEAR_EVERY` for unset/invalid values.
    A value <= 0 means "never clear" (the allocator config is the safety net).
    """
    raw = os.environ.get(_CADENCE_ENV)
    if raw is None:
        return DEFAULT_CACHE_CLEAR_EVERY
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid %s=%r; using default %d",
            _CADENCE_ENV, raw, DEFAULT_CACHE_CLEAR_EVERY,
        )
        return DEFAULT_CACHE_CLEAR_EVERY


class CacheClearGate:
    """Counter gate that fires (clears) once every ``every`` calls.

    Replaces a per-forward ``torch.cuda.empty_cache()`` in MCTS leaf-eval hot
    paths. ``step()`` increments the counter; when it reaches the period it
    invokes ``clear_fn`` and returns ``True``, else returns ``False``.

    ``every <= 0`` disables clearing entirely.
    """

    def __init__(self, every: int, clear_fn: Callable[[], None]):
        self._every = int(every)
        self._clear_fn = clear_fn
        self._count = 0

    @classmethod
    def for_device(
        cls,
        device: "object",
        every: int | None = None,
        clear_fn: Callable[[], None] | None = None,
    ) -> "CacheClearGate":
        """Build a gate appropriate for ``device``.

        On a non-CUDA device the gate is permanently disabled (clearing the CUDA
        cache is meaningless there). On CUDA, the default ``clear_fn`` is
        ``torch.cuda.empty_cache`` and the default cadence comes from
        :func:`cache_clear_cadence`.
        """
        on_cuda = getattr(device, "type", None) == "cuda"
        if not on_cuda:
            return cls(every=0, clear_fn=lambda: None)

        if every is None:
            every = cache_clear_cadence()
        if clear_fn is None:
            import torch

            clear_fn = torch.cuda.empty_cache
        return cls(every=every, clear_fn=clear_fn)

    def step(self) -> bool:
        """Advance the counter; clear + return True on the cadence boundary."""
        if self._every <= 0:
            return False
        self._count += 1
        if self._count >= self._every:
            self._count = 0
            self._clear_fn()
            return True
        return False
