"""Torch-free helpers for the native (hexo-infer) serving path."""
import math


def softmax(xs):
    """Numerically-stable softmax over a Python list of floats."""
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    if total <= 0.0:
        n = len(xs)
        return [1.0 / n] * n
    return [e / total for e in exps]


class NativeModel:
    """Lightweight stand-in for a torch HeXONet on the native serve path.

    Carries the ``_hexo_native`` engine the serving hot-paths already look for.
    It is intentionally NOT callable: every torch ``model(...)`` fallback site
    is guarded by ``if native is not None``, so reaching ``__call__`` means a
    site was missed — fail loud rather than silently import torch.
    """

    def __init__(self, infer):
        self._hexo_native = infer

    def __call__(self, *a, **k):
        raise RuntimeError("torch eval path unreachable in native mode")
