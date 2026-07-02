"""Serializing inference guard.

A single shared instance caps GPU inference to one in-flight call per process.
Bot moves block until the guard is free; analysis endpoints can request
non-blocking service and map ``InferenceBusy`` to HTTP 503.

Known limitation: this only serializes within one process. It does NOT arbitrate
GPU contention with other processes (e.g. a concurrent training run or a second
server). A true cross-process cap would require a dedicated inference service.
"""
import threading


class InferenceBusy(Exception):
    """Raised when the guard is held and ``blocking=False`` was requested."""


class InferenceGuard:
    """Process-local serializing guard for GPU inference.

    This is also the swappable seam for a future batched inference queue:
    callers pass a thunk ``fn`` that performs model forward passes, and the
    guard decides when it runs.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def run(self, fn, blocking: bool = True):
        """Execute ``fn`` under the guard.

        Args:
            fn: Callable returning the inference result.
            blocking: If True (default), wait until the guard is free. If False,
                raise ``InferenceBusy`` immediately when another call is active.

        Returns:
            The value returned by ``fn``.

        Raises:
            InferenceBusy: If ``blocking=False`` and the guard is held.
        """
        acquired = self._lock.acquire(blocking=blocking)
        if not acquired:
            raise InferenceBusy("inference guard is busy")
        try:
            return fn()
        finally:
            self._lock.release()
