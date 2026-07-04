"""Serializing inference guard.

A single shared instance caps model inference to ``capacity`` in-flight calls
per process (default 1). Bot moves block until a slot is free; analysis
endpoints can bound their wait with ``timeout`` (or non-block outright) and map
``InferenceBusy`` to HTTP 503. Capacities above 1 are for CPU inference, where
concurrent forward passes/solves merely share cores — on a GPU keep capacity 1.

Known limitation: this only arbitrates within one process. It does NOT cap
contention with other processes (e.g. a concurrent training run or a second
server). A true cross-process cap would require a dedicated inference service.
"""
import threading


class InferenceBusy(Exception):
    """Raised when no slot frees up within the requested wait."""


class InferenceGuard:
    """Process-local guard capping concurrent inference.

    This is also the swappable seam for a future batched inference queue:
    callers pass a thunk ``fn`` that performs model forward passes, and the
    guard decides when it runs.
    """

    def __init__(self, capacity: int = 1):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1 (got {capacity})")
        self.capacity = capacity
        self._sem = threading.BoundedSemaphore(capacity)

    def run(self, fn, blocking: bool = True, timeout: float | None = None):
        """Execute ``fn`` under the guard.

        Args:
            fn: Callable returning the inference result.
            blocking: If True (default), wait until a slot is free. If False,
                raise ``InferenceBusy`` immediately when all slots are taken.
            timeout: If set, wait at most this many seconds for a slot, then
                raise ``InferenceBusy``. Overrides ``blocking``.

        Returns:
            The value returned by ``fn``.

        Raises:
            InferenceBusy: If no slot was acquired within the requested wait.
        """
        if timeout is not None:
            acquired = self._sem.acquire(timeout=timeout)
        else:
            acquired = self._sem.acquire(blocking=blocking)
        if not acquired:
            raise InferenceBusy("inference guard is busy")
        try:
            return fn()
        finally:
            self._sem.release()
