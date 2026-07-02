"""Pure dispatch/drain/inflight bookkeeping for parallel SPRT pair-eval.

The SPRT daemon (sprt_daemon.py) plays games serially. When sprt_eval_workers
> 1 it instead farms whole *pairs* (model-as-P1 + model-as-P2) out to a pool of
worker processes. This module holds the process-free coordination logic so the
tricky invariants are unit-testable without real subprocesses:

  - inflight accounting that survives worker death (reclaim + re-dispatch),
  - champion-epoch staleness (discard pairs played against a swapped-out
    champion before they pollute the fresh round),
  - dispatch gating (no new work once a decision is reached),
  - per-pair timeout (reissue a silently-stuck pair).

A "pair" is recorded atomically as two consecutive outcomes, which keeps the
pentanomial positional pairing in SPRTState color-aligned regardless of which
worker finishes first.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Inflight:
    worker_id: object
    epoch: int
    dispatched_at: float


class PairCoordinator:
    """Tracks which pairs are in flight, on which worker, under which epoch."""

    def __init__(
        self,
        n_workers: int,
        target_inflight: int | None = None,
        pair_timeout: float = 600.0,
    ) -> None:
        self.n_workers = n_workers
        self.target_inflight = target_inflight if target_inflight is not None else 2 * n_workers
        self.pair_timeout = pair_timeout
        self._inflight: dict[int, _Inflight] = {}

    def inflight_count(self) -> int:
        return len(self._inflight)

    def on_dispatch(self, pair_id: int, worker_id: object, epoch: int, now: float) -> None:
        self._inflight[pair_id] = _Inflight(worker_id, epoch, now)

    def reclaim_dead(self, dead_worker_ids: set) -> list[int]:
        """Remove and return pair_ids assigned to dead workers for re-dispatch."""
        reclaimed = [
            pid for pid, inf in self._inflight.items() if inf.worker_id in dead_worker_ids
        ]
        for pid in reclaimed:
            del self._inflight[pid]
        return reclaimed

    def on_result(self, pair_id: int, played_epoch: int, current_epoch: int) -> bool:
        """Retire an inflight pair. Return True if it should be recorded.

        Discarded (return False, no record) when:
          - the pair_id is no longer in flight: it was already retired by a
            prior result, or reclaimed (dead/timed-out worker) and re-dispatched.
            Both the original and the reissue can complete; only the first to
            arrive counts, else the pair double-counts and mis-phases pairing.
          - the pair was played against a since-swapped champion (played_epoch
            != current_epoch): stale evidence for the fresh round.
        """
        if self._inflight.pop(pair_id, None) is None:
            return False  # already retired (duplicate / reissued original)
        return played_epoch == current_epoch

    def reclaim_timed_out(self, now: float) -> list[int]:
        """Remove and return pair_ids in flight longer than pair_timeout.

        Guards against a worker stuck in a native call that neither returns a
        result nor dies — without this the inflight slot leaks and dispatch
        eventually wedges.
        """
        stuck = [
            pid for pid, inf in self._inflight.items()
            if now - inf.dispatched_at > self.pair_timeout
        ]
        for pid in stuck:
            del self._inflight[pid]
        return stuck

    def dispatch_quota(self, decision: str) -> int:
        """How many new pairs to dispatch this cycle.

        Zero once a decision is reached (drain-only until the round resets).

        Deliberately NOT gated on the trainee-swap freeze: the freeze only
        exits via played games (LLR moving past a bound, or frozen_games
        reaching a window for stall-unfreeze), so halting dispatch while
        frozen deadlocks the round once the inflight pairs drain. The held
        trainee snapshot is exactly what the frozen round must keep playing.
        """
        if decision != "continue":
            return 0
        return max(0, self.target_inflight - len(self._inflight))
