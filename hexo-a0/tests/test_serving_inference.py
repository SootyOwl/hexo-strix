"""Tests for the serializing inference guard.

One shared instance serializes all GPU inference in the process. Bot moves
block; analysis endpoints can non-block and surface HTTP 503 via InferenceBusy.
"""
import threading
import time

from hexo_a0.serving.inference import InferenceGuard, InferenceBusy


def test_run_returns_value():
    assert InferenceGuard().run(lambda: 42) == 42


def test_non_blocking_raises_when_held():
    g = InferenceGuard()
    held = threading.Event()
    release = threading.Event()

    def hog():
        held.set()
        release.wait()
        return 1

    t = threading.Thread(target=lambda: g.run(hog))
    t.start()
    held.wait()
    try:
        with __import__("pytest").raises(InferenceBusy):
            g.run(lambda: 2, blocking=False)
    finally:
        release.set()
        t.join()


def test_serializes_calls():
    g = InferenceGuard()
    overlap = []
    inside = [0]
    lk = threading.Lock()

    def work():
        with lk:
            inside[0] += 1
            overlap.append(inside[0])
        time.sleep(0.02)
        with lk:
            inside[0] -= 1
        return 1

    ts = [threading.Thread(target=lambda: g.run(work)) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert max(overlap) == 1  # never two inside the guard at once


def test_capacity_two_admits_two_concurrent_runs():
    g = InferenceGuard(capacity=2)
    # Both workers must be inside the guard at the same time to pass the
    # barrier; a serializing guard would deadlock it (→ BrokenBarrierError).
    barrier = threading.Barrier(2, timeout=2.0)
    results = []
    lk = threading.Lock()

    def work():
        barrier.wait()
        return 1

    def runner():
        v = g.run(work)
        with lk:
            results.append(v)

    ts = [threading.Thread(target=runner) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert results == [1, 1]


def test_capacity_two_still_caps_at_two():
    g = InferenceGuard(capacity=2)
    overlap = []
    inside = [0]
    lk = threading.Lock()

    def work():
        with lk:
            inside[0] += 1
            overlap.append(inside[0])
        time.sleep(0.02)
        with lk:
            inside[0] -= 1
        return 1

    ts = [threading.Thread(target=lambda: g.run(work)) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert max(overlap) == 2


def test_timeout_raises_busy_after_bounded_wait():
    import pytest
    g = InferenceGuard()
    held = threading.Event()
    release = threading.Event()

    def hog():
        held.set()
        release.wait()
        return 1

    t = threading.Thread(target=lambda: g.run(hog))
    t.start()
    held.wait()
    try:
        t0 = time.perf_counter()
        with pytest.raises(InferenceBusy):
            g.run(lambda: 2, timeout=0.05)
        waited = time.perf_counter() - t0
        assert waited >= 0.04  # actually waited, didn't fail fast
    finally:
        release.set()
        t.join()


def test_timeout_succeeds_when_guard_frees_in_time():
    g = InferenceGuard()
    held = threading.Event()

    def hog():
        held.set()
        time.sleep(0.05)
        return 1

    t = threading.Thread(target=lambda: g.run(hog))
    t.start()
    held.wait()
    try:
        assert g.run(lambda: 2, timeout=2.0) == 2
    finally:
        t.join()
