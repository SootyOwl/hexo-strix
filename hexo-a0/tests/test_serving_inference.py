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
