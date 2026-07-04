"""solve_forcing must release the GIL while it searches.

The play server runs forcing solves inside HTTP handler threads; a solve that
holds the GIL freezes every other thread in the process (all request handling,
bot moves, even 503 responses) for its full runtime — seconds at the analysis
budgets. These tests pin the binding's ``py.allow_threads`` behavior.
"""
import threading
import time

import pytest


def _budget_burning_state():
    """A position with no forced win but a large forcing-candidate set: eight
    separated P1 three-in-a-row segments make the AND-OR search exhaust a 250k
    node budget (~1s) instead of exiting early."""
    import hexo_rs as hr
    cfg = hr.GameConfig(6, 8, 400)
    stones = [((0, 0), "P1")]
    for k in range(8):
        base_r = (k + 1) * 7
        for i in range(3):
            stones.append(((i, base_r), "P1"))
        stones.append(((5, base_r), "P2"))
        stones.append(((-2, base_r), "P2"))
    return hr.GameState.from_state(stones, "P1", 2, cfg)


def test_solve_forcing_releases_gil():
    import hexo_rs as hr
    state = _budget_burning_state()

    # Calibrate: starvation is only observable if the solve itself is long.
    t0 = time.perf_counter()
    assert hr.solve_forcing(state, 12, 250_000) is None
    solve_s = time.perf_counter() - t0
    if solve_s < 0.3:
        pytest.skip(f"probe position solves too fast ({solve_s:.3f}s) "
                    "to observe GIL starvation")

    go = threading.Event()
    done = threading.Event()

    def solver():
        go.wait()
        hr.solve_forcing(state, 12, 250_000)
        done.set()

    t = threading.Thread(target=solver)
    t.start()
    go.set()
    # sleep() releases the GIL and must RE-acquire it to return. If the solver
    # holds the GIL for its whole runtime, this 50ms sleep can't return until
    # the solve finishes (~solve_s); with allow_threads it returns on schedule.
    t1 = time.perf_counter()
    time.sleep(0.05)
    blocked = time.perf_counter() - t1
    still_running = not done.is_set()
    t.join()

    assert blocked < 0.5 * solve_s, (
        f"main thread starved for {blocked:.3f}s during a {solve_s:.3f}s "
        "solve — solve_forcing is holding the GIL"
    )
    assert still_running, "solve finished before we could measure; probe too fast"
