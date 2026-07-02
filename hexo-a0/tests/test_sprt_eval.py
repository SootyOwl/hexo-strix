"""Tests for SPRT decision math, sliding-window sizing, and swap-freeze logic.

These lock in the 2026-06-04 sizing decision: the sliding window must exceed
the natural decision horizon for the H1 effect size (~412 games at s1=0.55),
otherwise the windowed LLR plateaus below the accept bound and a genuinely
stronger trainee can never be promoted (the "dead band"). See
docs/research/ for the derivation.
"""
from __future__ import annotations

from hexo_a0.sprt_eval import (
    SPRTConfig,
    SPRTState,
    checkpoint_step,
    restore_decision,
    swap_decision,
)


def _sustained_stream(n_games: int, win_rate: float) -> list[str]:
    """Deterministic W/L stream with W's distributed evenly (Bresenham-style).

    Keeps the running mean close to `win_rate` throughout so there are no
    transient LLR excursions — the decision reflects the steady state, not
    an ordering artifact.
    """
    out: list[str] = []
    num = round(win_rate * 100)  # e.g. 0.55 -> 55 per 100
    den = 100
    for i in range(n_games):
        out.append("W" if (i * num) % den < num else "L")
    return out


def test_default_window_exceeds_s1_decision_horizon():
    """Default window must clear the ~412-game accept horizon for s1=0.55."""
    cfg = SPRTConfig()
    # Decision horizon = U / per-pair drift at s1, in games.
    assert cfg.window_size is not None
    assert cfg.window_size >= 412
    assert cfg.window_size == 1000


def test_sustained_s1_stream_accepts_with_default_window():
    """A sustained s1=0.55 trainee reaches accept_h1 under the default window."""
    cfg = SPRTConfig()  # window 1000
    state = SPRTState()
    for o in _sustained_stream(1000, 0.55):
        state.record(o, cfg)
    assert state.decision == "accept_h1"


def test_sustained_s1_stream_stalls_under_narrow_window():
    """The old window=400 plateaus an s1 trainee below accept — the dead band."""
    cfg = SPRTConfig(window_size=400)
    state = SPRTState()
    for o in _sustained_stream(1000, 0.55):
        state.record(o, cfg)
    _, upper = cfg.bounds()
    assert state.decision == "continue"
    assert state.llr < upper


def test_swap_freeze_threshold_is_fraction_of_upper():
    cfg = SPRTConfig()
    _, upper = cfg.bounds()
    assert cfg.swap_freeze_fraction == 0.9
    assert cfg.swap_freeze_threshold() == 0.9 * upper


def test_swap_not_frozen_below_threshold():
    cfg = SPRTConfig()
    freeze_active, stall = swap_decision(
        llr=cfg.swap_freeze_threshold() - 0.01, frozen_games=0, cfg=cfg
    )
    assert freeze_active is False
    assert stall is False


def test_swap_frozen_in_zone_before_stall():
    cfg = SPRTConfig()  # window 1000
    freeze_active, stall = swap_decision(
        llr=cfg.swap_freeze_threshold() + 0.01, frozen_games=500, cfg=cfg
    )
    assert freeze_active is True
    assert stall is False


def test_swap_stall_unfreezes_after_full_window_frozen():
    cfg = SPRTConfig()  # window 1000
    freeze_active, stall = swap_decision(
        llr=cfg.swap_freeze_threshold() + 0.01, frozen_games=1000, cfg=cfg
    )
    assert freeze_active is False
    assert stall is True


def test_swap_stall_uses_fallback_limit_when_unbounded():
    cfg = SPRTConfig(window_size=None)
    high = cfg.swap_freeze_threshold() + 0.01
    # Below fallback → still frozen
    fa_lo, st_lo = swap_decision(llr=high, frozen_games=999, cfg=cfg, stall_fallback=1000)
    assert (fa_lo, st_lo) == (True, False)
    # At/above fallback → stall-unfreeze
    fa_hi, st_hi = swap_decision(llr=high, frozen_games=1000, cfg=cfg, stall_fallback=1000)
    assert (fa_hi, st_hi) == (False, True)


def test_from_dict_drops_trailing_odd_outcome_to_keep_pairs_aligned():
    """A daemon killed mid-pair can persist an odd-length outcome deque.

    With an unbounded window (no eviction ever re-aligns it) a single orphaned
    outcome would mis-phase every subsequent pentanomial pair for the rest of
    the round. Restore must drop a trailing odd outcome so the deque is always
    an even number of (P1, P2) units.
    """
    cfg = SPRTConfig(window_size=None, pentanomial=True)
    state = SPRTState()
    for o in "WLWLW":  # odd length: 5 outcomes
        state.record(o, cfg)
    restored = SPRTState.from_dict(state.to_dict())
    assert len(restored._outcomes) == 4
    assert "".join(restored._outcomes) == "WLWL"


def test_checkpoint_step_parses_step_and_rejects_other_names():
    assert checkpoint_step("runs/x/checkpoints/checkpoint_00024900.pt") == 24900
    assert checkpoint_step("checkpoint_0.pt") == 0
    assert checkpoint_step("runs/x/checkpoints/self_play/champion.pt") is None


def test_restore_decision_allows_same_champion_fresh_trainee():
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=24900, latest_trainee_step=26198,
    )
    assert ok is True
    assert reasons == []


def test_restore_decision_blocks_changed_champion():
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=False,
        prior_trainee_step=24900, latest_trainee_step=24900,
    )
    assert ok is False
    assert any("champion" in r for r in reasons)


def test_restore_decision_blocks_terminal_decision():
    ok, reasons = restore_decision(
        prior_decision="accept_h1", same_champion=True,
        prior_trainee_step=24900, latest_trainee_step=24900,
    )
    assert ok is False
    assert any("accept_h1" in r for r in reasons)


def test_restore_decision_blocks_obsolete_trainee_evidence():
    """The staleness gate is TRAINING PROGRESS, not wall-clock: a round whose
    evidence was earned >max_trainee_lag steps behind the current latest
    checkpoint describes an obsolete network. Wall-clock was the wrong proxy
    (2026-06-11: a wedged-then-restarted round was silently discarded at
    4742s > 3600s while champion and trainee were both still current)."""
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=10000, latest_trainee_step=20000,
    )
    assert ok is False
    assert any("step" in r for r in reasons)
    # exactly at the limit is still fine
    ok, _ = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=10000, latest_trainee_step=15000,
    )
    assert ok is True


def test_restore_decision_unknown_steps_do_not_block():
    """Champion identity is the load-bearing gate; unparseable checkpoint
    names must not discard an otherwise-valid round."""
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=None, latest_trainee_step=26198,
    )
    assert ok is True and reasons == []
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=24900, latest_trainee_step=None,
    )
    assert ok is True and reasons == []


def test_restore_decision_collects_all_failed_gates():
    ok, reasons = restore_decision(
        prior_decision="reject_h1", same_champion=False,
        prior_trainee_step=0, latest_trainee_step=99999,
    )
    assert ok is False
    assert len(reasons) == 3
