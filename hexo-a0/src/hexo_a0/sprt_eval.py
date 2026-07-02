"""Sequential Probability Ratio Test for continuous model evaluation.

Pure library: SPRT math and state management. No I/O, no subprocess concerns.

SPRT tests two hypotheses about the trainee's per-game score (W=1, D=0.5, L=0):
  H0: mean score = s0 (typically 0.50, equal strength)
  H1: mean score = s1 (typically 0.55, ~35 Elo advantage — worth promoting)

The log-likelihood ratio accumulates additively across games under a Normal
approximation. Wald's bounds give early-stopping:
  accept H1 when LLR >= log((1-beta)/alpha)
  reject H1 when LLR <= log(beta/(1-alpha))
  (alpha=beta=0.05 → bounds ≈ ±2.944)

Trainee-swap assumption: the daemon reloads the trainee when newer checkpoints
appear but does NOT reset SPRT state on a swap. Under monotonic improvement
this is conservative (old games with weaker trainee underestimate H1). To
guard against the degradation case — where stale high-win games could
spuriously inflate LLR — SPRTState maintains a sliding window of the last N
outcomes. Older games are dropped as new ones arrive, bounding how much
history a single (possibly bad) trainee can influence. Set window_size=None
for unbounded accumulation.

Pentanomial: when enabled (default), games are grouped into consecutive pairs
(trainee-as-P1 followed by trainee-as-P2 against the same opponent), and the
pair score sum ∈ {0, 0.5, 1.0, 1.5, 2.0} is the statistical unit. This cancels
out side asymmetry: a trainee benefiting from first-player advantage in game
N and losing it in game N+1 produces a pair score of 1.0 (neutral), while
two wins (WW=2.0) or two losses (LL=0.0) give unambiguously strong signal.
Lower variance than trinomial for a given skill gap when P1/P2 are asymmetric.
"""
from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Deque, Iterable, Literal


Decision = Literal["continue", "accept_h1", "reject_h1"]


_OUTCOME_SCORE = {"W": 1.0, "D": 0.5, "L": 0.0}


@dataclass
class SPRTConfig:
    """SPRT hypothesis + decision parameters."""
    s0: float = 0.50
    s1: float = 0.55
    alpha: float = 0.05
    beta: float = 0.05
    # Conservative upper bound on per-game score variance. 0.25 is the max
    # for [0,1] outcomes (Bernoulli at p=0.5). Used when pentanomial=False.
    variance: float = 0.25
    # Per-pair score variance for pentanomial mode. Pair scores are in
    # [0, 2]; the independent-games upper bound is 2 * variance = 0.5, but
    # actual pair variance is lower when pairs are correlated (P1/P2
    # asymmetry: winning as P1 and losing as P2 reduces the {0, 2} tails).
    # Chess-engine tournament data typically puts pair variance at 0.30–0.40;
    # 0.35 is a reasonable default. Lowering this makes pentanomial
    # converge faster for a given true skill gap; too low risks
    # over-confident accepts on spurious pair correlation.
    pair_variance: float = 0.35
    # Sliding window: drop oldest GAMES once this many have been recorded.
    # In pentanomial mode, pairs are formed from consecutive games within the
    # current window. None = unbounded accumulation.
    #
    # Sizing constraint: the window is also the averaging horizon for a moving
    # (swapped) trainee, so it must exceed the natural decision horizon for the
    # H1 effect size. At s1=0.55 the per-pair LLR drift is
    # 2(s1-s0)/pair_variance * (2*s1 - (s0+s1)) ≈ 0.0143, so accept needs
    # ~U/0.0143 ≈ 206 pairs = ~412 games. A window below that plateaus a
    # genuinely-stronger trainee below the accept bound forever (the "dead
    # band"). 1000 (≈2.5× the horizon) gives full power at s1 while still
    # bounding staleness to a few hundred recent games. See docs/research/.
    window_size: int | None = 1000
    # Trainee-swap freeze: the SPRT daemon stops loading newer trainee
    # checkpoints once LLR climbs into the near-accept zone, so a converging
    # round isn't diluted by a weaker mid-training snapshot. Expressed as a
    # fraction of the accept (upper) bound. 0.9 keeps the freeze narrow — it
    # only fires when a round is genuinely about to close — which avoids a
    # deadlock sliver just below the accept threshold and lets the daemon keep
    # swapping in fresh checkpoints for a current view of the network.
    swap_freeze_fraction: float = 0.9
    # Pentanomial mode: group consecutive games into pairs (trainee-as-P1
    # then trainee-as-P2) and use pair scores as the statistical unit.
    # Cancels out first-player advantage. False = classic per-game trinomial.
    pentanomial: bool = True
    # Adaptive pair variance: when True, the daemon tracks an EMA of round-end
    # empirical pair variance (clamped to [floor, ceil]) and updates
    # pair_variance between rounds. Resets on champion change. Library doesn't
    # act on these — daemon does — but they live here for single source of truth.
    adaptive_pair_variance: bool = False
    adaptive_pair_variance_floor: float = 0.20
    adaptive_pair_variance_ceil: float = 0.50
    adaptive_pair_variance_ema_alpha: float = 0.30

    def bounds(self) -> tuple[float, float]:
        """Return (lower, upper) LLR thresholds. Lower → reject H1, upper → accept H1."""
        lower = math.log(self.beta / (1.0 - self.alpha))
        upper = math.log((1.0 - self.beta) / self.alpha)
        return lower, upper

    def swap_freeze_threshold(self) -> float:
        """LLR at/above which trainee swaps are frozen (fraction of the accept bound)."""
        _, upper = self.bounds()
        return self.swap_freeze_fraction * upper


@dataclass
class SPRTState:
    """Running SPRT statistics over a (possibly sliding) window of games.

    Internal state: a deque of recent outcomes (W/D/L). Summary counters
    (games, wins, draws, losses, total_score) are kept in sync with the
    deque and are what gets serialized — they always describe the *current
    window*, not lifetime totals.

    In pentanomial mode, consecutive outcomes are grouped into pairs at LLR
    computation time; the `pairs` / `pair_score_total` fields are derived
    from the outcome deque and refreshed on each record. Pair alignment uses
    the first two outcomes in the deque as pair 1, next two as pair 2, etc.
    If the deque has odd length (during transition), the trailing single
    outcome is ignored until its partner arrives.
    """
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    total_score: float = 0.0
    # Pentanomial summaries, derived from _outcomes on each record().
    # 0 in trinomial mode.
    pairs: int = 0
    pair_score_total: float = 0.0
    llr: float = 0.0
    decision: Decision = "continue"
    # Not serialized directly; reconstructable from the summary if ever needed.
    _outcomes: Deque[str] = field(default_factory=deque, repr=False)

    def record(self, outcome: Literal["W", "D", "L"], sprt_cfg: SPRTConfig) -> None:
        """Append one game; drop the oldest if window is exceeded; update LLR."""
        if outcome not in _OUTCOME_SCORE:
            raise ValueError(f"outcome must be W/D/L, got {outcome!r}")
        self._outcomes.append(outcome)
        self._add_counts(outcome, +1)

        if sprt_cfg.window_size is not None:
            while len(self._outcomes) > sprt_cfg.window_size:
                dropped = self._outcomes.popleft()
                self._add_counts(dropped, -1)

        if sprt_cfg.pentanomial:
            self.pair_score_total, self.pairs = _pair_stats(self._outcomes)
            self.llr = compute_llr_pentanomial(
                self.pair_score_total, self.pairs, sprt_cfg,
            )
        else:
            self.pairs = 0
            self.pair_score_total = 0.0
            self.llr = compute_llr(self.total_score, self.games, sprt_cfg)
        self.decision = classify(self.llr, sprt_cfg)

    def _add_counts(self, outcome: str, delta: int) -> None:
        """Apply +1/-1 to the summary counters for a given outcome."""
        if outcome == "W":
            self.wins += delta
        elif outcome == "D":
            self.draws += delta
        elif outcome == "L":
            self.losses += delta
        self.games += delta
        self.total_score += delta * _OUTCOME_SCORE[outcome]

    def reset(self) -> None:
        """Clear all counters. Use on champion promotion."""
        self._outcomes.clear()
        self.games = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.total_score = 0.0
        self.llr = 0.0
        self.decision = "continue"

    @property
    def score(self) -> float:
        return self.total_score / self.games if self.games > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_outcomes", None)
        # Serialize the outcomes deque under a non-underscore key so the
        # daemon can restore mid-round state across restarts. Encoded as a
        # single string ("WLDWLD...") rather than a list to keep the state
        # file compact (~6× smaller at full window vs JSON list with indent).
        # The summary counters alone aren't sufficient in pentanomial mode:
        # pair_stats is recomputed from the deque on each record().
        d["outcomes"] = "".join(self._outcomes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SPRTState":
        """Reconstruct a state from a to_dict() payload.

        Round-trip preserves the outcomes deque so subsequent record() calls
        see the correct sliding-window and pair history. Unknown keys are
        ignored. Missing keys default to dataclass defaults. Accepts both
        the new string encoding ("WLDW...") and the legacy list form
        (["W", "L", "D", "W"]) for backward-compatibility.
        """
        outcomes_raw = d.get("outcomes") or ""
        # Backward-compat: legacy list form is auto-detected
        if isinstance(outcomes_raw, list):
            outcomes_iter: Iterable[str] = outcomes_raw
        else:
            outcomes_iter = str(outcomes_raw)
        state = cls(
            games=int(d.get("games", 0)),
            wins=int(d.get("wins", 0)),
            draws=int(d.get("draws", 0)),
            losses=int(d.get("losses", 0)),
            total_score=float(d.get("total_score", 0.0)),
            pairs=int(d.get("pairs", 0)),
            pair_score_total=float(d.get("pair_score_total", 0.0)),
            llr=float(d.get("llr", 0.0)),
            decision=d.get("decision", "continue"),
        )
        state._outcomes = deque(o for o in outcomes_iter if o in _OUTCOME_SCORE)
        # Drop a trailing odd outcome so the deque is always an even number of
        # (P1, P2) pentanomial units. A daemon killed mid-pair can persist an
        # odd-length deque; with an unbounded window no eviction ever re-aligns
        # it, so a single orphan would mis-phase every subsequent pair for the
        # rest of the round. The orphan was never part of a pair (_pair_stats
        # ignores it), so pairs/pair_score_total/llr are unaffected — only the
        # per-outcome summary counters need fixing up.
        if len(state._outcomes) % 2 == 1:
            orphan = state._outcomes.pop()
            state._add_counts(orphan, -1)
        return state


def compute_llr(total_score: float, n_games: int, cfg: SPRTConfig) -> float:
    """Normal-approximation SPRT LLR for per-game scores (trinomial mode).

    Derived from assuming each game score ~ N(mu, variance) with mu=s0 under H0
    and mu=s1 under H1. LLR per game = (s1-s0)/variance * (score - (s0+s1)/2),
    accumulated linearly over games.
    """
    if cfg.variance <= 0 or n_games == 0:
        return 0.0
    midpoint = 0.5 * (cfg.s0 + cfg.s1)
    return (cfg.s1 - cfg.s0) / cfg.variance * (total_score - n_games * midpoint)


def compute_llr_pentanomial(
    pair_score_total: float, n_pairs: int, cfg: SPRTConfig,
) -> float:
    """Normal-approximation SPRT LLR for per-pair scores (pentanomial mode).

    Pair score ∈ {0, 0.5, 1.0, 1.5, 2.0} has mean 2*s under hypothesis s.
    So s0_pair = 2*cfg.s0, s1_pair = 2*cfg.s1, midpoint_pair = cfg.s0 + cfg.s1.
    LLR per pair = (s1_pair - s0_pair)/pair_variance * (pair_score - midpoint_pair)
                 = 2*(s1 - s0)/pair_variance * (pair_score - (s0 + s1))
    accumulated over pairs.
    """
    if cfg.pair_variance <= 0 or n_pairs == 0:
        return 0.0
    midpoint = cfg.s0 + cfg.s1
    return 2.0 * (cfg.s1 - cfg.s0) / cfg.pair_variance * (
        pair_score_total - n_pairs * midpoint
    )


def _pair_stats(outcomes: Deque[str]) -> tuple[float, int]:
    """Sum of pair scores and pair count from an outcome deque.

    Pairs are (outcomes[0], outcomes[1]), (outcomes[2], outcomes[3]), ...
    A trailing unpaired outcome (odd length) is ignored.
    """
    n_pairs = len(outcomes) // 2
    total = 0.0
    # deque indexing is O(n) for middle indices but we iterate sequentially.
    it = iter(outcomes)
    for _ in range(n_pairs):
        a = _OUTCOME_SCORE[next(it)]
        b = _OUTCOME_SCORE[next(it)]
        total += a + b
    return total, n_pairs


def sample_pair_variance(outcomes: Deque[str]) -> float | None:
    """Empirical (population) variance of pair scores in the outcome deque.

    Returns None if there are fewer than 2 complete pairs (variance undefined).
    Useful for calibrating cfg.pair_variance to observed data: if this
    consistently reports values much lower or higher than the configured
    pair_variance, SPRT is being too slow or too aggressive respectively.
    """
    n_pairs = len(outcomes) // 2
    if n_pairs < 2:
        return None
    scores: list[float] = []
    it = iter(outcomes)
    for _ in range(n_pairs):
        scores.append(_OUTCOME_SCORE[next(it)] + _OUTCOME_SCORE[next(it)])
    mean = sum(scores) / n_pairs
    return sum((s - mean) ** 2 for s in scores) / n_pairs


def classify(llr: float, cfg: SPRTConfig) -> Decision:
    """Map current LLR to a decision given the config's Wald bounds."""
    lower, upper = cfg.bounds()
    if llr >= upper:
        return "accept_h1"
    if llr <= lower:
        return "reject_h1"
    return "continue"


def swap_decision(
    llr: float,
    frozen_games: int,
    cfg: SPRTConfig,
    *,
    stall_fallback: int = 1000,
) -> tuple[bool, bool]:
    """Decide whether to hold (freeze) the current trainee or allow a swap.

    Returns (freeze_active, stall_unfreeze):
      - freeze_active: hold the current trainee, don't load newer checkpoints.
      - stall_unfreeze: the round has stayed frozen for a full window of games
        without closing (likely stuck at a sub-accept plateau in the residual
        dead band), so force-allow a swap to a fresher checkpoint.

    `frozen_games` is the count of consecutive games played while LLR was in
    the freeze zone. `stall_fallback` is used as the stall limit when the
    window is unbounded (window_size is None).
    """
    in_zone = llr >= cfg.swap_freeze_threshold()
    limit = cfg.window_size if cfg.window_size is not None else stall_fallback
    stall_unfreeze = in_zone and frozen_games >= limit
    freeze_active = in_zone and not stall_unfreeze
    return freeze_active, stall_unfreeze


def checkpoint_step(path: str | Path) -> int | None:
    """Step number embedded in a checkpoint_<step>.pt filename, or None."""
    m = re.match(r"checkpoint_(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def restore_decision(
    prior_decision: str,
    same_champion: bool,
    prior_trainee_step: int | None,
    latest_trainee_step: int | None,
    *,
    max_trainee_lag: int = 5000,
) -> tuple[bool, list[str]]:
    """Decide whether a saved mid-round SPRT state may be restored on start.

    Returns (restore, reasons): restore is True when every gate passes;
    reasons lists each failed gate for the operator-facing skip log.

    Gates:
      - champion identity: a different champion means a different round; all
        prior evidence is about the wrong opponent.
      - terminal decision: terminal rounds are handled by the daemon's normal
        post-game flow; restoring one directly would re-trigger that handling
        without a matching game record.
      - trainee staleness, measured in TRAINING STEPS, not wall-clock: a
        round whose games were earned more than max_trainee_lag steps behind
        the current latest checkpoint describes an obsolete network — under
        non-monotonic improvement its evidence could promote a regressed
        checkpoint. Wall-clock was the wrong proxy: a stalled round erodes
        the clock before shutdown, and a paused-then-resumed run trips it
        with perfectly fresh evidence (both bit on 2026-06-11). A round that
        spans a pause where no training happened is exactly as valid as one
        played straight through.

    Unknown steps (unparseable checkpoint names) do not block — champion
    identity is the load-bearing safety gate.
    """
    reasons: list[str] = []
    if not same_champion:
        reasons.append("champion file changed since state was written")
    if prior_decision != "continue":
        reasons.append(f"prior round already decided ({prior_decision})")
    if (
        prior_trainee_step is not None
        and latest_trainee_step is not None
        and latest_trainee_step - prior_trainee_step > max_trainee_lag
    ):
        reasons.append(
            f"trainee advanced {latest_trainee_step - prior_trainee_step} steps "
            f"since the round's last game (limit {max_trainee_lag})"
        )
    return not reasons, reasons
