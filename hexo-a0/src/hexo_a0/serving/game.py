"""Game records, errors, and the in-memory game manager.

Ported from scripts/play_server.py: GameRecord/GameError (274-307), difficulty
tiers (264-271), make_bot_turn_fn (310-400), and GameManager (456-663).

``make_bot_turn_fn`` is rewritten to accept an ``InferenceGuard`` and to reuse
``hexo_a0.serving.model.make_graph_fn`` instead of inlining graph construction.
"""
import json
import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from hexo_a0.serving.inference import InferenceGuard
from hexo_a0.serving.model import make_graph_fn

logger = logging.getLogger(__name__)


# Difficulty tiers. Labels are stable (UI-visible); sim counts are tunable via
# --difficulty-sims at startup. See
# docs/research/2026-05-14-sim-count-strength-curve.md for the Elo gaps.
DIFFICULTY_ORDER = ("casual", "easy", "standard", "strong")
DEFAULT_DIFFICULTY_SIMS: dict[str, int] = {
    "casual":   16,
    "easy":     32,
    "standard": 64,
    "strong": 128,
}
DEFAULT_DIFFICULTY = "standard"

# Per-difficulty forcing-solver DEPTH cap (see LIVE_NODE_BUDGET/DEF_NODE_BUDGET
# below for why node budgets stay flat while depth is the tier lever): all
# tiers get the SAME honest forcing tactics (own-win execution + defense, see
# `_play`), but a shallower depth cap means weaker tiers only catch shallow
# mates and let deeper ones through, same as a human of that strength would.
# "casual" only executes/blocks depth-2 (a single-move) mate.
#
# standard 6->8 / strong 10->16 raised 2026-07-05: at the FLAT 20k node
# budget, depth is wall-clock-free (p99 352-392 ms and max <750 ms at every
# depth in 10..18 over 640 real-game prefix solves, both sides — see
# scripts/forcing_depth_sweep.py + docs/research/). The same sweep showed the
# extra depth finds nothing EARLIER on the strongloss fixtures (threats built
# from quiet moves aren't VCFs until complete), so this is headroom for rare
# longer fully-forcing lines, not the fix for those losses — that was the
# pair-aware `hr.solve_defense` (see `_defensive_move`).
DEFAULT_DIFFICULTY_FORCING_DEPTH: dict[str, int] = {
    "casual":   2,
    "easy":     4,
    "standard": 8,
    "strong":  16,
}

# Live-play forcing (VCF) search NODE budgets, re-tuned 2026-07-04 from r=8
# NN-guided-position measurement (the actual live config), replacing the
# original from-first-principles 2_000_000 and 200_000 guesses:
#
#   depth/budget      p99 solve    max solve   win coverage
#   16 / 2_000_000     10.4 s        15.9 s        22.5%
#   12 /   250_000      2.5 s         8.6 s        22.5%
#   10 /    20_000    354 ms       780 ms          23.5%  <- best coverage
#
# Deeper budgets find NOTHING extra on real positions (coverage is flat or
# worse) while costing an order of magnitude more wall-clock — 20_000 wins on
# both axes, for both the own-win solve and the defensive sweep. Node budget
# stays FLAT across difficulty tiers (it's a search-efficiency cap, not a
# strength lever); DEPTH is the per-tier lever (see
# `DEFAULT_DIFFICULTY_FORCING_DEPTH` above and `_forcing_depth_for` below).
# Re-confirmed 2026-07-05 by the depth x budget grid sweep
# (scripts/forcing_depth_sweep.py): at 20k, p99 stays ~350-390 ms for every
# depth 10..18, while 50k/100k/250k blow the tail out to p99 0.9-2.7 s
# (max 13 s) with zero first-sighting gain.
LIVE_NODE_BUDGET = 20_000
DEF_NODE_BUDGET = 20_000
# Escalated budget for CONFIRMING the one defense candidate about to be
# played (`_verified_defense`). solve_defense's internal candidate re-solves
# run at the flat DEF budget where BudgetExceeded counts as refuted — sound
# enough for triage, not for trusting: on the 2026-07-06 chao position 4 of
# the 5 "verified" pairs were losing, their refutations budget artifacts that
# 30k-75k nodes expose. 250k is the ANALYSIS-grade budget with margin (the
# knee sweeps measured threat SIGHTING, a different axis from refutation
# soundness). Cheap where it counts: a GENUINE refutation has a small forcing
# tree that exhausts in ~100 ms at any budget; only rejecting a false
# candidate pays ~0.4-2 s, once, and saves the game.
VERIFY_NODE_BUDGET = 250_000
# Fallback depth cap for a difficulty tier missing from the map (defensive;
# should be unreachable given DEFAULT_DIFFICULTY_FORCING_DEPTH covers every
# DIFFICULTY_ORDER entry) — the "strong"/full-measured-budget depth.
FALLBACK_FORCING_DEPTH = DEFAULT_DIFFICULTY_FORCING_DEPTH["strong"]
# Wall-clock backstop for the defensive analysis (passed to
# ``hr.solve_defense`` as its time limit): even at the tuned budget above, a
# pathological position could stack enough candidate/pair re-solves to blow
# past a live turn's latency budget, so the solver stops after this long and
# returns whatever killers/anchors/survivors were verified so far.
DEFENSE_DEADLINE_S = 3.0
# Per-tier override of the defense deadline (tiers not listed use the flat
# DEFENSE_DEADLINE_S). Raised for "strong" after the 2026-07-06 chao loss:
# on the production host (Oracle Ampere, ~2-3x slower single-core than the
# dev box) the 3 s deadline expired mid-verification, solve_defense returned
# a partial result with no verified killers, and the bot played the max-delay
# survivor into a proven loss. The full pair sweep on that position takes
# ~5 s even on the dev box, so strong gets 10 s — at most twice per bot turn,
# still well inside the 60 s request timeout, and the pair-partner cache
# (see `_forcing_move`) makes the 2nd-placement solve usually unnecessary.
DIFFICULTY_DEFENSE_DEADLINE_S: dict[str, float] = {
    "strong": 10.0,
}

# Rate-limit for the "forcing check failed" exception log (see
# `_log_forcing_exception`): a systematic bug would otherwise flood the log
# with a full traceback on every bot placement.
_FORCING_LOG_INTERVAL_S = 60.0
_forcing_log_lock = threading.Lock()
_last_forcing_log_ts = 0.0


def _log_forcing_exception(game_id: str, exc: Exception) -> None:
    """Log a live-play forcing-check failure, at most one full traceback per
    `_FORCING_LOG_INTERVAL_S` module-wide; suppressed occurrences in between
    still get a one-line warning so the signal isn't lost entirely, just
    throttled. Must be called from within the `except` block that caught
    `exc` (relies on the ambient exception context for `logger.exception`).
    """
    global _last_forcing_log_ts
    now = time.monotonic()
    with _forcing_log_lock:
        emit_full = (now - _last_forcing_log_ts) >= _FORCING_LOG_INTERVAL_S
        if emit_full:
            _last_forcing_log_ts = now
    if emit_full:
        logger.exception(
            "forcing check failed for game %s; falling back to MCTS", game_id,
        )
    else:
        logger.warning(
            "forcing check failed for game %s; falling back to MCTS: %s",
            game_id, exc,
        )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.isoformat()


def _sanitize_name(name) -> str | None:
    if name is None:
        return None
    cleaned = "".join(ch for ch in str(name) if ch.isprintable() and ch != "\x00").strip()
    if not cleaned:
        return None
    return cleaned[:64]


@dataclass
class GameRecord:
    game_id: str
    created_at: datetime
    last_active_at: datetime
    state: object  # hexo_rs.GameState — not type-annotated to avoid import-at-decl
    human_side: str
    bot_side: str
    human_name: str | None
    move_log: list[tuple[int, int, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    terminal_recorded: bool = False  # set after Recorder.record_completed
    evicted: bool = False  # detached from GameManager._games; never re-persist
    difficulty: str = DEFAULT_DIFFICULTY
    # Cached forced-win PV from a previous solve_forcing call: the bot replays
    # it move-for-move as long as move_log stays in lockstep, without
    # re-invoking the solver. In-memory only — a restart just re-solves.
    forcing_pv: list[tuple[int, int]] | None = None
    forcing_base: int = 0  # len(move_log) at the time forcing_pv was solved
    # Cached second cell of a verified defensive killer PAIR (solve_defense):
    # after the bot plays the pair's anchor, the partner is a proven killer at
    # the SAME turn's next placement — only the bot's own anchor stone was
    # added since the joint verification — so it is played directly instead of
    # re-derived by a fresh, deadline-vulnerable solve (the 2026-07-06 chao
    # loss: the re-solve expired on the slow production host and the bot
    # played the max-delay survivor instead of the already-proven partner).
    # In-memory only, single-shot, valid only when len(move_log) still equals
    # defense_partner_base when consumed.
    defense_partner: tuple[int, int] | None = None
    defense_partner_base: int = 0
    # Phase 5: optional self-reported opponent Elo. Added now so the record is
    # forward-compatible; defaults are anonymous.
    opp_elo: float | None = None
    elo_source: str | None = None
    opp_handle: str | None = None


class GameError(Exception):
    """Base class for play-server errors with HTTP status hints."""
    http_status = 400


class UnknownGameError(GameError):
    http_status = 404


class NotYourTurnError(GameError):
    http_status = 409


class IllegalMoveError(GameError):
    http_status = 400


class GameAlreadyOverError(GameError):
    http_status = 410


class ServerBusyError(GameError):
    """Server is at game capacity with no idle games to reclaim → HTTP 503."""
    http_status = 503


def make_bot_turn_fn(
    *,
    model,
    model_config,
    m_actions: int,
    mcts_sims: int = 0,
    difficulty_sims: dict[str, int] | None = None,
    difficulty_forcing_depth: dict[str, int] | None = None,
    guard: InferenceGuard,
    game_kwargs: dict,
    live_forcing: bool = True,
):
    """Build a closure that runs a full bot turn under the inference guard.

    When ``difficulty_sims`` is provided, the sim count for each move is looked
    up from ``rec.difficulty``; otherwise the legacy fixed ``mcts_sims`` is used
    for every game. ``game_kwargs`` (the same dict given to ``GameManager``) is
    needed to build hypothetical opponent-perspective positions for the
    defensive forcing check.

    ``difficulty_forcing_depth`` (default ``None`` -> ``DEFAULT_DIFFICULTY_FORCING_DEPTH``)
    maps ``rec.difficulty`` to the forcing solver's DEPTH cap, used for both the
    own-win solve and the defensive solves (node budget stays flat across
    tiers — see ``_forcing_depth_for``). Unlike ``difficulty_sims``, there is no
    "off" state: forcing is always graded by tier, never flattened to a single
    depth, since honesty of tactics (not depth) is what stays uniform.

    ``live_forcing`` (default True) is a kill-switch: when False, `_forcing_move`
    is never consulted and bot placements come exclusively from the pre-forcing-
    feature MCTS/argmax path (`--no-live-forcing` on the CLI), byte-identical to
    serving before this feature existed.

    The closure mutates ``rec.state`` and ``rec.move_log`` in place. Caller must
    already hold ``rec.lock``; the guard serializes the actual GPU work.
    """
    import hexo_rs as hr

    forcing_depth_map = difficulty_forcing_depth or DEFAULT_DIFFICULTY_FORCING_DEPTH

    def _eval_fn(states):
        # Torch-only leaf evaluator (native path uses _pick_move's native
        # branch and never reaches here) — import torch + build the graph
        # builder lazily so the native serve path never loads torch.
        import torch
        from torch_geometric.data import Batch
        graph_fn = make_graph_fn(model_config)
        data_list = [graph_fn(s) for s in states]
        batch = Batch.from_data_list(data_list).to("cpu")
        with torch.inference_mode():
            try:
                logits_list, values = model.forward_batch(batch)
            except Exception:
                # Per-graph fallback (matches policy_viewer). One bad graph
                # must not sink the whole MCTS batch, so guard each forward.
                logits_list = []
                values_list = []
                for d in data_list:
                    with torch.no_grad():
                        try:
                            l, v = model(
                                d.x, d.edge_index, d.legal_mask,
                                stone_mask=d.stone_mask,
                                edge_attr=getattr(d, "edge_attr", None),
                            )
                            logits_list.append(l.tolist())
                            values_list.append(float(v.item()))
                        except Exception:
                            n = int(d.legal_mask.sum().item())
                            logits_list.append([0.0] * max(n, 1))
                            values_list.append(0.0)
                return logits_list, values_list
        return ([p.tolist() for p in logits_list],
                [float(v.item()) for v in values])

    def _policy_argmax(state, restrict_to: set | None = None):
        """Raw-argmax move choice: one forward pass, no MCTS.

        Shared by ``_pick_move``'s fallback (``restrict_to=None``) and the
        defensive forcing check, which restricts the choice to a specific set
        of legal killer cells.
        """
        native = getattr(model, "_hexo_native", None)
        if native is not None:
            logit_map, _value = native.forward(state)
            if not logit_map:
                raise RuntimeError("no legal moves on a non-terminal state")
            # Sorted for deterministic tie-breaking (the dict's key order is
            # process-randomized); max() keeps the first, i.e. lowest, coord.
            coords = sorted(logit_map)
            if restrict_to is not None:
                coords = [c for c in coords
                          if (int(c[0]), int(c[1])) in restrict_to]
                if not coords:
                    raise RuntimeError("no legal moves within restrict_to")
            q, r = max(coords, key=lambda c: logit_map[c])
            return (int(q), int(r))
        # Torch fallback (native returned above): build graph + import torch lazily.
        import torch
        graph_fn = make_graph_fn(model_config)
        data = graph_fn(state)
        with torch.no_grad():
            logits, _ = model(
                data.x, data.edge_index, data.legal_mask,
                stone_mask=data.stone_mask,
                edge_attr=getattr(data, "edge_attr", None),
            )
        legal_coords = data.coords[data.legal_mask].tolist()
        if not legal_coords:
            raise RuntimeError("no legal moves on a non-terminal state")
        legal_logits = logits[data.legal_mask] if logits.shape[0] == data.x.shape[0] else logits
        if legal_logits.shape[0] != len(legal_coords):
            # Refuse rather than mis-index a logit onto the wrong coordinate.
            raise RuntimeError(
                f"logits/legal length mismatch: {legal_logits.shape[0]} vs {len(legal_coords)}")
        if restrict_to is not None:
            idxs = [i for i, c in enumerate(legal_coords)
                    if (int(c[0]), int(c[1])) in restrict_to]
            if not idxs:
                raise RuntimeError("no legal moves within restrict_to")
        else:
            idxs = range(len(legal_coords))
        best = max(idxs, key=lambda i: legal_logits[i].item())
        q, r = legal_coords[best]
        return (int(q), int(r))

    def _pick_move(state, sims: int):
        if sims > 0:
            mcts_cfg = hr.MCTSConfig(
                n_simulations=sims,
                m_actions=m_actions,
                c_visit=50,
                c_scale=1.0,
                # Deterministic, paper-style eval: no Gumbel noise on the bot's
                # move selection (reproducible + optimal play vs humans).
                disable_gumbel_noise=True,
            )
            native = getattr(model, "_hexo_native", None)
            if native is not None:
                # Full search in Rust: no Python re-entry per leaf batch.
                return native.mcts_with_diagnostics(state, mcts_cfg)[0]
            action, _improved = hr.gumbel_mcts(state, _eval_fn, mcts_cfg)
            return action
        # Raw argmax fallback (mcts_sims == 0).
        return _policy_argmax(state)

    def _forcing_depth_for(rec: GameRecord) -> int:
        return forcing_depth_map.get(
            rec.difficulty,
            forcing_depth_map.get(DEFAULT_DIFFICULTY, FALLBACK_FORCING_DEPTH),
        )

    def _verified_defense(rec: GameRecord, depth_cap: int, cells) -> bool:
        """Escalated confirmation of one defense candidate: place `cells` as
        bot stones on a hypothetical board, then re-solve the opponent's
        threat (their perspective, fresh turn) at VERIFY_NODE_BUDGET. True
        means the threat is no longer provable past this defense at the
        escalated budget — see VERIFY_NODE_BUDGET for why the flat-budget
        verification inside solve_defense is not enough to trust.
        """
        stones = list(rec.state.placed_stones())
        stones += [(tuple(c), rec.bot_side) for c in cells]
        flip = hr.GameState.from_state(
            stones, rec.human_side, 2, hr.GameConfig(**game_kwargs))
        return hr.solve_forcing(flip, depth_cap, VERIFY_NODE_BUDGET) is None

    def _defensive_move(rec: GameRecord, depth_cap: int):
        """Step 3: pre-emptive VCF defense, run only when the bot has no own
        forced win. Exploits monotonicity: the bot's stones only remove
        opponent threat cells, so "no VCF for the opponent right now
        (hypothetically to move)" implies "none at their next turn start"
        either — safe to fall through to normal MCTS in that case.

        The search itself lives in the solver (``hr.solve_defense``,
        forcing.rs): single killers first, then killer PAIRS when no single
        block refutes — the case the old Python max-delay survivor loop lost
        (strongloss_a, 2026-07-04: every killer pair was anchored on a cell
        the delay metric ranked below a far-away non-anchor, and the bot
        played the non-anchor). This wrapper only applies the NN policy
        tie-break among the solver's proven-equivalent choices.

        ``depth_cap`` is the per-difficulty forcing depth (see
        ``_forcing_depth_for``), shared with the own-win solve in
        ``_forcing_move`` — the SAME tier sees the SAME depth on both sides.
        """
        state = rec.state
        deadline_s = DIFFICULTY_DEFENSE_DEADLINE_S.get(
            rec.difficulty, DEFENSE_DEADLINE_S)
        res = hr.solve_defense(state, depth_cap, DEF_NODE_BUDGET,
                               int(deadline_s * 1000))
        if res is None:
            return None
        killers, pair_anchors, best_delay, _threat_pv = res
        # Escalated confirmation of the candidates below: one more deadline
        # window, checked before each (rare) rejection re-solve. On expiry the
        # policy-best remaining candidate is played UNVERIFIED — exactly the
        # pre-escalation behavior, never worse.
        verify_deadline = time.monotonic() + deadline_s
        if killers:
            remaining = {tuple(c) for c in killers}
            while remaining:
                chosen = _policy_argmax(state, restrict_to=remaining)
                if time.monotonic() >= verify_deadline:
                    return chosen, "killer-unverified"
                if _verified_defense(rec, depth_cap, (chosen,)):
                    return chosen, "killer"
                remaining.discard(chosen)
            # Every killer was a budget artifact: a pair may still hold.
        pairs = [(tuple(c1), tuple(c2)) for (c1, c2) in pair_anchors]
        while pairs:
            # Any confirmed pair works: it was verified JOINTLY, so after
            # playing the anchor its partner is a proven killer at this same
            # turn's next placement. Cache it on the record — `_forcing_move`
            # plays it directly instead of re-deriving it with a fresh solve
            # that could hit the deadline (the 2026-07-06 chao loss).
            chosen = _policy_argmax(
                state, restrict_to={c1 for (c1, _c2) in pairs})
            partner = next(c2 for (c1, c2) in pairs if c1 == chosen)
            timed_out = time.monotonic() >= verify_deadline
            if timed_out or _verified_defense(rec, depth_cap, (chosen, partner)):
                rec.defense_partner = partner
                # After the anchor is appended to move_log, its length is
                # len+1 — the partner applies at exactly that placement.
                rec.defense_partner_base = len(rec.move_log) + 1
                return chosen, ("pair-unverified" if timed_out else "pair")
            pairs = [p for p in pairs if p != (chosen, partner)]
        if best_delay is not None:
            # No refutation found this placement: play the survivor with the
            # longest surviving PV (max delay). The next placement re-runs
            # this whole check against the updated threat.
            return tuple(best_delay), "delay"
        # No candidates at all (or deadline expired before any could be
        # verified): fall through to MCTS.
        return None

    def _forcing_move(rec: GameRecord):
        """Steps 1-3 of the live-play forcing check, run before `_pick_move`
        on every bot placement. Returns ``(move, src)`` — ``src`` names the
        mechanism for the placement log line — or None to fall through to the
        existing MCTS / raw-argmax path unchanged.
        """
        state = rec.state
        legal = {tuple(c) for c in state.legal_moves()}
        depth_cap = _forcing_depth_for(rec)

        # Pop the pair-partner cache (single-shot): it is only valid at the
        # placement immediately after its anchor, in the same bot turn. It is
        # consumed below AFTER the own-win solve — a proven own win preempts
        # defending, same priority order as the rest of this function.
        partner, rec.defense_partner = rec.defense_partner, None
        if (partner is not None
                and (len(rec.move_log) != rec.defense_partner_base
                     or partner not in legal)):
            partner = None

        # 1. PV-cache replay: if the opponent has followed the cached forced
        # line so far, play the next PV move without calling the solver.
        if rec.forcing_pv is not None:
            k = len(rec.move_log) - rec.forcing_base
            played = [(m[0], m[1]) for m in rec.move_log[rec.forcing_base:]]
            pv_prefix = [tuple(m) for m in rec.forcing_pv[:k]]
            if 0 <= k < len(rec.forcing_pv) and played == pv_prefix:
                nxt = tuple(rec.forcing_pv[k])
                if nxt in legal:
                    return nxt, "pv"
            # Deviation (or an exhausted/stale cache): re-solve from scratch.
            rec.forcing_pv = None

        # 2. Own-win solve, at this tier's forcing depth.
        res = hr.solve_forcing(state, depth_cap, LIVE_NODE_BUDGET)
        if res is not None:
            first_move, pv = res
            first_move = tuple(first_move)
            if first_move in legal:
                rec.forcing_pv = [tuple(m) for m in pv]
                rec.forcing_base = len(rec.move_log)
                return first_move, "win"
            logger.error(
                "solve_forcing returned illegal first_move %r for game %s",
                first_move, rec.game_id,
            )

        # 2.5. Verified pair partner from this turn's anchor placement: a
        # proven killer, played without any re-solve.
        if partner is not None:
            return partner, "partner"

        # 3. Defensive pre-emptive check (only when step 2 found nothing).
        return _defensive_move(rec, depth_cap)

    def _sims_for(rec: GameRecord) -> int:
        if difficulty_sims is None:
            return mcts_sims
        return difficulty_sims.get(
            rec.difficulty,
            difficulty_sims.get(DEFAULT_DIFFICULTY, mcts_sims),
        )

    def bot_turn_fn(rec: GameRecord):
        sims = _sims_for(rec)

        def _play():
            for _ in range(2):
                if rec.state.is_terminal():
                    return
                if rec.state.current_player() != rec.bot_side:
                    return
                move = src = None
                forcing_ms = 0.0
                # All difficulty tiers get forcing (win execution + defense) —
                # deliberately never OFF for weaker tiers: what varies is HOW
                # DEEP each tier's forcing search reaches (`_forcing_depth_for`,
                # graded like sim count) and how many sims back up MCTS, not
                # whether a tier is allowed to notice a forced win at all.
                if live_forcing:
                    t0 = time.perf_counter()
                    try:
                        res = _forcing_move(rec)
                        if res is not None:
                            move, src = res
                    except Exception as e:
                        # The solver must never take down live play.
                        _log_forcing_exception(rec.game_id, e)
                        move = None
                    forcing_ms = (time.perf_counter() - t0) * 1000
                mcts_ms = 0.0
                if move is not None:
                    q, r = move
                else:
                    src = "mcts" if sims > 0 else "argmax"
                    t0 = time.perf_counter()
                    q, r = _pick_move(rec.state, sims)
                    mcts_ms = (time.perf_counter() - t0) * 1000
                rec.state.apply_move(q, r)
                rec.move_log.append((q, r, rec.bot_side))
                # One line per placement: attribute slow turns to the solver
                # stack vs MCTS straight from the (docker) log.
                logger.info(
                    "bot placement game=%s diff=%s src=%s move=(%d,%d) "
                    "forcing_ms=%.0f mcts_ms=%.0f",
                    rec.game_id, rec.difficulty, src, q, r,
                    forcing_ms, mcts_ms,
                )

        guard.run(_play)

    return bot_turn_fn


class GameManager:
    """In-memory store of active games. Per-game locks. Lazy LRU eviction."""

    def __init__(
        self,
        *,
        game_kwargs: dict,           # passed to hexo_rs.GameConfig(**game_kwargs)
        bot_turn_fn,                 # callable(GameRecord) -> None, mutates state + move_log
        mcts_sims: int,
        m_actions: int,
        checkpoint_path: str,
        model_label: str = "unknown",
        model_step: int | None = None,
        difficulty_sims: dict[str, int] | None = None,
        default_difficulty: str = DEFAULT_DIFFICULTY,
        max_games: int = 100,
        idle_ttl_seconds: int = 24 * 3600,
        idle_grace_seconds: int = 60,
        recorder=None,
    ):
        if max_games < 1:
            raise ValueError(f"max_games must be >= 1 (got {max_games})")
        import hexo_rs as hr
        self._hr = hr
        self.game_kwargs = game_kwargs
        self.bot_turn_fn = bot_turn_fn
        self.mcts_sims = mcts_sims
        self.m_actions = m_actions
        self.checkpoint_path = checkpoint_path
        self.model_label = model_label
        self.model_step = model_step
        self.difficulty_sims = difficulty_sims  # None = single-tier legacy mode
        self.default_difficulty = default_difficulty
        self.max_games = max_games
        self.idle_ttl_seconds = idle_ttl_seconds
        # Overflow eviction never drops a game active within this grace window,
        # so a /new_game flood cannot evict a legitimate in-progress game.
        self.idle_grace_seconds = idle_grace_seconds
        self.recorder = recorder
        self._games: dict[str, GameRecord] = {}
        self._mgr_lock = threading.Lock()

    def _evict(self, *, evict_count: bool = False) -> tuple[bool, list[GameRecord]]:
        """Drop expired and (optionally) overflow games. Caller holds self._mgr_lock.

        Returns ``(ok, victims)``. ``ok`` is False if ``evict_count`` is set and
        the store is still at capacity after evicting every eligible idle game
        (i.e. all remaining games are active within the grace window) — the
        caller should then refuse the new game with ``ServerBusyError`` rather
        than drop an active game. ``victims`` are the records removed from the
        store; the caller MUST pass them to ``_cleanup_evicted`` after releasing
        _mgr_lock (their persisted rows are not touched here).
        """
        victims: list[GameRecord] = []
        now = _now_utc()
        cutoff = now.timestamp() - self.idle_ttl_seconds
        for gid in [g for g, r in self._games.items()
                    if r.last_active_at.timestamp() < cutoff]:
            victims.append(self._games.pop(gid))
        if evict_count:
            grace_cutoff = now.timestamp() - self.idle_grace_seconds
            while len(self._games) >= self.max_games:
                candidates = [r for r in self._games.values()
                              if r.last_active_at.timestamp() < grace_cutoff]
                if not candidates:
                    return False, victims
                oldest = min(candidates, key=lambda r: r.last_active_at)
                victims.append(self._games.pop(oldest.game_id))
        return True, victims

    def _cleanup_evicted(self, victims: list[GameRecord]) -> None:
        """Finish evictions with _mgr_lock released. Taking each victim's lock
        first serializes behind any in-flight move on it, so the row delete
        cannot lose the race with that move's write-through re-inserting it;
        ``evicted`` then blocks any later re-persist for good."""
        for victim in victims:
            with victim.lock:
                victim.evicted = True
                self._drop_active(victim.game_id)

    # ---- in-progress persistence (active_games write-through) --------------
    # Every request that leaves a game live upserts its snapshot; ending or
    # evicting a game deletes it. All of it is best-effort: persistence exists
    # to survive restarts and must never take down live play.

    def _sync_active(self, rec: GameRecord) -> None:
        """Write-through ``rec`` to the active_games table (caller holds rec.lock):
        upsert while the game is live, delete once it is over."""
        if self.recorder is None or rec.evicted:
            return
        try:
            if rec.terminal_recorded or rec.state.is_terminal():
                self.recorder.delete_active(rec.game_id)
            else:
                self.recorder.save_active(
                    game_id=rec.game_id,
                    created_at=_iso(rec.created_at),
                    last_active_at=_iso(rec.last_active_at),
                    human_name=rec.human_name,
                    human_side=rec.human_side,
                    bot_side=rec.bot_side,
                    difficulty=rec.difficulty,
                    opp_elo=rec.opp_elo,
                    elo_source=rec.elo_source,
                    opp_handle=rec.opp_handle,
                    win_length=self.game_kwargs["win_length"],
                    placement_radius=self.game_kwargs["placement_radius"],
                    max_moves=self.game_kwargs["max_moves"],
                    move_log=rec.move_log,
                )
        except Exception:
            logger.exception("active-game write-through failed for %s", rec.game_id)

    def _drop_active(self, game_id: str) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.delete_active(game_id)
        except Exception:
            logger.exception("active-game delete failed for %s", game_id)

    def restore_active_games(self) -> int:
        """Rebuild in-memory games persisted by a previous process. Call once at
        startup, before serving. Rows that are TTL-expired, rule-mismatched,
        terminal, bot-to-move (nothing would ever schedule their bot turn), over
        capacity, or fail replay are deleted instead of restored."""
        if self.recorder is None:
            return 0
        try:
            rows = self.recorder.load_active()   # newest first
        except Exception:
            logger.exception("active-game load failed; starting with no restored games")
            return 0
        cutoff = _now_utc().timestamp() - self.idle_ttl_seconds
        restored = 0
        for row in rows:
            rec = self._rebuild_record(row, cutoff) if restored < self.max_games else None
            if rec is None:
                self._drop_active(row["game_id"])
                continue
            with self._mgr_lock:
                self._games[rec.game_id] = rec
            restored += 1
        return restored

    def _rebuild_record(self, row: dict, cutoff: float) -> GameRecord | None:
        """Replay one active_games row into a live GameRecord, or None to drop it."""
        try:
            last_active = datetime.fromisoformat(row["last_active_at"])
            if last_active.timestamp() < cutoff:
                return None
            if self.recorder.has_completed(row["game_id"]):
                # Already finished — e.g. a resignation whose active-row delete
                # was lost to a crash. Its board replays as live (resigning
                # doesn't place a stone), and restoring it would collide with
                # the UNIQUE completed row when it finishes again.
                return None
            if {row["human_side"], row["bot_side"]} != {"P1", "P2"}:
                return None
            for k in ("win_length", "placement_radius", "max_moves"):
                if int(row[k]) != int(self.game_kwargs[k]):
                    return None
            move_log = [(int(q), int(r), str(p)) for q, r, p in json.loads(row["moves_json"])]
            if not move_log or move_log[0] != (0, 0, "P1"):
                return None
            state = self._hr.GameState(self._hr.GameConfig(**self.game_kwargs))
            for q, r, _p in move_log[1:]:
                state.apply_move(q, r)
            if state.is_terminal() or state.current_player() != row["human_side"]:
                return None
            difficulty = row["difficulty"]
            if self.difficulty_sims is not None and difficulty not in self.difficulty_sims:
                difficulty = self.default_difficulty
            return GameRecord(
                game_id=row["game_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_active_at=last_active,
                state=state,
                human_side=row["human_side"],
                bot_side=row["bot_side"],
                human_name=row["human_name"],
                move_log=move_log,
                difficulty=difficulty,
                opp_elo=row["opp_elo"],
                elo_source=row["elo_source"],
                opp_handle=row["opp_handle"],
            )
        except Exception:
            logger.exception("failed to restore active game %s", row.get("game_id"))
            return None

    def create_game(
        self,
        human_side_request: str,
        name=None,
        rng=None,
        difficulty: str | None = None,
        self_reported_elo: float | None = None,
    ) -> GameRecord:
        if human_side_request not in ("P1", "P2", "random"):
            raise GameError(
                f"invalid human_side {human_side_request!r}; expected P1/P2/random")
        if human_side_request == "random":
            rng = rng or random.Random()
            human_side = rng.choice(("P1", "P2"))
        else:
            human_side = human_side_request
        bot_side = "P2" if human_side == "P1" else "P1"

        if self.difficulty_sims is None:
            chosen = self.default_difficulty
        else:
            chosen = difficulty if difficulty in self.difficulty_sims else self.default_difficulty

        cfg = self._hr.GameConfig(**self.game_kwargs)
        state = self._hr.GameState(cfg)
        move_log = [(0, 0, "P1")]
        now = _now_utc()
        rec = GameRecord(
            game_id=str(uuid.uuid4()),
            created_at=now,
            last_active_at=now,
            state=state,
            human_side=human_side,
            bot_side=bot_side,
            human_name=_sanitize_name(name),
            move_log=move_log,
            difficulty=chosen,
            opp_elo=self_reported_elo,
            elo_source="self_reported" if self_reported_elo is not None else None,
        )
        with self._mgr_lock:
            ok, evicted = self._evict(evict_count=True)
            if ok:
                self._games[rec.game_id] = rec
                # Acquire rec.lock BEFORE releasing _mgr_lock so there is no
                # window in which a concurrent /resign can sneak in and record a
                # bot-win-by-resign before the opening bot move runs (which would
                # append a bot move after the resign and corrupt the move log).
                rec.lock.acquire()
        if not ok:
            # At capacity with no idle game to reclaim — refuse rather than
            # evict an active in-progress game (eviction-DoS mitigation). The
            # TTL pass may still have evicted expired games; finish those.
            self._cleanup_evicted(evicted)
            raise ServerBusyError("server at game capacity; try again shortly")
        # _mgr_lock released; rec.lock held continuously since insertion.
        try:
            if rec.bot_side == state.current_player() and not state.is_terminal():
                # Hold rec.lock for the initial bot turn: the record is already
                # visible in self._games, so concurrent /state or /move must not
                # observe a half-applied bot move. Matches apply_human_move's
                # discipline of holding rec.lock across bot_turn_fn.
                self.bot_turn_fn(rec)
                rec.last_active_at = _now_utc()
                # An opening bot move could (pathologically) end the game; record
                # it the same way apply_human_move does after a bot turn.
                self._maybe_record_engine_terminal(rec)
            self._sync_active(rec)
        finally:
            rec.lock.release()
            self._cleanup_evicted(evicted)
        return rec

    def get_game(self, game_id: str) -> GameRecord | None:
        with self._mgr_lock:
            rec = self._games.get(game_id)
        return rec

    def snapshot(self, rec: GameRecord, *, placement_radius: int, win_length: int,
                include_htttx: bool = False) -> dict:
        """Serialize ``rec`` to the API state dict under ``rec.lock``.

        Use this instead of calling ``state_to_dict`` directly so a concurrent
        bot move cannot produce a torn read of ``rec.state`` / ``rec.move_log``.
        """
        from hexo_a0.serving.serialize import state_to_dict
        with rec.lock:
            return state_to_dict(rec, placement_radius=placement_radius,
                                 win_length=win_length, include_htttx=include_htttx)

    def htttx(self, game_id: str) -> str:
        """Return the HTTTX move-log for a completed or in-progress game."""
        from hexo_a0.serving.htttx import serialize_htttx
        rec = self.get_game(game_id)
        if rec is None:
            raise UnknownGameError(f"unknown game_id {game_id}")
        # Hold rec.lock: move_log is mutated under it by bot_turn_fn, and a
        # concurrent append can tear the pairing inside serialize_htttx.
        with rec.lock:
            return serialize_htttx(rec.move_log)

    def active_games(self) -> list[dict]:
        """Snapshot of in-memory games that have NOT been terminal-recorded."""
        with self._mgr_lock:
            records = sorted(
                self._games.values(),
                key=lambda r: r.last_active_at,
                reverse=True,
            )
        out: list[dict] = []
        for rec in records:
            with rec.lock:
                if rec.terminal_recorded or rec.state.is_terminal():
                    continue
                out.append({
                    "game_id": rec.game_id,
                    "created_at": _iso(rec.created_at),
                    "last_active_at": _iso(rec.last_active_at),
                    "human_name": rec.human_name,
                    "human_side": rec.human_side,
                    "bot_side": rec.bot_side,
                    "current_player": rec.state.current_player(),
                    "moves_remaining": int(rec.state.moves_remaining_this_turn()),
                    "n_moves": len(rec.move_log),
                    "difficulty": rec.difficulty,
                })
        return out

    def _record_terminal(self, rec: GameRecord, *, winner, result_type):
        """Write the row to SQLite if recorder configured. Idempotent per game.

        The ``terminal_recorded`` flag is set ONLY after a successful write (or
        when no recorder is configured). Setting it before the write would mark
        a game recorded even if ``commit()`` raised on disk-full/lock-contention,
        permanently losing the row with no retry path.
        """
        if rec.terminal_recorded:
            return
        if self.recorder is None:
            rec.terminal_recorded = True
            return
        sims = (self.difficulty_sims.get(rec.difficulty, self.mcts_sims)
                if self.difficulty_sims is not None else self.mcts_sims)
        self.recorder.record_completed(
            game_id=rec.game_id,
            created_at=_iso(rec.created_at),
            completed_at=_iso(_now_utc()),
            human_name=rec.human_name,
            human_side=rec.human_side,
            bot_side=rec.bot_side,
            winner=winner,
            result_type=result_type,
            n_moves=len(rec.move_log),
            mcts_sims=sims,
            win_length=self.game_kwargs["win_length"],
            placement_radius=self.game_kwargs["placement_radius"],
            max_moves=self.game_kwargs["max_moves"],
            checkpoint_path=self.checkpoint_path,
            model_label=self.model_label,
            step=self.model_step,
            move_log=rec.move_log,
            opp_elo=rec.opp_elo,
            elo_source=rec.elo_source,
            opp_handle=rec.opp_handle,
        )
        rec.terminal_recorded = True

    def _maybe_record_engine_terminal(self, rec: GameRecord):
        """If hexo_rs marks terminal, write the row with derived result_type."""
        if not rec.state.is_terminal():
            return
        winner = rec.state.winner()
        if winner is None:
            self._record_terminal(rec, winner=None, result_type="max_moves")
        else:
            self._record_terminal(rec, winner=winner, result_type="win")

    def apply_human_move(self, game_id: str, q: int, r: int) -> GameRecord:
        rec = self.get_game(game_id)
        if rec is None:
            raise UnknownGameError(f"unknown game_id {game_id}")
        with rec.lock:
            if rec.terminal_recorded or rec.state.is_terminal():
                raise GameAlreadyOverError("game already over")
            cur = rec.state.current_player()
            if cur != rec.human_side:
                raise NotYourTurnError(f"current_player={cur}, human_side={rec.human_side}")
            legal = {tuple(c) for c in rec.state.legal_moves()}
            if (q, r) not in legal:
                raise IllegalMoveError(f"({q},{r}) not in legal_moves")
            try:
                rec.state.apply_move(q, r)
            except Exception as e:
                raise IllegalMoveError(str(e)) from e
            rec.move_log.append((q, r, rec.human_side))
            rec.last_active_at = _now_utc()
            self._maybe_record_engine_terminal(rec)
            if (
                not rec.terminal_recorded
                and rec.state.current_player() == rec.bot_side
            ):
                self.bot_turn_fn(rec)
                self._maybe_record_engine_terminal(rec)
            self._sync_active(rec)
        return rec

    def resign(self, game_id: str) -> GameRecord:
        rec = self.get_game(game_id)
        if rec is None:
            raise UnknownGameError(f"unknown game_id {game_id}")
        with rec.lock:
            # Guard on the engine terminal state too, not just terminal_recorded:
            # _record_terminal sets the flag only AFTER a successful DB write, so
            # a failed write (disk full / lock contention) leaves terminal_recorded
            # False while state.is_terminal() is True. Without this guard a /resign
            # would overwrite the real result (e.g. a human win) with a bot resign.
            if rec.terminal_recorded or rec.state.is_terminal():
                raise GameAlreadyOverError("game already over")
            self._record_terminal(rec, winner=rec.bot_side, result_type="resign")
            rec.last_active_at = _now_utc()
            self._sync_active(rec)
        return rec
