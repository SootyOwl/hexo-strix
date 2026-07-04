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

# Live-play forcing (VCF) search budgets. Post-9x-speedup these are sub-second
# at live board sizes; the defensive check runs up to ~1+|PV| solves per bot
# placement, so it gets the smaller budget.
LIVE_DEPTH_CAP = 16
LIVE_NODE_BUDGET = 2_000_000
DEF_DEPTH_CAP = 10
DEF_NODE_BUDGET = 200_000


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
    guard: InferenceGuard,
    game_kwargs: dict,
):
    """Build a closure that runs a full bot turn under the inference guard.

    When ``difficulty_sims`` is provided, the sim count for each move is looked
    up from ``rec.difficulty``; otherwise the legacy fixed ``mcts_sims`` is used
    for every game. ``game_kwargs`` (the same dict given to ``GameManager``) is
    needed to build hypothetical opponent-perspective positions for the
    defensive forcing check.

    The closure mutates ``rec.state`` and ``rec.move_log`` in place. Caller must
    already hold ``rec.lock``; the guard serializes the actual GPU work.
    """
    import torch
    from torch_geometric.data import Batch
    import hexo_rs as hr

    graph_fn = make_graph_fn(model_config)

    def _eval_fn(states):
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
            action, _improved = hr.gumbel_mcts(state, _eval_fn, mcts_cfg)
            return action
        # Raw argmax fallback (mcts_sims == 0).
        return _policy_argmax(state)

    def _defensive_move(rec: GameRecord, legal: set):
        """Step 3: pre-emptive VCF defense, run only when the bot has no own
        forced win. Exploits monotonicity: the bot's stones only remove
        opponent threat cells, so "no VCF for the opponent right now
        (hypothetically to move)" implies "none at their next turn start"
        either — safe to fall through to normal MCTS in that case.
        """
        state = rec.state
        opp = "P1" if rec.bot_side == "P2" else "P2"
        cfg = hr.GameConfig(**game_kwargs)
        opp_state = hr.GameState.from_state(state.placed_stones(), opp, 2, cfg)
        opp_win = hr.solve_forcing(opp_state, DEF_DEPTH_CAP, DEF_NODE_BUDGET)
        if opp_win is None:
            return None

        _opp_first, opp_pv = opp_win
        seen: set = set()
        candidates: list = []
        for c in opp_pv:
            c = tuple(c)
            if c in legal and c not in seen:
                seen.add(c)
                candidates.append(c)
        if not candidates:
            return None

        killers = []
        survivors = []  # (cell, surviving PV length) — delay/disruption proxy
        for c in candidates:
            trial = state.clone()
            trial.apply_move(*c)
            if trial.is_terminal():
                # The placer can only WIN by placing (the engine has no
                # self-loss rule), so a terminal trial means this cell
                # completes a bot win — play it outright. Also keeps the
                # from_state call below on its documented non-terminal-only
                # contract.
                return c
            trial_opp = hr.GameState.from_state(trial.placed_stones(), opp, 2, cfg)
            re_res = hr.solve_forcing(trial_opp, DEF_DEPTH_CAP, DEF_NODE_BUDGET)
            if re_res is None:
                killers.append(c)
            else:
                survivors.append((c, len(re_res[1])))

        if killers:
            return _policy_argmax(state, restrict_to=set(killers))
        # No killer this placement: play the survivor with the longest
        # surviving PV (max delay). The next placement re-runs this whole
        # check, giving the pair a second chance to kill the win.
        best_cell, _pv_len = max(survivors, key=lambda t: t[1])
        return best_cell

    def _forcing_move(rec: GameRecord):
        """Steps 1-3 of the live-play forcing check, run before `_pick_move`
        on every bot placement. Returns a move to play, or None to fall
        through to the existing MCTS / raw-argmax path unchanged.
        """
        state = rec.state
        legal = {tuple(c) for c in state.legal_moves()}

        # 1. PV-cache replay: if the opponent has followed the cached forced
        # line so far, play the next PV move without calling the solver.
        if rec.forcing_pv is not None:
            k = len(rec.move_log) - rec.forcing_base
            played = [(m[0], m[1]) for m in rec.move_log[rec.forcing_base:]]
            pv_prefix = [tuple(m) for m in rec.forcing_pv[:k]]
            if 0 <= k < len(rec.forcing_pv) and played == pv_prefix:
                nxt = tuple(rec.forcing_pv[k])
                if nxt in legal:
                    return nxt
            # Deviation (or an exhausted/stale cache): re-solve from scratch.
            rec.forcing_pv = None

        # 2. Own-win solve.
        res = hr.solve_forcing(state, LIVE_DEPTH_CAP, LIVE_NODE_BUDGET)
        if res is not None:
            first_move, pv = res
            first_move = tuple(first_move)
            if first_move in legal:
                rec.forcing_pv = [tuple(m) for m in pv]
                rec.forcing_base = len(rec.move_log)
                return first_move
            logger.error(
                "solve_forcing returned illegal first_move %r for game %s",
                first_move, rec.game_id,
            )

        # 3. Defensive pre-emptive check (only when step 2 found nothing).
        return _defensive_move(rec, legal)

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
                try:
                    move = _forcing_move(rec)
                except Exception:
                    # The solver must never take down live play.
                    logger.exception(
                        "forcing check failed for game %s; falling back to MCTS",
                        rec.game_id,
                    )
                    move = None
                q, r = move if move is not None else _pick_move(rec.state, sims)
                rec.state.apply_move(q, r)
                rec.move_log.append((q, r, rec.bot_side))

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
