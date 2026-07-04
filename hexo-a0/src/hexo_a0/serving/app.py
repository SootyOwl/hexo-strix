"""HTTP app for ``hexo-a0 serve`` (public play server + analysis).

Ports the frozen ``scripts/play_server.py`` HTTP layer, admin page, and
bootstrap into the new ``serving`` module. The frontend is fully external: the
HTML shells live in ``templates/`` (loaded via ``_load_template``) and the CSS/JS
under ``static/``; the ``/`` and ``/admin`` routes fill placeholders per request.
"""
import html
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from hexo_a0.serving.analysis import (
    analyze_game_full, analyze_position, analyze_trajectory, build_state,
    end_of_turn_indices, TRAJECTORY_DEPTH_CAP, TRAJECTORY_NODE_BUDGET)
from hexo_a0.serving.dbmigrate import apply_migrations
from hexo_a0.serving import hds
from hexo_a0.serving.game import DEFAULT_DIFFICULTY, DEFAULT_DIFFICULTY_SIMS, DIFFICULTY_ORDER, GameError, GameManager, ServerBusyError, UnknownGameError, make_bot_turn_fn
from hexo_a0.serving.inference import InferenceGuard, InferenceBusy
from hexo_a0.serving.model import load_model
from hexo_a0.serving.recorder import Recorder

# Static frontend assets (CSS/JS/fonts) live beside this module and are served
# under /static/. Resolved once; the per-request handler guards against path
# traversal before reading anything.
_STATIC_DIR = (Path(__file__).resolve().parent / "static")

_TEMPLATES_DIR = (Path(__file__).resolve().parent / "templates")


def _load_template(name: str) -> str:
    """Load a serve-time HTML shell (index/admin) from templates/. Read once at
    import; placeholders are filled per request by the `/` and `/admin` routes."""
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


_STATIC_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}


def _asset_version() -> str:
    """Short cache-busting token from the newest static asset mtime, so a
    redeploy of CSS/JS invalidates browser caches without manual versioning."""
    try:
        latest = max((p.stat().st_mtime_ns for p in _STATIC_DIR.glob("*.*")), default=0)
    except OSError:
        latest = 0
    return format(latest & 0xFFFFFFFF, "x")


MAX_BODY_BYTES = 262144  # 256 KiB — public request body cap
ELO_MIN, ELO_MAX = 0, 3500
COORD_MAX = 1000  # reject coordinates beyond the playable board at the trust boundary
MAX_CONCURRENT = 32        # cap concurrent in-flight requests (slowloris / flood backstop)
ANALYZE_RATE_PER_MIN = 30  # per-client cap on /analyze + /analyze_trajectory (GPU-lock starvation backstop)
NEW_GAME_RATE_PER_MIN = 20  # per-client cap on /new_game (bot-turn flood backstop)
ANALYZE_GAME_RATE_PER_MIN = 6  # per-client cap on /analyze_game (full-game MCTS job is heavy)
ANALYZE_GAME_MAX_INFLIGHT = 4  # cap concurrent /analyze_game requests (queue-slot exhaustion backstop)

logger = logging.getLogger("hexo_a0.serving")


class RateLimiter:
    """Simple per-key token-bucket limiter (thread-safe). Refills to max_per_min.

    The per-key state dict is capped (``max_keys``); when exceeded, the oldest
    entries are pruned so a flood of distinct keys can't grow it unbounded.
    """

    def __init__(self, max_per_min: int, max_keys: int = 8192):
        self.max_per_min = max_per_min
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, int]] = {}  # key -> (window_start, count)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if len(self._state) > self.max_keys:
                # Prune oldest windows (also drops stale entries > 60s).
                cutoff = now - 60.0
                self._state = {k: v for k, v in self._state.items() if v[0] > cutoff}
                if len(self._state) >= self.max_keys:
                    # Still too many — drop down to max_keys-1 so the current
                    # key's add below leaves us at <= max_keys.
                    drop = len(self._state) - (self.max_keys - 1)
                    for k, _ in sorted(self._state.items(), key=lambda kv: kv[1][0])[:drop]:
                        self._state.pop(k, None)
            start, count = self._state.get(key, (now, 0))
            if now - start >= 60.0:
                self._state[key] = (now, 1)
                return True
            if count >= self.max_per_min:
                return False
            self._state[key] = (start, count + 1)
            return True


class BadRequestError(GameError):
    """Malformed request body / missing fields. Maps to HTTP 400."""
    http_status = 400


class PayloadTooLargeError(GameError):
    """Request body exceeds the public cap. Maps to HTTP 413."""
    http_status = 413


def _validate_elo(raw) -> float | None:
    """Coerce a self-reported Elo to a finite float in [0, 3500], or None.

    Rejects NaN/Inf, booleans (``bool`` is an ``int`` subclass), negatives, and
    out-of-range values. Raises ``BadRequestError`` on invalid input.
    """
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BadRequestError("self_reported_elo must be a number")
    try:
        val = float(raw)
    except (ValueError, OverflowError) as e:
        # float(10**400) raises OverflowError; float('inf') is caught by isfinite.
        raise BadRequestError("self_reported_elo must be a finite number") from e
    if not math.isfinite(val) or val < ELO_MIN or val > ELO_MAX:
        raise BadRequestError(f"self_reported_elo must be in [{ELO_MIN},{ELO_MAX}]")
    return val


def _derive_step(ckpt_path: str, ckpt: dict) -> int | None:
    """Training step from the filename (checkpoint_NNNN.pt) or the ckpt dict's
    ``train_steps`` (used by champion.pt, whose filename carries no step)."""
    m = re.match(r"checkpoint_(\d+)\.pt$", Path(ckpt_path).name)
    if m:
        return int(m.group(1))
    v = ckpt.get("train_steps")
    return int(v) if isinstance(v, int) else None


def _derive_model_label(ckpt_path: str, ckpt: dict) -> str:
    """Best-effort '<run>@<step>' label from path + ckpt dict."""
    p = Path(ckpt_path).resolve()
    parts = p.parts
    run_name: str | None = None
    for i, seg in enumerate(parts):
        if seg == "runs" and i + 1 < len(parts):
            run_name = parts[i + 1]
            break
    if run_name is None:
        cur = p.parent
        while cur.name in ("checkpoints", "self_play") and cur.parent != cur:
            cur = cur.parent
        run_name = cur.name or "model"

    step = _derive_step(ckpt_path, ckpt)
    if step is not None:
        return f"{run_name}@{step}"
    return f"{run_name}@{p.stem}"


def _parse_difficulty_sims(spec: str) -> dict[str, int]:
    """Parse '16,32,64,128' into a {label: sims} dict in canonical order."""
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    if not parts:
        raise SystemExit("--difficulty-sims must list at least one sim count")
    if len(parts) > len(DIFFICULTY_ORDER):
        raise SystemExit(
            f"--difficulty-sims takes at most {len(DIFFICULTY_ORDER)} values "
            f"(one per tier in {DIFFICULTY_ORDER}); got {len(parts)}"
        )
    out: dict[str, int] = {}
    for label, raw in zip(DIFFICULTY_ORDER, parts):
        try:
            out[label] = int(raw)
        except ValueError as e:
            raise SystemExit(f"--difficulty-sims: '{raw}' is not an integer") from e
    return out


def _model_config_fallback(cfg) -> dict | None:
    """Read the ``[model]`` section of ``cfg.config`` as the architecture fallback.

    Mirrors the frozen ``scripts/play_server.py`` loader: an embedded
    checkpoint ``model_config`` always wins; this TOML section is only used
    when the checkpoint carries none. Returns ``None`` if no config path is
    configured or the file is missing/unreadable, so ``load_model`` falls back
    to ``ModelConfig`` defaults. A MALFORMED config (TOMLDecodeError) is NOT
    swallowed — it propagates so the server crashes loudly at startup instead
    of silently serving a wrong-architecture model (the frozen script does the
    same).
    """
    config_path = getattr(cfg, "config", None)
    if not config_path:
        return None
    try:
        import tomllib
        raw = tomllib.loads(Path(config_path).read_text())
    except (FileNotFoundError, OSError):
        return None
    return raw.get("model", {}) or None


def _assemble_result(entries: list, sims: int) -> dict:
    """Rebuild an analyze_game_full-shaped result from per-position entries
    collected along a line (entries[i] = the analysis of position i). Boundaries
    are recomputed from the players; each entry already carries its own quality."""
    boundary_indices = end_of_turn_indices([e["current_player"] for e in entries])
    return {
        "trajectory": entries,
        "boundary_indices": boundary_indices,
        "evaluated_prefixes": len(entries),
        "mcts_sims": sims,
        "cancelled": False,
        "completed_prefixes": len(entries),
    }


def _m0_positions(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS positions("
        " model TEXT NOT NULL, sims INTEGER NOT NULL, macts INTEGER NOT NULL,"
        " path TEXT NOT NULL, entry TEXT NOT NULL, ts INTEGER NOT NULL,"
        " PRIMARY KEY(model, sims, macts, path))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pos_ts ON positions(ts)")


_POSITIONS_MIGRATIONS = [_m0_positions]


class _PositionStore:
    """SQLite-backed persistence for the per-position analysis cache, so analysed
    positions survive a restart. Keyed by (model, sims, m_actions, path): reads
    and writes filter by the current model, so a changed model/checkpoint or
    search depth never reuses another's entries — but other models' rows are
    KEPT, so switching between checkpoints preserves each one's cache (bounded by
    a shared size cap). A row is only overwritten by a re-analysis of the same
    position under the same model + search settings."""

    def __init__(self, path, model_id, sims, m_actions, disk_max=50000):
        self.model_id, self.sims, self.m_actions, self.disk_max = model_id, sims, m_actions, disk_max
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        apply_migrations(self._conn, _POSITIONS_MIGRATIONS)
        row = self._conn.execute("SELECT COALESCE(MAX(ts), 0) FROM positions").fetchone()
        self._seq = int(row[0]) if row else 0

    @staticmethod
    def _path(prefix_tuple):
        return json.dumps([list(c) for c in prefix_tuple], separators=(",", ":"))

    def get(self, prefix_tuple):
        with self._lock:
            row = self._conn.execute(
                "SELECT entry FROM positions WHERE model=? AND sims=? AND macts=? AND path=?",
                (self.model_id, self.sims, self.m_actions, self._path(prefix_tuple))).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except (ValueError, TypeError):
            return None

    def put_many(self, items):
        """items: iterable of (prefix_tuple, entry_dict)."""
        with self._lock:
            for prefix_tuple, entry in items:
                try:
                    payload = json.dumps(entry, separators=(",", ":"))
                except (TypeError, ValueError):
                    continue
                self._seq += 1
                self._conn.execute(
                    "INSERT OR REPLACE INTO positions(model,sims,macts,path,entry,ts)"
                    " VALUES(?,?,?,?,?,?)",
                    (self.model_id, self.sims, self.m_actions,
                     self._path(prefix_tuple), payload, self._seq))
            # Cap the store: drop the oldest rows beyond disk_max.
            self._conn.execute(
                "DELETE FROM positions WHERE ts <= "
                "(SELECT ts FROM positions ORDER BY ts DESC LIMIT 1 OFFSET ?)",
                (self.disk_max,))
            self._conn.commit()

    def load_recent(self, n):
        """The newest ``n`` (prefix_tuple, entry), oldest-first (so replaying them
        into an LRU reproduces the recency order)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, entry FROM positions WHERE model=? AND sims=? AND macts=?"
                " ORDER BY ts DESC LIMIT ?",
                (self.model_id, self.sims, self.m_actions, n)).fetchall()
        out = []
        for path, entry in reversed(rows):
            try:
                pt = tuple(tuple(c) for c in json.loads(path))
                out.append((pt, json.loads(entry)))
            except (ValueError, TypeError):
                continue
        return out


class AnalyzeContext:
    """Container for the analysis endpoints' model + game rules + inference guard.

    In ``run()`` the ``guard`` MUST be the same instance the bot uses so
    analysis and bot play share one inference cap (analysis waits up to
    ``analyze_wait_timeout`` seconds for a slot → 503 only if the guard stays
    saturated). ``from_checkpoint`` builds its OWN guard and is intended for
    standalone analysis (no bot running) — do NOT wire it into ``run()``
    alongside a bot, or the two will oversubscribe the inference budget.
    """

    def __init__(self, model, model_config, guard: InferenceGuard,
                 win_length: int, placement_radius: int, max_moves: int,
                 max_analyze_moves: int = 220, mcts_sims: int = 64, m_actions: int = 16,
                 cache_size: int = 4096, model_id: str | None = None,
                 cache_db_path: str | None = None,
                 analyze_wait_timeout: float = 8.0):
        self.model = model
        self.model_config = model_config
        self.guard = guard
        self.win_length = win_length
        self.placement_radius = placement_radius
        self.max_moves = max_moves
        self.max_analyze_moves = max_analyze_moves
        self.mcts_sims = mcts_sims
        self.m_actions = m_actions
        # How long /analyze and /analyze_trajectory wait for a guard slot
        # before 503ing. Bounded (not infinite) so a stampede can't pile
        # threads up behind a long solve indefinitely.
        self.analyze_wait_timeout = analyze_wait_timeout
        # Serializes full-game analysis jobs across sessions (only one at a time);
        # a second /analyze_game streams `queued` events while blocked on this.
        self._job_lock = threading.Lock()
        # Per-POSITION LRU cache. Each analysed position is stored once, keyed by
        # the full move-path to it: (moves[:i+1], sims, m) -> trajectory[i].
        # This is a hash-trie: exact hits, prefix-serving, and warm-start are all
        # just O(depth) path walks (no scan), and overlapping lines (mainline +
        # variations + extensions) share their common head instead of each
        # storing a redundant copy. The model is fixed per process, so the key
        # need not include it. Eviction is plain LRU over positions; shared heads
        # are touched on every walk, so they stay hottest and evict last.
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_max = cache_size
        # Optional on-disk backing so the cache survives restarts. The store is
        # keyed by model_id (checkpoint identity), so a model change won't reuse
        # another model's entries (but keeps them for switching back); memory is
        # a hot LRU subset warmed from disk on startup.
        self._store = None
        if cache_db_path and model_id and cache_size > 0:
            try:
                self._store = _PositionStore(cache_db_path, model_id, mcts_sims, m_actions)
                for pt, entry in self._store.load_recent(cache_size):
                    self._cache[(pt, mcts_sims, m_actions)] = entry
                logger.info("analysis cache: %d positions warmed from %s",
                            len(self._cache), cache_db_path)
            except Exception:
                logger.exception("analysis cache: disk store disabled")
                self._store = None

    def cache_get(self, moves):
        """Assemble the full analysis for ``moves`` iff EVERY position on the
        path is cached (in memory or on disk). Serves exact hits and prefixes of
        longer games alike. Returns None on the first missing position."""
        if self._cache_max <= 0 or not moves:
            return None
        mt = tuple(moves)
        entries = []
        with self._cache_lock:
            for i in range(len(mt)):
                pt = mt[:i + 1]
                key = (pt, self.mcts_sims, self.m_actions)
                e = self._cache.get(key)
                if e is None and self._store is not None:
                    e = self._store.get(pt)   # disk fallback → promote to memory
                if e is None:
                    return None
                entries.append(e)
                self._cache[key] = e
                self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return _assemble_result(entries, self.mcts_sims)

    def cache_warm(self, moves):
        """The longest contiguous cached HEAD of ``moves`` (positions 0..k-1),
        passed to analyze_game_full so the shared head skips MCTS. Stops at the
        first uncached position (checking disk too). None if even the seed is
        uncached."""
        if self._cache_max <= 0 or not moves:
            return None
        mt = tuple(moves)
        entries = []
        with self._cache_lock:
            for i in range(len(mt)):
                pt = mt[:i + 1]
                key = (pt, self.mcts_sims, self.m_actions)
                e = self._cache.get(key)
                if e is None and self._store is not None:
                    e = self._store.get(pt)
                if e is None:
                    break
                entries.append(e)
                self._cache[key] = e
                self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return entries or None

    def cache_put(self, moves, result):
        """Explode a result into per-position entries (memory + disk). Only
        positions that were actually evaluated (terminal, or MCTS produced
        q_hat) are stored, so a partial run contributes its completed head but
        never a raw-policy stub that would later masquerade as fully analysed."""
        if self._cache_max <= 0:
            return
        mt = tuple(moves)
        traj = result.get("trajectory", [])
        to_disk = []
        with self._cache_lock:
            for i, entry in enumerate(traj):
                if not (entry.get("terminal") or entry.get("q_hat") is not None):
                    continue
                pt = mt[:i + 1]
                self._cache[(pt, self.mcts_sims, self.m_actions)] = entry
                self._cache.move_to_end((pt, self.mcts_sims, self.m_actions))
                to_disk.append((pt, entry))
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        if self._store is not None and to_disk:
            self._store.put_many(to_disk)   # DB write outside the memory lock

    @classmethod
    def from_checkpoint(cls, checkpoint: str, *, win_length: int, placement_radius: int,
                        max_moves: int, max_analyze_moves: int = 220,
                        mcts_sims: int = 64, m_actions: int = 16, device: str = "cpu"):
        """Standalone analysis context with its own guard (no bot alongside)."""
        model, mc, _ = load_model(checkpoint, device)
        guard = InferenceGuard()
        return cls(model, mc, guard, win_length, placement_radius, max_moves,
                   max_analyze_moves=max_analyze_moves, mcts_sims=mcts_sims, m_actions=m_actions)


def _normalize_moves(payload: dict) -> list[tuple[int, int]]:
    """Accept moves as list of [q,r] or list of [q,r,player].

    Raises ``ValueError`` (caught by the handlers → HTTP 400) on any malformed
    entry — including non-numeric elements that would otherwise raise
    ``TypeError`` — and rejects coordinates outside ``[-COORD_MAX, COORD_MAX]``
    before they reach the engine.
    """
    raw = payload.get("moves", [])
    if not isinstance(raw, (list, tuple)):
        raise ValueError("moves must be a list")
    out = []
    for m in raw:
        try:
            if isinstance(m, (list, tuple)) and len(m) >= 2:
                a, b = m[0], m[1]
            elif isinstance(m, dict) and "q" in m and "r" in m:
                a, b = m["q"], m["r"]
            else:
                raise ValueError(f"bad move entry: {m!r}")
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"bad move entry: {m!r}") from e
        # Reject bool (int subclass) and non-integer floats so [true,0] / [1.5,0]
        # don't silently become (1,0) — mirrors /move's stricter handling.
        if isinstance(a, bool) or isinstance(b, bool):
            raise ValueError(f"bad move entry: {m!r}")
        if isinstance(a, float) and not a.is_integer():
            raise ValueError(f"bad move entry: {m!r}")
        if isinstance(b, float) and not b.is_integer():
            raise ValueError(f"bad move entry: {m!r}")
        try:
            q, r = int(a), int(b)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"bad move entry: {m!r}") from e
        if abs(q) > COORD_MAX or abs(r) > COORD_MAX:
            raise ValueError(f"coordinate out of bounds: ({q},{r})")
        out.append((q, r))
    return out


def handle_convert_hds(payload: dict, *, fetch=None) -> tuple[int, dict]:
    """Pure helper for POST /convert_hds: a hexo.did.science sandbox URL/code
    -> HTTTX. ``fetch`` is injectable for tests. Returns (status, body)."""
    if not isinstance(payload, dict):
        return 400, {"error": "request body must be a JSON object"}
    src = payload.get("src", "")
    if not isinstance(src, str):
        return 400, {"error": "src must be a string"}
    try:
        return 200, hds.hds_to_htttx(src, fetch=fetch)
    except hds.HdsError as e:
        return e.http_status, {"error": str(e)}


def handle_analyze(payload: dict, ctx: AnalyzeContext) -> tuple[int, dict]:
    """Pure helper for POST /analyze. Returns (status, body)."""
    try:
        moves = _normalize_moves(payload)
    except ValueError as e:
        return 400, {"error": str(e)}
    if len(moves) > ctx.max_analyze_moves:
        return 400, {"error": f"too many moves: {len(moves)} > {ctx.max_analyze_moves}"}
    if len(moves) == 0:
        moves = [(0, 0)]

    def _infer():
        import hexo_rs as hr
        state = build_state(moves, ctx.win_length, ctx.placement_radius, ctx.max_moves)
        cfg = hr.GameConfig(ctx.win_length, ctx.placement_radius, ctx.max_moves)
        result = analyze_position(
            ctx.model, ctx.model_config, state,
            mcts_sims=ctx.mcts_sims, mcts_m_actions=ctx.m_actions,
            game_config=cfg,
        )
        return result.to_json()

    try:
        body = ctx.guard.run(_infer, timeout=ctx.analyze_wait_timeout)
    except InferenceBusy:
        return 503, {"error": "analysis busy — another inference is in flight"}
    except ValueError as e:
        # build_state rejects illegal move sequences with ValueError; surface as
        # 400 rather than letting it escape to the generic 500 handler.
        return 400, {"error": str(e)}

    return 200, body


def handle_trajectory(payload: dict, ctx: AnalyzeContext) -> tuple[int, dict]:
    """Pure helper for POST /analyze_trajectory. Returns (status, body)."""
    try:
        moves = _normalize_moves(payload)
    except ValueError as e:
        return 400, {"error": str(e)}
    if len(moves) > ctx.max_analyze_moves:
        return 400, {"error": f"too many moves: {len(moves)} > {ctx.max_analyze_moves}"}
    if len(moves) == 0:
        return 400, {"error": "moves required"}

    def _infer():
        return analyze_trajectory(
            ctx.model, ctx.model_config, moves,
            ctx.win_length, ctx.placement_radius, ctx.max_moves,
        )

    try:
        body = ctx.guard.run(_infer, timeout=ctx.analyze_wait_timeout)
    except InferenceBusy:
        return 503, {"error": "analysis busy — another inference is in flight"}
    except ValueError as e:
        return 400, {"error": str(e)}

    return 200, body


def _sse_frame(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def handle_analyze_game_sse(handler, ctx: AnalyzeContext, body: dict,
                            job_semaphore: threading.Semaphore):
    """Stream a full-game MCTS analysis over Server-Sent Events.

    Frames: ``queued`` (while waiting for the job lock), ``progress``
    ({done,total,eta_seconds}), ``result`` (the analyze_game_full payload),
    and ``error`` (on bad input / busy). The connection closes after the job.

    The MCTS loop acquires ``ctx.guard`` PER prefix (blocking=True) so bot moves
    interleave between prefixes. ``ctx._job_lock`` serializes jobs across
    sessions; ``job_semaphore`` caps how many sessions can be queued at once.
    Cancellation: a write to a disconnected client raises BrokenPipeError, which
    sets the cancel flag and stops the job early (partial results discarded).
    """
    wfile = handler.wfile

    # Validate input before sending any SSE headers (so we can still 400/413).
    try:
        moves = _normalize_moves(body)
    except ValueError as e:
        handler._send_json(400, {"error": str(e)})
        return
    if len(moves) > ctx.max_analyze_moves:
        handler._send_json(400, {"error": f"too many moves: {len(moves)} > {ctx.max_analyze_moves}"})
        return
    if len(moves) == 0:
        handler._send_json(400, {"error": "moves required"})
        return

    if not job_semaphore.acquire(blocking=False):
        handler._send_json(503, {"error": "analysis queue full — try again shortly"})
        return

    try:
        # SSE headers.
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler._security_headers()
        handler.end_headers()

        disconnected = {"flag": False}

        def emit(event, data):
            """Write an SSE frame; flag disconnect on a broken pipe."""
            try:
                wfile.write(_sse_frame(event, data))
                wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                disconnected["flag"] = True

        # Cache hit: stream the stored trajectory immediately — no job lock, no
        # GPU. This is what keeps shared deep links / reloads from re-running
        # MCTS over the whole game.
        cached = ctx.cache_get(moves)
        if cached is not None:
            n = len(moves)
            emit("progress", {"done": n, "total": n, "eta_seconds": 0})
            emit("result", cached)
            return

        # Acquire the cross-session job lock, streaming `queued` while we wait.
        acquired = ctx._job_lock.acquire(blocking=False)
        while not acquired and not disconnected["flag"]:
            emit("queued", {"message": "another analysis is running"})
            acquired = ctx._job_lock.acquire(timeout=2.0)
        if disconnected["flag"]:
            return  # client left before we even started

        try:
            start = time.monotonic()

            def progress_cb(done, total):
                elapsed = time.monotonic() - start
                eta = (elapsed / done) * (total - done) if done > 0 else None
                emit("progress", {
                    "done": done, "total": total,
                    "eta_seconds": round(eta, 1) if eta is not None else None,
                })

            def cancel_cb():
                return disconnected["flag"]

            import hexo_rs as hr
            forcing_cfg = hr.GameConfig(ctx.win_length, ctx.placement_radius, ctx.max_moves)

            def run_mcts_fn(state):
                # Per-prefix guard acquire (blocking=True): wait for bot moves,
                # release between prefixes so play interleaves. Uses the
                # tighter TRAJECTORY_* forcing budget: this solves twice per
                # placement across the whole game, unlike the single-position
                # /analyze path (ANALYSIS_* default) — see analysis.py's consts.
                return ctx.guard.run(
                    lambda: analyze_position(
                        ctx.model, ctx.model_config, state,
                        mcts_sims=ctx.mcts_sims, mcts_m_actions=ctx.m_actions,
                        game_config=forcing_cfg,
                        forcing_depth_cap=TRAJECTORY_DEPTH_CAP,
                        forcing_node_budget=TRAJECTORY_NODE_BUDGET),
                    blocking=True)

            # Warm-start from the longest cached prefix of this line (if any),
            # so extending a previously-analysed line only searches the new tail.
            warm = ctx.cache_warm(moves)
            result = analyze_game_full(
                ctx.model, ctx.model_config, moves,
                ctx.win_length, ctx.placement_radius, ctx.max_moves,
                mcts_sims=ctx.mcts_sims, mcts_m_actions=ctx.m_actions,
                progress_cb=progress_cb, cancel_cb=cancel_cb, run_mcts_fn=run_mcts_fn,
                warm_trajectory=warm,
            )
            if not disconnected["flag"]:
                # Only cache a fully-completed run (a cancelled job has holes).
                if not result.get("cancelled"):
                    ctx.cache_put(moves, result)
                emit("result", result)
        finally:
            ctx._job_lock.release()
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            wfile.write(_sse_frame("error", {"error": "internal server error"}))
            wfile.flush()
        except Exception:
            pass
    finally:
        job_semaphore.release()


def _fmt_duration(created_at: str, completed_at: str) -> str:
    try:
        t0 = datetime.fromisoformat(created_at)
        t1 = datetime.fromisoformat(completed_at)
        sec = int((t1 - t0).total_seconds())
        if sec < 60:
            return f"{sec}s"
        m, s = divmod(sec, 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    except Exception:
        return "?"


_ADMIN_HTML = _load_template("admin.html")


def _render_admin(mgr: GameManager) -> str:
    active = mgr.active_games()
    if active:
        active_rows = []
        for g in active:
            name = html.escape(g["human_name"] or "—")
            human_side = g["human_side"]
            side_cls = "p1" if human_side == "P1" else "p2"
            cur = g["current_player"] or "—"
            cur_cls = "p1" if cur == "P1" else "p2" if cur == "P2" else ""
            whose = "human" if cur == human_side else "bot" if cur in ("P1", "P2") else ""
            when = html.escape(g["created_at"].replace("T", " ").split("+")[0])
            short_id = html.escape(g["game_id"][:8])
            active_rows.append(
                f'<tr class="row">'
                f'<td><code>{short_id}</code></td>'
                f'<td>{when}</td>'
                f'<td>{name}</td>'
                f'<td><span class="{side_cls}">{human_side}</span></td>'
                f'<td><span class="{cur_cls}">{cur}</span> ({whose})</td>'
                f'<td>{g["moves_remaining"]}</td>'
                f'<td>{g["n_moves"]}</td>'
                f'</tr>'
            )
        active_html = (
            f'<h2 style="color:#00d4ff;margin:8px 0">Active games ({len(active)})</h2>'
            "<table><thead><tr>"
            "<th>id</th><th>Started (UTC)</th><th>Name</th><th>Human side</th>"
            "<th>To move</th><th>Remaining</th><th>Moves so far</th>"
            "</tr></thead><tbody>"
            + "".join(active_rows)
            + "</tbody></table>"
        )
    else:
        active_html = ""

    recorder = mgr.recorder
    if recorder is None:
        return _ADMIN_HTML.format(
            summary="(no recorder configured)",
            active=active_html,
            table='<div class="empty">No database.</div>',
        )
    stats = recorder.summary()
    summary = (
        f"<span><b>{stats['total']}</b> completed</span>"
        f"<span>Active: <b>{len(active)}</b></span>"
        f"<span>Human wins: <b>{stats['human_wins']}</b></span>"
        f"<span>Bot wins: <b>{stats['bot_wins']}</b></span>"
        f"<span>Draws: <b>{stats['draws']}</b></span>"
        f"<span>Resigns: <b>{stats['resigns']}</b></span>"
    )
    games = recorder.recent_games(limit=200)
    if not games:
        table = '<div class="empty">No games played yet.</div>'
    else:
        rows = []
        for g in games:
            human_won = g["winner"] == g["human_side"]
            bot_won = g["winner"] == g["bot_side"]
            result_cls = "win-human" if human_won else ("win-bot" if bot_won else "")
            winner_label = (
                f'<span class="{result_cls}">'
                f'{html.escape(g["winner"] or "draw")} {"(human)" if human_won else "(bot)" if bot_won else ""}'
                f'</span>'
            )
            result_label = html.escape(g["result_type"])
            if g["result_type"] == "resign":
                result_label = f'<span class="result-resign">{result_label}</span>'
            duration = _fmt_duration(g["created_at"], g["completed_at"])
            when = html.escape(g["created_at"].replace("T", " ").split("+")[0])
            name = html.escape(g["human_name"] or "—")
            human_side = g["human_side"]
            side_cls = "p1" if human_side == "P1" else "p2"
            gid = int(g["id"])  # coerce: gid is interpolated into JS/HTML below
            model = html.escape(g.get("model_label") or "—")
            step = g.get("step")
            bot_cell = model + (f' <span class="step">· step {int(step)}</span>'
                                if step is not None else "")
            elo = g.get("opp_elo")
            src = g.get("elo_source")
            elo_cell = "—" if elo is None else (
                f'{int(elo)}' + (f' <span class="elo-src">({html.escape(str(src))})</span>'
                                 if src else ""))
            rows.append(
                f'<tr class="row">'
                f'<td>{gid}</td>'
                f'<td>{when}</td>'
                f'<td>{name}</td>'
                f'<td><span class="{side_cls}">{human_side}</span></td>'
                f'<td>{winner_label}</td>'
                f'<td>{result_label}</td>'
                f'<td>{g["n_moves"]}</td>'
                f'<td>{duration}</td>'
                f'<td>{g["mcts_sims"]}</td>'
                f'<td>{bot_cell}</td>'
                f'<td>{elo_cell}</td>'
                f'<td><button onclick="toggleHtttx({gid})">show</button>'
                f' <button onclick="copyHtttx({gid})">copy</button>'
                f' <span class="copy-msg" id="msg-{gid}"></span></td>'
                f'</tr>'
                f'<tr class="htttx-row" id="htttx-{gid}">'
                f'<td colspan="12"><pre id="htttx-text-{gid}">{html.escape(g["htttx"])}</pre></td>'
                f'</tr>'
            )
        table = (
            "<table><thead><tr>"
            "<th>#</th><th>When (UTC)</th><th>Name</th><th>Human side</th>"
            "<th>Winner</th><th>Result</th><th>Moves</th><th>Duration</th>"
            "<th>MCTS sims</th><th>Bot</th><th>Opp Elo</th><th>HTTTX</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    return _ADMIN_HTML.format(summary=summary, active=active_html, table=table)


HTML = _load_template("index.html")


def make_handler_class(mgr: GameManager, admin_token: str = "", url_prefix: str = "",
                       analyze_ctx: AnalyzeContext | None = None, max_body_bytes: int = MAX_BODY_BYTES,
                       max_concurrent: int = MAX_CONCURRENT,
                       analyze_limiter: RateLimiter | None = None,
                       new_game_limiter: RateLimiter | None = None,
                       analyze_game_limiter: RateLimiter | None = None):
    concurrency = threading.BoundedSemaphore(max_concurrent)
    # Caps concurrently-queued full-game analysis jobs (queue-slot exhaustion).
    analyze_game_semaphore = threading.Semaphore(ANALYZE_GAME_MAX_INFLIGHT)

    class Handler(BaseHTTPRequestHandler):
        timeout = 60

        def log_request(self, code='-', size='-'):
            # Restore a minimal access log for incident response (the frozen
            # script silenced this). Log method + path WITHOUT the query string
            # so an admin token passed as ?token=... never reaches the access log.
            logger.info("%s %s %s %s", self.address_string(), self.command,
                        urlparse(self.path).path, code)

        def log_message(self, fmt, *args):
            # Suppress the default stderr error-page logging; log_request covers access.
            pass

        def _client_key(self) -> str:
            return self.client_address[0] if self.client_address else "?"

        def _begin_request(self) -> bool:
            """Acquire a concurrency slot (non-blocking). Returns False if at cap."""
            return concurrency.acquire(blocking=False)

        def _end_request(self):
            try:
                concurrency.release()
            except ValueError:
                pass

        def _security_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            # no-store prevents a caching proxy from serving an authenticated
            # /admin response (keyed on path alone) to a later wrong-token request.
            self.send_header("Cache-Control", "no-store, private")
            # The play UI is inline <script>/<style>, so 'unsafe-inline' is
            # required; the policy still blocks external-script injection.
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                             "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                             "connect-src 'self'")

        def _send_json(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._security_headers()
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, body: str):
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self._security_headers()
            self.end_headers()
            self.wfile.write(data)

        def _send_static(self, rel: str):
            """Serve a file from the static/ dir. Guards against path traversal
            (resolved path must stay within static/) and only serves known asset
            types. Cacheable (these are content-hashed-by-mtime via ETag)."""
            if not rel or rel.endswith("/") or "\x00" in rel:
                self.send_error(404)
                return
            base = _STATIC_DIR.resolve()
            try:
                target = (base / rel).resolve()
                target.relative_to(base)          # raises if rel escapes the dir
            except (ValueError, OSError):
                self.send_error(404)
                return
            ctype = _STATIC_TYPES.get(target.suffix.lower())
            if ctype is None or not target.is_file():
                self.send_error(404)
                return
            try:
                data = target.read_bytes()
                st = target.stat()
            except OSError:
                self.send_error(404)
                return
            etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("ETag", etag)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            # Force the connection to close after this response. stdlib
            # http.server doesn't decode chunked bodies and we may reject before
            # reading the body; closing the connection flushes any unread bytes
            # so they can't be reinterpreted as a follow-up request (smuggling
            # defense). Safe for HTTP/1.0 (closes anyway) and future HTTP/1.1.
            self.close_connection = True
            # Reject chunked bodies: stdlib http.server doesn't decode
            # Transfer-Encoding: chunked, so an unread chunked body would bleed
            # into the next request (request-smuggling / desync).
            if self.headers.get("Transfer-Encoding"):
                raise BadRequestError("Transfer-Encoding not supported")
            raw = self.headers.get("Content-Length")
            try:
                length = int(raw) if raw is not None else 0
            except (TypeError, ValueError):
                raise BadRequestError("malformed Content-Length")
            if length < 0:
                # A negative Content-Length would make rfile.read(-1) consume
                # the whole stream and bypass the body cap — reject outright.
                raise BadRequestError("negative Content-Length")
            if length > max_body_bytes:
                raise PayloadTooLargeError(f"request body too large: {length} bytes")
            # Require a JSON content type BEFORE the empty-body short-circuit:
            # a cross-origin <form method=POST> with no fields sends Content-Length 0
            # and an application/x-www-form-urlencoded type — without this order the
            # CSRF defense (non-simple Content-Type) would be bypassed by empty bodies.
            ctype_full = (self.headers.get("Content-Type") or "").lower()
            if not ctype_full.startswith("application/json"):
                raise BadRequestError("Content-Type must be application/json")
            if length == 0:
                return {}
            data = self.rfile.read(length)
            try:
                obj = json.loads(data)
            except json.JSONDecodeError as e:
                raise BadRequestError(f"malformed JSON: {e}")
            if not isinstance(obj, dict):
                raise BadRequestError("request body must be a JSON object")
            return obj

        def _route_path(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if url_prefix:
                if path == url_prefix:
                    path = "/"
                elif path.startswith(url_prefix + "/"):
                    path = path[len(url_prefix):]
            return path, parsed.query

        def do_GET(self):
            # Close the connection after every response. This flushes any
            # unread body bytes so they can't be reinterpreted as a follow-up
            # request (smuggling/desync defense) on every path, including the
            # concurrency-cap 503 early return that bypasses _read_json.
            self.close_connection = True
            if not self._begin_request():
                self._send_json(503, {"error": "server busy — too many concurrent requests"})
                return
            try:
                self._do_get()
            finally:
                self._end_request()

        def _do_get(self):
            path, query = self._route_path()
            # Play ("/") and analysis ("/analysis") are the same SPA shell served
            # at two paths; the client selects the view from location.pathname.
            if path in ("/", "", "/analysis"):
                diffs = mgr.difficulty_sims or {}
                ordered = {k: diffs[k] for k in DIFFICULTY_ORDER if k in diffs}
                # json.dumps escapes " and \ but not <, and these land inside a
                # <script>; escape < so a crafted url_prefix can't break out of
                # the script context (defense-in-depth; values are operator-controlled).
                def _js(s):
                    return json.dumps(s).replace("<", "\\u003c").replace(">", "\\u003e")
                # Absolute origin for the OG/Twitter social-card tags (scrapers need
                # absolute image/URL). Built from the request's Host + X-Forwarded-Proto
                # (Cloudflare/proxy sets the latter), so it works for any domain; empty
                # on a hostless request, which degrades OG URLs to relative.
                host = self.headers.get("Host", "")
                proto = self.headers.get("X-Forwarded-Proto", "http")
                base_url = f"{proto}://{host}" if host else ""
                html_out = (HTML
                            .replace('"__URL_PREFIX__"', _js(url_prefix))
                            .replace('"__DIFFICULTY_SIMS__"', _js(ordered))
                            .replace('"__DEFAULT_DIFFICULTY__"', _js(mgr.default_difficulty))
                            # absolute origin for OG tags; bare prefix for attribute contexts
                            .replace('__BASE_URL__', html.escape(base_url, quote=True))
                            .replace('__PREFIX_RAW__', html.escape(url_prefix, quote=True))
                            # cache-bust assets on change (max static mtime)
                            .replace('__ASSET_V__', _asset_version()))
                self._send_html(html_out)
            elif path.startswith("/static/"):
                self._send_static(path[len("/static/"):])
            elif path == "/stats":
                self._send_json(200, mgr.recorder.bot_brag(mgr.model_label, mgr.model_step)
                                if mgr.recorder else {"model_label": mgr.model_label})
            elif path == "/admin":
                if not admin_token:
                    self.send_error(404)
                    return
                # Accept the token via X-Admin-Token header (preferred — keeps it
                # out of proxy access logs / browser history) or ?token= (back-compat).
                token = self.headers.get("X-Admin-Token", "")
                if not token:
                    qs = parse_qs(query)
                    token = qs.get("token", [""])[0]
                if not secrets.compare_digest(token, admin_token):
                    self.send_error(404)
                    return
                admin_html = (_render_admin(mgr)
                              .replace('__PREFIX_RAW__', html.escape(url_prefix, quote=True))
                              .replace('__ASSET_V__', _asset_version()))
                self._send_html(admin_html)
            else:
                self.send_error(404)

        def do_POST(self):
            # Close the connection after every response — see do_GET. Critical
            # here: the rate-limit/concurrency 503 early returns send a response
            # WITHOUT reading the body, and an unread POST body left on a kept-open
            # connection is a request-smuggling primitive.
            self.close_connection = True
            if not self._begin_request():
                self._send_json(503, {"error": "server busy — too many concurrent requests"})
                return
            try:
                path, _ = self._route_path()
                if path == "/new_game":
                    if new_game_limiter is not None and not new_game_limiter.allow(self._client_key()):
                        self._send_json(503, {"error": "rate limit: too many new games"})
                        return
                    body = self._read_json()
                    # Validate Elo before creating the game so a bad value does
                    # not leak a half-created record.
                    elo_val = _validate_elo(body.get("self_reported_elo"))
                    rec = mgr.create_game(
                        human_side_request=body.get("human_side", "random"),
                        name=body.get("name"),
                        difficulty=body.get("difficulty"),
                        self_reported_elo=elo_val,
                    )
                    self._send_json(200, {
                        "game_id": rec.game_id,
                        "state": mgr.snapshot(rec, placement_radius=mgr.game_kwargs["placement_radius"],
                                               win_length=mgr.game_kwargs["win_length"], include_htttx=True),
                    })
                elif path == "/state":
                    body = self._read_json()
                    gid = body.get("game_id")
                    if not isinstance(gid, str):
                        raise BadRequestError("missing game_id")
                    rec = mgr.get_game(gid)
                    if rec is None:
                        self._send_json(404, {"error": "unknown game_id"})
                        return
                    self._send_json(200, {"state": mgr.snapshot(rec, placement_radius=mgr.game_kwargs["placement_radius"],
                                                                win_length=mgr.game_kwargs["win_length"], include_htttx=True)})
                elif path == "/move":
                    body = self._read_json()
                    gid = body.get("game_id")
                    if not isinstance(gid, str):
                        raise BadRequestError("missing game_id")
                    q_raw = body.get("q")
                    r_raw = body.get("r")
                    # Reject bool (a bool is an int subclass) and non-integers
                    # so true/1.5 don't silently become legal coords.
                    if isinstance(q_raw, bool) or isinstance(r_raw, bool) \
                            or not isinstance(q_raw, int) or not isinstance(r_raw, int):
                        raise BadRequestError("q/r must be integers")
                    rec = mgr.apply_human_move(gid, q_raw, r_raw)
                    self._send_json(200, {"state": mgr.snapshot(rec, placement_radius=mgr.game_kwargs["placement_radius"],
                                                                win_length=mgr.game_kwargs["win_length"], include_htttx=True)})
                elif path == "/resign":
                    body = self._read_json()
                    gid = body.get("game_id")
                    if not isinstance(gid, str):
                        raise BadRequestError("missing game_id")
                    rec = mgr.resign(gid)
                    self._send_json(200, {"state": mgr.snapshot(rec, placement_radius=mgr.game_kwargs["placement_radius"],
                                                                win_length=mgr.game_kwargs["win_length"], include_htttx=True)})
                elif path == "/analyze":
                    if analyze_limiter is not None and not analyze_limiter.allow(self._client_key()):
                        self._send_json(503, {"error": "rate limit: too many analysis requests"})
                        return
                    body = self._read_json()
                    if analyze_ctx is None:
                        self._send_json(503, {"error": "analysis disabled"})
                        return
                    status, resp = handle_analyze(body, analyze_ctx)
                    self._send_json(status, resp)
                elif path == "/analyze_trajectory":
                    if analyze_limiter is not None and not analyze_limiter.allow(self._client_key()):
                        self._send_json(503, {"error": "rate limit: too many analysis requests"})
                        return
                    body = self._read_json()
                    if analyze_ctx is None:
                        self._send_json(503, {"error": "analysis disabled"})
                        return
                    status, resp = handle_trajectory(body, analyze_ctx)
                    self._send_json(status, resp)
                elif path == "/analyze_game":
                    if analyze_game_limiter is not None and not analyze_game_limiter.allow(self._client_key()):
                        self._send_json(503, {"error": "rate limit: too many full-game analyses"})
                        return
                    body = self._read_json()
                    if analyze_ctx is None:
                        self._send_json(503, {"error": "analysis disabled"})
                        return
                    # Streams its own SSE response (status/headers/body).
                    handle_analyze_game_sse(self, analyze_ctx, body, analyze_game_semaphore)
                elif path == "/game_htttx":
                    body = self._read_json()
                    try:
                        htttx = mgr.htttx(body.get("game_id", ""))
                        self._send_json(200, {"htttx": htttx})
                    except UnknownGameError as e:
                        self._send_json(404, {"error": str(e)})
                elif path == "/convert_hds":
                    body = self._read_json()
                    status, resp = handle_convert_hds(body)
                    self._send_json(status, resp)
                else:
                    self.send_error(404)
            except GameError as e:
                self._send_json(e.http_status, {"error": str(e)})
            except Exception:
                import traceback
                traceback.print_exc()
                # Never leak internal exception text to the public client.
                self._send_json(500, {"error": "internal server error"})
            finally:
                self._end_request()

    return Handler


def run(cfg, analyze_ctx: AnalyzeContext | None = None):
    """Start the public play server.

    ``cfg`` is any object exposing the attributes listed in the plan:
    config, checkpoint, win_length, placement_radius, max_moves,
    mcts_sims, m_actions, db, port, bind, url_prefix, admin_token,
    model_label, difficulty_sims, default_difficulty, request_timeout,
    live_forcing (optional, defaults to True), difficulty_forcing_depth
    (optional, defaults to game.py's DEFAULT_DIFFICULTY_FORCING_DEPTH).

    The advisory flock on ``<db>.lock`` prevents two ``hexo-a0 serve``
    processes from sharing the same SQLite DB. It does NOT protect against
    the frozen ``scripts/play_server.py``, which takes no lock; the cutover
    runbook's manual "stop the old server" step is the real guarantee there.
    """
    import fcntl
    lock_path = f"{cfg.db}.lock"
    # O_NOFOLLOW: refuse to follow a symlink planted at <db>.lock (a local
    # attacker with write access to the db dir could otherwise point it at a
    # sensitive file and have us truncate it).
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o644)
    except OSError as e:
        print(f"ERROR: cannot open lock file {lock_path} (is it a symlink?): {e}")
        return 1
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        print(f"ERROR: another `hexo-a0 serve` is using this DB (lock held on {lock_path}).")
        print("Stop the other `hexo-a0 serve` instance before starting a new one.")
        print("NOTE: this lock does NOT detect a concurrently-running frozen "
              "`scripts/play_server.py` (which takes no lock) — stop it manually.")
        return 1

    try:
        return _serve(cfg, analyze_ctx)
    finally:
        # Release the advisory lock on every exit path (startup error, signal,
        # normal shutdown) so a restart isn't blocked by a stale lock.
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _serve(cfg, analyze_ctx):
    """Server body, run under run()'s flock try/finally."""
    model, mc, ckpt = load_model(cfg.checkpoint, "cpu",
                                 model_config_dict=_model_config_fallback(cfg))

    # CPU inference: N slots let analysis run beside the bot instead of 503ing.
    # Cap torch's intra-op threads so N concurrent forward passes don't
    # oversubscribe the cores (torch defaults to using all of them per op).
    inference_workers = max(1, getattr(cfg, "inference_workers", 2))
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // inference_workers))
    guard = InferenceGuard(capacity=inference_workers)
    recorder = Recorder(cfg.db)
    difficulty_sims = _parse_difficulty_sims(cfg.difficulty_sims)
    if cfg.default_difficulty not in difficulty_sims:
        raise SystemExit(
            f"--default-difficulty {cfg.default_difficulty!r} is not in the "
            f"configured tiers {list(difficulty_sims)}"
        )

    model_label = cfg.model_label or _derive_model_label(cfg.checkpoint, ckpt)
    model_step = _derive_step(cfg.checkpoint, ckpt)

    game_kwargs = dict(
        win_length=cfg.win_length,
        placement_radius=cfg.placement_radius,
        max_moves=cfg.max_moves,
    )
    bot_fn = make_bot_turn_fn(
        model=model, model_config=mc,
        mcts_sims=cfg.mcts_sims, m_actions=cfg.m_actions,
        difficulty_sims=difficulty_sims,
        # No CLI knob for this (yet): every tier's forcing depth comes from
        # game.py's measured DEFAULT_DIFFICULTY_FORCING_DEPTH map unless a
        # caller-built cfg opts into a custom one.
        difficulty_forcing_depth=getattr(cfg, "difficulty_forcing_depth", None),
        guard=guard,
        game_kwargs=game_kwargs,
        live_forcing=getattr(cfg, "live_forcing", True),
    )
    mgr = GameManager(
        game_kwargs=game_kwargs,
        bot_turn_fn=bot_fn,
        mcts_sims=cfg.mcts_sims,
        m_actions=cfg.m_actions,
        checkpoint_path=str(Path(cfg.checkpoint).resolve()),
        model_label=model_label,
        model_step=model_step,
        difficulty_sims=difficulty_sims,
        default_difficulty=cfg.default_difficulty,
        recorder=recorder,
    )
    restored_games = mgr.restore_active_games()

    if analyze_ctx is None:
        # Model identity for cache invalidation: label (run@step) + the
        # checkpoint file's size+mtime, so retraining the same path also busts it.
        model_id = model_label
        try:
            cst = Path(cfg.checkpoint).stat()
            model_id = f"{model_label}:{cst.st_size}:{int(cst.st_mtime)}"
        except OSError:
            pass
        # Persist the analysis cache beside the games DB (skip for :memory:).
        cache_db = None
        if cfg.db and cfg.db != ":memory:":
            cache_db = str(Path(cfg.db).with_suffix(".analysis.sqlite"))
        analyze_ctx = AnalyzeContext(
            model, mc, guard,
            win_length=cfg.win_length,
            placement_radius=cfg.placement_radius,
            max_moves=cfg.max_moves,
            mcts_sims=cfg.mcts_sims,
            m_actions=cfg.m_actions,
            model_id=model_id,
            cache_db_path=cache_db,
        )

    admin_token = secrets.token_urlsafe(16) if cfg.admin_token is None else cfg.admin_token
    url_prefix = cfg.url_prefix.rstrip("/")
    if url_prefix and not url_prefix.startswith("/"):
        raise SystemExit(f"--url-prefix must start with '/' (got: {url_prefix!r})")

    analyze_limiter = RateLimiter(ANALYZE_RATE_PER_MIN)
    new_game_limiter = RateLimiter(NEW_GAME_RATE_PER_MIN)
    analyze_game_limiter = RateLimiter(ANALYZE_GAME_RATE_PER_MIN)
    handler_cls = make_handler_class(mgr, admin_token=admin_token, url_prefix=url_prefix,
                                     analyze_ctx=analyze_ctx,
                                     analyze_limiter=analyze_limiter,
                                     new_game_limiter=new_game_limiter,
                                     analyze_game_limiter=analyze_game_limiter)
    # Per-connection socket timeout comes from Handler.timeout (a class attr),
    # NOT server.timeout (which only governs serve_forever's poll interval).
    # Wire --request-timeout to the one that actually bounds a stuck request.
    handler_cls.timeout = getattr(cfg, "request_timeout", 60)
    server = ThreadingHTTPServer((cfg.bind, cfg.port), handler_cls)
    server.timeout = getattr(cfg, "request_timeout", 60)

    print(f"Play server running at http://{cfg.bind}:{cfg.port}{url_prefix}/")
    print(f"  checkpoint: {cfg.checkpoint}")
    print(f"  model:      {model_label}")
    print(f"  rules: win={cfg.win_length} radius={cfg.placement_radius} "
          f"max={cfg.max_moves}  m_actions={cfg.m_actions}")
    tier_str = "  ".join(f"{k}={v}" for k, v in difficulty_sims.items())
    print(f"  difficulty: {tier_str}  (default: {cfg.default_difficulty})")
    print(f"  db: {cfg.db}")
    if restored_games:
        print(f"  restored:   {restored_games} in-progress game(s)")
    if url_prefix:
        print(f"  url-prefix: {url_prefix}")
    if admin_token:
        # Print the admin path WITHOUT the token to stdout (avoid proxy/journal
        # capture); emit the token itself to stderr so the operator sees it in
        # the terminal without it landing in piped stdout logs.
        print(f"  admin: http://{cfg.bind}:{cfg.port}{url_prefix}/admin?token=<TOKEN>")
        import sys as _sys
        print(f"  admin token (stderr only): {admin_token}", file=_sys.stderr)
    else:
        print("  admin: disabled (--admin-token empty)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
    return 0
