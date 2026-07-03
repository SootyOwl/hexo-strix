"""HTTP-level tests for the /analyze_game SSE endpoint.

Uses a fake AnalyzeContext (no real model) and monkeypatches
``analyze_game_full`` so the streaming framing, job-lock queue, rate limiting,
and cancel-on-disconnect can be exercised without a GPU/checkpoint.
"""
import http.client
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import hexo_a0.serving.app as app
from hexo_a0.serving.app import AnalyzeContext, _PositionStore, make_handler_class
from hexo_a0.serving.game import GameManager
from hexo_a0.serving.inference import InferenceGuard


def _full_result(moves, sims=16):
    """A synthetic analyze_game_full-shaped result: one trajectory entry per
    position (len == len(moves)), all non-terminal with MCTS fields present."""
    n = len(moves)
    traj = [{"current_player": "P2" if i % 2 == 0 else "P1", "terminal": False,
             "value": 0.1 * i, "q_hat": [0.2, 0.1], "legal": [[0, 0], [1, 0]],
             "probs": [0.6, 0.4]} for i in range(n)]
    return {"trajectory": traj,
            "boundary_indices": [i for i in range(n) if i % 2 == 0],
            "evaluated_prefixes": n, "mcts_sims": sims,
            "cancelled": False, "completed_prefixes": n}


def _real_ctx(sims=16, m=16):
    return AnalyzeContext(object(), object(), InferenceGuard(),
                          win_length=6, placement_radius=8, max_moves=400,
                          mcts_sims=sims, m_actions=m)


def _noop_mgr():
    return GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=lambda rec: None, recorder=None,
        mcts_sims=64, m_actions=16, checkpoint_path="x.pt", model_label="t",
        difficulty_sims={"standard": 64}, default_difficulty="standard",
        idle_ttl_seconds=3600, max_games=10,
    )


def _fake_ctx(cache=None):
    """An AnalyzeContext-shaped stub. model/model_config are unused because we
    monkeypatch analyze_game_full; guard is real so per-prefix run works.

    ``cache`` is an optional dict keyed by ``tuple(moves)``; supplying it lets a
    test exercise the cache-hit fast path (no job lock, no analyze_game_full)."""
    cache = {} if cache is None else cache
    return SimpleNamespace(
        model=object(), model_config=object(), guard=InferenceGuard(),
        win_length=6, placement_radius=8, max_moves=400,
        max_analyze_moves=220, mcts_sims=16, m_actions=16,
        _job_lock=threading.Lock(),
        cache_get=lambda moves: cache.get(tuple(moves)),
        cache_put=lambda moves, result: cache.__setitem__(tuple(moves), result),
        cache_warm=lambda moves: None,
    )


@contextmanager
def _server(analyze_ctx, analyze_game_limiter=None):
    Handler = make_handler_class(
        _noop_mgr(), admin_token="", url_prefix="",
        analyze_ctx=analyze_ctx, analyze_game_limiter=analyze_game_limiter)
    srv = __import__("http.server", fromlist=["ThreadingHTTPServer"]).ThreadingHTTPServer(
        ("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()


def _get(port, path, headers=None, timeout=5):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    out = (resp.status, resp.getheader("Content-Type"),
           resp.getheader("ETag"), resp.getheader("Cache-Control"), body)
    conn.close()
    return out


def test_static_serves_css_and_js():
    with _server(analyze_ctx=None) as port:
        s, ctype, etag, cache, body = _get(port, "/static/observatory.css")
        assert s == 200 and ctype == "text/css; charset=utf-8"
        assert etag and "max-age" in (cache or "") and len(body) > 0
        s2, ctype2, _, _, _ = _get(port, "/static/play.js")
        assert s2 == 200 and ctype2 == "text/javascript; charset=utf-8"


def test_static_etag_304():
    with _server(analyze_ctx=None) as port:
        _, _, etag, _, _ = _get(port, "/static/play.js")
        s, _, _, cache, body = _get(port, "/static/play.js", headers={"If-None-Match": etag})
        assert s == 304 and body == b"" and "max-age" in (cache or "")


def test_static_blocks_traversal_and_unknown_types():
    with _server(analyze_ctx=None) as port:
        assert _get(port, "/static/../app.py")[0] == 404
        assert _get(port, "/static/..%2fapp.py")[0] == 404
        assert _get(port, "/static/nope.txt")[0] == 404
        assert _get(port, "/static/")[0] == 404


def _read_sse(port, body=b'{"moves":[[0,0],[1,0],[2,0]]}', timeout=5):
    """POST /analyze_game and return the raw SSE response body text."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", "/analyze_game", body=body,
                 headers={"Content-Type": "application/json",
                          "Content-Length": str(len(body))})
    resp = conn.getresponse()
    ctype = resp.getheader("Content-Type")
    raw = resp.read().decode()
    status = resp.status
    conn.close()
    return status, ctype, raw


def _fake_analyze_game_full(progress=3):
    """Build a fake analyze_game_full that fires progress_cb and respects cancel_cb."""
    def fake(model, mc, moves, win, radius, mx, *, mcts_sims=64, mcts_m_actions=16,
             progress_cb=None, cancel_cb=None, run_mcts_fn=None, warm_trajectory=None):
        total = len(moves)
        completed = 0
        for i in range(total):
            if cancel_cb and cancel_cb():
                return {"trajectory": [], "boundary_indices": [], "evaluated_prefixes": total,
                        "mcts_sims": mcts_sims, "cancelled": True, "completed_prefixes": completed}
            completed = i + 1
            if progress_cb:
                progress_cb(completed, total)
        return {"trajectory": [{"value": 0.0, "current_player": "P1", "terminal": False}],
                "boundary_indices": [0], "evaluated_prefixes": total,
                "mcts_sims": mcts_sims, "cancelled": False, "completed_prefixes": completed}
    return fake


def test_analyze_game_streams_progress_and_result(monkeypatch):
    monkeypatch.setattr(app, "analyze_game_full", _fake_analyze_game_full())
    with _server(_fake_ctx()) as port:
        status, ctype, raw = _read_sse(port)
        assert status == 200
        assert ctype == "text/event-stream"
        assert "event: progress" in raw
        assert "event: result" in raw
        assert '"eta_seconds"' in raw


def test_analyze_game_cache_hit_skips_analysis(monkeypatch):
    # A pre-populated cache entry must be streamed back without ever calling
    # analyze_game_full (no MCTS, no job lock).
    called = {"n": 0}

    def must_not_run(*a, **k):
        called["n"] += 1
        raise AssertionError("analyze_game_full ran on a cache hit")

    monkeypatch.setattr(app, "analyze_game_full", must_not_run)
    moves = [[0, 0], [1, 0], [2, 0]]
    cached_result = {"trajectory": [{"value": 0.42, "current_player": "P1",
                                     "terminal": False}],
                     "boundary_indices": [0], "evaluated_prefixes": 3,
                     "mcts_sims": 16, "cancelled": False, "completed_prefixes": 3}
    cache = {tuple((q, r) for q, r in moves): cached_result}
    with _server(_fake_ctx(cache=cache)) as port:
        status, ctype, raw = _read_sse(port)
        assert status == 200
        assert ctype == "text/event-stream"
        assert "event: result" in raw
        assert "0.42" in raw
        assert called["n"] == 0


def test_analyze_game_caches_completed_run(monkeypatch):
    # After a successful run, a second identical request is served from cache.
    runs = {"n": 0}

    def counting_fake(model, mc, moves, win, radius, mx, *, mcts_sims=64,
                      mcts_m_actions=16, progress_cb=None, cancel_cb=None, run_mcts_fn=None, warm_trajectory=None):
        runs["n"] += 1
        if progress_cb:
            progress_cb(len(moves), len(moves))
        return {"trajectory": [{"value": 0.1, "current_player": "P1", "terminal": False}],
                "boundary_indices": [0], "evaluated_prefixes": len(moves),
                "mcts_sims": mcts_sims, "cancelled": False, "completed_prefixes": len(moves)}

    monkeypatch.setattr(app, "analyze_game_full", counting_fake)
    with _server(_fake_ctx()) as port:
        s1, _, r1 = _read_sse(port)
        s2, _, r2 = _read_sse(port)
        assert s1 == 200 and s2 == 200
        assert "event: result" in r1 and "event: result" in r2
        assert runs["n"] == 1  # second request hit the cache


def test_analyze_game_does_not_cache_cancelled(monkeypatch):
    # A cancelled (partial) run must NOT be cached, so a later request re-runs.
    runs = {"n": 0}

    def cancel_fake(model, mc, moves, win, radius, mx, *, mcts_sims=64,
                    mcts_m_actions=16, progress_cb=None, cancel_cb=None, run_mcts_fn=None, warm_trajectory=None):
        runs["n"] += 1
        return {"trajectory": [], "boundary_indices": [], "evaluated_prefixes": len(moves),
                "mcts_sims": mcts_sims, "cancelled": True, "completed_prefixes": 0}

    monkeypatch.setattr(app, "analyze_game_full", cancel_fake)
    cache = {}
    with _server(_fake_ctx(cache=cache)) as port:
        _read_sse(port)
        _read_sse(port)
        assert runs["n"] == 2  # neither run was cached
        assert cache == {}


# --- per-position prefix cache: pure, no model ----------------------------

def test_cache_get_serves_exact_and_prefix():
    ctx = _real_ctx()
    long_moves = [(0, 0), (1, 0), (2, 0), (0, 1)]
    ctx.cache_put(long_moves, _full_result(long_moves))
    # exact: every position present -> assembled full result
    full = ctx.cache_get(long_moves)
    assert full and full["evaluated_prefixes"] == 4 and len(full["trajectory"]) == 4
    # prefix: a shorter line is served from the longer game's stored positions
    pref = ctx.cache_get([(0, 0), (1, 0)])
    assert pref and len(pref["trajectory"]) == 2 and pref["evaluated_prefixes"] == 2
    # a divergent line misses on its first uncached position
    assert ctx.cache_get([(0, 0), (9, 9)]) is None
    # an extension misses (its last position was never analysed)
    assert ctx.cache_get(long_moves + [(5, 5)]) is None


def test_cache_warm_returns_longest_cached_head():
    ctx = _real_ctx()
    long_moves = [(0, 0), (1, 0), (2, 0), (0, 1)]
    ctx.cache_put(long_moves, _full_result(long_moves))
    # extending a cached line warm-starts from its whole cached head
    warm = ctx.cache_warm(long_moves + [(5, 5)])
    assert warm is not None and len(warm) == 4
    # a divergent line still warms from the shared head (here just the seed)
    assert len(ctx.cache_warm([(0, 0), (9, 9)])) == 1
    # an empty cache warms nothing
    assert _real_ctx().cache_warm([(0, 0), (1, 0)]) is None


def test_cache_keyed_by_sims():
    ctx = _real_ctx(sims=16)
    ctx.cache_put([(0, 0), (1, 0), (2, 0)], _full_result([(0, 0), (1, 0), (2, 0)], sims=16))
    ctx.mcts_sims = 64  # a different search depth must not reuse the 16-sim entries
    assert ctx.cache_get([(0, 0), (1, 0)]) is None
    assert ctx.cache_warm([(0, 0), (1, 0), (2, 0), (0, 1)]) is None


def test_cache_put_stores_only_evaluated_positions():
    # A partial run: positions past the cut lack q_hat -> not stored. The
    # completed head is still reusable; the full line remains a miss.
    ctx = _real_ctx()
    moves = [(0, 0), (1, 0), (2, 0), (0, 1)]
    res = _full_result(moves)
    for e in res["trajectory"][2:]:   # simulate cancellation after 2 positions
        e.pop("q_hat", None)
    ctx.cache_put(moves, res)
    assert ctx.cache_get(moves) is None                       # tail absent
    assert len(ctx.cache_get([(0, 0), (1, 0)])["trajectory"]) == 2   # head served
    warm = ctx.cache_warm(moves)
    assert warm is not None and len(warm) == 2                # head warms a re-run


def _ctx_db(db_path, model_id="m1", sims=16, m=16):
    return AnalyzeContext(object(), object(), InferenceGuard(),
                          win_length=6, placement_radius=8, max_moves=400,
                          mcts_sims=sims, m_actions=m,
                          model_id=model_id, cache_db_path=str(db_path))


def test_position_store_roundtrip(tmp_path):
    st = _PositionStore(str(tmp_path / "c.sqlite"), "m1", 16, 16)
    pt = ((0, 0), (1, 0))
    st.put_many([(pt, {"q_hat": [0.5], "current_player": "P1"})])
    assert st.get(pt)["current_player"] == "P1"
    assert st.get(((0, 0), (9, 9))) is None
    assert any(k == pt for k, _ in st.load_recent(10))


def test_cache_persists_across_restart(tmp_path):
    db = tmp_path / "a.analysis.sqlite"
    moves = [(0, 0), (1, 0), (2, 0), (0, 1)]
    _ctx_db(db).cache_put(moves, _full_result(moves))
    # A fresh instance (simulated restart) with the same db + model warms from disk.
    ctx2 = _ctx_db(db)
    got = ctx2.cache_get(moves)
    assert got is not None and got["evaluated_prefixes"] == 4
    assert len(ctx2.cache_get([(0, 0), (1, 0)])["trajectory"]) == 2   # prefix served too


def test_cache_disk_fallback_when_not_warmed(tmp_path):
    # Even if a position isn't warmed into memory, cache_get falls through to disk.
    db = tmp_path / "a.analysis.sqlite"
    moves = [(0, 0), (1, 0), (2, 0)]
    _ctx_db(db).cache_put(moves, _full_result(moves))
    ctx2 = _ctx_db(db)
    ctx2._cache.clear()                         # drop the memory warm
    assert ctx2.cache_get(moves) is not None    # served from disk


def test_cache_not_reused_across_models(tmp_path):
    db = tmp_path / "a.analysis.sqlite"
    moves = [(0, 0), (1, 0), (2, 0)]
    _ctx_db(db, model_id="m1").cache_put(moves, _full_result(moves))
    # A different model id must not reuse m1's entries (keyed by model) -> miss.
    assert _ctx_db(db, model_id="m2").cache_get(moves) is None


def test_cache_keeps_other_model_rows_for_switching(tmp_path):
    db = tmp_path / "a.analysis.sqlite"
    moves = [(0, 0), (1, 0), (2, 0)]
    _ctx_db(db, model_id="m1").cache_put(moves, _full_result(moves))
    _ctx_db(db, model_id="m2")                       # opening as m2 must NOT drop m1's rows
    got = _ctx_db(db, model_id="m1").cache_get(moves)  # switching back still hits
    assert got is not None and got["evaluated_prefixes"] == 3


def test_analyze_game_prefix_served_from_cache_over_http(monkeypatch):
    # End-to-end: a cached longer line serves a shorter prefix request through
    # the SSE handler without ever invoking analyze_game_full.
    def must_not_run(*a, **k):
        raise AssertionError("ran analysis on a prefix cache hit")

    monkeypatch.setattr(app, "analyze_game_full", must_not_run)
    ctx = _real_ctx()
    long_moves = [(0, 0), (1, 0), (2, 0), (0, 1)]
    ctx.cache_put(long_moves, _full_result(long_moves))
    with _server(ctx) as port:
        status, ctype, raw = _read_sse(port, body=b'{"moves":[[0,0],[1,0]]}')
        assert status == 200
        assert "event: result" in raw


def test_analyze_game_disabled_without_ctx():
    with _server(analyze_ctx=None) as port:
        body = b'{"moves":[[0,0],[1,0]]}'
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/analyze_game", body=body,
                     headers={"Content-Type": "application/json",
                              "Content-Length": str(len(body))})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 503
        conn.close()


def test_analyze_game_empty_moves_400(monkeypatch):
    monkeypatch.setattr(app, "analyze_game_full", _fake_analyze_game_full())
    with _server(_fake_ctx()) as port:
        status, ctype, raw = _read_sse(port, body=b'{"moves":[]}')
        assert status == 400


def test_analyze_game_rate_limited(monkeypatch):
    monkeypatch.setattr(app, "analyze_game_full", _fake_analyze_game_full())
    from hexo_a0.serving.app import RateLimiter
    limiter = RateLimiter(1)  # 1/min
    with _server(_fake_ctx(), analyze_game_limiter=limiter) as port:
        s1, _, _ = _read_sse(port)
        assert s1 == 200
        s2, _, _ = _read_sse(port)
        assert s2 == 503  # over the per-minute cap


def test_analyze_game_queues_when_busy(monkeypatch):
    # A job that blocks until released; a second concurrent request must get
    # `queued` frames while the first holds the job lock.
    release = threading.Event()

    def blocking_fake(model, mc, moves, win, radius, mx, *, mcts_sims=64,
                      mcts_m_actions=16, progress_cb=None, cancel_cb=None, run_mcts_fn=None, warm_trajectory=None):
        release.wait(timeout=5)
        return {"trajectory": [], "boundary_indices": [], "evaluated_prefixes": 1,
                "mcts_sims": mcts_sims, "cancelled": False, "completed_prefixes": 1}

    monkeypatch.setattr(app, "analyze_game_full", blocking_fake)
    ctx = _fake_ctx()
    with _server(ctx) as port:
        out = {}

        def first():
            out["first"] = _read_sse(port, timeout=10)

        t1 = threading.Thread(target=first)
        t1.start()
        time.sleep(0.5)  # let the first acquire the job lock
        # Second request: read a couple of frames, expect `queued`.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        body = b'{"moves":[[0,0],[1,0]]}'
        conn.request("POST", "/analyze_game", body=body,
                     headers={"Content-Type": "application/json",
                              "Content-Length": str(len(body))})
        resp = conn.getresponse()
        # Read a bit of the stream (queued frames flush every ~2s; the fake holds
        # the lock until released).
        resp.fp.readline()  # status line already consumed; read first frames
        chunk = b""
        deadline = time.time() + 4
        while b"event: queued" not in chunk and time.time() < deadline:
            try:
                line = resp.fp.readline()
            except Exception:
                break
            if not line:
                break
            chunk += line
        assert b"event: queued" in chunk
        release.set()
        conn.close()
        t1.join(timeout=10)


def test_analyze_game_cancels_on_disconnect(monkeypatch):
    # When the client disconnects mid-job, cancel_cb must flip True so the job
    # stops early. We detect this via a flag the fake sets when cancelled.
    cancelled_seen = {"flag": False}
    started = threading.Event()

    def long_fake(model, mc, moves, win, radius, mx, *, mcts_sims=64,
                  mcts_m_actions=16, progress_cb=None, cancel_cb=None, run_mcts_fn=None, warm_trajectory=None):
        started.set()
        for i in range(1000):
            if cancel_cb and cancel_cb():
                cancelled_seen["flag"] = True
                return {"trajectory": [], "boundary_indices": [], "evaluated_prefixes": 1000,
                        "mcts_sims": mcts_sims, "cancelled": True, "completed_prefixes": i}
            if progress_cb:
                progress_cb(i + 1, 1000)
            time.sleep(0.01)
        return {"trajectory": [], "boundary_indices": [], "evaluated_prefixes": 1000,
                "mcts_sims": mcts_sims, "cancelled": False, "completed_prefixes": 1000}

    monkeypatch.setattr(app, "analyze_game_full", long_fake)
    with _server(_fake_ctx()) as port:
        # Open a raw socket, send the request, read a frame, then slam it shut.
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        body = b'{"moves":[[0,0],[1,0],[2,0]]}'
        req = (b"POST /analyze_game HTTP/1.1\r\nHost: x\r\n"
               b"Content-Type: application/json\r\n"
               b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        s.sendall(req)
        assert started.wait(timeout=5)
        time.sleep(0.2)  # let a few progress frames flush
        s.close()  # disconnect
        # The job loop writes the next progress frame, hits BrokenPipe, cancels.
        deadline = time.time() + 5
        while not cancelled_seen["flag"] and time.time() < deadline:
            time.sleep(0.05)
        assert cancelled_seen["flag"]
