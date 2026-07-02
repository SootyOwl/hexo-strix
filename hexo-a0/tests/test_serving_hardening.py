"""HTTP-level hardening tests for the public play server.

Exercises request parsing, status-code mapping, and security headers against a
real ``ThreadingHTTPServer`` backed by a no-op-bot ``GameManager`` (no model /
no GPU). These cover the adversarial-review findings that the existing
smoke tests missed: negative Content-Length, oversized bodies, malformed JSON,
missing payload keys, error-message leakage, Elo validation, and headers.
"""
import http.client
import json
import socket
import threading
import time
from contextlib import contextmanager

import pytest

from hexo_a0.serving.game import GameManager
from hexo_a0.serving.app import make_handler_class, MAX_BODY_BYTES


def _noop_mgr():
    return GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=lambda rec: None,      # no model: bot never moves
        recorder=None,
        mcts_sims=64,
        m_actions=16,
        checkpoint_path="x.pt",
        model_label="test",
        difficulty_sims={"standard": 64},
        default_difficulty="standard",
        idle_ttl_seconds=3600,
        max_games=10,
    )


@contextmanager
def _server(mgr=None, analyze_ctx=None, max_body_bytes=MAX_BODY_BYTES, admin_token=""):
    mgr = mgr or _noop_mgr()
    Handler = make_handler_class(mgr, admin_token=admin_token, url_prefix="",
                                 analyze_ctx=analyze_ctx,
                                 max_body_bytes=max_body_bytes)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()


def _post(port, body_bytes, content_length=None, headers=None):
    """POST /move with full control over Content-Length. Returns (status, body_dict_or_raw)."""
    if content_length is None:
        content_length = len(body_bytes)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    conn.request("POST", "/move", body=body_bytes, headers={**hdrs, "Content-Length": str(content_length)})
    resp = conn.getresponse()
    raw = resp.read()
    status = resp.status
    resp_headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    try:
        return status, json.loads(raw), resp_headers
    except Exception:
        return status, raw, resp_headers


def test_negative_content_length_rejected_not_hung():
    # Content-Length: -1 must NOT cause an unbounded read (bypass of body cap).
    with _server() as port:
        status, body, _ = _post(port, b"{}", content_length=-1)
        assert status == 400
        assert isinstance(body, dict) and "error" in body


def test_oversized_body_returns_413_not_500():
    payload = b'{"game_id":"x"}'  # 15 bytes
    with _server(max_body_bytes=8) as port:  # cap below payload size
        status, body, _ = _post(port, payload)
        assert status == 413
        assert isinstance(body, dict) and "error" in body


def test_malformed_json_returns_400_not_500():
    with _server() as port:
        status, body, _ = _post(port, b"not-json-at-all")
        assert status == 400


def test_move_missing_keys_returns_400_not_500():
    with _server() as port:
        status, body, _ = _post(port, b"{}")
        assert status == 400
        assert "error" in body


def test_move_bad_coord_type_returns_400_not_500():
    with _server() as port:
        status, body, _ = _post(port, b'{"game_id":"nope","q":"x","r":0}')
        assert status == 400


def test_resign_missing_game_id_returns_400_not_500():
    conn_pool = []

    def _post_path(port, path, body_bytes):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", path, body=body_bytes,
                      headers={"Content-Type": "application/json",
                               "Content-Length": str(len(body_bytes))})
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
        conn.close()
        try:
            return status, json.loads(raw)
        except Exception:
            return status, raw

    with _server() as port:
        status, body = _post_path(port, "/resign", b"{}")
        assert status == 400
        assert isinstance(body, dict) and "error" in body


def test_new_game_rejects_invalid_elo():
    with _server() as port:
        for bad in ('NaN', '-5', '99999', 'Infinity'):
            body_bytes = json.dumps({"self_reported_elo": float(bad) if bad in ('NaN', 'Infinity') else float(bad)}).encode()
            # json.dumps emits NaN/Infinity literally; emulate by hand for those:
            if bad == 'NaN':
                body_bytes = b'{"self_reported_elo": NaN}'
            elif bad == 'Infinity':
                body_bytes = b'{"self_reported_elo": Infinity}'
            elif bad == '-5':
                body_bytes = b'{"self_reported_elo": -5}'
            else:
                body_bytes = b'{"self_reported_elo": 99999}'
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("POST", "/new_game", body=body_bytes,
                          headers={"Content-Type": "application/json",
                                   "Content-Length": str(len(body_bytes))})
            resp = conn.getresponse()
            raw = resp.read()
            status = resp.status
            conn.close()
            assert status == 400, f"elo={bad} should be 400, got {status}: {raw!r}"


def test_500_does_not_leak_internal_exception():
    # Force a RuntimeError inside the manager; the client must NOT see its message.
    mgr = _noop_mgr()

    def boom(game_id, q, r):
        raise RuntimeError("secret internal detail: db/password=hunter2")

    mgr.apply_human_move = boom
    with _server(mgr=mgr) as port:
        status, body, _ = _post(port, b'{"game_id":"x","q":1,"r":0}')
        assert status == 500
        assert "hunter2" not in json.dumps(body)
        assert "secret" not in json.dumps(body)


def test_security_headers_present_on_html():
    with _server() as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        conn.close()
        assert hdrs.get("x-content-type-options") == "nosniff"
        assert "x-frame-options" in hdrs
        assert "referrer-policy" in hdrs


def test_security_headers_present_on_json():
    with _server() as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/stats")
        resp = conn.getresponse()
        resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        conn.close()
        assert hdrs.get("x-content-type-options") == "nosniff"


def test_unknown_game_move_returns_404_not_500():
    with _server() as port:
        status, body, _ = _post(port, b'{"game_id":"does-not-exist","q":1,"r":0}')
        assert status == 404


def _post_json(port, path, payload):
    data = json.dumps(payload).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=data,
                 headers={"Content-Type": "application/json", "Content-Length": str(len(data))})
    resp = conn.getresponse()
    raw = resp.read()
    status = resp.status
    conn.close()
    try:
        return status, json.loads(raw)
    except Exception:
        return status, raw


def test_valid_move_returns_200_over_http():
    # Regression guard: a VALID /move must return 200 with the updated state.
    # A prior edit left a stale duplicate apply_human_move(gid, q, r) call after
    # the renamed q_raw/r_raw call — the second line raised NameError on every
    # valid move (illegal-move tests missed it because they raise on the first
    # call). This test exercises the happy path over real HTTP.
    with _server() as port:
        status, ng = _post_json(port, "/new_game", {"human_side": "P2", "name": "t"})
        assert status == 200 and "game_id" in ng
        gid = ng["game_id"]
        # (1,0) is within placement_radius=8 of the origin seed -> legal for P2.
        status, mb = _post_json(port, "/move", {"game_id": gid, "q": 1, "r": 0})
        assert status == 200, f"valid move should be 200, got {status}: {mb!r}"
        assert isinstance(mb, dict) and "state" in mb
        assert any(s[0] == [1, 0] for s in mb["state"]["stones"])


def test_normalize_moves_typeerror_returns_400():
    # [[null,0]] -> int(None) raises TypeError; must surface as 400, not 500.
    from hexo_a0.serving.app import handle_analyze
    status, body = handle_analyze({"moves": [[None, 0]]}, ctx=None)
    assert status == 400
    assert "error" in body


def test_normalize_moves_non_list_returns_400():
    from hexo_a0.serving.app import handle_analyze
    status, body = handle_analyze({"moves": 5}, ctx=None)
    assert status == 400


def test_huge_coordinates_rejected_before_engine():
    # A 10**18 coordinate must be rejected at the trust boundary, not handed to
    # the Rust engine (which could allocate/panic indexing by it).
    from hexo_a0.serving.app import handle_analyze
    status, body = handle_analyze({"moves": [[10**18, 0]]}, ctx=None)
    assert status == 400
    assert "bound" in body["error"]


def test_admin_token_correct_returns_200_wrong_returns_404():
    with _server(admin_token="secret-token") as port:
        for token, expected in (("wrong", 404), ("secret-token", 200)):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", f"/admin?token={token}")
            resp = conn.getresponse()
            resp.read()
            assert resp.status == expected, f"token={token!r} -> {resp.status}"
            conn.close()


def test_admin_token_via_header():
    # The token may be supplied via X-Admin-Token header (preferred — keeps it
    # out of proxy access logs / browser history) in addition to ?token=.
    with _server(admin_token="secret-token") as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/admin", headers={"X-Admin-Token": "secret-token"})
        resp = conn.getresponse(); resp.read()
        assert resp.status == 200
        conn.close()
        # wrong header token -> 404
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/admin", headers={"X-Admin-Token": "nope"})
        resp = conn.getresponse(); resp.read()
        assert resp.status == 404
        conn.close()


def test_rate_limiter_caps_bursts():
    from hexo_a0.serving.app import RateLimiter
    rl = RateLimiter(3)
    assert rl.allow("1.2.3.4") is True
    assert rl.allow("1.2.3.4") is True
    assert rl.allow("1.2.3.4") is True
    assert rl.allow("1.2.3.4") is False  # over the per-minute cap
    # a different key has its own bucket
    assert rl.allow("9.9.9.9") is True


def test_rate_limiter_state_is_bounded():
    # The per-key state dict must be capped so a flood of distinct keys can't
    # grow it unbounded (memory-DoS backstop).
    from hexo_a0.serving.app import RateLimiter
    rl = RateLimiter(60, max_keys=4)
    for i in range(20):
        rl.allow(f"10.0.0.{i}")
    assert len(rl._state) <= rl.max_keys


def test_csrf_rejects_empty_body_with_wrong_content_type():
    # An empty-body POST with a non-JSON Content-Type must be rejected — the
    # Content-Type check must run before the Content-Length:0 short-circuit,
    # or a cross-origin <form method=POST> (no fields) bypasses the CSRF gate.
    with _server() as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/new_game", body=b"",
                     headers={"Content-Type": "application/x-www-form-urlencoded",
                              "Content-Length": "0"})
        resp = conn.getresponse(); resp.read()
        assert resp.status == 400
        conn.close()


def test_normalize_moves_rejects_float_and_bool():
    # /analyze must reject non-integer floats and bools, not silently truncate.
    from hexo_a0.serving.app import handle_analyze
    for bad in ([[1.5, 0]], [[True, 0]], [[0, False]]):
        status, body = handle_analyze({"moves": bad}, ctx=None)
        assert status == 400, f"{bad} should be 400, got {status}: {body!r}"


def test_cache_control_no_store_on_responses():
    with _server() as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        conn.close()
        assert "no-store" in hdrs.get("cache-control", "")


def test_inference_busy_returns_503():
    # When the shared guard is held (bot thinking), analysis non-blocking → 503.
    from types import SimpleNamespace
    import threading as _t
    from hexo_a0.serving.app import handle_analyze
    from hexo_a0.serving.inference import InferenceGuard

    g = InferenceGuard()
    held = _t.Event()
    release = _t.Event()

    def hog():
        held.set()
        release.wait()

    th = _t.Thread(target=lambda: g.run(hog))
    th.start()
    held.wait()
    try:
        ctx = SimpleNamespace(guard=g, max_analyze_moves=220)
        status, body = handle_analyze({"moves": [[0, 0], [1, 0]]}, ctx=ctx)
        assert status == 503
        assert "busy" in body["error"]
    finally:
        release.set()
        th.join()


def test_csrf_rejects_non_json_content_type():
    # A cross-origin form POST (Content-Type: text/plain) must be rejected so a
    # CSRF "simple request" can't drive state-changing endpoints.
    body = b'{"game_id":"x","q":1,"r":0}'
    with _server() as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/move", body=body,
                     headers={"Content-Type": "text/plain",
                              "Content-Length": str(len(body))})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 400
        conn.close()


def test_chunked_transfer_encoding_rejected():
    # stdlib http.server doesn't decode chunked bodies; reject the header to
    # avoid request-smuggling/desync via an unread chunked body.
    with _server() as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/move", body=b'{}',
                     headers={"Transfer-Encoding": "chunked",
                              "Content-Length": "2",
                              "Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 400
        conn.close()


def test_analyze_disabled_when_no_ctx():
    # If analyze_ctx is None (handler built without one), /analyze must return
    # 503, not a 500 from dereferencing None.
    body = b'{"moves":[[0,0],[1,0]]}'
    with _server(analyze_ctx=None) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/analyze", body=body,
                     headers={"Content-Type": "application/json",
                              "Content-Length": str(len(body))})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 503
        conn.close()


def test_load_model_rejects_non_checkpoint(tmp_path):
    # A file that loads under weights_only=True but isn't a HeXONet checkpoint
    # (no model_state_dict) must raise a clear ValueError, not a raw KeyError.
    import torch as _t
    from hexo_a0.serving.model import load_model, _MODEL_CACHE
    junk = tmp_path / "junk.pt"
    _t.save({"foo": "bar"}, str(junk))
    _MODEL_CACHE.clear()
    try:
        with pytest.raises(ValueError):
            load_model(str(junk), "cpu")
    finally:
        _MODEL_CACHE.clear()


def test_model_config_fallback_raises_on_malformed_toml(tmp_path):
    # A malformed --config must crash loudly (matching the frozen script), not
    # be swallowed into a silent wrong-architecture default.
    import tomllib
    from types import SimpleNamespace
    from hexo_a0.serving.app import _model_config_fallback
    p = tmp_path / "bad.toml"
    p.write_text("this is = = not valid toml [[[")
    with pytest.raises(tomllib.TOMLDecodeError):
        _model_config_fallback(SimpleNamespace(config=str(p)))