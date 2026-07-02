"""HDS sandbox -> HTTTX converter.

hexo.did.science stores a shared "sandbox" position behind a random 7-char
key (e.g. ``5knldz6``), resolvable via its public, no-auth REST endpoint
``GET /api/sandbox-positions/<id>``. This module fetches such a position and
converts it to this project's HTTTX move-log notation.

HDS axial coordinates ``(x, y)`` map by identity onto HeXO ``(q, r)``: their
win axes are ``[1,0], [0,1], [1,-1]`` — identical to HeXO's. The board is
translation-invariant, so a position whose opening stone is not at the origin
is shifted so the first move sits at ``(0, 0)``.
"""
import json
import re
import urllib.error
import urllib.request

from .htttx import serialize_htttx

_API_BASE = "https://hexo.did.science/api/sandbox-positions/"
_CODE_RE = re.compile(r"^[a-z0-9]{7}$", re.IGNORECASE)
_SANDBOX_MARKER = "/sandbox/"
_PLAYER_MAP = {"player-1": "P1", "player-2": "P2"}
# A sandbox position is tiny JSON; cap the body we read so a compromised /
# misbehaving upstream cannot exhaust memory.
_MAX_RESPONSE_BYTES = 5_000_000
# hexo.did.science sits behind Cloudflare, which rejects the stdlib default
# "Python-urllib/x.y" User-Agent with a 403 (error 1010). Any other UA is
# accepted, so we send an honest one that identifies this tool rather than
# impersonating a browser.
_USER_AGENT = "hexo-a0-hds-converter/1.0 (HeXO sandbox->HTTTX importer)"


class HdsError(Exception):
    """Base for all HDS-conversion errors."""
    http_status = 502


class InvalidSandboxCode(HdsError):
    """The supplied URL/code is not a valid sandbox code. -> HTTP 400."""
    http_status = 400


class SandboxNotFound(HdsError):
    """hexo.did.science has no position for this code. -> HTTP 404."""
    http_status = 404


class HdsUnreachable(HdsError):
    """Could not reach hexo.did.science (timeout / connection). -> HTTP 502."""
    http_status = 502


class HdsBadResponse(HdsError):
    """hexo.did.science returned an unexpected / malformed response. -> 502."""
    http_status = 502


class IllegalPosition(HdsError):
    """The position is not a legal HeXO game. -> HTTP 422."""
    http_status = 422


def extract_sandbox_code(url_or_code: str) -> str:
    """Extract and validate the 7-char sandbox code from a URL or bare code."""
    s = (url_or_code or "").strip()
    s = s.split("?", 1)[0].split("#", 1)[0]
    if _SANDBOX_MARKER in s:
        s = s.split(_SANDBOX_MARKER, 1)[1].split("/", 1)[0]
    elif "/" in s:
        s = s.rstrip("/").rsplit("/", 1)[-1]
    if not _CODE_RE.match(s):
        raise InvalidSandboxCode(f"invalid sandbox code: {url_or_code!r}")
    return s.lower()


def _default_fetch(url: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SandboxNotFound("sandbox position not found") from e
        raise HdsBadResponse(f"HTTP {e.code} from hexo.did.science") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise HdsUnreachable(f"could not reach hexo.did.science: {e}") from e
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise HdsBadResponse("response from hexo.did.science too large")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise HdsBadResponse("non-JSON response from hexo.did.science") from e


def fetch_sandbox_position(code: str, *, timeout: float = 10.0, fetch=None) -> dict:
    """GET the sandbox position for ``code``. ``fetch`` is an injectable
    ``callable(url, timeout) -> dict`` (defaults to a stdlib urllib GET)."""
    url = _API_BASE + code
    fetcher = fetch if fetch is not None else _default_fetch
    try:
        resp = fetcher(url, timeout)
    except HdsError:
        raise
    except Exception as e:  # an injected fetch may raise anything else
        raise HdsUnreachable(f"could not reach hexo.did.science: {e}") from e
    game_position = resp.get("gamePosition") if isinstance(resp, dict) else None
    if not isinstance(game_position, dict) or not isinstance(game_position.get("cells"), list):
        raise HdsBadResponse("unexpected response shape from hexo.did.science")
    return resp


def sandbox_to_move_log(game_position: dict):
    """Convert an HDS ``gamePosition`` to a HeXO move log + metadata.

    Returns ``(move_log, meta)`` where ``move_log`` is a list of
    ``(q, r, "P1"|"P2")`` in move order with the opening stone centered at the
    origin, and ``meta`` carries ``offset_applied``, ``current_turn``,
    ``placements_remaining`` and ``move_count``. Raises ``IllegalPosition`` if
    the cells do not form a legal HeXO game.
    """
    cells = game_position.get("cells") if isinstance(game_position, dict) else None
    if not isinstance(cells, list) or not cells:
        raise IllegalPosition("position has no cells")
    for c in cells:
        if not isinstance(c, dict) or not all(k in c for k in ("x", "y", "player", "moveId")):
            raise HdsBadResponse("cell missing required field")
        # bool is an int subclass; reject it so True/False can't masquerade as a coord.
        if any(not isinstance(c[k], int) or isinstance(c[k], bool) for k in ("x", "y", "moveId")):
            raise HdsBadResponse("cell x/y/moveId must be integers")
        # isinstance guard first: an unhashable player (list/dict from malformed
        # JSON) would make the `in` membership test raise TypeError.
        if not isinstance(c["player"], str) or c["player"] not in _PLAYER_MAP:
            raise IllegalPosition(f"unknown player slot: {c['player']!r}")

    ordered = sorted(cells, key=lambda c: c["moveId"])
    for idx, c in enumerate(ordered, start=1):
        if c["moveId"] != idx:
            raise IllegalPosition("move ids are not contiguous 1..N")

    # Center the opening stone on the origin (board is translation-invariant).
    dx, dy = -ordered[0]["x"], -ordered[0]["y"]
    offset_applied = None if (dx == 0 and dy == 0) else (dx, dy)
    coords = [(c["x"] + dx, c["y"] + dy) for c in ordered]
    if len(set(coords)) != len(coords):
        raise IllegalPosition("duplicate stone coordinate")

    # Legal HeXO turn structure: a lone opening move, then strictly alternating
    # pairs (an optional trailing single = an incomplete turn). HDS lets either
    # colour open; HTTTX/HeXO always treat the opener as P1, so we relabel the
    # opener to P1 (a colour swap when HDS's player-2 started) rather than
    # rejecting it.
    opener = ordered[0]["player"]
    other = "player-2" if opener == "player-1" else "player-1"
    for i, c in enumerate(ordered, start=1):
        expected = opener if (i == 1 or ((i - 2) // 2) % 2 == 1) else other
        if c["player"] != expected:
            raise IllegalPosition("placements do not form a legal alternating-pairs sequence")

    # Move order, not the HDS colour slot, defines P1: opener -> P1, other -> P2.
    label = {opener: "P1", other: "P2"}
    move_log = [
        (coords[i][0], coords[i][1], label[ordered[i]["player"]])
        for i in range(len(ordered))
    ]
    # current_turn is lenient metadata; guard isinstance so an unhashable value
    # can't make the dict lookup raise TypeError.
    cur_player = game_position.get("currentTurnPlayer")
    meta = {
        "offset_applied": offset_applied,
        "current_turn": label.get(cur_player) if isinstance(cur_player, str) else None,
        "placements_remaining": game_position.get("placementsRemaining"),
        "move_count": len(move_log),
        "colors_swapped": opener == "player-2",
    }
    return move_log, meta


def hds_to_htttx(url_or_code: str, *, timeout: float = 10.0, fetch=None) -> dict:
    """Fetch an HDS sandbox position and convert it to HTTTX notation."""
    code = extract_sandbox_code(url_or_code)
    resp = fetch_sandbox_position(code, timeout=timeout, fetch=fetch)
    move_log, meta = sandbox_to_move_log(resp.get("gamePosition"))
    return {
        "code": code,
        "name": resp.get("name"),
        "htttx": serialize_htttx(move_log),
        "move_count": meta["move_count"],
        "current_turn": meta["current_turn"],
        "placements_remaining": meta["placements_remaining"],
        "offset_applied": meta["offset_applied"],
        "colors_swapped": meta["colors_swapped"],
    }
