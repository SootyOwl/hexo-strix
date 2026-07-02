"""HTTTX move-log codec (ported verbatim from scripts/play_server.py:51-77).

HTTTX is the text format for a HeXO game: a version header followed by one line
per turn, each turn being one or two `[q,r]` placements. The engine-seeded
origin stone (move_log[0]) is omitted from the serialized form.
"""
import re

_HTTTX_COORD_RE = re.compile(r"\[(-?\d+),(-?\d+)\]")


def serialize_htttx(move_log: list[tuple[int, int, str]]) -> str:
    """Convert a move log to HTTTX text. Skips the engine-seeded (0,0) stone."""
    body = "version[1];\n"
    if len(move_log) <= 1:
        return body
    # Pair up moves after the engine seed: (1,2), (3,4), ...
    rest = move_log[1:]
    turn = 1
    i = 0
    while i < len(rest):
        a = rest[i]
        b = rest[i + 1] if i + 1 < len(rest) else None
        if b is None:
            body += f"{turn}. [{a[0]},{a[1]}];\n"
        else:
            body += f"{turn}. [{a[0]},{a[1]}][{b[0]},{b[1]}];\n"
        turn += 1
        i += 2
    return body


def parse_htttx(text: str) -> list[tuple[int, int]]:
    """Parse HTTTX text into a flat list of (q, r) coords (engine seed not included)."""
    return [(int(m.group(1)), int(m.group(2))) for m in _HTTTX_COORD_RE.finditer(text)]
