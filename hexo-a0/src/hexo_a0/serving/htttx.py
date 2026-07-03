"""HTTTX move-log codec.

HTTTX is the text format for a HeXO game: a version header followed by one line
per turn, each turn being one or two `[q,r]` placements. The engine-seeded
origin stone (move_log[0]) is omitted from the serialized form.

HTTTX text uses explore.htttx.io's coordinate convention, which is a mirror of
our internal axial frame: internal (q, r) <-> HTTTX (q + r, -r). The map is an
involution, so serialize and parse apply the same transform.
"""
import re

_HTTTX_COORD_RE = re.compile(r"\[(-?\d+),(-?\d+)\]")


def mirror_axial(q: int, r: int) -> tuple[int, int]:
    """Reflect between our internal axial frame and htttx.io's (self-inverse)."""
    return q + r, -r


def serialize_htttx(move_log: list[tuple[int, int, str]]) -> str:
    """Convert a move log to HTTTX text. Skips the engine-seeded (0,0) stone."""
    body = "version[1];\n"
    if len(move_log) <= 1:
        return body
    # Pair up moves after the engine seed: (1,2), (3,4), ...
    rest = [mirror_axial(m[0], m[1]) for m in move_log[1:]]
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
    """Parse HTTTX text into internal (q, r) coords (engine seed not included)."""
    return [
        mirror_axial(int(m.group(1)), int(m.group(2)))
        for m in _HTTTX_COORD_RE.finditer(text)
    ]
