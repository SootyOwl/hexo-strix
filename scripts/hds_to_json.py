#!/usr/bin/env python3
"""Thin Python wrapper around the Rust `prove` deep-prover CLI.

The prover binary is pure file I/O with no network access (design decision: no
Rust-side network I/O). This wrapper covers the two boundaries the binary
deliberately leaves out:

  1. `fetch`  — pull an HDS sandbox position (a live URL or a 7-char code, or a
                local HDS JSON fixture) and write it in the prover's own JSON
                formats: a position snapshot and/or an ordered line log.
  2. render   — take a prover report (`--out report.json`) and turn its PV into a
                shareable ``hexo.tyto.cc/analysis#c=`` URL, so a proof can be
                eyeballed on the analysis board.

It reuses the tested converters rather than re-deriving them:
  * `hexo_a0.serving.hds` for the HDS fetch + (x,y)->(q,r) / colour-by-move-order
    conversion, and
  * `encode_moves_compact` / `load_hds` / `load_puzzle` from
    `scripts/forcing_search_prototype.py` for the `#c=` analysis-URL codec.

Run under the project environment so `hexo_a0` is importable, e.g.:

    uv run --no-sync python scripts/hds_to_json.py fetch 0l4291i --position pos.json
    uv run --no-sync python scripts/hds_to_json.py render report.json --code 0l4291i
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make sibling `forcing_search_prototype` importable regardless of cwd; it pulls
# in `hexo_a0.serving.hds` lazily, so this stays cheap.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASE_URL = "https://hexo.tyto.cc/analysis#c="


def _load(source: str | None, code: str | None):
    """Return (board, attacker, placements, move_log) from a local HDS JSON file
    (`source`) or a live HDS code/URL (`code`), reusing the prototype helpers."""
    from forcing_search_prototype import load_hds, load_puzzle

    if source:
        return load_puzzle(source)
    if code:
        return load_hds(code)
    raise SystemExit("need an HDS source: a --file HDS.json or a code/URL")


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"wrote {path}", file=sys.stderr)


def cmd_fetch(args: argparse.Namespace) -> int:
    _board, attacker, placements, move_log = _load(args.file, args.source)
    if not (args.position or args.line):
        raise SystemExit("fetch: give --position and/or --line")
    if args.position:
        # Prover position format: stones as a set of [q, r, "P1"|"P2"].
        pos = {
            "stones": [[q, r, p] for (q, r, p) in move_log],
            "attacker": attacker,
            "placements_remaining": placements,
        }
        _write_json(args.position, pos)
    if args.line:
        # Prover line format: the ordered game log from the seed.
        _write_json(args.line, {"moves": [[q, r, p] for (q, r, p) in move_log]})
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from forcing_search_prototype import encode_moves_compact

    with open(args.report) as f:
        report = json.load(f)
    pv = report.get("pv", [])
    if not pv:
        raise SystemExit(f"render: report {args.report!r} has no `pv` to render")

    # The ordered position prefix (seed .. current stones, in play order) comes
    # from the HDS source; the report's PV is the continuation the prover found.
    _board, _attacker, _placements, move_log = _load(args.file, args.code)
    prefix = [(q, r) for (q, r, _p) in move_log]
    full = prefix + [(q, r) for (q, r) in pv]
    url = args.base_url + encode_moves_compact(full)
    if args.out:
        with open(args.out, "w") as f:
            f.write(url + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    print(url)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="HDS sandbox -> prover position/line JSON")
    f.add_argument("source", nargs="?", help="HDS sandbox code or URL")
    f.add_argument("--file", help="local HDS sandbox JSON fixture instead of the network")
    f.add_argument("--position", help="write the position snapshot JSON here")
    f.add_argument("--line", help="write the ordered line-log JSON here")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("render", help="prover report PV -> analysis-board URL")
    r.add_argument("report", help="a prover report JSON (from prove --out)")
    r.add_argument("--code", help="HDS code/URL giving the ordered position prefix")
    r.add_argument("--file", help="local HDS sandbox JSON fixture (ordered prefix)")
    r.add_argument("--base-url", default=DEFAULT_BASE_URL, help="analysis URL prefix")
    r.add_argument("--out", help="also write the URL to this file")
    r.set_defaults(func=cmd_render)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
