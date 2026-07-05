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

Run under the project environment so `hexo_a0` is importable. The default
action is the ONE-STEP prove (fetch the sandbox, run the binary, render the
PV as an analysis URL); `fetch`/`render` remain available as explicit
subcommands for the individual steps:

    uv run --no-sync python scripts/prove.py 0l4291i -- --driver pdspn
    uv run --no-sync python scripts/prove.py 0l4291i --lines d9ci11d 7zoidcw -- --time-limit 7200
    uv run --no-sync python scripts/prove.py fetch 0l4291i --position pos.json
    uv run --no-sync python scripts/prove.py render report.json --code 0l4291i
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


def _default_prove_bin() -> str:
    """The prove binary at its scoped build location, relative to the repo root
    (this file lives in <root>/scripts/). Overridable via $PROVE_BIN."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get(
        "PROVE_BIN",
        os.path.join(root, "hexo-rs", "target-forcing", "release", "prove"))


def cmd_prove(args: argparse.Namespace) -> int:
    """One-step: fetch the sandbox, run the prover, render the PV URL."""
    import subprocess
    import tempfile

    prove_bin = args.prove_bin or _default_prove_bin()
    if not os.path.exists(prove_bin):
        raise SystemExit(
            f"prove binary not found at {prove_bin}\n"
            "build it first (scoped target dir — do NOT clobber target/):\n"
            "  cd hexo-rs && CARGO_TARGET_DIR=target-forcing "
            "cargo build --release -p hexo-mcts --bin prove")

    workdir = args.workdir or tempfile.mkdtemp(prefix="hexo-prove-")
    os.makedirs(workdir, exist_ok=True)

    def _slug(code: str) -> str:
        from hexo_a0.serving.hds import extract_sandbox_code
        try:
            return extract_sandbox_code(code)
        except Exception:
            return os.path.splitext(os.path.basename(code))[0]

    # Position (and its ordered prefix, needed later for the URL render).
    _b, attacker, placements, move_log = _load(None, args.source)
    slug = _slug(args.source)
    pos_path = os.path.join(workdir, f"{slug}.json")
    _write_json(pos_path, {
        "stones": [[q, r, p] for (q, r, p) in move_log],
        "attacker": attacker,
        "placements_remaining": placements,
    })

    # Optional composer lines (each its own sandbox code/URL).
    line_paths = []
    for lc in args.lines or []:
        _lb, _la, _lp, llog = _load(None, lc)
        lp = os.path.join(workdir, f"{_slug(lc)}_line.json")
        _write_json(lp, {"moves": [[q, r, p] for (q, r, p) in llog]})
        line_paths.append(lp)

    report_path = args.out or os.path.join(workdir, f"{slug}_report.json")
    cmd = [prove_bin, "--position", pos_path, "--out", report_path]
    if line_paths:
        cmd += ["--lines", *line_paths]
    cmd += args.prove_args  # passthrough: --driver, --time-limit, --width, ...
    print("+ " + " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd)  # streams the prover's own output live
    if rc != 0:
        return rc

    with open(report_path) as f:
        report = json.load(f)
    print(f"\nverdict: {report.get('verdict')}  (report: {report_path})")
    pv = report.get("pv") or []
    if pv:
        from forcing_search_prototype import encode_moves_compact
        prefix = [(q, r) for (q, r, _p) in move_log]
        url = args.base_url + encode_moves_compact(prefix + [(q, r) for q, r in pv])
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

    v = sub.add_parser(
        "prove",
        help="one-step: fetch sandbox, run the prover, render the PV URL")
    v.add_argument("source", help="HDS sandbox code or URL of the position")
    v.add_argument("--lines", nargs="*",
                   help="sandbox codes/URLs of composer mainbranch lines (hybrid)")
    v.add_argument("--workdir", help="keep the JSONs here (default: a temp dir)")
    v.add_argument("--out", help="prover report path (default: <workdir>/<code>_report.json)")
    v.add_argument("--prove-bin", help="prove binary (default: $PROVE_BIN or the "
                                       "target-forcing release build)")
    v.add_argument("--base-url", default=DEFAULT_BASE_URL, help="analysis URL prefix")
    # Flags for the prove binary itself go after a literal `--`
    # (e.g. `prove 0l4291i -- --driver pdspn --time-limit 600`); they're split
    # off in main() before argparse runs (argparse.REMAINDER greedily eats
    # this wrapper's own optionals, so it can't be used here).
    v.set_defaults(func=cmd_prove)

    r = sub.add_parser("render", help="prover report PV -> analysis-board URL")
    r.add_argument("report", help="a prover report JSON (from prove --out)")
    r.add_argument("--code", help="HDS code/URL giving the ordered position prefix")
    r.add_argument("--file", help="local HDS sandbox JSON fixture (ordered prefix)")
    r.add_argument("--base-url", default=DEFAULT_BASE_URL, help="analysis URL prefix")
    r.add_argument("--out", help="also write the URL to this file")
    r.set_defaults(func=cmd_render)

    if argv is None:
        argv = sys.argv[1:]
    # Everything after a literal `--` is passed through to the prove binary.
    passthrough: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]
    # One-step prove is the default action: if the first argument isn't a
    # known subcommand (or a help flag), treat the invocation as
    # `prove <sandbox> ...`.
    if argv and argv[0] not in {"fetch", "render", "prove", "-h", "--help"}:
        argv = ["prove", *argv]
    args = p.parse_args(argv)
    args.prove_args = passthrough
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
