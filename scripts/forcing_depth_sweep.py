"""Depth × node-budget sweep for the live-play forcing (VCF) solver tiers.

Motivation (2026-07-05): the `strong` tier lost real games because its
depth-10 forcing cap couldn't see the decisive line in time (see the
`strongloss_*_line.json` fixtures). This sweep answers two questions the
2026-07-04 r8 tuning bench couldn't (it only measured depth/budget PAIRED —
12/250k, 16/2M — never deep-with-small-budget):

  1. Must-solve: at which (depth, budget) does the eventual winner's forced
     win in each fixture game first become provable, and how many placements
     before the end is that? (Earlier = the bot at that tier would have
     executed it / started defending against it.)
  2. Latency guard: per-solve wall-clock p50/p99/max over every prefix of
     every fixture game, BOTH sides (live play solves mover-side for its own
     win and flipped for defense), per (depth, budget). The per-tier depth
     bump only ships if p99 stays comfortably inside live budgets
     (DEFENSE_DEADLINE_S = 3 s; current strong p99 is ~0.35 ms..350 ms).

Usage:
    uv run --no-sync python scripts/forcing_depth_sweep.py [--out report.json]
    (add --quick for a 2-config smoke run)

Findings go to docs/research/ — this script is the reproducible source.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "forcing_puzzles"
# Real games only (full move lines, live wl=6/r=8 config): the three losses
# this sweep exists for, plus the two long deep-puzzle lines for latency
# coverage of long games. hayes_20260712 is the LIVE_NODE_BUDGET 20k->50k
# counterexample (own win at prefix 30/31 first provable at 35k nodes).
LINE_FIXTURES = [
    "strongloss_a_line.json",
    "strongloss_b_line.json",
    "hayes_20260712_line.json",
    "7zoidcw_line.json",
    "d9ci11d_line.json",
]

DEPTHS = [10, 12, 14, 16, 18]
BUDGETS = [20_000, 50_000, 100_000, 250_000]
WIN_LENGTH, RADIUS, MAX_MOVES = 6, 8, 400


def load_lines():
    games = []
    for name in LINE_FIXTURES:
        path = FIXTURE_DIR / name
        moves = json.loads(path.read_text())["moves"]
        games.append((name.removesuffix("_line.json"), moves))
    return games


def replay_prefixes(hr, cfg, moves):
    """Yield (prefix_index, state, winner_so_far) for every non-terminal prefix.

    prefix_index i = position after moves[0..i] (moves[0] is the engine's own
    P1 seed, skipped on apply).
    """
    state = hr.GameState(cfg)
    prefixes = [(0, state.clone())]
    for i, (q, r, _owner) in enumerate(moves[1:], start=1):
        state.apply_move(q, r)
        if state.is_terminal():
            break
        prefixes.append((i, state.clone()))
    return prefixes, state


def flipped(hr, cfg, state, side):
    """State with `side` to move on the same stones (fresh 2-placement turn)."""
    return hr.GameState.from_state(state.placed_stones(), side, 2, cfg)


def solve_timed(hr, state, depth, budget):
    t0 = time.perf_counter()
    res = hr.solve_forcing(state, depth, budget)
    return res, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="2-config smoke run (depth 10/16 x budget 20k)")
    ap.add_argument("--map", dest="map_game", default=None,
                    help="per-prefix solve map for one game (fixture stem, "
                         "e.g. strongloss_a) instead of the grid sweep")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--budget", type=int, default=20_000)
    ap.add_argument("--defend", nargs=2, metavar=("GAME", "PREFIX"), default=None,
                    help="block-survival analysis at one prefix: re-solve the "
                         "opponent's VCF after every candidate block (PV cells "
                         "first — live play's candidate set — then ALL legal "
                         "moves) and report killers")
    ap.add_argument("--simulate", nargs=2, metavar=("GAME", "PREFIX"),
                    default=None,
                    help="replay game.py's _defensive_move greedy (killer "
                         "else longest-PV survivor, re-checked per placement) "
                         "for the mover's full turn at PREFIX; report whether "
                         "the opponent's VCF survives the turn")
    args = ap.parse_args()

    import hexo_rs as hr
    cfg = hr.GameConfig(WIN_LENGTH, RADIUS, MAX_MOVES)
    games = load_lines()

    if args.simulate:
        # game.py _defensive_move, minus the NN policy tie-break for killers
        # (first killer stands in for _policy_argmax) and minus the 3 s
        # deadline (we want the algorithm's verdict, not its timeout).
        game_name, prefix_target = args.simulate[0], int(args.simulate[1])
        moves = dict(games)[game_name]
        prefixes, _final = replay_prefixes(hr, cfg, moves)
        state = dict(prefixes)[prefix_target].clone()
        mover = state.current_player()
        opp = "P1" if mover == "P2" else "P2"
        print(f"{game_name}[{prefix_target}]: {mover} defends vs {opp} "
              f"(depth={args.depth}, budget={args.budget:,})")
        for placement in range(state.moves_remaining_this_turn()):
            res, _ = solve_timed(hr, flipped(hr, cfg, state, opp),
                                 args.depth, args.budget)
            if res is None:
                print(f"  placement {placement + 1}: no opp VCF visible — "
                      f"defense stands down (falls through to MCTS)")
                break
            _f, pv = res
            legal = {tuple(c) for c in state.legal_moves()}
            cands = []
            for c in pv:
                c = tuple(c)
                if c in legal and c not in cands:
                    cands.append(c)
            killers, survivors = [], []
            for c in cands:
                trial = state.clone()
                trial.apply_move(*c)
                if trial.is_terminal():
                    killers.append(c)  # terminal = bot win/draw, play it
                    continue
                re_res, _ = solve_timed(hr, flipped(hr, cfg, trial, opp),
                                        args.depth, args.budget)
                if re_res is None:
                    killers.append(c)
                else:
                    survivors.append((c, len(re_res[1])))
            if killers:
                pick, verdict = killers[0], "KILLER"
            elif survivors:
                pick, verdict = max(survivors, key=lambda t: t[1])[0], "survivor(max-delay)"
            else:
                print(f"  placement {placement + 1}: opp VCF pv={len(pv)} but "
                      f"no legal PV candidates — falls through to MCTS")
                break
            print(f"  placement {placement + 1}: opp VCF pv={len(pv)}, "
                  f"{len(cands)} candidates -> plays {pick} [{verdict}]")
            state.apply_move(*pick)
        final_check, _ = solve_timed(hr, flipped(hr, cfg, state, opp),
                                     args.depth, args.budget)
        print(f"  after turn: opp VCF "
              f"{'STILL PROVEN pv=' + str(len(final_check[1])) if final_check else 'KILLED — defense succeeded'}")
        return 0

    if args.defend:
        # Mirrors game.py's _defensive_move verification loop (block, flip,
        # re-solve) but exhaustively: PV-cell candidates (what live play
        # actually tries) vs. every legal move (what a perfect candidate
        # generator could try). Killers = blocks after which the opponent's
        # VCF is no longer provable. One placement deep — a surviving pair
        # of blocks would need the second placement's re-check.
        game_name, prefix_target = args.defend[0], int(args.defend[1])
        moves = dict(games)[game_name]
        prefixes, final = replay_prefixes(hr, cfg, moves)
        state = dict(prefixes)[prefix_target]
        mover = state.current_player()
        opp = "P1" if mover == "P2" else "P2"
        res, _dt = solve_timed(hr, flipped(hr, cfg, state, opp),
                               args.depth, args.budget)
        if res is None:
            print(f"no opponent VCF at prefix {prefix_target} "
                  f"(depth={args.depth}, budget={args.budget:,})")
            return 0
        _first, pv = res
        legal = {tuple(c) for c in state.legal_moves()}
        pv_cells = [c for c in (tuple(m) for m in pv) if c in legal]
        print(f"{game_name}[{prefix_target}]: {mover} to move, {opp} VCF "
              f"pv={len(pv)}; {len(pv_cells)} PV candidates, "
              f"{len(legal)} legal moves (depth={args.depth}, "
              f"budget={args.budget:,})")
        for label, cands in (("PV-cell", pv_cells),
                             ("all-legal", sorted(legal))):
            killers, survivors, t0 = [], 0, time.perf_counter()
            for c in cands:
                trial = state.clone()
                trial.apply_move(*c)
                if trial.is_terminal():
                    continue
                re_res, _ = solve_timed(hr, flipped(hr, cfg, trial, opp),
                                        args.depth, args.budget)
                if re_res is None:
                    killers.append(c)
                else:
                    survivors += 1
            print(f"  {label}: {len(killers)} killer(s), {survivors} "
                  f"surviving block(s), {time.perf_counter()-t0:.1f}s"
                  + (f" — killers: {killers[:8]}" if killers else ""))
        # Pair defense: the mover really places TWO stones before the
        # opponent moves (when rem=2) — try every unordered PV-cell pair,
        # then flip and re-solve. A killer pair means the game was still
        # saveable even though no single block killed the VCF.
        if state.moves_remaining_this_turn() == 2:
            pair_killers, tried, t0 = [], 0, time.perf_counter()
            for i in range(len(pv_cells)):
                for j in range(i + 1, len(pv_cells)):
                    trial = state.clone()
                    trial.apply_move(*pv_cells[i])
                    if trial.is_terminal():
                        continue
                    if tuple(pv_cells[j]) not in {tuple(c) for c in trial.legal_moves()}:
                        continue
                    trial.apply_move(*pv_cells[j])
                    if trial.is_terminal():
                        continue
                    tried += 1
                    re_res, _ = solve_timed(hr, flipped(hr, cfg, trial, opp),
                                            args.depth, args.budget)
                    if re_res is None:
                        pair_killers.append((pv_cells[i], pv_cells[j]))
            print(f"  PV-cell PAIRS: {len(pair_killers)} killer pair(s) of "
                  f"{tried} tried, {time.perf_counter()-t0:.1f}s"
                  + (f" — e.g. {pair_killers[:4]}" if pair_killers else ""))
        return 0

    if args.map_game:
        # Per-prefix map: who moves, what the mover-side solve proves (their
        # own win, the REAL mid-turn state), and what the flipped solve
        # proves (the opponent's threat, fresh-turn flip — what live play's
        # defensive sweep consults at the mover's turn). PV length shows how
        # deep the proven line is.
        moves = dict(games)[args.map_game]
        prefixes, final = replay_prefixes(hr, cfg, moves)
        print(f"{args.map_game}: winner={final.winner()}, depth={args.depth}, "
              f"budget={args.budget:,}")
        for idx, state in prefixes:
            mover = state.current_player()
            opp = "P1" if mover == "P2" else "P2"
            own, own_dt = solve_timed(hr, state, args.depth, args.budget)
            thr, thr_dt = solve_timed(hr, flipped(hr, cfg, state, opp),
                                      args.depth, args.budget)
            own_s = f"WIN pv={len(own[1])}" if own else "-"
            thr_s = f"THREAT pv={len(thr[1])}" if thr else "-"
            print(f"  [{idx:3d}] {mover} to move "
                  f"(rem={state.moves_remaining_this_turn()}): "
                  f"own={own_s:<12} opp-threat={thr_s:<15} "
                  f"({own_dt*1e3:.0f}/{thr_dt*1e3:.0f} ms)")
        return 0

    depths = [10, 16] if args.quick else DEPTHS
    budgets = [20_000] if args.quick else BUDGETS

    report = {"config": {"win_length": WIN_LENGTH, "radius": RADIUS},
              "games": {}, "grid": []}

    # Pre-replay every game once.
    replayed = []
    for name, moves in games:
        prefixes, final = replay_prefixes(hr, cfg, moves)
        winner = final.winner() if final.is_terminal() else None
        replayed.append((name, prefixes, winner, len(moves)))
        report["games"][name] = {"n_moves": len(moves), "winner": winner,
                                 "n_prefixes": len(prefixes)}
        print(f"{name}: {len(moves)} moves, winner={winner}, "
              f"{len(prefixes)} non-terminal prefixes", flush=True)

    for depth in depths:
        for budget in budgets:
            times = []
            first_seen = {}   # game -> earliest prefix idx proving winner's win
            for name, prefixes, winner, n_moves in replayed:
                for idx, state in prefixes:
                    mover = state.current_player()
                    # Winner-perspective solve: mover-side when the eventual
                    # winner is to move, else flipped (the defender-side view
                    # live play uses to spot incoming wins).
                    for side_state, side in (
                        (state, mover),
                        (flipped(hr, cfg, state, "P1" if mover == "P2" else "P2"),
                         "P1" if mover == "P2" else "P2"),
                    ):
                        res, dt = solve_timed(hr, side_state, depth, budget)
                        times.append(dt)
                        if (res is not None and winner is not None
                                and side == winner and name not in first_seen):
                            first_seen[name] = idx
            times.sort()
            n = len(times)
            entry = {
                "depth": depth, "budget": budget, "n_solves": n,
                "p50_ms": round(times[n // 2] * 1e3, 2),
                "p99_ms": round(times[int(n * 0.99)] * 1e3, 2),
                "max_ms": round(times[-1] * 1e3, 2),
                "total_s": round(sum(times), 1),
                "first_seen": {g: {"prefix": i,
                                   "placements_before_end":
                                       report["games"][g]["n_moves"] - 1 - i}
                               for g, i in first_seen.items()},
            }
            report["grid"].append(entry)
            fs = "  ".join(f"{g}@{v['prefix']}(-{v['placements_before_end']})"
                           for g, v in entry["first_seen"].items()) or "none"
            print(f"depth={depth:2d} budget={budget:>7,}: "
                  f"p50={entry['p50_ms']}ms p99={entry['p99_ms']}ms "
                  f"max={entry['max_ms']}ms total={entry['total_s']}s "
                  f"first-seen: {fs}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
