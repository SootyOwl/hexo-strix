"""Exploratory measurement: own-dead node prevalence in axis-window graphs.

This script is throwaway analysis to inform a pruning design, not part of
production code. It reads real positions from games.analysis.sqlite and
classifies stones / empty cells as own-dead for their controlling player.

Run: uv run --no-sync python scripts/explore_own_dead.py
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from statistics import mean, median

WIN_AXES = [(1, 0), (0, 1), (1, -1)]
WIN_LENGTH = 6


def is_own_dead(cell, player, stones_by_coord, win_length):
    """Return True if no length-W window through cell is clear of opponent stones.

    The board is treated as infinite / sparse: any cell not occupied by an
    opponent stone is empty, and empties are potential future own stones.
    """
    opp = "P2" if player == "P1" else "P1"
    cq, cr = cell
    for dq, dr in WIN_AXES:
        for offset in range(win_length):
            clear = True
            for step in range(win_length):
                tq = cq + dq * (step - offset)
                tr = cr + dr * (step - offset)
                if stones_by_coord.get((tq, tr)) == opp:
                    clear = False
                    break
            if clear:
                return False
    return True


def classify_entry(entry):
    stones = [((q, r), p) for [q, r], p in entry["stones"]]
    stones_by_coord = {c: p for c, p in stones}
    legal = set(tuple(c) for c in entry["legal"])

    p1_dead_stones = 0
    p2_dead_stones = 0
    both_dead_empty = 0
    p1_only_dead_empty = 0
    p2_only_dead_empty = 0
    alive_empty = 0

    for c, p in stones:
        if p == "P1" and is_own_dead(c, "P1", stones_by_coord, WIN_LENGTH):
            p1_dead_stones += 1
        if p == "P2" and is_own_dead(c, "P2", stones_by_coord, WIN_LENGTH):
            p2_dead_stones += 1

    for c in legal:
        d1 = is_own_dead(c, "P1", stones_by_coord, WIN_LENGTH)
        d2 = is_own_dead(c, "P2", stones_by_coord, WIN_LENGTH)
        if d1 and d2:
            both_dead_empty += 1
        elif d1:
            p1_only_dead_empty += 1
        elif d2:
            p2_only_dead_empty += 1
        else:
            alive_empty += 1

    return {
        "n_stones": len(stones),
        "n_legal": len(legal),
        "p1_dead_stones": p1_dead_stones,
        "p2_dead_stones": p2_dead_stones,
        "both_dead_empty": both_dead_empty,
        "p1_only_dead_empty": p1_only_dead_empty,
        "p2_only_dead_empty": p2_only_dead_empty,
        "alive_empty": alive_empty,
        "model": entry.get("model", "?"),
    }


def main():
    conn = sqlite3.connect("/var/home/tyto/Development/personal/hexo-strix/games.analysis.sqlite")
    rows = conn.execute("SELECT entry, model, sims FROM positions").fetchall()
    conn.close()

    stats = []
    for entry_json, model, sims in rows:
        entry = json.loads(entry_json)
        entry["model"] = model
        stats.append(classify_entry(entry))

    print(f"Positions sampled: {len(stats)}")
    print(f"Models: {sorted(set(s['model'] for s in stats))}")
    print()
    print("Node counts (median / mean)")
    for k in ["n_stones", "n_legal"]:
        vals = [s[k] for s in stats]
        print(f"  {k:20s}: median={median(vals):7.1f} mean={mean(vals):7.1f}")
    print()
    print("Own-dead stones (% of all stones)")
    for k in ["p1_dead_stones", "p2_dead_stones"]:
        ratios = [s[k] / max(1, s["n_stones"]) * 100 for s in stats]
        abs_vals = [s[k] for s in stats]
        print(f"  {k:20s}: median={median(abs_vals):5.1f} ({median(ratios):5.1f}%)")
    print()
    print("Empty-cell deadness (% of all legal moves)")
    for k in [
        "both_dead_empty",
        "p1_only_dead_empty",
        "p2_only_dead_empty",
        "alive_empty",
    ]:
        ratios = [s[k] / max(1, s["n_legal"]) * 100 for s in stats]
        abs_vals = [s[k] for s in stats]
        print(f"  {k:20s}: median={median(abs_vals):5.1f} ({median(ratios):5.1f}%)")
    print()

    total_nodes = sum(s["n_stones"] + s["n_legal"] for s in stats)
    prunable = sum(
        s["p1_dead_stones"]
        + s["p2_dead_stones"]
        + s["both_dead_empty"]
        for s in stats
    )
    print(f"Rough prunable real nodes: {prunable}/{total_nodes} ({prunable/total_nodes*100:.1f}%)")

    print("\nBy stone count bucket (median prunable-node share):")
    buckets = Counter()
    bucket_prune = Counter()
    for s in stats:
        b = s["n_stones"] // 25 * 25
        buckets[b] += 1
        bucket_prune[b] += (
            s["p1_dead_stones"] + s["p2_dead_stones"] + s["both_dead_empty"]
        ) / max(1, s["n_stones"] + s["n_legal"])
    for b in sorted(buckets):
        print(f"  stones {b:3d}-{b+24:3d}: n={buckets[b]:4d} median share={bucket_prune[b]/buckets[b]*100:5.1f}%")


if __name__ == "__main__":
    main()
