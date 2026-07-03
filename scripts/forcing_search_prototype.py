"""Fully-forcing (VCF) solver prototype for HeXO.

Validates the methodology in docs/superpowers/specs/2026-07-03-forcing-line-search-design.md
before a Rust port. Pure Python; no hexo_rs / cargo. The HDS loader (added later)
imports hexo_a0.serving.hds lazily so the kernel/search are usable standalone.

Board: dict[(q, r) -> "P1" | "P2"].  Cells are axial (q, r) int tuples.
"""
import base64
from collections import namedtuple
from itertools import combinations

# Win axes, copied from hexo-engine types.rs WIN_AXES.
WIN_AXES = [(1, 0), (0, 1), (1, -1)]

# Solver result. outcome in {"WIN", "NO", "BUDGET_EXCEEDED"}; depth in attacker
# turns (only meaningful on WIN).
Result = namedtuple("Result", ["outcome", "depth"])


def _wide_partners(board, atk, wl):
    """Empty cells within win_length-1 along an axis of any attacker stone. The
    'wide' partner set — a superset of hot cells that also admits build stones one
    step off an existing line (needed when a forcing line grows a fresh threat)."""
    cells = set()
    for (sq, sr), owner in board.items():
        if owner != atk:
            continue
        for (dq, dr) in WIN_AXES:
            for k in range(1, wl):
                for c in ((sq + k * dq, sr + k * dr), (sq - k * dq, sr - k * dr)):
                    if board.get(c) is None:
                        cells.add(c)
    return cells


def _threat_table(board, player, wl):
    """Precompute the player's near-complete lines once per position: enemy-free
    length-wl windows already holding win_length-4..win_length-3 player stones (so
    1-2 placements push them to a completion). Returns a dict mapping each empty cell
    to the (empties, attacker_count) of every candidate window through it — so a
    move's mandatory-block count is read from the table, not rescanned."""
    index = {}
    seen = set()
    bget = board.get
    for (sq, sr) in [c for c, p in board.items() if p == player]:
        for (dq, dr) in WIN_AXES:
            coords = [(sq + o * dq, sr + o * dr) for o in range(1 - wl, wl)]
            owners = [bget(c) for c in coords]
            for k in range(wl):  # windows owners[k:k+wl] all include the stone
                pc, empties, enemy = 0, [], False
                for j in range(k, k + wl):
                    o = owners[j]
                    if o is None:
                        empties.append(coords[j])
                    elif o == player:
                        pc += 1
                    else:
                        enemy = True
                        break
                if enemy or not (wl - 4 <= pc <= wl - 3):
                    continue
                window = tuple(coords[k:k + wl])
                if window in seen:
                    continue
                seen.add(window)
                entry = (frozenset(empties), pc)
                for e in empties:
                    index.setdefault(e, []).append(entry)
    return index


def _move_B(index, move, wl):
    """Mandatory-block count B after the attacker plays `move`, read from the threat
    table (no board scan): a candidate window becomes a completion once the move
    fills enough of its empties to reach win_length-2."""
    placed = frozenset(move)
    comps = set()
    for cell in move:
        for empties, pc in index.get(cell, ()):
            if pc + len(empties & placed) >= wl - 2:
                comps.add(empties - placed)
    return min_covers(comps)[0]


def _attacker_turns(board, placements, state, pk=None):
    """Return the forcing attacker placements (B >= 2), strongest threat first.

    Every forcing move places >= 1 hot cell (a threat-table empty). The partner is
    another hot cell or a cell blocking a standing defender threat, so the attacker
    can force while defending. Best-first ordering (higher B first) lets the OR
    search short-circuit on strong threats. B is read from the threat table."""
    cache = state.get("gencache")
    ckey = None
    if cache is not None:
        if pk is None:
            pk = frozenset(board.items())
        ckey = (pk, placements)
        if ckey in cache:  # move list is budget-independent
            return cache[ckey]
    atk, dfn, wl = state["atk"], state["def"], state["wl"]
    index = _threat_table(board, atk, wl)
    hot = list(index)  # hot cells = empties of candidate windows

    if placements == 1:
        candidates = [(c,) for c in hot]
    else:
        partners = set(hot)
        for comp in completions(board, dfn, wl):  # defender block cells
            partners |= set(comp)
        if state.get("width") == "wide":
            partners |= _wide_partners(board, atk, wl)
        candidates, seen = [], set()
        for h in hot:
            for a in partners:
                if a == h:
                    continue
                move = (h, a) if h < a else (a, h)
                if move not in seen:
                    seen.add(move)
                    candidates.append(move)
    scored = sorted(((_move_B(index, m, wl), m) for m in candidates), key=lambda s: -s[0])
    moves = [m for b, m in scored if b >= 2]  # forcing only, strongest first
    if cache is not None:
        cache[ckey] = moves
    return moves


def _tick(state):
    """Count a node; return True if the budget is now exhausted."""
    state["nodes"] += 1
    if state["nodes"] > state["budget"]:
        state["exceeded"] = True
    return state["exceeded"]


def _node_comps(board, state):
    """(pos_key, attacker_completions, defender_completions), completions cached per
    position so the full-board scan runs once per position instead of once per visit
    (each position is revisited across deepening rounds and transpositions)."""
    pk = frozenset(board.items())
    c = state["comps"].get(pk)
    if c is None:
        wl = state["wl"]
        c = (completions(board, state["atk"], wl), completions(board, state["def"], wl))
        state["comps"][pk] = c
    return pk, c[0], c[1]


def _atk_within(board, placements, budget, state):
    """Return (won, cutoff): won = attacker has a forced win within `budget` attacker
    turns; cutoff = a deeper budget might change a False to True. (won, cutoff) is a
    function of (position, side, budget), so it is memoized across deepening rounds."""
    if state["exceeded"] or _tick(state):
        return (False, False)
    pk, atk_comps, _ = _node_comps(board, state)
    key = (pk, "A", placements, budget)
    memo = state["tt"].get(key)
    if memo is not None:
        return memo
    if any(len(c) <= placements for c in atk_comps):
        state["tt"][key] = (True, False)  # immediate win, using this turn
        return (True, False)
    if budget < 2:  # a non-immediate forced win needs >=2 turns (force, then finish)
        return (False, True)  # a deeper search might still find one
    atk = state["atk"]
    won, cut = False, False
    for move in _attacker_turns(board, placements, state, pk):
        for c in move:
            board[c] = atk
        w, subcut = _def_within(board, budget - 1, state)
        for c in move:
            del board[c]
        cut = cut or subcut
        if w:
            won = True
            break
    result = (True, False) if won else (False, cut)
    if not state["exceeded"]:
        state["tt"][key] = result
    return result


def _def_within(board, budget, state):
    """Return (won, cutoff) from the defender's turn: won = attacker still wins
    against every defender reply, with `budget` attacker turns left to finish."""
    if state["exceeded"] or _tick(state):
        return (False, False)
    pk, atk_comps, def_comps = _node_comps(board, state)
    # Defender to move: if they can complete their own win this turn, they take it
    # (they move before the attacker finishes) — a definitive refutation, no cutoff.
    if any(len(c) <= 2 for c in def_comps):
        return (False, False)
    B, covers = min_covers(atk_comps)
    if B < 2:  # not forcing (shouldn't occur: attacker_turns guarantees B >= 2)
        return (False, False)
    if B >= 3:  # defender cannot cover; attacker completes on the next turn
        return (True, False) if budget >= 1 else (False, True)
    key = (pk, "D", budget)
    memo = state["tt"].get(key)
    if memo is not None:
        return memo
    dfn = state["def"]
    result = (True, False)
    for cover in covers:  # every minimal cover must still lose (defender prolongs)
        cells = list(cover)
        for c in cells:
            board[c] = dfn
        w, subcut = _atk_within(board, 2, budget, state)
        for c in cells:
            del board[c]
        if not w:
            result = (False, subcut)  # defender escapes here
            break
    if not state["exceeded"]:
        state["tt"][key] = result
    return result


def solve(board, attacker, placements_remaining=2, win_length=6, *,
          mode="find_shortest", node_budget=5_000_000, max_depth=40,
          partner_width="tight"):
    """Solve for a fully-forcing (VCF) win for `attacker` from `board`.

    Returns Result(outcome, depth). find_shortest iterative-deepens, so depth is the
    shortest forced win in attacker turns; a depth that finds no win with no
    depth-cutoff proves NO. find_any does one deep pass (WIN carries depth=None).
    partner_width "tight" = hot + defender-block cells; "wide" also admits build
    stones within win_length-1 of an attacker stone (slower, more complete).
    """
    defender = "P2" if attacker == "P1" else "P1"
    state = {
        "wl": win_length, "atk": attacker, "def": defender, "mode": mode,
        "nodes": 0, "budget": node_budget, "exceeded": False, "tt": {},
        "gencache": {}, "comps": {}, "width": partner_width,
    }
    if mode == "find_any":
        won, _ = _atk_within(dict(board), placements_remaining, max_depth, state)
        if state["exceeded"]:
            return Result("BUDGET_EXCEEDED", None)
        return Result("WIN", None) if won else Result("NO", None)
    for depth in range(1, max_depth + 1):
        # TT persists across rounds: (won, cutoff) is keyed by budget, so entries
        # from a shallower round stay valid and skip re-search.
        won, cut = _atk_within(dict(board), placements_remaining, depth, state)
        if won:
            return Result("WIN", depth)
        if state["exceeded"]:
            return Result("BUDGET_EXCEEDED", None)
        if not cut:
            return Result("NO", None)  # whole forcing tree explored, no win
    return Result("BUDGET_EXCEEDED", None)  # still cut off at max_depth


class _PN:
    """Proof-number-search node. Stores only the move that enters it (the board is
    reconstructed by a working board during descent/backup), not a board copy."""
    __slots__ = ("is_or", "placements", "move", "parent", "children", "pn", "dn")

    def __init__(self, is_or, placements, move, parent):
        self.is_or = is_or            # OR = attacker to move; AND = defender to move
        self.placements = placements
        self.move = move              # cells applied to reach this node (None at root)
        self.parent = parent
        self.children = None          # None = unexpanded; [] = terminal
        self.pn = 1
        self.dn = 1


def solve_pns(board, attacker, placements_remaining=2, win_length=6, node_budget=2_000_000):
    """Proof-number search: a best-first AND-OR search that always expands the
    most-proving node. Finds *a* forcing win (WIN) or proves none exists (NO) — no
    depth. Basic (tree) PNS: transpositions are re-expanded, not deduped."""
    INF = 10 ** 9
    dfn = "P2" if attacker == "P1" else "P1"
    wl = win_length
    gen = {"atk": attacker, "def": dfn, "wl": wl, "gencache": {}, "width": "tight"}
    work = dict(board)

    def apply(node):  # OR nodes are entered by a defender cover; AND by an attacker move
        player = dfn if node.is_or else attacker
        for c in node.move:
            work[c] = player

    def undo(node):
        for c in node.move:
            del work[c]

    def numbers(node):  # recompute pn/dn from children
        if node.is_or:
            node.pn = min(c.pn for c in node.children)
            node.dn = min(sum(c.dn for c in node.children), INF)
        else:
            node.pn = min(sum(c.pn for c in node.children), INF)
            node.dn = min(c.dn for c in node.children)

    def expand(node):  # `work` is positioned at this node
        if node.is_or:
            if any(len(c) <= node.placements for c in completions(work, attacker, wl)):
                node.pn, node.dn, node.children = 0, INF, []          # immediate win
                return
            moves = _attacker_turns(work, node.placements, gen)
            if not moves:
                node.pn, node.dn, node.children = INF, 0, []          # attack fizzles
                return
            node.children = [_PN(False, 2, m, node) for m in moves]
        else:
            if any(len(c) <= 2 for c in completions(work, dfn, wl)):
                node.pn, node.dn, node.children = INF, 0, []          # defender wins first
                return
            B, covers = min_covers(completions(work, attacker, wl))
            if B >= 3:
                node.pn, node.dn, node.children = 0, INF, []          # unstoppable
                return
            node.children = [_PN(True, 2, tuple(sorted(c)), node) for c in covers]
        numbers(node)

    root = _PN(True, placements_remaining, None, None)
    expand(root)
    nodes = 0
    while root.pn and root.dn:
        if nodes >= node_budget:
            return Result("BUDGET_EXCEEDED", None)
        node = root  # descend to the most-proving leaf, applying moves
        while node.children:
            node = min(node.children, key=(lambda c: c.pn) if node.is_or else (lambda c: c.dn))
            apply(node)
        expand(node)
        nodes += 1
        while node.parent is not None:  # back up, reverting the board
            undo(node)
            node = node.parent
            numbers(node)
    return Result("WIN", None) if root.pn == 0 else Result("NO", None)


def solve_dfpn(board, attacker, placements_remaining=2, win_length=6, node_budget=2_000_000):
    """Depth-first proof-number search (df-pn): DAG-aware PNS with a position-keyed
    (pn, dn) transposition table and proof/disproof thresholds — no explicit tree.
    Sound for HeXO because a position's forcing value is path-independent (stones only
    add: no cycles, no graph-history interaction). Finds a forcing WIN or proves NO."""
    INF = 10 ** 9
    dfn = "P2" if attacker == "P1" else "P1"
    wl = win_length
    st = {"atk": attacker, "def": dfn, "wl": wl, "gencache": {}, "comps": {}, "width": "tight"}
    tt = {}
    work = dict(board)
    calls = [0]

    def key(is_or, placements):
        return (frozenset(work.items()), is_or, placements if is_or else 0)

    def terminal_or_children(is_or, placements):
        _pk, atk_comps, def_comps = _node_comps(work, st)
        if is_or:
            if any(len(c) <= placements for c in atk_comps):
                return (0, INF)                         # immediate win
            moves = _attacker_turns(work, placements, st)
            return [(m, False, 2) for m in moves] if moves else (INF, 0)  # else fizzle
        if any(len(c) <= 2 for c in def_comps):
            return (INF, 0)                             # defender wins first
        B, covers = min_covers(atk_comps)
        if B >= 3:
            return (0, INF)                             # unstoppable
        if B < 2:
            return (INF, 0)
        return [(tuple(sorted(c)), True, 2) for c in covers]

    def child_val(child):  # (pn, dn) of a child position from the TT (1,1 if unseen)
        move, cis, _ = child
        player = attacker if not cis else dfn  # AND child entered by attacker, OR by defender
        for c in move:
            work[c] = player
        v = tt.get(key(cis, child[2]), (1, 1))
        for c in move:
            del work[c]
        return v

    def dfpn(is_or, placements, thpn, thdn):
        calls[0] += 1
        k = key(is_or, placements)
        tc = terminal_or_children(is_or, placements)
        if isinstance(tc, tuple):
            tt[k] = tc
            return
        children = tc
        while calls[0] < node_budget:
            vals = [child_val(ch) for ch in children]
            if is_or:
                pn = min(v[0] for v in vals)
                dn = min(sum(v[1] for v in vals), INF)
            else:
                pn = min(sum(v[0] for v in vals), INF)
                dn = min(v[1] for v in vals)
            tt[k] = (pn, dn)
            if pn >= thpn or dn >= thdn:
                return
            if is_or:  # descend into the min-pn child; bound it by the runner-up + 1
                best = min(range(len(children)), key=lambda i: vals[i][0])
                second = min((vals[i][0] for i in range(len(children)) if i != best), default=INF)
                cthpn, cthdn = min(thpn, second + 1), thdn - (dn - vals[best][1])
            else:      # descend into the min-dn child
                best = min(range(len(children)), key=lambda i: vals[i][1])
                second = min((vals[i][1] for i in range(len(children)) if i != best), default=INF)
                cthpn, cthdn = thpn - (pn - vals[best][0]), min(thdn, second + 1)
            move, cis, cpl = children[best]
            player = attacker if not cis else dfn
            for c in move:
                work[c] = player
            dfpn(cis, cpl, cthpn, cthdn)
            for c in move:
                del work[c]

    dfpn(True, placements_remaining, INF, INF)
    if calls[0] >= node_budget:
        return Result("BUDGET_EXCEEDED", None)
    pn, _ = tt[key(True, placements_remaining)]
    return Result("WIN", None) if pn == 0 else Result("NO", None)


def _fresh_state(atk, dfn, wl, node_budget):
    return {"wl": wl, "atk": atk, "def": dfn, "mode": "find_shortest", "nodes": 0,
            "budget": node_budget, "exceeded": False, "tt": {}, "gencache": {},
            "comps": {}, "width": "tight"}


def principal_variation(board, attacker, placements_remaining=2, win_length=6,
                        node_budget=5_000_000, max_depth=40):
    """Extract one winning line under best (prolonging) defense as a list of
    (player, (cell, ...)) placements. Returns (Result, pv). Only meaningful on WIN;
    re-queries sub-depths to pick the defender cover that drags the win out longest,
    so the line's length matches the reported winning-move-in-N depth."""
    res = solve(board, attacker, placements_remaining, win_length,
                node_budget=node_budget, max_depth=max_depth)
    if res.outcome != "WIN":
        return res, []
    dfn = "P2" if attacker == "P1" else "P1"
    wl, b, pv = win_length, dict(board), []
    pr, remaining = placements_remaining, res.depth
    while True:
        fin = sorted(tuple(sorted(c)) for c in completions(b, attacker, wl) if len(c) <= pr)
        if fin:  # attacker completes now
            for c in fin[0]:
                b[c] = attacker
            pv.append((attacker, fin[0]))
            break
        chosen = None
        for move in _attacker_turns(b, pr, _fresh_state(attacker, dfn, wl, node_budget)):
            for c in move:
                b[c] = attacker
            ok, _ = _def_within(b, remaining - 1, _fresh_state(attacker, dfn, wl, node_budget))
            for c in move:
                del b[c]
            if ok:
                chosen = move
                break
        if chosen is None:
            break
        for c in chosen:
            b[c] = attacker
        pv.append((attacker, chosen))
        pr = 2
        B, covers = min_covers(completions(b, attacker, wl))
        if B >= 3:  # unstoppable: show a futile 2-cell block, then the completion
            threat = sorted(set().union(*(set(c) for c in completions(b, attacker, wl))))
            dmove = tuple(threat[:2])
            for c in dmove:
                b[c] = dfn
            pv.append((dfn, dmove))
            remaining -= 1
            continue
        best = None  # pick the cover that prolongs the win the most
        for cover in covers:
            for c in cover:
                b[c] = dfn
            sub = next((d for d in range(1, remaining)
                        if _atk_within(dict(b), 2, d, _fresh_state(attacker, dfn, wl, node_budget))[0]),
                       None)
            for c in cover:
                del b[c]
            if sub is not None and (best is None or sub > best[0]):
                best = (sub, cover)
        if best is None:
            break
        for c in best[1]:
            b[c] = dfn
        pv.append((dfn, tuple(sorted(best[1]))))
        remaining = best[0]
    return res, pv


def render_line(move_log, pv, base_url="https://hexo.tyto.cc/analysis#c="):
    """Render a solution line (puzzle move_log + pv placements) as (htttx, url)."""
    from hexo_a0.serving.htttx import serialize_htttx
    full_log = list(move_log) + [(q, r, player) for (player, cells) in pv for (q, r) in cells]
    coords = [(q, r) for (q, r, _p) in full_log]
    return serialize_htttx(full_log), base_url + encode_moves_compact(coords)


def encode_moves_compact(moves):
    """Port of encodeMovesCompact (static/shared.js): the `#c=` analysis-URL codec.

    `moves` includes the (0,0) seed at index 0, which is dropped. Each remaining
    (q, r) is zig-zag + LEB128-varint encoded, concatenated, base64url'd (trailing
    '=' stripped). Must byte-match the JS."""
    out = bytearray()
    for (q, r) in moves[1:]:
        for v in (q * 2 if q >= 0 else -q * 2 - 1, r * 2 if r >= 0 else -r * 2 - 1):
            while v > 0x7F:
                out.append((v & 0x7F) | 0x80)
                v //= 128
            out.append(v)
    return base64.urlsafe_b64encode(bytes(out)).decode().rstrip("=")


def decode_moves_compact(s):
    """Inverse of encode_moves_compact (for round-trip tests)."""
    data = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    ints, i = [], 0
    while i < len(data):
        shift = val = 0
        while True:
            b = data[i]
            i += 1
            val += (b & 0x7F) * (2 ** shift)
            shift += 7
            if not b & 0x80:
                break
        ints.append(-((val + 1) // 2) if val % 2 else val // 2)
    return [(ints[j], ints[j + 1]) for j in range(0, len(ints) - 1, 2)]


def load_puzzle(path):
    """Load an HDS sandbox JSON fixture -> (board, attacker, placements_remaining).

    Reuses the tested hexo_a0.serving.hds converter (identity (x,y)->(q,r), opener
    relabeled to P1 by move order). Attacker is the side to move. Imported lazily so
    the kernel/search stay usable without the hexo_a0 package.
    """
    import json
    from hexo_a0.serving.hds import sandbox_to_move_log
    with open(path) as f:
        data = json.load(f)
    move_log, meta = sandbox_to_move_log(data["gamePosition"])
    board = {(q, r): p for (q, r, p) in move_log}
    return board, meta["current_turn"], meta["placements_remaining"], move_log


def load_hds(url_or_code):
    """Fetch a live HDS sandbox position (URL or 7-char code) -> the same tuple as
    load_puzzle. Reuses hexo_a0.serving.hds for the network fetch + conversion."""
    from hexo_a0.serving.hds import (
        extract_sandbox_code, fetch_sandbox_position, sandbox_to_move_log)
    resp = fetch_sandbox_position(extract_sandbox_code(url_or_code))
    move_log, meta = sandbox_to_move_log(resp["gamePosition"])
    board = {(q, r): p for (q, r, p) in move_log}
    return board, meta["current_turn"], meta["placements_remaining"], move_log


def completions(board, player, win_length):
    """Empty-cell sets that `player` can fill (<= 2 cells) to reach win_length.

    A completion is a length-`win_length` window along an axis that is enemy-free
    and already holds >= win_length-2 of `player`'s stones; the emitted set is the
    window's empty cells (size 1 or 2). Filling them all yields win_length in a row.
    """
    L = win_length
    result = set()
    bget = board.get
    for (sq, sr) in [c for c, p in board.items() if p == player]:
        for (dq, dr) in WIN_AXES:
            # Scan the axis into a 2L-1 strip once, then slide the L windows that
            # cover the stone over the list (no per-cell board.get per window).
            coords = [(sq + o * dq, sr + o * dr) for o in range(1 - L, L)]
            owners = [bget(c) for c in coords]
            for k in range(L):  # windows owners[k:k+L] all include the center
                pc = 0
                empties = []
                enemy = False
                for j in range(k, k + L):
                    o = owners[j]
                    if o is None:
                        empties.append(coords[j])
                    elif o == player:
                        pc += 1
                    else:
                        enemy = True
                        break
                if not enemy and L - 2 <= pc <= L - 1:  # 1 or 2 empties to fill
                    result.add(frozenset(empties))
    return result


def min_covers(completion_sets):
    """(B, covers): minimum hitting-set size B and all covers of that size.

    A cover intersects every completion set. Inputs are small (a handful of
    size-1/2 sets), so enumerate k-subsets of the involved cells by increasing k.
    """
    comps = [set(c) for c in completion_sets]
    if not comps:
        return 0, [frozenset()]
    universe = sorted(set().union(*comps))
    for k in range(1, len(universe) + 1):
        covers = [
            frozenset(sub)
            for sub in combinations(universe, k)
            if all(not c.isdisjoint(sub) for c in comps)
        ]
        if covers:
            return k, covers
    return len(universe), [frozenset(universe)]  # unreachable: universe hits all


def _main(argv):
    if len(argv) < 2:
        print("usage: python scripts/forcing_search_prototype.py <hds-url-or-code>")
        return 2
    board, attacker, pr, mlog = load_hds(argv[1])
    print(f"loaded {len(board)} stones; {attacker} to move ({pr} placement(s)); solving...")
    res, pv = principal_variation(board, attacker, placements_remaining=pr,
                                  node_budget=20_000_000)
    if res.outcome != "WIN":
        print(f"no fully-forcing win found ({res.outcome})")
        return 0
    htttx, url = render_line(mlog, pv)
    print(f"\nFORCING WIN — winning move in {res.depth} (attacker = {attacker})\n")
    for player, cells in pv:
        who = "attacker" if player == attacker else "defender"
        print(f"  {who} ({player}): " + "  ".join(f"({q},{r})" for q, r in cells))
    print(f"\nHTTTX:\n{htttx}\n\nsolution URL: {url}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv))
