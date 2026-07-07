"""Analysis core: per-position and trajectory evaluation.

Mirrors the committed ``scripts/policy_viewer.py`` spike. The meaningful
per-move signal is ``q_hat`` on ``candidate_set`` (visit_count > 0) — visit
counts and improved_policy under Gumbel-Top-K Sequential Halving are not a
quality heatmap.
"""
from dataclasses import dataclass, field
import math

from hexo_a0.serving.model import make_graph_fn
from hexo_a0.serving.nativeutil import softmax


# Forcing-solver (VCF) budgets for the analysis display.
#
# Bench (2026-07-03, release build, 1,501 real wl6/r2 corpus roots, per-solve):
#   depth= 6 / nodes=    2,000 ->  93/1501 wins found, max   49ms
#   depth=10 / nodes=   20,000 ->  98/1501 wins found, max  0.4s
#   depth=12 / nodes=  250,000 -> 100/1501 wins found, max  5.5s
#   depth=16 / nodes=1,000,000 -> 100/1501 wins found, max 16.7s (zero extra
#                                  over depth=12 — the tail beyond it is flat)
#
# ANALYSIS_* is for single-position solves (the /analyze endpoint — the user
# is looking at one position, so the rare 5.5s worst case is acceptable).
# Note this is now ABOVE live play's LIVE_DEPTH_CAP/LIVE_NODE_BUDGET (game.py,
# re-tuned 2026-07-04 to 10/20,000 from r=8 measurement showing deeper
# budgets find nothing extra on real positions) — analysis affords the
# richer budget because it solves once for a human looking at one position,
# whereas analyze_game_full below solves twice per prefix across a whole
# game and needs a much tighter budget to stay bounded (see TRAJECTORY_*,
# which happens to land on the same depth/budget as live play's).
ANALYSIS_DEPTH_CAP = 12
ANALYSIS_NODE_BUDGET = 250_000
# Per-prefix budget for analyze_game_full's forcing solves (run twice per
# placement inside the SSE loop) — the depth=10/20k row above caps the
# per-solve worst case at ~0.4s, keeping a whole-game analysis bounded.
TRAJECTORY_DEPTH_CAP = 10
TRAJECTORY_NODE_BUDGET = 20_000


def _finite(x):
    """Coerce NaN/Inf to 0.0 so JSON output stays standards-compliant."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    return xf if math.isfinite(xf) else 0.0


@dataclass
class Threat:
    owner: str
    length: int
    axis: tuple[int, int]
    start: tuple[int, int]
    extensions: list[dict]


@dataclass
class AnalysisResult:
    legal: list[list[int]]
    value: float
    probs: list[float]
    stones: list = field(default_factory=list)  # [[ [q,r], "P1"/"P2" ], ...]
    q_hat: list[float] | None = None
    improved_policy: list[float] | None = None
    candidate_set: list[bool] | None = None
    threats: list[Threat] = field(default_factory=list)
    terminal: bool = False
    winner: str | None = None
    current_player: str | None = None  # side to move (for P1-perspective flip)
    # Both-side VCF solve for display: {"winner", "attacker_is_mover",
    # "first_move", "pv", "pv_len", "pv_owners"} or None (no forced win found
    # either way). attacker_is_mover=True is a PROVEN win for the side to
    # move; False is a perspective-flipped THREAT (often defensible — the
    # client renders those behind a default-off toggle).
    # "pv_owners" is a parallel array of "P1"/"P2" (or None if the replay that
    # derives it failed) — the PV's per-cell chunk lengths are NOT a fixed
    # pairs-of-2 cadence (the first chunk is however many placements the
    # analyzed side has left THIS turn, and the final chunk can be a single
    # cell that ends the game), so ownership can't be inferred from position
    # alone. See _solve_forcing.
    forcing: dict | None = None

    def to_json(self) -> dict:
        return {
            "legal": self.legal,
            "stones": self.stones,
            "value": _finite(self.value),
            "current_player": self.current_player,
            "probs": [_finite(p) for p in self.probs],
            "q_hat": [_finite(q) for q in self.q_hat] if self.q_hat is not None else None,
            "improved_policy": ([_finite(p) for p in self.improved_policy]
                                if self.improved_policy is not None else None),
            "candidate_set": self.candidate_set,
            "threats": [
                {
                    "owner": t.owner,
                    "length": t.length,
                    "axis": list(t.axis),
                    "start": list(t.start),
                    "extensions": t.extensions,
                }
                for t in self.threats
            ],
            "terminal": self.terminal,
            "winner": self.winner,
            "forcing": self.forcing,
        }


def build_state(moves: list[tuple[int, int]], win_length: int,
                placement_radius: int, max_moves: int):
    """Replay ``moves`` into a fresh ``GameState``.

    The engine seeds P1 at (0,0); ``moves[0]`` is typically that seed and is
    skipped if so.
    """
    import hexo_rs as hr
    cfg = hr.GameConfig(win_length, placement_radius, max_moves)
    state = hr.GameState(cfg)
    for i, (q, r) in enumerate(moves):
        if i == 0 and q == 0 and r == 0:
            continue
        try:
            state.apply_move(q, r)
        except Exception as e:
            raise ValueError(f"Move {i} ({q},{r}): {e}") from e
    return state


def _stones_of(state) -> list:
    """Stone list in the same shape the play-board frontend expects."""
    return [[[int(q), int(r)], p] for ((q, r), p) in state.placed_stones()]


def _eval_fn_for_model(model, graph_fn):
    """Build an MCTS eval_fn that batches graphs and falls back to per-state."""
    def eval_fn(states):
        data_list = [graph_fn(s) for s in states]
        import torch
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(data_list).to("cpu")
        with torch.inference_mode():
            try:
                policy_logits_list, values = model.forward_batch(batch)
            except Exception:
                policy_logits_list = []
                values_list = []
                for d in data_list:
                    with torch.no_grad():
                        try:
                            logits, val = model(
                                d.x, d.edge_index, d.legal_mask,
                                stone_mask=d.stone_mask,
                                edge_attr=getattr(d, "edge_attr", None),
                            )
                            policy_logits_list.append(logits.tolist())
                            values_list.append(float(val.item()))
                        except Exception:
                            n = int(d.legal_mask.sum().item())
                            policy_logits_list.append([0.0] * max(n, 1))
                            values_list.append(0.0)
                return (policy_logits_list, values_list)
        logits_list = [p.tolist() for p in policy_logits_list]
        values_list = [float(v.item()) for v in values]
        return (logits_list, values_list)
    return eval_fn


def _run_mcts(model, model_config, state, mcts_sims: int, mcts_m_actions: int):
    """Run Gumbel MCTS and return diagnostic tensors.

    Returns: (improved_policy, root_value, legal_coords, visit_counts,
              per_child_q, candidate_indices)
    """
    import hexo_rs as hr
    mcts_config = hr.MCTSConfig(
        n_simulations=mcts_sims,
        m_actions=mcts_m_actions,
        c_visit=50,
        c_scale=1.0,
        disable_gumbel_noise=True,
    )
    native = getattr(model, "_hexo_native", None)
    if native is not None:
        # Search + root forward run fully in Rust (GIL released, leaf batches
        # threaded). Same return contract as the torch path below.
        (_action, improved_policy, visit_counts, per_child_q,
         _per_child_prior, candidate_indices, _forced) = (
            native.mcts_with_diagnostics(state, mcts_config)
        )
        _logit_map, root_value = native.forward(state)
        return (list(improved_policy), float(root_value),
                [list(c) for c in state.legal_moves()],
                list(visit_counts), list(per_child_q), list(candidate_indices))
    graph_fn = make_graph_fn(model_config)
    eval_fn = _eval_fn_for_model(model, graph_fn)
    (_action, improved_policy, visit_counts, per_child_q,
     _per_child_prior, candidate_indices, _forced) = (
        hr.gumbel_mcts_with_diagnostics(state, eval_fn, mcts_config)
    )
    legal_moves = state.legal_moves()
    data = graph_fn(state)
    import torch
    with torch.no_grad():
        _, value = model(
            data.x, data.edge_index, data.legal_mask,
            stone_mask=data.stone_mask,
            edge_attr=getattr(data, "edge_attr", None),
        )
    return (improved_policy, value.item(), [list(c) for c in legal_moves],
            list(visit_counts), list(per_child_q), list(candidate_indices))


def _scan_threats(data, legal_coords, probs, *, current_player, relative_stones):
    """Detect 3+-in-a-row threats with open extensions on any axis.

    Owner labelling respects the stone encoding: under absolute encoding col0 is
    P1 and col1 is P2; under ``relative_stone_encoding`` col0 is the side to move
    (``current_player``) and col1 is the opponent.
    """
    features = data.x
    if relative_stones:
        own = current_player
        opp = "P2" if current_player == "P1" else "P1"
    else:
        own, opp = "P1", "P2"
    stone_sets = {own: set(), opp: set()}
    for i in range(data.x.shape[0]):
        if data.stone_mask[i]:
            c = tuple(data.coords[i].tolist())
            if features[i, 0] > 0.5:
                stone_sets[own].add(c)
            elif features[i, 1] > 0.5:
                stone_sets[opp].add(c)
    return _threats_from_stone_sets(stone_sets, legal_coords, probs)


def _threats_from_stone_sets(stone_sets, legal_coords, probs):
    """Shared 3+-in-a-row open-threat walk over pre-bucketed stone sets.

    ``stone_sets`` maps owner-label -> set of (q, r); ``legal_coords`` is the
    move ordering ``probs`` is indexed in.
    """
    AXES = [(1, 0), (0, 1), (1, -1)]
    legal_set = set(tuple(c) for c in legal_coords)
    legal_list = [tuple(c) for c in legal_coords]

    threats = []
    for owner, stones in stone_sets.items():
        for (q, r) in stones:
            for dq, dr in AXES:
                length = 1
                while (q + length * dq, r + length * dr) in stones:
                    length += 1
                if length < 3:
                    continue
                prev = (q - dq, r - dr)
                if prev in stones:
                    continue
                exts = []
                for ext in ((q + length * dq, r + length * dr), prev):
                    if ext in legal_set:
                        idx = next((i for i, c in enumerate(legal_list) if c == ext), None)
                        # Guard against probs being shorter than legal_list
                        # (e.g. empty MCTS visit_counts) — never IndexError.
                        p = (float(probs[idx])
                             if idx is not None and 0 <= idx < len(probs)
                             else 0.0)
                        exts.append({"coord": list(ext), "prob": round(_finite(p), 4)})
                if exts:
                    threats.append(Threat(
                        owner=owner, length=length, axis=(dq, dr), start=(q, r),
                        extensions=exts,
                    ))
    return threats


def _scan_threats_from_state(state, legal_coords, probs, *, current_player, relative_stones):
    """Torch-free threat scan: buckets stones from state.placed_stones()."""
    if relative_stones:
        own, opp = current_player, ("P2" if current_player == "P1" else "P1")
    else:
        own, opp = "P1", "P2"
    own_abs = current_player if relative_stones else "P1"
    stone_sets = {own: set(), opp: set()}
    for (q, r), p in state.placed_stones():
        stone_sets[own if p == own_abs else opp].add((int(q), int(r)))
    return _threats_from_stone_sets(stone_sets, legal_coords, probs)


def _solve_forcing(state, depth_cap=ANALYSIS_DEPTH_CAP,
                   node_budget=ANALYSIS_NODE_BUDGET):
    """Both-side VCF solve for the analysis display.

    Solves for the side to move first; only if THAT side has no proven win is
    the opponent's perspective-flipped THREAT checked (``hr.solve_threat`` —
    the flip pretends the mover skips their turn, so a threat is often
    defensible, NOT a proven loss; ``attacker_is_mover`` distinguishes the
    two and the client renders threats behind a default-off toggle). Never
    raises: any solver failure or absence of a forced win on either side is
    reported as ``None`` — analysis must never crash because the display
    solve had trouble.
    """
    try:
        import hexo_rs as hr
        res = hr.solve_forcing(state, depth_cap, node_budget)
        attacker_is_mover = True
        winner, solved_state = state.current_player(), state
        if res is None:
            res = hr.solve_threat(state, depth_cap, node_budget)
            if res is None:
                return None
            attacker_is_mover = False
            opp = "P1" if winner == "P2" else "P2"
            winner = opp
            solved_state = hr.GameState.from_state(
                state.placed_stones(), opp, 2, state.config())
        first_move, pv = res
        pv = [[int(q), int(r)] for (q, r) in pv]
        # Per-cell ownership: the PV's chunk lengths are NOT a fixed pairs-of-2
        # cadence (the first chunk is whatever moves_remaining_this_turn() was
        # on the solved side, and the final chunk can be a single cell that
        # ends the game), so replay the PV on a clone to get the real mover at
        # each step. Never let a replay hiccup take down the rest of `forcing`.
        pv_owners = None
        try:
            replay = solved_state.clone()
            owners = []
            for (q, r) in pv:
                owners.append(replay.current_player())
                replay.apply_move(q, r)
            pv_owners = owners
        except Exception:
            pv_owners = None
        return {
            "winner": winner,
            "attacker_is_mover": attacker_is_mover,
            "first_move": [int(first_move[0]), int(first_move[1])],
            "pv": pv,
            "pv_len": len(pv),
            "pv_owners": pv_owners,
        }
    except Exception:
        return None


def analyze_position(model, model_config, state, *, mcts_sims: int = 0,
                     mcts_m_actions: int = 16,
                     forcing_depth_cap=ANALYSIS_DEPTH_CAP,
                     forcing_node_budget=ANALYSIS_NODE_BUDGET) -> AnalysisResult:
    """Evaluate a single position.

    With ``mcts_sims=0`` the result is a raw forward pass (softmax policy + value
    head). With ``mcts_sims > 0`` it runs Gumbel MCTS and returns visit-normalized
    ``probs`` plus ``q_hat`` / ``improved_policy`` / ``candidate_set`` for the
    visited candidates.

    Additionally solves for a forced win (VCF) on either side via
    ``hr.solve_forcing`` / ``hr.solve_threat`` — never the ``gumbel_mcts``
    shortcut, whose one-hot ``improved_policy`` would flatten this eval
    heatmap. See ``AnalysisResult.forcing``. ``forcing_depth_cap`` /
    ``forcing_node_budget`` default to the single-position ``ANALYSIS_*``
    budgets; ``analyze_game_full`` overrides them with the tighter
    ``TRAJECTORY_*`` budgets since it solves up to twice per prefix across a
    whole game.
    """
    if state.is_terminal():
        return AnalysisResult(
            legal=[], value=0.0, probs=[],
            stones=_stones_of(state),
            terminal=True, winner=state.winner(),
            current_player=None,
        )

    native = getattr(model, "_hexo_native", None)
    if native is not None:
        logit_map, raw_value = native.forward(state)
        legal_coords = [list(c) for c in state.legal_moves()]
        logits_list = [logit_map[(int(q), int(r))] for q, r in legal_coords]
        value = float(raw_value)
        data = None
    else:
        data = make_graph_fn(model_config)(state)
        import torch
        with torch.no_grad():
            raw_logits, raw_value = model(
                data.x, data.edge_index, data.legal_mask,
                stone_mask=data.stone_mask,
                edge_attr=getattr(data, "edge_attr", None),
            )
        logits_list = raw_logits.tolist()
        value = float(raw_value.item())
        legal_coords = data.coords[data.legal_mask].tolist()

    q_hat = None
    improved_pol = None
    candidate_set = None
    # ``legal_out`` is the move ordering that ``probs`` / ``q_hat`` /
    # ``candidate_set`` are indexed in. The frontend maps legal[i] -> probs[i],
    # so these MUST share an ordering. In the MCTS branch the diagnostic arrays
    # come back in ``state.legal_moves()`` order, not graph-builder order, so we
    # use that; in the raw branch we use graph-builder order (torch) or
    # legal_moves order (native) — both aligned to logits_list.
    legal_out = legal_coords

    if mcts_sims > 0 and not state.is_terminal():
        (improved_policy, mcts_value, mcts_legal, visit_counts, per_child_q,
         _candidate_indices) = _run_mcts(
            model, model_config, state, mcts_sims, mcts_m_actions)
        n_legal = len(mcts_legal)
        total_visits = sum(visit_counts)
        if total_visits > 0:
            probs = [v / total_visits for v in visit_counts]
        elif visit_counts and n_legal > 0:
            probs = [1.0 / n_legal] * n_legal
        else:
            probs = [0.0] * n_legal
        # Defensive length alignment: every per-child array must index the same
        # as mcts_legal so the frontend never mis-pairs a move with its Q/prob.
        # Pad short arrays and truncate long ones to exactly n_legal.
        probs = (probs + [0.0] * max(0, n_legal - len(probs)))[:n_legal]
        value = mcts_value
        q_hat = [round(_finite(q), 4) for q in per_child_q]
        q_hat = (q_hat + [0.0] * max(0, n_legal - len(q_hat)))[:n_legal]
        improved_pol = [round(_finite(p), 6) for p in improved_policy]
        improved_pol = (improved_pol + [0.0] * max(0, n_legal - len(improved_pol)))[:n_legal]
        candidate_set = [bool(vc > 0) for vc in visit_counts]
        candidate_set = (candidate_set + [False] * max(0, n_legal - len(candidate_set)))[:n_legal]
        legal_out = mcts_legal
    else:
        if native is not None:
            probs = softmax(logits_list)
        else:
            import torch
            probs_t = torch.softmax(raw_logits, dim=-1)
            # Some heads return [N, 1] logits; squeeze so probs is a flat list
            # (a nested list would crash _scan_threats' float(probs[idx])).
            if probs_t.dim() > 1:
                probs_t = probs_t.squeeze(-1)
            # HeXONet may return either legal-only or all-node logits; mask to
            # the legal nodes when the logits cover the full graph so probs[i]
            # lines up with legal_coords[i] (mirrors _pick_move's shape
            # handling).
            if probs_t.shape[0] == data.x.shape[0]:
                probs = probs_t[data.legal_mask].tolist()
            else:
                probs = probs_t.tolist()

    relative = bool(getattr(model_config, "relative_stone_encoding", False))
    if data is not None:
        threats = _scan_threats(data, legal_out, probs,
                                current_player=state.current_player(),
                                relative_stones=relative)
    else:
        threats = _scan_threats_from_state(state, legal_out, probs,
                                           current_player=state.current_player(),
                                           relative_stones=relative)
    forcing = _solve_forcing(state, forcing_depth_cap, forcing_node_budget)
    return AnalysisResult(
        legal=[list(c) for c in legal_out],
        value=value,
        probs=probs,
        stones=_stones_of(state),
        q_hat=q_hat,
        improved_policy=improved_pol,
        candidate_set=candidate_set,
        threats=threats,
        terminal=False,
        winner=None,
        current_player=state.current_player(),
        forcing=forcing,
    )


def end_of_turn_indices(current_players: list[str]) -> list[int]:
    """Return boundary indices where the current-player identity changes.

    The first index is always a boundary; the last index is always appended.
    An empty input yields an empty list (no turns).
    """
    if not current_players:
        return []
    out = []
    for i, cp in enumerate(current_players):
        if i == 0 or cp != current_players[i - 1]:
            out.append(i)
    if not out or out[-1] != len(current_players) - 1:
        out.append(len(current_players) - 1)
    return out


# Move-quality thresholds (loss = best-available q_hat minus achieved eval,
# both from the mover's perspective). See classify_turn_quality.
QUALITY_BLUNDER = 0.40
QUALITY_MISTAKE = 0.15
QUALITY_LABELS = {"best": "★", "good": "✓", "mistake": "?", "blunder": "✗"}
# Observatory palette: gold best, green good, amber mistake, crimson blunder.
QUALITY_COLORS = {"best": "#f2c14e", "good": "#79cf9a",
                  "mistake": "#e0a23a", "blunder": "#e25c5c"}


def _p1_perspective(value, current_player):
    """Value-head output is side-to-move perspective; flip to P1's perspective."""
    return -value if current_player == "P2" else value


def _engine_best_move(entry):
    """(coord, q_of_coord) for ``argmax improved_policy`` over the visited
    candidates at ``entry``, or (None, None). The engine's CHOICE is the MCTS
    visit-argmax (what it actually plays), NOT ``argmax q_hat``: the value head
    can rate an under-visited move higher while the search correctly avoids it."""
    ip, cs, lg, qh = (entry.get("improved_policy"), entry.get("candidate_set"),
                      entry.get("legal"), entry.get("q_hat"))
    if not ip or not lg:
        return None, None
    best_i, best_p = -1, float("-inf")
    for i in range(len(ip)):
        if cs and i < len(cs) and not cs[i]:
            continue
        if ip[i] > best_p:
            best_p, best_i = ip[i], i
    if best_i < 0:
        return None, None
    q = _finite(qh[best_i]) if qh and best_i < len(qh) else None
    return tuple(lg[best_i]), q


def _q_of_move(entry, coord):
    """q_hat of ``coord`` at ``entry`` (mover-perspective), or None."""
    lg, qh = entry.get("legal"), entry.get("q_hat")
    if not lg or not qh:
        return None
    for i, c in enumerate(lg):
        if tuple(c) == coord and i < len(qh):
            return _finite(qh[i])
    return None


def classify_turn_quality(trajectory, boundary_indices, moves, *,
                          turn_end_idx, prev_boundary_idx, eval_after_fn=None):
    """Classify a turn's quality by comparing END-OF-TURN BOARDS, order-free.

    HeXO plays two placements per turn and the intra-turn order never changes the
    resulting board, so a turn is judged as a SET of placements, not a sequence:
    ``[A, B]`` and ``[B, A]`` reach the same position and get the same verdict.
    We compare the player's placed-set for the turn against the engine's own best
    placed-set (its greedy policy line from the turn-start position) — equal sets
    (identical board) ⇒ "best", regardless of the order either side played them.

    The engine's choice at each placement is ``argmax improved_policy`` (the MCTS
    visit distribution — what it actually plays), NOT ``argmax q_hat``. q_hat only
    SIZES the loss, and only between the two full turns' END positions, where the
    estimates are directly comparable (both mover-perspective, both after the
    turn's placements).

    Args:
        trajectory: per-prefix entries; each non-terminal one carries
            ``improved_policy``, ``q_hat``, ``candidate_set`` and ``legal`` (as
            emitted by ``analyze_game_full``). q_hat is mover-perspective.
        boundary_indices: turn-boundary indices (from end_of_turn_indices).
        moves: full move list incl. leading (0,0) seed (analysisMoves).
        turn_end_idx: the boundary index whose turn we're classifying.
        prev_boundary_idx: the boundary index at the START of this turn.
        eval_after_fn: optional ``(prefix_index, first_move) -> mcts_json | None``
            used to evaluate the position AFTER the engine's first pick when the
            player did not play that pick first (so the engine's second pick isn't
            on the observed game line). Without it the engine line falls back to
            the observed positions, which can only confirm a forward-order match.

    Returns a quality dict (see below) or None if it can't be classified. The
    dict carries enough for the frontend to render order-independently:
    ``{"label", "icon", "color", "matched", "engine_pair", "played_pair",
       "loss", "player_end_q", "engine_end_q", "turn_start_depth"}``.
    """
    if turn_end_idx <= 0 or turn_end_idx not in boundary_indices:
        return None
    a, b = prev_boundary_idx, turn_end_idx
    if a < 0 or b >= len(trajectory):
        return None
    mover = trajectory[a].get("current_player")
    if not mover:
        return None

    # The turn's placements (moves[a+1 .. b]) as an ordered list + a set. moves
    # entries may carry a player tag as a 3rd element; take the (q, r) prefix.
    played = []
    for k in range(a, b):
        if k + 1 >= len(moves) or len(moves[k + 1]) < 2:
            return None
        played.append(tuple(moves[k + 1][:2]))
    played_set = set(played)

    # The engine's best line of the same length, greedily from the turn start.
    # Placement 1: argmax at the turn-start position (a).
    e0, _ = _engine_best_move(trajectory[a])
    if e0 is None:
        return None
    engine_line = [e0]
    engine_end_q = _q_of_move(trajectory[a], e0)
    if len(played) >= 2:
        # Placement 2: argmax at the position AFTER the engine's first pick.
        if e0 == played[0]:
            # The engine's first pick is what the player played first, so that
            # position is already on the observed line (trajectory[a+1]).
            e1, _ = _engine_best_move(trajectory[a + 1])
            e1_q = _q_of_move(trajectory[a + 1], e1) if e1 else None
        elif eval_after_fn is not None:
            # Off-line: evaluate the engine's own second placement.
            js = eval_after_fn(a, e0)
            if js:
                e1, e1_q = _engine_best_move(js)
            else:
                e1, e1_q = None, None
        else:
            e1, e1_q = None, None
        if e1 is None:
            # Can't determine the engine's second placement (its first pick wasn't
            # played first and no off-line evaluator was supplied), so the engine's
            # end-of-turn board is unknown — don't guess a verdict.
            return None
        engine_line.append(e1)
        engine_end_q = e1_q

    engine_set = set(engine_line)
    matched = len(engine_line) == len(played) and engine_set == played_set

    # Player's achieved end-of-turn value (mover-perspective): q_hat of the LAST
    # placement, evaluated at the position just before it (trajectory[b-1]).
    player_end_q = _q_of_move(trajectory[b - 1], played[-1])

    loss = 0.0
    if not matched and engine_end_q is not None and player_end_q is not None:
        loss = max(0.0, engine_end_q - player_end_q)  # both end-of-turn, comparable

    if matched:
        label = "best"
    elif loss >= QUALITY_BLUNDER:
        label = "blunder"
    elif loss >= QUALITY_MISTAKE:
        label = "mistake"
    else:
        label = "good"
    return {
        "label": label, "icon": QUALITY_LABELS[label], "color": QUALITY_COLORS[label],
        "matched": matched,
        "engine_pair": [list(c) for c in engine_line],
        "played_pair": [list(c) for c in played],
        "loss": _finite(loss),
        "player_end_q": player_end_q, "engine_end_q": engine_end_q,
        "turn_start_depth": a,
    }


def _build_prefix_states(moves, cfg):
    """Replay ``moves`` into prefix states. Returns (prefix_states, error_at).

    ``prefix_states[0]`` is the freshly-seeded game (P1 at origin). A leading
    ``(0,0)`` in ``moves`` is the engine seed and is skipped (mirrors build_state).
    Shared by analyze_trajectory and analyze_game_full.
    """
    import hexo_rs as hr
    state = hr.GameState(cfg)  # auto-seeds P1 at (0,0)
    prefix_states = [state.clone()]  # prefix 0: the seed alone
    error_at = None
    for i in range(len(moves)):
        q, r = moves[i]
        if i == 0 and q == 0 and r == 0:
            continue
        try:
            state.apply_move(int(q), int(r))
        except Exception as e:
            error_at = {"index": i, "message": str(e)}
            break
        prefix_states.append(state.clone())
    return prefix_states, error_at


def _current_player_at(st, i, prefix_states):
    """current_player at prefix ``i``, deriving it for terminal positions.

    The engine reports ``None`` for terminal states; derive the side-to-move
    (the player who would move next) as the opponent of whoever just moved.
    """
    cp = st.current_player()
    if cp is not None:
        return cp
    if i == 0:
        return "P2"  # only the P1 seed; cannot be terminal for win>=2
    mover = prefix_states[i - 1].current_player()
    return "P2" if mover == "P1" else ("P1" if mover else "P2")


def analyze_trajectory(model, model_config, moves: list[tuple[int, int]],
                       win_length: int, placement_radius: int, max_moves: int):
    """Batched raw-policy value over every prefix of ``moves``.

    Returns one entry per prefix length 1..len(moves). Terminal prefixes get
    exact +1/-1/0; non-terminal prefixes are evaluated with one ``forward_batch``.
    """
    import hexo_rs as hr
    graph_fn = make_graph_fn(model_config)
    cfg = hr.GameConfig(win_length, placement_radius, max_moves)
    prefix_states, error_at = _build_prefix_states(moves, cfg)

    non_terminal_idx = [i for i, s in enumerate(prefix_states) if not s.is_terminal()]
    non_term_values: dict[int, float] = {}
    if non_terminal_idx:
        graphs = [graph_fn(prefix_states[i]) for i in non_terminal_idx]
        try:
            import torch
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(graphs).to("cpu")
            with torch.inference_mode():
                _logits_list, values = model.forward_batch(batch)
            for idx, v in zip(non_terminal_idx, values):
                non_term_values[idx] = float(v.item() if hasattr(v, "item") else v)
        except Exception:
            # Per-graph fallback: one bad graph must not sink the whole
            # trajectory, so evaluate each independently and skip failures.
            for idx, d in zip(non_terminal_idx, graphs):
                try:
                    with torch.no_grad():
                        _logits, val = model(
                            d.x, d.edge_index, d.legal_mask,
                            stone_mask=d.stone_mask,
                            edge_attr=getattr(d, "edge_attr", None),
                        )
                    non_term_values[idx] = float(val.item())
                except Exception:
                    non_term_values[idx] = 0.0

    trajectory = []
    for i, st in enumerate(prefix_states):
        cp = _current_player_at(st, i, prefix_states)
        if st.is_terminal():
            winner = st.winner()
            if winner is None:
                val = 0.0
            elif winner == cp:
                val = 1.0
            else:
                val = -1.0
            trajectory.append({"value": _finite(val), "current_player": cp,
                               "terminal": True, "winner": winner})
        else:
            # Sanitize: a NaN/Inf value-head output would otherwise emit a
            # bare `NaN`/`Infinity` token (invalid RFC 8259 JSON) and break the
            # browser analysis view.
            trajectory.append({"value": _finite(non_term_values.get(i, 0.0)),
                               "current_player": cp,
                               "terminal": False, "winner": None})

    boundary_indices = end_of_turn_indices([t["current_player"] for t in trajectory])
    out = {
        "trajectory": trajectory,
        "boundary_indices": boundary_indices,
        "evaluated_prefixes": len(prefix_states),
    }
    if error_at is not None:
        out["error_at"] = error_at
    return out


def analyze_game_full(model, model_config, moves, win_length, placement_radius,
                      max_moves, *, mcts_sims=64, mcts_m_actions=16,
                      progress_cb=None, cancel_cb=None, run_mcts_fn=None,
                      warm_trajectory=None):
    """Full per-position analysis over every prefix of ``moves``.

    Two passes:
      1. ONE batched ``forward_batch`` over all non-terminal prefixes gives every
         position value + raw policy + legal (cheap, instant).
      2. A loop of full Gumbel MCTS per non-terminal prefix (via ``run_mcts_fn``)
         upgrades each to q_hat / improved_policy / candidate_set.
         ``cancel_cb()`` is checked between prefixes so a disconnected client
         stops early; ``progress_cb(done, total)`` fires after each prefix.

    ``run_mcts_fn`` defaults to ``analyze_position``; the SSE handler injects a
    guard-wrapped version so the InferenceGuard is acquired/released PER prefix
    (not held for the whole job), letting bot moves interleave.

    ``warm_trajectory`` (optional) is the trajectory of a previously-analysed
    line that is a PREFIX of ``moves``; its per-position MCTS results are reused
    for the shared head so only the appended tail runs MCTS (warm-start).

    Returns the extended trajectory shape: per-prefix
    {value, current_player, terminal, winner, legal, probs, q_hat,
     improved_policy, candidate_set, stones, threats, forcing, quality} (MCTS
     fields only for non-terminal prefixes that completed; quality only at
     turn boundaries), plus boundary_indices, evaluated_prefixes, mcts_sims,
    cancelled, completed_prefixes, error_at.
    """
    import hexo_rs as hr
    graph_fn = make_graph_fn(model_config)
    cfg = hr.GameConfig(win_length, placement_radius, max_moves)
    prefix_states, error_at = _build_prefix_states(moves, cfg)
    total = len(prefix_states)

    if run_mcts_fn is None:
        def run_mcts_fn(st):
            # Trajectory solves run up to twice per placement across the
            # whole game, so they use the tighter TRAJECTORY_* budget, not
            # ANALYSIS_* (the single-position /analyze default) — see the
            # consts' bench.
            return analyze_position(model, model_config, st,
                                    mcts_sims=mcts_sims, mcts_m_actions=mcts_m_actions,
                                    forcing_depth_cap=TRAJECTORY_DEPTH_CAP,
                                    forcing_node_budget=TRAJECTORY_NODE_BUDGET)

    # Pass 1: batched value + raw policy + legal over all non-terminal prefixes.
    non_terminal_idx = [i for i, s in enumerate(prefix_states) if not s.is_terminal()]
    raw_by_idx = {}  # idx -> {value, legal, probs}
    if non_terminal_idx:
        graphs = [graph_fn(prefix_states[i]) for i in non_terminal_idx]
        try:
            import torch
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(graphs).to("cpu")
            with torch.inference_mode():
                logits_list, values = model.forward_batch(batch)
            for idx, d, lg, v in zip(non_terminal_idx, graphs, logits_list, values):
                legal = d.coords[d.legal_mask].tolist()
                probs_t = torch.softmax(lg, dim=-1)
                if probs_t.dim() > 1:
                    probs_t = probs_t.squeeze(-1)
                if probs_t.shape[0] == d.x.shape[0]:
                    probs = probs_t[d.legal_mask].tolist()
                else:
                    probs = probs_t.tolist()
                raw_by_idx[idx] = {
                    "value": float(v.item() if hasattr(v, "item") else v),
                    "legal": [list(c) for c in legal],
                    "probs": [_finite(p) for p in probs],
                }
        except Exception:
            # Per-graph fallback (one bad graph must not sink the batch).
            for idx, d in zip(non_terminal_idx, graphs):
                try:
                    with torch.no_grad():
                        lg, val = model(
                            d.x, d.edge_index, d.legal_mask,
                            stone_mask=d.stone_mask,
                            edge_attr=getattr(d, "edge_attr", None),
                        )
                    legal = d.coords[d.legal_mask].tolist()
                    probs_t = torch.softmax(lg, dim=-1)
                    if probs_t.dim() > 1:
                        probs_t = probs_t.squeeze(-1)
                    if probs_t.shape[0] == d.x.shape[0]:
                        probs = probs_t[d.legal_mask].tolist()
                    else:
                        probs = probs_t.tolist()
                    raw_by_idx[idx] = {
                        "value": float(val.item()),
                        "legal": [list(c) for c in legal],
                        "probs": [_finite(p) for p in probs],
                    }
                except Exception:
                    raw_by_idx[idx] = {"value": 0.0, "legal": [], "probs": []}

    # Build the base trajectory (value + raw policy + stones for every prefix).
    trajectory = []
    for i, st in enumerate(prefix_states):
        cp = _current_player_at(st, i, prefix_states)
        entry = {"current_player": cp, "terminal": bool(st.is_terminal())}
        if st.is_terminal():
            winner = st.winner()
            val = 0.0 if winner is None else (1.0 if winner == cp else -1.0)
            # Carry stones (and an empty legal list) for terminal prefixes too, so
            # the final winning position renders instead of blanking the board —
            # this mirrors analyze_position's terminal shape (the /analyze path).
            entry.update({
                "value": _finite(val), "winner": winner,
                "legal": [], "stones": _stones_of(st),
            })
        else:
            raw = raw_by_idx.get(i, {"value": 0.0, "legal": [], "probs": []})
            entry.update({
                "value": _finite(raw["value"]), "winner": None,
                "legal": raw["legal"], "probs": raw["probs"],
                "stones": _stones_of(st),
            })
        trajectory.append(entry)

    boundary_indices = end_of_turn_indices([t["current_player"] for t in trajectory])

    # Pass 2: full MCTS per non-terminal prefix (cancellable, with progress).
    # ``warm_trajectory`` is the per-position MCTS result of a previously-cached
    # line that is a PREFIX of ``moves`` (e.g. the same game before it was
    # extended). Because position i depends only on moves[0..i], those shared
    # prefix positions are identical, so we copy their MCTS fields and skip the
    # (expensive) search entirely — only the newly-appended tail runs MCTS.
    warm = warm_trajectory or []
    _MCTS_FIELDS = ("q_hat", "improved_policy", "candidate_set", "legal",
                    "probs", "value", "threats", "forcing")
    cancelled = False
    completed = 0
    for i, st in enumerate(prefix_states):
        if not st.is_terminal():
            wi = warm[i] if i < len(warm) else None
            # forcing is independent of MCTS (computed even at mcts_sims=0), so
            # a warm entry that carries one is reusable regardless of whether
            # q_hat/MCTS diagnostics were also cached for this position.
            if wi is not None and "forcing" in wi:
                trajectory[i]["forcing"] = wi["forcing"]
            if wi is not None and wi.get("q_hat") is not None:
                # Reuse the cached MCTS for this shared-prefix position.
                for k in _MCTS_FIELDS:
                    if k in wi:
                        trajectory[i][k] = wi[k]
            else:
                # Only the uncached tail incurs MCTS cost; bail if the client left.
                if cancel_cb and cancel_cb():
                    cancelled = True
                    break
                try:
                    res = run_mcts_fn(st)
                    js = res.to_json() if hasattr(res, "to_json") else res
                    # forcing is independent of q_hat/MCTS success — always set
                    # it (a fresh compute at mcts_sims=0 still has one to report).
                    trajectory[i]["forcing"] = js.get("forcing")
                    if js.get("q_hat") is not None:
                        trajectory[i]["q_hat"] = js["q_hat"]
                        trajectory[i]["improved_policy"] = js.get("improved_policy")
                        trajectory[i]["candidate_set"] = js.get("candidate_set")
                        # MCTS legal/probs/value are the authoritative per-move data.
                        trajectory[i]["legal"] = js.get("legal", trajectory[i].get("legal"))
                        if js.get("probs") is not None:
                            trajectory[i]["probs"] = js["probs"]
                        trajectory[i]["value"] = _finite(js.get("value", trajectory[i]["value"]))
                        if js.get("threats") is not None:
                            trajectory[i]["threats"] = js["threats"]
                except Exception:
                    pass  # leave the raw-policy entry in place
        completed = i + 1
        if progress_cb:
            progress_cb(completed, total)

    # Off-line evaluator for the engine's SECOND placement: when the engine's top
    # pick for a turn wasn't what the player played first, its second pick lies off
    # the observed game line, so we search the position after the engine's own first
    # pick. Cached per (prefix_index, first_move) and skipped once the client leaves.
    _after_cache = {}

    def eval_after_fn(prefix_index, first_move):
        if cancelled:
            return None
        key = (prefix_index, tuple(first_move))
        if key in _after_cache:
            return _after_cache[key]
        js = None
        try:
            st = prefix_states[prefix_index].clone()
            st.apply_move(int(first_move[0]), int(first_move[1]))
            if not st.is_terminal():
                res = run_mcts_fn(st)
                js = res.to_json() if hasattr(res, "to_json") else res
        except Exception:
            js = None
        _after_cache[key] = js
        return js

    # Pre-compute move-quality at turn boundaries. Order-free: the turn is judged
    # by its placed-set vs the engine's best placed-set (see classify_turn_quality).
    for bi, b in enumerate(boundary_indices):
        if b == 0:
            continue
        a = boundary_indices[bi - 1] if bi > 0 else 0
        # Reuse a shared-head verdict verbatim: a turn fully inside the warm prefix
        # has identical positions, so its verdict (and any off-line evals it needed)
        # is unchanged — don't recompute it when extending a cached line.
        if b < len(warm) and warm[b].get("quality") is not None:
            trajectory[b]["quality"] = warm[b]["quality"]
            continue
        q = classify_turn_quality(trajectory, boundary_indices, moves,
                                  turn_end_idx=b, prev_boundary_idx=a,
                                  eval_after_fn=eval_after_fn)
        if q is not None:
            trajectory[b]["quality"] = q

    # Missed-win walk: pure post-processing over the trajectory, no new
    # solves (Task 3). For each non-terminal prefix i where the side to move
    # X had a proven forced win (trajectory[i]["forcing"] with
    # attacker_is_mover == True), check whether the move actually played from
    # i abandoned it, and if so flag the RESULTING placement
    # (trajectory[i + 1], matching classify_turn_quality's moves[i+1]
    # convention) with a ``missed_win`` dict carrying the squandered line.
    #
    # Deviating from the proven `first_move` is not itself damning: X might
    # play a DIFFERENT winning move on the same (turn-ending) placement, or
    # keep the win alive as a threat into the opponent's reply. So rather than
    # judging the single next entry, forward-scan from i + 1 to the first
    # entry that can actually settle it:
    #   (a) a terminal entry — exempt iff its winner is X (a draw counts as a
    #       squandered win: draw is strictly worse than the win X held).
    #   (b) an entry where X is to move again (mid-turn continuation at
    #       i + 1 itself, or X's next turn after the opponent replies) —
    #       exempt iff that entry still has a proven win for X (its
    #       ``forcing`` has attacker_is_mover == True and winner == X).
    # An opponent-to-move entry with attacker_is_mover == False and
    # winner == X (X still holds the win as a THREAT, opponent to move) does
    # NOT exempt by itself — it only defers the verdict to X's next
    # to-move entry, which the scan reaches next. If the scan runs off the
    # end of the trajectory without hitting (a) or (b) (game record ends
    # non-terminal before X moves again), there isn't enough information to
    # call it squandered — don't flag (conservative).
    #
    # Recomputed over the FULL merged trajectory on every call — warm_trajectory
    # reuse above only ever copies MCTS/forcing/quality fields, never
    # missed_win, but strip any that leaked in from an earlier response
    # anyway (defensive: a stale flag would be wrong the moment a later
    # request's forcing solve disagrees with an earlier one, and this walk is
    # cheap enough that recomputing it is free).
    #
    # Budget note: the ``forcing`` fields this walk reads were solved at the
    # tight TRAJECTORY_* budgets (far below live play's), so a win beyond
    # that budget just never flags here (conservative, fine). The converse
    # bit players in the field (chao, 2026-07-04): an ALTERNATIVE winning
    # move can fail to re-prove at TRAJECTORY budget (the solver's own line
    # move maximizes forcing pressure; an equally-winning human move may
    # enter a wider tree), producing a false "Missed win!". So before
    # flagging, the scan's landing entry is re-solved once at the full
    # ANALYSIS_* budget — would-flag events are a handful per game at most,
    # so the cost stays bounded — and a proven kept win both exempts and is
    # back-filled into that entry's ``forcing`` for display consistency.
    for entry in trajectory:
        entry.pop("missed_win", None)
    for i in range(len(trajectory) - 1):
        forcing = trajectory[i].get("forcing")
        if not forcing or not forcing.get("attacker_is_mover"):
            continue
        if i + 1 >= len(moves) or len(moves[i + 1]) < 2:
            continue  # bounds guard: no placement follows this prefix
        played = tuple(moves[i + 1][:2])
        if played == tuple(forcing["first_move"]):
            continue  # on the proven line
        mover = forcing["winner"]
        exempt = None  # None = scan ran off the end -> conservative, no flag
        for j in range(i + 1, len(trajectory)):
            entry = trajectory[j]
            if entry.get("terminal"):
                exempt = entry.get("winner") == mover
                break
            if entry.get("current_player") == mover:
                nf = entry.get("forcing")
                exempt = bool(nf and nf.get("attacker_is_mover")
                              and nf.get("winner") == mover)
                if not exempt and j < len(prefix_states):
                    # Escalated exemption check (see budget note above): the
                    # per-prefix TRAJECTORY-budget solve may have missed an
                    # alternative kept win — re-solve this one position at
                    # the full ANALYSIS budget before daring to flag.
                    escalated = _solve_forcing(
                        prefix_states[j],
                        depth_cap=ANALYSIS_DEPTH_CAP,
                        node_budget=ANALYSIS_NODE_BUDGET,
                    )
                    if (escalated and escalated.get("attacker_is_mover")
                            and escalated.get("winner") == mover):
                        entry["forcing"] = escalated
                        exempt = True
                break
        if exempt is None or exempt:
            continue  # inconclusive, or X kept/reclaimed the win — not missed
        trajectory[i + 1]["missed_win"] = {
            "by": mover,
            "at_prefix": i,
            "first_move": forcing["first_move"],
            "pv": forcing["pv"],
            "pv_len": forcing["pv_len"],
            "pv_owners": forcing.get("pv_owners"),
        }

    out = {
        "trajectory": trajectory,
        "boundary_indices": boundary_indices,
        "evaluated_prefixes": total,
        "mcts_sims": mcts_sims,
        "win_length": win_length,
        "cancelled": cancelled,
        "completed_prefixes": completed,
    }
    if error_at is not None:
        out["error_at"] = error_at
    return out
