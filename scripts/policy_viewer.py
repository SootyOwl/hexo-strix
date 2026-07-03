"""Interactive hex board for visualising model policy + value.

Usage:
    uv run python scripts/policy_viewer.py [--config configs/ablations/baseline.toml] [--port 8765]

Opens a browser tab. Click hexes to place stones (alternating P1/P2 per HeXO
rules). Select a checkpoint, adjust game params, hit Analyze to see the policy
heatmap and value head output overlaid on the board.
"""
import argparse
import json
import struct
import sys
import tomllib
import webbrowser
from functools import lru_cache
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

# Lazy imports to keep startup fast
_hexo_rs = None
_model_mod = None
_graph_mod = None
_config_mod = None


def _ensure_imports():
    global _hexo_rs, _model_mod, _graph_mod, _config_mod
    if _hexo_rs is None:
        import hexo_rs as hr
        from hexo_a0 import model as mm, graph as gm, config as cm
        _hexo_rs = hr
        _model_mod = mm
        _graph_mod = gm
        _config_mod = cm


_model_config_dict: dict = {}  # Set from parsed TOML at startup


@lru_cache(maxsize=4)
def _load_model(ckpt_path: str):
    _ensure_imports()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Prefer the checkpoint's embedded model_config (authoritative for graph
    # flags like threat_features/relative_stone_encoding); fall back to TOML.
    mc = _config_mod.model_config_from_checkpoint(ckpt, _model_config_dict)
    model = _model_mod.HeXONet(mc)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, mc


def _make_graph_fn(mc):
    """Per-state graph builder threading all graph flags from the model config."""
    prune = getattr(mc, "prune_empty_edges", False)
    threat = getattr(mc, "threat_features", False)
    rel = getattr(mc, "relative_stone_encoding", False)
    if mc.graph_type == "axis":
        return lambda g: _graph_mod.game_to_axis_graph(
            g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
    return lambda g: _graph_mod.game_to_graph(
        g, threat_features=threat, relative_stones=rel)


def _build_state(moves, win_length, placement_radius, max_moves,
                 setup_stones=None, setup_current_player="P2", setup_moves_remaining=2):
    """Build a GameState. Returns (state, GameConfig).

    Two modes:
      - Default: replay `moves` (list of [q,r]) via apply_move. Enforces turn
        order and placement radius.
      - Setup:   if `setup_stones` (list of [q, r, "P1"|"P2"]) is provided,
        construct the state directly via GameState.from_state, bypassing
        turn/radius rules. Used for puzzle / arbitrary-position analysis.
    """
    _ensure_imports()
    cfg = _hexo_rs.GameConfig(win_length, placement_radius, max_moves)
    if setup_stones is not None:
        typed = [((int(s[0]), int(s[1])), str(s[2])) for s in setup_stones]
        state = _hexo_rs.GameState.from_state(
            typed, str(setup_current_player), int(setup_moves_remaining), cfg,
        )
        return state, cfg
    state = _hexo_rs.GameState(cfg)
    for i, (q, r) in enumerate(moves):
        if i == 0 and q == 0 and r == 0:
            continue
        try:
            state.apply_move(q, r)
        except Exception as e:
            raise ValueError(f"Move {i} ({q},{r}): {e} — stones so far: {state.placed_stones()}") from e
    return state, cfg


def _run_mcts(model, mc, state, mcts_sims, mcts_m_actions):
    """Run Gumbel MCTS and return (improved_policy, value, legal_coords,
    visit_counts, per_child_q, candidate_indices).

    `visit_counts` is the root-node child visit count per legal move, in the
    same order as `improved_policy`.

    `per_child_q` is Q(root, action) from the root player's perspective —
    visited children use empirical Q, unvisited use the v_mix fallback. This is
    the meaningful per-move *evaluation* to display: visit counts under Gumbel +
    Sequential Halving reflect the halving bracket, not move quality, and the
    improved policy is near-one-hot by design (σ-amplified at c_visit=50).

    `candidate_indices` are the Gumbel-Top-K survivors that actually received
    search budget (empirical Q); all other entries are the prior-value fallback.
    See `gumbel_mcts_with_diagnostics` in the Rust bindings.
    """
    _ensure_imports()
    from torch_geometric.data import Batch

    graph_fn = _make_graph_fn(mc)

    mcts_config = _hexo_rs.MCTSConfig(
        n_simulations=mcts_sims,
        m_actions=mcts_m_actions,
        c_visit=50,
        c_scale=1.0,
        # Deterministic, paper-style eval: no Gumbel noise, so the displayed
        # search (action + improved policy) is reproducible across reloads.
        disable_gumbel_noise=True,
    )

    def eval_fn(states):
        data_list = [graph_fn(s) for s in states]
        batch = Batch.from_data_list(data_list).to("cpu")
        with torch.inference_mode():
            try:
                policy_logits_list, values = model.forward_batch(batch)
            except Exception as e:
                # Fallback: evaluate one at a time
                policy_logits_list = []
                values_list = []
                for d in data_list:
                    with torch.no_grad():
                        logits, val = model(
                            d.x, d.edge_index, d.legal_mask,
                            stone_mask=d.stone_mask,
                            edge_attr=getattr(d, "edge_attr", None),
                        )
                    policy_logits_list.append(logits.tolist())
                    values_list.append(float(val.item()))
                return (policy_logits_list, values_list)
        logits_list = [p.tolist() for p in policy_logits_list]
        values_list = [float(v.item()) for v in values]
        return (logits_list, values_list)

    (_action, improved_policy, visit_counts, per_child_q,
     _per_child_prior, candidate_indices,
     _forced) = _hexo_rs.gumbel_mcts_with_diagnostics(
        state, eval_fn, mcts_config)
    legal_moves = state.legal_moves()

    # Get value from raw network for the root position
    data = graph_fn(state)
    with torch.no_grad():
        _, value = model(
            data.x, data.edge_index, data.legal_mask,
            stone_mask=data.stone_mask,
            edge_attr=getattr(data, "edge_attr", None),
        )

    return (improved_policy, value.item(), [list(c) for c in legal_moves],
            list(visit_counts), list(per_child_q), list(candidate_indices))


def _analyze(moves, checkpoint, win_length, placement_radius, max_moves,
             mcts_sims=0, mcts_m_actions=16,
             setup_stones=None, setup_current_player="P2", setup_moves_remaining=2):
    _ensure_imports()
    model, mc = _load_model(checkpoint)
    state, cfg = _build_state(
        moves, win_length, placement_radius, max_moves,
        setup_stones=setup_stones,
        setup_current_player=setup_current_player,
        setup_moves_remaining=setup_moves_remaining,
    )

    if state.is_terminal():
        winner = state.winner()
        return {
            "legal": [], "probs": [], "logits": [], "value": 0.0,
            "stones": state.placed_stones(),
            "entropy": 0.0, "max_entropy": 0.0,
            "threats": [], "graph_edges": [], "node_info": [],
            "terminal": True, "winner": winner,
        }

    data = _make_graph_fn(mc)(state)

    # Always get raw logits + value from the network
    with torch.no_grad():
        raw_logits, raw_value = model(
            data.x, data.edge_index, data.legal_mask,
            stone_mask=data.stone_mask,
            edge_attr=getattr(data, "edge_attr", None),
        )
    logits_list = raw_logits.tolist()          # list[float]
    value = float(raw_value.item())            # float
    legal_coords = data.coords[data.legal_mask].tolist()  # list[[q,r]]

    q_hat = None       # per-legal-move Q(root, action), root-player perspective
    candidate_set = None  # bool per legal move: received search budget
    improved_pol = None   # σ-amplified improved policy; argmax = greedy played move
    if mcts_sims > 0 and not state.is_terminal():
        # MCTS root visit counts replace raw softmax for the heatmap. We ALSO
        # surface per_child_q (q_hat) — the meaningful per-move evaluation to
        # display, since visit counts under Sequential Halving reflect the
        # halving bracket and the improved policy is near-one-hot by design.
        (improved_policy, mcts_value, _coords, visit_counts, per_child_q,
         _candidate_indices) = _run_mcts(
            model, mc, state, mcts_sims, mcts_m_actions)
        total_visits = sum(visit_counts)
        if total_visits > 0:
            probs = [v / total_visits for v in visit_counts]
        else:
            # Pathological: zero visits everywhere (shouldn't happen with
            # n_simulations > 0). Fall back to uniform over legals.
            n = len(visit_counts) or 1
            probs = [1.0 / n] * n
        value = mcts_value                     # float
        q_hat = [round(float(q), 4) for q in per_child_q]
        improved_pol = [round(float(p), 6) for p in improved_policy]
        # "Searched" = actually VISITED (visit_count > 0), NOT Gumbel-Top-K
        # membership. Only visited moves carry an empirical Q; unvisited ones
        # fall back to v_mix (the value-head estimate), which in a won position
        # is ~+1 for EVERY move and falsely paints the whole board as winning.
        # The forced-win shortcut returns a one-hot policy with a single visited
        # move, so gating on visits correctly shows only that move.
        candidate_set = [vc > 0 for vc in visit_counts]
        probs_t = torch.tensor(probs, dtype=torch.float32)
        log_probs_t = torch.log(probs_t + 1e-8)
        entropy = float(-(probs_t * log_probs_t).sum().item() / 0.6931)
    else:
        probs_t = torch.softmax(raw_logits, dim=-1)
        probs = probs_t.tolist()               # list[float]
        log_probs_t = torch.log_softmax(raw_logits, dim=-1)
        entropy = float(-(probs_t * log_probs_t).sum().item() / 0.6931)

    n_legal = len(legal_coords)
    max_entropy = float(torch.tensor(max(n_legal, 1)).float().log2().item())

    # Detect threats: 3+ in a row with open extension on any axis
    AXES = [(1, 0), (0, 1), (1, -1)]
    stone_coords = data.coords[data.stone_mask].tolist()
    features = data.x
    stone_set_p1 = set()
    stone_set_p2 = set()
    for i in range(data.x.shape[0]):
        if data.stone_mask[i]:
            c = tuple(data.coords[i].tolist())
            if features[i, 0] > 0.5:
                stone_set_p1.add(c)
            elif features[i, 1] > 0.5:
                stone_set_p2.add(c)
    legal_set = set(tuple(c) for c in legal_coords)

    threats = []
    for owner, stones in (("P1", stone_set_p1), ("P2", stone_set_p2)):
        for (q, r) in stones:
            for dq, dr in AXES:
                # Find consecutive run starting here
                length = 1
                while (q + length * dq, r + length * dr) in stones:
                    length += 1
                if length < 3:
                    continue
                # Only report from the "start" of the run
                prev = (q - dq, r - dr)
                if prev in stones:
                    continue
                # Check extensions
                ext_fwd = (q + length * dq, r + length * dr)
                ext_back = prev
                exts = []
                for ext in (ext_fwd, ext_back):
                    if ext in legal_set:
                        # Find prob for this coord
                        idx = next((i for i, c in enumerate(legal_coords) if tuple(c) == ext), None)
                        p = float(probs[idx]) if idx is not None else 0.0
                        exts.append({"coord": list(ext), "prob": round(p, 4)})
                if exts:
                    threats.append({
                        "owner": owner, "length": length,
                        "axis": [dq, dr],
                        "start": [q, r],
                        "extensions": exts,
                    })

    # Build graph edge data for visualization
    all_coords = data.coords.tolist()  # [[q,r], ...]
    ei = data.edge_index  # (2, E)
    ea = getattr(data, "edge_attr", None)
    n_nodes = data.x.shape[0]
    # Axis graphs have a dummy node as the last node — skip its edges
    has_dummy = ea is not None
    dummy_idx = n_nodes - 1 if has_dummy else -1
    graph_edges = []
    for e in range(ei.shape[1]):
        src_i, dst_i = int(ei[0, e].item()), int(ei[1, e].item())
        if src_i == dst_i:
            continue  # skip self-loops
        if src_i == dummy_idx or dst_i == dummy_idx:
            continue  # skip dummy node edges
        src_c = all_coords[src_i]
        dst_c = all_coords[dst_i]
        edge_info = {"src": src_c, "dst": dst_c}
        if ea is not None:
            a = ea[e].tolist()
            edge_info["axis"] = int(max(range(3), key=lambda k: a[k])) if max(a[:3]) > 0.5 else -1
            edge_info["dist"] = round(a[3], 2)
            edge_info["src_player"] = round(a[4])  # +1=P1, -1=P2, 0=empty
        graph_edges.append(edge_info)

    # Node info for hover ([0]/[1] = P1/P2 absolute, own/opp relative)
    _rel = getattr(mc, "relative_stone_encoding", False)
    a_lbl, b_lbl = ("own", "opp") if _rel else ("P1", "P2")
    node_info = []
    for i in range(data.x.shape[0]):
        f = data.x[i].tolist()
        c = all_coords[i]
        kind = a_lbl if f[0] > 0.5 else b_lbl if f[1] > 0.5 else "."
        node_info.append({"coord": c, "kind": kind, "stone": bool(data.stone_mask[i]),
                          "legal": bool(data.legal_mask[i])})

    return {
        "legal": legal_coords,
        "probs": probs,
        "q_hat": q_hat,
        "improved_policy": improved_pol,
        "candidate_set": candidate_set,
        "logits": logits_list,
        "value": value,
        "stones": state.placed_stones(),
        "entropy": round(entropy, 2),
        "max_entropy": round(max_entropy, 2),
        "threats": threats,
        "graph_edges": graph_edges,
        "node_info": node_info,
    }


def _analyze_trajectory(moves, checkpoint, win_length, placement_radius, max_moves):
    """Batched raw-policy value over every prefix of ``moves``.

    Returns one entry per prefix length 1..len(moves). For non-terminal prefixes
    the value comes from a single ``forward_batch`` over all prefixes; terminal
    prefixes get +1/-1/0 derived from the winner relative to the would-be
    current player. Values are from the current-player perspective at that
    prefix (the frontend flips to a fixed P1 reference for the bar).
    """
    _ensure_imports()
    from torch_geometric.data import Batch

    model, mc = _load_model(checkpoint)
    cfg = _hexo_rs.GameConfig(win_length, placement_radius, max_moves)
    graph_fn = _make_graph_fn(mc)

    # Build one state per prefix length 1..N. The seed at moves[0]=(0,0) is the
    # engine-placed P1 opening; constructing GameState already applies it.
    state = _hexo_rs.GameState(cfg)
    prefix_states = [state.clone()]
    error_at = None
    for i in range(1, len(moves)):
        q, r = moves[i]
        try:
            state.apply_move(int(q), int(r))
        except Exception as e:
            error_at = {"index": i, "message": str(e)}
            break
        prefix_states.append(state.clone())

    non_terminal_idx = [i for i, s in enumerate(prefix_states) if not s.is_terminal()]
    non_term_values: dict[int, float] = {}
    if non_terminal_idx:
        graphs = [graph_fn(prefix_states[i]) for i in non_terminal_idx]
        try:
            batch = Batch.from_data_list(graphs).to("cpu")
            with torch.inference_mode():
                _logits_list, values = model.forward_batch(batch)
            for idx, v in zip(non_terminal_idx, values):
                non_term_values[idx] = float(v.item() if hasattr(v, "item") else v)
        except Exception:
            # forward_batch can fail on edge-case batch shapes; fall back to per-state.
            for idx, d in zip(non_terminal_idx, graphs):
                with torch.no_grad():
                    _logits, val = model(
                        d.x, d.edge_index, d.legal_mask,
                        stone_mask=d.stone_mask,
                        edge_attr=getattr(d, "edge_attr", None),
                    )
                non_term_values[idx] = float(val.item())

    trajectory = []
    for i, st in enumerate(prefix_states):
        prefix_len = i + 1
        cp = st.current_player()
        if cp is None:
            # Terminal: pattern is P1 places (0,0); then P2 plays 2, P1 plays 2, alternating.
            cp = "P2" if ((prefix_len - 1) % 4) < 2 else "P1"
        if st.is_terminal():
            winner = st.winner()
            if winner is None:
                val = 0.0
            elif winner == cp:
                val = 1.0
            else:
                val = -1.0
            trajectory.append({"value": val, "current_player": cp,
                               "terminal": True, "winner": winner})
        else:
            trajectory.append({"value": non_term_values.get(i, 0.0),
                               "current_player": cp,
                               "terminal": False, "winner": None})

    out = {"trajectory": trajectory, "evaluated_prefixes": len(prefix_states)}
    if error_at is not None:
        out["error_at"] = error_at
    return out


def _ai_move(moves, checkpoint, win_length, placement_radius, max_moves,
             mcts_sims=0, mcts_m_actions=16,
             setup_stones=None, setup_current_player="P2", setup_moves_remaining=2):
    """Run model forward and return argmax(policy) as the AI's move."""
    _ensure_imports()
    model, mc = _load_model(checkpoint)
    state, cfg = _build_state(
        moves, win_length, placement_radius, max_moves,
        setup_stones=setup_stones,
        setup_current_player=setup_current_player,
        setup_moves_remaining=setup_moves_remaining,
    )

    if state.is_terminal():
        winner = state.winner()
        return {"terminal": True, "winner": winner}

    if mcts_sims > 0:
        improved_policy, value, legal_coords, _vc, _q, _ci = _run_mcts(
            model, mc, state, mcts_sims, mcts_m_actions)
        best_idx = max(range(len(improved_policy)), key=lambda i: improved_policy[i])
        best_coord = legal_coords[best_idx]
        best_prob = improved_policy[best_idx]
    else:
        data = _make_graph_fn(mc)(state)

        with torch.no_grad():
            logits, value_t = model(
                data.x, data.edge_index, data.legal_mask,
                stone_mask=data.stone_mask,
                edge_attr=getattr(data, "edge_attr", None),
            )
        probs = torch.softmax(logits, dim=-1)
        legal_coords = data.coords[data.legal_mask].tolist()
        value = value_t.item()
        best_idx = int(probs.argmax().item())
        best_coord = legal_coords[best_idx]
        best_prob = probs[best_idx].item()

    return {
        "move": best_coord,
        "value": value,
        "prob": best_prob,
        "terminal": False,
        "winner": None,
    }


def _list_checkpoints(ckpt_dir: str):
    d = Path(ckpt_dir)
    if not d.exists():
        return []
    pts = sorted(d.glob("checkpoint_*.pt"), key=lambda p: p.name)
    out = [str(p) for p in pts]
    champion = d / "self_play" / "champion.pt"
    if champion.exists():
        out.append(str(champion))
    return out


def _check_terminal(moves, win_length, placement_radius, max_moves):
    """Check if the position is terminal without loading a model."""
    _ensure_imports()
    state, _cfg = _build_state(moves, win_length, placement_radius, max_moves)
    if state.is_terminal():
        return {"terminal": True, "winner": state.winner()}
    return {"terminal": False, "winner": None}


_sealbot_loaded = False


def _ensure_sealbot_imports():
    global _sealbot_loaded
    if _sealbot_loaded:
        return
    from hexo_a0.sealbot_eval import _ensure_sealbot
    _ensure_sealbot()
    _sealbot_loaded = True


def _sealbot_move(moves, win_length, placement_radius, max_moves, time_limit):
    """Replay moves into a SealBot HexGame and ask SealBot for the next move(s).

    Returns a dict with either:
      - {"moves": [[q,r], ...], "terminal": False, "winner": None}
      - {"terminal": True, "winner": "P1"|"P2"|None}
    """
    _ensure_imports()
    _ensure_sealbot_imports()
    from game import HexGame, Player as SBPlayer
    from minimax_cpp import MinimaxBot as SBMinimaxBot

    game = HexGame(win_length=win_length)
    # Replay moves (including the auto-placed [0,0] at index 0)
    for (q, r) in moves:
        if game.game_over:
            break
        game.make_move(int(q), int(r))

    if game.game_over:
        winner = ("P1" if game.winner == SBPlayer.A else
                  "P2" if game.winner == SBPlayer.B else None)
        return {"terminal": True, "winner": winner}

    sealbot = SBMinimaxBot(time_limit=float(time_limit))
    result = sealbot.get_move(game)
    moves_to_apply = list(result) if sealbot.pair_moves else [result]
    out = []
    for m in moves_to_apply:
        if game.game_over:
            break
        q, r = int(m[0]), int(m[1])
        game.make_move(q, r)
        out.append([q, r])

    terminal = bool(game.game_over)
    winner = None
    if terminal:
        winner = ("P1" if game.winner == SBPlayer.A else
                  "P2" if game.winner == SBPlayer.B else None)
    return {"moves": out, "terminal": terminal, "winner": winner}


HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>HeXO Policy Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; display: flex; height: 100vh; }
#sidebar { width: 320px; padding: 16px; background: #16213e; overflow-y: auto; flex-shrink: 0; }
#sidebar h2 { margin-bottom: 12px; font-size: 18px; color: #00d4ff; }
#sidebar label { display: block; margin-top: 10px; font-size: 13px; color: #aaa; }
#sidebar select, #sidebar input { width: 100%; padding: 6px 8px; margin-top: 4px; background: #0f3460; border: 1px solid #333; color: #fff; border-radius: 4px; font-size: 13px; }
#sidebar button { margin-top: 14px; width: 100%; padding: 10px; background: #00d4ff; color: #1a1a2e; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 14px; }
#sidebar button:hover { background: #00b8d9; }
#info { margin-top: 16px; padding: 10px; background: #0f3460; border-radius: 6px; font-size: 13px; line-height: 1.6; }
#info .val { color: #00d4ff; font-weight: bold; }
#moves-list { margin-top: 10px; font-size: 12px; color: #888; max-height: 120px; overflow-y: auto; }
#board-container { flex: 1; overflow: hidden; position: relative; }
svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: grab; }
svg.panning { cursor: grabbing; }
.hex { stroke: #2a2a4a; stroke-width: 1.5; cursor: pointer; }
.hex:hover { stroke: #00d4ff; stroke-width: 2.5; }
.hex-empty { fill: #1e2748; }
.hex-p1 { fill: #e07020; }
.hex-p2 { fill: #2090d0; }
.hex-label { font-size: 10px; fill: #666; pointer-events: none; text-anchor: middle; dominant-baseline: middle; }
.hex-prob { font-size: 11px; fill: #fff; pointer-events: none; text-anchor: middle; dominant-baseline: middle; font-weight: bold; }
.top-marker { stroke: #fff; stroke-width: 2; fill: none; pointer-events: none; }
.edge-label { font-size: 12px; fill: #fff; pointer-events: none; text-anchor: middle; dominant-baseline: middle; font-weight: bold; }
#controls-row { display: flex; gap: 8px; margin-top: 10px; }
#controls-row button { flex: 1; padding: 8px; font-size: 12px; }
#undo-btn { background: #e07020; }
#clear-btn { background: #cc3333; }
#edge-legend { margin-top: 8px; font-size: 11px; line-height: 1.8; }
</style></head>
<body>
<div id="sidebar">
  <h2>HeXO Policy Viewer</h2>
  <label>Checkpoint
    <input id="ckpt" type="text" list="ckpt-list" autocomplete="off"
           placeholder="step (e.g. 147600)">
    <datalist id="ckpt-list"></datalist>
    <div id="ckpt-champion-row" style="display:none;margin-top:4px;font-size:12px">
      <label style="display:flex;align-items:center;gap:6px;color:#aaa;margin-top:0">
        <input type="checkbox" id="ckpt-use-champion" onchange="updateChampionToggle('ckpt')" style="width:auto">
        <span>Use current champion (<span style="color:#ffcc44">★</span>)</span>
      </label>
    </div>
  </label>
  <label>Win length <input id="win" type="number" value="6" min="3" max="8"></label>
  <label>Placement radius <input id="radius" type="number" value="8" min="1" max="10"></label>
  <label>Max moves <input id="maxmoves" type="number" value="400" min="10" max="1000"></label>
  <div style="margin-top:8px;padding:8px;background:#0a1a30;border-radius:4px;font-size:12px">
    <label><input type="checkbox" id="use-mcts" style="width:auto"> Use MCTS (root visit counts)</label>
    <div id="mcts-params" style="display:none;margin-top:4px">
      <label style="font-size:11px">Sims <input id="mcts-sims" type="number" value="128" min="1" max="1000" style="width:60px"></label>
      <label style="font-size:11px;margin-left:8px">m <input id="mcts-m" type="number" value="16" min="1" max="128" style="width:50px"></label>
    </div>
  </div>
  <label><input type="checkbox" id="auto-analyze" style="width:auto"> Auto-analyze on click</label>
  <div style="margin-top:10px;display:flex;gap:4px">
    <button id="mode-policy" onclick="setMode('policy')" style="flex:1;padding:6px;font-size:12px">Policy</button>
    <button id="mode-graph" onclick="setMode('graph')" style="flex:1;padding:6px;font-size:12px;background:#555">Graph</button>
    <button id="mode-play" onclick="setMode('play')" style="flex:1;padding:6px;font-size:12px;background:#555">Play</button>
    <button id="mode-setup" onclick="setMode('setup')" style="flex:1;padding:6px;font-size:12px;background:#555">Setup</button>
  </div>
  <div id="setup-controls" style="display:none;margin-top:8px;border:2px solid #444;border-radius:6px;padding:8px">
    <div style="font-size:12px;color:#aaa;font-weight:bold;margin-bottom:4px">Setup brush</div>
    <div style="display:flex;gap:4px">
      <button onclick="setSetupBrush('P1')" id="brush-p1" style="flex:1;padding:5px;font-size:11px;background:#e07020;color:#fff;border:none;border-radius:3px;cursor:pointer">P1</button>
      <button onclick="setSetupBrush('P2')" id="brush-p2" style="flex:1;padding:5px;font-size:11px;background:#555;color:#ccc;border:none;border-radius:3px;cursor:pointer">P2</button>
      <button onclick="setSetupBrush('erase')" id="brush-erase" style="flex:1;padding:5px;font-size:11px;background:#555;color:#ccc;border:none;border-radius:3px;cursor:pointer">Erase</button>
    </div>
    <div style="font-size:12px;color:#aaa;margin-top:8px">To move</div>
    <div style="display:flex;gap:4px;margin-top:3px">
      <button onclick="setSetupTurn('P1')" id="setup-turn-p1" style="flex:1;padding:5px;font-size:11px;background:#555;color:#ccc;border:none;border-radius:3px;cursor:pointer">P1</button>
      <button onclick="setSetupTurn('P2')" id="setup-turn-p2" style="flex:1;padding:5px;font-size:11px;background:#2090d0;color:#fff;border:none;border-radius:3px;cursor:pointer">P2</button>
    </div>
    <div style="font-size:12px;color:#aaa;margin-top:8px">Moves remaining this turn</div>
    <div style="display:flex;gap:4px;margin-top:3px">
      <button onclick="setSetupMovesRemaining(1)" id="setup-mr-1" style="flex:1;padding:5px;font-size:11px;background:#555;color:#ccc;border:none;border-radius:3px;cursor:pointer">1</button>
      <button onclick="setSetupMovesRemaining(2)" id="setup-mr-2" style="flex:1;padding:5px;font-size:11px;background:#00d4ff;color:#1a1a2e;border:none;border-radius:3px;cursor:pointer">2</button>
    </div>
    <div style="display:flex;gap:4px;margin-top:8px">
      <button onclick="setupFromMoves()" style="flex:1;padding:5px;font-size:11px;background:#555;color:#ccc;border:none;border-radius:3px;cursor:pointer">Import played moves</button>
      <button onclick="clearSetup()" style="flex:1;padding:5px;font-size:11px;background:#cc3333;color:#fff;border:none;border-radius:3px;cursor:pointer">Clear stones</button>
    </div>
    <div style="font-size:11px;color:#888;margin-top:6px;line-height:1.4">
      Click hexes to place/replace stones. (0,0) is always P1 (engine seed).
    </div>
  </div>
  <div id="play-controls" style="display:none;margin-top:6px">
    <div id="p1-panel" style="border:2px solid #e07020;border-radius:6px;padding:8px;margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:13px;font-weight:bold;color:#e07020">Player 1 (orange)</span>
        <span id="p1-tomove" style="font-size:11px;font-weight:bold;color:#e07020;background:#e0702033;padding:2px 6px;border-radius:3px">TO MOVE</span>
      </div>
      <div style="font-size:11px;color:#aaa;margin-top:2px">Opens with 1 stone</div>
      <div style="display:flex;gap:4px;margin-top:6px">
        <button id="p1-human" onclick="setPlayerType('P1','human')" style="flex:1;padding:4px;font-size:11px;background:#e07020;color:#fff;border:none;border-radius:3px;cursor:pointer">Human</button>
        <button id="p1-bot" onclick="setPlayerType('P1','bot')" style="flex:1;padding:4px;font-size:11px;background:#555;color:#ccc;border:none;border-radius:3px;cursor:pointer">Bot</button>
        <button id="p1-sealbot" onclick="setPlayerType('P1','sealbot')" style="flex:1;padding:4px;font-size:11px;background:#555;color:#ccc;border:none;border-radius:3px;cursor:pointer">SealBot</button>
      </div>
      <div id="p1-ckpt-row" style="display:none;margin-top:4px">
        <label style="font-size:11px;color:#aaa;margin:0">Checkpoint
          <input id="p1-ckpt" type="text" list="p1-ckpt-list" autocomplete="off"
                 placeholder="step"
                 style="width:100%;padding:3px;font-size:11px;margin-top:2px">
          <datalist id="p1-ckpt-list"></datalist>
        </label>
        <div id="p1-ckpt-champion-row" style="display:none;margin-top:3px">
          <label style="display:flex;align-items:center;gap:4px;font-size:11px;color:#aaa;margin:0">
            <input type="checkbox" id="p1-ckpt-use-champion" onchange="updateChampionToggle('p1-ckpt')" style="width:auto">
            <span>Use champion (<span style="color:#ffcc44">★</span>)</span>
          </label>
        </div>
      </div>
      <div id="p1-mcts-row" style="display:none;margin-top:4px">
        <div style="display:flex;gap:6px;align-items:center">
          <label style="font-size:11px;color:#aaa;white-space:nowrap">Sims <input id="p1-sims" type="number" value="0" min="0" max="1000" style="width:50px;padding:2px;font-size:11px"></label>
          <label style="font-size:11px;color:#aaa;white-space:nowrap">m <input id="p1-m" type="number" value="16" min="1" max="128" style="width:40px;padding:2px;font-size:11px"></label>
          <span style="font-size:10px;color:#666">0 = raw</span>
        </div>
      </div>
      <div id="p1-sealbot-row" style="display:none;margin-top:4px">
        <div style="display:flex;gap:6px;align-items:center">
          <label style="font-size:11px;color:#aaa;white-space:nowrap">Time/move <input id="p1-time" type="number" value="0.5" min="0.01" max="60" step="0.1" style="width:60px;padding:2px;font-size:11px"></label>
          <span style="font-size:10px;color:#666">seconds</span>
        </div>
      </div>
    </div>
    <div id="p2-panel" style="border:2px solid #2090d0;border-radius:6px;padding:8px;margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:13px;font-weight:bold;color:#2090d0">Player 2 (blue)</span>
        <span id="p2-tomove" style="font-size:11px;font-weight:bold;color:#2090d0;background:#2090d033;padding:2px 6px;border-radius:3px;display:none">TO MOVE</span>
      </div>
      <div style="font-size:11px;color:#aaa;margin-top:2px">Responds with 2 stones</div>
      <div style="display:flex;gap:4px;margin-top:6px">
        <button id="p2-human" onclick="setPlayerType('P2','human')" style="flex:1;padding:4px;font-size:11px;background:#555;color:#ccc;border:none;border-radius:3px;cursor:pointer">Human</button>
        <button id="p2-bot" onclick="setPlayerType('P2','bot')" style="flex:1;padding:4px;font-size:11px;background:#2090d0;color:#fff;border:none;border-radius:3px;cursor:pointer">Bot</button>
        <button id="p2-sealbot" onclick="setPlayerType('P2','sealbot')" style="flex:1;padding:4px;font-size:11px;background:#555;color:#ccc;border:none;border-radius:3px;cursor:pointer">SealBot</button>
      </div>
      <div id="p2-ckpt-row" style="margin-top:4px">
        <label style="font-size:11px;color:#aaa;margin:0">Checkpoint
          <input id="p2-ckpt" type="text" list="p2-ckpt-list" autocomplete="off"
                 placeholder="step"
                 style="width:100%;padding:3px;font-size:11px;margin-top:2px">
          <datalist id="p2-ckpt-list"></datalist>
        </label>
        <div id="p2-ckpt-champion-row" style="display:none;margin-top:3px">
          <label style="display:flex;align-items:center;gap:4px;font-size:11px;color:#aaa;margin:0">
            <input type="checkbox" id="p2-ckpt-use-champion" onchange="updateChampionToggle('p2-ckpt')" style="width:auto">
            <span>Use champion (<span style="color:#ffcc44">★</span>)</span>
          </label>
        </div>
      </div>
      <div id="p2-mcts-row" style="margin-top:4px">
        <div style="display:flex;gap:6px;align-items:center">
          <label style="font-size:11px;color:#aaa;white-space:nowrap">Sims <input id="p2-sims" type="number" value="0" min="0" max="1000" style="width:50px;padding:2px;font-size:11px"></label>
          <label style="font-size:11px;color:#aaa;white-space:nowrap">m <input id="p2-m" type="number" value="16" min="1" max="128" style="width:40px;padding:2px;font-size:11px"></label>
          <span style="font-size:10px;color:#666">0 = raw</span>
        </div>
      </div>
      <div id="p2-sealbot-row" style="display:none;margin-top:4px">
        <div style="display:flex;gap:6px;align-items:center">
          <label style="font-size:11px;color:#aaa;white-space:nowrap">Time/move <input id="p2-time" type="number" value="0.5" min="0.01" max="60" step="0.1" style="width:60px;padding:2px;font-size:11px"></label>
          <span style="font-size:10px;color:#666">seconds</span>
        </div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
      <span style="font-size:11px;color:#aaa;white-space:nowrap">Move delay:</span>
      <input id="move-delay" type="number" value="500" min="0" max="5000" step="100" style="width:70px;padding:3px;font-size:11px">
      <span style="font-size:11px;color:#aaa">ms</span>
    </div>
    <div style="display:flex;gap:4px">
      <button id="new-game-btn" onclick="newGame()" style="flex:1;padding:6px;font-size:12px;background:#44aa44;border:none;border-radius:4px;color:#fff;cursor:pointer;font-weight:bold">New Game</button>
      <button id="pause-btn" onclick="togglePause()" style="flex:1;padding:6px;font-size:12px;background:#555;border:none;border-radius:4px;color:#ccc;cursor:pointer;font-weight:bold">Start</button>
    </div>
    <div id="play-status" style="margin-top:6px;padding:8px;background:#0f3460;border-radius:6px;font-size:12px;line-height:1.6"></div>
  </div>
  <div id="axis-filter" style="display:none;margin-top:6px">
    <span style="font-size:11px;color:#aaa">Axis filter:</span>
    <div style="display:flex;gap:3px;margin-top:3px">
      <button onclick="setAxisFilter(-1)" id="af-all" style="flex:1;padding:4px;font-size:11px;background:#00d4ff;color:#1a1a2e">All</button>
      <button onclick="setAxisFilter(0)" id="af-0" style="flex:1;padding:4px;font-size:11px;background:#555;color:#ff6b6b">(1,0)</button>
      <button onclick="setAxisFilter(1)" id="af-1" style="flex:1;padding:4px;font-size:11px;background:#555;color:#6bff6b">(0,1)</button>
      <button onclick="setAxisFilter(2)" id="af-2" style="flex:1;padding:4px;font-size:11px;background:#555;color:#6b6bff">(1,-1)</button>
    </div>
    <span style="font-size:11px;color:#aaa;margin-top:4px;display:block">Edge direction:</span>
    <div style="display:flex;gap:3px;margin-top:3px">
      <button onclick="setEdgeDir('both')" id="ed-both" style="flex:1;padding:4px;font-size:11px;background:#00d4ff;color:#1a1a2e">Both</button>
      <button onclick="setEdgeDir('in')" id="ed-in" style="flex:1;padding:4px;font-size:11px;background:#555;color:#ccc">In (receives)</button>
      <button onclick="setEdgeDir('out')" id="ed-out" style="flex:1;padding:4px;font-size:11px;background:#555;color:#ccc">Out (sends)</button>
    </div>
  </div>
  <button id="analyze-btn" onclick="analyze()">Analyze (Enter)</button>
  <div id="controls-row">
    <button id="undo-btn" onclick="undo()">Undo</button>
    <button id="clear-btn" onclick="clearBoard()">Clear</button>
  </div>
  <div style="display:flex;gap:4px;margin-top:6px">
    <button onclick="copyPosition()" style="flex:1;padding:6px;font-size:11px;background:#555;color:#ccc">Copy</button>
    <button onclick="pastePosition()" style="flex:1;padding:6px;font-size:11px;background:#555;color:#ccc">Paste</button>
  </div>
  <div id="replay-bar" style="display:none;position:relative;margin-top:6px;padding:6px;background:#1a2538;border-radius:4px">
    <canvas id="eval-bar-canvas" height="32"
            style="width:100%;height:32px;display:block;margin-bottom:4px;background:#0d1322;border-radius:2px;cursor:pointer"
            onmousemove="onEvalBarHover(event)"
            onmouseleave="hideEvalTooltip()"
            onclick="onEvalBarClick(event)"></canvas>
    <div id="eval-tooltip" style="display:none;position:absolute;pointer-events:none;background:#000a;color:#fff;font-size:11px;padding:2px 6px;border-radius:3px;white-space:nowrap;z-index:10;transform:translate(-50%,-100%);top:0"></div>
    <div style="display:flex;align-items:center;gap:3px">
      <button onclick="replayFirst()" title="First (Home)" style="padding:4px 6px;font-size:11px;background:#555;color:#ccc">⏮</button>
      <button onclick="replayPrev()" title="Previous (←)" style="padding:4px 6px;font-size:11px;background:#555;color:#ccc">◀</button>
      <input type="range" id="replay-slider" min="1" max="1" value="1"
             oninput="scrubTo(parseInt(this.value))"
             style="flex:1;min-width:0">
      <button onclick="replayNext()" title="Next (→)" style="padding:4px 6px;font-size:11px;background:#555;color:#ccc">▶</button>
      <button onclick="replayLast()" title="Last (End)" style="padding:4px 6px;font-size:11px;background:#555;color:#ccc">⏭</button>
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-top:3px">
      <div id="replay-counter" style="flex:1;font-size:11px;color:#aaa;text-align:center">Move 0 / 0</div>
      <button id="mainline-btn" onclick="returnToMainline()"
              style="display:none;padding:3px 8px;font-size:11px;background:#3a4a6a;color:#cce">↺ Mainline</button>
    </div>
    <div id="trajectory-status" style="display:none;font-size:10px;color:#888;text-align:center;margin-top:2px">Analysing game…</div>
  </div>
  <div id="info">Click hexes to place stones. P1 (orange) goes first with 1 stone, then each side places 2.<br><br>
    <span id="status">No analysis yet.</span>
  </div>
  <div id="threats-info"></div>
  <div id="moves-list"></div>
</div>
<div id="board-container">
  <svg id="board"></svg>
</div>
<script>
const S = 28; // hex size
const SQ3 = Math.sqrt(3);
let moves = [{q:0, r:0}]; // P1 auto-placed at origin by engine
let analysisResult = null;
let radius = 8, winLength = 6, maxMoves = 400; // default values, updated from sidebar inputs

// Pan/zoom state
let viewX = 0, viewY = 0, viewScale = 1;
let isPanning = false, panStartX = 0, panStartY = 0, panStartVX = 0, panStartVY = 0;
let wasDragging = false;
let selectedNode = null; // {q, r} of node whose edges to highlight
let viewMode = 'policy'; // 'policy' or 'graph'
const AXIS_COLORS = ['#ff6b6b', '#6bff6b', '#6b6bff']; // (1,0)=red, (0,1)=green, (1,-1)=blue
const AXIS_NAMES = ['(1,0)', '(0,1)', '(1,-1)'];

let axisFilter = -1; // -1 = all, 0/1/2 = specific axis
let edgeDir = 'both'; // 'both', 'in', 'out'

// --- Replay mode state ---
// replayMoves is the full game loaded via paste (includes (0,0) seed at [0]).
// While non-null, the slider scrubs through it: moves === replayMoves.slice(0, replayIndex).
// User-initiated moves (click/undo) keep replayMoves in sync, so the scrubber
// keeps working as you continue playing from a mid-game position.
let replayMoves = null;
let replayIndex = 0;
// Parallel to replayMoves: trajectory[i] = { value: float in [-1,1] from P1's
// perspective, currentPlayer: 'P1'|'P2' } for the position after replayMoves[i]
// has been played. Null entries are positions still pending (or that errored).
let replayEvalTrajectory = null;
let trajectoryGeneration = 0;  // bumped to cancel in-flight trajectory runs
// Snapshot of replayMoves captured at paste time. When the user branches by
// clicking a cell in policy mode, replayMoves diverges but this stays as the
// "original game" so it can be restored via the Mainline button.
let replayMainline = null;

// --- Setup mode state ---
let setupStones = [{q:0, r:0, player:'P1'}]; // (0,0)=P1 is the engine seed
let setupBrush = 'P1';
let setupTurn = 'P2';
let setupMovesRemaining = 2;

function setMode(m) {
  viewMode = m;
  selectedNode = null;
  for (const id of ['mode-policy', 'mode-graph', 'mode-play', 'mode-setup']) {
    const btn = document.getElementById(id);
    const active = id === 'mode-' + m;
    btn.style.background = active ? '#00d4ff' : '#555';
    btn.style.color = active ? '#1a1a2e' : '#ccc';
  }
  document.getElementById('axis-filter').style.display = m === 'graph' ? 'block' : 'none';
  document.getElementById('play-controls').style.display = m === 'play' ? 'block' : 'none';
  document.getElementById('setup-controls').style.display = m === 'setup' ? 'block' : 'none';
  if (m === 'play') {
    updatePlayStatus();
    if (isBotTurn() && !botLoopRunning) startBotLoop();
  }
  if (m === 'setup') {
    analysisResult = null; // analysis from previous mode no longer aligned
  }
  updateReplayUI();
  drawBoard();
}

function updateReplayUI() {
  const bar = document.getElementById('replay-bar');
  if (!bar) return;
  const show = (viewMode === 'policy' || viewMode === 'graph')
               && replayMoves !== null && replayMoves.length > 1;
  bar.style.display = show ? '' : 'none';
  if (!show) return;
  const slider = document.getElementById('replay-slider');
  slider.min = 1;
  slider.max = replayMoves.length;
  slider.value = replayIndex;
  // Counter displays played stones (excluding the engine-seeded (0,0)).
  const totalPlayed = replayMoves.length - 1;
  const curPlayed = replayIndex - 1;
  document.getElementById('replay-counter').textContent =
    `Move ${curPlayed} / ${totalPlayed}`;
  document.getElementById('mainline-btn').style.display = isBranched() ? '' : 'none';
  renderEvalBar();
}

function scrubTo(idx) {
  if (replayMoves === null) return;
  replayIndex = Math.max(1, Math.min(replayMoves.length, idx | 0));
  moves = replayMoves.slice(0, replayIndex);
  analysisResult = null;
  selectedNode = null;
  updateReplayUI();
  renderEvalBar();
  drawBoard();
  if (document.getElementById('auto-analyze').checked) analyze();
}
function replayFirst() { scrubTo(1); }
function replayPrev()  { scrubTo(replayIndex - 1); }
function replayNext()  { scrubTo(replayIndex + 1); }
function replayLast()  { scrubTo(replayMoves ? replayMoves.length : 1); }

function samePath(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].q !== b[i].q || a[i].r !== b[i].r) return false;
  }
  return true;
}

function isBranched() {
  return replayMainline !== null && replayMoves !== null
         && !samePath(replayMoves, replayMainline);
}

function returnToMainline() {
  if (replayMainline === null) return;
  replayMoves = replayMainline.slice();
  replayIndex = replayMoves.length;
  moves = replayMoves.slice();
  cancelReplayTrajectory();
  analysisResult = null;
  selectedNode = null;
  updateReplayUI();
  drawBoard();
  if (document.getElementById('auto-analyze').checked) analyze();
  computeReplayTrajectory();
}

// Hover / click on the eval bar canvas
function evalBarIndexFromEvent(ev) {
  if (replayEvalTrajectory === null || replayEvalTrajectory.length === 0) return -1;
  const canvas = ev.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const N = replayEvalTrajectory.length;
  if (N === 1) return 0;
  const frac = Math.max(0, Math.min(1, x / rect.width));
  return Math.round(frac * (N - 1));
}

function onEvalBarHover(ev) {
  const i = evalBarIndexFromEvent(ev);
  if (i < 0) { hideEvalTooltip(); return; }
  const e = replayEvalTrajectory[i];
  const tip = document.getElementById('eval-tooltip');
  const canvas = ev.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  let text;
  if (e === null) {
    text = `Move ${i}: …`;
  } else {
    const v = e.value;
    const vStr = v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3);
    const lean = v > 0.05 ? ' (P1 ahead)' : v < -0.05 ? ' (P2 ahead)' : '';
    text = `Move ${i}: ${vStr}${lean}`;
  }
  tip.textContent = text;
  tip.style.left = (canvas.offsetLeft + x) + 'px';
  tip.style.top  = (canvas.offsetTop) + 'px';
  tip.style.display = '';
}

function hideEvalTooltip() {
  const tip = document.getElementById('eval-tooltip');
  if (tip) tip.style.display = 'none';
}

function onEvalBarClick(ev) {
  const i = evalBarIndexFromEvent(ev);
  if (i < 0) return;
  // trajectory index i corresponds to slider value i+1
  scrubTo(i + 1);
}

// After prefix of length L (including the (0,0) seed), whose turn is it?
// Pattern: P1 places 1 stone (seed), then P2 places 2, P1 places 2, alternating.
//   L=1: P2 to move (2 stones); L=2: P2 to move (1 left);
//   L=3: P1 to move (2 stones); L=4: P1 to move (1 left); ...
function currentPlayerAtPrefix(L) {
  return (((L - 1) % 4) < 2) ? 'P2' : 'P1';
}

async function computeReplayTrajectory() {
  if (replayMoves === null || replayMoves.length < 2) return;
  const resolved = resolveCheckpointFor('ckpt');
  if (resolved.error || !resolved.path) return;
  const gen = ++trajectoryGeneration;
  const N = replayMoves.length;
  replayEvalTrajectory = new Array(N).fill(null);

  const body = {
    checkpoint: resolved.path,
    win_length: parseInt(document.getElementById('win').value),
    placement_radius: parseInt(document.getElementById('radius').value),
    max_moves: parseInt(document.getElementById('maxmoves').value),
    moves: replayMoves.map(m => [m.q, m.r]),
  };

  const status = document.getElementById('trajectory-status');
  status.style.display = '';
  status.textContent = `Analysing game…`;
  renderEvalBar();

  try {
    const resp = await fetch('/analyze_trajectory', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const result = await resp.json();
    if (gen !== trajectoryGeneration) return;
    if (result.error) {
      status.textContent = `Trajectory error: ${result.error}`;
      return;
    }
    const traj = result.trajectory || [];
    for (let i = 0; i < traj.length; i++) {
      // Only show end-of-turn values (after each player has placed their stones). 
      // The raw trajectory includes mid-turn values, which jump around and are less useful to display.
      // For mid-turn values just show the previous end-of-turn value (or null if none yet).
      const e = traj[i];
      if (e === null) continue;
      if (i > 0) {
        const prev = traj[i - 1];
        if (prev !== null && e.current_player === prev.current_player) {
          // Mid-turn: copy previous end-of-turn value.
          replayEvalTrajectory[i] = replayEvalTrajectory[i - 1];
          continue;
        }
      }
      // Server values are from current-player perspective; flip to fixed P1 reference.
      const p1Value = (e.current_player === 'P1') ? e.value : -e.value;
      replayEvalTrajectory[i] = { value: p1Value, currentPlayer: e.current_player };
    }
    renderEvalBar();
    if (result.error_at) {
      status.textContent = `Stopped at move ${result.error_at.index}: ${result.error_at.message}`;
    } else {
      status.style.display = 'none';
    }
  } catch (e) {
    if (gen === trajectoryGeneration) {
      status.textContent = `Trajectory error: ${e}`;
    }
  }
}

function cancelReplayTrajectory() {
  trajectoryGeneration++;
  replayEvalTrajectory = null;
  const status = document.getElementById('trajectory-status');
  if (status) status.style.display = 'none';
}

function renderEvalBar() {
  const canvas = document.getElementById('eval-bar-canvas');
  if (!canvas || canvas.offsetParent === null) return;  // hidden
  // Match canvas pixel buffer to its CSS size (slider is full-width).
  const cssW = canvas.clientWidth || canvas.parentElement.clientWidth;
  const cssH = canvas.clientHeight || 32;
  if (canvas.width !== cssW) canvas.width = cssW;
  if (canvas.height !== cssH) canvas.height = cssH;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  // Midline (value = 0)
  ctx.strokeStyle = '#2a3550';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();
  if (replayEvalTrajectory === null || replayEvalTrajectory.length === 0) return;
  const N = replayEvalTrajectory.length;
  // Filled area: P1-orange above midline, P2-blue below
  const colP1 = 'rgba(224,112,32,0.55)';   // matches stone-P1 hue
  const colP2 = 'rgba(32,144,208,0.55)';   // matches stone-P2 hue
  const stroke = '#e0e0e0';
  // Draw separate filled regions above/below by walking points
  function xAt(i) { return N === 1 ? W / 2 : (i / (N - 1)) * W; }
  function yAt(v) { return H / 2 - (v * (H / 2 - 1)); }
  // Plot END-OF-TURN states only. Each HeXO turn is two placements, so the
  // raw per-prefix trajectory jumps mid-turn; we keep only the settled value
  // after each completed turn. A boundary is where the player-to-move flips
  // (previous player just placed their 2nd stone). x still uses the full prefix
  // index so points stay aligned with the per-move scrubber. The current/last
  // prefix is always included so the live position shows.
  const plot = [];
  for (let i = 0; i < N; i++) {
    const e = replayEvalTrajectory[i];
    if (e === null) continue;
    const prev = i > 0 ? replayEvalTrajectory[i - 1] : null;
    const boundary = (i === 0) || (prev !== null && e.currentPlayer !== prev.currentPlayer);
    if (boundary || i === N - 1) plot.push(i);
  }
  // Filled regions between consecutive end-of-turn points.
  const mid = H / 2;
  for (let k = 0; k < plot.length - 1; k++) {
    const i = plot[k], j = plot[k + 1];
    const a = replayEvalTrajectory[i], b = replayEvalTrajectory[j];
    const x0 = xAt(i), x1 = xAt(j);
    const y0 = yAt(a.value), y1 = yAt(b.value);
    const avg = (a.value + b.value) / 2;
    ctx.fillStyle = avg >= 0 ? colP1 : colP2;
    ctx.beginPath();
    ctx.moveTo(x0, mid);
    ctx.lineTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1, mid);
    ctx.closePath();
    ctx.fill();
  }
  // Curve line on top, through the end-of-turn points.
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let k = 0; k < plot.length; k++) {
    const i = plot[k];
    const e = replayEvalTrajectory[i];
    const x = xAt(i), y = yAt(e.value);
    if (k === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // Current-position indicator
  if (replayIndex >= 1 && replayIndex <= N) {
    const x = xAt(replayIndex - 1);
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
}

function setSetupBrush(b) {
  setupBrush = b;
  for (const id of ['brush-p1', 'brush-p2', 'brush-erase']) {
    const btn = document.getElementById(id);
    const active = id === ('brush-' + (b === 'erase' ? 'erase' : b.toLowerCase()));
    let activeBg = '#555';
    if (active) activeBg = b === 'P1' ? '#e07020' : b === 'P2' ? '#2090d0' : '#cc3333';
    btn.style.background = activeBg;
    btn.style.color = active ? '#fff' : '#ccc';
  }
}

function setSetupTurn(p) {
  setupTurn = p;
  document.getElementById('setup-turn-p1').style.background = p === 'P1' ? '#e07020' : '#555';
  document.getElementById('setup-turn-p1').style.color = p === 'P1' ? '#fff' : '#ccc';
  document.getElementById('setup-turn-p2').style.background = p === 'P2' ? '#2090d0' : '#555';
  document.getElementById('setup-turn-p2').style.color = p === 'P2' ? '#fff' : '#ccc';
}

function setSetupMovesRemaining(n) {
  setupMovesRemaining = n;
  document.getElementById('setup-mr-1').style.background = n === 1 ? '#00d4ff' : '#555';
  document.getElementById('setup-mr-1').style.color = n === 1 ? '#1a1a2e' : '#ccc';
  document.getElementById('setup-mr-2').style.background = n === 2 ? '#00d4ff' : '#555';
  document.getElementById('setup-mr-2').style.color = n === 2 ? '#1a1a2e' : '#ccc';
}

function clearSetup() {
  // Always keep (0,0)=P1 (engine seeds it; from_state preserves it).
  setupStones = [{q:0, r:0, player:'P1'}];
  analysisResult = null;
  drawBoard();
}

function setupFromMoves() {
  // Convert the current played `moves` list into setup stones using
  // implicit player ordering (whoseStone). Lets the user start tweaking
  // from the position currently on the board.
  setupStones = moves.map((m, i) => ({q: m.q, r: m.r, player: whoseStone(i)}));
  analysisResult = null;
  // Reasonable default: whoever would play next in the played sequence.
  setupTurn = currentPlayer();
  setSetupTurn(setupTurn);
  drawBoard();
}

function setAxisFilter(ax) {
  axisFilter = ax;
  ['af-all','af-0','af-1','af-2'].forEach((id, i) => {
    document.getElementById(id).style.background = (ax === i - 1) ? '#00d4ff' : '#555';
  });
  drawBoard();
}

function setEdgeDir(dir) {
  edgeDir = dir;
  ['ed-both','ed-in','ed-out'].forEach(id => {
    document.getElementById(id).style.background = id === `ed-${dir}` ? '#00d4ff' : '#555';
    document.getElementById(id).style.color = id === `ed-${dir}` ? '#1a1a2e' : '#ccc';
  });
  drawBoard();
}

// HeXO turn logic: P1 places 1 stone first, then alternating 2 each.
// Stone index 0 = P1, index 1,2 = P2, index 3,4 = P1, ...
function whoseStone(idx) {
  if (idx === 0) return 'P1';
  return ((idx - 1) >> 1) % 2 === 0 ? 'P2' : 'P1';
}
function currentPlayer() { return whoseStone(moves.length); }

function axialToPixel(q, r) {
  return { x: S * (SQ3 * q + SQ3/2 * r), y: S * (3/2 * r) };
}
function hexCorners(cx, cy) {
  let pts = [];
  for (let i = 0; i < 6; i++) {
    let a = Math.PI / 180 * (60 * i - 30);
    pts.push(`${cx + S * Math.cos(a)},${cy + S * Math.sin(a)}`);
  }
  return pts.join(' ');
}
function hexDist(q1, r1, q2, r2) {
  if (q2 === undefined) { q2 = 0; r2 = 0; }
  const dq = q1-q2, dr = r1-r2;
  return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
}
function isLegalHex(q, r) {
  // Occupied?
  if (moves.some(m => m.q === q && m.r === r)) return false;
  // Within placement_radius of any stone (or origin if no stones)
  if (moves.length === 0) return hexDist(q, r) <= radius;
  return moves.some(m => hexDist(q, r, m.q, m.r) <= radius);
}

function updateTransform() {
  const g = document.getElementById('board-group');
  if (!g) return;
  const container = document.getElementById('board-container');
  const cx = container.clientWidth / 2;
  const cy = container.clientHeight / 2;
  g.setAttribute('transform', `translate(${cx + viewX},${cy + viewY}) scale(${viewScale})`);
}

function drawBoard() {
  radius = parseInt(document.getElementById('radius').value) || 2;
  const svg = document.getElementById('board');
  // Choose the stone source for the current mode.
  const inSetup = viewMode === 'setup';
  const stoneSeeds = inSetup
    ? setupStones.map(s => ({q: s.q, r: s.r}))
    : moves;
  // Compute all hex positions: start from stones + origin, expand by radius+1.
  // Setup mode wants more room because stones can be placed anywhere.
  let seeds = [{q:0, r:0}, ...stoneSeeds];
  let cellSet = new Set();
  const margin = (inSetup ? radius + 5 : radius + 2);
  for (const s of seeds) {
    for (let dq = -margin; dq <= margin; dq++) {
      for (let dr = -margin; dr <= margin; dr++) {
        if (hexDist(dq, dr) <= margin) {
          cellSet.add(`${s.q+dq},${s.r+dr}`);
        }
      }
    }
  }
  let cells = [...cellSet].map(k => { const [q,r] = k.split(',').map(Number); return {q,r}; });
  // SVG fills the container; the <g> handles centering + pan/zoom
  svg.removeAttribute('viewBox');
  svg.removeAttribute('width');
  svg.removeAttribute('height');

  // Build stone lookup
  const stoneMap = {};
  if (inSetup) {
    setupStones.forEach(s => { stoneMap[`${s.q},${s.r}`] = s.player; });
  } else {
    moves.forEach((m, i) => { stoneMap[`${m.q},${m.r}`] = whoseStone(i); });
  }

  // Build prob lookup (+ q_hat / candidate lookups when MCTS supplied them).
  const probMap = {};
  const qHatMap = {};
  const candMap = {};
  let maxProb = 0;
  const hasQ = !!(analysisResult && analysisResult.q_hat);
  if (analysisResult) {
    analysisResult.legal.forEach((c, i) => {
      const p = analysisResult.probs[i];
      const key = `${c[0]},${c[1]}`;
      probMap[key] = p;
      if (p > maxProb) maxProb = p;
      if (hasQ) {
        qHatMap[key] = analysisResult.q_hat[i];
        candMap[key] = analysisResult.candidate_set ? analysisResult.candidate_set[i] : false;
      }
    });
  }
  // Among searched candidates, only those "in contention" (near the best eval
  // and not clearly losing) get a numeric label + bright ring; the rest stay a
  // muted red fill. Keeps the heatmap on the moves worth considering instead of
  // a wall of identical -1.00 losing moves. The single best move is always
  // shown, so even a lost position highlights the most stubborn defence.
  let bestQ = -Infinity;
  if (hasQ) for (const k in qHatMap) if (candMap[k] && qHatMap[k] > bestQ) bestQ = qHatMap[k];
  const Q_MARGIN = 0.2, Q_FLOOR = -0.95;
  const inContention = (key) => candMap[key] && (qHatMap[key] === bestQ ||
      (qHatMap[key] >= bestQ - Q_MARGIN && qHatMap[key] > Q_FLOOR));

  const isGraphMode = viewMode === 'graph';
  let hexHtml = '';
  for (const c of cells) {
    const p = axialToPixel(c.q, c.r);
    const key = `${c.q},${c.r}`;
    const stone = stoneMap[key];
    // Setup mode: every rendered hex is interactable, so don't dim non-radius cells.
    const inRadius = inSetup ? true : (isLegalHex(c.q, c.r) || stone);
    let cls = 'hex-empty';
    let fill = '';
    let stroke = '';
    const isSelected = selectedNode && selectedNode.q === c.q && selectedNode.r === c.r;
    if (stone === 'P1') cls = 'hex-p1';
    else if (stone === 'P2') cls = 'hex-p2';
    else if (!isGraphMode && hasQ && candMap[key]) {
      // Diverging eval colormap on q_hat ∈ [-1,1] from root-player perspective:
      // red (losing) → grey (even) → green (winning). Contention moves get the
      // full color + bright ring; searched-but-also-ran moves are muted (still
      // visibly "considered & losing", but they don't shout).
      const q = qHatMap[key];
      const t = Math.max(-1, Math.min(1, q));
      // Continuous diverging map centered on neutral grey at q=0, so tiny ±
      // values read as ~even instead of a hard red/green flip at zero.
      let r = Math.round(t >= 0 ? 100 - 80 * t : 100 - 100 * t);
      let g = Math.round(t >= 0 ? 100 + 100 * t : 100 + 80 * t);
      let b = 90;
      if (inContention(key)) {
        fill = `fill:rgb(${r},${g},${b})`;
        stroke = `;stroke:#ffd23f;stroke-width:2`;
      } else {
        fill = `fill:rgb(${Math.round(r*0.45)},${Math.round(g*0.45)},${Math.round(b*0.45)})`;
      }
    }
    else if (!isGraphMode && probMap[key] !== undefined && maxProb > 0) {
      const prob = probMap[key];
      const t = prob / maxProb;
      const r = Math.round(20 + t * 20);
      const g = Math.round(30 + t * 200);
      const b = Math.round(40 + t * 40);
      fill = `fill:rgb(${r},${g},${b})`;
    }
    let opacity = inRadius ? 1.0 : 0.3;
    if (isGraphMode && isSelected) fill = `fill:#445588`;
    hexHtml += `<polygon class="hex ${cls}" points="${hexCorners(p.x, p.y)}" ` +
      `style="${fill}${stroke};opacity:${opacity}" ` +
      `onclick="clickHex(${c.q},${c.r})" data-q="${c.q}" data-r="${c.r}"/>`;

    // Labels differ by mode
    if (isGraphMode) {
      // Graph mode: coords on all hexes (stone labels hidden so edge distances are readable)
      hexHtml += `<text class="hex-label" x="${p.x}" y="${p.y}" style="opacity:${opacity*0.7}">${c.q},${c.r}</text>`;
    } else {
      // Policy / Setup mode: probs on legal, coords elsewhere, label on stones.
      // Policy stones get the HTTTX-style round number; setup stones have no
      // intrinsic order so they show their coord instead.
      if (stone) {
        if (inSetup) {
          hexHtml += `<text class="hex-label" x="${p.x}" y="${p.y}" style="fill:#fff;opacity:0.8">${c.q},${c.r}</text>`;
        } else {
          const idx = moves.findIndex(m => m.q === c.q && m.r === c.r);
          const round = idx === 0 ? 0 : (((idx - 1) >> 2) + 1);
          hexHtml += `<text class="hex-prob" x="${p.x}" y="${p.y}">${round}</text>`;
        }
      } else if (hasQ && inContention(key)) {
        // q_hat (eval) labeled only on in-contention candidates; also-rans keep
        // their muted color but no number.
        const q = Math.abs(qHatMap[key]) < 0.005 ? 0 : qHatMap[key];
        const qStr = (q > 0 ? '+' : '') + q.toFixed(2);
        hexHtml += `<text class="hex-prob" x="${p.x}" y="${p.y}">${qStr}</text>`;
      } else if (!hasQ && probMap[key] !== undefined && probMap[key] > 0.005) {
        const pct = (probMap[key] * 100).toFixed(probMap[key] > 0.1 ? 0 : 1);
        hexHtml += `<text class="hex-prob" x="${p.x}" y="${p.y}">${pct}%</text>`;
      } else {
        hexHtml += `<text class="hex-label" x="${p.x}" y="${p.y}" style="opacity:${opacity*0.7}">${c.q},${c.r}</text>`;
      }
    }
  }

  // Last-turn highlight, color-coded: P1's most recent turn in light orange,
  // P2's most recent turn in light blue. Walks back from the latest stone and
  // collects the two most recent contiguous same-player runs.
  let overlayHtml = '';
  if (!inSetup && moves.length > 0) {
    const lastTurnByPlayer = {P1: [], P2: []};
    let i = moves.length - 1;
    // First contiguous run (current/last turn)
    let runPlayer = whoseStone(i);
    while (i >= 0 && whoseStone(i) === runPlayer) {
      lastTurnByPlayer[runPlayer].push(i);
      i--;
    }
    // Second contiguous run (other player's most recent turn)
    if (i >= 0) {
      runPlayer = whoseStone(i);
      while (i >= 0 && whoseStone(i) === runPlayer) {
        lastTurnByPlayer[runPlayer].push(i);
        i--;
      }
    }
    const TURN_RING = {P1: '#ffcc99', P2: '#99ddff'};
    for (const player of ['P1', 'P2']) {
      for (const idx of lastTurnByPlayer[player]) {
        const m = moves[idx];
        const lp = axialToPixel(m.q, m.r);
        overlayHtml += `<polygon points="${hexCorners(lp.x, lp.y)}" ` +
          `fill="none" stroke="${TURN_RING[player]}" stroke-width="3" ` +
          `opacity="0.9" pointer-events="none"/>`;
      }
    }
  }

  // Winning line highlight (play mode only)
  if (!inSetup && playGameOver && playWinner) {
    const winLen = parseInt(document.getElementById('win').value) || 6;
    const AXES = [[1,0],[0,1],[1,-1]];
    const winnerStones = new Set();
    moves.forEach((m, i) => {
      if (whoseStone(i) === playWinner) winnerStones.add(`${m.q},${m.r}`);
    });
    let winLine = null;
    for (const [dq, dr] of AXES) {
      if (winLine) break;
      for (const key of winnerStones) {
        const [sq, sr] = key.split(',').map(Number);
        // Check if this is the start of a winning run (no predecessor in set)
        if (winnerStones.has(`${sq-dq},${sr-dr}`)) continue;
        let len = 0;
        let q = sq, r = sr;
        const line = [];
        while (winnerStones.has(`${q},${r}`)) {
          line.push({q, r});
          q += dq; r += dr; len++;
        }
        if (len >= winLen) { winLine = line.slice(0, winLen); break; }
      }
    }
    if (winLine) {
      const winColor = playWinner === 'P1' ? '#ffaa44' : '#44ccff';
      for (const c of winLine) {
        const wp = axialToPixel(c.q, c.r);
        overlayHtml += `<polygon points="${hexCorners(wp.x, wp.y)}" ` +
          `fill="none" stroke="${winColor}" stroke-width="3" opacity="0.9" pointer-events="none"/>`;
      }
      // Connect winning stones with a thick line
      const first = axialToPixel(winLine[0].q, winLine[0].r);
      const last = axialToPixel(winLine[winLine.length-1].q, winLine[winLine.length-1].r);
      overlayHtml += `<line x1="${first.x}" y1="${first.y}" x2="${last.x}" y2="${last.y}" ` +
        `stroke="${winColor}" stroke-width="4" opacity="0.7" stroke-linecap="round" pointer-events="none"/>`;
    }
  }

  // Move markers. With MCTS, the GREEDY PLAYED move is argmax(improved_policy)
  // — NOT argmax(visit counts): under Gumbel/Sequential-Halving the played move
  // is argmax(logit + σ(Q)), which the improved policy encodes. We mark that in
  // white, and separately mark the best-EVAL move (argmax q_hat over searched
  // candidates) in cyan, so policy-vs-value disagreement is legible.
  if (!isGraphMode && analysisResult) {
    const ip = analysisResult.improved_policy;
    if (ip) {
      let playIdx = 0;
      for (let i = 1; i < ip.length; i++) if (ip[i] > ip[playIdx]) playIdx = i;
      const pc = analysisResult.legal[playIdx];
      const pp = axialToPixel(pc[0], pc[1]);
      overlayHtml += `<polygon class="top-marker" points="${hexCorners(pp.x, pp.y)}" style="stroke-width:3;stroke:#fff"/>`;

      const qh = analysisResult.q_hat, cand = analysisResult.candidate_set;
      if (qh) {
        let bestIdx = -1;
        for (let i = 0; i < qh.length; i++) {
          if (cand && !cand[i]) continue;
          if (bestIdx < 0 || qh[i] > qh[bestIdx]) bestIdx = i;
        }
        if (bestIdx >= 0 && bestIdx !== playIdx) {
          const bc = analysisResult.legal[bestIdx];
          const bp = axialToPixel(bc[0], bc[1]);
          overlayHtml += `<polygon class="top-marker" points="${hexCorners(bp.x, bp.y)}" style="stroke-width:3;stroke:#00d4ff;stroke-dasharray:4 3"/>`;
        }
      }
    } else {
      // Raw-policy mode: top-3 by policy probability, as before.
      let ranked = analysisResult.legal.map((c, i) => ({q: c[0], r: c[1], p: analysisResult.probs[i]}));
      ranked.sort((a, b) => b.p - a.p);
      for (let i = 0; i < Math.min(3, ranked.length); i++) {
        if (ranked[i].p < 0.01) break;
        const px = axialToPixel(ranked[i].q, ranked[i].r);
        overlayHtml += `<polygon class="top-marker" points="${hexCorners(px.x, px.y)}" style="stroke-width:${3-i};stroke:${i===0?'#fff':i===1?'#ccc':'#999'}"/>`;
      }
    }
  }

  // Graph mode: when a node is selected, highlight connected nodes
  // by coloring their hex borders and drawing thin connector lines
  let edgeHtml = '';
  if (isGraphMode && analysisResult && analysisResult.graph_edges && selectedNode) {
    // Build edge map for selected node: all edges where selected is src OR dst
    const edges = analysisResult.graph_edges.filter(e => {
      const isSrc = e.src[0] === selectedNode.q && e.src[1] === selectedNode.r;
      const isDst = e.dst[0] === selectedNode.q && e.dst[1] === selectedNode.r;
      if (edgeDir === 'in' && !isDst) return false;  // only edges INTO selected
      if (edgeDir === 'out' && !isSrc) return false;  // only edges FROM selected
      return (isSrc || isDst) && (axisFilter === -1 || e.axis === axisFilter);
    });
    const sp = axialToPixel(selectedNode.q, selectedNode.r);

    // Sort: empty (0) first, friendly (1) next, opponent (-1) on top
    const zOrder = sp => sp !== 0 ? 1 : 0;  // stones on top of empties
    edges.sort((a, b) => zOrder(a.src_player) - zOrder(b.src_player));

    for (const e of edges) {
      // Show the "other" end of the edge
      const isSrc = e.src[0] === selectedNode.q && e.src[1] === selectedNode.r;
      const other = isSrc ? e.dst : e.src;
      const dp = axialToPixel(other[0], other[1]);
      const color = (e.axis >= 0) ? AXIS_COLORS[e.axis] : '#888';

      // Thin connector line
      edgeHtml += `<line x1="${sp.x}" y1="${sp.y}" x2="${dp.x}" y2="${dp.y}" ` +
        `stroke="${color}" stroke-width="2" opacity="0.5" pointer-events="none"/>`;

      // Highlight ring on destination hex — white outer + colored inner
      // so the ring is visible even on same-colored stones
      let ringColor = '#888';  // empty
      if (e.src_player === 1) ringColor = '#e07020';  // P1 (orange)
      else if (e.src_player === -1) ringColor = '#2090d0';  // P2 (blue)
      edgeHtml += `<polygon points="${hexCorners(dp.x, dp.y)}" ` +
        `fill="none" stroke="#fff" stroke-width="4" pointer-events="none"/>`;
      edgeHtml += `<polygon points="${hexCorners(dp.x, dp.y)}" ` +
        `fill="none" stroke="${ringColor}" stroke-width="3" pointer-events="none"/>`;

      // Distance label: show distance from selected node's perspective
      const dist = isSrc ? e.dist : -e.dist;
      const distStr = dist > 0 ? `+${dist}` : `${dist}`;
      edgeHtml += `<text class="edge-label" x="${dp.x}" y="${dp.y}">${distStr}</text>`;
    }

    // Bright ring on selected node
    edgeHtml += `<polygon points="${hexCorners(sp.x, sp.y)}" ` +
      `fill="none" stroke="#fff" stroke-width="3" pointer-events="none"/>`;
  }

  // Edges on top of hexes, overlays on top of edges
  svg.innerHTML = `<g id="board-group">${hexHtml}${edgeHtml}${overlayHtml}</g>`;
  updateTransform();
  updateMovesList();

  // Sidebar info for graph mode
  const legendEl = document.getElementById('threats-info');
  if (isGraphMode && analysisResult) {
    let info = '<div style="margin-top:10px;padding:10px;background:#0f3460;border-radius:6px;font-size:12px;line-height:1.8">';
    info += '<b>Rings:</b> <span style="color:#e07020">orange</span>=P1 · ' +
      '<span style="color:#2090d0">blue</span>=P2 · <span style="color:#888">grey</span>=empty<br>';
    info += '<b>Lines:</b> ' +
      `<span style="color:${AXIS_COLORS[0]}">red</span>=(1,0) ` +
      `<span style="color:${AXIS_COLORS[1]}">green</span>=(0,1) ` +
      `<span style="color:${AXIS_COLORS[2]}">blue</span>=(1,-1)<br>`;
    info += '<b>Numbers:</b> signed distance along axis<br>';
    if (selectedNode) {
      const allEdges = analysisResult.graph_edges.filter(e => {
        const isSrc = e.src[0]===selectedNode.q && e.src[1]===selectedNode.r;
        const isDst = e.dst[0]===selectedNode.q && e.dst[1]===selectedNode.r;
        if (edgeDir === 'in' && !isDst) return false;
        if (edgeDir === 'out' && !isSrc) return false;
        return isSrc || isDst;
      });
      const filtered = axisFilter === -1 ? allEdges : allEdges.filter(e => e.axis === axisFilter);
      const byAxis = [0,0,0];
      let sameCount=0, oppCount=0, emptyCount=0;
      for (const e of filtered) {
        if (e.axis >= 0) byAxis[e.axis]++;
        if (e.src_player === 1) sameCount++;  // reusing var name, now means P1
        else if (e.src_player === -1) oppCount++;  // P2
        else emptyCount++;
      }
      info += `<br><b>Node (${selectedNode.q},${selectedNode.r})</b> — ${filtered.length} edges shown<br>`;
      info += byAxis.map((n,i) => `<span style="color:${AXIS_COLORS[i]}">${AXIS_NAMES[i]}:${n}</span>`).join(' · ') + '<br>';
      info += `<span style="color:#e07020">P1=${sameCount}</span> · <span style="color:#2090d0">P2=${oppCount}</span> · empty=${emptyCount}`;
    } else {
      info += '<br>Click a hex to inspect its edges';
    }
    info += '</div>';
    legendEl.innerHTML = info;
  } else if (!isGraphMode) {
    legendEl.innerHTML = '';
  }
}

function clickHex(q, r) {
  if (wasDragging) { wasDragging = false; return; }
  if (viewMode === 'graph') {
    // Graph mode: click to select/deselect node for edge inspection
    if (selectedNode && selectedNode.q === q && selectedNode.r === r) {
      selectedNode = null;
    } else {
      selectedNode = {q, r};
    }
    drawBoard();
    return;
  }
  if (viewMode === 'setup') {
    // (0,0) is the engine-seeded P1 origin and cannot be removed/recoloured.
    if (q === 0 && r === 0) return;
    const idx = setupStones.findIndex(s => s.q === q && s.r === r);
    if (setupBrush === 'erase') {
      if (idx >= 0) setupStones.splice(idx, 1);
    } else if (idx >= 0) {
      setupStones[idx].player = setupBrush; // replace colour
    } else {
      setupStones.push({q, r, player: setupBrush});
    }
    analysisResult = null;
    drawBoard();
    return;
  }
  if (viewMode === 'play') {
    // Play mode: human places stones
    if (playGameOver || botThinking) return;
    const cp = currentPlayer();
    if (playerType[cp] !== 'human') return; // not a human's turn
    if (!isLegalHex(q, r)) return;
    moves.push({q, r});
    analysisResult = null;
    drawBoard();
    updatePlayStatus();
    // Check terminal or trigger bot turn
    checkTerminalThenBot();
    return;
  }
  // Policy mode: place stones
  if (!isLegalHex(q, r)) return;
  if (replayMoves !== null) {
    // Branching: drop any "future" moves and treat the current scrub
    // position as the new live tip. replayMoves stays valid so the
    // scrubber keeps working through the new continuation.
    replayMoves = moves.slice();
  }
  moves.push({q, r});
  if (replayMoves !== null) {
    replayMoves.push({q, r});
    replayIndex = replayMoves.length;
    updateReplayUI();
    // Re-analyse the branched continuation. The batched endpoint makes this
    // cheap (~sub-second for typical games); the gen counter in
    // computeReplayTrajectory cancels any prior in-flight run.
    computeReplayTrajectory();
  }
  analysisResult = null;
  selectedNode = null;
  drawBoard();
  if (document.getElementById('auto-analyze').checked) analyze();
}

function undo() {
  if (viewMode === 'play' && botThinking) return;
  if (viewMode === 'setup') {
    // Pop the most recently added setup stone, but never the (0,0) seed.
    for (let i = setupStones.length - 1; i >= 0; i--) {
      const s = setupStones[i];
      if (s.q === 0 && s.r === 0) continue;
      setupStones.splice(i, 1);
      break;
    }
    analysisResult = null;
    drawBoard();
    return;
  }
  if (replayMoves !== null && (viewMode === 'policy' || viewMode === 'graph')) {
    // In replay: undo scrubs backwards rather than mutating replayMoves.
    // Existing muscle-memory from "Undo to scan a game" keeps working.
    scrubTo(replayIndex - 1);
    return;
  }
  if (viewMode === 'play') botMoveGeneration++; // cancel any pending bot loop
  moves.pop();
  analysisResult = null;
  if (viewMode === 'play') {
    playGameOver = false;
    playWinner = null;
    botLoopRunning = false;
    updatePlayStatus();
  }
  drawBoard();
}

function clearBoard() {
  if (viewMode === 'setup') {
    clearSetup();
    return;
  }
  if (viewMode === 'play') { botMoveGeneration++; }
  moves = [{q:0, r:0}];
  replayMoves = null;
  replayMainline = null;
  replayIndex = 0;
  cancelReplayTrajectory();
  hideEvalTooltip();
  updateReplayUI();
  analysisResult = null;
  if (viewMode === 'play') {
    playGameOver = false;
    playWinner = null;
    aiLastValue = null;
    botThinking = false;
    botLoopRunning = false;
    updatePlayStatus();
  }
  drawBoard();
}

// HTTTX text uses explore.htttx.io's coordinate convention, a mirror of our
// internal axial frame: internal (q, r) <-> HTTTX (q + r, -r). Self-inverse,
// so both directions apply the same map (mirrors serving/htttx.py).
function mirrorAxial(q, r) { return {q: q + r, r: -r}; }

function movesToHTTTX() {
  // HTTTX format: "version[1];\n1. [q,r][q,r];\n2. [q,r][q,r];\n..."
  // Skip move 0 (auto-placed origin). Group remaining moves into turns of 2.
  // Turn 1 = moves 1-2, turn 2 = moves 3-4, etc.
  if (moves.length <= 1) return 'version[1];\n';
  const out = moves.map(m => mirrorAxial(m.q, m.r));
  let s = 'version[1];\n';
  let turn = 1;
  for (let i = 1; i < out.length; i += 2) {
    s += `${turn}.`;
    s += ` [${out[i].q},${out[i].r}]`;
    if (i + 1 < out.length) {
      s += `[${out[i+1].q},${out[i+1].r}]`;
    }
    s += ';\n';
    turn++;
  }
  return s;
}

function parseHTTTX(s) {
  const coords = [];
  for (const m of s.matchAll(/\[(-?\d+),(-?\d+)\]/g)) {
    coords.push(mirrorAxial(parseInt(m[1]), parseInt(m[2])));
  }
  return coords;
}

async function copyPosition() {
  const s = movesToHTTTX();
  await navigator.clipboard.writeText(s);
  document.getElementById('status').innerHTML = '<span style="color:#4f4">Copied to clipboard</span>';
}

async function pastePosition() {
  try {
    const s = await navigator.clipboard.readText();
    const coords = parseHTTTX(s);
    if (coords.length === 0) {
      document.getElementById('status').innerHTML = '<span style="color:#f55">No moves found in clipboard</span>';
      return;
    }
    // Reset and apply: move 0 is always origin
    moves = [{q:0, r:0}];
    for (const c of coords) {
      moves.push(c);
    }
    // Enter replay mode: keep the full game and start scrubbed to the final
    // position. Use Home / ⏮ to jump to the start and step forward from there.
    replayMoves = moves.slice();
    replayMainline = moves.slice();  // snapshot for the Mainline button
    replayIndex = replayMoves.length;
    cancelReplayTrajectory();
    analysisResult = null;
    selectedNode = null;
    updateReplayUI();
    drawBoard();
    if (document.getElementById('auto-analyze').checked) analyze();
    document.getElementById('status').innerHTML = `<span style="color:#4f4">Loaded ${coords.length} moves — use ◀ ▶ or ← → to scrub</span>`;
    computeReplayTrajectory();
  } catch(e) {
    document.getElementById('status').innerHTML = `<span style="color:#f55">Paste failed: ${e}</span>`;
  }
}

function updateMovesList() {
  const div = document.getElementById('moves-list');
  if (viewMode === 'setup') {
    const p1 = setupStones.filter(s => s.player === 'P1').length;
    const p2 = setupStones.filter(s => s.player === 'P2').length;
    const turnCol = setupTurn === 'P1' ? '#e07020' : '#2090d0';
    div.innerHTML = `<b>Setup:</b> ` +
      `<span style="color:#e07020">P1=${p1}</span> · ` +
      `<span style="color:#2090d0">P2=${p2}</span><br>` +
      `To move: <b style="color:${turnCol}">${setupTurn}</b> ` +
      `(${setupMovesRemaining} placement${setupMovesRemaining===1?'':'s'} left)`;
    return;
  }
  if (!moves.length) { div.innerHTML = ''; return; }
  let s = '<b>Moves:</b> ';
  moves.forEach((m, i) => {
    const who = whoseStone(i);
    const col = who === 'P1' ? '#e07020' : '#2090d0';
    s += `<span style="color:${col}">${i}.[${m.q},${m.r}]</span> `;
  });
  s += `<br>Next: <b>${currentPlayer()}</b>`;
  div.innerHTML = s;
}

function getMctsParams() {
  const useMcts = document.getElementById('use-mcts').checked;
  if (!useMcts) return {};
  return {
    mcts_sims: parseInt(document.getElementById('mcts-sims').value) || 128,
    mcts_m_actions: parseInt(document.getElementById('mcts-m').value) || 32,
  };
}

// Toggle MCTS params visibility
document.addEventListener('DOMContentLoaded', () => {
  const cb = document.getElementById('use-mcts');
  if (cb) cb.addEventListener('change', () => {
    document.getElementById('mcts-params').style.display = cb.checked ? 'block' : 'none';
  });
});

// Latest-wins guard for analyze(): scrubbing fires one /analyze per position,
// and responses can return out of order. Each call takes a sequence number and
// aborts the previous in-flight request; only the newest response renders.
let analyzeSeq = 0;
let analyzeAbort = null;

async function analyze() {
  const resolved = resolveCheckpointFor('ckpt');
  if (resolved.error || !resolved.path) {
    alert(resolved.error || 'Select a checkpoint');
    return;
  }
  const mySeq = ++analyzeSeq;
  if (analyzeAbort) analyzeAbort.abort();
  analyzeAbort = new AbortController();
  const mySignal = analyzeAbort.signal;
  const ckpt = resolved.path;
  const ckptLabel = resolved.label || ckpt.split('/').pop();
  const win = parseInt(document.getElementById('win').value);
  radius = parseInt(document.getElementById('radius').value);
  const maxm = parseInt(document.getElementById('maxmoves').value);
  const mcts = getMctsParams();
  const label = mcts.mcts_sims ? `Analyzing (MCTS n=${mcts.mcts_sims})...` : 'Analyzing...';
  document.getElementById('status').textContent = label;
  try {
    const payload = {checkpoint: ckpt,
      win_length: win, placement_radius: radius, max_moves: maxm, ...mcts};
    if (viewMode === 'setup') {
      payload.moves = [[0, 0]]; // ignored when setup_stones is set, but keep schema
      payload.setup_stones = setupStones.map(s => [s.q, s.r, s.player]);
      payload.setup_current_player = setupTurn;
      payload.setup_moves_remaining = setupMovesRemaining;
    } else {
      payload.moves = moves.map(m => [m.q, m.r]);
    }
    const resp = await fetch('/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
      signal: mySignal
    });
    const result = await resp.json();
    // Drop stale responses: the user has scrubbed past this position.
    if (mySeq !== analyzeSeq) return;
    analysisResult = result;
    if (analysisResult.error) {
      document.getElementById('status').innerHTML = `<span style="color:#f55">${analysisResult.error}</span>`;
      analysisResult = null;
    } else {
      const v = analysisResult.value;
      const who = currentPlayer();
      const vStr = v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3);
      const hasQtop = !!analysisResult.q_hat;
      const cand = analysisResult.candidate_set || [];
      const topMoves = analysisResult.legal
        .map((c, i) => ({c, p: analysisResult.probs[i], l: analysisResult.logits[i],
                         q: hasQtop ? analysisResult.q_hat[i] : null, cand: !!cand[i]}))
        // With MCTS, only searched candidates carry a meaningful q_hat.
        .filter(m => hasQtop ? m.cand : true)
        .sort((a, b) => hasQtop ? (b.q - a.q) : (b.p - a.p))
        .slice(0, 5)
        .map((m, i) => {
          if (hasQtop) {
            const q = Math.abs(m.q) < 0.0005 ? 0 : m.q;
            const qStr = (q > 0 ? '+' : '') + q.toFixed(3);
            return `${i+1}. (${m.c[0]},${m.c[1]}) <b class="val">${qStr}</b> <span style="color:#888">${(m.p*100).toFixed(0)}% visits</span>`;
          }
          return `${i+1}. (${m.c[0]},${m.c[1]}) ${(m.p*100).toFixed(1)}% <span style="color:#888">[${m.l.toFixed(2)}]</span>`;
        })
        .join('<br>');
      const ent = analysisResult.entropy;
      const maxEnt = analysisResult.max_entropy;
      const entPct = maxEnt > 0 ? (ent / maxEnt * 100).toFixed(0) : '?';
      const mctsLabel = getMctsParams().mcts_sims ? ` <span style="color:#00d4ff">[MCTS q_hat — white=plays, cyan=best eval]</span>` : ' <span style="color:#888">[raw policy]</span>';
      document.getElementById('status').innerHTML =
        `<b>Loaded:</b> <span style="color:#00d4ff">${ckptLabel}</span><br>` +
        `<b>Value:</b> <span class="val">${vStr}</span> (${who} to move, +1 = ${who} winning)<br>` +
        `<b>Entropy:</b> ${ent} / ${maxEnt} bits (${entPct}% of uniform)${mctsLabel}<br><br>` +
        `<b>Top moves:</b><br>${topMoves}`;

      // Threats panel
      const threats = analysisResult.threats || [];
      let thtml = '';
      if (threats.length) {
        thtml = '<div style="margin-top:10px;padding:10px;background:#0f3460;border-radius:6px;font-size:12px">';
        thtml += `<b style="color:#ff6b6b">Threats detected (${threats.length}):</b><br>`;
        for (const t of threats) {
          const col = t.owner === 'P1' ? '#e07020' : '#2090d0';
          const exts = t.extensions.map(e => `(${e.coord[0]},${e.coord[1]})=${(e.prob*100).toFixed(1)}%`).join(', ');
          thtml += `<span style="color:${col}">${t.owner}</span> ${t.length}-in-a-row — block at: ${exts}<br>`;
        }
        thtml += '</div>';
      }
      document.getElementById('threats-info').innerHTML = thtml;
    }
    drawBoard();
  } catch(e) {
    // Aborted requests (superseded by a newer scrub) are expected — ignore.
    // Also suppress errors from stale requests so they can't clobber the UI.
    if (e.name === 'AbortError' || mySeq !== analyzeSeq) return;
    document.getElementById('status').innerHTML = `<span style="color:#f55">${e}</span>`;
  }
}

function isChampionPath(p) { return p.endsWith('/self_play/champion.pt'); }

function checkpointStep(p) {
  const m = p.match(/checkpoint_(\d+)\.pt$/);
  return m ? parseInt(m[1], 10) : null;
}

let _allCheckpoints = []; // cached from /checkpoints, used by resolveCheckpoint

function updateChampionToggle(prefix) {
  // prefix is 'ckpt', 'p1-ckpt', or 'p2-ckpt'.
  const cb = document.getElementById(prefix + '-use-champion');
  const inp = document.getElementById(prefix);
  if (!cb || !inp) return;
  inp.disabled = cb.checked;
  inp.style.opacity = cb.checked ? '0.5' : '1';
}

function resolveCheckpointFor(inputId) {
  const cb = document.getElementById(inputId + '-use-champion');
  if (cb && cb.checked) {
    const champion = _allCheckpoints.find(isChampionPath) || null;
    if (!champion) return {path: null, error: 'No champion.pt found'};
    return {path: champion, label: 'champion'};
  }
  const inp = document.getElementById(inputId);
  return resolveCheckpoint(inp ? inp.value : '');
}

function populateCheckpointDatalist(datalistId, data) {
  // Datalist holds training step numbers. Champion is selected via the
  // separate "Use current champion" checkbox, not through the datalist.
  const dl = document.getElementById(datalistId);
  if (!dl) return;
  dl.innerHTML = '';
  // Latest checkpoints first so the dropdown bias is recent
  const training = data.filter(p => !isChampionPath(p))
                       .map(p => ({p, step: checkpointStep(p)}))
                       .filter(x => x.step !== null)
                       .sort((a, b) => b.step - a.step);
  for (const {step} of training) {
    const opt = document.createElement('option');
    opt.value = String(step);
    opt.label = `step ${step}`;
    dl.appendChild(opt);
  }
}

function setChampionRowVisibility(data) {
  // Show "Use current champion" rows iff a champion.pt exists.
  const visible = data.some(isChampionPath);
  for (const id of ['ckpt-champion-row', 'p1-ckpt-champion-row', 'p2-ckpt-champion-row']) {
    const el = document.getElementById(id);
    if (el) el.style.display = visible ? 'block' : 'none';
    if (!visible) {
      const cb = document.getElementById(id.replace('-row', '') + '-use-champion');
      if (cb && cb.checked) { cb.checked = false; updateChampionToggle(id.replace('-champion-row','')); }
    }
  }
}

function resolveCheckpoint(typed) {
  // Map free-form input to a full checkpoint path.
  //   ''            -> latest training checkpoint
  //   'champion'/c  -> champion.pt
  //   '147600'      -> closest checkpoint by step (exact preferred)
  //   'checkpoint_147600.pt' / full path -> that path
  // Returns {path, label, error}
  const data = _allCheckpoints;
  if (!data.length) return {path: null, error: 'No checkpoints available'};
  const champion = data.find(isChampionPath) || null;
  const training = data.filter(p => !isChampionPath(p));

  const raw = (typed || '').trim();
  if (!raw) {
    const latest = training[training.length - 1];
    if (!latest) return {path: champion, label: 'champion'};
    return {path: latest, label: `step ${checkpointStep(latest)} (latest)`};
  }
  const t = raw.toLowerCase();
  if (t === 'champion' || t === 'c' || t === '★') {
    if (!champion) return {path: null, error: 'No champion.pt found'};
    return {path: champion, label: 'champion'};
  }
  // Exact full-path match
  if (data.includes(raw)) {
    return {path: raw, label: raw.split('/').pop()};
  }
  // Filename match (e.g. "checkpoint_147600.pt")
  const fname = data.find(p => p.split('/').pop() === raw);
  if (fname) return {path: fname, label: fname.split('/').pop()};
  // Numeric step → exact, else closest
  const num = parseInt(t, 10);
  if (!isNaN(num) && /^\d+$/.test(t)) {
    const candidates = training
      .map(p => ({p, step: checkpointStep(p)}))
      .filter(x => x.step !== null);
    if (!candidates.length) return {path: null, error: 'No training checkpoints'};
    candidates.sort((a, b) => Math.abs(a.step - num) - Math.abs(b.step - num));
    const c = candidates[0];
    const exact = c.step === num;
    return {path: c.p,
            label: exact ? `step ${c.step}` : `step ${c.step} (closest to ${num})`};
  }
  return {path: null, error: `No checkpoint matches "${raw}"`};
}

function maybeSetInitialCheckpointInput(inputId, data) {
  const inp = document.getElementById(inputId);
  if (!inp || inp.value) return;
  const training = data.filter(p => !isChampionPath(p))
                       .map(p => ({p, step: checkpointStep(p)}))
                       .filter(x => x.step !== null)
                       .sort((a, b) => a.step - b.step);
  if (training.length) inp.value = String(training[training.length - 1].step);
  else if (data.find(isChampionPath)) inp.value = 'champion';
}

async function loadCheckpoints() {
  const resp = await fetch('/checkpoints');
  const data = await resp.json();
  _allCheckpoints = data;
  populateCheckpointDatalist('ckpt-list', data);
  maybeSetInitialCheckpointInput('ckpt', data);
  setChampionRowVisibility(data);
}

document.addEventListener('keydown', e => {
  // Don't hijack keys while the user is typing in a text field.
  const tag = (e.target.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable) {
    if (e.key === 'Enter') analyze();
    return;
  }
  if (e.key === 'Enter') { analyze(); return; }
  if (replayMoves === null) return;
  if (viewMode !== 'policy' && viewMode !== 'graph') return;
  switch (e.key) {
    case 'ArrowLeft':  e.preventDefault(); replayPrev(); break;
    case 'ArrowRight': e.preventDefault(); replayNext(); break;
    case 'Home':       e.preventDefault(); replayFirst(); break;
    case 'End':        e.preventDefault(); replayLast(); break;
  }
});

// --- Play mode state ---
let playGameOver = false;
let playWinner = null;
let aiLastValue = null;
let botThinking = false;
let paused = true;
let pauseResolve = null; // resolve function to unpause
let botLoopRunning = false;
let botMoveGeneration = 0; // incremented to cancel stale bot loops

// Player config: 'human' or 'bot' per side
let playerType = { P1: 'human', P2: 'bot' };

function setPlayerType(side, type) {
  const prev = playerType[side];
  playerType[side] = type;
  // Update button styling
  const color = side === 'P1' ? '#e07020' : '#2090d0';
  const sideLow = side.toLowerCase();
  const humanBtn = document.getElementById(sideLow + '-human');
  const botBtn = document.getElementById(sideLow + '-bot');
  const sealBtn = document.getElementById(sideLow + '-sealbot');
  humanBtn.style.background = type === 'human' ? color : '#555';
  humanBtn.style.color = type === 'human' ? '#fff' : '#ccc';
  botBtn.style.background = type === 'bot' ? color : '#555';
  botBtn.style.color = type === 'bot' ? '#fff' : '#ccc';
  sealBtn.style.background = type === 'sealbot' ? color : '#555';
  sealBtn.style.color = type === 'sealbot' ? '#fff' : '#ccc';
  // Show/hide per-type rows
  const ckptRow = document.getElementById(sideLow + '-ckpt-row');
  const mctsRow = document.getElementById(sideLow + '-mcts-row');
  const sealRow = document.getElementById(sideLow + '-sealbot-row');
  ckptRow.style.display = type === 'bot' ? 'block' : 'none';
  mctsRow.style.display = type === 'bot' ? 'block' : 'none';
  sealRow.style.display = type === 'sealbot' ? 'block' : 'none';
  updatePlayStatus();
  // If switching to a bot type and it's this player's turn, start the bot loop
  const isBotType = (t) => t === 'bot' || t === 'sealbot';
  if (isBotType(type) && !isBotType(prev) && !playGameOver && currentPlayer() === side) {
    startBotLoop();
  }
  // If switching away from a bot type, cancel any pending bot action for this side
  if (!isBotType(type) && isBotType(prev)) {
    botMoveGeneration++;
  }
}

function isBotTurn() {
  if (playGameOver || botThinking) return false;
  const t = playerType[currentPlayer()];
  return t === 'bot' || t === 'sealbot';
}

function getPlayerCheckpoint(side) {
  const r = resolveCheckpointFor(side.toLowerCase() + '-ckpt');
  return r.path || '';
}

function getMoveDelay() {
  const v = parseInt(document.getElementById('move-delay').value);
  return isNaN(v) ? 500 : Math.max(0, v);
}

function pauseBtnLabel() {
  if (!paused) return 'Pause';
  return moves.length <= 1 ? 'Start' : 'Resume';
}

function togglePause() {
  paused = !paused;
  const btn = document.getElementById('pause-btn');
  btn.textContent = pauseBtnLabel();
  btn.style.background = paused ? '#e07020' : '#555';
  btn.style.color = paused ? '#fff' : '#ccc';
  // If unpausing, resolve the pause promise and restart bot loop
  if (!paused && pauseResolve) {
    pauseResolve();
    pauseResolve = null;
  }
  if (!paused && !botLoopRunning && isBotTurn()) {
    startBotLoop();
  }
  updatePlayStatus();
}

function waitForUnpause() {
  if (!paused) return Promise.resolve();
  return new Promise(resolve => { pauseResolve = resolve; });
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function newGame() {
  botMoveGeneration++;
  moves = [{q:0, r:0}];
  analysisResult = null;
  selectedNode = null;
  playGameOver = false;
  playWinner = null;
  aiLastValue = null;
  botThinking = false;
  botLoopRunning = false;
  paused = true;
  const pauseBtn = document.getElementById('pause-btn');
  if (pauseBtn) { pauseBtn.textContent = 'Start'; pauseBtn.style.background = '#555'; pauseBtn.style.color = '#ccc'; }
  updatePlayStatus();
  drawBoard();
}

function updatePlayStatus() {
  const el = document.getElementById('play-status');
  if (!el) return;
  // Update TO MOVE indicators
  const cp = currentPlayer();
  const p1tm = document.getElementById('p1-tomove');
  const p2tm = document.getElementById('p2-tomove');
  if (p1tm) { p1tm.style.display = (cp === 'P1' && !playGameOver) ? 'inline' : 'none'; }
  if (p2tm) { p2tm.style.display = (cp === 'P2' && !playGameOver) ? 'inline' : 'none'; }
  if (playGameOver) {
    const winColor = playWinner === 'P1' ? '#e07020' : playWinner === 'P2' ? '#2090d0' : '#ccc';
    const msg = playWinner ? `<span style="color:${winColor}">${playWinner} wins!</span>` : 'Draw!';
    el.innerHTML = msg;
    return;
  }
  if (botThinking) {
    const side = currentPlayer();
    const col = side === 'P1' ? '#e07020' : '#2090d0';
    el.innerHTML = `<span style="color:${col}">${side} bot thinking...</span>${paused ? ' <span style="color:#ffcc00">(paused)</span>' : ''}`;
    return;
  }
  const isBot = playerType[cp] === 'bot';
  const label = isBot ? `<b>${cp}</b> (bot)` : `<b>${cp}</b> (your turn)`;
  let html = `Turn: ${label}<br>`;
  html += `Move #${moves.length}`;
  if (aiLastValue !== null) {
    const vStr = aiLastValue > 0 ? `+${aiLastValue.toFixed(3)}` : aiLastValue.toFixed(3);
    html += `<br>Last bot value: <span style="color:#00d4ff;font-weight:bold">${vStr}</span>`;
  }
  if (paused) html += '<br><span style="color:#ffcc00">Paused</span>';
  el.innerHTML = html;
}

async function checkTerminal() {
  const win = parseInt(document.getElementById('win').value);
  const rad = parseInt(document.getElementById('radius').value);
  const maxm = parseInt(document.getElementById('maxmoves').value);
  try {
    const resp = await fetch('/check_terminal', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({moves: moves.map(m => [m.q, m.r]),
        win_length: win, placement_radius: rad, max_moves: maxm})
    });
    return await resp.json();
  } catch(e) { return {terminal: false, winner: null}; }
}

async function checkTerminalThenBot() {
  // After a human move, peek for terminal then maybe start the bot loop.
  if (playGameOver) return;
  const peek = await checkTerminal();
  if (peek && peek.terminal) {
    playGameOver = true;
    playWinner = peek.winner;
    updatePlayStatus();
    drawBoard();
    return;
  }
  if (isBotTurn()) startBotLoop();
}

function getPlayerMctsParams(player) {
  const prefix = player.toLowerCase();
  const sims = parseInt(document.getElementById(prefix + '-sims').value) || 0;
  const m = parseInt(document.getElementById(prefix + '-m').value) || 16;
  if (sims <= 0) return {};
  return { mcts_sims: sims, mcts_m_actions: m };
}

async function fetchBotMove(checkpoint, player) {
  if (!checkpoint) { alert('Select a checkpoint for the bot'); return null; }
  const win = parseInt(document.getElementById('win').value);
  const rad = parseInt(document.getElementById('radius').value);
  const maxm = parseInt(document.getElementById('maxmoves').value);
  const mcts = player ? getPlayerMctsParams(player) : getMctsParams();
  const resp = await fetch('/ai_move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({moves: moves.map(m => [m.q, m.r]), checkpoint: checkpoint,
      win_length: win, placement_radius: rad, max_moves: maxm, ...mcts})
  });
  return await resp.json();
}

function getPlayerTimeLimit(player) {
  const el = document.getElementById(player.toLowerCase() + '-time');
  const v = el ? parseFloat(el.value) : 0.5;
  return isNaN(v) || v <= 0 ? 0.5 : v;
}

async function fetchSealbotMove(player) {
  const win = parseInt(document.getElementById('win').value);
  const rad = parseInt(document.getElementById('radius').value);
  const maxm = parseInt(document.getElementById('maxmoves').value);
  const time_limit = getPlayerTimeLimit(player);
  const resp = await fetch('/sealbot_move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({moves: moves.map(m => [m.q, m.r]),
      win_length: win, placement_radius: rad, max_moves: maxm, time_limit})
  });
  return await resp.json();
}

async function fetchPlayerMove(player) {
  // Dispatch by playerType. Returns a unified shape:
  //   { terminal, winner, moves: [[q,r], ...], value? }
  if (playerType[player] === 'sealbot') {
    const r = await fetchSealbotMove(player);
    if (!r) return null;
    if (r.error) return r;
    if (r.terminal && (!r.moves || !r.moves.length)) {
      return {terminal: true, winner: r.winner, moves: []};
    }
    return {terminal: !!r.terminal, winner: r.winner ?? null,
            moves: r.moves || [], value: null};
  }
  const ckpt = getPlayerCheckpoint(player) || document.getElementById('ckpt').value;
  const r = await fetchBotMove(ckpt, player);
  if (!r) return null;
  if (r.error) return r;
  if (r.terminal) return {terminal: true, winner: r.winner, moves: []};
  return {terminal: false, winner: null, moves: [r.move], value: r.value};
}

async function startBotLoop() {
  if (botLoopRunning) return;
  botLoopRunning = true;
  const gen = botMoveGeneration;
  const isBotType = (t) => t === 'bot' || t === 'sealbot';
  while (!playGameOver && botLoopRunning && gen === botMoveGeneration) {
    const cp = currentPlayer();
    if (!isBotType(playerType[cp])) break; // it's a human's turn now

    // Wait for unpause
    if (paused) {
      updatePlayStatus();
      await waitForUnpause();
      if (gen !== botMoveGeneration) break;
    }

    // Add move delay (between bot moves, not before the first)
    const delayMs = getMoveDelay();
    if (delayMs > 0) {
      await delay(delayMs);
      if (gen !== botMoveGeneration) break;
    }

    botThinking = true;
    updatePlayStatus();

    const result = await fetchPlayerMove(cp);
    if (gen !== botMoveGeneration) break; // cancelled

    if (!result || result.error) {
      const el = document.getElementById('play-status');
      el.innerHTML = `<span style="color:#f55">Bot error: ${result ? result.error : 'no response'}</span>`;
      botThinking = false;
      botLoopRunning = false;
      return;
    }
    if (result.terminal && (!result.moves || !result.moves.length)) {
      playGameOver = true;
      playWinner = result.winner;
      botThinking = false;
      botLoopRunning = false;
      updatePlayStatus();
      drawBoard();
      return;
    }
    // Apply each move (sealbot may return a pair). Animate with delay between halves.
    aiLastValue = (result.value !== undefined) ? result.value : null;
    for (let i = 0; i < result.moves.length; i++) {
      moves.push({q: result.moves[i][0], r: result.moves[i][1]});
      drawBoard();
      updatePlayStatus();
      if (i + 1 < result.moves.length && delayMs > 0) {
        await delay(delayMs);
        if (gen !== botMoveGeneration) { botThinking = false; botLoopRunning = false; return; }
      }
    }
    botThinking = false;
    if (result.terminal) {
      playGameOver = true;
      playWinner = result.winner;
      botLoopRunning = false;
      updatePlayStatus();
      drawBoard();
      return;
    }
    drawBoard();
    updatePlayStatus();
  }
  botLoopRunning = false;
  // Final terminal check (e.g. when control passes to a human after a bot win)
  if (!playGameOver && gen === botMoveGeneration) {
    const peek = await checkTerminal();
    if (peek && peek.terminal) {
      playGameOver = true;
      playWinner = peek.winner;
      updatePlayStatus();
      drawBoard();
    }
  }
  updatePlayStatus();
}

// Load checkpoints into per-player datalists
async function loadPlayerCheckpoints() {
  const resp = await fetch('/checkpoints');
  const data = await resp.json();
  _allCheckpoints = data;
  for (const side of ['p1', 'p2']) {
    populateCheckpointDatalist(side + '-ckpt-list', data);
    maybeSetInitialCheckpointInput(side + '-ckpt', data);
  }
  setChampionRowVisibility(data);
}

loadCheckpoints();
loadPlayerCheckpoints();
drawBoard();

// Pan: mousedown on SVG background starts drag
const svgEl = document.getElementById('board');
svgEl.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  // Only pan if clicking on SVG background or the board-group, not on a hex
  isPanning = true;
  panStartX = e.clientX; panStartY = e.clientY;
  panStartVX = viewX; panStartVY = viewY;
  svgEl.classList.add('panning');
  e.preventDefault();
});
window.addEventListener('mousemove', e => {
  if (!isPanning) return;
  viewX = panStartVX + (e.clientX - panStartX);
  viewY = panStartVY + (e.clientY - panStartY);
  updateTransform();
});
window.addEventListener('mouseup', e => {
  if (!isPanning) return;
  isPanning = false;
  svgEl.classList.remove('panning');
  const dx = e.clientX - panStartX, dy = e.clientY - panStartY;
  if (dx*dx + dy*dy > 9) wasDragging = true; // suppress click if dragged >3px
});
// Zoom: scroll wheel
svgEl.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
  const container = document.getElementById('board-container');
  const cx = container.clientWidth / 2;
  const cy = container.clientHeight / 2;
  // Zoom toward mouse position
  const mx = e.clientX - container.getBoundingClientRect().left;
  const my = e.clientY - container.getBoundingClientRect().top;
  // Adjust pan so the point under the mouse stays fixed
  viewX = mx - factor * (mx - viewX - cx) - cx;
  viewY = my - factor * (my - viewY - cy) - cy;
  viewScale *= factor;
  viewScale = Math.max(0.2, Math.min(5, viewScale));
  updateTransform();
}, {passive: false});
// Resize handler
window.addEventListener('resize', () => updateTransform());

// Auto-refresh checkpoint list every 30s. Inputs preserve the user's typed
// value automatically; we just rebuild the datalists.
setInterval(async () => {
  await loadCheckpoints();
  await loadPlayerCheckpoints();
}, 30000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    checkpoint_dir = "runs/ablation-baseline/checkpoints"

    def log_message(self, fmt, *args):
        # Quieter logging
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif parsed.path == "/checkpoints":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_list_checkpoints(self.checkpoint_dir)).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/analyze":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            try:
                import traceback as _tb
                result = _analyze(
                    moves=body["moves"],
                    checkpoint=body["checkpoint"],
                    win_length=body.get("win_length", 5),
                    placement_radius=body.get("placement_radius", 2),
                    max_moves=body.get("max_moves", 40),
                    mcts_sims=body.get("mcts_sims", 0),
                    mcts_m_actions=body.get("mcts_m_actions", 16),
                    setup_stones=body.get("setup_stones"),
                    setup_current_player=body.get("setup_current_player", "P2"),
                    setup_moves_remaining=body.get("setup_moves_remaining", 2),
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                _tb.print_exc()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == "/analyze_trajectory":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            try:
                import traceback as _tb
                result = _analyze_trajectory(
                    moves=body["moves"],
                    checkpoint=body["checkpoint"],
                    win_length=body.get("win_length", 6),
                    placement_radius=body.get("placement_radius", 8),
                    max_moves=body.get("max_moves", 200),
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                _tb.print_exc()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == "/ai_move":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            try:
                result = _ai_move(
                    moves=body["moves"],
                    checkpoint=body["checkpoint"],
                    win_length=body.get("win_length", 5),
                    placement_radius=body.get("placement_radius", 2),
                    max_moves=body.get("max_moves", 40),
                    mcts_sims=body.get("mcts_sims", 0),
                    mcts_m_actions=body.get("mcts_m_actions", 16),
                    setup_stones=body.get("setup_stones"),
                    setup_current_player=body.get("setup_current_player", "P2"),
                    setup_moves_remaining=body.get("setup_moves_remaining", 2),
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == "/check_terminal":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            try:
                result = _check_terminal(
                    moves=body["moves"],
                    win_length=body.get("win_length", 5),
                    placement_radius=body.get("placement_radius", 2),
                    max_moves=body.get("max_moves", 40),
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == "/sealbot_move":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            try:
                import traceback as _tb
                result = _sealbot_move(
                    moves=body["moves"],
                    win_length=body.get("win_length", 5),
                    placement_radius=body.get("placement_radius", 2),
                    max_moves=body.get("max_moves", 40),
                    time_limit=body.get("time_limit", 0.5),
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                _tb.print_exc()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_error(404)


def main():
    global _model_config_dict
    parser = argparse.ArgumentParser(description="HeXO Policy Viewer")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default="configs/ablations/baseline.toml",
                        help="Curriculum TOML config (extracts checkpoint_dir and model config)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint dir from config")
    args = parser.parse_args()

    raw = tomllib.loads(Path(args.config).read_text())
    _model_config_dict = raw.get("model", {})
    ckpt_dir = args.checkpoint_dir or raw.get("run", {}).get("checkpoint_dir", "checkpoints")
    Handler.checkpoint_dir = ckpt_dir

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Policy viewer running at {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
