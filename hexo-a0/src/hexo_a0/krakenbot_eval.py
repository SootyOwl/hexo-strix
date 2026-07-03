"""Evaluate HeXO model against the hosted KrakenBot via the 6-tac.com API.

Proof-of-concept: defaults to a single game. The endpoint at
``https://6-tac.com/api/v1/compute/best-move`` is operated by mteam88 (the
sealbot/Kraken author) — Modal compute on someone else's tab — so don't run
batched evals against it without their OK.

API contract (verified live 2026-05-12)::

    POST /api/v1/compute/best-move
    {"position": {"turnsJson": "<json string>"},
     "config":   {"botName":  "kraken"}}
    -> {"stones":[Cube,Cube],"modelVersion":"kraken_v1","positionId":"..."}

``turnsJson`` is JSON of ``{"turns":[{"stones":[Cube,Cube]}, ...]}`` — the
pair-move history AFTER the implicit (0,0) P1 opening. Cube hex coords are
``{"x":int,"y":int,"z":int}`` with ``x+y+z=0``; axial conversion is
``q=x, r=z`` (Red-Blob-Games convention).

Strength is fixed at the upstream's deploy-time ``KRAKEN_N_SIMS=200`` and is
not parameterizable from the API.
"""

from __future__ import annotations

import json
import logging
import time as _time

import requests

from hexo_a0.sealbot_eval import BatchInferenceServer, _win_rate_stats

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://6-tac.com/api/v1/compute/best-move"
DEFAULT_TIMEOUT = 60.0  # cold start ~11s; warm ~1-2s — generous headroom


def _axial_to_cube(q: int, r: int) -> dict[str, int]:
    return {"x": q, "y": -q - r, "z": r}


def _cube_to_axial(cube: dict) -> tuple[int, int]:
    return int(cube["x"]), int(cube["z"])


def build_turns_json(turns_so_far: list[list[tuple[int, int]]]) -> str:
    """Serialize accumulated pair-move history into six-tac's turnsJson format."""
    return json.dumps({
        "turns": [
            {"stones": [_axial_to_cube(q, r) for (q, r) in pair]}
            for pair in turns_so_far
        ]
    })


def turns_to_htttx(turns_so_far: list[list[tuple[int, int]]]) -> str:
    """Serialize pair-move history into HTTTX text (policy_viewer paste format).

    HTTTX implicitly assumes P1's (0,0) opening, so turns_so_far[0] is P2's
    first 2-stone turn (HTTTX turn 1), turns_so_far[1] is P1's first 2-stone
    turn (HTTTX turn 2), etc. Matches the format produced by
    ``hexo_a0.serving.htttx.serialize_htttx``, including its mirror into
    htttx.io's coordinate convention.
    """
    from hexo_a0.serving.htttx import mirror_axial

    body = "version[1];\n"
    for i, pair in enumerate(turns_so_far, start=1):
        if not pair:
            continue
        coords = "".join(f"[{mq},{mr}]" for mq, mr in (mirror_axial(q, r) for q, r in pair))
        body += f"{i}. {coords};\n"
    return body


def kraken_pair_move(
    turns_so_far: list[list[tuple[int, int]]],
    api_url: str = DEFAULT_API_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[tuple[int, int]]:
    """Call six-tac.com and return Kraken's two-stone response in axial coords."""
    payload = {
        "position": {"turnsJson": build_turns_json(turns_so_far)},
        "config": {"botName": "kraken"},
    }
    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return [_cube_to_axial(s) for s in data["stones"]]


def _play_one_game(
    *,
    game_idx: int,
    eval_fn,
    gc_dict: dict,
    n_simulations: int,
    m_actions: int,
    api_url: str,
    timeout: float,
) -> dict:
    """Play one game vs Kraken. Hexo is P1 on even game_idx, P2 on odd.

    Note: ``hexo_rs.GameState`` auto-seeds P1 at (0,0) on construction (P2 to
    move, 2 stones remaining), matching six-tac's implicit opening. The very
    first 2-stone turn in turnsJson is therefore P2's, not P1's.
    """
    import hexo_rs

    hexo_is_p1 = (game_idx % 2 == 0)
    hexo_player = "P1" if hexo_is_p1 else "P2"
    hexo_gc = hexo_rs.GameConfig(**gc_dict)
    mcts_config = hexo_rs.MCTSConfig(
        n_simulations=n_simulations,
        m_actions=m_actions,
        c_visit=50,
        c_scale=1.0,
    )

    gs = hexo_rs.GameState(hexo_gc)  # auto-seeds P1@(0,0); P2 to play

    turns_so_far: list[list[tuple[int, int]]] = []
    move_count = 0  # stones we apply in this loop, NOT counting the engine opening
    error: str | None = None

    while not gs.is_terminal() and move_count < gc_dict["max_moves"]:
        cur = gs.current_player()
        if cur is None:
            break
        is_hexo_turn = (cur == hexo_player)

        if is_hexo_turn:
            stones: list[tuple[int, int]] = []
            for _ in range(2):
                if gs.is_terminal() or move_count >= gc_dict["max_moves"]:
                    break
                action, _ = hexo_rs.gumbel_mcts(gs, eval_fn, mcts_config)
                gs.apply_move(action[0], action[1])
                stones.append((action[0], action[1]))
                move_count += 1
            turns_so_far.append(stones)
        else:
            try:
                pair = kraken_pair_move(turns_so_far, api_url=api_url, timeout=timeout)
                applied: list[tuple[int, int]] = []
                for q, r in pair:
                    if gs.is_terminal() or move_count >= gc_dict["max_moves"]:
                        break
                    gs.apply_move(q, r)
                    applied.append((q, r))
                    move_count += 1
                turns_so_far.append(applied)
            except Exception as exc:
                error = f"kraken move failed at turn {len(turns_so_far)}: {exc}"
                logger.warning(error)
                break

    winner = gs.winner() if gs.is_terminal() else None
    hexo_player = "P1" if hexo_is_p1 else "P2"
    if error is not None:
        # Treat any network/protocol error as a forfeit by hexo so that PoC
        # runs surface the failure rather than hiding it as a "draw".
        is_win, is_draw, is_loss = False, False, True
    else:
        is_win = winner == hexo_player
        is_draw = winner is None
        is_loss = (winner is not None) and (winner != hexo_player)

    return {
        "moves": move_count,
        "is_win": is_win,
        "is_loss": is_loss,
        "is_draw": is_draw,
        "p1_won": winner == "P1",
        "p2_won": winner == "P2",
        "hexo_side": hexo_player,
        "winner": winner,
        "error": error,
        "turns_json": build_turns_json(turns_so_far),
        "htttx": turns_to_htttx(turns_so_far),
    }


def evaluate_vs_krakenbot(
    model,
    game_config,
    device,
    n_games: int = 1,
    n_simulations: int = 64,
    m_actions: int = 16,
    model_config=None,
    api_url: str = DEFAULT_API_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Play the model against the hosted KrakenBot. Defaults to one game.

    Increase ``n_games`` only after the upstream operator (mteam88) has given
    permission — the endpoint is unauthenticated and the compute cost lands on
    them. Strength is fixed upstream at KRAKEN_N_SIMS=200.

    Returns the same dict shape as ``evaluate_vs_sealbot``, plus an ``errors``
    count and ``model_version`` string and a per-game ``games`` list with the
    final turnsJson (useful for replaying the position in policy_viewer).
    """
    was_training = model.training
    model.eval()

    gc_dict = {
        "win_length": game_config.win_length,
        "placement_radius": game_config.placement_radius,
        "max_moves": game_config.max_moves,
    }

    from hexo_a0.graph import game_to_axis_graph, game_to_graph

    graph_type = model_config.graph_type if model_config else "hex"
    prune = model_config.prune_empty_edges if model_config else False
    threat = getattr(model_config, "threat_features", False) if model_config else False
    rel = getattr(model_config, "relative_stone_encoding", False) if model_config else False
    graph_fn = (
        (lambda g: game_to_axis_graph(g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel))
        if graph_type == "axis"
        else (lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel))
    )

    start_wall = _time.time()
    logger.info(
        "Kraken eval starting: n_games=%d, sims=%d, m_actions=%d, api_url=%s",
        n_games, n_simulations, m_actions, api_url,
    )

    server = BatchInferenceServer(model, device, graph_fn)
    results: list[dict] = []
    try:
        for i in range(n_games):
            r = _play_one_game(
                game_idx=i,
                eval_fn=server.eval_fn,
                gc_dict=gc_dict,
                n_simulations=n_simulations,
                m_actions=m_actions,
                api_url=api_url,
                timeout=timeout,
            )
            results.append(r)
            outcome = "WIN" if r["is_win"] else ("LOSS" if r["is_loss"] else "DRAW")
            logger.info(
                "Kraken game %d/%d: %s (hexo=%s, winner=%s, moves=%d) [%.1fs]",
                i + 1, n_games, outcome, r["hexo_side"], r["winner"],
                r["moves"], _time.time() - start_wall,
            )
    finally:
        server.stop()

    if was_training:
        model.train()

    wins = sum(1 for r in results if r["is_win"])
    losses = sum(1 for r in results if r["is_loss"])
    draws = sum(1 for r in results if r["is_draw"])
    p1_wins = sum(1 for r in results if r["p1_won"])
    p2_wins = sum(1 for r in results if r["p2_won"])
    total_moves = sum(r["moves"] for r in results)
    errors = sum(1 for r in results if r["error"] is not None)

    win_rate, ci_lo, ci_hi, elo_diff = _win_rate_stats(wins, losses, draws)
    decided = p1_wins + p2_wins
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "p1_decided_rate": p1_wins / decided if decided > 0 else 0.0,
        "win_rate": win_rate,
        "elo_diff": elo_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_game_length": total_moves / max(n_games, 1),
        "errors": errors,
        "model_version": "kraken_v1",
        "games": results,
    }
