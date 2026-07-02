"""Evaluate HeXO model against the Sealbot minimax engine.

Always uses a BatchInferenceServer: a dedicated thread that collects eval
requests from game threads, batches them onto the GPU, and returns results.
With workers=1 the games run in a serial loop sharing the same server (no
batching across games, but the model is loaded once); with workers>1 a
ThreadPoolExecutor runs games concurrently and the server batches their
requests. Computes win rate, Elo estimate, and Wilson confidence interval.

Requires the Sealbot repo to be cloned alongside the hexo project:
    hexo/
      hexo-a0/
      hexo-rs/
      sealbot/   ← https://github.com/Ramora0/SealBot (branch master),
                   the C++ SealBot minimax engine. Run its build.sh so that
                   best/ contains the compiled minimax_cpp module.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import queue
import sys
import threading
import time as _time
from pathlib import Path

import torch
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)

# Sealbot repo path — resolved relative to this file's location
_SEALBOT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "sealbot"


def _ensure_sealbot():
    """Add Sealbot directory to sys.path if available."""
    if not _SEALBOT_DIR.is_dir():
        raise FileNotFoundError(
            f"Sealbot repo not found at {_SEALBOT_DIR}. "
            "Clone https://github.com/Ramora0/SealBot there and run build.sh."
        )
    sealbot_str = str(_SEALBOT_DIR)
    if sealbot_str not in sys.path:
        sys.path.insert(0, sealbot_str)
    # The compiled module lives in best/
    current_str = str(_SEALBOT_DIR / "best")
    if current_str not in sys.path:
        sys.path.insert(0, current_str)


class BatchInferenceServer:
    """Collects eval requests from game threads, batches on GPU, returns results.

    Game threads call ``eval_fn(states)`` which blocks until the batch
    inference thread has processed the request.  The server thread waits
    briefly to accumulate requests from multiple game threads, then runs a
    single batched forward pass.
    """

    def __init__(self, model, device, graph_fn, max_batch_wait: float = 0.005):
        self._model = model
        self._device = device
        self._graph_fn = graph_fn
        self._max_batch_wait = max_batch_wait
        self._queue: queue.Queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def eval_fn(self, states):
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((states, future))
        return future.result()

    def _serve_loop(self):
        while self._running:
            batch: list[tuple] = []
            try:
                item = self._queue.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                continue
            deadline = _time.perf_counter() + self._max_batch_wait
            while _time.perf_counter() < deadline:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    if len(batch) >= 8:
                        break
                    _time.sleep(0.0005)

            all_states = []
            splits = []
            for states, _ in batch:
                splits.append(len(states))
                all_states.extend(states)

            try:
                graphs = [self._graph_fn(s) for s in all_states]
                pyg_batch = Batch.from_data_list(graphs).to(self._device)
                with torch.inference_mode():
                    logits_list, values = self._model.forward_batch(pyg_batch)

                # Sync once for the whole forward pass: concat the per-state
                # logits, transfer all logits and all values together, then
                # split back to per-future chunks. Avoids 2N CUDA syncs (one
                # .cpu() per logit + one .item() per value) in favour of 2.
                sizes = [pl.shape[0] for pl in logits_list]
                starts = [0]
                for s in sizes:
                    starts.append(starts[-1] + s)
                flat_logits = (
                    torch.cat(logits_list, dim=0).cpu().tolist()
                    if logits_list else []
                )
                flat_values = values.cpu().tolist()

                state_off = 0
                for (_, future), n in zip(batch, splits):
                    chunk_logits = [
                        flat_logits[starts[state_off + i]:starts[state_off + i + 1]]
                        for i in range(n)
                    ]
                    chunk_values = flat_values[state_off:state_off + n]
                    future.set_result((chunk_logits, chunk_values))
                    state_off += n
            except Exception as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)

    def stop(self):
        self._running = False
        self._thread.join(timeout=5)


def _play_one_game_threaded(args: dict) -> dict:
    """Play a single SealBot game using a shared eval_fn (thread-safe).

    Does not load a model — uses the ``eval_fn`` callable provided in args,
    which points to a ``BatchInferenceServer`` shared across all games in
    the current evaluation. The serial (workers=1) and parallel paths both
    flow through this function.
    """
    import hexo_rs
    from game import HexGame, Player as SBPlayer
    from minimax_cpp import MinimaxBot as SBMinimaxBot

    game_idx = args["game_idx"]
    eval_fn = args["eval_fn"]
    gc_dict = args["gc_dict"]
    n_simulations = args["n_simulations"]
    m_actions = args["m_actions"]
    sealbot_time_limit = args["sealbot_time_limit"]

    sealbot = SBMinimaxBot(time_limit=sealbot_time_limit)
    hexo_gc = hexo_rs.GameConfig(**gc_dict)
    mcts_config = hexo_rs.MCTSConfig(
        n_simulations=n_simulations,
        m_actions=m_actions,
        c_visit=50,
        c_scale=1.0,
    )

    hexo_is_a = (game_idx % 2 == 0)
    game = HexGame(win_length=gc_dict["win_length"])
    gs = hexo_rs.GameState(hexo_gc)
    moves = 0

    while not game.game_over and moves < gc_dict["max_moves"]:
        is_hexo_turn = (
            (hexo_is_a and game.current_player == SBPlayer.A)
            or (not hexo_is_a and game.current_player == SBPlayer.B)
        )
        if is_hexo_turn:
            if game.move_count == 0:
                gs = hexo_rs.GameState(hexo_gc)
                game.make_move(0, 0)
                moves += 1
                continue
            action, _ = hexo_rs.gumbel_mcts(gs, eval_fn, mcts_config)
            gs.apply_move(action[0], action[1])
            game.make_move(action[0], action[1])
            moves += 1
        else:
            if game.move_count == 0:
                game.make_move(0, 0)
                moves += 1
                gs = hexo_rs.GameState(hexo_gc)
                continue
            result = sealbot.get_move(game)
            moves_to_apply = result if sealbot.pair_moves else [result]
            for m in moves_to_apply:
                if game.game_over:
                    break
                game.make_move(m[0], m[1])
                moves += 1
                gs.apply_move(m[0], m[1])

    is_win = (
        (game.winner != SBPlayer.NONE)
        and (
            (hexo_is_a and game.winner == SBPlayer.A)
            or (not hexo_is_a and game.winner == SBPlayer.B)
        )
    )
    is_draw = game.winner == SBPlayer.NONE
    return {
        "moves": moves,
        "is_win": is_win,
        "is_loss": (not is_win) and (not is_draw),
        "is_draw": is_draw,
        "p1_won": game.winner == SBPlayer.A,
        "p2_won": game.winner == SBPlayer.B,
    }


def _win_rate_stats(wins: int, losses: int, draws: int):
    """Win rate, 95% Wilson CI, and Elo difference.

    Tournament scoring: win=1, draw=0.5, loss=0.
    """
    n = wins + losses + draws
    if n == 0:
        return 0.5, 0.0, 1.0, 0.0
    score = wins + 0.5 * draws
    p_hat = score / n
    z = 1.96
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n) / denom
    ci_lo = max(0.0, centre - spread)
    ci_hi = min(1.0, centre + spread)
    # Jeffreys prior for Elo: add 0.5 wins and 0.5 losses to avoid inf
    elo_p = (score + 0.5) / (n + 1.0)
    elo_diff = -400 * math.log10(1.0 / elo_p - 1.0)
    return p_hat, ci_lo, ci_hi, elo_diff


def evaluate_vs_sealbot(
    model,
    game_config,
    device,
    n_games: int = 10,
    sealbot_time_limit: float = 0.5,
    n_simulations: int = 64,
    m_actions: int = 16,
    model_config: object = None,
    workers: int = 1,
) -> dict:
    """Play the model against Sealbot and return evaluation metrics.

    Args:
        model:              HeXONet in eval mode.
        game_config:        hexo_rs.GameConfig.
        device:             Torch device for model inference.
        n_games:            Number of games to play (half as each side).
        sealbot_time_limit: Time limit per move for Sealbot's minimax.
        workers:            Parallel game threads. 1 = sequential; >1 uses a
                            ThreadPoolExecutor of game threads sharing the
                            single BatchInferenceServer.

    Returns:
        Dict with keys: wins, losses, draws, win_rate, elo_diff,
        ci_lo, ci_hi, mean_game_length.
    """
    _ensure_sealbot()

    was_training = model.training
    model.eval()

    gc_dict = {
        "win_length": game_config.win_length,
        "placement_radius": game_config.placement_radius,
        "max_moves": game_config.max_moves,
    }

    start_wall = _time.time()
    logger.info(
        "Sealbot eval starting: %d games, time_limit=%.3fs, sims=%d, workers=%d, device=%s",
        n_games, sealbot_time_limit, n_simulations, workers, str(device),
    )

    from hexo_a0.graph import game_to_graph, game_to_axis_graph

    graph_type = model_config.graph_type if model_config else "hex"
    prune = model_config.prune_empty_edges if model_config else False
    threat = getattr(model_config, "threat_features", False) if model_config else False
    rel = getattr(model_config, "relative_stone_encoding", False) if model_config else False
    graph_fn = (
        (lambda g: game_to_axis_graph(g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel))
        if graph_type == "axis"
        else (lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel))
    )

    server = BatchInferenceServer(model, device, graph_fn)
    results: list[dict] = []
    try:
        tasks = [
            {
                "game_idx": i,
                "eval_fn": server.eval_fn,
                "gc_dict": gc_dict,
                "n_simulations": n_simulations,
                "m_actions": m_actions,
                "sealbot_time_limit": sealbot_time_limit,
            }
            for i in range(n_games)
        ]
        if workers <= 1:
            for i, t in enumerate(tasks, start=1):
                results.append(_play_one_game_threaded(t))
                if i % max(1, n_games // 10) == 0 or i == n_games:
                    logger.info(
                        "Sealbot eval progress: %d/%d games (%.0fs elapsed)",
                        i, n_games, _time.time() - start_wall,
                    )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_play_one_game_threaded, t) for t in tasks]
                for f in concurrent.futures.as_completed(futures):
                    results.append(f.result())
                    if len(results) % max(1, n_games // 10) == 0 or len(results) == n_games:
                        logger.info(
                            "Sealbot eval progress: %d/%d games (%.0fs elapsed)",
                            len(results), n_games, _time.time() - start_wall,
                        )
    finally:
        server.stop()

    elapsed = _time.time() - start_wall
    logger.info(
        "Sealbot eval done: %d games in %.1fs (%.2f g/s, %.1fs/game avg)",
        n_games, elapsed, n_games / elapsed if elapsed > 0 else 0.0,
        elapsed / max(n_games, 1),
    )

    wins = sum(1 for r in results if r["is_win"])
    losses = sum(1 for r in results if r["is_loss"])
    draws = sum(1 for r in results if r["is_draw"])
    p1_wins = sum(1 for r in results if r["p1_won"])
    p2_wins = sum(1 for r in results if r["p2_won"])
    total_moves = sum(r["moves"] for r in results)

    if was_training:
        model.train()

    win_rate, ci_lo, ci_hi, elo_diff = _win_rate_stats(wins, losses, draws)
    mean_length = total_moves / max(n_games, 1)
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
        "mean_game_length": mean_length,
    }
