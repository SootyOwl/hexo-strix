"""Continuous SPRT evaluation daemon.

Runs as a long-lived subprocess alongside training. Watches a checkpoint
directory for the latest trainee and a fixed champion path. Plays games
one at a time (alternating trainee side), updates a JSON state file, and
resets statistics on champion change.

Phase 2a scope: standalone daemon that can be launched manually to watch
a live training run without any trainer modifications. Phase 2b will
have the trainer launch + consume this.

Usage (manual, while training runs):
    uv run python -m hexo_a0.sprt_daemon \\
        --config configs/gine-full/baseline.toml \\
        --trainee-dir runs/gine-champion-mini/checkpoints \\
        --champion-path runs/gine-champion-mini/checkpoints/self_play/champion.pt \\
        --state-file runs/gine-champion-mini/sprt_state.json \\
        --stage-index 2
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import tomllib
from pathlib import Path

import torch

from hexo_a0.config import ModelConfig
from hexo_a0.evaluate import play_eval_game, sample_opening


def _opening_generator_for_pair(opening_generator, pair_id, trainee_pack, champion_pack):
    """Pick the ``(model, model_config)`` that generates a pair's opening.

    ``"alternate"`` (default) uses the trainee on even pairs and the champion on
    odd — fine during healthy training where trainee ≈ champion (both in-domain).
    ``"champion"`` / ``"trainee"`` fix the generator (use ``"champion"`` if the
    trainee may be much weaker / out-of-domain, which would inject refuted
    openings).
    """
    if opening_generator == "champion":
        return champion_pack
    if opening_generator == "trainee":
        return trainee_pack
    return trainee_pack if pair_id % 2 == 0 else champion_pack


def _maybe_sample_opening(opening_generator, pair_id, trainee_pack, champion_pack,
                          game_config, device, opening_plies, opening_temperature):
    """Sample one shared opening for a pentanomial pair, or None if disabled.

    Seeded by ``pair_id`` so the two games of a pair use the identical opening.
    A sampling failure (no non-terminal opening found) falls back to no opening
    rather than crashing the daemon.
    """
    if opening_plies <= 0:
        return None
    gen_model, gen_mc = _opening_generator_for_pair(
        opening_generator, pair_id, trainee_pack, champion_pack)
    try:
        return sample_opening(gen_model, game_config, device, opening_plies,
                              opening_temperature, seed=pair_id, model_config=gen_mc)
    except RuntimeError as e:
        logging.getLogger(__name__).warning(
            "opening sample failed (pair %d): %s; playing without opening", pair_id, e)
        return None
from hexo_a0.model import HeXONet
from hexo_a0.sprt_eval import (
    SPRTConfig,
    SPRTState,
    checkpoint_step,
    restore_decision,
    sample_pair_variance,
    swap_decision,
)
from hexo_a0.sprt_parallel import PairCoordinator

logger = logging.getLogger("sprt_daemon")


def _torch_load_with_retry(path: Path, *, weights_only: bool):
    """torch.load with retry on partial-file zip read errors.

    For the race with concurrent checkpoint writes: even though the trainer
    now writes atomically, this is a defensive belt-and-suspenders for older
    runs or other writers. 5 retries with exponential backoff starting at 1s.
    Only catches the specific 'failed finding central directory' RuntimeError;
    other errors propagate immediately.
    """
    delay = 1.0
    last_err: RuntimeError | None = None
    for attempt in range(5):
        try:
            return torch.load(path, map_location="cpu", weights_only=weights_only)
        except RuntimeError as e:
            if "failed finding central directory" not in str(e):
                raise
            last_err = e
            logger.warning(
                "torch.load(%s) hit partial-write race (attempt %d/5); retrying in %.1fs",
                path, attempt + 1, delay,
            )
            time.sleep(delay)
            delay *= 2
    assert last_err is not None
    raise last_err


def _load_model(path: Path, fallback_mc: ModelConfig, device: torch.device) -> tuple[HeXONet, ModelConfig]:
    """Load a HeXONet checkpoint, preferring the checkpoint's own ModelConfig."""
    try:
        peek = _torch_load_with_retry(path, weights_only=True)
    except Exception:
        peek = _torch_load_with_retry(path, weights_only=False)
    mc_dict = peek.get("model_config", {}) if isinstance(peek, dict) else {}
    mc = ModelConfig(**mc_dict) if mc_dict else fallback_mc
    model = HeXONet(mc).to(device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in peek["model_state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, mc


def _latest_checkpoint(trainee_dir: Path) -> Path | None:
    """Return the most recently modified checkpoint_*.pt, or None."""
    candidates = sorted(trainee_dir.glob("checkpoint_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically via tempfile + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _resolve_game_config(raw: dict, stage_index: int):
    """Pull win_length / placement_radius / max_moves from the requested stage."""
    stages = raw.get("curriculum", {}).get("stages", [])
    stage = stages[stage_index] if stages and 0 <= stage_index < len(stages) else {}
    base = raw.get("game", {})
    win_length = stage.get("game_win_length", base.get("win_length", 6))
    radius = stage.get("game_placement_radius", base.get("placement_radius", 8))
    max_moves = stage.get("game_max_moves", base.get("max_moves", 200))
    return win_length, radius, max_moves


def _wait_for_file(path: Path, kind: str, poll: float = 2.0) -> None:
    """Poll until path exists. Logs once per attempt."""
    logged = False
    while not path.exists():
        if not logged:
            logger.info("Waiting for %s at %s", kind, path)
            logged = True
        time.sleep(poll)


def _pair_worker(
    worker_id: int,
    task_q,
    result_q,
    config_path: str,
    stage_index: int,
    mcts_sims: int,
    m_actions: int,
    opening_plies: int = 0,
    opening_temperature: float = 0.5,
    opening_generator: str = "alternate",
) -> None:
    """Worker process: play whole (P1, P2) pairs against the champion.

    Each task is ``(pair_id, trainee_path, champion_path, epoch)``. The worker
    plays the trainee as P1 then as P2 — a complete pentanomial pair — and
    returns ``(pair_id, epoch, p1_outcome, p2_outcome, total_moves)``. Models
    are cached: the trainee by path (checkpoint filenames carry the step), the
    champion by epoch (its path is fixed and overwritten in place on promotion).

    CPU-only by construction. Thread counts are pinned to 1 so N workers don't
    oversubscribe the cores (OMP is set in the parent before spawn; this is the
    runtime backstop). A ``None`` task is the poison pill.
    """
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # interop pool already started; not fatal

    import hexo_rs
    raw = tomllib.loads(Path(config_path).read_text())
    mc_fallback = ModelConfig(**raw["model"])
    win_length, radius, max_moves = _resolve_game_config(raw, stage_index)
    game_config = hexo_rs.GameConfig(win_length, radius, max_moves)
    device = torch.device("cpu")

    t_cache: dict = {}  # trainee_path -> (model, mc)
    c_cache: dict = {}  # epoch -> (model, mc)

    def _trainee(path: str):
        if path not in t_cache:
            t_cache.clear()
            t_cache[path] = _load_model(Path(path), mc_fallback, device)
        return t_cache[path]

    def _champion(path: str, epoch: int):
        if epoch not in c_cache:
            c_cache.clear()
            c_cache[epoch] = _load_model(Path(path), mc_fallback, device)
        return c_cache[epoch]

    while True:
        task = task_q.get()
        if task is None:
            return
        pair_id, trainee_path, champion_path, epoch = task
        trainee, tmc = _trainee(trainee_path)
        champion, cmc = _champion(champion_path, epoch)
        opening = _maybe_sample_opening(
            opening_generator, pair_id, (trainee, tmc), (champion, cmc),
            game_config, device, opening_plies, opening_temperature)
        outcomes = []
        total_moves = 0
        for side in ("P1", "P2"):
            result = play_eval_game(
                trainee, game_config, device,
                opponent=champion, model_side=side,
                model_config=tmc, opponent_config=cmc,
                mcts_sims=mcts_sims, mcts_m_actions=m_actions,
                opening=opening, disable_gumbel_noise=opening_plies > 0,
            )
            winner = result["winner"]
            outcomes.append("D" if winner is None else "W" if winner == side else "L")
            total_moves += result["moves"]
        result_q.put((pair_id, epoch, outcomes[0], outcomes[1], total_moves))


class _EvalPool:
    """Spawns and feeds the pair-eval worker processes.

    Owns per-worker task queues (so a dead worker's in-flight pairs are known
    and can be re-dispatched — a shared queue would lose that attribution) and
    one shared result queue. The pure dispatch/drain/epoch bookkeeping lives in
    PairCoordinator; this class is the process/queue plumbing around it.
    """

    def __init__(
        self,
        n_workers: int,
        config_path: Path,
        stage_index: int,
        mcts_sims: int,
        m_actions: int,
        poll_interval: float,
        pair_timeout: float = 600.0,
        opening_plies: int = 0,
        opening_temperature: float = 0.5,
        opening_generator: str = "alternate",
    ) -> None:
        # Pin BLAS/OMP thread pools to 1 before spawning so each worker's torch
        # import (which initialises OpenMP) sees it — setting it inside the
        # worker after import is too late. N workers × many threads would thrash
        # the APU cores and make the pool slower than serial. setdefault so an
        # operator who deliberately exported a value keeps it.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        # spawn (not fork): forking after libtorch/HIP init is a known crash
        # source in this stack.
        self._ctx = mp.get_context("spawn")
        self._worker_args = (config_path, stage_index, mcts_sims, m_actions,
                             opening_plies, opening_temperature, opening_generator)
        self.poll_interval = poll_interval
        self.result_q = self._ctx.Queue()
        self.task_qs = [self._ctx.Queue() for _ in range(n_workers)]
        self.procs = [self._spawn(i) for i in range(n_workers)]
        self.coord = PairCoordinator(n_workers, pair_timeout=pair_timeout)
        self._next_pair_id = 0
        self._rr = 0
        # Crash-loop backstop: a worker that dies on libtorch/HIP import will be
        # respawned every cycle forever, making zero progress. Cap total
        # respawns; past it, give up so the daemon exits and the watcher's
        # is_running() flips false (correctly reported as dead, not "slow").
        self._respawns = 0
        self._respawn_cap = max(20, 5 * n_workers)

    def _spawn(self, i: int):
        p = self._ctx.Process(
            target=_pair_worker,
            args=(i, self.task_qs[i], self.result_q, *self._worker_args),
            daemon=True,
        )
        p.start()
        return p

    def _dispatch(self, pair_id: int, trainee_path, champion_path, epoch: int, now: float) -> None:
        wid = self._rr % len(self.task_qs)
        self._rr += 1
        self.task_qs[wid].put((pair_id, str(trainee_path), str(champion_path), epoch))
        self.coord.on_dispatch(pair_id, wid, epoch, now)

    def pump(self, trainee_path, champion_path, epoch, decision) -> list:
        """One service cycle. Returns completed, epoch-current (side, outcome,
        moves) units to record — empty if nothing finished this cycle."""
        now = time.time()

        # 1. Reclaim dead workers; respawn their slots and re-dispatch their
        #    unreturned pairs (else the inflight slots leak and dispatch wedges).
        #    Replace the dead worker's queue before respawning: its unpulled
        #    tasks are reclaimed below and reissued, so leaving them in the
        #    reused queue would double-execute (the respawned worker would pull
        #    them too).
        dead = {i for i, p in enumerate(self.procs) if not p.is_alive()}
        if dead:
            self._respawns += len(dead)
            if self._respawns > self._respawn_cap:
                raise RuntimeError(
                    f"pair-eval workers respawned {self._respawns} times "
                    f"(cap {self._respawn_cap}); likely a crash loop — giving up"
                )
            for i in dead:
                logger.warning("pair-eval worker %d died; respawning (%d total)",
                               i, self._respawns)
                try:
                    self.task_qs[i].cancel_join_thread()
                except Exception:
                    pass
                self.task_qs[i] = self._ctx.Queue()
                self.procs[i] = self._spawn(i)
        reissue = self.coord.reclaim_dead(dead) if dead else []
        # 2. Reclaim silently-stuck pairs (worker alive but wedged in a native call).
        reissue += self.coord.reclaim_timed_out(now)
        for pair_id in reissue:
            self._dispatch(pair_id, trainee_path, champion_path, epoch, now)

        # 3. Top up to target (gated: no new work once decided).
        for _ in range(self.coord.dispatch_quota(decision)):
            self._dispatch(self._next_pair_id, trainee_path, champion_path, epoch, now)
            self._next_pair_id += 1

        # 4. Drain: block up to poll_interval for the first result, then sweep
        #    the rest non-blocking. Discard pairs played against a stale champion.
        units: list = []
        try:
            msg = self.result_q.get(timeout=self.poll_interval)
        except queue.Empty:
            return units
        while True:
            pair_id, played_epoch, o1, o2, moves = msg
            if self.coord.on_result(pair_id, played_epoch, epoch):
                units.append(("P1", o1, moves))
                units.append(("P2", o2, moves))
            try:
                msg = self.result_q.get_nowait()
            except queue.Empty:
                break
        return units

    def close(self) -> None:
        # Workers are daemon=True and may be mid-game (a queued poison pill sits
        # behind a backlog they'd have to play through first), so terminate
        # directly rather than waiting on a graceful drain.
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()
        # Drop buffered data so the queues' feeder threads don't block exit.
        for q in (*self.task_qs, self.result_q):
            try:
                q.cancel_join_thread()
            except Exception:
                pass


def run_daemon(
    config_path: Path,
    trainee_dir: Path,
    champion_path: Path,
    state_file: Path,
    stage_index: int,
    sprt_cfg: SPRTConfig,
    mcts_sims: int,
    mcts_m_actions: int,
    device_str: str,
    poll_interval: float,
    eval_workers: int = 1,
    opening_plies: int = 0,
    opening_temperature: float = 0.5,
    opening_generator: str = "alternate",
) -> None:
    import hexo_rs  # noqa: F401 — lazy import, same pattern as evaluate.py

    raw = tomllib.loads(config_path.read_text())
    mc_fallback = ModelConfig(**raw["model"])
    win_length, radius, max_moves = _resolve_game_config(raw, stage_index)
    game_config = hexo_rs.GameConfig(win_length, radius, max_moves)
    device = torch.device(device_str)

    logger.info("SPRT daemon starting")
    logger.info("Config: %s  stage_index=%d  game: L=%d r=%d moves=%d",
                config_path, stage_index, win_length, radius, max_moves)
    logger.info("SPRT: s0=%.3f s1=%.3f alpha=%.3f beta=%.3f",
                sprt_cfg.s0, sprt_cfg.s1, sprt_cfg.alpha, sprt_cfg.beta)
    logger.info("MCTS sims=%d device=%s", mcts_sims, device_str)
    if opening_plies > 0:
        logger.info("Openings: noise-OFF, plies=%d T=%.2g generator=%s",
                    opening_plies, opening_temperature, opening_generator)
    else:
        logger.info("Openings: disabled (legacy noise-on gate)")
    logger.info("State file: %s", state_file)

    _wait_for_file(champion_path, "champion")
    _wait_for_file(trainee_dir, "trainee dir")
    while _latest_checkpoint(trainee_dir) is None:
        logger.info("Waiting for trainee checkpoints in %s", trainee_dir)
        time.sleep(poll_interval)

    state = SPRTState()
    # Counts SPRT rounds that terminated in reject_h1 since the current
    # champion was last promoted. Resets to 0 on champion change.
    # Exposed in the state file so the trainer can gate stage advance
    # on a patience layer (sprt_reject_patience).
    reject_count = 0
    # Per-cycle reject_count peaks at each champion change (the peak
    # captured just before promo). The curriculum uses this for an
    # auto-calibrated dual-threshold plateau detector (warn_thr=⌈μ+σ⌉,
    # decl_thr=max+1) — see curriculum.py. Restored from any pre-existing
    # state file so a daemon restart mid-stage doesn't wipe accumulated
    # history. Requires the watcher to NOT unlink the state file on start.
    peak_history: list[int] = []
    # Lifetime game counter for side alternation and log identity. Distinct
    # from state.games, which is window-scoped and can shrink when the
    # sliding window drops old games. Resets on champion change and after
    # reject_h1 (matches state.reset() semantics).
    lifetime_games = 0
    if state_file.exists():
        try:
            prior = json.loads(state_file.read_text())
            ph = prior.get("peak_history")
            if isinstance(ph, list) and all(isinstance(x, int) for x in ph):
                peak_history = list(ph)
                logger.info("Restored peak_history from prior state file: %s", peak_history)

            # Restore in-flight round state when the prior daemon was
            # shut down mid-round against the same champion. Gates live in
            # restore_decision() (sprt_eval.py, unit-tested): same champion,
            # decision still "continue", and trainee staleness measured in
            # training steps — NOT wall-clock, which a stalled round erodes
            # before shutdown and a paused run trips spuriously.
            prior_state = prior.get("state") or {}
            prior_decision = prior_state.get("decision", "continue")
            prior_champion_mtime = prior.get("champion_mtime")
            current_champion_mtime = (
                champion_path.stat().st_mtime if champion_path.exists() else None
            )
            same_champion = (
                prior_champion_mtime is not None
                and current_champion_mtime is not None
                and abs(prior_champion_mtime - current_champion_mtime) < 1.0
            )
            latest_ckpt = _latest_checkpoint(trainee_dir)
            restore_ok, skip_reasons = restore_decision(
                prior_decision=prior_decision,
                same_champion=same_champion,
                prior_trainee_step=checkpoint_step(prior.get("trainee_path", "")),
                latest_trainee_step=(
                    checkpoint_step(latest_ckpt) if latest_ckpt is not None else None
                ),
            )
            if restore_ok and prior_state.get("games", 0) > 0:
                state = SPRTState.from_dict(prior_state)
                reject_count = int(prior.get("reject_count", 0))
                # Use the post-restore count: from_dict may drop a trailing odd
                # outcome, so the serialized games value can be one too high —
                # which would flip the serial side-alternation parity.
                lifetime_games = state.games
                logger.info(
                    "Restored mid-round state: games=%d wins=%d draws=%d losses=%d "
                    "llr=%.3f reject_count=%d (same champion, trainee current)",
                    state.games, state.wins, state.draws, state.losses, state.llr,
                    reject_count,
                )
            elif same_champion and prior.get("reject_count", 0) > 0:
                # Even if we don't restore the in-flight round (e.g. prior
                # decision was terminal, or games=0), reject_count is still
                # valid as long as the champion hasn't changed.
                reject_count = int(prior.get("reject_count", 0))
                logger.info(
                    "Restored reject_count=%d against current champion (prior round "
                    "decision=%s, not restoring round state)",
                    reject_count, prior_decision,
                )
            elif prior_state.get("games", 0) > 0:
                # A mid-round state existed but failed a restore gate. Say
                # which one — a silently discarded round looks like data loss
                # to the operator (live incident 2026-06-11: 238-game round at
                # LLR 4.43 dropped silently by the old wall-clock guard).
                logger.warning(
                    "Prior mid-round state (games=%d, llr=%.3f) NOT restored: %s "
                    "— starting a fresh round",
                    prior_state.get("games", 0), prior_state.get("llr", 0.0),
                    "; ".join(skip_reasons),
                )
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.warning("Could not restore from %s: %s — starting fresh", state_file, e)
    # Adaptive pair variance state: original cfg value to restore on champion
    # change, plus running EMA estimate. None ema means "no rounds completed
    # since last reset, still using initial pair_variance".
    initial_pair_variance = sprt_cfg.pair_variance
    ema_pair_variance: float | None = None
    champion_mtime: float | None = None
    # Monotonic champion generation. Bumped on every champion change so parallel
    # workers can tag each pair with the champion it was played against; pairs
    # that come back after a swap (stale epoch) are discarded rather than scored
    # into the fresh round. Unused in serial mode.
    champion_epoch = 0
    champion = None
    champion_mc: ModelConfig | None = None
    trainee_path: Path | None = None
    trainee = None
    trainee_mc: ModelConfig | None = None

    lower, upper = sprt_cfg.bounds()
    logger.info("Wald bounds: reject_h1 if LLR<=%.3f, accept_h1 if LLR>=%.3f", lower, upper)

    # Trainee-swap freeze zone: once a converging-toward-accept round
    # crosses LLR >= swap_freeze_fraction * upper (0.9 by default), stop
    # loading new trainee checkpoints until the round closes (accept_h1 /
    # reject_h1) or LLR drifts back below the threshold. Prevents a
    # near-promoting round from being diluted by a weaker mid-training
    # snapshot when the trainee oscillates across an offensive-breakthrough
    # boundary. The narrow 0.9 fraction keeps the freeze close to the accept
    # bound so the daemon keeps swapping in fresh checkpoints for a current
    # view of the network until a round is genuinely about to close.
    #
    # Asymmetric: we do NOT freeze on the lower-bound side. Approaching
    # reject is something the patience layer wants resolved quickly, and
    # letting a potentially-stronger newer trainee swap in mid-descent
    # may either avert the reject or hit it faster — both fine outcomes.
    #
    # Unfreeze-on-stall: a round can plateau in the freeze zone just below
    # the accept bound (the residual dead band) and never close. To avoid a
    # permanent stall, `frozen_games` counts consecutive games spent in the
    # freeze zone; once it reaches a full window the swap is force-allowed so
    # a fresher (possibly clearly-stronger) checkpoint can replace the stuck
    # trainee. swap_decision() encapsulates this logic (tested in
    # test_sprt_eval.py).
    swap_freeze_upper = sprt_cfg.swap_freeze_threshold()
    swap_frozen_logged = False
    frozen_games = 0
    current_opening = None  # serial path: shared opening cached across a pair

    state_file.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def _signal_handler(signum, _frame):
        nonlocal stop
        logger.info("Received signal %d, shutting down after current game", signum)
        stop = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    pool = None
    if eval_workers > 1:
        logger.info("Starting parallel pair-eval pool: %d workers", eval_workers)
        pool = _EvalPool(
            n_workers=eval_workers,
            config_path=config_path,
            stage_index=stage_index,
            mcts_sims=mcts_sims,
            m_actions=mcts_m_actions,
            poll_interval=poll_interval,
            opening_plies=opening_plies,
            opening_temperature=opening_temperature,
            opening_generator=opening_generator,
        )

    while not stop:
        # --- Champion sync ---
        current_mtime = champion_path.stat().st_mtime
        if champion_mtime is None or current_mtime != champion_mtime:
            if champion_mtime is not None:
                # The reject_count value at champion-change time is the
                # peak of the just-completed cycle. Record it for the
                # curriculum's auto-calibrated plateau detector.
                # (Persisted via the next _atomic_write_json below.)
                peak_history.append(reject_count)
                logger.info("Champion changed (mtime %.0f → %.0f), resetting SPRT state "
                            "(prior reject_count=%d, peak_history=%s)",
                            champion_mtime, current_mtime, reject_count, peak_history)
                state.reset()
                reject_count = 0
                lifetime_games = 0
                swap_frozen_logged = False
                frozen_games = 0
                # Reset adaptive pair variance to its initial config value;
                # new champion = new dynamics, prior empirical estimates are stale.
                if sprt_cfg.adaptive_pair_variance:
                    sprt_cfg.pair_variance = initial_pair_variance
                    ema_pair_variance = None
                    logger.info(
                        "Adaptive pair variance reset to initial %.3f for new champion",
                        initial_pair_variance,
                    )
            champion, champion_mc = _load_model(champion_path, mc_fallback, device)
            champion_mtime = current_mtime
            champion_epoch += 1
            logger.info("Loaded champion: %s (epoch %d)", champion_path, champion_epoch)

        # --- Trainee sync (don't reset state on swap) ---
        # Freeze swaps in the upper near-decision zone so a converging
        # round isn't diluted by a weaker mid-training snapshot.
        # The freeze is about preventing mid-round swaps to a DIFFERENT
        # trainee — it must not block the initial load on daemon startup
        # or resume, otherwise we deadlock with `trainee is None` forever
        # if the restored LLR is already in the freeze zone.
        freeze_active, stall_unfreeze = swap_decision(state.llr, frozen_games, sprt_cfg)
        if trainee is None:
            latest = _latest_checkpoint(trainee_dir)
            if latest is not None:
                trainee, trainee_mc = _load_model(latest, mc_fallback, device)
                trainee_path = latest
                if freeze_active:
                    logger.info(
                        "Initial trainee load (LLR=%.3f in freeze zone but no "
                        "trainee yet, bypassing): %s",
                        state.llr, trainee_path,
                    )
                    swap_frozen_logged = True
                else:
                    logger.info("Loaded trainee: %s", trainee_path)
        elif freeze_active:
            if not swap_frozen_logged:
                logger.info(
                    "Trainee-swap frozen: LLR=%.3f >= %.3f "
                    "(near-accept zone); current trainee %s held until round closes",
                    state.llr, swap_freeze_upper, trainee_path,
                )
                swap_frozen_logged = True
        else:
            if swap_frozen_logged:
                if stall_unfreeze:
                    logger.info(
                        "Trainee-swap stall-unfreeze: LLR=%.3f held %d games "
                        "(>= window) in freeze zone without closing; allowing swap "
                        "to a fresher checkpoint",
                        state.llr, frozen_games,
                    )
                else:
                    logger.info(
                        "Trainee-swap unfrozen: LLR=%.3f back below %.3f",
                        state.llr, swap_freeze_upper,
                    )
                swap_frozen_logged = False
            latest = _latest_checkpoint(trainee_dir)
            if latest is not None and latest != trainee_path:
                trainee, trainee_mc = _load_model(latest, mc_fallback, device)
                trainee_path = latest
                # Fresh trainee gets a full window before another stall check.
                frozen_games = 0
                logger.info("Loaded trainee: %s", trainee_path)

        if trainee is None or champion is None:
            time.sleep(poll_interval)
            continue

        # --- Produce this iteration's completed unit(s) ---
        # Serial: play one game inline. Parallel: drain whole (P1, P2) pairs
        # from the worker pool (epoch-filtered so stale-champion pairs are
        # already discarded). `pending` is a list of (side, outcome, moves).
        if pool is None:
            side = "P1" if lifetime_games % 2 == 0 else "P2"
            # Start of a pair (P1): sample one opening, reused for the P2 game.
            if opening_plies > 0 and lifetime_games % 2 == 0:
                current_opening = _maybe_sample_opening(
                    opening_generator, lifetime_games // 2,
                    (trainee, trainee_mc), (champion, champion_mc),
                    game_config, device, opening_plies, opening_temperature)
            result = play_eval_game(
                trainee, game_config, device,
                opponent=champion, model_side=side,
                model_config=trainee_mc, opponent_config=champion_mc,
                mcts_sims=mcts_sims, mcts_m_actions=mcts_m_actions,
                opening=current_opening if opening_plies > 0 else None,
                disable_gumbel_noise=opening_plies > 0,
            )
            winner = result["winner"]
            if winner is None:
                outcome = "D"
            elif winner == side:
                outcome = "W"
            else:
                outcome = "L"
            pending = [(side, outcome, result["moves"])]
        else:
            pending = pool.pump(
                trainee_path=trainee_path, champion_path=champion_path,
                epoch=champion_epoch, decision=state.decision,
            )
            if not pending:
                continue

        for side, outcome, moves in pending:
            lifetime_games += 1
            state.record(outcome, sprt_cfg)

            # Track consecutive games in the swap-freeze zone for unfreeze-on-stall.
            if state.llr >= swap_freeze_upper:
                frozen_games += 1
            else:
                frozen_games = 0

            # --- Emit state ---
            payload = {
                "timestamp": time.time(),
                "trainee_path": str(trainee_path),
                "champion_path": str(champion_path),
                "champion_mtime": champion_mtime,
                "sprt_config": {
                    "s0": sprt_cfg.s0, "s1": sprt_cfg.s1,
                    "alpha": sprt_cfg.alpha, "beta": sprt_cfg.beta,
                    "variance": sprt_cfg.variance,
                    "pair_variance": sprt_cfg.pair_variance,
                    "pentanomial": sprt_cfg.pentanomial,
                },
                "state": state.to_dict(),
                "score": state.score,
                "bounds": {"lower": lower, "upper": upper},
                "last_game": {"side": side, "outcome": outcome, "moves": moves},
                "reject_count": reject_count,
                "peak_history": peak_history,
                "empirical_pair_variance": (
                    sample_pair_variance(state._outcomes) if sprt_cfg.pentanomial else None
                ),
            }
            _atomic_write_json(state_file, payload)

            logger.info("game=%d side=%s %s  window W-D-L=%d-%d-%d  score=%.3f  LLR=%.3f  %s",
                        lifetime_games, side, outcome, state.wins, state.draws, state.losses,
                        state.score, state.llr, state.decision)

            # accept_h1: idle until champion changes (trainer promotes → fresh
            # round against new champion). reject_h1: log, increment counter,
            # reset state, and immediately start a new round against the same
            # champion — the trainer uses reject_count for patience-gated stage
            # advance rather than treating a single reject as terminal.
            if sprt_cfg.pentanomial:
                emp_var = sample_pair_variance(state._outcomes)
                if emp_var is not None and state.decision != "continue":
                    logger.info(
                        "Empirical pair variance: %.3f (configured pair_variance=%.3f; "
                        "lower empirical → could lower config for faster convergence)",
                        emp_var, sprt_cfg.pair_variance,
                    )
                    if sprt_cfg.adaptive_pair_variance:
                        a = sprt_cfg.adaptive_pair_variance_ema_alpha
                        if ema_pair_variance is None:
                            ema_pair_variance = emp_var
                        else:
                            ema_pair_variance = (1 - a) * ema_pair_variance + a * emp_var
                        clamped = max(
                            sprt_cfg.adaptive_pair_variance_floor,
                            min(sprt_cfg.adaptive_pair_variance_ceil, ema_pair_variance),
                        )
                        logger.info(
                            "Adaptive: ema_pair_variance=%.3f → clamped %.3f "
                            "(floor=%.2f ceil=%.2f); applying for next round",
                            ema_pair_variance, clamped,
                            sprt_cfg.adaptive_pair_variance_floor,
                            sprt_cfg.adaptive_pair_variance_ceil,
                        )
                        sprt_cfg.pair_variance = clamped

            if state.decision == "accept_h1":
                logger.info("Decision accept_h1 reached — idling until champion changes")
                # Heartbeat the state file every ~60s while idle so the trainer's
                # staleness check (if strict) doesn't misinterpret idle as dead.
                last_heartbeat = time.time()
                while not stop:
                    time.sleep(poll_interval)
                    current_mtime = champion_path.stat().st_mtime
                    if current_mtime != champion_mtime:
                        break
                    if time.time() - last_heartbeat > 60.0:
                        payload["timestamp"] = time.time()
                        _atomic_write_json(state_file, payload)
                        last_heartbeat = time.time()
                # Champion changed (or stopping): drop any remaining drained
                # units in this batch — they belong to the just-superseded
                # round — and let the outer loop re-sync + reset.
                break
            elif state.decision == "reject_h1":
                reject_count += 1
                logger.info(
                    "Decision reject_h1 reached (round %d, final score=%.3f LLR=%.3f "
                    "over %d games) — resetting SPRT state and starting new round",
                    reject_count, state.score, state.llr, state.games,
                )
                state.reset()
                lifetime_games = 0
                swap_frozen_logged = False
                frozen_games = 0
                # Drop the rest of this batch so a fresh round starts clean.
                break

    logger.info("SPRT daemon exiting after %d games (final LLR=%.3f, decision=%s)",
                state.games, state.llr, state.decision)
    if pool is not None:
        pool.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous SPRT eval daemon")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--trainee-dir", required=True, type=Path,
                    help="Directory containing checkpoint_*.pt files")
    ap.add_argument("--champion-path", required=True, type=Path)
    ap.add_argument("--state-file", required=True, type=Path)
    ap.add_argument("--stage-index", type=int, default=-1,
                    help="Curriculum stage for game config (-1 = last)")
    ap.add_argument("--s0", type=float, default=0.50)
    ap.add_argument("--s1", type=float, default=0.55)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--window-size", type=int, default=1000,
                    help="Sliding window for SPRT state (in games). Must exceed the "
                         "~412-game accept horizon at s1 or a stronger trainee never "
                         "promotes (dead band). 0 = unbounded.")
    ap.add_argument("--pentanomial", action="store_true", default=True,
                    help="Pair consecutive P1/P2 games and use 5-bucket pair scores "
                         "as the statistical unit (default: on).")
    ap.add_argument("--trinomial", dest="pentanomial", action="store_false",
                    help="Use per-game trinomial scores instead of pentanomial pairs.")
    ap.add_argument("--pair-variance", type=float, default=0.35,
                    help="Per-pair score variance for pentanomial LLR (default 0.35, "
                         "matching chess-engine-testing convention).")
    ap.add_argument("--adaptive-pair-variance", action="store_true",
                    help="Adaptively track pair variance via EMA of round-end "
                         "empirical estimates, clamped to [floor, ceil].")
    ap.add_argument("--adaptive-pair-variance-floor", type=float, default=0.20)
    ap.add_argument("--adaptive-pair-variance-ceil", type=float, default=0.50)
    ap.add_argument("--adaptive-pair-variance-ema-alpha", type=float, default=0.30,
                    help="EMA weight on the latest empirical observation.")
    ap.add_argument("--mcts-sims", type=int, default=16)
    ap.add_argument("--mcts-m-actions", type=int, default=16)
    ap.add_argument("--opening-plies", type=int, default=8,
                    help="Noise-off SPRT: sample this many opening plies per pair "
                         "(replayed into both swapped-side games) and play the rest "
                         "with Gumbel noise OFF. 0 = legacy noise-on gate.")
    ap.add_argument("--opening-temperature", type=float, default=0.5)
    ap.add_argument("--opening-generator", default="alternate",
                    choices=["alternate", "champion", "trainee"],
                    help="Which model samples the pair's opening ('alternate' = "
                         "trainee on even pairs, champion on odd).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-workers", type=int, default=1,
                    help="Parallel pair-eval worker processes. 1 = inline serial. "
                         ">1 clamped to 1 unless --device cpu.")
    ap.add_argument("--poll-interval", type=float, default=2.0,
                    help="Seconds between champion/trainee dir polls when idle")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    window = None if args.window_size <= 0 else args.window_size
    sprt_cfg = SPRTConfig(
        s0=args.s0, s1=args.s1, alpha=args.alpha, beta=args.beta,
        window_size=window,
        pentanomial=args.pentanomial,
        pair_variance=args.pair_variance,
        adaptive_pair_variance=args.adaptive_pair_variance,
        adaptive_pair_variance_floor=args.adaptive_pair_variance_floor,
        adaptive_pair_variance_ceil=args.adaptive_pair_variance_ceil,
        adaptive_pair_variance_ema_alpha=args.adaptive_pair_variance_ema_alpha,
    )
    # >1 workers on a GPU device = N processes each at batch-1 contending for
    # the same accelerator (the occupancy-contention failure mode). Keep the
    # pool CPU-only; clamp to inline serial otherwise.
    eval_workers = args.eval_workers
    if eval_workers > 1 and args.device != "cpu":
        logging.getLogger(__name__).warning(
            "eval_workers=%d requested on device=%s; clamping to 1 "
            "(parallel pair-eval is CPU-only).", eval_workers, args.device,
        )
        eval_workers = 1

    run_daemon(
        config_path=args.config,
        trainee_dir=args.trainee_dir,
        champion_path=args.champion_path,
        state_file=args.state_file,
        stage_index=args.stage_index,
        sprt_cfg=sprt_cfg,
        mcts_sims=args.mcts_sims,
        mcts_m_actions=args.mcts_m_actions,
        device_str=args.device,
        poll_interval=args.poll_interval,
        eval_workers=eval_workers,
        opening_plies=args.opening_plies,
        opening_temperature=args.opening_temperature,
        opening_generator=args.opening_generator,
    )


if __name__ == "__main__":
    main()
