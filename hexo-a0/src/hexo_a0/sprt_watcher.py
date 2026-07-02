"""Trainer-side SPRT daemon lifecycle + state-file consumer.

Spawns `python -m hexo_a0.sprt_daemon` as a subprocess at stage start,
reads its JSON state file, and reports decisions back to the curriculum
loop. On stage end (or trainer shutdown) the subprocess is terminated.

The daemon itself handles checkpoint watching, game playing, and SPRT
math — this module is purely orchestration.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Run the daemon at a lower CPU priority than the trainer + self-play workers.
# Its CPU-side MCTS otherwise saturates ~12 cores on the shared APU and starves
# the 31 self-play threads (the real throughput bottleneck). nice caps its share
# under contention; SCHED_BATCH stops it preempting the self-play threads on
# wakeup. Together they let it yield cores yet still make gating progress.
SPRT_DAEMON_NICE = 10


def _deprioritise_daemon() -> None:
    """preexec_fn: lower the daemon's scheduling priority (runs in the child)."""
    try:
        os.sched_setscheduler(0, os.SCHED_BATCH, os.sched_param(0))
    except (AttributeError, OSError):
        pass  # non-Linux or kernel without SCHED_BATCH; nice still applies
    os.nice(SPRT_DAEMON_NICE)


@dataclass
class SPRTDecision:
    """Snapshot of daemon state at a given moment."""
    decision: str  # "continue" | "accept_h1" | "reject_h1"
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    llr: float
    timestamp: float
    trainee_path: str | None
    champion_mtime: float | None
    reject_count: int  # SPRT rounds terminated in reject_h1 since current champion was promoted
    pairs: int  # completed pentanomial pairs (0 in trinomial mode)
    empirical_pair_variance: float | None  # sample variance of pair scores in current window
    peak_history: list[int] = dataclasses.field(default_factory=list)  # reject_count peaks at each prior promotion in stage


class SPRTWatcher:
    """Lifecycle manager for the SPRT daemon subprocess.

    Usage:
        watcher = SPRTWatcher(...)
        watcher.start()
        while training:
            d = watcher.check()
            if d and d.decision == "accept_h1": ...
        watcher.stop()
    """

    def __init__(
        self,
        config_path: Path,
        trainee_dir: Path,
        champion_path: Path,
        state_file: Path,
        stage_index: int,
        s0: float,
        s1: float,
        alpha: float,
        beta: float,
        window_size: int,
        mcts_sims: int,
        mcts_m_actions: int,
        device: str,
        poll_interval: float,
        eval_workers: int = 1,
        pentanomial: bool = True,
        pair_variance: float = 0.5,
        adaptive_pair_variance: bool = False,
        adaptive_pair_variance_floor: float = 0.20,
        adaptive_pair_variance_ceil: float = 0.50,
        adaptive_pair_variance_ema_alpha: float = 0.30,
        stale_after: float = 300.0,
    ) -> None:
        self.config_path = config_path
        self.trainee_dir = trainee_dir
        self.champion_path = champion_path
        self.state_file = state_file
        self.stage_index = stage_index
        self.s0 = s0
        self.s1 = s1
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.mcts_sims = mcts_sims
        self.mcts_m_actions = mcts_m_actions
        self.device = device
        self.eval_workers = eval_workers
        self.poll_interval = poll_interval
        self.pentanomial = pentanomial
        self.pair_variance = pair_variance
        self.adaptive_pair_variance = adaptive_pair_variance
        self.adaptive_pair_variance_floor = adaptive_pair_variance_floor
        self.adaptive_pair_variance_ceil = adaptive_pair_variance_ceil
        self.adaptive_pair_variance_ema_alpha = adaptive_pair_variance_ema_alpha
        self.stale_after = stale_after
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    def start(self) -> None:
        """Launch the daemon subprocess.

        The existing state file (if any) is preserved so that long-running
        per-stage state (notably `peak_history`, which seeds the
        curriculum's auto-calibrated plateau detector) survives daemon
        restarts. The daemon atomically rewrites the state file on its
        first iteration, and `check()` has its own staleness guard for
        cases where the prior daemon's last-written decision is too old
        to trust.

        Daemon stdout/stderr are redirected to `sprt_daemon.log` next to the
        state file so its logging output does not interleave with the
        trainer's TUI display on the parent terminal.
        """
        if self._proc is not None and self._proc.poll() is None:
            logger.warning("SPRT daemon already running (pid=%d), skipping start", self._proc.pid)
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "hexo_a0.sprt_daemon",
            "--config", str(self.config_path),
            "--trainee-dir", str(self.trainee_dir),
            "--champion-path", str(self.champion_path),
            "--state-file", str(self.state_file),
            "--stage-index", str(self.stage_index),
            "--s0", str(self.s0),
            "--s1", str(self.s1),
            "--alpha", str(self.alpha),
            "--beta", str(self.beta),
            "--window-size", str(self.window_size),
            "--mcts-sims", str(self.mcts_sims),
            "--mcts-m-actions", str(self.mcts_m_actions),
            "--device", self.device,
            "--eval-workers", str(self.eval_workers),
            "--poll-interval", str(self.poll_interval),
            "--pair-variance", str(self.pair_variance),
            "--pentanomial" if self.pentanomial else "--trinomial",
        ]
        if self.adaptive_pair_variance:
            cmd += [
                "--adaptive-pair-variance",
                "--adaptive-pair-variance-floor", str(self.adaptive_pair_variance_floor),
                "--adaptive-pair-variance-ceil", str(self.adaptive_pair_variance_ceil),
                "--adaptive-pair-variance-ema-alpha", str(self.adaptive_pair_variance_ema_alpha),
            ]
        log_path = self.state_file.parent / "sprt_daemon.log"
        logger.info("Starting SPRT daemon (log: %s)", log_path)
        self._log_fh = open(log_path, "a", buffering=1)
        self._proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=_deprioritise_daemon,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the daemon subprocess (SIGTERM, then SIGKILL on timeout)."""
        if self._proc is None or self._proc.poll() is not None:
            self._proc = None
            return
        logger.info("Stopping SPRT daemon (pid=%d)", self._proc.pid)
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("SPRT daemon did not stop on SIGTERM; sending SIGKILL")
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._proc.wait(timeout=timeout)
        self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def check(self) -> SPRTDecision | None:
        """Return the current SPRT decision, or None if state is missing/stale."""
        if not self.state_file.exists():
            return None
        try:
            payload = json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        state = payload.get("state", {})
        decision = state.get("decision", "continue")
        ts = payload.get("timestamp", 0.0)
        age = time.time() - ts

        # Staleness only matters when the daemon is supposed to be making
        # progress (decision == "continue"). Terminal decisions (accept_h1,
        # reject_h1) are idle-by-design: the daemon stops writing while
        # waiting for the champion to change, so the timestamp naturally
        # freezes. Treating that as stale loses valid decisions.
        if decision == "continue" and age > self.stale_after:
            if self.is_running():
                # Daemon is up but hasn't finished a game yet — likely slow
                # or just started. Treat as "no decision yet."
                return None
            logger.warning("SPRT state file is stale (%.0fs old) and daemon not running",
                           age)
            return None
        ph = payload.get("peak_history") or []
        return SPRTDecision(
            decision=state.get("decision", "continue"),
            games=int(state.get("games", 0)),
            wins=int(state.get("wins", 0)),
            draws=int(state.get("draws", 0)),
            losses=int(state.get("losses", 0)),
            score=float(payload.get("score", 0.0)),
            llr=float(state.get("llr", 0.0)),
            timestamp=ts,
            trainee_path=payload.get("trainee_path"),
            champion_mtime=payload.get("champion_mtime"),
            reject_count=int(payload.get("reject_count", 0)),
            pairs=int(state.get("pairs", 0)),
            empirical_pair_variance=payload.get("empirical_pair_variance"),
            peak_history=[int(x) for x in ph if isinstance(x, int)],
        )
