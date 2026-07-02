"""Two-line overwriting terminal display for training progress."""

import logging
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SPINNER_PLACEHOLDER = "\x00SPIN\x00"

# ANSI colour codes
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _color_delta(current: str, previous: float | None, lower_is_better: bool = True,
                  deadband: float = 0.02) -> str:
    """Colorize a value based on whether it improved or worsened.

    Changes within `deadband` (as a fraction of the previous value) are
    left uncoloured to reduce visual noise.
    """
    try:
        val = float(current)
    except (ValueError, TypeError):
        return current
    if previous is None or previous == 0.0:
        return current
    if abs(val - previous) / abs(previous) < deadband:
        return current
    improved = (val < previous) if lower_is_better else (val > previous)
    color = _GREEN if improved else _RED
    return f"{color}{current}{_RESET}"


class TrainingDisplay:
    """Manages a two-line terminal display: eval line + status line with spinner.

    The display overwrites itself in place. Stage headers and important events
    are printed above and persist. Everything is also sent to a log file when
    configured via setup_file_logging().

    The spinner appears on whichever line is currently "active" — the status
    line during training, and the eval line during evaluation. A background
    thread redraws every 200ms to keep the spinner animated.
    """

    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    TICK_INTERVAL = 0.2  # seconds between spinner redraws

    def __init__(self, stream=None):
        self._stream = stream or sys.stdout
        self._spinner_idx = 0
        self._eval_text: str = ""
        self._eval_summary: str = ""  # eval results without convergence suffix
        self._convergence_text: str = ""  # appended to eval line
        self._status_text: str = ""
        self._active_line: str = "status"  # which line gets the spinner
        self._has_lines = False
        self._file_handler: logging.FileHandler | None = None
        self._lock = threading.Lock()
        self._tick_stop = threading.Event()
        self._tick_thread: threading.Thread | None = None
        # Previous values for delta colouring
        self._prev_loss: float | None = None
        self._prev_policy_loss: float | None = None
        self._prev_value_loss: float | None = None
        self._prev_games_per_sec: float | None = None
        self._prev_steps_per_sec: float | None = None

    def _start_ticker(self):
        """Start the background spinner thread if not already running."""
        if self._tick_thread is not None and self._tick_thread.is_alive():
            return
        self._tick_stop.clear()
        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def _tick_loop(self):
        while not self._tick_stop.wait(self.TICK_INTERVAL):
            with self._lock:
                if self._has_lines:
                    self._redraw()

    def _spinner_frame(self) -> str:
        frame = self.SPINNER_FRAMES[self._spinner_idx % len(self.SPINNER_FRAMES)]
        self._spinner_idx += 1
        return frame

    def _timestamp(self) -> str:
        return time.strftime("%H:%M:%S")

    def _inject_spinner(self, text: str) -> str:
        """Replace the spinner placeholder with the current frame."""
        return text.replace(_SPINNER_PLACEHOLDER, self._spinner_frame())

    def _build_eval_line(self) -> str:
        if self._active_line == "eval":
            return self._inject_spinner(self._eval_text)
        return self._eval_text.replace(_SPINNER_PLACEHOLDER, " ")

    def _build_status_line(self) -> str:
        if self._active_line == "status":
            return self._inject_spinner(self._status_text)
        return self._status_text.replace(_SPINNER_PLACEHOLDER, " ")

    def _redraw(self):
        """Redraw both lines in place. Caller must hold _lock."""
        if self._has_lines:
            # Move to saved position and clear everything below. This
            # handles long lines that wrap across multiple terminal rows
            # (where a simple \033[2A wouldn't go back far enough).
            self._stream.write("\033[u\033[J")
        # Save cursor position before writing the two lines.
        self._stream.write("\033[s")
        self._stream.write(self._build_eval_line() + "\n" + self._build_status_line() + "\n")
        self._stream.flush()
        self._has_lines = True

    def _safe_float(self, s: str) -> float | None:
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def update_status(
        self, *, stage: str, step: int, loss: str, policy_loss: str,
        value_loss: str, games: str, games_per_sec: str, buf_size: str,
        steps_per_sec: str, lr: str,
    ):
        with self._lock:
            ts = self._timestamp()
            # Colorize values based on direction
            c_loss = _color_delta(loss, self._prev_loss, lower_is_better=True)
            c_ploss = _color_delta(policy_loss, self._prev_policy_loss, lower_is_better=True)
            c_vloss = _color_delta(value_loss, self._prev_value_loss, lower_is_better=True)
            c_gps = _color_delta(games_per_sec, self._prev_games_per_sec, lower_is_better=False)
            c_sps = _color_delta(steps_per_sec, self._prev_steps_per_sec, lower_is_better=False)

            self._status_text = (
                f"{ts} {stage}  {_SPINNER_PLACEHOLDER}  │ step {step} │ "
                f"loss {c_loss} (π {c_ploss} v {c_vloss}) │ "
                f"{games} games ({c_gps}/s) │ "
                f"buf {buf_size} │ {c_sps} step/s │ lr {lr}"
            )
            # Update previous values
            self._prev_loss = self._safe_float(loss)
            self._prev_policy_loss = self._safe_float(policy_loss)
            self._prev_value_loss = self._safe_float(value_loss)
            self._prev_games_per_sec = self._safe_float(games_per_sec)
            self._prev_steps_per_sec = self._safe_float(steps_per_sec)

            self._active_line = "status"
            self._redraw()
        self._start_ticker()

    def update_eval(self, eval_summary: str):
        with self._lock:
            ts = self._timestamp()
            self._eval_summary = eval_summary
            self._eval_text = f"{ts} eval  │ {eval_summary}{self._convergence_text}"
            self._active_line = "status"
            self._redraw()

    def update_convergence(self, mean_win_rate: float, threshold: float, window: int, full_window: bool):
        """Append convergence info to the eval line.

        Args:
            mean_win_rate: rolling average of win rate vs lagged checkpoint
            threshold: improve_threshold (below this = plateau)
            window: number of evals currently in the rolling window
            full_window: whether the window has reached its target size (i.e. convergence check is active)
        """
        with self._lock:
            if full_window:
                # Color: green if still improving, red if plateaued
                color = _GREEN if mean_win_rate >= threshold else _RED
                status = "improving" if mean_win_rate >= threshold else "plateau"
                self._convergence_text = (
                    f" │ {color}⌀{window}: {mean_win_rate:.2f} ({status}){_RESET}"
                )
            else:
                self._convergence_text = f" │ ⌀: {mean_win_rate:.2f} ({window} evals)"
            # Rebuild eval text with new convergence suffix
            if self._eval_summary:
                ts_match = self._eval_text.split(" eval")[0] if " eval" in self._eval_text else self._timestamp()
                self._eval_text = f"{ts_match} eval  │ {self._eval_summary}{self._convergence_text}"
            self._redraw()

    def update_sprt(
        self, score: float, llr: float, games: int, decision: str,
        s0: float, s1: float, upper: float,
        reject_count: int = 0, reject_patience: int = 0,
        reject_status: str = "active",
    ) -> None:
        """Update the eval line with live SPRT state.

        Args:
            score: current running score (W+0.5D)/N
            llr: log-likelihood ratio
            games: games played so far
            decision: "continue" | "accept_h1" | "reject_h1"
            s0, s1: null and alternative hypothesis scores (for context)
            upper: |LLR| threshold (symmetric Wald bounds)
        """
        with self._lock:
            ts = self._timestamp()
            if decision == "accept_h1":
                color = _GREEN
                dec_str = "ACCEPT H1 → promote"
            elif decision == "reject_h1":
                color = _RED
                dec_str = "REJECT H1 → plateau"
            else:
                color = ""
                dec_str = "continue"
            progress_pct = min(100, max(0, int(abs(llr) / upper * 100)))
            if reject_patience > 0:
                if reject_status == "active":
                    reject_color = _RED if reject_count >= reject_patience else ""
                    reject_end = _RESET if reject_color else ""
                    reject_info = (
                        f" │ {reject_color}rejects {reject_count}/{reject_patience}{reject_end}"
                    )
                else:
                    # warmup or grace — rejects don't count yet
                    reject_info = (
                        f" │ rejects {reject_count}/{reject_patience} ({reject_status})"
                    )
            else:
                # Reject-patience detector disabled (patience <= 0): the count
                # is still useful context, just no threshold to show.
                suffix = "" if reject_status == "active" else f" ({reject_status})"
                reject_info = f" │ rejects {reject_count}{suffix}"
            summary = (
                f"vs champion │ n={games} │ score={score:.3f} "
                f"(H0={s0:.2f} H1={s1:.2f}) │ LLR={llr:+.2f}/{upper:.2f} "
                f"({progress_pct}%){reject_info} │ {color}{dec_str}{_RESET}"
            )
            self._eval_summary = summary
            self._eval_text = f"{ts} sprt  │ {summary}"
            self._active_line = "status"
            self._redraw()

    def show_eval_busy(self, message: str = "evaluating..."):
        """Show spinner on the eval line while evaluation is running."""
        with self._lock:
            ts = self._timestamp()
            self._eval_text = f"{ts} eval  │ {message}  {_SPINNER_PLACEHOLDER}"
            self._active_line = "eval"
            self._redraw()

    def print_stage_header(
        self, *, stage_num: int, total_stages: int, name: str,
        win_length: int, radius: int, max_moves: int, lr: float,
    ):
        with self._lock:
            sep = "=" * 60
            header = (
                f"\n{sep}\n"
                f"CURRICULUM STAGE {stage_num}/{total_stages}: {name}\n"
                f"  win_length={win_length}, radius={radius}, max_moves={max_moves}\n"
                f"  lr={lr:.6f}\n"
                f"{sep}\n"
            )
            self._stream.write(header)
            self._stream.flush()
            self._has_lines = False
            self._eval_text = ""
            self._eval_summary = ""
            self._convergence_text = ""
            self._status_text = ""
            self._active_line = "status"
            # Reset previous values for new stage
            self._prev_loss = None
            self._prev_policy_loss = None
            self._prev_value_loss = None
            self._prev_games_per_sec = None
            self._prev_steps_per_sec = None

    def print_event(self, message: str):
        """Print a one-off event line above the display (e.g. checkpoint saved)."""
        with self._lock:
            ts = self._timestamp()
            if self._has_lines:
                self._stream.write("\033[2A\033[J")
                self._has_lines = False
            self._stream.write(f"{ts} {message}\n")
            self._stream.flush()
            self._redraw()

    def setup_file_logging(self, log_path: str):
        """Add a file handler that captures all hexo_a0 logger output."""
        self.close_file_logging()
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(path), mode="a")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logging.getLogger("hexo_a0").addHandler(handler)
        self._file_handler = handler

    def close_file_logging(self):
        """Remove file handler if present."""
        if self._file_handler is not None:
            logging.getLogger("hexo_a0").removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def close(self):
        """Stop the ticker thread and remove file handler."""
        self._tick_stop.set()
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=1)
            self._tick_thread = None
        self.close_file_logging()
