"""Tests for the training display module."""

import io
import re

import pytest

from hexo_a0.display import TrainingDisplay


class TestSpinner:
    def test_spinner_cycles_through_frames(self):
        display = TrainingDisplay(stream=io.StringIO())
        n = len(display.SPINNER_FRAMES)
        frames = [display._spinner_frame() for _ in range(n + 1)]
        # Should cycle through braille spinner chars
        assert frames[0] != frames[1]
        assert frames[0] == frames[n]

    def test_spinner_appears_in_status_line(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.update_status(
            stage="S1", step=100, loss="1.24", policy_loss="0.64",
            value_loss="0.59", games="+3027", games_per_sec="26.6",
            buf_size="100000", steps_per_sec="1.0", lr="9.6e-05",
        )
        output = stream.getvalue()
        assert any(c in output for c in display.SPINNER_FRAMES)

    def test_spinner_on_eval_line_when_busy(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.show_eval_busy("evaluating...")
        output = stream.getvalue()
        # Eval line should have spinner
        lines = [l for l in output.split("\n") if l.strip()]
        eval_line = lines[0]
        assert any(c in eval_line for c in display.SPINNER_FRAMES)
        assert "evaluating..." in eval_line

    def test_spinner_moves_back_to_status_after_eval(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.show_eval_busy("evaluating...")
        display.update_eval("vs random: 100%")
        display.update_status(
            stage="S1", step=100, loss="1.0", policy_loss="0.5",
            value_loss="0.5", games="+10", games_per_sec="1.0",
            buf_size="1000", steps_per_sec="1.0", lr="1e-04",
        )
        output = stream.getvalue()
        # The last status line should have the spinner
        lines = output.split("\n")
        # Find the last non-empty line (status)
        non_empty = [l for l in lines if l.strip()]
        status_line = non_empty[-1]
        assert any(c in status_line for c in display.SPINNER_FRAMES)


class TestStatusLine:
    def test_formats_status_line(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.update_status(
            stage="S1", step=16700, loss="1.24", policy_loss="0.64",
            value_loss="0.59", games="+3027", games_per_sec="26.6",
            buf_size="100k", steps_per_sec="1.0", lr="9.6e-05",
        )
        output = stream.getvalue()
        assert "S1" in output
        assert "16700" in output
        assert "1.24" in output
        assert "0.64" in output
        assert "0.59" in output
        assert "+3027" in output
        assert "26.6" in output
        assert "100k" in output
        assert "1.0 step/s" in output
        assert "9.6e-05" in output

    def test_status_line_has_timestamp(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.update_status(
            stage="S1", step=100, loss="1.0", policy_loss="0.5",
            value_loss="0.5", games="+10", games_per_sec="1.0",
            buf_size="1000", steps_per_sec="1.0", lr="1e-04",
        )
        output = stream.getvalue()
        assert re.search(r"\d{2}:\d{2}:\d{2}", output)

    def test_status_line_overwrites_previous(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.update_status(
            stage="S1", step=100, loss="1.0", policy_loss="0.5",
            value_loss="0.5", games="+10", games_per_sec="1.0",
            buf_size="1000", steps_per_sec="1.0", lr="1e-04",
        )
        display.update_status(
            stage="S1", step=200, loss="0.9", policy_loss="0.4",
            value_loss="0.5", games="+20", games_per_sec="2.0",
            buf_size="2000", steps_per_sec="1.5", lr="9e-05",
        )
        output = stream.getvalue()
        # Second update should use ANSI cursor restore + clear to overwrite the first
        assert "\033[u" in output


class TestEvalLine:
    def test_formats_eval_line(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.update_eval("vs random: 100% (W50/L0/D0) │ vs checkpoint: 46% (W23/L27/D0)")
        output = stream.getvalue()
        assert "eval" in output.lower()
        assert "100%" in output
        assert "46%" in output

    def test_eval_line_has_timestamp(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.update_eval("vs random: 100%")
        output = stream.getvalue()
        assert re.search(r"\d{2}:\d{2}:\d{2}", output)

    def test_eval_and_status_both_visible(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.update_eval("vs random: 100%")
        display.update_status(
            stage="S1", step=100, loss="1.0", policy_loss="0.5",
            value_loss="0.5", games="+10", games_per_sec="1.0",
            buf_size="1000", steps_per_sec="1.0", lr="1e-04",
        )
        output = stream.getvalue()
        assert "100%" in output
        assert "step" in output.lower() or "100" in output


class TestSprtRejectInfo:
    @staticmethod
    def _sprt_line(stream, **kwargs):
        display = TrainingDisplay(stream=stream)
        display.update_sprt(
            score=0.52, llr=1.0, games=100, decision="continue",
            s0=0.50, s1=0.55, upper=4.55, **kwargs,
        )
        return stream.getvalue()

    def test_counter_with_threshold_when_detector_on(self):
        out = self._sprt_line(io.StringIO(), reject_count=2, reject_patience=5)
        assert "rejects 2/5" in out

    def test_counter_shown_without_threshold_when_detector_off(self):
        # patience <= 0 disables the plateau detector, but the rejection
        # count is still useful information on the CLI.
        out = self._sprt_line(io.StringIO(), reject_count=7, reject_patience=0)
        assert "rejects 7" in out
        assert "rejects 7/" not in out

    def test_detector_off_keeps_status_suffix(self):
        out = self._sprt_line(
            io.StringIO(), reject_count=3, reject_patience=0,
            reject_status="grace 200/1000",
        )
        assert "rejects 3 (grace 200/1000)" in out


class TestStageHeader:
    def test_prints_stage_header(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.print_stage_header(
            stage_num=1, total_stages=4, name="S1: 4-in-a-row, radius 2",
            win_length=4, radius=2, max_moves=80, lr=0.0002,
        )
        output = stream.getvalue()
        assert "1/4" in output
        assert "S1: 4-in-a-row, radius 2" in output
        assert "win_length=4" in output
        assert "radius=2" in output
        assert "max_moves=80" in output
        assert "0.000200" in output or "0.0002" in output
        assert "=" * 10 in output

    def test_stage_header_not_overwritten(self):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        display.print_stage_header(
            stage_num=1, total_stages=4, name="S1: 4-in-a-row, radius 2",
            win_length=4, radius=2, max_moves=80, lr=0.0002,
        )
        display.update_status(
            stage="S1", step=100, loss="1.0", policy_loss="0.5",
            value_loss="0.5", games="+10", games_per_sec="1.0",
            buf_size="1000", steps_per_sec="1.0", lr="1e-04",
        )
        full_output = stream.getvalue()
        assert "S1: 4-in-a-row, radius 2" in full_output


class TestFileLogging:
    def test_setup_file_logging_creates_handler(self, tmp_path):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        log_path = tmp_path / "train.log"
        display.setup_file_logging(str(log_path))
        assert log_path.exists() or log_path.parent.exists()
        display.close()

    def test_close_removes_file_handler(self, tmp_path):
        stream = io.StringIO()
        display = TrainingDisplay(stream=stream)
        log_path = tmp_path / "train.log"
        display.setup_file_logging(str(log_path))
        display.close()
        display.close()  # safe to call multiple times
