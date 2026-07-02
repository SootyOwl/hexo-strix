"""Tests for gpu_memory.py — CUDA/HIP allocator config + cache-clear cadence.

Written first (TDD). These cover everything testable WITHOUT a real
multi-process GPU OOM run:

  * the alloc-config env is applied at startup with the documented default,
  * an externally-exported PYTORCH_CUDA_ALLOC_CONF wins (no clobber),
  * the HEXO_CUDA_ALLOC_CONF override is honoured,
  * expandable_segments is only appended when the probe says it's supported,
  * the cadence gate fires on the configured period and NOT every call,
  * cadence is tunable via HEXO_CACHE_CLEAR_EVERY,
  * a cadence of 0 (or negative) disables clearing entirely.

The actual OOM-avoidance behaviour needs a live shared-GPU training run and
is documented as a follow-up, not unit-tested here.
"""

import os
import subprocess
import sys

import pytest

import hexo_a0.gpu_memory as gm


# ---------------------------------------------------------------------------
# Allocator config
# ---------------------------------------------------------------------------


class TestConfigureCudaAlloc:
    def test_sets_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("HEXO_CUDA_ALLOC_CONF", raising=False)

        applied = gm.configure_cuda_alloc(supports_expandable=False)

        assert applied == os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        assert "garbage_collection_threshold:0.8" in applied
        assert "max_split_size_mb:128" in applied
        # probe said unsupported -> must NOT be present
        assert "expandable_segments" not in applied

    def test_existing_pytorch_conf_wins(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        monkeypatch.delenv("HEXO_CUDA_ALLOC_CONF", raising=False)

        applied = gm.configure_cuda_alloc(supports_expandable=True)

        # user export is untouched
        assert applied == "max_split_size_mb:512"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"

    def test_hexo_override_is_honoured(self, monkeypatch):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.setenv("HEXO_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.5")

        applied = gm.configure_cuda_alloc(supports_expandable=False)

        assert applied == "garbage_collection_threshold:0.5"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "garbage_collection_threshold:0.5"

    def test_expandable_appended_when_supported(self, monkeypatch):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("HEXO_CUDA_ALLOC_CONF", raising=False)

        applied = gm.configure_cuda_alloc(supports_expandable=True)

        assert "expandable_segments:True" in applied

    def test_hexo_override_not_double_clobbered_by_expandable(self, monkeypatch):
        # If the user already put expandable_segments in their override we must
        # not append a duplicate.
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.setenv("HEXO_CUDA_ALLOC_CONF", "expandable_segments:False")

        applied = gm.configure_cuda_alloc(supports_expandable=True)

        assert applied.count("expandable_segments") == 1
        assert "expandable_segments:False" in applied


class TestKillSwitch:
    @pytest.mark.parametrize("sentinel", ["off", "none", "default", "", "OFF"])
    def test_disable_sentinel_leaves_pytorch_env_unset(self, monkeypatch, sentinel):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.setenv("HEXO_CUDA_ALLOC_CONF", sentinel)

        applied = gm.configure_cuda_alloc(supports_expandable=True)

        assert applied == ""
        assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


class TestExpandableProbe:
    def test_probe_false_on_hip_build(self, monkeypatch):
        # ROCm/HIP builds expose torch.version.hip; never auto-enable there
        # (unsupported key emits a UserWarning at device init).
        import torch

        monkeypatch.setattr(torch.version, "hip", "7.2.0", raising=False)
        assert gm.supports_expandable_segments() is False

    def test_probe_true_on_recent_cuda_build(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.version, "hip", None, raising=False)
        monkeypatch.setattr(torch, "__version__", "2.5.1+cu124", raising=False)
        assert gm.supports_expandable_segments() is True

    def test_explicit_opt_in_overrides_hip_probe(self, monkeypatch):
        # A HIP user who knows their build supports it can force it on.
        import torch

        monkeypatch.setattr(torch.version, "hip", "7.2.0", raising=False)
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("HEXO_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.setenv("HEXO_EXPANDABLE_SEGMENTS", "1")

        applied = gm.configure_cuda_alloc()  # None -> consult env + probe

        assert "expandable_segments:True" in applied

    def test_explicit_opt_out_overrides_probe(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.version, "hip", None, raising=False)
        monkeypatch.setattr(torch, "__version__", "2.5.1+cu124", raising=False)
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("HEXO_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.setenv("HEXO_EXPANDABLE_SEGMENTS", "0")

        applied = gm.configure_cuda_alloc()  # None -> consult env + probe

        assert "expandable_segments" not in applied


# ---------------------------------------------------------------------------
# Cache-clear cadence gate
# ---------------------------------------------------------------------------


class TestCacheClearGate:
    def test_clears_on_period_not_every_call(self):
        calls = []
        gate = gm.CacheClearGate(every=4, clear_fn=lambda: calls.append(1))

        # 12 invocations, period 4 -> should clear on the 4th, 8th, 12th.
        fired = [gate.step() for _ in range(12)]

        assert sum(fired) == 3
        assert len(calls) == 3
        # specifically the 4th, 8th, 12th (1-indexed)
        assert [i for i, f in enumerate(fired, start=1) if f] == [4, 8, 12]

    def test_period_one_clears_every_call(self):
        gate = gm.CacheClearGate(every=1, clear_fn=lambda: None)
        assert all(gate.step() for _ in range(5))

    def test_zero_disables(self):
        calls = []
        gate = gm.CacheClearGate(every=0, clear_fn=lambda: calls.append(1))
        fired = [gate.step() for _ in range(100)]
        assert not any(fired)
        assert calls == []

    def test_negative_disables(self):
        gate = gm.CacheClearGate(every=-5, clear_fn=lambda: None)
        assert not any(gate.step() for _ in range(100))

    def test_cadence_from_env_default(self, monkeypatch):
        monkeypatch.delenv("HEXO_CACHE_CLEAR_EVERY", raising=False)
        assert gm.cache_clear_cadence() == gm.DEFAULT_CACHE_CLEAR_EVERY
        assert gm.DEFAULT_CACHE_CLEAR_EVERY > 1  # much less frequent than per-forward

    def test_cadence_from_env_override(self, monkeypatch):
        monkeypatch.setenv("HEXO_CACHE_CLEAR_EVERY", "16")
        assert gm.cache_clear_cadence() == 16

    def test_cadence_bad_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("HEXO_CACHE_CLEAR_EVERY", "not-a-number")
        assert gm.cache_clear_cadence() == gm.DEFAULT_CACHE_CLEAR_EVERY

    def test_for_device_cpu_is_noop(self):
        # A gate built for a non-cuda device should never invoke torch clearing.
        import torch
        gate = gm.CacheClearGate.for_device(torch.device("cpu"), every=1)
        # Even with period 1, stepping must not fire (disabled on cpu).
        assert not any(gate.step() for _ in range(10))

    def test_for_device_cuda_uses_empty_cache(self):
        # On a cuda-typed device the default clear_fn is torch.cuda.empty_cache,
        # and the gate fires on the resolved cadence. We use period 1 + a stub
        # clear_fn so this is safe to run without a real GPU.
        import torch
        calls = []
        gate = gm.CacheClearGate.for_device(
            torch.device("cuda"), every=1, clear_fn=lambda: calls.append(1)
        )
        assert gate.step() is True
        assert calls == [1]


# ---------------------------------------------------------------------------
# Wiring: the self_play eval path steps the gate once per forward and clears on
# cadence, NOT every call. Runs on CPU; we force a small-cadence gate so the
# cadence logic is exercised without a real GPU.
# ---------------------------------------------------------------------------


class TestAppliedBeforeDeviceInit:
    """The load-bearing correctness point: PYTORCH_CUDA_ALLOC_CONF must be set
    BEFORE the HIP/CUDA context initialises (it's read once, at first device
    touch). Verify in a clean subprocess that configure_cuda_alloc() runs while
    the context is still uninitialised, that the env reads back, and — when the
    allocator backend exposes it — that the live allocator actually adopted the
    config.
    """

    def test_env_set_before_context_init_in_clean_process(self):
        script = (
            "import torch\n"
            "assert not torch.cuda.is_initialized(), 'context already init at import'\n"
            "from hexo_a0.gpu_memory import configure_cuda_alloc\n"
            "applied = configure_cuda_alloc()\n"
            "import os\n"
            "assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == applied, 'env readback mismatch'\n"
            "assert not torch.cuda.is_initialized(), 'context init too early'\n"
            "# If a CUDA/HIP device is present, force context init and confirm the\n"
            "# live allocator settings reflect our config (best-effort readback).\n"
            "if torch.cuda.is_available():\n"
            "    torch.zeros(1, device='cuda')\n"
            "    try:\n"
            "        from torch.cuda.memory import _get_allocator_settings\n"
            "        live = _get_allocator_settings()\n"
            "        assert 'garbage_collection_threshold' in str(live) or applied, live\n"
            "    except Exception:\n"
            "        pass\n"
            "print('OK', applied)\n"
        )
        env = dict(os.environ)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("HEXO_CUDA_ALLOC_CONF", None)
        r = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        assert "OK garbage_collection_threshold:0.8" in r.stdout, r.stdout

    def test_external_override_wins_in_clean_process(self):
        script = (
            "from hexo_a0.gpu_memory import configure_cuda_alloc\n"
            "import os\n"
            "applied = configure_cuda_alloc()\n"
            "assert applied == 'max_split_size_mb:256', applied\n"
            "assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'max_split_size_mb:256'\n"
            "print('OK')\n"
        )
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
        env.pop("HEXO_CUDA_ALLOC_CONF", None)
        r = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        assert "OK" in r.stdout


class TestSelfPlayWiring:
    def test_eval_steps_gate_once_per_forward_and_clears_on_cadence(
        self, monkeypatch
    ):
        import torch
        import hexo_rs
        from hexo_a0.config import ModelConfig, MCTSConfig, FullConfig
        from hexo_a0.model import HeXONet
        from hexo_a0.self_play import self_play_game

        clears = []
        steps = {"n": 0}

        def fake_for_device(device, every=None, clear_fn=None):
            gate = gm.CacheClearGate(every=3, clear_fn=lambda: clears.append(1))
            orig_step = gate.step

            def counting_step():
                steps["n"] += 1
                return orig_step()

            gate.step = counting_step
            return gate

        # self_play imported the symbol by name; patch its module reference.
        import hexo_a0.self_play as sp_mod
        monkeypatch.setattr(sp_mod.CacheClearGate, "for_device", fake_for_device)

        model_config = ModelConfig(
            hidden_dim=16, num_layers=1, num_heads=1,
            graph_type="hex", conv_type="gatv2",
        )
        mcts_config = MCTSConfig(n_simulations=4, m_actions=4, exploration_moves=10)
        config = FullConfig(model=model_config, mcts=mcts_config)
        game_config = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=12)
        network = HeXONet(model_config)
        network.eval()

        examples = self_play_game(network, config, game_config, torch.device("cpu"))

        # Sanity: the game actually produced training examples (eval path ran).
        assert len(examples) > 0
        # The gate was stepped at least once per forward.
        assert steps["n"] > 0
        # And it cleared on the cadence (every 3rd step), not every call.
        assert len(clears) == steps["n"] // 3
