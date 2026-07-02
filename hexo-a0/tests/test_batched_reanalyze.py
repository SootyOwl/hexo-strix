"""Batched-reanalyze wrapper: lockstep MCTS over many positions."""
import pytest
import torch

import hexo_rs
from hexo_a0.config import MCTSConfig, ModelConfig
from hexo_a0.model import HeXONet


def _tiny_model():
    m = HeXONet(ModelConfig(
        hidden_dim=16, num_layers=1, num_heads=1,
        policy_hidden=8, value_hidden=8, graph_type="axis",
    ))
    m.eval()
    return m


def _games(n=3):
    """n DIFFERENT single-move positions: origin (0,0) is pre-seeded in all,
    then each game gets one move at a distinct coordinate (i+1, 0)."""
    gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    out = []
    for i in range(n):
        g = hexo_rs.GameState(gc)
        g.apply_move(i + 1, 0)
        out.append(g)
    return out


class TestRustBatchedGumbelMcts:
    def test_one_policy_per_state_matching_legal_count(self):
        from hexo_a0.self_play import _rust_batched_gumbel_mcts
        from hexo_a0.graph import game_to_axis_graph, game_to_axis_graph_batch
        mcts = MCTSConfig(n_simulations=4, m_actions=4)
        games = _games(3)
        policies = _rust_batched_gumbel_mcts(
            games, _tiny_model(), mcts, torch.device("cpu"),
            graph_fn=game_to_axis_graph, graph_batch_fn=game_to_axis_graph_batch,
            seed=3,
        )
        assert len(policies) == 3
        for pol, g in zip(policies, games):
            assert pol.shape[0] == len(g.legal_moves())
            assert torch.isfinite(pol).all()
            assert pol.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_sims_override_reaches_rust(self, monkeypatch):
        # hexo_rs.MCTSConfig is a PyO3 class without attribute getters
        # (probed: cfg.n_simulations raises AttributeError), so capture the
        # constructor kwargs via a fake class instead.
        from hexo_a0 import self_play as sp_mod
        captured = {}

        class FakeMCTSConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        def fake_batched(games, eval_fn, cfg, seed=None):
            return [
                ((0, 0), [1.0 / len(g.legal_moves())] * len(g.legal_moves()))
                for g in games
            ]

        monkeypatch.setattr(hexo_rs, "MCTSConfig", FakeMCTSConfig)
        monkeypatch.setattr(hexo_rs, "batched_gumbel_mcts", fake_batched)
        mcts = MCTSConfig(n_simulations=96, m_actions=16)
        sp_mod._rust_batched_gumbel_mcts(
            _games(1), _tiny_model(), mcts, torch.device("cpu"), n_simulations=8,
        )
        assert captured["n_simulations"] == 8

    def test_parity_with_serial_path(self):
        """Same position, same seed, same deterministic model: serial vs batched.

        Probed 2026-06-10 (seeds 3/7/42, n=8, m=4, CPU): the single-state
        batched path is EXACTLY equal to the serial path — same RNG stream.
        Lock that in; if a future Rust change diverges the streams, relax this
        to length + sum-to-1 (the structural invariants asserted first).
        """
        from hexo_a0.self_play import _rust_batched_gumbel_mcts, _rust_gumbel_mcts
        from hexo_a0.graph import game_to_axis_graph, game_to_axis_graph_batch

        torch.manual_seed(0)
        model = _tiny_model()
        mcts = MCTSConfig(n_simulations=8, m_actions=4)

        g_serial = _games(1)[0]
        g_batched = _games(1)[0]

        _action, serial_pol = _rust_gumbel_mcts(
            g_serial, model, mcts, torch.device("cpu"),
            graph_fn=game_to_axis_graph, graph_batch_fn=game_to_axis_graph_batch,
            seed=7,
        )
        (batched_pol,) = _rust_batched_gumbel_mcts(
            [g_batched], model, mcts, torch.device("cpu"),
            graph_fn=game_to_axis_graph, graph_batch_fn=game_to_axis_graph_batch,
            seed=7,
        )

        assert batched_pol.shape == serial_pol.shape
        assert batched_pol.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert serial_pol.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert torch.allclose(batched_pol, serial_pol, atol=1e-6)


# ---------------------------------------------------------------------------
# Trainer-level chunked reanalyze
# ---------------------------------------------------------------------------


def _dummy_data(num_legal: int = 2):
    from torch_geometric.data import Data

    n_nodes = num_legal + 1
    legal_mask = torch.tensor([False] + [True] * num_legal)
    return Data(
        x=torch.zeros((n_nodes, 7)),
        edge_index=torch.zeros((2, 0), dtype=torch.int64),
        legal_mask=legal_mask,
        coords=torch.zeros((n_nodes, 2), dtype=torch.int32),
    )


class TestChunkedReanalyze:
    """Trainer._reanalyze_batch: chunked lockstep dispatch, cheap-search
    overrides, sample_weight preservation, and per-chunk failure isolation.
    The batched MCTS callout is stubbed (no real forward pass)."""

    # Distance-1 neighbours of the pre-seeded origin stone — all legal,
    # all non-terminal after a single placement.
    _COORDS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    def _make_trainer(
        self,
        tmp_path,
        monkeypatch,
        reanalyze_fraction: float,
        reanalyze_chunk: int = 32,
        reanalyze_sims: int = 0,
        reanalyze_m_actions: int = 0,
    ):
        """Construct a minimal Trainer with reanalyze enabled."""
        from hexo_a0.config import (
            EvalConfig,
            FullConfig,
            MCTSConfig,
            ModelConfig,
            PythonSelfPlayConfig,
            RunConfig,
            SelfPlayConfig,
            TrainingConfig,
        )
        from hexo_a0.trainer import Trainer

        model_cfg = ModelConfig(
            hidden_dim=16, num_layers=1, num_heads=1,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
            prune_empty_edges=False, dropout=0.0, pre_norm=True,
        )
        training_cfg = TrainingConfig(
            batch_size=4, buffer_capacity=100,
            edge_budget=0, grad_accumulation=False,
            lr=2e-4, weight_decay=1e-4, max_grad_norm=1.0,
            lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0,
            step_delay=0.0, target_reuse=0.0, draw_value=0.0,
            augment_symmetries=False, reanalyze_fraction=reanalyze_fraction,
            reanalyze_chunk=reanalyze_chunk,
            reanalyze_sims=reanalyze_sims,
            reanalyze_m_actions=reanalyze_m_actions,
            pc_loss_weight=0.0,
        )
        full_cfg = FullConfig(
            model=model_cfg,
            training=training_cfg,
            mcts=MCTSConfig(
                n_simulations=2, m_actions=2,
                exploration_moves=0, exploration_fraction=0.0, exploration_max=0.0,
            ),
            self_play=SelfPlayConfig(
                device="cpu", workers=1, precision="fp32",
                python=PythonSelfPlayConfig(games_per_write=1),
            ),
            eval=EvalConfig(interval=0, checkpoint_interval=0, games=0, sims=0),
            run=RunConfig(device="cpu", fp16=False, compile=False),
        )
        net = HeXONet(model_cfg)
        net.eval()
        trainer = Trainer(
            model=net,
            config=full_cfg,
            game_config=hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=20),
            device=torch.device("cpu"),
            log_dir=None,
            ckpt_dir=None,
        )
        return trainer

    def _example(self, i: int, sample_weight: float = 1.0):
        """A TrainingExample with a real non-terminal game_state and a
        2-legal dummy graph (fake policies of length 2 pass the trainer's
        policy-length guard)."""
        from hexo_a0.self_play import TrainingExample

        game = hexo_rs.GameState(
            hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=20)
        )
        q, r = self._COORDS[i % len(self._COORDS)]
        game.apply_move(q, r)  # origin is pre-seeded; one stone, non-terminal
        return TrainingExample(
            data=_dummy_data(num_legal=2),
            policy_target=torch.tensor([1.0, 0.0], dtype=torch.float32),
            value_target=0.5,
            game_state=game,
            trajectory_id=i,
            sample_weight=sample_weight,
        )

    def _n_legal(self, i: int) -> int:
        """Legal-move count for the game_state built by ``_example(i)``.

        The reanalyze policy-length guard now derives n_legal from the
        reconstructed game_state (graphs are no longer carried on examples),
        so mock policies must match this length to pass the guard.
        """
        game = hexo_rs.GameState(
            hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=20)
        )
        q, r = self._COORDS[i % len(self._COORDS)]
        game.apply_move(q, r)
        return len(game.legal_moves())

    @staticmethod
    def _patch_deterministic_sample(monkeypatch):
        """Make random.sample order-preserving so chunk membership is fixed.

        population is always range(n) at this call site, so [:k] is
        order-preserving."""
        monkeypatch.setattr(
            "random.sample", lambda population, k: list(population)[:k]
        )

    def test_chunking_splits_calls(self, tmp_path, monkeypatch):
        trainer = self._make_trainer(
            tmp_path, monkeypatch, reanalyze_fraction=1.0, reanalyze_chunk=2
        )
        calls = []

        def fake_batched(states, network, mcts, device, **kwargs):
            calls.append(len(states))
            return [torch.tensor([0.5, 0.5]) for _ in states]

        from hexo_a0 import self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", fake_batched)

        examples = [self._example(i) for i in range(5)]
        trainer._reanalyze_batch(examples)
        assert calls == [2, 2, 1]

    def test_sims_override_forwarded(self, tmp_path, monkeypatch):
        trainer = self._make_trainer(
            tmp_path, monkeypatch, reanalyze_fraction=1.0,
            reanalyze_sims=8, reanalyze_m_actions=4,
        )
        captured = {}

        def fake_batched(states, network, mcts, device, **kwargs):
            captured["n_simulations"] = kwargs.get("n_simulations")
            captured["m_actions"] = kwargs.get("m_actions")
            return [torch.tensor([0.5, 0.5]) for _ in states]

        from hexo_a0 import self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", fake_batched)

        trainer._reanalyze_batch([self._example(0)])
        assert (captured["n_simulations"], captured["m_actions"]) == (8, 4)

    def test_zero_sims_means_none(self, tmp_path, monkeypatch):
        # Default reanalyze_sims=0 / reanalyze_m_actions=0 → the trainer
        # converts 0 to None (= use mcts.n_simulations / mcts.m_actions).
        trainer = self._make_trainer(tmp_path, monkeypatch, reanalyze_fraction=1.0)
        captured = {}

        def fake_batched(states, network, mcts, device, **kwargs):
            captured["n_simulations"] = kwargs.get("n_simulations", "MISSING")
            captured["m_actions"] = kwargs.get("m_actions", "MISSING")
            return [torch.tensor([0.5, 0.5]) for _ in states]

        from hexo_a0 import self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", fake_batched)

        trainer._reanalyze_batch([self._example(0)])
        assert captured["n_simulations"] is None
        assert captured["m_actions"] is None

    def test_sample_weight_preserved(self, tmp_path, monkeypatch):
        trainer = self._make_trainer(tmp_path, monkeypatch, reanalyze_fraction=1.0)
        sentinel_policy = torch.zeros(self._n_legal(0), dtype=torch.float32)
        sentinel_policy[0] = 0.123
        sentinel_policy[1] = 0.877

        def fake_batched(states, network, mcts, device, **kwargs):
            return [sentinel_policy.clone() for _ in states]

        from hexo_a0 import self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", fake_batched)

        examples = [self._example(0, sample_weight=0.25)]
        out = trainer._reanalyze_batch(examples)
        # The rewrite must have happened AND kept the sub-cap loss weight.
        assert torch.allclose(out[0].policy_target, sentinel_policy)
        assert out[0].sample_weight == 0.25

    def test_chunk_failure_is_counted_not_fatal(self, tmp_path, monkeypatch):
        trainer = self._make_trainer(
            tmp_path, monkeypatch, reanalyze_fraction=1.0, reanalyze_chunk=2
        )
        self._patch_deterministic_sample(monkeypatch)
        call_count = {"n": 0}

        def _sentinel_for(state):
            p = torch.zeros(len(state.legal_moves()), dtype=torch.float32)
            p[0] = 0.123
            p[1] = 0.877
            return p

        def fake_batched(states, network, mcts, device, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated MCTS failure")
            return [_sentinel_for(s) for s in states]

        from hexo_a0 import self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", fake_batched)

        original_policy = torch.tensor([1.0, 0.0], dtype=torch.float32)
        examples = [self._example(i) for i in range(4)]
        out = trainer._reanalyze_batch(examples)

        assert len(out) == 4
        assert call_count["n"] == 2
        # First chunk (examples 0, 1) failed → policy targets untouched.
        assert torch.allclose(out[0].policy_target, original_policy)
        assert torch.allclose(out[1].policy_target, original_policy)
        # Second chunk (examples 2, 3) succeeded → rewritten.
        assert torch.allclose(out[2].policy_target, _sentinel_for(out[2].game_state))
        assert torch.allclose(out[3].policy_target, _sentinel_for(out[3].game_state))

    def test_policy_length_mismatch_skipped(self, tmp_path, monkeypatch):
        trainer = self._make_trainer(tmp_path, monkeypatch, reanalyze_fraction=1.0)
        self._patch_deterministic_sample(monkeypatch)

        def _sentinel_for(state):
            p = torch.zeros(len(state.legal_moves()), dtype=torch.float32)
            p[0] = 0.123
            p[1] = 0.877
            return p

        def fake_batched(states, network, mcts, device, **kwargs):
            # Valid length (matches game_state.legal_moves()) for all states
            # except the first, which gets a deliberately wrong length so the
            # trainer's policy-length guard skips it.
            policies = [_sentinel_for(s) for s in states]
            policies[0] = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
            return policies

        from hexo_a0 import self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", fake_batched)

        original_policy = torch.tensor([1.0, 0.0], dtype=torch.float32)
        examples = [self._example(i) for i in range(3)]
        out = trainer._reanalyze_batch(examples)

        # Mismatched example untouched; the others rewritten.
        assert torch.allclose(out[0].policy_target, original_policy)
        assert torch.allclose(out[1].policy_target, _sentinel_for(out[1].game_state))
        assert torch.allclose(out[2].policy_target, _sentinel_for(out[2].game_state))
