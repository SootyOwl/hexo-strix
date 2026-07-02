"""Tests for AlphaZero-style reanalyze enablement.

Verifies that:
  1. `hexo_rs.GameState` supports pickle round-trip (via __reduce__ over from_state).
  2. `TrainingExample` round-trips its `game_state` through pickle (no __getstate__ strip).
  3. `ReplayBuffer.add() + sample()` preserves `game_state` (pickle is the storage format).
  4. `Trainer._reanalyze_batch` actually rewrites `policy_target` on examples with
     non-None `game_state`.

Written before the Step-1/2/3 fixes — these tests are RED until the fix is applied.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from torch_geometric.data import Data

import hexo_rs
from hexo_a0.replay_buffer import ReplayBuffer
from hexo_a0.self_play import TrainingExample


# ---------------------------------------------------------------------------
# 1. PyGameState pickle round-trip
# ---------------------------------------------------------------------------


def _state_signature(state):
    """Stable summary used to compare two reconstructed PyGameStates."""
    stones = sorted(state.placed_stones())
    return (
        stones,
        state.current_player(),
        state.moves_remaining_this_turn(),
        state.legal_move_count(),
        state.is_terminal(),
    )


class TestPyGameStatePickle:
    def test_default_config_initial_state_round_trips(self):
        s = hexo_rs.GameState()
        s2 = pickle.loads(pickle.dumps(s))
        assert _state_signature(s) == _state_signature(s2)

    def test_after_moves_round_trips(self):
        """A mid-game state with several stones round-trips losslessly."""
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        s = hexo_rs.GameState(cfg)
        s.apply_move(1, 0)   # P2 #1
        s.apply_move(-1, 0)  # P2 #2
        s.apply_move(0, 1)   # P1 #1
        s.apply_move(0, -1)  # P1 #2
        # Now P2 to move with 2 remaining.
        s2 = pickle.loads(pickle.dumps(s))
        assert _state_signature(s) == _state_signature(s2)

    def test_mid_turn_state_preserves_moves_remaining(self):
        """A state in the middle of a turn (moves_remaining=1) round-trips."""
        s = hexo_rs.GameState()
        s.apply_move(1, 0)  # only one of P2's two placements
        assert s.moves_remaining_this_turn() == 1
        s2 = pickle.loads(pickle.dumps(s))
        assert _state_signature(s) == _state_signature(s2)
        assert s2.moves_remaining_this_turn() == 1

    def test_custom_config_round_trips(self):
        cfg = hexo_rs.GameConfig(win_length=5, placement_radius=3, max_moves=15)
        s = hexo_rs.GameState(cfg)
        s.apply_move(1, 0)
        s.apply_move(-1, 0)
        s2 = pickle.loads(pickle.dumps(s))
        assert _state_signature(s) == _state_signature(s2)
        cfg2 = s2.config()
        assert (cfg2.win_length, cfg2.placement_radius, cfg2.max_moves) == (5, 3, 15)


# ---------------------------------------------------------------------------
# 2. TrainingExample round-trip preserves game_state
# ---------------------------------------------------------------------------


def _dummy_data(num_legal: int = 2) -> Data:
    n_nodes = num_legal + 1
    legal_mask = torch.tensor([False] + [True] * num_legal)
    return Data(
        x=torch.zeros((n_nodes, 7)),
        edge_index=torch.zeros((2, 0), dtype=torch.int64),
        legal_mask=legal_mask,
        coords=torch.zeros((n_nodes, 2), dtype=torch.int32),
    )


class TestTrainingExamplePickle:
    def test_game_state_survives_pickle(self):
        game = hexo_rs.GameState()
        ex = TrainingExample(
            data=_dummy_data(num_legal=2),
            policy_target=torch.tensor([0.5, 0.5]),
            value_target=0.0,
            game_state=game,
            trajectory_id=42,
        )
        ex2 = pickle.loads(pickle.dumps(ex))
        assert ex2.game_state is not None, "game_state was stripped on pickle"
        assert _state_signature(ex.game_state) == _state_signature(ex2.game_state)
        assert ex2.trajectory_id == 42
        assert ex2.value_target == 0.0

    def test_none_game_state_still_works(self):
        """Examples without game_state must still round-trip cleanly."""
        ex = TrainingExample(
            data=_dummy_data(num_legal=2),
            policy_target=torch.tensor([0.5, 0.5]),
            value_target=1.0,
            game_state=None,
        )
        ex2 = pickle.loads(pickle.dumps(ex))
        assert ex2.game_state is None
        assert ex2.value_target == 1.0


# ---------------------------------------------------------------------------
# 3. ReplayBuffer round-trip preserves game_state
# ---------------------------------------------------------------------------


class TestReplayBufferReanalyze:
    def test_sample_preserves_game_state(self):
        buf = ReplayBuffer(capacity=10)
        game = hexo_rs.GameState()
        game.apply_move(1, 0)
        ex = TrainingExample(
            data=_dummy_data(num_legal=2),
            policy_target=torch.tensor([0.5, 0.5]),
            value_target=0.0,
            game_state=game,
            trajectory_id=7,
        )
        buf.add(ex)
        sampled = buf.sample(1)
        assert len(sampled) == 1
        out = sampled[0]
        assert out.game_state is not None, (
            "ReplayBuffer.sample() stripped game_state — reanalyze is non-functional"
        )
        assert _state_signature(out.game_state) == _state_signature(game)


# ---------------------------------------------------------------------------
# 4. _reanalyze_batch actually mutates policy_target
# ---------------------------------------------------------------------------


class TestReanalyzeBatch:
    """Verifies _reanalyze_batch rewrites the policy_target for examples
    that carry a `game_state`. Stubs the MCTS callout so the test is fast
    and deterministic (no GPU / no real network forward)."""

    def _make_trainer(self, tmp_path, monkeypatch, reanalyze_fraction: float):
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
        from hexo_a0.model import HeXONet
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

    def test_reanalyze_rewrites_policy_for_examples_with_game_state(
        self, tmp_path, monkeypatch
    ):
        trainer = self._make_trainer(tmp_path, monkeypatch, reanalyze_fraction=1.0)

        # Stub the batched Rust MCTS so we don't pay for a real forward pass.
        # Returns a sentinel policy tensor so we can detect the rewrite.
        # The trainer's policy-length guard now derives n_legal from the
        # reconstructed game_state (data is no longer carried), so the mock
        # must return a policy whose length matches game_state.legal_moves().
        call_count = {"n": 0}

        game = hexo_rs.GameState()
        game.apply_move(1, 0)  # P2's first placement, so state is non-terminal
        n_legal = len(game.legal_moves())
        sentinel_policy = torch.zeros(n_legal, dtype=torch.float32)
        sentinel_policy[0] = 0.123
        sentinel_policy[1] = 0.877

        def fake_batched(states, network, mcts, device, **kwargs):
            call_count["n"] += 1
            return [sentinel_policy.clone() for _ in states]

        # _reanalyze_batch does `from hexo_a0.self_play import
        # _rust_batched_gumbel_mcts` at call time, so we patch the symbol on
        # the source module.
        import hexo_a0.self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", fake_batched)

        original_policy = torch.tensor([1.0, 0.0], dtype=torch.float32)
        ex_with_state = TrainingExample(
            data=_dummy_data(num_legal=2),
            policy_target=original_policy.clone(),
            value_target=0.5,
            game_state=game,
            trajectory_id=1,
        )
        ex_without_state = TrainingExample(
            data=_dummy_data(num_legal=2),
            policy_target=original_policy.clone(),
            value_target=-0.5,
            game_state=None,
            trajectory_id=2,
        )
        examples = [ex_with_state, ex_without_state]

        out = trainer._reanalyze_batch(examples)

        # The example with game_state should have been rewritten to sentinel.
        # value_target must be preserved (only policy changes).
        assert call_count["n"] >= 1, "Reanalyze never called _rust_batched_gumbel_mcts"
        assert torch.allclose(out[0].policy_target, sentinel_policy), (
            f"policy_target was not rewritten by reanalyze: {out[0].policy_target}"
        )
        assert out[0].value_target == 0.5
        # The example without game_state should be untouched.
        assert torch.allclose(out[1].policy_target, original_policy)
        assert out[1].game_state is None

    def test_reanalyze_disabled_is_noop(self, tmp_path, monkeypatch):
        trainer = self._make_trainer(tmp_path, monkeypatch, reanalyze_fraction=0.0)

        called = {"n": 0}

        def fake_batched(states, network, mcts, device, **kwargs):
            called["n"] += 1
            return [torch.tensor([0.5, 0.5]) for _ in states]

        import hexo_a0.self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", fake_batched)

        original_policy = torch.tensor([1.0, 0.0])
        ex = TrainingExample(
            data=_dummy_data(num_legal=2),
            policy_target=original_policy.clone(),
            value_target=0.0,
            game_state=hexo_rs.GameState(),
        )
        out = trainer._reanalyze_batch([ex])
        assert called["n"] == 0
        assert torch.allclose(out[0].policy_target, original_policy)

    def test_reanalyze_passes_mcts_config_not_training_config(
        self, tmp_path, monkeypatch
    ):
        """Regression: ``_reanalyze_batch`` must pass the *MCTS* config (which
        carries ``n_simulations`` / ``m_actions``), not the ``TrainingConfig``.

        The real ``_rust_batched_gumbel_mcts`` reads ``mcts.n_simulations`` /
        ``mcts.m_actions`` to build ``hexo_rs.MCTSConfig``, so passing
        ``TrainingConfig`` raises ``AttributeError`` that the bare ``except``
        in ``_reanalyze_batch`` swallows — silently turning reanalyze into a
        no-op. This test's fake mirrors that contract (it accesses the same
        attributes), unlike the other tests' blind fakes."""
        trainer = self._make_trainer(tmp_path, monkeypatch, reanalyze_fraction=1.0)

        game = hexo_rs.GameState()
        game.apply_move(1, 0)  # non-terminal
        # Policy length must match game_state.legal_moves() (the new guard
        # derives n_legal from game_state, not the dropped data graph).
        n_legal = len(game.legal_moves())
        sentinel_policy = torch.zeros(n_legal, dtype=torch.float32)
        sentinel_policy[0] = 0.2
        sentinel_policy[1] = 0.8
        received = {}

        def faithful_fake(states, network, mcts, device, **kwargs):
            _ = mcts.n_simulations   # real wrapper reads these to build hexo_rs.MCTSConfig;
            _ = mcts.m_actions       # TrainingConfig lacks them -> AttributeError, the bug class under test
            received["mcts"] = mcts
            return [sentinel_policy.clone() for _ in states]

        import hexo_a0.self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", faithful_fake)
        ex = TrainingExample(
            data=_dummy_data(num_legal=2),
            policy_target=torch.tensor([1.0, 0.0], dtype=torch.float32),
            value_target=0.5,
            game_state=game,
            trajectory_id=1,
        )

        out = trainer._reanalyze_batch([ex])

        assert "mcts" in received, (
            "reanalyze never reached MCTS — it passed a config object lacking "
            "n_simulations (TrainingConfig instead of MCTSConfig); the "
            "AttributeError was swallowed by the bare except"
        )
        assert received["mcts"].n_simulations == 2
        assert torch.allclose(out[0].policy_target, sentinel_policy)


# ---------------------------------------------------------------------------
# 5. reconstruct_game_state: rebuild a PyGameState from graph + GameConfig
#
# Used to recover game_state for legacy TrainingExamples that were pickled
# before PyGameState gained pickle support (which silently dropped state).
# ---------------------------------------------------------------------------


class TestReconstructGameState:
    """Round-trip: GameState -> graph Data -> reconstruct_game_state -> GameState."""

    def _assert_states_equivalent(self, original, reconstructed):
        assert set(reconstructed.placed_stones()) == set(original.placed_stones())
        assert reconstructed.current_player() == original.current_player()
        assert (
            reconstructed.moves_remaining_this_turn()
            == original.moves_remaining_this_turn()
        )
        assert set(reconstructed.legal_moves()) == set(original.legal_moves())

    def test_hex_graph_round_trip_mid_turn(self):
        """moves_remaining = 1 (a turn was started but not finished)."""
        from hexo_a0.graph import game_to_graph, reconstruct_game_state

        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)   # P2 #1
        game.apply_move(-1, 0)  # P2 #2 -> P1 to move
        game.apply_move(0, 1)   # P1 #1 -> still P1, moves_remaining=1
        assert game.current_player() == "P1"
        assert game.moves_remaining_this_turn() == 1

        data = game_to_graph(game)
        reconstructed = reconstruct_game_state(data, cfg)
        self._assert_states_equivalent(game, reconstructed)

    def test_hex_graph_round_trip_start_of_turn(self):
        """moves_remaining = 2 (a fresh turn)."""
        from hexo_a0.graph import game_to_graph, reconstruct_game_state

        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)   # P2 #1
        game.apply_move(-1, 0)  # P2 #2 -> P1 starts new turn with 2 moves
        assert game.current_player() == "P1"
        assert game.moves_remaining_this_turn() == 2

        data = game_to_graph(game)
        reconstructed = reconstruct_game_state(data, cfg)
        self._assert_states_equivalent(game, reconstructed)

    def test_axis_graph_round_trip(self):
        """axis-window graph uses the same 8-dim node feature schema."""
        from hexo_a0.graph import game_to_axis_graph, reconstruct_game_state

        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)
        game.apply_move(-1, 0)
        game.apply_move(0, 1)
        game.apply_move(0, -1)
        # P2 now to move, 2 remaining
        assert game.current_player() == "P2"
        assert game.moves_remaining_this_turn() == 2

        data = game_to_axis_graph(game)
        reconstructed = reconstruct_game_state(data, cfg)
        self._assert_states_equivalent(game, reconstructed)

    def test_custom_game_config(self):
        """A non-default GameConfig should reconstruct identically."""
        from hexo_a0.graph import game_to_graph, reconstruct_game_state

        cfg = hexo_rs.GameConfig(win_length=5, placement_radius=3, max_moves=15)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)
        game.apply_move(0, 1)
        data = game_to_graph(game)
        reconstructed = reconstruct_game_state(data, cfg)
        self._assert_states_equivalent(game, reconstructed)
        # Verify config propagated correctly: legal_moves should respect radius=3.
        rcfg = reconstructed.config()
        assert (rcfg.win_length, rcfg.placement_radius, rcfg.max_moves) == (5, 3, 15)


class TestReconstructGameStateRelative(TestReconstructGameState):
    """Relative-encoding graphs (7/11-dim) drop the to_move feature; the side
    to move is recovered from stone-count parity (origin pre-seeded P1, P2
    places first, two placements per turn): with m = stones − 1 placements,
    P1 moves iff m % 4 ∈ {2, 3}, and moves_remaining = 2 − (m % 2) must match
    the stored feature."""

    # Moves from the seeded origin: P2, P2, P1, P1, P2, ...
    _MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)]

    def _game_after(self, n_placements, cfg):
        game = hexo_rs.GameState(cfg)
        for q, r in self._MOVES[:n_placements]:
            game.apply_move(q, r)
        return game

    def test_round_trip_all_parity_phases(self):
        """m % 4 ∈ {0,1,2,3} covers P2/P1 × start-of-turn/mid-turn."""
        from hexo_a0.graph import game_to_graph, reconstruct_game_state

        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        expected = [("P2", 2), ("P2", 1), ("P1", 2), ("P1", 1), ("P2", 2)]
        for m, (player, mr) in enumerate(expected):
            game = self._game_after(m, cfg)
            assert game.current_player() == player
            assert game.moves_remaining_this_turn() == mr

            data = game_to_graph(game, relative_stones=True)
            assert data.x.shape[1] == 7
            reconstructed = reconstruct_game_state(data, cfg)
            self._assert_states_equivalent(game, reconstructed)

    def test_round_trip_with_threat_features(self):
        """11-dim relative layout decodes via the same dims 0-3."""
        from hexo_a0.graph import game_to_graph, reconstruct_game_state

        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game = self._game_after(3, cfg)
        data = game_to_graph(game, relative_stones=True, threat_features=True)
        assert data.x.shape[1] == 11
        reconstructed = reconstruct_game_state(data, cfg)
        self._assert_states_equivalent(game, reconstructed)

    def test_axis_graph_round_trip_relative(self):
        from hexo_a0.graph import game_to_axis_graph, reconstruct_game_state

        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game = self._game_after(4, cfg)
        data = game_to_axis_graph(game, relative_stones=True)
        reconstructed = reconstruct_game_state(data, cfg)
        self._assert_states_equivalent(game, reconstructed)

    def test_parity_mismatch_refuses(self):
        """A moves_remaining feature contradicting stone-count parity means
        the position is not from an origin-seeded alternating game."""
        import pytest

        from hexo_a0.graph import game_to_graph, reconstruct_game_state

        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game = self._game_after(2, cfg)  # P1 to move, mr=2
        data = game_to_graph(game, relative_stones=True)
        data.x[:, 3] = 0.5  # tamper: claim mr=1 on every row
        with pytest.raises(ValueError, match="parity"):
            reconstruct_game_state(data, cfg)

    # Absolute-layout tests from the base class run unchanged against the
    # updated implementation (inherited), keeping the absolute path covered.


# ---------------------------------------------------------------------------
# 6. _reanalyze_batch falls back to reconstruction for legacy game_state=None
# ---------------------------------------------------------------------------


class TestReanalyzeReconstructsLegacyExamples(TestReanalyzeBatch):
    """For legacy examples with game_state=None but valid `data`, reanalyze
    must reconstruct the state from the graph + current GameConfig and call
    MCTS on the reconstruction."""

    def test_reanalyze_reconstructs_when_game_state_is_none(
        self, tmp_path, monkeypatch
    ):
        from hexo_a0.graph import game_to_graph

        trainer = self._make_trainer(tmp_path, monkeypatch, reanalyze_fraction=1.0)

        sentinel_policy = torch.tensor([0.42, 0.58], dtype=torch.float32)
        received_states: list = []

        def fake_batched(states, network, mcts, device, **kwargs):
            received_states.extend(states)
            return [sentinel_policy.clone() for _ in states]

        import hexo_a0.self_play as sp_mod
        monkeypatch.setattr(sp_mod, "_rust_batched_gumbel_mcts", fake_batched)

        # Build an original game state and graph; deliberately drop game_state
        # to simulate a legacy example pickled before PyGameState had pickle.
        # game_config on the trainer is win_length=4, placement_radius=4.
        cfg = trainer.game_config
        original = hexo_rs.GameState(cfg)
        original.apply_move(1, 0)
        original.apply_move(-1, 0)
        original.apply_move(0, 1)

        data = game_to_graph(original)
        n_legal = int(data.legal_mask.sum().item())

        ex_legacy = TrainingExample(
            data=data,
            policy_target=torch.ones(n_legal) / n_legal,
            value_target=0.25,
            game_state=None,           # legacy -- stripped
            trajectory_id=1,
        )

        out = trainer._reanalyze_batch([ex_legacy])

        # The fake returns a 2-vector but this example has num_legal != 2, so
        # the trainer's policy-length guard skips the rewrite (counted as a
        # failure, not fatal). So instead assert the MCTS was called at least
        # once with a reconstructed GameState that matches `original`.
        assert len(received_states) >= 1, (
            "Reanalyze did not reconstruct legacy example with game_state=None"
        )
        recon = received_states[0]
        assert set(recon.placed_stones()) == set(original.placed_stones())
        assert recon.current_player() == original.current_player()
        assert (
            recon.moves_remaining_this_turn()
            == original.moves_remaining_this_turn()
        )
        assert set(recon.legal_moves()) == set(original.legal_moves())
