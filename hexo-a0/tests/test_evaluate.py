"""Tests for evaluate.py: evaluation games against random and checkpoint opponents."""

import pytest
import torch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_a0.evaluate import (
    _mcts_argmax_move,
    _model_move,
    _random_move,
    play_eval_game,
    sample_opening,
    evaluate_vs_random,
    evaluate_vs_checkpoint,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_MODEL_CONFIG = ModelConfig(
    hidden_dim=32, num_layers=2, num_heads=4,
    policy_hidden=16, value_hidden=16,
    graph_type="hex", conv_type="gatv2",
)
FAST_GAME_CONFIG = hexo_rs.GameConfig(
    win_length=4, placement_radius=2, max_moves=80,
)
DEVICE = torch.device("cpu")


@pytest.fixture
def model():
    net = HeXONet(TINY_MODEL_CONFIG).to(DEVICE)
    net.eval()
    return net


@pytest.fixture
def game():
    return hexo_rs.GameState(FAST_GAME_CONFIG)


# ---------------------------------------------------------------------------
# _random_move
# ---------------------------------------------------------------------------


class TestRandomMove:
    def test_returns_legal_move(self, game):
        legal = game.legal_moves()
        move = _random_move(game)
        assert move in legal

    def test_returns_tuple_of_two_ints(self, game):
        move = _random_move(game)
        assert isinstance(move, tuple)
        assert len(move) == 2
        assert isinstance(move[0], int)
        assert isinstance(move[1], int)


# ---------------------------------------------------------------------------
# _model_move
# ---------------------------------------------------------------------------


class TestModelMove:
    def test_greedy_deterministic(self, model, game):
        """Temperature=0 (greedy) returns the same move every time."""
        moves = [_model_move(model, game, DEVICE, temperature=0.0) for _ in range(5)]
        assert all(m == moves[0] for m in moves)

    def test_greedy_returns_legal_move(self, model, game):
        move = _model_move(model, game, DEVICE, temperature=0.0)
        assert move in game.legal_moves()

    def test_stochastic_returns_legal_move(self, model, game):
        """Temperature>0 still returns a legal move."""
        move = _model_move(model, game, DEVICE, temperature=1.0)
        assert move in game.legal_moves()

    def test_stochastic_path_runs(self, model, game):
        """Temperature>0 exercises the multinomial sampling branch without error."""
        for _ in range(10):
            move = _model_move(model, game, DEVICE, temperature=0.5)
            assert move in game.legal_moves()


# ---------------------------------------------------------------------------
# play_eval_game
# ---------------------------------------------------------------------------


class TestPlayEvalGame:
    def test_model_vs_random_p1(self, model):
        result = play_eval_game(
            model, FAST_GAME_CONFIG, DEVICE,
            opponent="random", model_side="P1",
        )
        assert "winner" in result
        assert "moves" in result
        assert "model_side" in result
        assert result["model_side"] == "P1"
        assert result["winner"] in ("P1", "P2", None)
        assert result["moves"] > 0

    def test_model_vs_random_p2(self, model):
        result = play_eval_game(
            model, FAST_GAME_CONFIG, DEVICE,
            opponent="random", model_side="P2",
        )
        assert result["model_side"] == "P2"
        assert result["winner"] in ("P1", "P2", None)

    def test_model_vs_model(self, model):
        """When opponent is another nn.Module, both sides use _model_move."""
        opponent = HeXONet(TINY_MODEL_CONFIG).to(DEVICE)
        opponent.eval()
        result = play_eval_game(
            model, FAST_GAME_CONFIG, DEVICE,
            opponent=opponent, model_side="P1",
        )
        assert result["winner"] in ("P1", "P2", None)
        assert result["moves"] > 0

    def test_per_side_search_config_mcts_smoke(self, model):
        """Per-side search-config overrides thread through the MCTS path.

        Exercises both sides running Gumbel MCTS at *different* operating
        points (model m=4/c_scale=1.0 vs opponent m=2/c_scale=0.5) — would
        raise if any of the new kwargs weren't accepted or weren't wired into
        the opponent's MCTSConfig.
        """
        result = play_eval_game(
            model, FAST_GAME_CONFIG, DEVICE,
            opponent=model, model_side="P1",
            model_config=TINY_MODEL_CONFIG, opponent_config=TINY_MODEL_CONFIG,
            mcts_sims=8, mcts_m_actions=4, mcts_c_scale=1.0,
            opponent_mcts_sims=8, opponent_mcts_m_actions=2,
            opponent_mcts_c_scale=0.5,
        )
        assert result["winner"] in ("P1", "P2", None)
        assert result["moves"] > 0

    def test_correct_winner_attribution(self, model):
        """Winner is one of the players or None (draw), never something else."""
        for side in ("P1", "P2"):
            result = play_eval_game(
                model, FAST_GAME_CONFIG, DEVICE,
                opponent="random", model_side=side,
            )
            assert result["winner"] in ("P1", "P2", None)

    def test_unknown_opponent_raises(self, model):
        with pytest.raises(ValueError, match="Unknown opponent type"):
            play_eval_game(
                model, FAST_GAME_CONFIG, DEVICE,
                opponent="bad_value", model_side="P1",
            )

    def test_game_terminates(self, model):
        """Game always finishes (max_moves ensures termination)."""
        short_config = hexo_rs.GameConfig(
            win_length=4, placement_radius=2, max_moves=10,
        )
        result = play_eval_game(
            model, short_config, DEVICE,
            opponent="random", model_side="P1",
        )
        assert result["moves"] <= 10
        assert result["winner"] in ("P1", "P2", None)


# ---------------------------------------------------------------------------
# evaluate_vs_random
# ---------------------------------------------------------------------------


class TestEvaluateVsRandom:
    def test_result_keys(self, model):
        result = evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=4)
        assert "win_rate" in result
        assert "wins" in result
        assert "losses" in result
        assert "draws" in result
        assert "mean_game_length" in result

    def test_counts_add_up(self, model):
        n = 6
        result = evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=n)
        assert result["wins"] + result["losses"] + result["draws"] == n

    def test_win_rate_range(self, model):
        result = evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=4)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_half_p1_half_p2(self, model):
        """Plays half as P1, half as P2 (verified via mock/counting)."""
        n = 6
        sides_seen = []
        original_play = play_eval_game.__wrapped__ if hasattr(play_eval_game, "__wrapped__") else None

        import hexo_a0.evaluate as eval_mod
        original_fn = eval_mod.play_eval_game

        def tracking_play(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            sides_seen.append(result["model_side"])
            return result

        eval_mod.play_eval_game = tracking_play
        try:
            evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=n)
        finally:
            eval_mod.play_eval_game = original_fn

        assert sides_seen.count("P1") == n // 2
        assert sides_seen.count("P2") == n - n // 2

    def test_mean_game_length_positive(self, model):
        result = evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=4)
        assert result["mean_game_length"] > 0

    def test_zero_games(self, model):
        result = evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=0)
        assert result["wins"] == 0
        assert result["losses"] == 0
        assert result["draws"] == 0
        assert result["win_rate"] == 0.0
        assert result["mean_game_length"] == 0.0

    def test_always_uses_raw_policy_never_mcts(self, model, monkeypatch):
        """evaluate_vs_random must exclusively use the raw policy head — no MCTS.

        The vs-random baseline is a vibe check on the policy itself, so
        MCTS is deliberately not allowed here (see design note in
        `evaluate.py`). If this test starts failing, someone re-enabled
        MCTS for vs-random and the metric is no longer a raw-policy
        signal.
        """
        from hexo_a0 import evaluate as ev

        calls = {"mcts": 0, "raw": 0}

        def fake_mcts_move(_m, game, *_a, **_k):
            calls["mcts"] += 1
            return game.legal_moves()[0]

        def fake_raw_move(_m, game, *_a, **_k):
            calls["raw"] += 1
            return game.legal_moves()[0]

        monkeypatch.setattr(ev, "_mcts_argmax_move", fake_mcts_move)
        monkeypatch.setattr(ev, "_model_move", fake_raw_move)

        evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=4)
        assert calls["mcts"] == 0, "evaluate_vs_random must not invoke MCTS"
        assert calls["raw"] > 0, "evaluate_vs_random must invoke the raw policy path"

    def test_does_not_accept_mcts_kwargs(self, model):
        """The mcts_sims/mcts_m_actions kwargs were removed — passing them
        should be a clear TypeError so stale callers fail loudly instead
        of silently being ignored."""
        with pytest.raises(TypeError):
            evaluate_vs_random(
                model, FAST_GAME_CONFIG, DEVICE, n_games=2, mcts_sims=4,
            )


# ---------------------------------------------------------------------------
# evaluate_vs_checkpoint
# ---------------------------------------------------------------------------


class TestPerSideTracking:
    """Per-side (P1 vs P2) win tracking via mocked play_eval_game."""

    def _patch(self, monkeypatch, winners):
        """Patch play_eval_game to return winners from a sequence."""
        import hexo_a0.evaluate as eval_mod
        it = iter(winners)

        def fake_play(*args, **kwargs):
            return {
                "winner": next(it),
                "moves": 5,
                "model_side": kwargs.get("model_side", "P1"),
            }

        monkeypatch.setattr(eval_mod, "play_eval_game", fake_play)

    def test_random_p1_p2_wins_tracked(self, model, monkeypatch):
        # 4 games: 2 as P1, 2 as P2. Winners: P1, P2, P1, P2
        # → p1_wins=2, p2_wins=2, p1_decided_rate=0.5
        self._patch(monkeypatch, ["P1", "P2", "P1", "P2"])
        result = evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=4)
        assert result["p1_wins"] == 2
        assert result["p2_wins"] == 2
        assert result["p1_decided_rate"] == 0.5

    def test_random_p1_dominates(self, model, monkeypatch):
        self._patch(monkeypatch, ["P1", "P1", "P1", "P1"])
        result = evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=4)
        assert result["p1_wins"] == 4
        assert result["p2_wins"] == 0
        assert result["p1_decided_rate"] == 1.0

    def test_random_all_draws_edge_case(self, model, monkeypatch):
        self._patch(monkeypatch, [None, None, None, None])
        result = evaluate_vs_random(model, FAST_GAME_CONFIG, DEVICE, n_games=4)
        assert result["p1_wins"] == 0
        assert result["p2_wins"] == 0
        assert result["p1_decided_rate"] == 0.0
        assert result["draws"] == 4

    def test_checkpoint_p1_p2_tracked(self, model, tmp_path, monkeypatch):
        ckpt_path = str(tmp_path / "opponent.pt")
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        self._patch(monkeypatch, ["P1", "P1", "P2", None])
        result = evaluate_vs_checkpoint(
            model, ckpt_path, TINY_MODEL_CONFIG, FAST_GAME_CONFIG,
            DEVICE, n_games=4,
        )
        assert result["p1_wins"] == 2
        assert result["p2_wins"] == 1
        assert result["p1_decided_rate"] == pytest.approx(2 / 3)

    def test_checkpoint_all_draws_edge_case(self, model, tmp_path, monkeypatch):
        ckpt_path = str(tmp_path / "opponent.pt")
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        self._patch(monkeypatch, [None, None, None, None])
        result = evaluate_vs_checkpoint(
            model, ckpt_path, TINY_MODEL_CONFIG, FAST_GAME_CONFIG,
            DEVICE, n_games=4,
        )
        assert result["p1_wins"] == 0
        assert result["p2_wins"] == 0
        assert result["p1_decided_rate"] == 0.0


class TestEvaluateVsCheckpoint:
    def test_loads_opponent_and_plays(self, model, tmp_path):
        """Save a checkpoint, then evaluate against it."""
        ckpt_path = str(tmp_path / "opponent.pt")
        # Save a minimal checkpoint with just model_state_dict
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        result = evaluate_vs_checkpoint(
            model, ckpt_path, TINY_MODEL_CONFIG, FAST_GAME_CONFIG,
            DEVICE, n_games=4,
        )
        assert "win_rate" in result
        assert "wins" in result
        assert "losses" in result
        assert "draws" in result
        assert "opponent_path" in result
        assert result["opponent_path"] == ckpt_path

    def test_counts_add_up(self, model, tmp_path):
        ckpt_path = str(tmp_path / "opponent.pt")
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        n = 6
        result = evaluate_vs_checkpoint(
            model, ckpt_path, TINY_MODEL_CONFIG, FAST_GAME_CONFIG,
            DEVICE, n_games=n,
        )
        assert result["wins"] + result["losses"] + result["draws"] == n

    def test_same_model_mostly_draws_or_balanced(self, model, tmp_path):
        """When model plays itself, results should be roughly balanced."""
        ckpt_path = str(tmp_path / "opponent.pt")
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        result = evaluate_vs_checkpoint(
            model, ckpt_path, TINY_MODEL_CONFIG, FAST_GAME_CONFIG,
            DEVICE, n_games=4,
        )
        # Just verify it completes and returns valid data
        assert 0.0 <= result["win_rate"] <= 1.0
        assert result["mean_game_length"] > 0

    def test_with_trainer_checkpoint(self, model, tmp_path):
        """Use Trainer.save_checkpoint to create a realistic checkpoint file."""
        from hexo_a0.trainer import Trainer
        from hexo_a0.config import FullConfig, MCTSConfig

        config = FullConfig(
            model=TINY_MODEL_CONFIG,
            mcts=MCTSConfig(n_simulations=2, m_actions=4, exploration_moves=10),
        )
        trainer = Trainer(
            model=model,
            config=config,
            game_config=FAST_GAME_CONFIG,
            device=DEVICE,
        )
        ckpt_path = str(tmp_path / "trainer_ckpt.pt")
        trainer.save_checkpoint(ckpt_path)

        result = evaluate_vs_checkpoint(
            model, ckpt_path, TINY_MODEL_CONFIG, FAST_GAME_CONFIG,
            DEVICE, n_games=4,
        )
        assert result["wins"] + result["losses"] + result["draws"] == 4
        assert result["opponent_path"] == ckpt_path

    def test_orig_mod_checkpoint_loads_weights_not_random(self, model, tmp_path):
        """Checkpoint with _orig_mod. prefixed keys must load real weights, not random init."""
        from hexo_a0.evaluate import _model_move
        from hexo_a0.graph import game_to_graph

        ckpt_path = str(tmp_path / "compiled_opponent.pt")
        # Simulate a checkpoint saved from a compiled model (old bug)
        prefixed_sd = {
            f"_orig_mod.{k}": v for k, v in model.state_dict().items()
        }
        torch.save({"model_state_dict": prefixed_sd}, ckpt_path)

        # Load the opponent the same way evaluate_vs_checkpoint does
        opponent = HeXONet(TINY_MODEL_CONFIG).to(DEVICE)
        raw = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        ckpt_sd = {k.removeprefix("_orig_mod."): v for k, v in raw["model_state_dict"].items()}
        opponent.load_state_dict(ckpt_sd, strict=False)
        opponent.eval()

        # Verify every parameter matches the original model
        for name, param in model.named_parameters():
            loaded = dict(opponent.named_parameters())[name]
            assert torch.equal(param, loaded), (
                f"Parameter '{name}' was not loaded — opponent has random weights"
            )


# ---------------------------------------------------------------------------
# MCTS-based eval (argmax of improved policy — Gumbel AZ "no exploration")
# ---------------------------------------------------------------------------


class TestMctsArgmaxMove:
    def test_returns_legal_move(self, model, game):
        from hexo_a0.graph import game_to_graph

        mcts_config = hexo_rs.MCTSConfig(
            n_simulations=8, m_actions=8, c_visit=50, c_scale=1.0,
        )
        action = _mcts_argmax_move(model, game, DEVICE, game_to_graph, mcts_config)
        assert action in game.legal_moves()

    def test_argmax_of_improved_policy(self, model, game, monkeypatch):
        """The played action must be legal_moves[argmax(pi')], not the
        Gumbel-sampled action returned by the Rust call.

        Uses ``gumbel_mcts_with_diagnostics`` (the richer 7-tuple variant)
        because ``_mcts_argmax_move`` was switched to that to expose the
        ``chosen_action_forced_candidates`` capture output (E.4-inject).
        """
        from hexo_a0.graph import game_to_graph

        legal = game.legal_moves()
        # Fake improved policy peaked on index 2; Rust-returned action is a
        # different legal move to make sure we ignore it.
        fake_pi = [0.0] * len(legal)
        fake_pi[2] = 1.0
        fake_action = legal[5]

        def fake_mcts(_game, _eval_fn, _cfg, seed=None, forced_candidates=None):
            return (
                fake_action,    # action
                fake_pi,        # improved_policy
                [0] * len(legal),   # visit_counts
                [0.0] * len(legal), # per_child_q
                [0.0] * len(legal), # per_child_prior
                [0],            # candidate_indices
                [],             # chosen_action_forced_candidates
            )

        monkeypatch.setattr(hexo_rs, "gumbel_mcts_with_diagnostics", fake_mcts)

        mcts_config = hexo_rs.MCTSConfig(
            n_simulations=4, m_actions=4, c_visit=50, c_scale=1.0,
        )
        action = _mcts_argmax_move(model, game, DEVICE, game_to_graph, mcts_config)
        assert action == legal[2], "Should pick argmax(pi'), not the Rust-returned action"

    def test_play_eval_game_routes_to_mcts_when_sims_positive(self, model, monkeypatch):
        """play_eval_game with mcts_sims>0 must call _mcts_argmax_move, not _model_move."""
        from hexo_a0 import evaluate as ev

        calls = {"mcts": 0, "raw": 0}

        def fake_mcts_move(_m, game, *_a, **_k):
            calls["mcts"] += 1
            return game.legal_moves()[0]

        def fake_raw_move(_m, game, *_a, **_k):
            calls["raw"] += 1
            return game.legal_moves()[0]

        monkeypatch.setattr(ev, "_mcts_argmax_move", fake_mcts_move)
        monkeypatch.setattr(ev, "_model_move", fake_raw_move)

        play_eval_game(
            model, FAST_GAME_CONFIG, DEVICE,
            opponent="random", model_side="P1",
            mcts_sims=4, mcts_m_actions=4,
        )
        assert calls["mcts"] > 0
        assert calls["raw"] == 0

    def test_play_eval_game_raw_path_when_sims_zero(self, model, monkeypatch):
        from hexo_a0 import evaluate as ev

        calls = {"mcts": 0, "raw": 0}

        def fake_mcts_move(_m, game, *_a, **_k):
            calls["mcts"] += 1
            return game.legal_moves()[0]

        def fake_raw_move(_m, game, *_a, **_k):
            calls["raw"] += 1
            return game.legal_moves()[0]

        monkeypatch.setattr(ev, "_mcts_argmax_move", fake_mcts_move)
        monkeypatch.setattr(ev, "_model_move", fake_raw_move)

        play_eval_game(
            model, FAST_GAME_CONFIG, DEVICE,
            opponent="random", model_side="P1",
            mcts_sims=0,
        )
        assert calls["raw"] > 0
        assert calls["mcts"] == 0


# ---------------------------------------------------------------------------
# E.4-inject: joint p₂ candidate injection plumbing
# ---------------------------------------------------------------------------


class TestE4InjectPlumbing:
    """Verify that `mcts_forced_candidate_capture_k` flows from the
    `play_eval_game` kwargs through to the MCTS call, and the resulting
    captured p₂ candidates re-appear in the next-move (mr=1) candidate set.

    We don't try to assert anything about win/loss — just that the
    plumbing is wired and the cross-move state passes correctly.
    """

    def test_capture_k_zero_smoke(self, model):
        """Smoke: capture_k=0 (default) leaves play_eval_game unchanged
        and games complete normally."""
        result = play_eval_game(
            model, FAST_GAME_CONFIG, DEVICE,
            opponent=model, model_side="P1",
            mcts_sims=4, mcts_m_actions=4,
            model_config=TINY_MODEL_CONFIG, opponent_config=TINY_MODEL_CONFIG,
            mcts_forced_candidate_capture_k=0,
            opponent_mcts_forced_candidate_capture_k=0,
        )
        assert result["winner"] in ("P1", "P2", None)
        assert result["moves"] > 0

    def test_capture_k_positive_completes_game(self, model):
        """End-to-end smoke (spec test #9): a multi-turn game with
        capture_k>0 for the model side completes without error.

        Uses MCTS sims small enough to be fast but large enough for SH
        to run; m_actions=4 keeps the candidate count tight so we can
        observe candidate displacement.
        """
        result = play_eval_game(
            model, FAST_GAME_CONFIG, DEVICE,
            opponent=model, model_side="P1",
            mcts_sims=4, mcts_m_actions=4,
            model_config=TINY_MODEL_CONFIG, opponent_config=TINY_MODEL_CONFIG,
            mcts_forced_candidate_capture_k=2,
            opponent_mcts_forced_candidate_capture_k=0,
        )
        assert result["winner"] in ("P1", "P2", None)
        assert result["moves"] >= 2, "need at least a full turn (2 placements) to exercise capture"

    def test_captured_candidates_appear_in_next_move_candidate_set(self, model, monkeypatch):
        """After a mr=2 root MCTS call captures top-K p₂s, the next call
        (mr=1, same side) must receive those coords as `forced_candidates`
        and the resulting `candidate_indices` must include them.

        We mock the MCTS call to record per-call: (mr, forced_candidates_arg,
        chosen_action_forced_candidates_returned, candidate_indices).
        """
        import hexo_rs
        import hexo_a0.evaluate as eval_mod

        # Use the real Rust gumbel_mcts_with_diagnostics, but wrap it so
        # we can inspect every call's args and returns.
        calls = []  # list of (mr, forced_arg_or_none, captured_returned, candidate_indices)
        real_fn = hexo_rs.gumbel_mcts_with_diagnostics

        def wrap(game, eval_fn, config, seed=None, forced_candidates=None):
            mr = game.moves_remaining_this_turn()
            forced_in = (
                list(forced_candidates) if forced_candidates is not None else None
            )
            res = real_fn(game, eval_fn, config, seed=seed, forced_candidates=forced_candidates)
            (action, policy, visits, q, prior, cand_idx, captured) = res
            calls.append({
                "mr": mr,
                "forced_in": forced_in,
                "captured": list(captured),
                "cand_idx": list(cand_idx),
                "coords": game.legal_moves(),
            })
            return res

        monkeypatch.setattr(hexo_rs, "gumbel_mcts_with_diagnostics", wrap)
        # Also patch the lazy import inside evaluate.py.
        monkeypatch.setattr(eval_mod.hexo_rs if hasattr(eval_mod, "hexo_rs") else hexo_rs,
                            "gumbel_mcts_with_diagnostics", wrap)

        play_eval_game(
            model, FAST_GAME_CONFIG, DEVICE,
            opponent=model, model_side="P1",
            mcts_sims=4, mcts_m_actions=6,
            model_config=TINY_MODEL_CONFIG, opponent_config=TINY_MODEL_CONFIG,
            mcts_forced_candidate_capture_k=3,
            opponent_mcts_forced_candidate_capture_k=0,
        )

        # Find a (mr=2, mr=1) pair where:
        # - the mr=2 call's `captured` is non-empty
        # - the mr=1 call's `forced_in` matches `captured`
        # - the mr=1 call's `cand_idx` (translated to coords) contains all the captured
        found = False
        for i in range(len(calls) - 1):
            a = calls[i]
            b = calls[i + 1]
            if a["mr"] != 2 or b["mr"] != 1:
                continue
            if not a["captured"]:
                continue
            # Translate the b call's cand_idx to coords
            b_coords = set(b["coords"][i] for i in b["cand_idx"])
            captured_set = set(tuple(c) for c in a["captured"])
            # Some captured coords may have been played as p₁ by the mr=2
            # call, removing them from the mr=1 legal set; only assert
            # that the still-legal ones appear.
            still_legal = captured_set & set(b["coords"])
            if not still_legal:
                continue
            assert still_legal.issubset(b_coords), (
                f"captured p₂s {still_legal} must appear in next mr=1 candidate_indices "
                f"(got cand_coords={b_coords})"
            )
            # The forced_in arg must match (or be a superset of) the captured set
            assert b["forced_in"] is not None, "mr=1 call must receive forced_candidates"
            forced_in_set = set(tuple(c) for c in b["forced_in"])
            assert captured_set.issubset(forced_in_set), (
                f"mr=1 forced_candidates must include all captured ({captured_set}); "
                f"got {forced_in_set}"
            )
            found = True
            break
        assert found, (
            "Test setup: no (mr=2, mr=1) pair with non-empty captured candidates was "
            f"observed. Calls:\n{calls}"
        )


# ---------------------------------------------------------------------------
# sample_opening — seeded opening generator for noise-off SPRT
# ---------------------------------------------------------------------------


class TestSampleOpening:
    def test_returns_k_legal_replayable_nonterminal(self, model):
        opening = sample_opening(
            model, FAST_GAME_CONFIG, DEVICE, k=4, temperature=1.0,
            seed=0, model_config=TINY_MODEL_CONFIG,
        )
        assert len(opening) == 4
        g = hexo_rs.GameState(FAST_GAME_CONFIG)
        for (q, r) in opening:
            assert (q, r) in g.legal_moves()
            g.apply_move(q, r)
        assert not g.is_terminal()

    def test_deterministic_given_seed(self, model):
        a = sample_opening(model, FAST_GAME_CONFIG, DEVICE, k=4,
                           temperature=1.0, seed=7, model_config=TINY_MODEL_CONFIG)
        b = sample_opening(model, FAST_GAME_CONFIG, DEVICE, k=4,
                           temperature=1.0, seed=7, model_config=TINY_MODEL_CONFIG)
        assert a == b

    def test_different_seeds_give_diversity(self, model):
        openings = {
            tuple(sample_opening(model, FAST_GAME_CONFIG, DEVICE, k=4,
                                 temperature=1.5, seed=s, model_config=TINY_MODEL_CONFIG))
            for s in range(16)
        }
        assert len(openings) > 1

    def test_raises_when_cannot_stay_nonterminal(self, model):
        # k far exceeds any non-terminal opening on this tiny board (max_moves=80)
        with pytest.raises(RuntimeError):
            sample_opening(model, FAST_GAME_CONFIG, DEVICE, k=500,
                           temperature=1.0, seed=0, model_config=TINY_MODEL_CONFIG,
                           max_resamples=3)


# ---------------------------------------------------------------------------
# play_eval_game — forced opening replay + noise-off determinism
# ---------------------------------------------------------------------------


class TestPlayEvalGameOpening:
    def test_opening_replayed_into_both_sides(self, model):
        opening = sample_opening(model, FAST_GAME_CONFIG, DEVICE, k=4,
                                 temperature=1.0, seed=1, model_config=TINY_MODEL_CONFIG)
        for side in ("P1", "P2"):
            res = play_eval_game(
                model, FAST_GAME_CONFIG, DEVICE,
                opponent="random", model_side=side,
                model_config=TINY_MODEL_CONFIG,
                opening=opening,
            )
            assert res["move_log"][:4] == opening

    def test_noise_off_same_opening_is_deterministic(self, model):
        opening = sample_opening(model, FAST_GAME_CONFIG, DEVICE, k=4,
                                 temperature=1.0, seed=2, model_config=TINY_MODEL_CONFIG)
        runs = [
            play_eval_game(
                model, FAST_GAME_CONFIG, DEVICE,
                opponent=model, model_side="P1",
                model_config=TINY_MODEL_CONFIG, opponent_config=TINY_MODEL_CONFIG,
                mcts_sims=8, mcts_m_actions=4,
                opening=opening, disable_gumbel_noise=True,
            )["move_log"]
            for _ in range(2)
        ]
        assert runs[0] == runs[1]
