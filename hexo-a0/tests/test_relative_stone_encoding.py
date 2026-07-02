"""Side-to-move-relative stone encoding (model.relative_stone_encoding).

Base node layout becomes 7-dim [own, opp, empty, moves_remaining, norm_q,
norm_r, inv_stone_dist] (own = current player; absolute to_move dropped);
threat features append after the base (11-dim). From-scratch-run-only:
incompatible with training.graft_threat_features.
"""
import math

import pytest
import torch
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.config import ModelConfig, node_feature_dim
from hexo_a0.graph import (
    game_to_axis_graph,
    game_to_axis_graph_batch,
    game_to_graph,
    game_to_graph_batch,
)
from hexo_a0.model import HeXONet


@pytest.fixture
def small_game():
    # NOTE: origin (0,0) is pre-seeded with a P1 stone; P2 moves first.
    # After P2's full turn (2 placements) it is P1's turn.
    gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    g = hexo_rs.GameState(gc)
    g.apply_move(1, 0)
    g.apply_move(2, 0)
    return g


@pytest.fixture
def fresh_game():
    # Only the pre-seeded P1 origin stone; P2 to move.
    gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    return hexo_rs.GameState(gc)


def _model_cfg(*, relative: bool, threat: bool = False) -> ModelConfig:
    return ModelConfig(
        hidden_dim=32, num_layers=2, num_heads=4,
        policy_hidden=16, value_hidden=16,
        graph_type="axis",
        threat_features=threat,
        relative_stone_encoding=relative,
    )


class TestGraphFeatureDim:
    def test_axis_graph_7dim(self, small_game):
        d = game_to_axis_graph(small_game, relative_stones=True)
        assert d.x.shape[1] == 7

    def test_hex_graph_7dim(self, small_game):
        d = game_to_graph(small_game, relative_stones=True)
        assert d.x.shape[1] == 7

    def test_axis_graph_11dim_with_threat(self, small_game):
        d = game_to_axis_graph(small_game, threat_features=True, relative_stones=True)
        assert d.x.shape[1] == 11

    def test_hex_graph_11dim_with_threat(self, small_game):
        d = game_to_graph(small_game, threat_features=True, relative_stones=True)
        assert d.x.shape[1] == 11

    def test_batch_builders_match_single(self, small_game):
        for single, batch, kwargs in (
            (game_to_graph, game_to_graph_batch, {}),
            (game_to_axis_graph, game_to_axis_graph_batch, {"prune_empty_edges": True}),
        ):
            ds = single(small_game, relative_stones=True, threat_features=True, **kwargs)
            (db,) = batch([small_game], relative_stones=True, threat_features=True, **kwargs)
            torch.testing.assert_close(db.x, ds.x, rtol=0.0, atol=0.0)


class TestRelativeLayout:
    """Base features must match the absolute builder at shifted indices."""

    def test_base_features_match_absolute_p1_to_move(self, small_game):
        d_abs = game_to_axis_graph(small_game)
        d_rel = game_to_axis_graph(small_game, relative_stones=True)
        # P1 to move (abs dim 3 == +1) => own = P1: stone one-hot unchanged.
        assert d_abs.x[0, 3].item() == 1.0
        torch.testing.assert_close(d_rel.x[:, 0], d_abs.x[:, 0])  # own == p1
        torch.testing.assert_close(d_rel.x[:, 1], d_abs.x[:, 1])  # opp == p2
        torch.testing.assert_close(d_rel.x[:, 2], d_abs.x[:, 2])  # empty
        torch.testing.assert_close(d_rel.x[:, 3], d_abs.x[:, 4])  # moves_remaining
        torch.testing.assert_close(d_rel.x[:, 4:7], d_abs.x[:, 5:8])  # coords + dist

    def test_own_opp_flip_with_side_to_move(self, fresh_game):
        # P2 to move (abs dim 3 == -1) => own = P2: one-hot columns swap.
        d_abs = game_to_axis_graph(fresh_game)
        d_rel = game_to_axis_graph(fresh_game, relative_stones=True)
        assert d_abs.x[0, 3].item() == -1.0
        torch.testing.assert_close(d_rel.x[:, 0], d_abs.x[:, 1])  # own == p2
        torch.testing.assert_close(d_rel.x[:, 1], d_abs.x[:, 0])  # opp == p1
        torch.testing.assert_close(d_rel.x[:, 2], d_abs.x[:, 2])
        torch.testing.assert_close(d_rel.x[:, 3], d_abs.x[:, 4])
        torch.testing.assert_close(d_rel.x[:, 4:7], d_abs.x[:, 5:8])

    def test_same_stone_flips_between_positions(self, fresh_game, small_game):
        # The pre-seeded P1 origin stone is the FIRST node (stones sort
        # first; only stone in fresh_game) in both graphs of fresh_game.
        d_fresh = game_to_axis_graph(fresh_game, relative_stones=True)
        # fresh: P2 to move -> origin P1 stone is opp.
        torch.testing.assert_close(
            d_fresh.x[0, 0:3], torch.tensor([0.0, 1.0, 0.0])
        )
        # small_game: P1 to move -> the origin stone is own. Stones sort by
        # (q, r): (0,0) < (1,0) < (2,0), so it is still row 0.
        d_small = game_to_axis_graph(small_game, relative_stones=True)
        torch.testing.assert_close(
            d_small.x[0, 0:3], torch.tensor([1.0, 0.0, 0.0])
        )

    def test_threat_dims_match_absolute(self, small_game):
        d_abs = game_to_axis_graph(small_game, threat_features=True)
        d_rel = game_to_axis_graph(
            small_game, threat_features=True, relative_stones=True
        )
        # Threat features are side-to-move relative in BOTH layouts; only
        # their position shifts (abs 8:12 -> rel 7:11).
        torch.testing.assert_close(
            d_rel.x[:, 7:11], d_abs.x[:, 8:12], rtol=0.0, atol=0.0
        )


class TestAugmentRelativeLayout:
    def test_random_augment_recomputes_shifted_coord_dims(self, small_game, monkeypatch):
        # random_augment must recompute the centroid-relative coord features
        # at dims 4,5 for the relative layout (5,6 absolute). Force the same
        # transform on both layouts and compare row-for-row: node ordering
        # is coords-determined, hence identical across layouts.
        import random

        from hexo_a0.graph import random_augment

        d_abs = game_to_axis_graph(small_game, threat_features=True)
        d_rel = game_to_axis_graph(
            small_game, threat_features=True, relative_stones=True
        )
        n_legal = int(d_abs.legal_mask.sum())
        policy = torch.full((n_legal,), 1.0 / n_legal)

        monkeypatch.setattr(random, "choice", lambda seq: seq[2])  # rot180
        aug_abs, pol_abs = random_augment(d_abs, policy)
        monkeypatch.setattr(random, "choice", lambda seq: seq[2])
        aug_rel, pol_rel = random_augment(d_rel, policy)

        assert aug_rel.x.shape[1] == 11
        torch.testing.assert_close(aug_rel.coords, aug_abs.coords)
        # Recomputed coord features land at the shifted indices...
        torch.testing.assert_close(aug_rel.x[:, 4:6], aug_abs.x[:, 5:7])
        # ...and the remaining base + threat dims ride along unchanged.
        torch.testing.assert_close(aug_rel.x[:, 2], aug_abs.x[:, 2])
        torch.testing.assert_close(aug_rel.x[:, 3], aug_abs.x[:, 4])
        torch.testing.assert_close(aug_rel.x[:, 6], aug_abs.x[:, 7])
        torch.testing.assert_close(aug_rel.x[:, 7:11], aug_abs.x[:, 8:12])
        torch.testing.assert_close(pol_rel, pol_abs)


class TestReconstructRelative:
    def test_reconstruct_round_trips_relative_graphs(self, small_game):
        """Relative graphs reconstruct absolutely via stone-count parity
        (origin pre-seeded P1, P2 first, 2-placement turns). Full parity-phase
        and tamper-refusal coverage lives in test_reanalyze.py
        TestReconstructGameStateRelative."""
        from hexo_a0.graph import reconstruct_game_state

        gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        d = game_to_axis_graph(small_game, relative_stones=True)
        rec = reconstruct_game_state(d, gc)
        assert set(rec.placed_stones()) == set(small_game.placed_stones())
        assert rec.current_player() == small_game.current_player()
        assert rec.moves_remaining_this_turn() == small_game.moves_remaining_this_turn()


class TestModelInDim:
    @pytest.mark.parametrize(
        "relative,threat,expected",
        [(False, False, 8), (True, False, 7), (False, True, 12), (True, True, 11)],
    )
    def test_in_dim_all_flag_combos(self, relative, threat, expected):
        cfg = _model_cfg(relative=relative, threat=threat)
        assert node_feature_dim(cfg) == expected
        model = HeXONet(cfg)
        assert model.representation.input_proj.weight.shape[1] == expected

    def test_forward_on_7dim_graph(self, small_game):
        model = HeXONet(_model_cfg(relative=True))
        model.eval()
        d = game_to_axis_graph(small_game, relative_stones=True)
        with torch.no_grad():
            logits_list, values = model.forward_batch(Batch.from_data_list([d]))
        assert logits_list[0].shape[0] == int(d.legal_mask.sum())
        assert torch.isfinite(values).all()

    def test_forward_on_11dim_graph(self, small_game):
        model = HeXONet(_model_cfg(relative=True, threat=True))
        model.eval()
        d = game_to_axis_graph(small_game, threat_features=True, relative_stones=True)
        with torch.no_grad():
            logits_list, values = model.forward_batch(Batch.from_data_list([d]))
        assert logits_list[0].shape[0] == int(d.legal_mask.sum())
        assert torch.isfinite(values).all()


class TestGraftRefusal:
    def test_graft_refuses_relative_model_config(self):
        from hexo_a0.model import graft_input_proj_for_threat_features

        m8 = HeXONet(_model_cfg(relative=False))
        sd = {k: v.clone() for k, v in m8.state_dict().items()}
        with pytest.raises(ValueError, match="relative_stone_encoding"):
            graft_input_proj_for_threat_features(
                sd, _model_cfg(relative=True, threat=True)
            )

    def test_graft_refuses_relative_model_config_dict(self):
        from hexo_a0.model import graft_input_proj_for_threat_features

        m8 = HeXONet(_model_cfg(relative=False))
        sd = {k: v.clone() for k, v in m8.state_dict().items()}
        with pytest.raises(ValueError, match="relative_stone_encoding"):
            graft_input_proj_for_threat_features(
                sd, {"relative_stone_encoding": True}
            )

    def test_graft_refuses_relative_width_checkpoint(self):
        from hexo_a0.model import graft_input_proj_for_threat_features

        m7 = HeXONet(_model_cfg(relative=True))
        sd = {k: v.clone() for k, v in m7.state_dict().items()}
        assert sd["representation.input_proj.weight"].shape[1] == 7
        with pytest.raises(ValueError, match="relative_stone_encoding width"):
            graft_input_proj_for_threat_features(sd)

    def test_graft_still_works_with_absolute_config(self, small_game):
        from hexo_a0.model import graft_input_proj_for_threat_features

        m8 = HeXONet(_model_cfg(relative=False))
        sd = {k: v.clone() for k, v in m8.state_dict().items()}
        assert graft_input_proj_for_threat_features(
            sd, _model_cfg(relative=False, threat=True)
        ) is True
        assert sd["representation.input_proj.weight"].shape[1] == 12


class TestNodeFeatureNames:
    def test_layout_aware_names(self):
        from hexo_a0.trainer import _node_feature_names

        abs_names = _node_feature_names(_model_cfg(relative=False, threat=True))
        rel_names = _node_feature_names(_model_cfg(relative=True, threat=True))
        assert abs_names[:8] == [
            "p1_stone", "p2_stone", "empty", "to_move", "moves_remaining",
            "norm_q", "norm_r", "inv_stone_dist",
        ]
        assert rel_names[:7] == [
            "own_stone", "opp_stone", "empty", "moves_remaining",
            "norm_q", "norm_r", "inv_stone_dist",
        ]
        assert abs_names[8:] == rel_names[7:] == [
            "own_max_line", "opp_max_line", "own_threat_axes", "opp_threat_axes",
        ]


class TestTrainerSmoke:
    def test_train_steps_with_relative_graphs(self):
        """Trainer trains on relative(+threat) graphs end-to-end (mirrors
        test_training_integration.test_full_training_loop)."""
        from hexo_a0 import HeXONet, Trainer, TrainingExample
        from hexo_a0.config import (
            EvalConfig,
            FullConfig,
            MCTSConfig,
            PythonSelfPlayConfig,
            SelfPlayConfig,
            TrainingConfig,
        )

        game_config = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=30)
        model_config = _model_cfg(relative=True, threat=True)
        config = FullConfig(
            model=model_config,
            mcts=MCTSConfig(n_simulations=2, m_actions=4, exploration_moves=5),
            training=TrainingConfig(
                batch_size=8, buffer_capacity=500, edge_budget=0,
                grad_accumulation=False, lr_schedule="constant",
                lr_warmup_steps=0, total_train_steps=0, target_reuse=0.0,
                augment_symmetries=True,  # exercises random_augment at 11-dim
            ),
            self_play=SelfPlayConfig(
                device="cpu", workers=1, precision="fp32",
                python=PythonSelfPlayConfig(games_per_write=2),
            ),
            eval=EvalConfig(interval=0, games=0, sims=0, checkpoint_interval=0),
        )
        device = torch.device("cpu")
        model = HeXONet(model_config).to(device)
        trainer = Trainer(model, config, game_config, device)

        game = hexo_rs.GameState(game_config)
        game.apply_move(1, 0)
        game.apply_move(2, 0)
        data = game_to_axis_graph(
            game,
            prune_empty_edges=model_config.prune_empty_edges,
            threat_features=True,
            relative_stones=True,
        )
        n_legal = int(data.legal_mask.sum())
        for _ in range(20):
            trainer.buffer.add(TrainingExample(
                data=data,
                policy_target=torch.ones(n_legal) / n_legal,
                value_target=0.0,
            ))

        metrics = trainer.train(steps=3)
        assert math.isfinite(metrics["total_loss"])
        assert math.isfinite(metrics["policy_loss"])
        assert math.isfinite(metrics["value_loss"])
        assert trainer.train_steps == 3


class TestGameStatesToBatch:
    """The raw hex collation path (hexo_rs.game_states_to_batch) must honour
    the threat/relative flags so the self-play raw path feeds the model
    graphs at node_feature_dim width."""

    @pytest.mark.parametrize("threat,relative,dim", [
        (False, False, 8),
        (False, True, 7),
        (True, False, 12),
        (True, True, 11),
    ])
    def test_feature_width_per_flag_combo(self, small_game, threat, relative, dim):
        out = hexo_rs.game_states_to_batch(
            [small_game, small_game.clone()],
            threat_features=threat, relative_stones=relative,
        )
        features, stone_mask = out[0], out[4]
        total_nodes = len(stone_mask)
        assert total_nodes > 0
        assert len(features) == total_nodes * dim

    def test_defaults_are_absolute_8dim(self, small_game):
        out = hexo_rs.game_states_to_batch([small_game])
        features, stone_mask = out[0], out[4]
        assert len(features) == len(stone_mask) * 8

    @pytest.mark.parametrize("threat,relative", [
        (False, True), (True, False), (True, True),
    ])
    def test_features_match_single_graph_builder(self, small_game, threat, relative):
        out = hexo_rs.game_states_to_batch(
            [small_game], threat_features=threat, relative_stones=relative,
        )
        features = out[0]
        d = game_to_graph(small_game, threat_features=threat, relative_stones=relative)
        assert features == pytest.approx(d.x.flatten().tolist())


class TestRawHexSelfPlayPath:
    """End-to-end: batched_self_play_games with graph_type='hex' uses the raw
    tensor path; the collated features must match the model's input width for
    every flag combo (was hardcoded to 8 before the fix)."""

    @pytest.mark.parametrize("threat,relative", [
        (False, False), (True, False), (False, True), (True, True),
    ])
    def test_batched_self_play_hex_raw(self, threat, relative):
        from hexo_a0.config import (
            EvalConfig, FullConfig, MCTSConfig, SelfPlayConfig,
        )
        from hexo_a0.self_play import batched_self_play_games

        mc = ModelConfig(
            hidden_dim=16, num_layers=1, num_heads=1,
            policy_hidden=8, value_hidden=8,
            graph_type="hex", conv_type="gatv2",
            threat_features=threat,
            relative_stone_encoding=relative,
        )
        config = FullConfig(
            model=mc,
            mcts=MCTSConfig(n_simulations=2, m_actions=2, exploration_moves=2),
            self_play=SelfPlayConfig(device="cpu", workers=1, precision="fp32"),
            eval=EvalConfig(interval=0, games=0, sims=0, checkpoint_interval=0),
        )
        device = torch.device("cpu")
        net = HeXONet(mc).to(device)
        gc = hexo_rs.GameConfig(win_length=3, placement_radius=2, max_moves=12)

        from hexo_a0.graph import graph_fn_from_model_config
        trajectories = batched_self_play_games(net, config, gc, device, n_games=1)
        assert len(trajectories) == 1
        assert len(trajectories[0]) > 0
        expected_dim = node_feature_dim(mc)
        gfn = graph_fn_from_model_config(mc)
        for ex in trajectories[0]:
            d = gfn(ex.game_state)
            assert d.x.shape[1] == expected_dim
