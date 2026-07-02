"""Threat-encoding node features (8 -> 12 dim) end-to-end."""
import pytest
import torch
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_axis_graph, game_to_graph
from hexo_a0.model import HeXONet


@pytest.fixture
def small_game():
    # NOTE: origin (0,0) is pre-seeded with a P1 stone; P2 moves first.
    gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    g = hexo_rs.GameState(gc)
    g.apply_move(1, 0)
    g.apply_move(2, 0)   # P2's full turn (2 placements)
    return g


def _model_cfg(threat: bool) -> ModelConfig:
    return ModelConfig(
        hidden_dim=32, num_layers=2, num_heads=4,
        policy_hidden=16, value_hidden=16,
        graph_type="axis", threat_features=threat,
    )


def _node_row(data, q: int, r: int) -> int:
    """Index of the node at hex coordinate (q, r) via data.coords.

    Only safe for coords != (0,0): axis graphs append a dummy node whose
    coords are (0,0), colliding with the pre-seeded origin stone.
    """
    assert (q, r) != (0, 0)
    match = (data.coords[:, 0] == q) & (data.coords[:, 1] == r)
    idx = torch.nonzero(match, as_tuple=False).flatten()
    assert idx.numel() == 1, f"expected exactly one node at ({q},{r})"
    return int(idx.item())


class TestGraphFeatureDim:
    def test_axis_graph_12dim(self, small_game):
        d = game_to_axis_graph(small_game, threat_features=True)
        assert d.x.shape[1] == 12

    def test_axis_graph_default_8dim(self, small_game):
        d = game_to_axis_graph(small_game)
        assert d.x.shape[1] == 8

    def test_hex_graph_12dim(self, small_game):
        d = game_to_graph(small_game, threat_features=True)
        assert d.x.shape[1] == 12

    def test_first_8_dims_unchanged(self, small_game):
        d8 = game_to_axis_graph(small_game)
        d12 = game_to_axis_graph(small_game, threat_features=True)
        torch.testing.assert_close(d12.x[:, :8], d8.x)

    def test_known_threat_values(self, small_game):
        # Position: P1 stone pre-seeded at (0,0); P2 stones at (1,0),(2,0);
        # win_length = 4. After P2's two placements it is P1's turn, so
        # own = P1, opp = P2. Confirm side to move via node feature dim 3
        # (+1.0 = P1 to move, global on every node).
        d = game_to_axis_graph(small_game, threat_features=True)
        # Any row index works — dim 3 is a global feature replicated per node.
        assert d.x[0, 3].item() == 1.0  # P1 to move => own = P1

        # Hand derivation for the EMPTY node (3,0), wl=4 (threat.rs
        # algorithm: per axis, slide every wl-window containing the node
        # over the 2*wl-1 cells centred on it; a window counts for a
        # player only if it is free of the other player's stones).
        #
        # Axis (1,0): cells (0,0)..(6,0) =
        #   (0,0)=P1, (1,0)=P2, (2,0)=P2, (3,0)=e, (4,0)=e, (5,0)=e, (6,0)=e
        # Windows of length 4 containing (3,0):
        #   [(0,0)..(3,0)]: own=1 opp=2 -> mixed, counts for NEITHER
        #   [(1,0)..(4,0)]: own=0 opp=2 -> clean for opp -> axis_opp=2
        #   [(2,0)..(5,0)]: own=0 opp=1 -> axis_opp stays 2
        #   [(3,0)..(6,0)]: own=0 opp=0 -> clean but empty
        #   => axis_own=0, axis_opp=2
        # Axis (0,1): cells (3,-3)..(3,3) are all empty => 0 / 0
        # Axis (1,-1): cells (0,3),(1,2),(2,1),(3,0),(4,-1),(5,-2),(6,-3)
        #   are all empty => 0 / 0
        #
        # own_max = 0, opp_max = 2
        # own_threat_axes: no axis with axis_own >= wl-2 = 2 -> 0
        # opp_threat_axes: axis (1,0) has axis_opp = 2 >= 2 -> 1
        # Normalised: [0/4, 2/4, 0/3, 1/3] (exact in f32).
        i = _node_row(d, 3, 0)
        expected = torch.tensor([0.0, 2.0 / 4.0, 0.0, 1.0 / 3.0])
        torch.testing.assert_close(d.x[i, 8:12], expected, rtol=0.0, atol=0.0)

        # The P2 STONE node (1,0) happens to derive to the same values:
        # Axis (1,0): cells (-2,0)..(4,0); windows containing (1,0):
        #   [(-2,0)..(1,0)]: own=1 opp=1 -> mixed
        #   [(-1,0)..(2,0)]: own=1 opp=2 -> mixed
        #   [( 0,0)..(3,0)]: own=1 opp=2 -> mixed
        #   [( 1,0)..(4,0)]: own=0 opp=2 -> axis_opp=2
        #   => axis_own=0, axis_opp=2
        # Axes (0,1) and (1,-1): only stone in range is (1,0)=P2 itself,
        # and every window contains it => axis_own=0 (blocked), axis_opp=1.
        # own_max=0, opp_max=2, own_axes=0, opp_axes=1 (only axis (1,0)
        # reaches 2 >= wl-2; the other axes hold 1 < 2).
        j = _node_row(d, 1, 0)
        torch.testing.assert_close(d.x[j, 8:12], expected, rtol=0.0, atol=0.0)


class TestAugmentPreservesThreatDims:
    def test_random_augment_carries_threat_dims(self, small_game, monkeypatch):
        # Threat values are per-node invariants under D6: random_augment must
        # carry dims 8-11 along with each node's permuted row (it only
        # recomputes the centroid-relative coord dims 5-6).
        import random

        from hexo_a0.graph import random_augment

        d = game_to_axis_graph(small_game, threat_features=True)
        n_legal = int(d.legal_mask.sum())
        policy = torch.full((n_legal,), 1.0 / n_legal)

        # Force rot180 (_transforms[2]: (q, r) -> (-q, -r)) so the node
        # mapping is known exactly.
        monkeypatch.setattr(random, "choice", lambda seq: seq[2])
        aug, _ = random_augment(d, policy)

        assert aug.x.shape[1] == 12
        # Real nodes exclude the trailing axis-graph dummy node (last row in
        # both graphs; its (0,0) coords would collide with the origin stone).
        orig_coords = d.coords[:-1]
        aug_coords = aug.coords[:-1]
        aug_x = aug.x[:-1]
        for i in range(orig_coords.shape[0]):
            q, r = int(orig_coords[i, 0]), int(orig_coords[i, 1])
            match = (aug_coords[:, 0] == -q) & (aug_coords[:, 1] == -r)
            idx = torch.nonzero(match, as_tuple=False).flatten()
            assert idx.numel() == 1, f"node ({q},{r}) not mapped 1:1 under rot180"
            torch.testing.assert_close(
                aug_x[int(idx.item()), 8:12], d.x[i, 8:12], rtol=0.0, atol=0.0
            )


class TestInputProjGraft:
    def test_graft_preserves_forward_bit_identical(self, small_game):
        from hexo_a0.model import graft_input_proj_for_threat_features
        m8 = HeXONet(_model_cfg(threat=False))
        m8.eval()
        sd = {k: v.clone() for k, v in m8.state_dict().items()}
        assert graft_input_proj_for_threat_features(sd) is True
        m12 = HeXONet(_model_cfg(threat=True))
        m12.load_state_dict(sd)
        m12.eval()
        d8 = game_to_axis_graph(small_game)
        d12 = game_to_axis_graph(small_game, threat_features=True)
        with torch.no_grad():
            l8, v8 = m8.forward_batch(Batch.from_data_list([d8]))
            l12, v12 = m12.forward_batch(Batch.from_data_list([d12]))
        torch.testing.assert_close(l8[0], l12[0], rtol=0, atol=0)
        torch.testing.assert_close(v8, v12, rtol=0, atol=0)

    def test_graft_noop_on_already_12dim(self):
        from hexo_a0.model import graft_input_proj_for_threat_features
        m12 = HeXONet(_model_cfg(threat=True))
        sd = m12.state_dict()
        assert graft_input_proj_for_threat_features(sd) is False

    def test_graft_raises_on_unexpected_width(self):
        from hexo_a0.model import graft_input_proj_for_threat_features
        m8 = HeXONet(_model_cfg(threat=False))
        sd = {k: v.clone() for k, v in m8.state_dict().items()}
        w = sd["representation.input_proj.weight"]
        sd["representation.input_proj.weight"] = torch.zeros(w.shape[0], 10)
        with pytest.raises(ValueError, match="expected 8"):
            graft_input_proj_for_threat_features(sd)

    def test_graft_raises_on_missing_bias_key(self):
        from hexo_a0.model import graft_input_proj_for_threat_features
        m8 = HeXONet(_model_cfg(threat=False))
        sd = {k: v.clone() for k, v in m8.state_dict().items()}
        del sd["representation.input_proj.bias"]
        with pytest.raises(ValueError, match="missing from checkpoint"):
            graft_input_proj_for_threat_features(sd)

    def _make_trainer(self, threat: bool, graft: bool):
        """Minimal CPU trainer (mirrors tests/test_reanalyze.py recipe)."""
        from hexo_a0.config import (
            EvalConfig,
            FullConfig,
            MCTSConfig,
            PythonSelfPlayConfig,
            RunConfig,
            SelfPlayConfig,
            TrainingConfig,
        )
        from hexo_a0.trainer import Trainer

        model_cfg = _model_cfg(threat=threat)
        training_cfg = TrainingConfig(
            batch_size=4, buffer_capacity=100,
            edge_budget=0, grad_accumulation=False,
            lr=2e-4, weight_decay=1e-4, max_grad_norm=1.0,
            lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0,
            step_delay=0.0, target_reuse=0.0, draw_value=0.0,
            augment_symmetries=False, reanalyze_fraction=0.0,
            pc_loss_weight=0.0, graft_threat_features=graft,
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
        return Trainer(
            model=net,
            config=full_cfg,
            game_config=hexo_rs.GameConfig(
                win_length=4, placement_radius=4, max_moves=20
            ),
            device=torch.device("cpu"),
            log_dir=None,
            ckpt_dir=None,
        )

    def test_trainer_load_checkpoint_grafts_input_proj(self, tmp_path):
        t8 = self._make_trainer(threat=False, graft=False)
        ckpt = tmp_path / "ckpt_8dim.pt"
        t8.save_checkpoint(ckpt)

        t12 = self._make_trainer(threat=True, graft=True)
        t12.load_checkpoint(ckpt)  # must not raise (incl. optimizer routing)

        w = t12.model.representation.input_proj.weight
        assert w.shape[1] == 12
        torch.testing.assert_close(
            w[:, :8],
            t8.model.representation.input_proj.weight,
            rtol=0.0, atol=0.0,
        )
        assert torch.all(w[:, 8:12] == 0)


class TestModelForward12:
    def test_forward_on_12dim_graph(self, small_game):
        model = HeXONet(_model_cfg(threat=True))
        model.eval()
        d = game_to_axis_graph(small_game, threat_features=True)
        batch = Batch.from_data_list([d])
        with torch.no_grad():
            logits_list, values = model.forward_batch(batch)
        assert len(logits_list) == 1
        assert logits_list[0].shape[0] == int(d.legal_mask.sum())
        assert torch.isfinite(values).all()

    def test_default_model_stays_8dim(self, small_game):
        model = HeXONet(_model_cfg(threat=False))
        model.eval()
        d = game_to_axis_graph(small_game)
        batch = Batch.from_data_list([d])
        with torch.no_grad():
            logits_list, values = model.forward_batch(batch)
        assert logits_list[0].shape[0] == int(d.legal_mask.sum())
        assert torch.isfinite(values).all()
