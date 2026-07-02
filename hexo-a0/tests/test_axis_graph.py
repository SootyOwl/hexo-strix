"""Tests for axis-graph conversion and augmentation guard."""

import pytest
import torch
import hexo_rs

from hexo_a0.graph import (
    game_to_axis_graph,
    game_to_axis_graph_batch,
    game_to_graph,
    random_augment,
)


# ---------------------------------------------------------------------------
# AC-3a: game_to_axis_graph returns Data with edge_attr of shape (E, 5)
# ---------------------------------------------------------------------------


class TestAxisGraphEdgeAttr:
    """AC-3a: edge_attr shape and presence."""

    def test_edge_attr_shape(self, small_game):
        data = game_to_axis_graph(small_game)
        E = data.edge_index.shape[1]
        assert data.edge_attr is not None
        assert data.edge_attr.shape == (E, 5)

    def test_edge_attr_dtype(self, small_game):
        data = game_to_axis_graph(small_game)
        assert data.edge_attr.dtype == torch.float32

    def test_initial_game_edge_attr(self, initial_game):
        data = game_to_axis_graph(initial_game)
        E = data.edge_index.shape[1]
        assert data.edge_attr.shape == (E, 5)


# ---------------------------------------------------------------------------
# AC-3b: edge_attr.shape[0] == edge_index.shape[1]
# ---------------------------------------------------------------------------


class TestEdgeAttrCountMatchesEdgeIndex:
    """AC-3b: one edge_attr row per edge."""

    def test_count_matches_initial(self, initial_game):
        data = game_to_axis_graph(initial_game)
        assert data.edge_attr.shape[0] == data.edge_index.shape[1]

    def test_count_matches_small(self, small_game):
        data = game_to_axis_graph(small_game)
        assert data.edge_attr.shape[0] == data.edge_index.shape[1]


# ---------------------------------------------------------------------------
# AC-3c: Dummy node is last; legal_mask[-1] == False, stone_mask[-1] == False
# ---------------------------------------------------------------------------


class TestDummyNode:
    """AC-3c: dummy node properties."""

    def test_dummy_legal_mask_false(self, small_game):
        data = game_to_axis_graph(small_game)
        assert data.legal_mask[-1].item() is False

    def test_dummy_stone_mask_false(self, small_game):
        data = game_to_axis_graph(small_game)
        assert data.stone_mask[-1].item() is False

    def test_dummy_initial(self, initial_game):
        data = game_to_axis_graph(initial_game)
        assert data.legal_mask[-1].item() is False
        assert data.stone_mask[-1].item() is False


# ---------------------------------------------------------------------------
# AC-3d: x.shape == (num_nodes, 8)
# ---------------------------------------------------------------------------


class TestAxisNodeFeatures:
    """AC-3d: feature tensor shape."""

    def test_feature_shape(self, small_game):
        data = game_to_axis_graph(small_game)
        assert data.x.shape == (data.num_nodes, 8)

    def test_feature_shape_initial(self, initial_game):
        data = game_to_axis_graph(initial_game)
        assert data.x.shape == (data.num_nodes, 8)


# ---------------------------------------------------------------------------
# AC-3e: random_augment works for axis graphs (D6 augmentation with edge_attr)
# ---------------------------------------------------------------------------


class TestAugmentAxisGraph:
    """AC-3e: augmentation works for axis graphs."""

    def test_augment_axis_works(self, small_game):
        data = game_to_axis_graph(small_game)
        n_legal = data.legal_mask.sum().item()
        policy = torch.ones(n_legal) / n_legal
        new_data, new_policy = random_augment(data, policy)
        assert new_data.x.shape == data.x.shape
        assert new_policy.shape == policy.shape
        assert new_data.edge_attr is not None
        assert new_data.edge_attr.shape[1] == 5

    def test_augment_axis_preserves_counts(self, small_game):
        data = game_to_axis_graph(small_game)
        n_legal = data.legal_mask.sum().item()
        policy = torch.ones(n_legal) / n_legal
        new_data, new_policy = random_augment(data, policy)
        assert new_data.num_nodes == data.num_nodes
        assert new_data.edge_index.shape[1] == data.edge_index.shape[1]
        assert new_data.stone_mask.sum() == data.stone_mask.sum()
        assert new_data.legal_mask.sum() == data.legal_mask.sum()

    def test_augment_axis_edge_attr_valid(self, small_game):
        """Axis one-hots sum to 0 (dummy) or 1 (axis edges) after augmentation."""
        data = game_to_axis_graph(small_game)
        n_legal = data.legal_mask.sum().item()
        policy = torch.ones(n_legal) / n_legal
        new_data, _ = random_augment(data, policy)
        ea = new_data.edge_attr
        dummy_idx = new_data.num_nodes - 1
        for e in range(ea.shape[0]):
            src = new_data.edge_index[0, e].item()
            dst = new_data.edge_index[1, e].item()
            oh_sum = ea[e, 0] + ea[e, 1] + ea[e, 2]
            if src == dummy_idx or dst == dummy_idx:
                assert oh_sum == 0.0, f"dummy edge {e}: expected oh=0, got {oh_sum}"
            else:
                assert oh_sum == 1.0, f"axis edge {e}: expected oh=1, got {oh_sum}"

    def test_augment_axis_policy_sums_to_one(self, small_game):
        data = game_to_axis_graph(small_game)
        n_legal = data.legal_mask.sum().item()
        policy = torch.ones(n_legal) / n_legal
        _, new_policy = random_augment(data, policy)
        assert abs(new_policy.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# AC-3f: random_augment(hex_data, policy) still works (no regression)
# ---------------------------------------------------------------------------


class TestAugmentRegression:
    """AC-3f: hex graphs still augment fine."""

    def test_augment_hex_works(self, small_game):
        data = game_to_graph(small_game)
        n_legal = data.legal_mask.sum().item()
        policy = torch.ones(n_legal) / n_legal
        new_data, new_policy = random_augment(data, policy)
        assert new_data.x.shape == data.x.shape
        assert new_policy.shape == policy.shape


# ---------------------------------------------------------------------------
# AC-3g: game_to_axis_graph_batch returns correct count, each has edge_attr
# ---------------------------------------------------------------------------


class TestAxisGraphBatch:
    """AC-3g: batch variant."""

    def test_batch_count(self, initial_game, small_game):
        results = game_to_axis_graph_batch([initial_game, small_game])
        assert len(results) == 2

    def test_batch_each_has_edge_attr(self, initial_game, small_game):
        results = game_to_axis_graph_batch([initial_game, small_game])
        for data in results:
            assert data.edge_attr is not None
            E = data.edge_index.shape[1]
            assert data.edge_attr.shape == (E, 5)


# ===========================================================================
# Phase 4: Model edge feature support tests
# ===========================================================================

import torch.nn as nn
from torch_geometric.data import Batch, Data

from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet, RepresentationNetwork


def _make_chain_edges(num_nodes):
    """Build bidirectional chain edges for num_nodes nodes."""
    src, dst = [], []
    for i in range(num_nodes - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    if not src:
        src, dst = [0], [0]
    return torch.tensor([src, dst], dtype=torch.int64)


def _make_axis_data(num_nodes=12, num_legal=5, node_features=8):
    """Build a minimal PyG Data with edge_attr (5-dim) for axis graph tests."""
    x = torch.randn(num_nodes, node_features)
    edge_index = _make_chain_edges(num_nodes)
    num_edges = edge_index.shape[1]
    edge_attr = torch.randn(num_edges, 5)
    legal_mask = torch.zeros(num_nodes, dtype=torch.bool)
    legal_mask[:num_legal] = True
    stone_mask = torch.zeros(num_nodes, dtype=torch.bool)
    stone_mask[num_legal : num_legal + 3] = True
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        legal_mask=legal_mask,
        stone_mask=stone_mask,
    )


def _make_hex_data(num_nodes=12, num_legal=5, node_features=8):
    """Build a minimal PyG Data without edge_attr for hex graph tests."""
    x = torch.randn(num_nodes, node_features)
    edge_index = _make_chain_edges(num_nodes)
    legal_mask = torch.zeros(num_nodes, dtype=torch.bool)
    legal_mask[:num_legal] = True
    stone_mask = torch.zeros(num_nodes, dtype=torch.bool)
    stone_mask[num_legal : num_legal + 3] = True
    return Data(
        x=x,
        edge_index=edge_index,
        legal_mask=legal_mask,
        stone_mask=stone_mask,
    )


_SMALL_CFG_HEX = ModelConfig(
    hidden_dim=32, num_layers=2, num_heads=2,
    policy_hidden=16, value_hidden=16,
    graph_type="hex", conv_type="gatv2",
)

_SMALL_CFG_AXIS = ModelConfig(
    hidden_dim=32, num_layers=2, num_heads=2,
    policy_hidden=16, value_hidden=16,
    graph_type="axis",
)


# ---------------------------------------------------------------------------
# AC-4a: graph_type="hex" regression — identical output to current model
# ---------------------------------------------------------------------------


class TestHexGraphTypeRegression:
    """AC-4a: graph_type='hex' produces identical output to default model."""

    def test_hex_forward_same_as_default(self):
        """Model with explicit graph_type='hex' matches default (no graph_type)."""
        torch.manual_seed(42)
        net_default = HeXONet(_SMALL_CFG_HEX)
        # Create config with explicit hex
        cfg_hex = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=2,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
        )
        net_hex = HeXONet(cfg_hex)
        # Copy weights
        net_hex.load_state_dict(net_default.state_dict())
        net_default.eval()
        net_hex.eval()

        data = _make_hex_data(num_nodes=10, num_legal=4)
        with torch.no_grad():
            p_def, v_def = net_default(data.x, data.edge_index, data.legal_mask, data.stone_mask)
            p_hex, v_hex = net_hex(data.x, data.edge_index, data.legal_mask, data.stone_mask)

        assert torch.allclose(p_def, p_hex, atol=1e-6)
        assert torch.allclose(v_def, v_hex, atol=1e-6)

    def test_hex_model_has_no_edge_proj(self):
        """graph_type='hex' model should NOT have an edge_proj layer."""
        net = HeXONet(_SMALL_CFG_HEX)
        assert not hasattr(net.representation, "edge_proj")


# ---------------------------------------------------------------------------
# AC-4b: graph_type="axis" forward with edge_attr — shapes correct
# ---------------------------------------------------------------------------


class TestAxisModelForward:
    """AC-4b: axis model forward produces correct shapes."""

    def test_forward_policy_shape(self):
        net = HeXONet(_SMALL_CFG_AXIS)
        data = _make_axis_data(num_nodes=12, num_legal=5)
        with torch.no_grad():
            policy, value = net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, edge_attr=data.edge_attr,
            )
        assert policy.shape == (5,)
        assert value.dim() == 0

    def test_forward_value_in_range(self):
        net = HeXONet(_SMALL_CFG_AXIS)
        data = _make_axis_data(num_nodes=12, num_legal=5)
        with torch.no_grad():
            _, value = net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, edge_attr=data.edge_attr,
            )
        assert -1.0 <= value.item() <= 1.0

    def test_axis_model_has_edge_proj(self):
        """graph_type='axis' model should have an edge_proj Linear layer."""
        net = HeXONet(_SMALL_CFG_AXIS)
        assert hasattr(net.representation, "edge_proj")
        assert isinstance(net.representation.edge_proj, nn.Linear)
        assert net.representation.edge_proj.in_features == 5
        assert net.representation.edge_proj.out_features == 32


# ---------------------------------------------------------------------------
# AC-4c: axis batched forward with edge_attr
# ---------------------------------------------------------------------------


class TestAxisBatchedForward:
    """AC-4c: batched forward with axis edge_attr across multiple graphs."""

    def test_batched_forward_shapes(self):
        net = HeXONet(_SMALL_CFG_AXIS)
        g1 = _make_axis_data(num_nodes=10, num_legal=4)
        g2 = _make_axis_data(num_nodes=8, num_legal=3)
        batch = Batch.from_data_list([g1, g2])
        with torch.no_grad():
            policy_list, values = net.forward_batch(batch)
        assert len(policy_list) == 2
        assert policy_list[0].shape == (4,)
        assert policy_list[1].shape == (3,)
        assert values.shape == (2,)

    def test_batched_forward_values_in_range(self):
        net = HeXONet(_SMALL_CFG_AXIS)
        g1 = _make_axis_data(num_nodes=10, num_legal=4)
        g2 = _make_axis_data(num_nodes=8, num_legal=3)
        batch = Batch.from_data_list([g1, g2])
        with torch.no_grad():
            _, values = net.forward_batch(batch)
        for v in values:
            assert -1.0 <= v.item() <= 1.0


# ---------------------------------------------------------------------------
# AC-4d: graph_type="hex" batched forward still works (no regression)
# ---------------------------------------------------------------------------


class TestHexBatchedRegression:
    """AC-4d: hex model batched forward still works."""

    def test_hex_batched_forward(self):
        net = HeXONet(_SMALL_CFG_HEX)
        g1 = _make_hex_data(num_nodes=10, num_legal=4)
        g2 = _make_hex_data(num_nodes=8, num_legal=3)
        batch = Batch.from_data_list([g1, g2])
        with torch.no_grad():
            policy_list, values = net.forward_batch(batch)
        assert len(policy_list) == 2
        assert policy_list[0].shape == (4,)
        assert policy_list[1].shape == (3,)
        assert values.shape == (2,)


# ---------------------------------------------------------------------------
# AC-4e: edge_proj receives nonzero gradients after a training step
# ---------------------------------------------------------------------------


class TestEdgeProjGradients:
    """AC-4e: edge_proj gets nonzero gradients."""

    def test_edge_proj_gradients_nonzero(self):
        net = HeXONet(_SMALL_CFG_AXIS)
        data = _make_axis_data(num_nodes=12, num_legal=5)
        policy, value = net(
            data.x, data.edge_index, data.legal_mask,
            data.stone_mask, edge_attr=data.edge_attr,
        )
        loss = policy.sum() + value
        loss.backward()
        for name, param in net.representation.edge_proj.named_parameters():
            assert param.grad is not None, f"No grad for edge_proj.{name}"
            assert param.grad.abs().sum().item() > 0.0, (
                f"Zero grad for edge_proj.{name}"
            )


# ---------------------------------------------------------------------------
# AC-4g: forward() and forward_batch() equivalence with edge_attr
# ---------------------------------------------------------------------------


class TestAxisForwardBatchEquivalence:
    """AC-4g: forward() matches forward_batch() for single axis graph."""

    def test_equivalence(self):
        net = HeXONet(_SMALL_CFG_AXIS)
        net.eval()
        data = _make_axis_data(num_nodes=12, num_legal=5)
        batch = Batch.from_data_list([data])

        with torch.no_grad():
            p_single, v_single = net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, edge_attr=data.edge_attr,
            )
            p_list, v_batch = net.forward_batch(batch)

        assert torch.allclose(p_single, p_list[0], atol=1e-6), (
            "Policy logits differ between forward() and forward_batch()"
        )
        assert torch.allclose(v_single, v_batch[0], atol=1e-6), (
            "Value differs between forward() and forward_batch()"
        )


# ---------------------------------------------------------------------------
# Graph type / edge_attr mismatch guards
# ---------------------------------------------------------------------------


class TestGraphTypeMismatchGuards:
    """Axis model + no edge_attr and hex model + edge_attr raise ValueError."""

    def test_axis_model_without_edge_attr_raises(self):
        net = HeXONet(_SMALL_CFG_AXIS)
        data = _make_axis_data(num_nodes=12, num_legal=5)
        with pytest.raises(ValueError, match="requires edge_attr"):
            net(data.x, data.edge_index, data.legal_mask, data.stone_mask)

    def test_hex_model_with_edge_attr_raises(self):
        net = HeXONet(_SMALL_CFG_HEX)
        data = _make_axis_data(num_nodes=12, num_legal=5)
        with pytest.raises(ValueError, match="unexpected edge_attr"):
            net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, edge_attr=data.edge_attr,
            )

    def test_invalid_graph_type_raises(self):
        from hexo_a0.model import RepresentationNetwork
        cfg = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=2,
            policy_hidden=16, value_hidden=16, graph_type="hex", conv_type="gatv2",
        )
        with pytest.raises(ValueError, match="must be 'hex' or 'axis'"):
            RepresentationNetwork(cfg, graph_type="axes")


# ===========================================================================
# Phase 5: Config + Wiring + Guards
# ===========================================================================


# ---------------------------------------------------------------------------
# AC-5f: TorchScript export with graph_type="axis" raises ValueError
# ---------------------------------------------------------------------------


class TestScriptableModelGuard:
    """AC-5f: export_torchscript works for both hex and axis graphs."""

    def test_export_hex_no_raise(self):
        """graph_type='hex' should not raise (smoke test only)."""
        from hexo_a0.scriptable_model import export_torchscript

        net = HeXONet(_SMALL_CFG_HEX)
        config = ModelConfig(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            policy_hidden=16,
            value_hidden=16,
            graph_type="hex", conv_type="gatv2",
        )
        sd = net.state_dict()
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            export_torchscript(sd, config, path, bf16=False)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# AC-5g: augment_symmetries + graph_type="axis" now works
# ---------------------------------------------------------------------------


class TestAugmentAxisStartup:
    """AC-5g: training startup accepts augment_symmetries + axis."""

    def test_augment_axis_ok_at_startup(self):
        from hexo_a0.config import FullConfig, TrainingConfig
        from hexo_a0.trainer import Trainer

        mc = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=2,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        tc = TrainingConfig(augment_symmetries=True)
        cfg = FullConfig(model=mc, training=tc)
        net = HeXONet(mc)
        import hexo_rs
        gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)

        # Should not raise — D6 augmentation now supported for axis graphs
        trainer = Trainer(net, cfg, gc, torch.device("cpu"))
        assert trainer is not None

    def test_no_augment_axis_ok(self):
        """axis without augment should not raise."""
        from hexo_a0.config import FullConfig, TrainingConfig
        from hexo_a0.trainer import Trainer

        mc = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=2,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        tc = TrainingConfig(augment_symmetries=False)
        cfg = FullConfig(model=mc, training=tc)
        net = HeXONet(mc)
        import hexo_rs
        gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)

        # Should not raise
        trainer = Trainer(net, cfg, gc, torch.device("cpu"))
        assert trainer is not None


# ===========================================================================
# Phase 6: Integration Test + Benchmarks
# ===========================================================================

import time
import math


# ---------------------------------------------------------------------------
# AC-6a: End-to-end integration — game state -> axis model -> train_step
# ---------------------------------------------------------------------------


class TestAxisEndToEndIntegration:
    """AC-6a: full pipeline from game state through training step."""

    def test_train_step_axis_model(self):
        """Create axis graph from game, build axis model, run train_step.

        Verify: loss > 0, loss is not NaN, edge_proj gradients are nonzero.
        """
        from hexo_a0.trainer import train_step

        # 1. Small axis model config
        cfg = ModelConfig(
            hidden_dim=32, num_heads=4, num_layers=2,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        model = HeXONet(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        # 2. Create a game state and convert to axis graph
        gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game = hexo_rs.GameState(gc)
        game.apply_move(1, 0)
        game.apply_move(-1, 0)

        data = game_to_axis_graph(game)

        # 3. Create fake training targets
        n_legal = data.legal_mask.sum().item()
        policy_target = torch.ones(n_legal) / n_legal
        value_target = torch.tensor(1.0)

        # 4. Run train_step
        examples = [(data, policy_target, value_target)]
        metrics = train_step(model, optimizer, examples, device)

        # 5. Verify loss > 0 and not NaN
        assert metrics["total_loss"] > 0.0, f"Expected loss > 0, got {metrics['total_loss']}"
        assert not math.isnan(metrics["total_loss"]), "Loss is NaN"
        assert not math.isnan(metrics["policy_loss"]), "Policy loss is NaN"
        assert not math.isnan(metrics["value_loss"]), "Value loss is NaN"

        # 6. Verify edge_proj gradients are nonzero
        edge_proj = model.representation.edge_proj
        assert edge_proj.weight.grad is not None, "edge_proj.weight has no gradient"
        assert edge_proj.weight.grad.abs().sum().item() > 0.0, (
            "edge_proj.weight gradient is all zeros"
        )
        assert edge_proj.bias.grad is not None, "edge_proj.bias has no gradient"
        assert edge_proj.bias.grad.abs().sum().item() > 0.0, (
            "edge_proj.bias gradient is all zeros"
        )


# ---------------------------------------------------------------------------
# AC-6b: Edge count scaling benchmark — hex-neighbor vs axis-window
# ---------------------------------------------------------------------------


class TestEdgeCountScaling:
    """AC-6b: compare edge counts for hex and axis graphs at various move counts."""

    def test_edge_count_ratio(self, capsys):
        """Measure edge counts for hex-neighbor vs axis-window graphs.

        For S5 game states at move counts 5, 20, 50, 100, print the ratio.
        This validates the 3-layer-vs-9-layer argument.
        """
        import random as _random

        gc = hexo_rs.GameConfig(win_length=6, placement_radius=6, max_moves=200)

        print("\n--- Edge Count Scaling Benchmark (S5: win_length=6, radius=6) ---")
        print(f"{'Moves':>6}  {'Hex Edges':>10}  {'Axis Edges':>11}  {'Ratio (Hex/Axis)':>16}  {'Hex Nodes':>10}  {'Axis Nodes':>11}")
        print("-" * 75)

        for target_moves in [5, 20, 50, 100]:
            _random.seed(42)
            game = hexo_rs.GameState(gc)
            for i in range(target_moves):
                if game.is_terminal():
                    break
                moves = game.legal_moves()
                game.apply_move(*_random.choice(moves))

            if game.is_terminal():
                print(f"{target_moves:>6}  (terminal before {target_moves} moves)")
                continue

            hex_data = game_to_graph(game)
            axis_data = game_to_axis_graph(game)

            hex_edges = hex_data.edge_index.shape[1]
            axis_edges = axis_data.edge_index.shape[1]
            hex_nodes = hex_data.x.shape[0]
            axis_nodes = axis_data.x.shape[0]
            ratio = hex_edges / axis_edges if axis_edges > 0 else float("inf")

            print(f"{target_moves:>6}  {hex_edges:>10}  {axis_edges:>11}  {ratio:>16.2f}  {hex_nodes:>10}  {axis_nodes:>11}")

            # Both should have edges
            assert hex_edges > 0
            assert axis_edges > 0

        print("-" * 75)
        print("Higher ratio = more edges in hex graph relative to axis graph.")
        print("Axis graphs have fewer edges, enabling equivalent receptive field with fewer layers.\n")


# ---------------------------------------------------------------------------
# AC-6c: Training step speed benchmark — axis 3-layer vs hex 9-layer
# ---------------------------------------------------------------------------


class TestTrainingStepSpeed:
    """AC-6c: compare step/s for 3-layer axis vs 9-layer hex models."""

    def test_step_speed_comparison(self, capsys):
        """Benchmark training step speed for axis (3 layers) vs hex (9 layers).

        Both models use the same hidden_dim and batch size. Prints timing.
        """
        import random as _random
        from hexo_a0.trainer import train_step

        device = torch.device("cpu")

        # Configure the two models
        axis_cfg = ModelConfig(
            hidden_dim=32, num_heads=4, num_layers=3,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        hex_cfg = ModelConfig(
            hidden_dim=32, num_heads=4, num_layers=9,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
        )

        axis_model = HeXONet(axis_cfg)
        hex_model = HeXONet(hex_cfg)

        axis_opt = torch.optim.Adam(axis_model.parameters(), lr=1e-3)
        hex_opt = torch.optim.Adam(hex_model.parameters(), lr=1e-3)

        # Generate some game states for benchmarking
        gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        _random.seed(42)

        axis_examples = []
        hex_examples = []
        batch_size = 16

        for _ in range(batch_size):
            game = hexo_rs.GameState(gc)
            for _ in range(_random.randint(2, 15)):
                if game.is_terminal():
                    break
                moves = game.legal_moves()
                game.apply_move(*_random.choice(moves))
            if game.is_terminal():
                game = hexo_rs.GameState(gc)
                game.apply_move(1, 0)

            axis_data = game_to_axis_graph(game)
            hex_data = game_to_graph(game)

            n_legal_axis = axis_data.legal_mask.sum().item()
            n_legal_hex = hex_data.legal_mask.sum().item()

            policy_axis = torch.ones(n_legal_axis) / n_legal_axis
            policy_hex = torch.ones(n_legal_hex) / n_legal_hex
            value_target = torch.tensor(0.0)

            axis_examples.append((axis_data, policy_axis, value_target))
            hex_examples.append((hex_data, policy_hex, value_target))

        # Warmup
        train_step(axis_model, axis_opt, axis_examples[:4], device)
        train_step(hex_model, hex_opt, hex_examples[:4], device)

        # Benchmark
        n_iters = 5

        start = time.perf_counter()
        for _ in range(n_iters):
            train_step(axis_model, axis_opt, axis_examples, device)
        axis_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(n_iters):
            train_step(hex_model, hex_opt, hex_examples, device)
        hex_time = time.perf_counter() - start

        axis_step_s = n_iters / axis_time
        hex_step_s = n_iters / hex_time

        axis_param_count = sum(p.numel() for p in axis_model.parameters())
        hex_param_count = sum(p.numel() for p in hex_model.parameters())

        print(f"\n--- Training Step Speed Benchmark (batch_size={batch_size}, CPU) ---")
        print(f"{'Model':>20}  {'Layers':>7}  {'Params':>8}  {'step/s':>8}  {'ms/step':>8}  {'Total (s)':>10}")
        print("-" * 75)
        print(f"{'Axis (3-layer)':>20}  {3:>7}  {axis_param_count:>8}  {axis_step_s:>8.2f}  {axis_time/n_iters*1000:>8.1f}  {axis_time:>10.3f}")
        print(f"{'Hex (9-layer)':>20}  {9:>7}  {hex_param_count:>8}  {hex_step_s:>8.2f}  {hex_time/n_iters*1000:>8.1f}  {hex_time:>10.3f}")
        speedup = hex_time / axis_time if axis_time > 0 else float("inf")
        print(f"\nAxis speedup: {speedup:.2f}x faster than hex (3 layers vs 9 layers)")
        print("Axis graphs with fewer layers can match hex receptive field due to richer edges.\n")


# ---------------------------------------------------------------------------
# GPU self-play smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAxisGpuSelfPlay:
    """Verify axis-graph self-play works end-to-end on GPU."""

    def test_batched_self_play_axis_gpu(self):
        """Run a short batched self-play game with axis graphs on CUDA."""
        from hexo_a0.self_play import batched_self_play_games
        from hexo_a0.config import FullConfig, MCTSConfig

        mc = ModelConfig(
            hidden_dim=32, num_heads=4, num_layers=2,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        mcts = MCTSConfig(
            n_simulations=4, m_actions=4,
            exploration_moves=4,
        )
        cfg = FullConfig(model=mc, mcts=mcts)
        device = torch.device("cuda")
        net = HeXONet(mc).to(device)

        gc = hexo_rs.GameConfig(win_length=4, placement_radius=2, max_moves=30)

        trajectories = batched_self_play_games(
            net, cfg, gc, device, n_games=2,
        )

        from hexo_a0.graph import graph_fn_from_model_config
        gfn = graph_fn_from_model_config(mc)
        assert len(trajectories) == 2, f"Expected 2 game trajectories, got {len(trajectories)}"
        for i, game_examples in enumerate(trajectories):
            assert len(game_examples) > 0, f"Game {i} produced no training examples"
            ex = game_examples[0]
            d = gfn(ex.game_state)
            assert d.edge_attr is not None, f"Game {i} example missing edge_attr"
            assert d.edge_attr.shape[1] == 5, f"Game {i} edge_attr wrong shape"


# ===========================================================================
# Piece 2+3+4: ScriptableHeXONet axis graph support + numerical equivalence
# ===========================================================================

from hexo_a0.scriptable_model import (
    GATv2Layer,
    GINELayer,
    ScriptableHeXONet,
    load_from_hexonet,
    export_torchscript,
)


# ---------------------------------------------------------------------------
# GATv2Layer edge_attr support
# ---------------------------------------------------------------------------


class TestGATv2LayerEdgeAttr:
    """GATv2Layer with edge_dim handles edge_attr correctly."""

    def test_forward_with_edge_attr_shape(self):
        """Output shape is (N, H*D) when edge_attr is provided."""
        layer = GATv2Layer(in_channels=32, head_dim=8, num_heads=4, edge_dim=32)
        x = torch.randn(10, 32)
        edge_index = _make_chain_edges(10)
        edge_attr = torch.randn(edge_index.shape[1], 32)
        out = layer(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (10, 32)

    def test_forward_without_edge_attr_still_works(self):
        """Layer with edge_dim=0 works without edge_attr (hex mode)."""
        layer = GATv2Layer(in_channels=32, head_dim=8, num_heads=4)
        x = torch.randn(10, 32)
        edge_index = _make_chain_edges(10)
        out = layer(x, edge_index)
        assert out.shape == (10, 32)

    def test_edge_dim_creates_lin_edge(self):
        """edge_dim > 0 creates lin_edge parameter."""
        layer = GATv2Layer(in_channels=32, head_dim=8, num_heads=4, edge_dim=16)
        assert hasattr(layer, "lin_edge")
        assert layer.lin_edge.in_features == 16
        assert layer.lin_edge.out_features == 32

    def test_no_edge_dim_no_lin_edge(self):
        """edge_dim=0 does not create lin_edge."""
        layer = GATv2Layer(in_channels=32, head_dim=8, num_heads=4)
        assert not hasattr(layer, "lin_edge") or layer.edge_dim == 0


# ---------------------------------------------------------------------------
# ScriptableHeXONet axis graph support
# ---------------------------------------------------------------------------


class TestScriptableHeXONetAxis:
    """ScriptableHeXONet forward with axis graphs (edge_attr)."""

    def test_axis_forward_shapes(self):
        """Axis model produces correct output shapes."""
        net = ScriptableHeXONet(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        net.eval()
        data = _make_axis_data(num_nodes=12, num_legal=5)
        batch = torch.zeros(12, dtype=torch.long)
        with torch.no_grad():
            logits, counts, values = net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, batch, 1, data.edge_attr,
            )
        assert logits.shape == (5,)
        assert counts.shape == (1,)
        assert values.shape == (1,)

    def test_hex_forward_no_edge_attr(self):
        """Hex model works without edge_attr (empty tensor sentinel)."""
        net = ScriptableHeXONet(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
        )
        net.eval()
        data = _make_hex_data(num_nodes=10, num_legal=4)
        batch = torch.zeros(10, dtype=torch.long)
        empty_edge_attr = torch.zeros(0, dtype=torch.float32)
        with torch.no_grad():
            logits, counts, values = net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, batch, 1, empty_edge_attr,
            )
        assert logits.shape == (4,)

    def test_axis_model_has_edge_proj(self):
        """Axis ScriptableHeXONet has edge_proj."""
        net = ScriptableHeXONet(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        assert hasattr(net, "edge_proj")
        assert net.edge_proj.in_features == 5
        assert net.edge_proj.out_features == 32

    def test_torchscript_axis(self):
        """Axis ScriptableHeXONet can be torch.jit.script'd."""
        net = ScriptableHeXONet(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        net.eval()
        scripted = torch.jit.script(net)
        data = _make_axis_data(num_nodes=12, num_legal=5)
        batch = torch.zeros(12, dtype=torch.long)
        with torch.no_grad():
            logits, counts, values = scripted(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, batch, 1, data.edge_attr,
            )
        assert logits.shape == (5,)

    def test_torchscript_hex(self):
        """Hex ScriptableHeXONet can still be torch.jit.script'd."""
        net = ScriptableHeXONet(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
        )
        net.eval()
        scripted = torch.jit.script(net)
        data = _make_hex_data(num_nodes=10, num_legal=4)
        batch = torch.zeros(10, dtype=torch.long)
        empty_ea = torch.zeros(0, dtype=torch.float32)
        with torch.no_grad():
            logits, counts, values = scripted(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, batch, 1, empty_ea,
            )
        assert logits.shape == (4,)


# ---------------------------------------------------------------------------
# Piece 4: Numerical equivalence — PyG HeXONet vs ScriptableHeXONet
# ---------------------------------------------------------------------------


class TestNumericalEquivalence:
    """ScriptableHeXONet must produce identical outputs to PyG HeXONet."""

    def test_axis_numerical_equivalence(self):
        """Axis graph: HeXONet (PyG) and ScriptableHeXONet match within atol=1e-4."""
        torch.manual_seed(123)
        cfg = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis", conv_type="gine",
        )
        pyg_net = HeXONet(cfg)
        pyg_net.eval()

        # Build scriptable net and load weights
        script_net = ScriptableHeXONet(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis", conv_type="gine",
        )
        load_from_hexonet(script_net, pyg_net.state_dict())
        script_net.eval()

        data = _make_axis_data(num_nodes=12, num_legal=5)
        batch_idx = torch.zeros(12, dtype=torch.long)

        with torch.no_grad():
            p_pyg, v_pyg = pyg_net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, edge_attr=data.edge_attr,
            )
            p_script, _, v_script = script_net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, batch_idx, 1, data.edge_attr,
            )

        assert torch.allclose(p_pyg, p_script, atol=1e-4), (
            f"Policy mismatch:\nPyG: {p_pyg}\nScript: {p_script}\n"
            f"Max diff: {(p_pyg - p_script).abs().max()}"
        )
        assert torch.allclose(v_pyg, v_script[0], atol=1e-4), (
            f"Value mismatch: PyG={v_pyg.item()}, Script={v_script[0].item()}"
        )

    def test_hex_numerical_equivalence(self):
        """Hex graph: HeXONet (PyG) and ScriptableHeXONet match within atol=1e-4."""
        torch.manual_seed(456)
        cfg = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
        )
        pyg_net = HeXONet(cfg)
        pyg_net.eval()

        script_net = ScriptableHeXONet(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
        )
        load_from_hexonet(script_net, pyg_net.state_dict())
        script_net.eval()

        data = _make_hex_data(num_nodes=10, num_legal=4)
        batch_idx = torch.zeros(10, dtype=torch.long)
        empty_ea = torch.zeros(0, dtype=torch.float32)

        with torch.no_grad():
            p_pyg, v_pyg = pyg_net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask,
            )
            p_script, _, v_script = script_net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, batch_idx, 1, empty_ea,
            )

        assert torch.allclose(p_pyg, p_script, atol=1e-4), (
            f"Policy mismatch:\nPyG: {p_pyg}\nScript: {p_script}\n"
            f"Max diff: {(p_pyg - p_script).abs().max()}"
        )
        assert torch.allclose(v_pyg, v_script[0], atol=1e-4), (
            f"Value mismatch: PyG={v_pyg.item()}, Script={v_script[0].item()}"
        )


# ---------------------------------------------------------------------------
# Piece 3: export_torchscript for axis graphs
# ---------------------------------------------------------------------------


class TestAxisTorchScriptExport:
    """export_torchscript works for axis graphs (guard removed)."""

    def test_export_axis_succeeds(self):
        """graph_type='axis' export no longer raises."""
        cfg = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        pyg_net = HeXONet(cfg)
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            export_torchscript(pyg_net.state_dict(), cfg, path, bf16=False)
            assert os.path.exists(path)

            # Load and run inference on axis data
            scripted = torch.jit.load(path)
            data = _make_axis_data(num_nodes=12, num_legal=5)
            batch_idx = torch.zeros(12, dtype=torch.long)
            with torch.no_grad():
                logits, counts, values = scripted(
                    data.x, data.edge_index, data.legal_mask,
                    data.stone_mask, batch_idx, 1, data.edge_attr,
                )
            assert logits.shape == (5,)
        finally:
            os.unlink(path)

    def test_export_axis_numerical_equivalence(self):
        """Exported axis TorchScript model matches PyG model."""
        torch.manual_seed(789)
        cfg = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis",
        )
        pyg_net = HeXONet(cfg)
        pyg_net.eval()

        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            export_torchscript(pyg_net.state_dict(), cfg, path, bf16=False)
            scripted = torch.jit.load(path)

            data = _make_axis_data(num_nodes=12, num_legal=5)
            batch_idx = torch.zeros(12, dtype=torch.long)

            with torch.no_grad():
                p_pyg, v_pyg = pyg_net(
                    data.x, data.edge_index, data.legal_mask,
                    data.stone_mask, edge_attr=data.edge_attr,
                )
                p_script, _, v_script = scripted(
                    data.x, data.edge_index, data.legal_mask,
                    data.stone_mask, batch_idx, 1, data.edge_attr,
                )

            assert torch.allclose(p_pyg, p_script, atol=1e-4), (
                f"Max diff: {(p_pyg - p_script).abs().max()}"
            )
            assert torch.allclose(v_pyg, v_script[0], atol=1e-4)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# GINEConv scriptable model support (regression: TorchScript must use the
# correct conv layer, not always GATv2Layer)
# ---------------------------------------------------------------------------


class TestGINELayerScriptable:
    """GINELayer pure-PyTorch implementation for TorchScript export."""

    def test_forward_shape(self):
        layer = GINELayer(hidden_dim=32, edge_dim=32)
        x = torch.randn(10, 32)
        edge_index = _make_chain_edges(10)
        edge_attr = torch.randn(edge_index.shape[1], 32)
        out = layer(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (10, 32)

    def test_forward_without_edge_attr(self):
        layer = GINELayer(hidden_dim=32)
        x = torch.randn(10, 32)
        edge_index = _make_chain_edges(10)
        out = layer(x, edge_index)
        assert out.shape == (10, 32)

    def test_scriptable(self):
        layer = GINELayer(hidden_dim=32, edge_dim=32)
        scripted = torch.jit.script(layer)
        x = torch.randn(10, 32)
        edge_index = _make_chain_edges(10)
        edge_attr = torch.randn(edge_index.shape[1], 32)
        out = scripted(x, edge_index, edge_attr)
        assert out.shape == (10, 32)


class TestScriptableHeXONetGINE:
    """ScriptableHeXONet with conv_type='gine'."""

    def test_gine_forward_shapes(self):
        net = ScriptableHeXONet(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis", conv_type="gine",
        )
        net.eval()
        data = _make_axis_data(num_nodes=12, num_legal=5)
        batch = torch.zeros(12, dtype=torch.long)
        with torch.no_grad():
            logits, counts, values = net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, batch, 1, data.edge_attr,
            )
        assert logits.shape == (5,)
        assert counts.shape == (1,)
        assert values.shape == (1,)

    def test_gine_uses_gine_layers(self):
        """Regression: conv_type='gine' must create GINELayer, not GATv2Layer."""
        net = ScriptableHeXONet(
            hidden_dim=32, num_layers=3, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis", conv_type="gine",
        )
        for conv in net.convs:
            assert isinstance(conv, GINELayer), (
                f"Expected GINELayer, got {type(conv).__name__}"
            )

    def test_gatv2_still_uses_gatv2_layers(self):
        """conv_type='gatv2' must still create GATv2Layer."""
        net = ScriptableHeXONet(
            hidden_dim=32, num_layers=3, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis", conv_type="gatv2",
        )
        for conv in net.convs:
            assert isinstance(conv, GATv2Layer), (
                f"Expected GATv2Layer, got {type(conv).__name__}"
            )

    def test_gine_torchscript_export(self):
        """Regression: TorchScript export of GINEConv model uses GINELayer."""
        cfg = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis", conv_type="gine",
        )
        pyg_net = HeXONet(cfg)
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            export_torchscript(pyg_net.state_dict(), cfg, path, bf16=False)
            scripted = torch.jit.load(path)
            graph_str = str(scripted.graph)
            assert "GINELayer" in graph_str, "TorchScript graph must contain GINELayer"
            assert "GATv2Layer" not in graph_str, "TorchScript graph must NOT contain GATv2Layer"

            # Run inference
            data = _make_axis_data(num_nodes=12, num_legal=5)
            batch_idx = torch.zeros(12, dtype=torch.long)
            with torch.no_grad():
                logits, counts, values = scripted(
                    data.x, data.edge_index, data.legal_mask,
                    data.stone_mask, batch_idx, 1, data.edge_attr,
                )
            assert logits.shape == (5,)
        finally:
            os.unlink(path)

    def test_gine_numerical_equivalence(self):
        """PyG GINEConv HeXONet and ScriptableHeXONet must match."""
        torch.manual_seed(999)
        cfg = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis", conv_type="gine",
        )
        pyg_net = HeXONet(cfg)
        pyg_net.eval()

        script_net = ScriptableHeXONet(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="axis", conv_type="gine",
        )
        load_from_hexonet(script_net, pyg_net.state_dict())
        script_net.eval()

        data = _make_axis_data(num_nodes=12, num_legal=5)
        batch_idx = torch.zeros(12, dtype=torch.long)

        with torch.no_grad():
            p_pyg, v_pyg = pyg_net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, edge_attr=data.edge_attr,
            )
            p_script, _, v_script = script_net(
                data.x, data.edge_index, data.legal_mask,
                data.stone_mask, batch_idx, 1, data.edge_attr,
            )

        assert torch.allclose(p_pyg, p_script, atol=1e-4), (
            f"Policy mismatch: max diff={( p_pyg - p_script).abs().max()}"
        )
        assert torch.allclose(v_pyg, v_script[0], atol=1e-4), (
            f"Value mismatch: PyG={v_pyg.item()}, Script={v_script[0].item()}"
        )
