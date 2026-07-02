"""Integration tests for HeXO AlphaZero: end-to-end, batched, overfit, GPU smoke."""

import pytest
import torch
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_graph
from hexo_a0.loss import alphazero_loss
from hexo_a0.model import HeXONet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_CFG = ModelConfig(
    hidden_dim=32,
    num_layers=2,
    num_heads=2,
    policy_hidden=16,
    value_hidden=16,
    graph_type="hex",
    conv_type="gatv2",
)


def _uniform_policy_target(num_legal: int) -> torch.Tensor:
    """Uniform probability distribution over legal moves."""
    return torch.full((num_legal,), 1.0 / num_legal)


def _value_target(val: float = 0.5) -> torch.Tensor:
    return torch.tensor(val)


def _initial_game() -> hexo_rs.GameState:
    return hexo_rs.GameState()


def _game_after_moves(n: int) -> hexo_rs.GameState:
    """Create a GameState with n moves applied (stops if terminal)."""
    game = hexo_rs.GameState()
    for _ in range(n):
        if game.is_terminal():
            break
        moves = game.legal_moves()
        game.apply_move(*moves[0])
    return game


# ---------------------------------------------------------------------------
# AC-14a: End-to-end single game
# ---------------------------------------------------------------------------

class TestEndToEndSingleGame:
    """AC-14a: full pipeline on a single GameState — loss is finite, grads flow."""

    def test_loss_is_finite(self):
        net = HeXONet(_SMALL_CFG)
        game = _initial_game()
        data = game_to_graph(game)

        policy_logits, value = net(data.x, data.edge_index, data.legal_mask)

        num_legal = data.legal_mask.sum().item()
        policy_target = _uniform_policy_target(num_legal)
        value_target = _value_target(0.5)

        loss, _ = alphazero_loss(policy_logits, policy_target, value, value_target)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_loss_backward_reaches_all_parameters(self):
        net = HeXONet(_SMALL_CFG)
        game = _initial_game()
        data = game_to_graph(game)

        policy_logits, value = net(data.x, data.edge_index, data.legal_mask)

        num_legal = data.legal_mask.sum().item()
        policy_target = _uniform_policy_target(num_legal)
        value_target = _value_target(0.5)

        loss, _ = alphazero_loss(policy_logits, policy_target, value, value_target)
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"

    def test_loss_components_are_finite(self):
        net = HeXONet(_SMALL_CFG)
        game = _initial_game()
        data = game_to_graph(game)

        policy_logits, value = net(data.x, data.edge_index, data.legal_mask)
        num_legal = data.legal_mask.sum().item()

        loss, components = alphazero_loss(
            policy_logits,
            _uniform_policy_target(num_legal),
            value,
            _value_target(0.5),
        )

        assert torch.isfinite(components["policy_loss"])
        assert torch.isfinite(components["value_loss"])


# ---------------------------------------------------------------------------
# AC-14a: End-to-end batched
# ---------------------------------------------------------------------------

class TestEndToEndBatched:
    """AC-14a: batched pipeline over multiple GameStates at different stages."""

    def _make_batch_and_targets(self):
        """Build a batch of 3 graphs and their fake targets."""
        games = [
            _initial_game(),
            _game_after_moves(4),
            _game_after_moves(8),
        ]
        graphs = [game_to_graph(g) for g in games]
        batch = Batch.from_data_list(graphs)
        # Uniform policy targets per graph
        targets = [
            (
                _uniform_policy_target(g.legal_mask.sum().item()),
                _value_target(0.5),
            )
            for g in graphs
        ]
        return batch, targets

    def test_batched_loss_is_finite(self):
        net = HeXONet(_SMALL_CFG)
        batch, targets = self._make_batch_and_targets()

        policy_list, values = net.forward_batch(batch)

        total_loss = torch.tensor(0.0)
        for i, (policy_logits, value) in enumerate(zip(policy_list, values)):
            policy_target, value_target = targets[i]
            loss, _ = alphazero_loss(policy_logits, policy_target, value, value_target)
            total_loss = total_loss + loss

        assert torch.isfinite(total_loss), f"Batched loss is not finite: {total_loss.item()}"

    def test_batched_loss_backward_reaches_all_parameters(self):
        net = HeXONet(_SMALL_CFG)
        batch, targets = self._make_batch_and_targets()

        policy_list, values = net.forward_batch(batch)

        total_loss = torch.tensor(0.0)
        for i, (policy_logits, value) in enumerate(zip(policy_list, values)):
            policy_target, value_target = targets[i]
            loss, _ = alphazero_loss(policy_logits, policy_target, value, value_target)
            total_loss = total_loss + loss

        total_loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"

    def test_batched_values_shape(self):
        net = HeXONet(_SMALL_CFG)
        batch, _ = self._make_batch_and_targets()

        with torch.no_grad():
            _, values = net.forward_batch(batch)

        assert values.shape == (3,)

    def test_batched_policy_shapes_match_legal_counts(self):
        net = HeXONet(_SMALL_CFG)
        games = [_initial_game(), _game_after_moves(4)]
        graphs = [game_to_graph(g) for g in games]
        batch = Batch.from_data_list(graphs)

        with torch.no_grad():
            policy_list, _ = net.forward_batch(batch)

        for i, (graph, policy_logits) in enumerate(zip(graphs, policy_list)):
            expected_num_legal = graph.legal_mask.sum().item()
            assert policy_logits.shape == (expected_num_legal,), (
                f"Graph {i}: expected ({expected_num_legal},), got {policy_logits.shape}"
            )


# ---------------------------------------------------------------------------
# AC-14c: Overfit test
# ---------------------------------------------------------------------------

class TestOverfit:
    """AC-14c: loss decreases when training on a single position for 50 steps."""

    def test_loss_decreases_over_50_steps(self):
        torch.manual_seed(42)
        net = HeXONet(_SMALL_CFG)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        game = _initial_game()
        data = game_to_graph(game)
        num_legal = data.legal_mask.sum().item()
        policy_target = _uniform_policy_target(num_legal)
        value_target = _value_target(0.5)

        initial_loss: float | None = None
        final_loss: float | None = None

        for step in range(50):
            optimizer.zero_grad()
            policy_logits, value = net(data.x, data.edge_index, data.legal_mask)
            loss, _ = alphazero_loss(policy_logits, policy_target, value, value_target)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"
        )


# ---------------------------------------------------------------------------
# AC-14b: GPU smoke tests
# ---------------------------------------------------------------------------

class TestGPUSmoke:
    """AC-14b: forward + backward on CUDA device; outputs land on cuda."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not available"
    )
    def test_single_forward_backward_on_gpu(self):
        device = torch.device("cuda")
        net = HeXONet(_SMALL_CFG).to(device)

        game = _initial_game()
        data = game_to_graph(game)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        legal_mask = data.legal_mask.to(device)

        policy_logits, value = net(x, edge_index, legal_mask)

        assert policy_logits.device.type == "cuda", (
            f"policy_logits not on cuda: {policy_logits.device}"
        )
        assert value.device.type == "cuda", (
            f"value not on cuda: {value.device}"
        )

        num_legal = legal_mask.sum().item()
        policy_target = _uniform_policy_target(num_legal).to(device)
        value_target = _value_target(0.5).to(device)

        loss, _ = alphazero_loss(policy_logits, policy_target, value, value_target)
        loss.backward()

        assert torch.isfinite(loss)
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient on GPU for parameter: {name}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not available"
    )
    def test_batched_forward_backward_on_gpu(self):
        device = torch.device("cuda")
        net = HeXONet(_SMALL_CFG).to(device)

        games = [_initial_game(), _game_after_moves(4)]
        graphs = [game_to_graph(g) for g in games]
        batch = Batch.from_data_list(graphs).to(device)

        policy_list, values = net.forward_batch(batch)

        assert values.device.type == "cuda", f"values not on cuda: {values.device}"
        for i, logits in enumerate(policy_list):
            assert logits.device.type == "cuda", (
                f"policy_list[{i}] not on cuda: {logits.device}"
            )

        total_loss = torch.tensor(0.0, device=device)
        for i, (logits, value) in enumerate(zip(policy_list, values)):
            num_legal = logits.shape[0]
            policy_target = _uniform_policy_target(num_legal).to(device)
            value_target = _value_target(0.5).to(device)
            loss, _ = alphazero_loss(logits, policy_target, value, value_target)
            total_loss = total_loss + loss

        total_loss.backward()

        assert torch.isfinite(total_loss)
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient on GPU for parameter: {name}"
