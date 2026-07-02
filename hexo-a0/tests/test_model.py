"""Tests for ModelConfig and GNN model components."""

import pytest
import torch
from torch_geometric.data import Batch, Data

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_graph
from hexo_a0.model import HeXONet, PolicyHead, RepresentationNetwork, ValueHead


class TestModelConfigDefaults:
    """AC-13a and AC-13b: default instantiation and default field values."""

    def test_default_instantiation(self):
        cfg = ModelConfig()
        assert cfg is not None

    def test_default_hidden_dim(self):
        assert ModelConfig().hidden_dim == 256

    def test_default_num_layers(self):
        assert ModelConfig().num_layers == 3

    def test_default_num_heads(self):
        assert ModelConfig().num_heads == 8

    def test_default_policy_hidden(self):
        assert ModelConfig().policy_hidden == 128

    def test_default_value_hidden(self):
        assert ModelConfig().value_hidden == 128

    def test_default_dropout(self):
        assert ModelConfig().dropout == 0.0


class TestModelConfigCustomValues:
    """AC-13c: custom values can be passed at construction."""

    def test_custom_hidden_dim(self):
        cfg = ModelConfig(hidden_dim=256)
        assert cfg.hidden_dim == 256

    def test_custom_num_layers(self):
        cfg = ModelConfig(num_layers=3)
        assert cfg.num_layers == 3

    def test_custom_num_heads(self):
        cfg = ModelConfig(num_heads=8)
        assert cfg.num_heads == 8

    def test_custom_policy_hidden(self):
        cfg = ModelConfig(policy_hidden=128)
        assert cfg.policy_hidden == 128

    def test_custom_value_hidden(self):
        cfg = ModelConfig(value_hidden=128)
        assert cfg.value_hidden == 128

    def test_custom_dropout(self):
        cfg = ModelConfig(dropout=0.1)
        assert cfg.dropout == 0.1

    def test_all_custom(self):
        cfg = ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            policy_hidden=32,
            value_hidden=32,
            dropout=0.5,
        )
        assert cfg.hidden_dim == 64
        assert cfg.num_layers == 2
        assert cfg.num_heads == 2
        assert cfg.policy_hidden == 32
        assert cfg.value_hidden == 32
        assert cfg.dropout == 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(num_nodes: int, node_features: int, num_edges: int = 0):
    """Return (x, edge_index) tensors for a random graph with real edges."""
    x = torch.randn(num_nodes, node_features, requires_grad=True)
    if num_edges == 0:
        # Chain edges (0→1, 1→0, 1→2, 2→1, ...) so GATv2Conv attention is exercised
        src, dst = [], []
        for i in range(num_nodes - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])
        if not src:
            # Single node: self-loop as fallback
            src, dst = [0], [0]
        edge_index = torch.tensor([src, dst], dtype=torch.int64)
    else:
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        edge_index = torch.stack([src, dst], dim=0)
    return x, edge_index


# ---------------------------------------------------------------------------
# REQ-6 / AC-6a / AC-6b  RepresentationNetwork tests
# ---------------------------------------------------------------------------

class TestRepresentationNetworkOutputShape:
    """AC-6a: output shape is (N, hidden_dim) for any graph size."""

    @pytest.mark.parametrize("num_nodes,hidden_dim,num_heads", [
        (1,  64, 4),
        (5,  64, 4),
        (20, 128, 4),
        (50, 128, 8),
    ])
    def test_output_shape(self, num_nodes, hidden_dim, num_heads):
        cfg = ModelConfig(hidden_dim=hidden_dim, num_heads=num_heads, graph_type="hex", conv_type="gatv2")
        net = RepresentationNetwork(cfg)
        x, edge_index = _make_graph(num_nodes, 8)
        with torch.no_grad():
            out = net(x, edge_index)
        assert out.shape == (num_nodes, hidden_dim)

    def test_different_graph_sizes_give_correct_shape(self):
        """Different N values all produce (N, hidden_dim)."""
        cfg = ModelConfig(hidden_dim=64, num_heads=4, num_layers=2, graph_type="hex", conv_type="gatv2")
        net = RepresentationNetwork(cfg)
        for n in (3, 7, 25):
            x, edge_index = _make_graph(n, 8)
            with torch.no_grad():
                out = net(x, edge_index)
            assert out.shape == (n, cfg.hidden_dim), (
                f"Expected ({n}, {cfg.hidden_dim}), got {out.shape}"
            )


class TestRepresentationNetworkDtype:
    """Output tensor dtype must be float32."""

    def test_output_dtype_is_float32(self):
        cfg = ModelConfig(hidden_dim=64, num_heads=4, num_layers=2, graph_type="hex", conv_type="gatv2")
        net = RepresentationNetwork(cfg)
        x, edge_index = _make_graph(10, 8)
        with torch.no_grad():
            out = net(x, edge_index)
        assert out.dtype == torch.float32


class TestRepresentationNetworkGradients:
    """AC-6b: gradients flow through all layers back to the input."""

    def test_gradients_flow_to_input(self):
        cfg = ModelConfig(hidden_dim=64, num_heads=4, num_layers=2, graph_type="hex", conv_type="gatv2")
        net = RepresentationNetwork(cfg)
        x, edge_index = _make_graph(10, 8)
        # x already has requires_grad=True from _make_graph
        out = net(x, edge_index)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradients_nonzero(self):
        # Use a fixed seed: GATv2Conv attention can produce zero input
        # gradients with certain random initialisations + inputs.
        torch.manual_seed(42)
        cfg = ModelConfig(hidden_dim=64, num_heads=4, num_layers=3, graph_type="hex", conv_type="gatv2")
        net = RepresentationNetwork(cfg)
        x, edge_index = _make_graph(8, 8)
        out = net(x, edge_index)
        out.sum().backward()
        assert x.grad.abs().sum().item() > 0.0


class TestRepresentationNetworkInvalidConfig:
    """ValueError raised when hidden_dim % num_heads != 0."""

    def test_invalid_head_dim_raises(self):
        cfg = ModelConfig(hidden_dim=65, num_heads=4, graph_type="hex", conv_type="gatv2")
        with pytest.raises(ValueError, match="hidden_dim"):
            RepresentationNetwork(cfg)

    def test_valid_head_dim_does_not_raise(self):
        cfg = ModelConfig(hidden_dim=64, num_heads=4, graph_type="hex", conv_type="gatv2")
        net = RepresentationNetwork(cfg)   # must not raise
        assert net is not None


class TestRepresentationNetworkDropout:
    """Dropout=0.0 uses nn.Identity; dropout>0.0 uses nn.Dropout."""

    def test_no_dropout_uses_identity(self):
        cfg = ModelConfig(hidden_dim=64, num_heads=4, dropout=0.0, graph_type="hex", conv_type="gatv2")
        net = RepresentationNetwork(cfg)
        assert isinstance(net.dropout, torch.nn.Identity)

    def test_positive_dropout_uses_dropout_layer(self):
        cfg = ModelConfig(hidden_dim=64, num_heads=4, dropout=0.1, graph_type="hex", conv_type="gatv2")
        net = RepresentationNetwork(cfg)
        assert isinstance(net.dropout, torch.nn.Dropout)


# ---------------------------------------------------------------------------
# Helpers shared by new head / HeXONet tests
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


def _make_embeddings(num_nodes: int, hidden_dim: int = 32) -> torch.Tensor:
    """Random embedding matrix with grad enabled."""
    return torch.randn(num_nodes, hidden_dim, requires_grad=True)


def _make_legal_mask(num_nodes: int, num_legal: int) -> torch.Tensor:
    """Boolean mask with exactly num_legal True entries."""
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[:num_legal] = True
    return mask


def _make_pyg_data(
    num_nodes: int,
    num_legal: int,
    node_features: int = 8,
) -> Data:
    """Build a minimal PyG Data object with chain edges and a legal_mask."""
    x = torch.randn(num_nodes, node_features)
    src, dst = [], []
    for i in range(num_nodes - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    if not src:
        src, dst = [0], [0]
    edge_index = torch.tensor([src, dst], dtype=torch.int64)
    legal_mask = _make_legal_mask(num_nodes, num_legal)
    return Data(x=x, edge_index=edge_index, legal_mask=legal_mask)


# ---------------------------------------------------------------------------
# REQ-7 / AC-7a / AC-7b  PolicyHead tests
# ---------------------------------------------------------------------------

class TestPolicyHeadOutputLength:
    """AC-7a: output length equals number of True entries in legal_mask."""

    @pytest.mark.parametrize("num_nodes,num_legal", [
        (10, 3),
        (10, 10),
        (10, 1),
        (25, 12),
    ])
    def test_output_length_matches_legal_count(self, num_nodes, num_legal):
        head = PolicyHead(hidden_dim=32, policy_hidden=16)
        embeddings = _make_embeddings(num_nodes)
        mask = _make_legal_mask(num_nodes, num_legal)
        with torch.no_grad():
            logits = head(embeddings, mask)
        assert logits.shape == (num_legal,), (
            f"Expected ({num_legal},), got {logits.shape}"
        )

    def test_output_is_1d(self):
        head = PolicyHead(hidden_dim=32, policy_hidden=16)
        embeddings = _make_embeddings(10)
        mask = _make_legal_mask(10, 4)
        with torch.no_grad():
            logits = head(embeddings, mask)
        assert logits.dim() == 1


class TestPolicyHeadSoftmax:
    """AC-7b: softmax of output sums to ~1.0."""

    def test_softmax_sums_to_one(self):
        head = PolicyHead(hidden_dim=32, policy_hidden=16)
        embeddings = _make_embeddings(15)
        mask = _make_legal_mask(15, 7)
        with torch.no_grad():
            logits = head(embeddings, mask)
        probs = torch.softmax(logits, dim=0)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_softmax_all_legal(self):
        head = PolicyHead(hidden_dim=32, policy_hidden=16)
        embeddings = _make_embeddings(8)
        mask = torch.ones(8, dtype=torch.bool)
        with torch.no_grad():
            logits = head(embeddings, mask)
        probs = torch.softmax(logits, dim=0)
        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestPolicyHeadGradients:
    """Gradients flow through PolicyHead back to embeddings."""

    def test_gradients_flow_to_embeddings(self):
        head = PolicyHead(hidden_dim=32, policy_hidden=16)
        embeddings = _make_embeddings(10)   # requires_grad=True
        mask = _make_legal_mask(10, 5)
        logits = head(embeddings, mask)
        logits.sum().backward()
        assert embeddings.grad is not None
        assert embeddings.grad.abs().sum().item() > 0.0

    def test_gradients_flow_to_parameters(self):
        head = PolicyHead(hidden_dim=32, policy_hidden=16)
        embeddings = _make_embeddings(10)
        mask = _make_legal_mask(10, 5)
        logits = head(embeddings, mask)
        logits.sum().backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# REQ-8 / AC-8a / AC-8b / AC-8c  ValueHead tests
# ---------------------------------------------------------------------------

class TestValueHeadScalarOutput:
    """AC-8a: ValueHead returns a 0-D scalar tensor."""

    def test_output_is_scalar(self):
        head = ValueHead(hidden_dim=32, value_hidden=16)
        embeddings = _make_embeddings(10)
        with torch.no_grad():
            value = head(embeddings)
        assert value.dim() == 0, f"Expected 0-D tensor, got shape {value.shape}"

    @pytest.mark.parametrize("num_nodes", [1, 5, 25, 100])
    def test_output_shape_independent_of_graph_size(self, num_nodes):
        head = ValueHead(hidden_dim=32, value_hidden=16)
        embeddings = _make_embeddings(num_nodes)
        with torch.no_grad():
            value = head(embeddings)
        assert value.dim() == 0


class TestValueHeadRange:
    """AC-8b: ValueHead output is in [-1, 1]."""

    def test_value_in_minus_one_to_one(self):
        head = ValueHead(hidden_dim=32, value_hidden=16)
        for _ in range(20):
            embeddings = _make_embeddings(10)
            with torch.no_grad():
                value = head(embeddings)
            assert -1.0 <= value.item() <= 1.0

    def test_value_range_with_large_inputs(self):
        head = ValueHead(hidden_dim=32, value_hidden=16)
        embeddings = torch.randn(50, 32) * 100.0
        with torch.no_grad():
            value = head(embeddings)
        assert -1.0 <= value.item() <= 1.0


class TestValueHeadGradients:
    """AC-8c: Gradients flow through the pooling operation."""

    def test_gradients_flow_through_pooling(self):
        head = ValueHead(hidden_dim=32, value_hidden=16)
        embeddings = _make_embeddings(10)   # requires_grad=True
        value = head(embeddings)
        value.backward()
        assert embeddings.grad is not None
        assert embeddings.grad.abs().sum().item() > 0.0

    def test_gradients_flow_to_all_parameters(self):
        head = ValueHead(hidden_dim=32, value_hidden=16)
        embeddings = _make_embeddings(10)
        value = head(embeddings)
        value.backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# REQ-9 / AC-9a / AC-9b  HeXONet tests
# ---------------------------------------------------------------------------

class TestHeXONetForward:
    """AC-9a: forward() returns (policy_logits, value) with correct shapes."""

    def test_forward_returns_tuple(self):
        net = HeXONet(_SMALL_CFG)
        data = _make_pyg_data(num_nodes=10, num_legal=4, node_features=8)
        result = net(data.x, data.edge_index, data.legal_mask)
        assert isinstance(result, tuple) and len(result) == 2

    def test_forward_policy_shape(self):
        net = HeXONet(_SMALL_CFG)
        num_legal = 5
        data = _make_pyg_data(num_nodes=12, num_legal=num_legal, node_features=8)
        policy, _ = net(data.x, data.edge_index, data.legal_mask)
        assert policy.shape == (num_legal,)

    def test_forward_value_is_scalar(self):
        net = HeXONet(_SMALL_CFG)
        data = _make_pyg_data(num_nodes=12, num_legal=5, node_features=8)
        _, value = net(data.x, data.edge_index, data.legal_mask)
        assert value.dim() == 0

    def test_forward_value_in_range(self):
        net = HeXONet(_SMALL_CFG)
        data = _make_pyg_data(num_nodes=12, num_legal=5, node_features=8)
        with torch.no_grad():
            _, value = net(data.x, data.edge_index, data.legal_mask)
        assert -1.0 <= value.item() <= 1.0


class TestHeXONetForwardBatch:
    """AC-9b: forward_batch() returns (list[policy_logits], values_tensor)."""

    def test_forward_batch_single_graph(self):
        net = HeXONet(_SMALL_CFG)
        data = _make_pyg_data(num_nodes=10, num_legal=4, node_features=8)
        batch = Batch.from_data_list([data])
        policy_list, values = net.forward_batch(batch)
        assert isinstance(policy_list, list) and len(policy_list) == 1
        assert policy_list[0].shape == (4,)
        assert values.shape == (1,)

    def test_forward_batch_multiple_graphs(self):
        net = HeXONet(_SMALL_CFG)
        graphs = [
            _make_pyg_data(num_nodes=8,  num_legal=3, node_features=8),
            _make_pyg_data(num_nodes=12, num_legal=6, node_features=8),
            _make_pyg_data(num_nodes=5,  num_legal=2, node_features=8),
        ]
        batch = Batch.from_data_list(graphs)
        policy_list, values = net.forward_batch(batch)
        assert len(policy_list) == 3
        assert policy_list[0].shape == (3,)
        assert policy_list[1].shape == (6,)
        assert policy_list[2].shape == (2,)
        assert values.shape == (3,)

    def test_forward_batch_values_in_range(self):
        net = HeXONet(_SMALL_CFG)
        graphs = [_make_pyg_data(10, 4), _make_pyg_data(8, 3)]
        batch = Batch.from_data_list(graphs)
        with torch.no_grad():
            _, values = net.forward_batch(batch)
        assert values.shape == (2,)
        for v in values:
            assert -1.0 <= v.item() <= 1.0


class TestHeXONetGradients:
    """Gradients flow to all parameters of HeXONet."""

    def test_gradients_flow_to_all_parameters(self):
        net = HeXONet(_SMALL_CFG)
        data = _make_pyg_data(num_nodes=10, num_legal=5, node_features=8)
        policy, value = net(data.x, data.edge_index, data.legal_mask)
        loss = policy.sum() + value
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No grad for {name}"
            assert param.grad.abs().sum().item() > 0.0, f"Zero grad for {name}"

    def test_gradients_flow_through_batched_forward(self):
        net = HeXONet(_SMALL_CFG)
        graphs = [_make_pyg_data(10, 4), _make_pyg_data(8, 3)]
        batch = Batch.from_data_list(graphs)
        policy_list, values = net.forward_batch(batch)
        loss = sum(p.sum() for p in policy_list) + values.sum()
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# AC-12 / AC-9c  HeXONet batched forward with real game graphs
# ---------------------------------------------------------------------------

_BATCHED_CFG = ModelConfig(hidden_dim=32, num_layers=2, num_heads=2, graph_type="hex", conv_type="gatv2")


def _make_real_graph():
    """Return a PyG Data object from a real GameState."""
    game = hexo_rs.GameState()
    return game_to_graph(game)


class TestHeXONetBatched:
    """AC-12 / AC-9c: forward_batch on real game graphs."""

    def test_forward_batch_policy_shapes(self):
        """AC-12a: each policy logits tensor has shape (num_legal_i,)."""
        net = HeXONet(_BATCHED_CFG)
        g1 = _make_real_graph()
        g2 = _make_real_graph()
        batch = Batch.from_data_list([g1, g2])
        with torch.no_grad():
            policy_list, values = net.forward_batch(batch)
        assert len(policy_list) == 2
        assert policy_list[0].shape == (g1.legal_mask.sum().item(),)
        assert policy_list[1].shape == (g2.legal_mask.sum().item(),)

    def test_forward_batch_values_shape(self):
        """AC-12b: values tensor has shape (batch_size,)."""
        net = HeXONet(_BATCHED_CFG)
        g1 = _make_real_graph()
        g2 = _make_real_graph()
        batch = Batch.from_data_list([g1, g2])
        with torch.no_grad():
            _, values = net.forward_batch(batch)
        assert values.shape == (2,)

    def test_forward_batch_core_with_precomputed_indices(self):
        """Index-tensor path produces identical output to boolean-mask path."""
        config = ModelConfig(
            hidden_dim=32, num_layers=2, num_heads=4,
            policy_hidden=16, value_hidden=16,
            graph_type="hex", conv_type="gatv2",
        )
        net = HeXONet(config)
        net.eval()

        # Build two small graphs
        graphs = []
        for n_stones, n_legal in [(3, 5), (2, 4)]:
            n = n_stones + n_legal
            x = torch.randn(n, 8)
            # Simple chain edges
            src = list(range(n - 1))
            dst = list(range(1, n))
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            legal_mask = torch.zeros(n, dtype=torch.bool)
            legal_mask[n_stones:] = True
            stone_mask = torch.zeros(n, dtype=torch.bool)
            stone_mask[:n_stones] = True
            data = Data(x=x, edge_index=edge_index, legal_mask=legal_mask, stone_mask=stone_mask)
            graphs.append(data)

        batch = Batch.from_data_list(graphs)

        with torch.no_grad():
            # Boolean path (existing)
            logits_bool, counts_bool, values_bool = net._forward_batch_core(batch)

            # Precompute indices
            legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
            stone_idx = batch.stone_mask.nonzero(as_tuple=False).squeeze(1)
            stone_batch = batch.batch[stone_idx]

            # Index path (new)
            logits_idx, counts_idx, values_idx = net._forward_batch_core(
                batch, legal_idx=legal_idx, stone_idx=stone_idx, stone_batch=stone_batch,
            )

        torch.testing.assert_close(logits_bool, logits_idx)
        torch.testing.assert_close(counts_bool, counts_idx)
        torch.testing.assert_close(values_bool, values_idx)

    def test_forward_vs_forward_batch_equivalence(self):
        """AC-9c: forward(graph) == forward_batch([graph]) for policy and value."""
        net = HeXONet(_BATCHED_CFG)
        net.eval()
        graph = _make_real_graph()
        batch = Batch.from_data_list([graph])
        stone_mask = graph.stone_mask if hasattr(graph, "stone_mask") else None
        with torch.no_grad():
            policy_single, value_single = net(
                graph.x, graph.edge_index, graph.legal_mask, stone_mask
            )
            policy_list, values_batch = net.forward_batch(batch)
        assert torch.allclose(policy_single, policy_list[0], atol=1e-6), (
            "Policy logits differ between forward() and forward_batch()"
        )
        assert torch.allclose(value_single, values_batch[0], atol=1e-6), (
            "Value differs between forward() and forward_batch()"
        )
