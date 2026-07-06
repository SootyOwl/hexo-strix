"""DedupGINEConv: per-layer edge linear over unique edge_attr rows + gather.

Must be state-dict compatible with stock GINEConv and numerically equivalent
(allclose, NOT bitwise: GEMM tiling differs with row count, esp. on GPU).
"""
import torch
from torch import nn
from torch_geometric.nn import GINEConv

from hexo_a0.model import DedupGINEConv


def _mlp(h):
    return nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))


def _fixture(h=32, n=50, e=400, n_unique=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, h, generator=g)
    edge_index = torch.stack([
        torch.randint(0, n, (e,), generator=g),
        torch.randint(0, n, (e,), generator=g),
    ])
    # e edges drawn from n_unique distinct attr rows (the axis-graph shape).
    table = torch.randn(n_unique, h, generator=g)
    inverse = torch.randint(0, n_unique, (e,), generator=g)
    edge_attr = table.index_select(0, inverse)
    return x, edge_index, edge_attr


def test_state_dict_keys_match_stock_gineconv():
    stock, dedup = GINEConv(_mlp(32), edge_dim=32), DedupGINEConv(_mlp(32), edge_dim=32)
    assert set(stock.state_dict()) == set(dedup.state_dict())


def test_output_allclose_to_stock_gineconv():
    torch.manual_seed(1)
    stock = GINEConv(_mlp(32), edge_dim=32)
    dedup = DedupGINEConv(_mlp(32), edge_dim=32)
    dedup.load_state_dict(stock.state_dict())
    x, edge_index, edge_attr = _fixture()
    want = stock(x, edge_index, edge_attr=edge_attr)
    uniq, inv = torch.unique(edge_attr, dim=0, return_inverse=True)
    got = dedup(x, edge_index, edge_attr=edge_attr,
                edge_attr_unique=uniq, edge_inverse=inv)
    assert torch.allclose(want, got, atol=1e-5), (want - got).abs().max()


def test_falls_back_to_stock_path_without_unique_args():
    torch.manual_seed(2)
    stock = GINEConv(_mlp(32), edge_dim=32)
    dedup = DedupGINEConv(_mlp(32), edge_dim=32)
    dedup.load_state_dict(stock.state_dict())
    x, edge_index, edge_attr = _fixture(seed=3)
    assert torch.allclose(stock(x, edge_index, edge_attr=edge_attr),
                          dedup(x, edge_index, edge_attr=edge_attr), atol=1e-5)


def test_gradients_flow_through_dedup_path():
    dedup = DedupGINEConv(_mlp(32), edge_dim=32)
    x, edge_index, edge_attr = _fixture(seed=4)
    uniq, inv = torch.unique(edge_attr, dim=0, return_inverse=True)
    out = dedup(x, edge_index, edge_attr=edge_attr,
                edge_attr_unique=uniq, edge_inverse=inv)
    out.sum().backward()
    assert dedup.lin.weight.grad is not None
    assert dedup.lin.weight.grad.abs().sum() > 0
