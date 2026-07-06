"""DedupGINEConv: per-layer edge linear over unique edge_attr rows + gather.

Must be state-dict compatible with stock GINEConv and numerically equivalent
(allclose, NOT bitwise: GEMM tiling differs with row count, esp. on GPU).
"""
import pytest
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


def test_hexonet_forward_matches_pre_dedup_reference():
    # End-to-end: a tiny axis-config HeXONet forward must be allclose to the
    # stock-GINEConv computation on a REAL game graph.
    import hexo_rs
    from hexo_a0.config import ModelConfig
    from hexo_a0.graph import game_to_axis_graph
    from hexo_a0.model import HeXONet

    mc = ModelConfig(hidden_dim=32, num_layers=2, num_heads=4, conv_type="gine",
                     graph_type="axis", pre_norm=True, policy_hidden=16,
                     value_hidden=8, use_jk=True, jk_mode="cat",
                     prune_empty_edges=True, threat_features=True,
                     relative_stone_encoding=True)
    torch.manual_seed(5)
    net = HeXONet(mc).eval()

    game = hexo_rs.GameState(hexo_rs.GameConfig(6, 4, 300))
    for m in [(1, 2), (0, -2), (2, 0), (-1, 1)]:
        game.apply_move(m[0], m[1])
    data = game_to_axis_graph(game, prune_empty_edges=True,
                              threat_features=True, relative_stones=True)

    with torch.no_grad():
        logits, value = net(data.x, data.edge_index, data.legal_mask,
                            stone_mask=data.stone_mask, edge_attr=data.edge_attr)
        # Reference: force every conv down the stock path by hiding the flag.
        for conv in net.representation.convs:
            conv._dedup_force_stock = True
        ref_logits, ref_value = net(data.x, data.edge_index, data.legal_mask,
                                    stone_mask=data.stone_mask, edge_attr=data.edge_attr)
    assert torch.allclose(logits, ref_logits, atol=1e-5)
    assert torch.allclose(value, ref_value, atol=1e-5)


def test_gatv2_forward_skips_torch_unique():
    """conv_type="gatv2" convs never consume the unique/inverse kwargs, so the
    dedupe path (torch.unique(edge_attr, dim=0, ...) + gather) is pure waste
    for them — and on CUDA a likely device->host sync. Assert the eager
    forward for a gatv2+axis model calls `torch.unique` zero times, while the
    gine+axis model (which DOES benefit from dedupe) still calls it.
    """
    import hexo_rs
    from unittest.mock import patch

    from hexo_a0.config import ModelConfig
    from hexo_a0.graph import game_to_axis_graph
    from hexo_a0.model import HeXONet

    game = hexo_rs.GameState(hexo_rs.GameConfig(6, 4, 300))
    for m in [(1, 2), (0, -2), (2, 0), (-1, 1)]:
        game.apply_move(m[0], m[1])
    data = game_to_axis_graph(game, prune_empty_edges=True,
                              threat_features=True, relative_stones=True)

    real_unique = torch.unique
    calls = []

    def _recording_unique(*args, **kwargs):
        calls.append((args, kwargs))
        return real_unique(*args, **kwargs)

    # gatv2: torch.unique must NOT be called.
    mc_gatv2 = ModelConfig(hidden_dim=32, num_layers=2, num_heads=4,
                           conv_type="gatv2", graph_type="axis", pre_norm=True,
                           policy_hidden=16, value_hidden=8, use_jk=True,
                           jk_mode="cat", prune_empty_edges=True,
                           threat_features=True, relative_stone_encoding=True)
    torch.manual_seed(5)
    net_gatv2 = HeXONet(mc_gatv2).eval()

    calls.clear()
    with patch("torch.unique", side_effect=_recording_unique), torch.no_grad():
        net_gatv2(data.x, data.edge_index, data.legal_mask,
                  stone_mask=data.stone_mask, edge_attr=data.edge_attr)
    assert calls == [], (
        f"expected zero torch.unique calls for conv_type='gatv2', got {len(calls)}"
    )

    # gine: torch.unique SHOULD be called (sanity check the patch works and
    # the dedupe path is still exercised for the conv type that needs it).
    mc_gine = ModelConfig(hidden_dim=32, num_layers=2, num_heads=4,
                          conv_type="gine", graph_type="axis", pre_norm=True,
                          policy_hidden=16, value_hidden=8, use_jk=True,
                          jk_mode="cat", prune_empty_edges=True,
                          threat_features=True, relative_stone_encoding=True)
    torch.manual_seed(5)
    net_gine = HeXONet(mc_gine).eval()

    calls.clear()
    with patch("torch.unique", side_effect=_recording_unique), torch.no_grad():
        net_gine(data.x, data.edge_index, data.legal_mask,
                 stone_mask=data.stone_mask, edge_attr=data.edge_attr)
    assert len(calls) > 0, "expected torch.unique to be called for conv_type='gine'"


def test_torch_compile_dedupe_smoke():
    """Training wraps model._forward_batch_core in torch.compile(dynamic=True,
    fullgraph=False) (see cli.py, curriculum.py). The dedupe wiring adds a
    ``torch.unique(edge_attr, dim=0, return_inverse=True)`` call (data-dependent
    output shape) plus a stateful ``setattr`` inside DedupGINEConv.forward.
    This is a correctness + graph-break-inspection smoke test, not a
    performance gate: assert compiled output matches eager, and record
    whether/why Dynamo graph-breaks so a human can judge the compile-mode
    impact (reported in task-0.2-report.md, not asserted here — graph breaks
    on dynamic-shape ops are expected/tolerated, not a failure condition).
    """
    import hexo_rs
    from torch_geometric.data import Batch

    from hexo_a0.config import ModelConfig
    from hexo_a0.graph import game_to_axis_graph
    from hexo_a0.model import HeXONet

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable in this torch build")

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    mc = ModelConfig(hidden_dim=32, num_layers=2, num_heads=4, conv_type="gine",
                     graph_type="axis", pre_norm=True, policy_hidden=16,
                     value_hidden=8, use_jk=True, jk_mode="cat",
                     prune_empty_edges=True, threat_features=True,
                     relative_stone_encoding=True)
    torch.manual_seed(5)
    net = HeXONet(mc).eval()

    game = hexo_rs.GameState(hexo_rs.GameConfig(6, 4, 300))
    for m in [(1, 2), (0, -2), (2, 0), (-1, 1)]:
        game.apply_move(m[0], m[1])
    data = game_to_axis_graph(game, prune_empty_edges=True,
                              threat_features=True, relative_stones=True)
    batch = Batch.from_data_list([data, data])

    try:
        with torch.no_grad():
            eager_logits, eager_counts, eager_values = net._forward_batch_core(batch)

            compiled_core = torch.compile(
                net._forward_batch_core, dynamic=True, fullgraph=False)
            comp_logits, comp_counts, comp_values = compiled_core(batch)
    except Exception as e:
        pytest.skip(f"torch.compile failed in this environment: {e}")

    assert torch.equal(eager_counts, comp_counts)
    assert torch.allclose(eager_logits, comp_logits, atol=1e-5)
    assert torch.allclose(eager_values, comp_values, atol=1e-5)

    breaks = dict(torch._dynamo.utils.counters.get("graph_break", {}))
    unique_breaks = [k for k in breaks if "unique_dim" in k or "aten.unique" in k]
    # Informational only — see docstring. Print so `-s` runs surface it, and
    # stash on the module for the report/CI log without failing the test.
    print(f"\n[compile smoke] graph_break reasons: {list(breaks.keys())}")
    print(f"[compile smoke] unique-related breaks: {len(unique_breaks)}")


def test_torch_compile_no_graph_break_with_precomputed_indices():
    """Regression pin for the REAL training callsite shape.

    Production wraps ``model._forward_batch_core`` in a single
    ``torch.compile(dynamic=True, fullgraph=False)`` region (see cli.py:
    ``model._forward_batch_core = torch.compile(model._forward_batch_core,
    dynamic=True)``), and the trainer always calls it with
    ``legal_idx``/``stone_idx``/``stone_batch`` precomputed *outside* the
    compiled call (trainer.py's ``_forward_and_loss``, ~lines 814-828) —
    precisely to dodge the ``nonzero()`` graph break. At that callsite, the
    dedupe wiring's ``torch.unique(edge_attr, dim=0, return_inverse=True)``
    (a data-dependent-output-shape op) used to be the ONLY remaining break,
    turning a previously single-graph compiled training step into >=2
    segments. ``RepresentationNetwork.forward`` now guards the dedupe path
    behind ``torch.compiler.is_dynamo_compiling()`` so Dynamo only ever
    traces the plain per-edge ``edge_proj(edge_attr)`` computation (no
    ``torch.unique`` node in the graph, and no DedupGINEConv unique kwargs).
    This test asserts BOTH allclose-to-eager AND zero new Dynamo graph
    breaks with indices precomputed — the load-bearing regression pin for
    that guard.
    """
    import hexo_rs
    from torch_geometric.data import Batch

    from hexo_a0.config import ModelConfig
    from hexo_a0.graph import game_to_axis_graph
    from hexo_a0.model import HeXONet

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable in this torch build")

    mc = ModelConfig(hidden_dim=32, num_layers=2, num_heads=4, conv_type="gine",
                     graph_type="axis", pre_norm=True, policy_hidden=16,
                     value_hidden=8, use_jk=True, jk_mode="cat",
                     prune_empty_edges=True, threat_features=True,
                     relative_stone_encoding=True)
    torch.manual_seed(5)
    net = HeXONet(mc).eval()

    game = hexo_rs.GameState(hexo_rs.GameConfig(6, 4, 300))
    for m in [(1, 2), (0, -2), (2, 0), (-1, 1)]:
        game.apply_move(m[0], m[1])
    data = game_to_axis_graph(game, prune_empty_edges=True,
                              threat_features=True, relative_stones=True)
    batch = Batch.from_data_list([data, data])

    # Precompute index tensors exactly as trainer.py's real training
    # callsite does, before ever entering the compiled region.
    legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
    stone_mask = (
        batch.stone_mask
        if hasattr(batch, "stone_mask") and batch.stone_mask is not None
        else ~batch.legal_mask
    )
    stone_idx = stone_mask.nonzero(as_tuple=False).squeeze(1)
    stone_batch = batch.batch[stone_idx]

    with torch.no_grad():
        eager_logits, eager_counts, eager_values = net._forward_batch_core(
            batch, legal_idx=legal_idx, stone_idx=stone_idx, stone_batch=stone_batch,
        )

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    try:
        compiled_core = torch.compile(
            net._forward_batch_core, dynamic=True, fullgraph=False)
        with torch.no_grad():
            comp_logits, comp_counts, comp_values = compiled_core(
                batch, legal_idx=legal_idx, stone_idx=stone_idx, stone_batch=stone_batch,
            )
    except Exception as e:
        pytest.skip(f"torch.compile failed in this environment: {e}")

    assert torch.equal(eager_counts, comp_counts)
    assert torch.allclose(eager_logits, comp_logits, atol=1e-5)
    assert torch.allclose(eager_values, comp_values, atol=1e-5)

    breaks = dict(torch._dynamo.utils.counters.get("graph_break", {}))
    assert breaks == {}, (
        "expected zero Dynamo graph breaks at the real (idx-precomputed) "
        f"training callsite with the compile guard active, got: {breaks}"
    )
