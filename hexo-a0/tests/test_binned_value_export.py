"""TorchScript-export + scripted-forward tests for the distributional value head.

Covers the wiring of the binned/categorical value head through
``ScriptableHeXONet`` and ``export_torchscript``:

1. Scalar head (value_bins=0) still exports/scripts and returns values in [-1,1].
2. Binned head: the in-forward softmax-expectation decode of the SCRIPTED model
   matches the eager ``HeXONet.forward_batch`` decoded scalar to atol=1e-4,
   proving the scriptable decode reproduces ``loss.decode_binned_value``.
3. Binned scripted values stay within [value_bin_min, value_bin_max].

Inputs are built by hand via PyG so the eager and scripted paths see byte-identical
tensors; ``hexo_rs`` is never imported.
"""

from __future__ import annotations

import tempfile

import pytest

torch = pytest.importorskip("torch")
from torch_geometric.data import Batch, Data

from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_a0.loss import decode_binned_value
from hexo_a0.scriptable_model import ScriptableHeXONet, export_torchscript


def _cfg(value_bins: int = 0, **kw) -> ModelConfig:
    base = dict(
        hidden_dim=32, num_layers=2, num_heads=2, policy_hidden=16,
        value_hidden=16, graph_type="hex", conv_type="gatv2",
    )
    base.update(kw)
    base["value_bins"] = value_bins
    return ModelConfig(**base)


def _data(num_nodes: int = 10, num_legal: int = 4, num_stones: int = 3,
          nf: int = 8, seed: int = 0) -> Data:
    """A tiny chain graph (no self-loops; hex GATv2 uses add_self_loops=False).

    Legal nodes and stone nodes are disjoint so eager/scriptable pool the same
    node set.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, nf, generator=g)
    src, dst = [], []
    for i in range(num_nodes - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    ei = torch.tensor([src, dst], dtype=torch.int64)
    legal_mask = torch.zeros(num_nodes, dtype=torch.bool)
    legal_mask[:num_legal] = True
    stone_mask = torch.zeros(num_nodes, dtype=torch.bool)
    stone_mask[num_legal:num_legal + num_stones] = True
    return Data(x=x, edge_index=ei, legal_mask=legal_mask, stone_mask=stone_mask)


def _strip_orig_mod(sd: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def _scriptable_inputs_from_batch(batch: Batch):
    """Raw tensors the ScriptableHeXONet.forward expects, from a PyG Batch.

    Mirrors the production calling convention: precomputed legal_idx/stone_idx/
    stone_batch so the heads take the index_select path.
    """
    legal_idx = torch.where(batch.legal_mask)[0]
    stone_idx = torch.where(batch.stone_mask)[0]
    stone_batch = batch.batch[stone_idx]
    return (
        batch.x,
        batch.edge_index,
        batch.legal_mask,
        batch.stone_mask,
        batch.batch,
        batch.num_graphs,
        torch.zeros(0),                       # edge_attr (hex: none)
        legal_idx,
        stone_idx,
        stone_batch,
    )


# ---------------------------------------------------------------------------
# 1. Scalar head still exports + scripts and stays in [-1, 1].
# ---------------------------------------------------------------------------


def test_scalar_export_unchanged():
    torch.manual_seed(1)
    cfg = _cfg(value_bins=0)
    net = HeXONet(cfg)
    net.eval()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        out_path = f.name
    export_torchscript(_strip_orig_mod(net.state_dict()), cfg, out_path)
    scripted = torch.jit.load(out_path)

    batch = Batch.from_data_list([_data(seed=1), _data(8, 3, 2, seed=2)])
    inputs = _scriptable_inputs_from_batch(batch)
    with torch.no_grad():
        _, _, values = scripted(*inputs)

    assert values.shape == (2,)
    assert torch.isfinite(values).all()
    assert (values >= -1.0).all() and (values <= 1.0).all()


# ---------------------------------------------------------------------------
# 2. Binned head: scripted decode matches eager HeXONet.forward_batch decode.
# ---------------------------------------------------------------------------


def test_binned_scripted_decode_matches_eager():
    torch.manual_seed(2)
    cfg = _cfg(value_bins=65)
    net = HeXONet(cfg)
    net.eval()
    assert net.value_bins == 65  # sanity: binned head really built

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        out_path = f.name
    export_torchscript(_strip_orig_mod(net.state_dict()), cfg, out_path)
    scripted = torch.jit.load(out_path)

    data_list = [_data(seed=3), _data(9, 3, 3, seed=4), _data(7, 2, 2, seed=5)]
    batch = Batch.from_data_list(data_list)

    with torch.no_grad():
        _, eager_values = net.forward_batch(batch)
        _, _, scripted_values = scripted(*_scriptable_inputs_from_batch(batch))

    assert scripted_values.shape == eager_values.shape == (3,)
    assert torch.allclose(scripted_values, eager_values, atol=1e-4), (
        f"scripted decode differs from eager; max|diff|="
        f"{(scripted_values - eager_values).abs().max().item():.2e}"
    )

    # And the eager values themselves are the softmax-expectation decode.
    _, _, _, extras = net._forward_batch_core(batch, return_train_extras=True)
    bin_logits = extras["value_logits"]
    assert torch.allclose(
        eager_values, decode_binned_value(bin_logits, net.value_bin_centers), atol=1e-5
    )


# ---------------------------------------------------------------------------
# 3. Binned scripted values stay within the bin-center range.
# ---------------------------------------------------------------------------


def test_binned_scripted_values_in_range():
    torch.manual_seed(6)
    vmin, vmax = -1.0, 1.0
    cfg = _cfg(value_bins=51, value_bin_min=vmin, value_bin_max=vmax)
    net = HeXONet(cfg)
    net.eval()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        out_path = f.name
    export_torchscript(_strip_orig_mod(net.state_dict()), cfg, out_path)
    scripted = torch.jit.load(out_path)

    batch = Batch.from_data_list([_data(seed=7), _data(6, 2, 2, seed=8)])
    with torch.no_grad():
        _, _, values = scripted(*_scriptable_inputs_from_batch(batch))

    assert torch.isfinite(values).all()
    assert (values >= vmin - 1e-5).all() and (values <= vmax + 1e-5).all()
