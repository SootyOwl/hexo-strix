"""Parity tests for the D6-invariant axis-relational path in ScriptableHeXONet.

The scriptable model (used by the self-play inference server) must reproduce
the eager ``HeXONet`` axis_relational path EXACTLY:

  - a tied-weight per-axis GINE message + symmetric SUM over the 3 hex axes,
  - a learned per-hop distance embedding indexed by ``edge_dist - 1``,
  - a separate global/dummy relation with its own untied GINE + edge embed,
  - a ``node_update`` MLP over ``cat(x, summed_messages)``,
  - the legacy -> lean conversion shim (wide ``x`` + ``edge_attr[E,5]`` over the
    wire -> lean ``x`` + ``edge_type``/``edge_dist``/``global_edge_index``).

Inputs are built by hand; ``hexo_rs`` is never imported.
"""

from __future__ import annotations

import itertools

import torch

from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_a0.scriptable_model import ScriptableHeXONet, load_from_hexonet

# Lean D6-invariant schema: relative + threat + compact_stone_onehot, no
# node_coords, node-scoped moves -> lean node dim == 8 (legacy width == 11).
LEAN_NODE_DIM = 8
LEGACY_NODE_DIM = 11
HIDDEN = 32
LAYERS = 3
WINDOW = 8


def _model_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=HIDDEN,
        num_layers=LAYERS,
        num_heads=4,
        conv_type="gine",
        graph_type="axis",
        policy_hidden=16,
        value_hidden=16,
        pre_norm=True,
        use_jk=True,
        jk_mode="cat",
        axis_relational=True,
        axis_window=WINDOW,
        relative_stone_encoding=True,
        threat_features=True,
        compact_stone_onehot=True,
        node_coords=False,
        moves_scope="node",
    )


def _build_pair(seed: int = 0):
    """Random-init eager HeXONet + weight-matched ScriptableHeXONet."""
    torch.manual_seed(seed)
    mc = _model_config()
    hexonet = HeXONet(mc)
    hexonet.eval()

    scriptable = ScriptableHeXONet(
        node_features=LEAN_NODE_DIM,
        hidden_dim=HIDDEN,
        num_layers=LAYERS,
        num_heads=4,
        policy_hidden=16,
        value_hidden=16,
        graph_type="axis",
        pre_norm=True,
        conv_type="gine",
        use_jk=True,
        jk_mode="cat",
        axis_relational=True,
        axis_window=WINDOW,
        relative_stone_encoding=True,
        threat_features=True,
        compact_stone_onehot=True,
        node_coords=False,
        moves_scope="node",
    )
    load_from_hexonet(scriptable, hexonet.state_dict())
    scriptable.eval()
    return hexonet, scriptable


# 12 bidirectional axis edges spanning all 3 axis types with varied hops.
_AXIS_EDGES = [
    (0, 1, 0, 1), (1, 0, 0, 1),
    (1, 2, 1, 2), (2, 1, 1, 2),
    (2, 3, 2, 1), (3, 2, 2, 1),
    (3, 4, 0, 3), (4, 3, 0, 3),
    (4, 5, 1, 1), (5, 4, 1, 1),
    (5, 6, 2, 2), (6, 5, 2, 2),
]
_GLOBAL_EDGES = [(0, 6), (6, 0), (3, 0)]
_N = 7
_NUM_LEGAL = 3
_NUM_STONES = 2


def _masks():
    legal_mask = torch.zeros(_N, dtype=torch.bool)
    legal_mask[:_NUM_LEGAL] = True
    stone_mask = torch.zeros(_N, dtype=torch.bool)
    stone_mask[_NUM_LEGAL : _NUM_LEGAL + _NUM_STONES] = True
    batch = torch.zeros(_N, dtype=torch.long)
    return legal_mask, stone_mask, batch


def _native_lean_inputs(seed: int = 3):
    """Lean inputs: 8-dim x + edge_type/edge_dist + separate global edges."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(_N, LEAN_NODE_DIM, generator=g)
    edge_index = torch.tensor(
        [[s for s, _, _, _ in _AXIS_EDGES], [d for _, d, _, _ in _AXIS_EDGES]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([t for _, _, t, _ in _AXIS_EDGES], dtype=torch.long)
    edge_dist = torch.tensor([w for _, _, _, w in _AXIS_EDGES], dtype=torch.long)
    global_edge_index = torch.tensor(
        [[s for s, _ in _GLOBAL_EDGES], [d for _, d in _GLOBAL_EDGES]],
        dtype=torch.long,
    )
    return x, edge_index, edge_type, edge_dist, global_edge_index


def _legacy_inputs(seed: int = 4):
    """Legacy wire graph: 11-dim x + edge_attr[E,5], no edge_type.

    edge_attr = [axis0, axis1, axis2, signed_dist, src_player]. Axis edges get a
    one-hot in cols 0:3 with a SIGNED distance (tests the abs()); global edges
    get an all-zero axis one-hot so the shim routes them to global_edge_index.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(_N, LEGACY_NODE_DIM, generator=g)

    axis_ei = [[s for s, _, _, _ in _AXIS_EDGES], [d for _, d, _, _ in _AXIS_EDGES]]
    axis_rows = []
    for _s, _d, t, w in _AXIS_EDGES:
        oh = [0.0, 0.0, 0.0]
        oh[t] = 1.0
        sign = -1.0 if (w % 2 == 1) else 1.0  # mix of signs to exercise abs()
        src_player = float(torch.randint(0, 2, (1,), generator=g).item())
        axis_rows.append(oh + [sign * float(w), src_player])

    glob_ei = [[s for s, _ in _GLOBAL_EDGES], [d for _, d in _GLOBAL_EDGES]]
    glob_rows = []
    for _s, _d in _GLOBAL_EDGES:
        src_player = float(torch.randint(0, 2, (1,), generator=g).item())
        glob_rows.append([0.0, 0.0, 0.0, 0.0, src_player])

    edge_index = torch.tensor(
        [axis_ei[0] + glob_ei[0], axis_ei[1] + glob_ei[1]], dtype=torch.long
    )
    edge_attr = torch.tensor(axis_rows + glob_rows, dtype=torch.float32)
    return x, edge_index, edge_attr


# ---------------------------------------------------------------------------
# 1. PARITY (the gate): eager HeXONet vs weight-matched ScriptableHeXONet.
# ---------------------------------------------------------------------------


def test_parity_native_lean():
    hexonet, scriptable = _build_pair(seed=0)
    x, edge_index, edge_type, edge_dist, global_edge_index = _native_lean_inputs()
    legal_mask, stone_mask, batch = _masks()

    with torch.no_grad():
        pol_e, val_e = hexonet(
            x, edge_index, legal_mask, stone_mask,
            edge_type=edge_type, edge_dist=edge_dist,
            global_edge_index=global_edge_index,
        )
        logits_s, counts_s, values_s = scriptable(
            x, edge_index, legal_mask, stone_mask, batch, 1,
            torch.zeros(0),  # edge_attr
            torch.zeros(0, dtype=torch.long),  # legal_idx
            torch.zeros(0, dtype=torch.long),  # stone_idx
            torch.zeros(0, dtype=torch.long),  # stone_batch
            edge_type, edge_dist, global_edge_index,
        )

    assert logits_s.shape == pol_e.shape
    assert torch.allclose(pol_e, logits_s, atol=1e-5), (
        f"policy mismatch; max|diff|={(pol_e - logits_s).abs().max().item():.2e}"
    )
    assert torch.allclose(val_e, values_s[0], atol=1e-5), (
        f"value mismatch; |diff|={(val_e - values_s[0]).abs().item():.2e}"
    )


def test_parity_legacy_shim():
    hexonet, scriptable = _build_pair(seed=1)
    x, edge_index, edge_attr = _legacy_inputs()
    legal_mask, stone_mask, batch = _masks()

    with torch.no_grad():
        pol_e, val_e = hexonet(
            x, edge_index, legal_mask, stone_mask, edge_attr=edge_attr,
        )
        logits_s, counts_s, values_s = scriptable(
            x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr,
        )

    assert logits_s.shape == pol_e.shape
    assert torch.allclose(pol_e, logits_s, atol=1e-5), (
        f"policy mismatch (legacy shim); max|diff|="
        f"{(pol_e - logits_s).abs().max().item():.2e}"
    )
    assert torch.allclose(val_e, values_s[0], atol=1e-5), (
        f"value mismatch (legacy shim); |diff|={(val_e - values_s[0]).abs().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 2. Scriptability: torch.jit.script compiles and matches eager-scriptable.
# ---------------------------------------------------------------------------


def test_scriptable_compiles_and_matches():
    _, scriptable = _build_pair(seed=2)
    scripted = torch.jit.script(scriptable)

    # Legacy input == the wire format the inference server actually receives.
    x, edge_index, edge_attr = _legacy_inputs(seed=9)
    legal_mask, stone_mask, batch = _masks()

    with torch.no_grad():
        le, ce, ve = scriptable(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)
        ls, cs, vs = scripted(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)

    assert torch.allclose(le, ls, atol=1e-5), (
        f"scripted logits differ; max|diff|={(le - ls).abs().max().item():.2e}"
    )
    assert torch.equal(ce, cs)
    assert torch.allclose(ve, vs, atol=1e-5), (
        f"scripted values differ; max|diff|={(ve - vs).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 3. Axis-permutation invariance (native-lean input).
# ---------------------------------------------------------------------------


def test_axis_permutation_invariance():
    _, scriptable = _build_pair(seed=5)
    x, edge_index, edge_type, edge_dist, global_edge_index = _native_lean_inputs(seed=6)
    legal_mask, stone_mask, batch = _masks()

    def run(et):
        with torch.no_grad():
            logits, _counts, values = scriptable(
                x, edge_index, legal_mask, stone_mask, batch, 1,
                torch.zeros(0),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                et, edge_dist, global_edge_index,
            )
        return logits, values

    base_logits, base_values = run(edge_type)
    for perm in itertools.permutations(range(3)):
        pmap = torch.tensor(perm, dtype=torch.long)
        permuted_type = pmap[edge_type]
        logits, values = run(permuted_type)
        assert torch.allclose(base_logits, logits, atol=1e-5), (
            f"axis permutation {perm} changed policy logits"
        )
        assert torch.allclose(base_values, values, atol=1e-5), (
            f"axis permutation {perm} changed value"
        )


# ---------------------------------------------------------------------------
# 4. torch.compile(fullgraph=True): the WHOLE POINT. The relational encoder
#    must compile with NO graph breaks / data-dependent guard failures, so the
#    inference server can run fullgraph=True (no ~3x self-play launch overhead).
# ---------------------------------------------------------------------------


def _legacy_forward_inputs(seed: int = 9):
    x, edge_index, edge_attr = _legacy_inputs(seed=seed)
    legal_mask, stone_mask, batch = _masks()
    return (x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)


def _native_forward_inputs(seed: int = 7):
    x, edge_index, edge_type, edge_dist, gei = _native_lean_inputs(seed=seed)
    legal_mask, stone_mask, batch = _masks()
    return (
        x, edge_index, legal_mask, stone_mask, batch, 1,
        torch.zeros(0),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        edge_type, edge_dist, gei,
    )


def test_fullgraph_compiles_and_matches_legacy():
    """torch.compile(fullgraph=True) must NOT raise the data-dependent guard
    error on the LEGACY wire graph (the exact path that forced the server onto
    fullgraph=False) and must match eager to atol=1e-5."""
    _, scriptable = _build_pair(seed=2)
    inputs = _legacy_forward_inputs(seed=9)

    torch._dynamo.reset()
    compiled = torch.compile(scriptable, fullgraph=True)
    with torch.no_grad():
        le, ce, ve = scriptable(*inputs)
        lc, cc, vc = compiled(*inputs)  # must not raise

    assert torch.allclose(le, lc, atol=1e-5), (
        f"fullgraph logits differ; max|diff|={(le - lc).abs().max().item():.2e}"
    )
    assert torch.equal(ce, cc)
    assert torch.allclose(ve, vc, atol=1e-5), (
        f"fullgraph values differ; max|diff|={(ve - vc).abs().max().item():.2e}"
    )


def test_fullgraph_compiles_and_matches_native_lean():
    """Same fullgraph guarantee on native-lean inputs (edge_type + separate
    global_edge_index folded into the unified bucket representation)."""
    _, scriptable = _build_pair(seed=2)
    inputs = _native_forward_inputs(seed=7)

    torch._dynamo.reset()
    compiled = torch.compile(scriptable, fullgraph=True)
    with torch.no_grad():
        le, ce, ve = scriptable(*inputs)
        lc, cc, vc = compiled(*inputs)  # must not raise

    assert torch.allclose(le, lc, atol=1e-5), (
        f"fullgraph (native) logits differ; max|diff|={(le - lc).abs().max().item():.2e}"
    )
    assert torch.equal(ce, cc)
    assert torch.allclose(ve, vc, atol=1e-5), (
        f"fullgraph (native) values differ; max|diff|={(ve - vc).abs().max().item():.2e}"
    )


def test_fullgraph_no_graph_breaks_legacy():
    """The relational forward on a legacy graph must compile into a SINGLE
    graph with zero break reasons (was graph_count=4, break_reasons=3 with the
    per-axis-mask + nonzero-split shim).

    Uses the PRODUCTION calling convention: the inference server's
    ``_prepare_tensors`` always passes precomputed ``legal_idx``/``stone_idx``/
    ``stone_batch`` so the policy/value heads take the ``index_select`` path
    (the empty-idx fallback boolean-masks -> ``nonzero``, an orthogonal head
    break present in the non-relational path too, never hit in production)."""
    _, scriptable = _build_pair(seed=2)
    x, edge_index, edge_attr = _legacy_inputs(seed=9)
    legal_mask, stone_mask, batch = _masks()
    legal_idx = torch.where(legal_mask)[0]
    stone_idx = torch.where(stone_mask)[0]
    stone_batch = batch[stone_idx]
    inputs = (
        x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr,
        legal_idx, stone_idx, stone_batch,
    )

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(scriptable)(*inputs)
    assert len(explanation.break_reasons) == 0, (
        f"expected 0 graph breaks, got {len(explanation.break_reasons)}: "
        f"{[br.reason for br in explanation.break_reasons]}"
    )
    assert explanation.graph_count == 1, (
        f"expected a single compiled graph, got {explanation.graph_count}"
    )
