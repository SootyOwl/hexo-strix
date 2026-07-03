"""Smoke test: the self-play inference server loads + runs the D6-invariant
relational model on the LEGACY wire graph it actually receives.

Exercises the real path: argparse flags -> _load_model builds
ScriptableHeXONet(axis_relational=True) -> load_from_hexonet from a lean
HeXONet checkpoint -> forward on a legacy (11-dim x + edge_attr[E,5]) graph,
matching the eager HeXONet. No hexo_rs rebuild; torch.compile disabled.
"""
import argparse

import torch

from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_a0.graph import game_to_axis_graph
from hexo_a0 import inference_server as isv
import hexo_rs as hr


def _lean_cfg():
    return ModelConfig(
        relative_stone_encoding=True, threat_features=True,
        compact_stone_onehot=True, node_coords=False, moves_scope="node",
        axis_relational=True, axis_window=8, num_layers=3, hidden_dim=32,
        num_heads=8, policy_hidden=16, value_hidden=16,
        use_jk=True, jk_mode="cat", graph_type="axis", conv_type="gine",
    )


def _legacy_graph():
    st = hr.GameState(hr.GameConfig(6, 6, 300))
    for q, r in [(1, 0), (0, 1), (2, 0), (-1, 1)]:
        st.apply_move(q, r)
    return game_to_axis_graph(st, prune_empty_edges=True, threat_features=True,
                              relative_stones=True)


def test_inference_server_serves_relational_model(tmp_path):
    torch.manual_seed(0)
    cfg = _lean_cfg()
    eager = HeXONet(cfg).eval()
    ckpt = tmp_path / "lean.pt"
    torch.save({"model_state_dict": eager.state_dict()}, ckpt)

    # Mirrors what the Rust self_play binary passes the server: legacy node_dim
    # (11, relative+threat) on the wire + the lean flags to build the model.
    args = argparse.Namespace(
        checkpoint=str(ckpt), hidden_dim=32, num_layers=3, num_heads=8,
        policy_hidden=16, value_hidden=16, graph_type="axis", conv_type="gine",
        device="cpu", no_compile=True, no_pre_norm=False, node_dim=11,
        use_jk=True, jk_mode="cat", dynamic_compile=False,
        axis_relational=True, axis_window=8, relative_stone_encoding=True,
        threat_features=True, compact_stone_onehot=True, node_coords=False,
        moves_scope="node",
    )
    model = isv._load_model(args)  # strict load fails -> load_from_hexonet

    d = _legacy_graph()
    assert d.x.shape[1] == 11 and d.edge_attr.shape[1] == 5
    n = d.x.shape[0]
    batch = torch.zeros(n, dtype=torch.long)

    with torch.no_grad():
        # Server call convention (see _warmup / model(*tensors)).
        logits, counts, values = model(
            d.x, d.edge_index, d.legal_mask, d.stone_mask, batch, 1, d.edge_attr
        )
        epol, eval_ = eager(
            d.x, d.edge_index, d.legal_mask, stone_mask=d.stone_mask,
            edge_attr=d.edge_attr,
        )

    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    assert int(counts[0]) == int(d.legal_mask.sum())
    assert torch.allclose(logits, epol, atol=1e-4), "server logits != eager"
    assert torch.allclose(values[0], eval_, atol=1e-4), "server value != eager"


def test_inference_server_reads_embedded_config(tmp_path):
    """The REAL self-play path: model_selfplay.pt embeds model_config, and the
    server is launched with LEGACY-default CLI flags — it must auto-configure
    the relational schema from the checkpoint (no self-play CLI plumbing)."""
    torch.manual_seed(0)
    cfg = _lean_cfg()
    eager = HeXONet(cfg).eval()
    mc_dict = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
    ckpt = tmp_path / "lean_emb.pt"
    torch.save({"model_state_dict": eager.state_dict(), "model_config": mc_dict}, ckpt)

    # argparse LEGACY defaults — NO lean flags passed on the CLI.
    args = argparse.Namespace(
        checkpoint=str(ckpt), hidden_dim=32, num_layers=3, num_heads=8,
        policy_hidden=16, value_hidden=16, graph_type="axis", conv_type="gine",
        device="cpu", no_compile=True, no_pre_norm=False, node_dim=11,
        use_jk=True, jk_mode="cat", dynamic_compile=False,
        axis_relational=False, axis_window=8, relative_stone_encoding=False,
        threat_features=False, compact_stone_onehot=False, node_coords=True,
        moves_scope="node",
    )
    model = isv._load_model(args)
    assert args.axis_relational is True, "embedded model_config not applied"

    d = _legacy_graph()
    n = d.x.shape[0]
    batch = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        logits, counts, values = model(
            d.x, d.edge_index, d.legal_mask, d.stone_mask, batch, 1, d.edge_attr
        )
        epol, eval_ = eager(
            d.x, d.edge_index, d.legal_mask, stone_mask=d.stone_mask,
            edge_attr=d.edge_attr,
        )
    assert torch.allclose(logits, epol, atol=1e-4)
    assert torch.allclose(values[0], eval_, atol=1e-4)


def test_inference_server_torch_compile_relational(tmp_path):
    """The server torch.compile's the model (NOT jit.script) with
    fullgraph=True. The relational encoder now unifies all edges into
    fixed-shape scatter buckets (no per-axis masking / nonzero split), so it
    compiles fullgraph — the exact path that forced fullgraph=False (and ~3x
    self-play launch overhead); it must compile, run, and match eager."""
    torch.manual_seed(0)
    cfg = _lean_cfg()
    eager = HeXONet(cfg).eval()
    mc_dict = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
    ckpt = tmp_path / "lean_c.pt"
    torch.save({"model_state_dict": eager.state_dict(), "model_config": mc_dict}, ckpt)

    # compile ON (no_compile=False), legacy CLI defaults (embedded config drives it).
    args = argparse.Namespace(
        checkpoint=str(ckpt), hidden_dim=32, num_layers=3, num_heads=8,
        policy_hidden=16, value_hidden=16, graph_type="axis", conv_type="gine",
        device="cpu", no_compile=False, no_pre_norm=False, node_dim=11,
        use_jk=True, jk_mode="cat", dynamic_compile=False,
        axis_relational=False, axis_window=8, relative_stone_encoding=False,
        threat_features=False, compact_stone_onehot=False, node_coords=True,
        moves_scope="node",
    )
    model = isv._load_model(args)  # must not raise (fullgraph=False for relational)

    d = _legacy_graph()
    n = d.x.shape[0]
    batch = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        logits, counts, values = model(
            d.x, d.edge_index, d.legal_mask, d.stone_mask, batch, 1, d.edge_attr
        )
        epol, eval_ = eager(
            d.x, d.edge_index, d.legal_mask, stone_mask=d.stone_mask,
            edge_attr=d.edge_attr,
        )
    assert torch.allclose(logits, epol, atol=1e-4), "compiled logits != eager"
    assert torch.allclose(values[0], eval_, atol=1e-4), "compiled value != eager"
