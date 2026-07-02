"""Parity: the bytes-collated axis eval_fn must match the legacy PyG path.

The fast path (graph.axis_states_to_batch -> _forward_batch_core) and the
legacy path (per-graph game_to_axis_graph -> Batch.from_data_list ->
forward_batch) must produce the same logits and values for the same states,
for every supported node-feature schema.
"""

import pytest
import torch

from hexo_a0.config import ModelConfig
from hexo_a0.evaluate import make_eval_fn
from hexo_a0.graph import game_to_axis_graph
from hexo_a0.model import HeXONet


def _mid_game_states(n: int = 6):
    import hexo_rs

    gc = hexo_rs.GameConfig(win_length=6, placement_radius=4, max_moves=200)
    gs = hexo_rs.GameState(gc)
    # Origin is pre-seeded; play a fixed legal sequence (two stones per turn).
    moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (2, 0), (1, 1), (-2, 0), (-1, -1),
             (3, 0), (2, 1), (-3, 0), (-2, -1)]
    states = []
    for q, r in moves:
        states.append(gs.clone())
        gs.apply_move(q, r)
    return states[-n:]


@pytest.mark.parametrize("threat,rel", [(False, False), (True, False), (True, True)])
def test_axis_eval_fn_matches_legacy(threat, rel):
    torch.manual_seed(0)
    mc = ModelConfig(
        hidden_dim=16, num_layers=2, conv_type="gine", graph_type="axis",
        policy_hidden=16, value_hidden=8, prune_empty_edges=True,
        threat_features=threat, relative_stone_encoding=rel,
    )
    model = HeXONet(mc)
    model.eval()
    states = _mid_game_states()

    fast = make_eval_fn(
        model, "cpu", graph_type="axis", prune_empty_edges=True,
        threat_features=threat, relative_stones=rel, graph_fn=None,
    )
    graph_fn = lambda g: game_to_axis_graph(
        g, prune_empty_edges=True, threat_features=threat, relative_stones=rel)
    legacy = make_eval_fn(
        model, "cpu", graph_type="hex-fallback", prune_empty_edges=True,
        threat_features=threat, relative_stones=rel, graph_fn=graph_fn,
    )

    fast_logits, fast_values = fast(states)
    legacy_logits, legacy_values = legacy(states)

    assert len(fast_logits) == len(legacy_logits) == len(states)
    for fl, ll in zip(fast_logits, legacy_logits):
        assert fl == pytest.approx(ll, abs=1e-5)
    assert fast_values == pytest.approx(legacy_values, abs=1e-5)
