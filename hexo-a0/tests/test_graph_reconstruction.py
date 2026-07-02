import queue, threading
import torch
import hexo_rs
from hexo_a0.config import FullConfig
from hexo_a0.graph import graph_fn_from_model_config
from hexo_a0.model import HeXONet
from hexo_a0.replay_buffer import ReplayBuffer
from hexo_a0.self_play import TrainingExample
from hexo_a0.trainer import _prefetch_worker


def _mk_example(game_state, with_graph, model_config):
    policy = torch.ones(len(game_state.legal_moves())) / len(game_state.legal_moves())
    data = graph_fn_from_model_config(model_config)(game_state) if with_graph else None
    return TrainingExample(data=data, policy_target=policy, value_target=1.0,
                           game_state=game_state)


def test_graph_batch_fn_matches_per_example():
    """Batch reconstruction must equal per-example reconstruction, element-wise."""
    import torch, hexo_rs
    from hexo_a0.config import ModelConfig
    from hexo_a0.graph import graph_fn_from_model_config, graph_batch_fn_from_model_config
    cfg = hexo_rs.GameConfig(6, 4, 120)
    states = []
    for k in range(5):
        g = hexo_rs.GameState(cfg)
        import random; rng = random.Random(k)
        for _ in range(10 + k):
            if g.is_terminal(): break
            g.apply_move(*rng.choice(g.legal_moves()))
        states.append(g)
    mc = ModelConfig(graph_type="axis", prune_empty_edges=True)
    single = graph_fn_from_model_config(mc); batch = graph_batch_fn_from_model_config(mc)
    bs = batch(states)
    assert len(bs) == len(states)
    for st, b in zip(states, bs):
        s = single(st)
        assert torch.equal(s.edge_index, b.edge_index)
        assert torch.equal(s.legal_mask, b.legal_mask)
        assert torch.allclose(s.x, b.x, atol=1e-5)
    # empty input -> empty list (prefetch may have all-legacy batches)
    assert batch([]) == [] or len(batch([])) == 0


def test_prefetch_shim_handles_both_forms():
    """Prefetch must reconstruct for new (data=None) AND pass through legacy (data set)."""
    from hexo_a0.graph import graph_batch_fn_from_model_config
    config = FullConfig(); config.model.graph_type = "axis"
    gc = hexo_rs.GameConfig(6, 4, 60)
    g = hexo_rs.GameState(gc)
    for _ in range(8):
        if g.is_terminal():
            break
        g.apply_move(*g.legal_moves()[0])
    graph_batch_fn = graph_batch_fn_from_model_config(config.model)
    buf = ReplayBuffer(100, db_path=":memory:")
    buf.add(_mk_example(g.clone(), with_graph=False, model_config=config.model))  # new
    buf.add(_mk_example(g.clone(), with_graph=True, model_config=config.model))   # legacy

    q, stop = queue.Queue(maxsize=2), threading.Event()
    t = threading.Thread(target=_prefetch_worker,
                         args=(buf, 2, False, q, stop, "cpu", graph_batch_fn), daemon=True)
    t.start()
    batch = q.get(timeout=30); stop.set()
    assert len(batch) == 2
    for data, policy, *_ in batch:
        assert data is not None
        assert policy.shape[0] == int(data.legal_mask.sum())


def test_reconstructed_batch_matches_direct_build():
    """Self-play -> buffer -> reconstruct must yield policy-aligned graphs."""
    import torch
    from hexo_a0.config import FullConfig
    from hexo_a0.graph import graph_fn_from_model_config
    from hexo_a0.replay_buffer import ReplayBuffer
    from hexo_a0.self_play import batched_self_play_games
    import hexo_rs
    config = FullConfig(); config.model.graph_type = "axis"
    gc = hexo_rs.GameConfig(6, 4, 60)
    model = HeXONet(config.model)
    model.eval()
    games = batched_self_play_games(model, config, gc, torch.device("cpu"),
                                    n_games=3, n_simulations_override=8)
    buf = ReplayBuffer(1000, db_path=":memory:")
    for g in games:
        buf.add_many(g)
    gfn = graph_fn_from_model_config(config.model)
    for ex in buf.sample(min(16, len(buf))):
        assert ex.game_state is not None and ex.data is None
        d = gfn(ex.game_state)
        assert ex.policy_target.shape[0] == int(d.legal_mask.sum())
