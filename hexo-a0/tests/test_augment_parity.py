"""Parity gate for batched augmented byte-buffer reconstruction.

Asserts the NEW single-Rust-call augmentation path (parsed into a
SimpleNamespace) is numerically identical to the OLD per-`Data`
`random_augment` path when both are forced to the SAME D6 transform.

Coverage (the acceptance contract for the live training loss path):
  - all four threat/relative encoding combos (the threat-dim offset differs
    between the absolute 8/12-dim and relative 7/11-dim layouts, so the two
    "mixed" combos are exercised independently),
  - both prune_empty_edges settings (prune changes the edge-walk),
  - every non-identity D6 transform (1..11), so a transform-specific
    axis-perm/sign bug cannot slip through,
  - structural tensors (x / edge_index / edge_attr / legal_mask), the
    reordered policy, AND the model's policy-KL loss + value-head outputs
    (the latter exercises stone_idx / stone_batch, which the structural
    tensor asserts do not touch).
"""

import queue
import random
import threading

import pytest
import torch
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import (
    augment_axis_states_to_batch,
    game_to_axis_graph_batch,
    graph_fn_from_model_config,
    random_augment,
)
from hexo_a0.model import HeXONet
from hexo_a0.self_play import TrainingExample
from hexo_a0.trainer import _AUGMENT_FALLBACK_MAX_STREAK, _prefetch_worker


def _mk_states(k_states: int, seed: int = 0) -> list:
    """Build K non-terminal mid-game axis GameStates."""
    cfg = hexo_rs.GameConfig(6, 4, 120)
    states = []
    for k in range(k_states):
        g = hexo_rs.GameState(cfg)
        rng = random.Random(seed * 100 + k)
        moves = 10 + k
        for _ in range(moves):
            if g.is_terminal():
                break
            g.apply_move(*rng.choice(g.legal_moves()))
        # Ensure non-terminal with legal moves.
        assert not g.is_terminal()
        assert len(g.legal_moves()) > 0
        states.append(g)
    return states


def _fixed_policies(states: list, seed: int = 1234) -> list:
    """One random-but-fixed normalised policy per state (length = legal count)."""
    rng = torch.Generator().manual_seed(seed)
    policies = []
    for st in states:
        n = len(st.legal_moves())
        p = torch.rand(n, generator=rng) + 1e-3
        p = p / p.sum()
        policies.append(p)
    return policies


def _policy_kl_and_value(model, batch_ns, legal_idx, stone_idx, stone_batch, target_policies):
    """Policy-KL loss + value outputs of the model, mirroring the trainer.

    Returns ``(kl_mean, values)`` so callers can assert parity on BOTH the
    policy head (legal-node ordering) and the value head (stone_idx /
    stone_batch correctness) — the latter is otherwise untouched by the
    structural tensor asserts.
    """
    all_logits, legal_counts, values = model._forward_batch_core(
        batch_ns, legal_idx=legal_idx, stone_idx=stone_idx, stone_batch=stone_batch,
    )
    n = legal_counts.shape[0]
    graph_idx = torch.arange(n).repeat_interleave(legal_counts)
    logits_f = all_logits.float()
    max_per_graph = torch.zeros(n).scatter_reduce(
        0, graph_idx, logits_f, reduce="amax", include_self=False,
    )
    shifted = logits_f - max_per_graph[graph_idx]
    exp_shifted = shifted.exp()
    sum_exp = torch.zeros(n)
    sum_exp.scatter_add_(0, graph_idx, exp_shifted)
    log_q = shifted - sum_exp[graph_idx].log()

    targets = torch.cat(target_policies).float()
    assert targets.shape[0] == logits_f.shape[0]
    ce = -(targets * log_q)
    ent = torch.special.xlogy(targets, targets)
    kl = torch.zeros(n)
    kl.scatter_add_(0, graph_idx, ce + ent)
    return kl.mean(), values


@pytest.mark.parametrize("prune", [True, False], ids=["prune", "noprune"])
@pytest.mark.parametrize(
    "threat,rel",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["plain", "threat", "rel", "threat_rel"],
)
@pytest.mark.parametrize("transform_k", list(range(1, 12)))
def test_augment_byte_buffer_parity(threat, rel, prune, transform_k):
    torch.manual_seed(0)
    K = 6

    states = _mk_states(K, seed=7)
    policies = _fixed_policies(states, seed=42)

    mc = ModelConfig(
        graph_type="axis",
        prune_empty_edges=prune,
        threat_features=threat,
        relative_stone_encoding=rel,
    )

    # ---- NEW path: single Rust call + permutation reorder ----
    batch_ns, aux, perms = augment_axis_states_to_batch(
        states,
        [transform_k] * K,
        prune_empty_edges=prune,
        threat_features=threat,
        relative_stones=rel,
    )
    # permutation[new_legal_idx] = old_legal_idx => new_policy = old_policy[perm]
    new_policies = [
        policies[i][torch.tensor(perms[i], dtype=torch.long)] for i in range(K)
    ]

    # ---- OLD path: per-Data random_augment forced to same transform ----
    graph_fn = graph_fn_from_model_config(mc)
    old_datas = []
    old_policies = []
    for st, pol in zip(states, policies):
        g = graph_fn(st)
        d, rp = random_augment(g, pol, transform_index=transform_k)
        old_datas.append(d)
        old_policies.append(rp)
    old_batch = Batch.from_data_list(old_datas)

    # ---- Structural parity ----
    assert torch.allclose(batch_ns.x, old_batch.x, atol=1e-5), (
        "x mismatch between new and old augmentation paths"
    )

    # edge_index + edge_attr together: pair each edge with its attr row and sort
    # by (src, dst, *attr) so two edges sharing a (src,dst) but on different
    # axes (distinct attr) stay distinguishable — comparing the sorted lists.
    def _edges_with_attr_sorted(ei, ea):
        keys = [tuple(e) + tuple(round(v, 5) for v in a)
                for e, a in zip(ei.t().tolist(), ea.tolist())]
        order = sorted(range(len(keys)), key=lambda j: keys[j])
        return ei.t()[order], ea[order]

    new_ei_s, new_ea_s = _edges_with_attr_sorted(batch_ns.edge_index, batch_ns.edge_attr)
    old_ei_s, old_ea_s = _edges_with_attr_sorted(old_batch.edge_index, old_batch.edge_attr)
    assert torch.equal(new_ei_s, old_ei_s), "edge_index mismatch"
    assert torch.allclose(new_ea_s, old_ea_s, atol=1e-5), "edge_attr mismatch"

    assert torch.equal(batch_ns.legal_mask, old_batch.legal_mask), "legal_mask mismatch"

    # ---- Policy parity (concatenated, in state order) ----
    new_pol_cat = torch.cat(new_policies)
    old_pol_cat = torch.cat(old_policies)
    pol_diff = (new_pol_cat - old_pol_cat).abs().max().item()
    assert torch.allclose(new_pol_cat, old_pol_cat, atol=1e-6), (
        f"reordered policy mismatch (max abs diff {pol_diff})"
    )

    # ---- The real guard: policy-KL loss AND value head through the model ----
    torch.manual_seed(123)
    model = HeXONet(mc)
    model.eval()

    with torch.no_grad():
        new_loss, new_values = _policy_kl_and_value(
            model, batch_ns, aux.legal_idx, aux.stone_idx, aux.stone_batch,
            new_policies,
        )

    # Old path: build index tensors the way _forward_and_loss does.
    old_legal_idx = old_batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
    old_stone_mask = old_batch.stone_mask
    old_stone_idx = old_stone_mask.nonzero(as_tuple=False).squeeze(1)
    old_stone_batch = old_batch.batch[old_stone_idx]
    with torch.no_grad():
        old_loss, old_values = _policy_kl_and_value(
            model, old_batch, old_legal_idx, old_stone_idx, old_stone_batch,
            old_policies,
        )

    loss_diff = (new_loss - old_loss).abs().item()
    assert torch.allclose(new_loss, old_loss, atol=1e-5), (
        f"policy-KL loss mismatch: new={new_loss.item()} old={old_loss.item()} "
        f"diff={loss_diff}"
    )
    # Value head parity (exercises stone_idx / stone_batch pooling).
    assert torch.allclose(new_values, old_values, atol=1e-5), (
        f"value head mismatch (max abs diff "
        f"{(new_values - old_values).abs().max().item()})"
    )


def _mk_state_with_moves(n_moves: int, seed: int):
    """Build one non-terminal axis GameState with ~n_moves random plies."""
    cfg = hexo_rs.GameConfig(6, 4, 120)
    g = hexo_rs.GameState(cfg)
    rng = random.Random(seed)
    for _ in range(n_moves):
        if g.is_terminal():
            break
        g.apply_move(*rng.choice(g.legal_moves()))
    assert not g.is_terminal()
    assert len(g.legal_moves()) > 0
    return g


# ---------------------------------------------------------------------------
# FIX T2: heterogeneous batch — a near-empty state alongside a dense state
# stresses the multi-graph index offsetting (legal_idx / stone_idx /
# stone_batch) far harder than K similarly-sized graphs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "threat,rel",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["plain", "threat", "rel", "threat_rel"],
)
def test_augment_byte_buffer_parity_heterogeneous(threat, rel):
    transform_k = 5
    prune = False

    # 1-stone near-empty state + a 30+ stone dense state (very different sizes).
    near_empty = _mk_state_with_moves(1, seed=11)
    dense = _mk_state_with_moves(34, seed=99)
    states = [near_empty, dense]
    assert len(dense.placed_stones()) >= 30
    K = len(states)
    policies = _fixed_policies(states, seed=77)

    mc = ModelConfig(
        graph_type="axis",
        prune_empty_edges=prune,
        threat_features=threat,
        relative_stone_encoding=rel,
    )

    batch_ns, aux, perms = augment_axis_states_to_batch(
        states,
        [transform_k] * K,
        prune_empty_edges=prune,
        threat_features=threat,
        relative_stones=rel,
    )
    new_policies = [
        policies[i][torch.tensor(perms[i], dtype=torch.long)] for i in range(K)
    ]

    graph_fn = graph_fn_from_model_config(mc)
    old_datas, old_policies = [], []
    for st, pol in zip(states, policies):
        g = graph_fn(st)
        d, rp = random_augment(g, pol, transform_index=transform_k)
        old_datas.append(d)
        old_policies.append(rp)
    old_batch = Batch.from_data_list(old_datas)

    # legal_idx / stone_idx / stone_batch offsetting parity.
    new_legal_idx = batch_ns.legal_mask.nonzero(as_tuple=False).squeeze(1)
    old_legal_idx = old_batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
    assert torch.equal(new_legal_idx, old_legal_idx), "legal_idx mismatch"
    assert torch.equal(aux.legal_idx, old_legal_idx), "aux.legal_idx mismatch"

    old_stone_idx = old_batch.stone_mask.nonzero(as_tuple=False).squeeze(1)
    old_stone_batch = old_batch.batch[old_stone_idx]
    assert torch.equal(aux.stone_idx, old_stone_idx), "stone_idx mismatch"
    assert torch.equal(aux.stone_batch, old_stone_batch), "stone_batch mismatch"

    # Structural parity on the heterogeneous collated batch (a mis-offset
    # edge_index between the differently-sized graphs would surface here even
    # if it happened not to perturb the final loss).
    assert torch.allclose(batch_ns.x, old_batch.x, atol=1e-5), "x mismatch"

    def _edges_with_attr_sorted(ei, ea):
        keys = [tuple(e) + tuple(round(v, 5) for v in a)
                for e, a in zip(ei.t().tolist(), ea.tolist())]
        order = sorted(range(len(keys)), key=lambda j: keys[j])
        return ei.t()[order], ea[order]

    new_ei_s, new_ea_s = _edges_with_attr_sorted(batch_ns.edge_index, batch_ns.edge_attr)
    old_ei_s, old_ea_s = _edges_with_attr_sorted(old_batch.edge_index, old_batch.edge_attr)
    assert torch.equal(new_ei_s, old_ei_s), "edge_index mismatch"
    assert torch.allclose(new_ea_s, old_ea_s, atol=1e-5), "edge_attr mismatch"
    assert torch.equal(batch_ns.legal_mask, old_batch.legal_mask), "legal_mask mismatch"

    # Model loss + value head parity.
    torch.manual_seed(123)
    model = HeXONet(mc)
    model.eval()
    with torch.no_grad():
        new_loss, new_values = _policy_kl_and_value(
            model, batch_ns, aux.legal_idx, aux.stone_idx, aux.stone_batch,
            new_policies,
        )
        old_loss, old_values = _policy_kl_and_value(
            model, old_batch, old_legal_idx, old_stone_idx, old_stone_batch,
            old_policies,
        )
    assert torch.allclose(new_loss, old_loss, atol=1e-5), (
        f"policy-KL loss mismatch: new={new_loss.item()} old={old_loss.item()}"
    )
    assert torch.allclose(new_values, old_values, atol=1e-5), "value head mismatch"


# ---------------------------------------------------------------------------
# FIX W2: the prefetch worker's batched-augment fast path falls back per-batch
# on failure, but a PERSISTENT failure must surface loudly. After
# _AUGMENT_FALLBACK_MAX_STREAK consecutive failures the worker RAISES; a single
# transient failure falls back cleanly without raising.
# ---------------------------------------------------------------------------

class _FakeBuffer:
    """Yields a fixed batch of game_state-bearing TrainingExamples."""

    def __init__(self, examples):
        self._examples = examples

    def sample(self, batch_size):
        return list(self._examples)


def _state_examples(k=2):
    states = _mk_states(k, seed=3)
    policies = _fixed_policies(states, seed=5)
    return [
        TrainingExample(policy_target=p, value_target=0.0, game_state=s)
        for s, p in zip(states, policies)
    ]


def test_prefetch_augment_persistent_failure_raises():
    examples = _state_examples(2)
    buf = _FakeBuffer(examples)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def always_raises(states, tindices):
        raise RuntimeError("simulated persistent augment bug")

    graph_batch_fn = lambda gs: game_to_axis_graph_batch(gs)

    with pytest.raises(RuntimeError, match="consecutively"):
        _prefetch_worker(
            buf, batch_size=2, augment_enabled=True, q=q, stop_event=stop,
            device_type="cpu", graph_batch_fn=graph_batch_fn,
            augment_fn=always_raises,
        )


def test_augment_fallback_escalate_increments_then_raises():
    # The shared escalation helper used by BOTH the prefetch worker and the
    # inline (reanalyze) path: increments below the cap, raises at the cap.
    from hexo_a0.trainer import (
        _augment_fallback_escalate,
        _AUGMENT_FALLBACK_MAX_STREAK,
    )

    streak = 0
    for expected in range(1, _AUGMENT_FALLBACK_MAX_STREAK):
        streak = _augment_fallback_escalate(streak, ValueError("boom"), "Test")
        assert streak == expected
    # The cap-th consecutive failure escalates to a raise.
    with pytest.raises(RuntimeError, match="consecutively"):
        _augment_fallback_escalate(streak, ValueError("boom"), "Test")


def test_prefetch_augment_single_transient_failure_falls_back():
    examples = _state_examples(2)
    buf = _FakeBuffer(examples)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    calls = {"n": 0}

    def fail_once(states, tindices):
        calls["n"] += 1
        # One iteration only: setting stop means the loop exits after this
        # batch is fully processed via the fallback path.
        stop.set()
        raise RuntimeError("transient")

    # Real fallback machinery (graph_batch_fn + random_augment) must succeed —
    # if it errored, the worker would PROPAGATE (the body is not try-wrapped),
    # so a clean return proves the single-failure fallback worked.
    graph_batch_fn = lambda gs: game_to_axis_graph_batch(gs)

    _prefetch_worker(
        buf, batch_size=2, augment_enabled=True, q=q, stop_event=stop,
        device_type="cpu", graph_batch_fn=graph_batch_fn, augment_fn=fail_once,
    )
    # A single transient failure does NOT raise; the fallback path ran once.
    assert calls["n"] == 1
    assert _AUGMENT_FALLBACK_MAX_STREAK > 1
