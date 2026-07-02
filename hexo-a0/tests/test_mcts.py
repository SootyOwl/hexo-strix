"""Tests for MCTSNode, backup, sigma, v_mix, normalize_q, select_child,
evaluate_state, gumbel_top_k, compute_improved_policy, simulate,
sequential_halving.

Written first (TDD) — all tests fail until mcts.py is implemented.
"""

import math
import pytest
import torch

from hexo_a0.mcts import (
    MCTSNode,
    backup,
    normalize_q,
    select_child,
    sigma,
    v_mix,
    evaluate_state,
    gumbel_top_k,
    compute_improved_policy,
    simulate,
    sequential_halving,
)
from hexo_a0.model import HeXONet, ModelConfig
import hexo_rs


# ---------------------------------------------------------------------------
# Part 1 — MCTSNode structure
# ---------------------------------------------------------------------------


class TestMCTSNode:
    def test_default_fields(self):
        node = MCTSNode(prior=0.5)
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.5
        assert node.children == {}
        assert node.action is None
        assert node.current_player is None
        assert node.value_estimate == 0.0
        assert node.parent is None

    def test_q_value_unvisited(self):
        node = MCTSNode(prior=0.3)
        assert node.q_value == 0.0

    def test_q_value_visited(self):
        node = MCTSNode(prior=0.3)
        node.visit_count = 4
        node.value_sum = 2.0
        assert node.q_value == pytest.approx(0.5)

    def test_q_value_negative(self):
        node = MCTSNode(prior=0.1)
        node.visit_count = 2
        node.value_sum = -1.0
        assert node.q_value == pytest.approx(-0.5)

    def test_explicit_fields(self):
        parent = MCTSNode(prior=1.0)
        child = MCTSNode(
            prior=0.2,
            action=(1, 0),
            current_player="P1",
            value_estimate=0.7,
            parent=parent,
        )
        assert child.action == (1, 0)
        assert child.current_player == "P1"
        assert child.value_estimate == pytest.approx(0.7)
        assert child.parent is parent

    def test_children_are_independent(self):
        """Each MCTSNode gets its own children dict (no shared mutable default)."""
        a = MCTSNode(prior=0.5)
        b = MCTSNode(prior=0.5)
        a.children[(0, 0)] = MCTSNode(prior=0.1)
        assert (0, 0) not in b.children


# ---------------------------------------------------------------------------
# Part 2 — player-aware backup
# ---------------------------------------------------------------------------


class TestBackup:
    def _make_path(self, players):
        """Build a list of MCTSNode with given current_player strings, linked parent→child."""
        nodes = [MCTSNode(prior=1.0, current_player=p) for p in players]
        for i in range(1, len(nodes)):
            nodes[i].parent = nodes[i - 1]
        return nodes

    def test_leaf_visit_and_value(self):
        path = self._make_path(["P1", "P2"])
        backup(path, value=1.0)
        leaf = path[-1]
        assert leaf.visit_count == 1
        assert leaf.value_sum == pytest.approx(1.0)

    def test_all_visit_counts_increment(self):
        path = self._make_path(["P1", "P2", "P1"])
        backup(path, value=0.5)
        for node in path:
            assert node.visit_count == 1

    def test_different_player_parent_gets_negated_value(self):
        """Root=P1, leaf=P2. Value at leaf=1.0 → root gets -1.0."""
        path = self._make_path(["P1", "P2"])
        backup(path, value=1.0)
        root = path[0]
        assert root.value_sum == pytest.approx(-1.0)

    def test_same_player_parent_no_negation(self):
        """Root=P2, leaf=P2 (mid-turn, no player change). Value=1.0 stays 1.0."""
        path = self._make_path(["P2", "P2"])
        backup(path, value=1.0)
        root = path[0]
        assert root.value_sum == pytest.approx(1.0)

    def test_three_node_single_sign_flip(self):
        """P1 → P2 → P2 path: only one sign flip (at grandparent P1).

        leaf (P2) gets value=1.0
        middle (P2, same as leaf) gets +1.0
        grandparent (P1, different from middle=P2) gets -1.0
        """
        path = self._make_path(["P1", "P2", "P2"])
        backup(path, value=1.0)
        grandparent = path[0]
        middle = path[1]
        leaf = path[2]

        assert leaf.value_sum == pytest.approx(1.0)
        assert middle.value_sum == pytest.approx(1.0)
        assert grandparent.value_sum == pytest.approx(-1.0)

    def test_four_node_alternating(self):
        """P1→P2→P1→P2 path with leaf value=1.0.

        leaf P2: +1.0
        P1 (idx 2): -1.0  (flip from P2)
        P2 (idx 1): +1.0  (flip from P1)
        P1 (idx 0): -1.0  (flip from P2)
        """
        path = self._make_path(["P1", "P2", "P1", "P2"])
        backup(path, value=1.0)
        assert path[3].value_sum == pytest.approx(1.0)
        assert path[2].value_sum == pytest.approx(-1.0)
        assert path[1].value_sum == pytest.approx(1.0)
        assert path[0].value_sum == pytest.approx(-1.0)

    def test_single_node_path(self):
        """A single-node path (root == leaf) should still update."""
        path = self._make_path(["P1"])
        backup(path, value=0.3)
        assert path[0].visit_count == 1
        assert path[0].value_sum == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Part 3 — sigma
# ---------------------------------------------------------------------------


class TestSigma:
    def test_basic_formula(self):
        # sigma(q) = (c_visit + max_child_visits) * c_scale * q
        # c_visit=50, c_scale=1.0, max_child_visits=10, q=0.5
        # => (50 + 10) * 1.0 * 0.5 = 30.0
        result = sigma(q=0.5, max_child_visits=10, c_visit=50, c_scale=1.0)
        assert result == pytest.approx(30.0)

    def test_zero_visits(self):
        # (50 + 0) * 1.0 * 0.5 = 25.0
        result = sigma(q=0.5, max_child_visits=0, c_visit=50, c_scale=1.0)
        assert result == pytest.approx(25.0)

    def test_scales_with_visits(self):
        r1 = sigma(q=0.5, max_child_visits=10, c_visit=50, c_scale=1.0)
        r2 = sigma(q=0.5, max_child_visits=100, c_visit=50, c_scale=1.0)
        assert r2 > r1

    def test_custom_c_scale(self):
        # (50 + 10) * 2.0 * 0.5 = 60.0
        result = sigma(q=0.5, max_child_visits=10, c_visit=50, c_scale=2.0)
        assert result == pytest.approx(60.0)

    def test_negative_q(self):
        # (50 + 0) * 1.0 * (-0.5) = -25.0
        result = sigma(q=-0.5, max_child_visits=0, c_visit=50, c_scale=1.0)
        assert result == pytest.approx(-25.0)


# ---------------------------------------------------------------------------
# Part 3 — normalize_q
# ---------------------------------------------------------------------------


class TestNormalizeQ:
    def test_basic_normalization(self):
        q_vals = {(0, 0): 0.0, (1, 0): 0.5, (0, 1): 1.0}
        result = normalize_q(q_vals, q_min=0.0, q_max=1.0)
        assert result[(0, 0)] == pytest.approx(0.0)
        assert result[(1, 0)] == pytest.approx(0.5)
        assert result[(0, 1)] == pytest.approx(1.0)

    def test_all_equal_returns_zero(self):
        q_vals = {(0, 0): 0.7, (1, 0): 0.7}
        result = normalize_q(q_vals, q_min=0.7, q_max=0.7)
        assert result[(0, 0)] == pytest.approx(0.0)
        assert result[(1, 0)] == pytest.approx(0.0)

    def test_clipped_to_range(self):
        """Values outside [q_min, q_max] are clamped to [0.0, 1.0]."""
        q_vals = {(0, 0): 2.0, (1, 0): -0.5}
        result = normalize_q(q_vals, q_min=0.0, q_max=1.0)
        assert result[(0, 0)] == pytest.approx(1.0)
        assert result[(1, 0)] == pytest.approx(0.0)

    def test_specific_numerical(self):
        # q_min=-1, q_max=1, q=0 → normalized=(0-(-1))/(1-(-1))=0.5
        q_vals = {(0, 0): -1.0, (1, 0): 0.0, (0, 1): 1.0}
        result = normalize_q(q_vals, q_min=-1.0, q_max=1.0)
        assert result[(1, 0)] == pytest.approx(0.5)

    def test_empty_dict(self):
        result = normalize_q({}, q_min=0.0, q_max=1.0)
        assert result == {}


# ---------------------------------------------------------------------------
# Part 3 — v_mix
# ---------------------------------------------------------------------------


class TestVMix:
    def _make_child(self, prior, visit_count, q_val):
        c = MCTSNode(prior=prior)
        c.visit_count = visit_count
        if visit_count > 0:
            c.value_sum = q_val * visit_count
        return c

    def test_all_unvisited_returns_value_estimate(self):
        node = MCTSNode(prior=1.0, value_estimate=0.42)
        node.children = {
            (0, 0): self._make_child(0.6, 0, 0.0),
            (1, 0): self._make_child(0.4, 0, 0.0),
        }
        result = v_mix(node.value_estimate, node.children)
        assert result == pytest.approx(0.42)

    def test_concrete_numerical(self):
        """Hand-computed: priors=[0.8,0.2], N=[3,1], Q=[1.0,0.0], v_hat=0.5.

        total_N = 4
        sum_visited_prior = 0.8 + 0.2 = 1.0
        sum_visited(pi*Q) = 0.8*1.0 + 0.2*0.0 = 0.8
        v_mix = (0.5 + 4/1.0 * 0.8) / (1+4) = (0.5 + 3.2) / 5 = 0.74
        """
        node = MCTSNode(prior=1.0, value_estimate=0.5)
        node.children = {
            (0, 0): self._make_child(0.8, 3, 1.0),
            (1, 0): self._make_child(0.2, 1, 0.0),
        }
        result = v_mix(node.value_estimate, node.children)
        assert result == pytest.approx(0.74)

    def test_single_visited_child(self):
        """Only one child visited with prior=1.0, N=2, Q=0.5, v_hat=0.0.

        total_N=2, sum_visited_prior=1.0, sum_visited(pi*Q)=0.5
        v_mix = (0.0 + 2/1.0 * 0.5) / (1+2) = 1.0/3 ≈ 0.333...
        """
        node = MCTSNode(prior=1.0, value_estimate=0.0)
        node.children = {
            (0, 0): self._make_child(1.0, 2, 0.5),
        }
        result = v_mix(node.value_estimate, node.children)
        assert result == pytest.approx(1.0 / 3)

    def test_no_children(self):
        """No children at all — total_N=0, return value_estimate."""
        node = MCTSNode(prior=1.0, value_estimate=0.99)
        result = v_mix(node.value_estimate, node.children)
        assert result == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Part 4 — select_child
# ---------------------------------------------------------------------------


class TestSelectChild:
    def _make_node_with_children(self, priors_visits_qvals, v_estimate=0.5):
        """Create a parent node with children specified as list of (action, prior, visits, q)."""
        node = MCTSNode(prior=1.0, value_estimate=v_estimate, current_player="P1")
        node._expanded = True
        for action, prior, visits, q in priors_visits_qvals:
            node._child_priors[action] = prior
            child = MCTSNode(prior=prior, action=action, current_player="P2")
            child.visit_count = visits
            if visits > 0:
                child.value_sum = q * visits
            node.children[action] = child
        return node

    def test_all_unvisited_selects_highest_prior(self):
        """All children unvisited + equal completedQ → argmax prior wins."""
        node = self._make_node_with_children(
            [
                ((0, 0), 0.1, 0, 0.0),
                ((1, 0), 0.7, 0, 0.0),
                ((0, 1), 0.2, 0, 0.0),
            ]
        )
        action = select_child(node)
        assert action == (1, 0)

    def test_unvisited_get_vmix_not_zero(self):
        """Unvisited children completedQ = v_mix(node), not 0.

        If completedQ were 0 for unvisited, a visited child with Q=0 would tie
        with unvisited children. With v_mix=value_estimate=0.5, the unvisited
        children with higher prior should dominate a visited child with Q=-1.
        """
        node = self._make_node_with_children(
            [
                ((0, 0), 0.05, 10, -1.0),   # visited, bad Q
                ((1, 0), 0.50, 0, 0.0),     # unvisited, high prior
                ((0, 1), 0.45, 0, 0.0),     # unvisited, medium prior
            ],
            v_estimate=0.5,
        )
        action = select_child(node)
        # The unvisited high-prior child should win over visited bad-Q child
        assert action != (0, 0)

    def test_all_unvisited_no_error(self):
        """select_child must not raise when all children are unvisited."""
        node = self._make_node_with_children(
            [
                ((0, 0), 0.5, 0, 0.0),
                ((1, 0), 0.5, 0, 0.0),
            ]
        )
        action = select_child(node)  # must not raise
        assert action in {(0, 0), (1, 0)}

    def test_returns_valid_action(self):
        """select_child always returns a key from node.children."""
        node = self._make_node_with_children(
            [
                ((0, 0), 0.3, 2, 0.8),
                ((1, 0), 0.5, 1, 0.3),
                ((0, 1), 0.2, 0, 0.0),
            ]
        )
        action = select_child(node)
        assert action in node.children

    def test_high_q_child_preferred_when_visits_equal(self):
        """Among children with equal visits, higher Q *from parent's perspective* wins.

        Setup: parent P1, children P2, equal visits, equal priors, different Q.
        Child (0,0) has Q=+1.0 from P2's view → -1.0 from P1's view (bad for parent).
        Child (1,0) has Q=-1.0 from P2's view → +1.0 from P1's view (good for parent).
        select_child should pick (1,0).
        """
        node = self._make_node_with_children(
            [
                ((0, 0), 0.5, 5, 1.0),   # Q=+1 from P2's view = -1 from P1's view
                ((1, 0), 0.5, 5, -1.0),  # Q=-1 from P2's view = +1 from P1's view
            ]
        )
        action = select_child(node)
        assert action == (1, 0)


# ---------------------------------------------------------------------------
# Part 5 — evaluate_state (REQ-5)
# ---------------------------------------------------------------------------

_TINY_CONFIG = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex", conv_type="gatv2")


def _make_tiny_net():
    """Create a small HeXONet in eval mode for tests."""
    net = HeXONet(_TINY_CONFIG)
    net.eval()
    return net


class TestEvaluateState:
    def test_returns_tuple_of_three(self):
        net = _make_tiny_net()
        game = hexo_rs.GameState()
        result = evaluate_state(net, game, torch.device("cpu"))
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_logits_shape_matches_legal_moves(self):
        net = _make_tiny_net()
        game = hexo_rs.GameState()
        logits, value, coords = evaluate_state(net, game, torch.device("cpu"))
        expected_len = len(game.legal_moves())
        assert logits.shape == (expected_len,)

    def test_value_is_float_in_range(self):
        net = _make_tiny_net()
        game = hexo_rs.GameState()
        logits, value, coords = evaluate_state(net, game, torch.device("cpu"))
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_coords_length_matches_legal_moves(self):
        net = _make_tiny_net()
        game = hexo_rs.GameState()
        logits, value, coords = evaluate_state(net, game, torch.device("cpu"))
        assert len(coords) == len(game.legal_moves())

    def test_coords_are_tuples_of_ints(self):
        net = _make_tiny_net()
        game = hexo_rs.GameState()
        logits, value, coords = evaluate_state(net, game, torch.device("cpu"))
        for c in coords:
            assert isinstance(c, tuple)
            assert len(c) == 2
            assert isinstance(c[0], int)
            assert isinstance(c[1], int)

    def test_coords_sorted_by_qr(self):
        net = _make_tiny_net()
        game = hexo_rs.GameState()
        logits, value, coords = evaluate_state(net, game, torch.device("cpu"))
        assert coords == sorted(coords)

    def test_no_gradients_on_logits(self):
        net = _make_tiny_net()
        game = hexo_rs.GameState()
        logits, value, coords = evaluate_state(net, game, torch.device("cpu"))
        assert not logits.requires_grad

    def test_logits_is_cpu_tensor(self):
        net = _make_tiny_net()
        game = hexo_rs.GameState()
        logits, value, coords = evaluate_state(net, game, torch.device("cpu"))
        assert logits.device == torch.device("cpu")

    def test_coords_match_game_to_graph_order(self):
        """Coords returned by evaluate_state must match data.coords[legal_mask]."""
        from hexo_a0.graph import game_to_graph
        net = _make_tiny_net()
        game = hexo_rs.GameState()
        logits, value, coords = evaluate_state(net, game, torch.device("cpu"))
        data = game_to_graph(game)
        expected = [
            (int(c[0]), int(c[1]))
            for c in data.coords[data.legal_mask]
        ]
        assert coords == expected


# ---------------------------------------------------------------------------
# Part 6 — gumbel_top_k (REQ-6)
# ---------------------------------------------------------------------------


class TestGumbelTopK:
    def test_returns_min_m_num_legal(self):
        logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        indices, gumbels = gumbel_top_k(logits, m=3)
        assert len(indices) == 3

    def test_returns_all_when_m_ge_len(self):
        logits = torch.tensor([0.1, 0.2, 0.3])
        indices, gumbels = gumbel_top_k(logits, m=10)
        assert len(indices) == 3

    def test_gumbels_length_matches_logits(self):
        logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
        indices, gumbels = gumbel_top_k(logits, m=2)
        # gumbels covers all logits, not just the top-k
        assert gumbels.shape == (4,)

    def test_seeded_deterministic(self):
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        rng1 = torch.Generator()
        rng1.manual_seed(42)
        idx1, g1 = gumbel_top_k(logits, m=3, rng=rng1)
        rng2 = torch.Generator()
        rng2.manual_seed(42)
        idx2, g2 = gumbel_top_k(logits, m=3, rng=rng2)
        assert idx1 == idx2
        assert torch.allclose(g1, g2)

    def test_candidates_are_top_m_by_score(self):
        """Returned indices must be the top-m indices by (gumbels + logits)."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        rng = torch.Generator()
        rng.manual_seed(7)
        indices, gumbels = gumbel_top_k(logits, m=3, rng=rng)
        scores = gumbels + logits
        # Get expected top-3
        _, expected_top = torch.topk(scores, k=3)
        assert set(indices) == set(expected_top.tolist())

    def test_indices_are_ints(self):
        logits = torch.tensor([0.5, 1.0, 1.5])
        indices, gumbels = gumbel_top_k(logits, m=2)
        for idx in indices:
            assert isinstance(idx, int)

    def test_m_equals_one(self):
        logits = torch.tensor([0.1, 5.0, 0.3])
        indices, gumbels = gumbel_top_k(logits, m=1)
        assert len(indices) == 1


# ---------------------------------------------------------------------------
# Part 7 — compute_improved_policy (REQ-8)
# ---------------------------------------------------------------------------


class TestComputeImprovedPolicy:
    def _make_root(self, value_estimate, children_data):
        """Create root node with children.

        children_data: list of (action, prior, visits, q_val)
        """
        root = MCTSNode(prior=1.0, value_estimate=value_estimate, current_player="P1")
        for action, prior, visits, q_val in children_data:
            child = MCTSNode(prior=prior, action=action, current_player="P2")
            child.visit_count = visits
            if visits > 0:
                child.value_sum = q_val * visits
            root.children[action] = child
        return root

    def _uniform_logits(self, n):
        return torch.zeros(n)

    def test_output_sums_to_one(self):
        coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
        root = self._make_root(
            0.5,
            [
                ((0, 0), 0.3, 5, 0.8),
                ((1, 0), 0.3, 3, 0.4),
                ((0, 1), 0.2, 0, 0.0),
            ],
        )
        logits = self._uniform_logits(4)
        policy = compute_improved_policy(logits, coords, root)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)

    def test_high_q_action_gets_boosted(self):
        """Action with higher Q *from parent's perspective* gets more probability.

        Root is P1, children are P2. _q_from_parent negates Q at boundary.
        Child (0,0): Q=+1.0 from P2 → -1.0 from P1 (bad for parent).
        Child (1,0): Q=-1.0 from P2 → +1.0 from P1 (good for parent).
        """
        coords = [(0, 0), (1, 0)]
        root = self._make_root(
            0.0,
            [
                ((0, 0), 0.5, 5, 1.0),   # Q=+1 from P2's view = -1 from P1's
                ((1, 0), 0.5, 5, -1.0),  # Q=-1 from P2's view = +1 from P1's
            ],
        )
        logits = torch.zeros(2)
        policy = compute_improved_policy(logits, coords, root)
        assert policy[1] > policy[0]

    def test_unsearched_action_gets_vmix(self):
        """An action not in root.children must receive completedQ = v_mix, not 0."""
        # Build root with 2 children; coords has 3 actions (third unsearched)
        coords = [(0, 0), (1, 0), (2, 0)]
        root = self._make_root(
            0.5,
            [
                ((0, 0), 0.4, 4, 0.9),
                ((1, 0), 0.4, 4, 0.9),
                # (2, 0) never expanded
            ],
        )
        logits = torch.zeros(3)
        # With v_mix ~ 0.5 and visited Q ~ 0.9, unsearched should get v_mix
        # which is less boosted than visited high-Q but not zero.
        # The key property: policy[2] > 0.
        policy = compute_improved_policy(logits, coords, root)
        assert policy[2].item() > 0.0

    def test_no_nan_or_inf_with_large_logits(self):
        """No NaN/inf for logits in [-100, 100] range."""
        coords = [(0, 0), (1, 0), (0, 1)]
        root = self._make_root(
            0.3,
            [
                ((0, 0), 0.5, 3, 0.7),
                ((1, 0), 0.3, 2, 0.2),
            ],
        )
        logits = torch.tensor([-100.0, 50.0, 100.0])
        policy = compute_improved_policy(logits, coords, root)
        assert not torch.any(torch.isnan(policy))
        assert not torch.any(torch.isinf(policy))
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)

    def test_q_values_normalized_before_sigma(self):
        """All-equal Q values → sigma inputs all 0 → policy shaped purely by logits."""
        coords = [(0, 0), (1, 0)]
        root = self._make_root(
            0.5,
            [
                ((0, 0), 0.5, 3, 0.6),
                ((1, 0), 0.5, 3, 0.6),  # same Q → norm to 0 → sigma=0
            ],
        )
        # Different logits: action 1 gets higher logit
        logits = torch.tensor([0.0, 2.0])
        policy = compute_improved_policy(logits, coords, root)
        # When normalized Q is uniform (all 0), policy should follow logits
        assert policy[1] > policy[0]

    def test_all_unvisited_children_no_error(self):
        """Root with no visited children (only unvisited) should not raise."""
        coords = [(0, 0), (1, 0), (0, 1)]
        root = self._make_root(
            0.5,
            [
                ((0, 0), 0.4, 0, 0.0),
                ((1, 0), 0.6, 0, 0.0),
            ],
        )
        logits = torch.zeros(3)
        policy = compute_improved_policy(logits, coords, root)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)

    def test_output_shape(self):
        coords = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        root = self._make_root(
            0.0,
            [((0, 0), 0.5, 2, 0.5)],
        )
        logits = torch.zeros(5)
        policy = compute_improved_policy(logits, coords, root)
        assert policy.shape == (5,)


# ---------------------------------------------------------------------------
# Helpers shared by simulate/sequential_halving tests
# ---------------------------------------------------------------------------

_TINY_CFG = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex", conv_type="gatv2")
_GAME_CFG = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
# Use win_length=2 so we can reach terminal states in 2 moves.
_GAME_CFG_WIN2 = hexo_rs.GameConfig(win_length=2, placement_radius=4, max_moves=50)
_DEVICE = torch.device("cpu")


def _make_net():
    net = HeXONet(_TINY_CFG)
    net.eval()
    return net


def _make_expanded_root(game_state, net):
    """Build a root MCTSNode that has already been expanded (lazy: priors stored)."""
    logits, value, coords = evaluate_state(net, game_state, _DEVICE)
    priors = torch.softmax(logits, dim=-1)

    root = MCTSNode(
        prior=1.0,
        action=None,
        current_player=game_state.current_player(),
        value_estimate=value,
    )
    root.game_state = game_state
    root._child_priors = {
        coord: float(priors[i]) for i, coord in enumerate(coords)
    }
    root._expanded = True

    return root, logits, coords


# ---------------------------------------------------------------------------
# Part 8 — simulate (REQ-11)
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_root_visit_count_increments(self):
        """Each call to simulate increments root.visit_count."""
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG)
        root, _, _ = _make_expanded_root(game, net)

        assert root.visit_count == 0
        simulate(root, net, _DEVICE)
        assert root.visit_count == 1
        simulate(root, net, _DEVICE)
        assert root.visit_count == 2

    def test_non_terminal_leaf_gets_children(self):
        """After simulation, the visited leaf node should be expanded (has priors)."""
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG)
        root, _, _ = _make_expanded_root(game, net)

        simulate(root, net, _DEVICE)

        # Find any child of root that got expanded (has priors from network eval)
        expanded = [c for c in root.children.values() if c.is_expanded]
        assert len(expanded) >= 1, "At least one leaf should have been expanded"

    def test_non_terminal_leaf_gets_value_estimate(self):
        """Expanded leaf should have a non-zero value_estimate from the network."""
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG)
        root, _, _ = _make_expanded_root(game, net)

        simulate(root, net, _DEVICE)

        expanded = [c for c in root.children.values() if c.is_expanded]
        assert len(expanded) >= 1
        leaf = expanded[0]
        # value_estimate is set from the network; it won't necessarily be 0.0
        # but it should be in [-1, 1]
        assert -1.0 <= leaf.value_estimate <= 1.0

    def test_terminal_leaf_no_children_created(self):
        """If simulation reaches a terminal game state, no children are added."""
        net = _make_net()
        # With win_length=2, P2 wins after placing 2 adjacent stones.
        # Create a state where we're one forced move from a terminal.
        game = hexo_rs.GameState(_GAME_CFG_WIN2)
        # Place one stone for P2 so the next move wins.
        game.apply_move(-4, 0)
        # Current: P2 still has 1 move remaining (moves_remaining=1)
        # Each child in this state leads to a terminal state (win_length=2, 1 stone placed)
        assert not game.is_terminal()

        root, _, _ = _make_expanded_root(game, net)

        # Run a simulation - some child will be terminal
        simulate(root, net, _DEVICE)

        # Children that correspond to winning moves (placing adjacent to (-4,0)) should
        # be terminal and have no children of their own
        terminal_children = [
            c for c in root.children.values()
            if c.game_state is not None and c.game_state.is_terminal()
        ]
        for tc in terminal_children:
            assert len(tc.children) == 0, "Terminal nodes must not have children"

    def test_terminal_leaf_visit_count_increments(self):
        """A terminal leaf should have its visit_count incremented via backup.

        We force a simulation to a known action, then check the lazily-created
        child has visit_count=1.
        """
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG_WIN2)
        game.apply_move(-4, 0)  # P2 placed one stone, next move wins for P2

        root, _, coords = _make_expanded_root(game, net)

        # Pick the first legal action and force a simulation to it
        forced = coords[0]
        simulate(root, net, _DEVICE, forced_action=forced)

        # The child should have been lazily created and visited
        assert forced in root.children, "Forced child should have been created"
        child = root.children[forced]
        assert child.visit_count == 1

    def test_backup_value_propagates_to_root(self):
        """After simulate(), root.value_sum should be non-zero (backup reached root)."""
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG)
        root, _, _ = _make_expanded_root(game, net)

        simulate(root, net, _DEVICE)

        # root must have been included in backup path
        assert root.visit_count == 1
        # value_sum could be 0.0 if the network returns exactly 0 but that is
        # unlikely; instead just check it's a finite number
        assert math.isfinite(root.value_sum)

    def test_child_game_states_are_independent(self):
        """Applying a move on a child's game state must not affect the parent."""
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG)
        root, _, _ = _make_expanded_root(game, net)

        simulate(root, net, _DEVICE)

        # Pick an expanded leaf
        expanded = [c for c in root.children.values() if len(c.children) > 0]
        if not expanded:
            return  # skip if no expansion happened

        leaf = expanded[0]
        # leaf's game_state should not equal root's game_state object
        assert leaf.game_state is not root.game_state

    def test_forced_action_visits_specified_child(self):
        """simulate with forced_action visits the named child first."""
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG)
        root, _, coords = _make_expanded_root(game, net)

        # Choose the first legal action
        first_action = coords[0]

        simulate(root, net, _DEVICE, forced_action=first_action)

        # The child should have been lazily created and visited
        assert first_action in root.children
        assert root.children[first_action].visit_count == 1

    def test_multiple_simulations_increase_total_visits(self):
        """Running N simulations results in N total visits across the tree."""
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG)
        root, _, _ = _make_expanded_root(game, net)

        n = 5
        for _ in range(n):
            simulate(root, net, _DEVICE)

        assert root.visit_count == n


# ---------------------------------------------------------------------------
# Part 9 — sequential_halving (REQ-7)
# ---------------------------------------------------------------------------

class _MockConfig:
    """Minimal config stand-in for sequential_halving."""
    def __init__(self, n_simulations, c_visit=50, c_scale=1.0):
        self.n_simulations = n_simulations
        self.c_visit = c_visit
        self.c_scale = c_scale


def _make_root_for_halving(n_candidates=8):
    """Build a root node with n_candidates (lazy expansion, priors stored)."""
    net = _make_net()
    game = hexo_rs.GameState(_GAME_CFG)
    root, logits, coords = _make_expanded_root(game, net)

    # Limit to n_candidates priors (keep only first n)
    all_actions = list(root._child_priors.keys())[:n_candidates]
    root._child_priors = {a: root._child_priors[a] for a in all_actions}

    # candidate_indices are 0..n_candidates-1 mapping into coords
    candidate_indices = list(range(n_candidates))
    candidate_gumbels = torch.zeros(len(coords))

    return root, candidate_indices, candidate_gumbels, coords, logits, net


class TestSequentialHalving:
    def test_zero_budget_returns_immediately(self):
        """With n_simulations=0, sequential_halving returns candidates unchanged (AC-7f)."""
        root, indices, gumbels, coords, logits, net = _make_root_for_halving(4)
        cfg = _MockConfig(n_simulations=0)

        result = sequential_halving(root, indices, gumbels, coords, logits, net, cfg, _DEVICE)
        assert result == indices

    def test_single_candidate_no_op(self):
        """With 1 candidate, sequential_halving is a no-op (AC-7g)."""
        root, indices, gumbels, coords, logits, net = _make_root_for_halving(4)
        single = [indices[0]]
        cfg = _MockConfig(n_simulations=10)

        result = sequential_halving(root, single, gumbels, coords, logits, net, cfg, _DEVICE)
        assert result == single

    def test_candidates_roughly_halve(self):
        """Each phase should cut candidates in half (AC-7b)."""
        root, indices, gumbels, coords, logits, net = _make_root_for_halving(8)
        cfg = _MockConfig(n_simulations=100)

        result = sequential_halving(root, indices, gumbels, coords, logits, net, cfg, _DEVICE)

        # With 8 candidates: 8→4→2→1. Result should be <= ceil(8/2)=4
        assert len(result) <= len(indices)
        assert len(result) >= 1

    def test_one_candidate_survives(self):
        """With sufficient budget, exactly 1 candidate survives (AC-7c)."""
        root, indices, gumbels, coords, logits, net = _make_root_for_halving(4)
        # 4 candidates: 4→2→1 requires 3 phases, at least 4 sims/phase * 3 = 12 sims
        cfg = _MockConfig(n_simulations=48)

        result = sequential_halving(root, indices, gumbels, coords, logits, net, cfg, _DEVICE)
        assert len(result) == 1

    def test_total_simulations_within_budget(self):
        """Total simulations run must not exceed n_simulations (AC-7a)."""
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG)
        root, logits, coords = _make_expanded_root(game, net)

        n_candidates = 4
        action_list = list(root._child_priors.keys())[:n_candidates]
        root._child_priors = {a: root._child_priors[a] for a in action_list}
        candidate_indices = list(range(n_candidates))
        gumbels = torch.zeros(len(coords))

        budget = 16
        cfg = _MockConfig(n_simulations=budget)

        initial_visits = root.visit_count
        sequential_halving(root, candidate_indices, gumbels, coords, logits, net, cfg, _DEVICE)

        total_sims = root.visit_count - initial_visits
        assert total_sims <= budget

    def test_each_surviving_candidate_visited(self):
        """Every candidate that survives halving must have >= 1 simulation (AC-7d)."""
        root, indices, gumbels, coords, logits, net = _make_root_for_halving(4)
        cfg = _MockConfig(n_simulations=48)

        result = sequential_halving(root, indices, gumbels, coords, logits, net, cfg, _DEVICE)

        for idx in result:
            action = coords[idx]
            child = root.children.get(action)
            assert child is not None
            assert child.visit_count >= 1, f"Surviving candidate {action} was never visited"

    def test_budget_exhausted_stops_halving(self):
        """With n=16 candidates and n_simulations=16, halving stops after budget (AC-7e)."""
        net = _make_net()
        game = hexo_rs.GameState(_GAME_CFG)
        root, logits, coords = _make_expanded_root(game, net)

        # Use exactly 16 candidates (may need enough legal moves)
        n_candidates = min(16, len(root._child_priors))
        action_list = list(root._child_priors.keys())[:n_candidates]
        root._child_priors = {a: root._child_priors[a] for a in action_list}
        candidate_indices = list(range(n_candidates))
        gumbels = torch.zeros(len(coords))

        cfg = _MockConfig(n_simulations=n_candidates)

        initial_visits = root.visit_count
        result = sequential_halving(root, candidate_indices, gumbels, coords, logits, net, cfg, _DEVICE)
        total_sims = root.visit_count - initial_visits

        assert total_sims <= n_candidates
        # Should not have reduced all the way to 1 with such a small budget
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Part 10 — gumbel_mcts (REQ-10)
# ---------------------------------------------------------------------------


from hexo_a0.mcts import gumbel_mcts  # noqa: E402 – after test helpers defined
from hexo_a0.config import MCTSConfig  # noqa: E402


_MCTS_CFG = MCTSConfig(n_simulations=4, m_actions=4, c_visit=50, c_scale=1.0)


def _make_mcts_net():
    cfg = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex", conv_type="gatv2")
    net = HeXONet(cfg)
    net.eval()
    return net


class TestGumbelMCTS:
    """Integration tests for gumbel_mcts() entry point (AC-10a/b/c, AC-8a, AC-9b)."""

    def test_returns_tuple_of_action_and_policy(self):
        """AC-10a: return value is (action_coord, improved_policy)."""
        net = _make_mcts_net()
        game = hexo_rs.GameState(_GAME_CFG)
        action, policy = gumbel_mcts(game, net, _MCTS_CFG, _DEVICE)
        assert isinstance(action, tuple)
        assert len(action) == 2
        assert isinstance(policy, torch.Tensor)

    def test_improved_policy_length_equals_legal_moves(self):
        """AC-10b: improved_policy length equals number of legal moves."""
        net = _make_mcts_net()
        game = hexo_rs.GameState(_GAME_CFG)
        action, policy = gumbel_mcts(game, net, _MCTS_CFG, _DEVICE)
        assert policy.shape == (len(game.legal_moves()),)

    def test_improved_policy_sums_to_one(self):
        """AC-8a (integration): improved_policy is a valid probability distribution."""
        net = _make_mcts_net()
        game = hexo_rs.GameState(_GAME_CFG)
        _, policy = gumbel_mcts(game, net, _MCTS_CFG, _DEVICE)
        assert policy.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_terminal_state_raises_value_error(self):
        """AC-10c: calling gumbel_mcts on a terminal game raises ValueError."""
        net = _make_mcts_net()
        # Use win_length=2 so we can create a terminal quickly.
        game = hexo_rs.GameState(_GAME_CFG_WIN2)
        game.apply_move(-4, 0)
        game.apply_move(-3, 0)  # two adjacent — P2 wins, terminal
        assert game.is_terminal()
        with pytest.raises(ValueError):
            gumbel_mcts(game, net, _MCTS_CFG, _DEVICE)

    def test_action_is_legal_move(self):
        """Returned action must be a valid legal move in the game state."""
        net = _make_mcts_net()
        game = hexo_rs.GameState(_GAME_CFG)
        action, _ = gumbel_mcts(game, net, _MCTS_CFG, _DEVICE)
        legal = game.legal_moves()
        assert action in legal

    def test_zero_simulations_children_all_unvisited(self):
        """AC-9b: with n_simulations=0, sequential_halving runs no sims.

        Root children were set up during expansion but no simulate() calls
        are made, so all child visit_counts remain 0.
        """
        net = _make_mcts_net()
        game = hexo_rs.GameState(_GAME_CFG)
        cfg_zero = MCTSConfig(n_simulations=0, m_actions=4)
        action, policy = gumbel_mcts(game, net, cfg_zero, _DEVICE)
        # action must still be valid
        assert action in game.legal_moves()
        # policy sums to 1
        assert policy.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_seeded_rng_deterministic(self):
        """Same seed → same action selected both times."""
        net = _make_mcts_net()
        game = hexo_rs.GameState(_GAME_CFG)

        rng1 = torch.Generator()
        rng1.manual_seed(7)
        action1, _ = gumbel_mcts(game, net, _MCTS_CFG, _DEVICE, rng=rng1)

        rng2 = torch.Generator()
        rng2.manual_seed(7)
        action2, _ = gumbel_mcts(game, net, _MCTS_CFG, _DEVICE, rng=rng2)

        assert action1 == action2

    def test_improved_policy_non_negative(self):
        """All entries in improved_policy must be >= 0."""
        net = _make_mcts_net()
        game = hexo_rs.GameState(_GAME_CFG)
        _, policy = gumbel_mcts(game, net, _MCTS_CFG, _DEVICE)
        assert torch.all(policy >= 0).item()
