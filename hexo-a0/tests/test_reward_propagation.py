"""Tests for reward propagation correctness across moves and turns.

HeXO has 2-placement turns: each player places two stones before the turn
passes. This means the MCTS tree alternates between same-player nodes
(mid-turn) and cross-player nodes (turn boundary). Rewards must:

  - NOT flip sign between same-player mid-turn nodes
  - Flip sign at turn boundaries (different players)
  - Correctly identify winning moves as positive from the winner's perspective
  - Correctly identify winning moves as negative from the loser's perspective
"""

import pytest
import hexo_rs

from hexo_a0.mcts import MCTSNode, backup, _q_from_parent, select_child


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(player: str, prior: float = 0.5) -> MCTSNode:
    """Create a minimal MCTSNode with the given player."""
    return MCTSNode(prior=prior, current_player=player)


def _make_expanded_node(
    player: str,
    child_actions: list[tuple[int, int]],
    priors: list[float] | None = None,
) -> MCTSNode:
    """Create an expanded node with child priors but no realized children."""
    if priors is None:
        priors = [1.0 / len(child_actions)] * len(child_actions)
    node = MCTSNode(prior=0.5, current_player=player)
    node._expanded = True
    node._child_priors = dict(zip(child_actions, priors))
    return node


# ---------------------------------------------------------------------------
# backup: mid-turn (same player) — no sign flip
# ---------------------------------------------------------------------------


class TestBackupMidTurn:
    """Backup through two same-player nodes (mid-turn, second placement)."""

    def test_mid_turn_no_sign_flip(self):
        """P1 → P1 path: value stays positive throughout."""
        root = _make_node("P1")
        child = _make_node("P1")  # same player — mid-turn
        path = [root, child]

        backup(path, value=1.0)

        # Child gets raw value
        assert child.value_sum == 1.0
        assert child.visit_count == 1
        # Root gets same value (no flip — same player)
        assert root.value_sum == 1.0
        assert root.visit_count == 1

    def test_mid_turn_negative_value_no_flip(self):
        """P2 → P2 path with negative value: stays negative."""
        root = _make_node("P2")
        child = _make_node("P2")
        path = [root, child]

        backup(path, value=-0.7)

        assert child.value_sum == -0.7
        assert root.value_sum == -0.7

    def test_mid_turn_q_from_parent_no_flip(self):
        """_q_from_parent returns Q unchanged for same-player child."""
        child = _make_node("P1")
        child.visit_count = 5
        child.value_sum = 3.0  # Q = 0.6

        q = _q_from_parent(child, parent_player="P1")
        assert q == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# backup: cross-turn (different players) — sign flip
# ---------------------------------------------------------------------------


class TestBackupCrossTurn:
    """Backup through a turn boundary where current_player changes."""

    def test_cross_turn_sign_flip(self):
        """P1 → P2 path: value negated for the parent."""
        root = _make_node("P1")
        child = _make_node("P2")  # different player — turn boundary
        path = [root, child]

        backup(path, value=1.0)

        # Child (P2) gets raw +1.0
        assert child.value_sum == 1.0
        # Root (P1) gets negated: -1.0 (P2's gain is P1's loss)
        assert root.value_sum == -1.0

    def test_cross_turn_q_from_parent_negates(self):
        """_q_from_parent negates Q for different-player child."""
        child = _make_node("P2")
        child.visit_count = 4
        child.value_sum = 2.0  # Q = 0.5

        q = _q_from_parent(child, parent_player="P1")
        assert q == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# backup: full 2-move turn cycle (P1 → P1 → P2 → P2)
# ---------------------------------------------------------------------------


class TestBackupFullTurnCycle:
    """Backup through a complete turn cycle: P1 plays twice, then P2 plays twice."""

    def test_p1_turn_then_p2_turn_positive_leaf(self):
        """Path: P1 → P1 → P2 → P2, leaf value +1.0 (good for P2).

        Expected propagation:
          P2 (leaf):   +1.0  (raw)
          P2 (depth 2): +1.0  (same player, no flip)
          P1 (depth 1): -1.0  (P2→P1 boundary, flip)
          P1 (root):    -1.0  (same player, no flip)
        """
        root = _make_node("P1")        # P1 first placement
        p1_mid = _make_node("P1")      # P1 second placement
        p2_first = _make_node("P2")    # P2 first placement
        p2_second = _make_node("P2")   # P2 second placement (leaf)
        path = [root, p1_mid, p2_first, p2_second]

        backup(path, value=1.0)

        assert p2_second.value_sum == pytest.approx(1.0)
        assert p2_first.value_sum == pytest.approx(1.0)
        assert p1_mid.value_sum == pytest.approx(-1.0)
        assert root.value_sum == pytest.approx(-1.0)

    def test_p1_turn_then_p2_turn_negative_leaf(self):
        """Same path, leaf value -1.0 (bad for P2 = good for P1).

        Expected:
          P2 (leaf):   -1.0
          P2 (depth 2): -1.0
          P1 (depth 1): +1.0  (flip at boundary)
          P1 (root):    +1.0
        """
        root = _make_node("P1")
        p1_mid = _make_node("P1")
        p2_first = _make_node("P2")
        p2_second = _make_node("P2")
        path = [root, p1_mid, p2_first, p2_second]

        backup(path, value=-1.0)

        assert p2_second.value_sum == pytest.approx(-1.0)
        assert p2_first.value_sum == pytest.approx(-1.0)
        assert p1_mid.value_sum == pytest.approx(1.0)
        assert root.value_sum == pytest.approx(1.0)

    def test_three_turn_path(self):
        """6-node path: P1→P1→P2→P2→P1→P1 — two boundary flips.

        Leaf value +1.0 (good for P1 at leaf).
        Expected:
          P1 (leaf):   +1.0
          P1 (depth 4): +1.0  (same)
          P2 (depth 3): -1.0  (flip at P1→P2)
          P2 (depth 2): -1.0  (same)
          P1 (depth 1): +1.0  (flip at P2→P1)
          P1 (root):    +1.0  (same)
        """
        nodes = [
            _make_node("P1"),  # root
            _make_node("P1"),  # P1 second
            _make_node("P2"),  # P2 first
            _make_node("P2"),  # P2 second
            _make_node("P1"),  # P1 first
            _make_node("P1"),  # P1 second (leaf)
        ]
        backup(nodes, value=1.0)

        expected = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
        for node, exp in zip(nodes, expected):
            assert node.value_sum == pytest.approx(exp), (
                f"Player {node.current_player} expected {exp}, got {node.value_sum}"
            )

    def test_visit_counts_always_increment(self):
        """Every node in the path gets visit_count += 1, regardless of flips."""
        nodes = [
            _make_node("P1"),
            _make_node("P1"),
            _make_node("P2"),
            _make_node("P2"),
        ]
        backup(nodes, value=0.5)

        for node in nodes:
            assert node.visit_count == 1


# ---------------------------------------------------------------------------
# backup: terminal outcomes
# ---------------------------------------------------------------------------


class TestBackupTerminal:
    """Test backup with terminal game outcomes (+1 win, -1 loss, 0 draw)."""

    def test_win_propagates_positive_to_winner(self):
        """P1 wins at the leaf (value=+1). Root is P1 → gets +1."""
        root = _make_node("P1")
        leaf = _make_node("P1")  # P1's second move completes a win
        path = [root, leaf]

        backup(path, value=1.0)

        assert root.value_sum == pytest.approx(1.0)
        assert root.q_value == pytest.approx(1.0)

    def test_win_propagates_negative_to_loser(self):
        """P1 wins at the leaf (value=+1), but root is P2 → gets -1."""
        # P2 → P2 → P1 path: P1 wins
        root = _make_node("P2")
        p2_mid = _make_node("P2")
        p1_win = _make_node("P1")
        path = [root, p2_mid, p1_win]

        # value=+1 is from P1's perspective (P1 just won)
        backup(path, value=1.0)

        assert p1_win.value_sum == pytest.approx(1.0)   # P1: +1
        assert p2_mid.value_sum == pytest.approx(-1.0)   # P2: -1 (flip at boundary)
        assert root.value_sum == pytest.approx(-1.0)      # P2: -1 (same player, no flip)

    def test_draw_propagates_zero(self):
        """Draw (value=0) propagates as 0 everywhere — no flip matters."""
        nodes = [_make_node("P1"), _make_node("P1"), _make_node("P2")]
        backup(nodes, value=0.0)

        for node in nodes:
            assert node.value_sum == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _q_from_parent: comprehensive
# ---------------------------------------------------------------------------


class TestQFromParent:
    """Detailed tests for the perspective-correction helper."""

    def test_unvisited_child_returns_zero(self):
        """Unvisited child has Q=0, _q_from_parent returns 0 regardless."""
        child = _make_node("P1")
        assert _q_from_parent(child, "P1") == 0.0
        assert _q_from_parent(child, "P2") == 0.0

    def test_positive_q_same_player(self):
        child = _make_node("P1")
        child.visit_count = 10
        child.value_sum = 8.0
        assert _q_from_parent(child, "P1") == pytest.approx(0.8)

    def test_positive_q_different_player(self):
        child = _make_node("P1")
        child.visit_count = 10
        child.value_sum = 8.0
        assert _q_from_parent(child, "P2") == pytest.approx(-0.8)

    def test_negative_q_different_player_double_negation(self):
        """Negative Q + different player → positive (double negation)."""
        child = _make_node("P2")
        child.visit_count = 4
        child.value_sum = -2.0  # Q = -0.5
        assert _q_from_parent(child, "P1") == pytest.approx(0.5)

    def test_symmetry(self):
        """_q_from_parent(child, P1) == -_q_from_parent(child, P2) when child is non-zero."""
        child = _make_node("P1")
        child.visit_count = 3
        child.value_sum = 1.5
        q_for_p1 = _q_from_parent(child, "P1")
        q_for_p2 = _q_from_parent(child, "P2")
        assert q_for_p1 == pytest.approx(-q_for_p2)


# ---------------------------------------------------------------------------
# Integration: backup on a real GameState turn sequence
# ---------------------------------------------------------------------------


class TestBackupWithRealGameState:
    """Use actual hexo_rs.GameState to verify player transitions match backup."""

    @pytest.fixture
    def game(self):
        """4-in-a-row, radius 2, max 80 moves."""
        return hexo_rs.GameState(
            hexo_rs.GameConfig(win_length=4, placement_radius=2, max_moves=80)
        )

    def test_real_game_four_moves_player_sequence(self, game):
        """Play 4 moves and verify the player sequence is P1→P1→P2→P2."""
        # Fresh game starts with P1 having placed at origin, P2 to move
        # Actually GameState() starts: P1 at origin, P2 to move, 2 remaining
        players = []
        states = []
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        current = game
        players.append(current.current_player())
        for q, r in moves:
            current.apply_move(q, r)
            players.append(current.current_player())

        # P2 moves twice, then P1 moves twice
        # Initial: P2 to move
        # After (1,0): P2 (still has 1 move)
        # After (-1,0): P1 (P2's turn done)
        # After (0,1): P1 (still has 1 move)
        # After (0,-1): P2 (P1's turn done)
        assert players == ["P2", "P2", "P1", "P1", "P2"]

    def test_backup_matches_real_game_transitions(self, game):
        """Build MCTS nodes mirroring a real game and verify backup correctness."""
        # Game starts: P2 to move, 2 moves remaining
        assert game.current_player() == "P2"

        # Build path: root(P2) → mid(P2) → cross(P1) → mid(P1)
        root = _make_node(game.current_player())  # P2

        game.apply_move(1, 0)  # P2's first move
        mid_p2 = _make_node(game.current_player())  # still P2

        game.apply_move(-1, 0)  # P2's second move → turn passes to P1
        cross_p1 = _make_node(game.current_player())  # P1

        game.apply_move(0, 1)  # P1's first move
        mid_p1 = _make_node(game.current_player())  # still P1

        path = [root, mid_p2, cross_p1, mid_p1]

        # Leaf value +1.0 from P1's perspective (good for P1)
        backup(path, value=1.0)

        # P1 nodes get +1 (good for them)
        assert mid_p1.value_sum == pytest.approx(1.0)
        assert cross_p1.value_sum == pytest.approx(1.0)
        # P2 nodes get -1 (flip at P1→P2 boundary)
        assert mid_p2.value_sum == pytest.approx(-1.0)
        assert root.value_sum == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Integration: winning move gets positive Q from parent's view
# ---------------------------------------------------------------------------


class TestWinningMoveQPerspective:
    """Verify that a winning move has positive Q from the winning parent's perspective."""

    def test_winning_child_positive_q_for_winner(self):
        """Parent (P1) has a child where P1 wins → _q_from_parent is positive."""
        parent = _make_node("P1")

        # Child is P1 (mid-turn win on second placement)
        # After backup of a +1.0 win from P1's perspective
        winning_child = _make_node("P1")
        winning_child.visit_count = 1
        winning_child.value_sum = 1.0  # +1 from P1's perspective

        q = _q_from_parent(winning_child, parent_player="P1")
        assert q > 0, f"Winning move should have positive Q, got {q}"

    def test_winning_child_positive_q_across_turn(self):
        """Parent (P2) has a child (P1) that leads to P2 winning eventually.

        If the path goes P2 → P1(leaf), and P2 wins, the leaf value is -1
        from P1's perspective. After backup: P1 child gets -1, P2 root gets +1.
        Reading child Q from P2's perspective: negate(-1) = +1. Correct.
        """
        parent = _make_node("P2")
        child = _make_node("P1")

        # Leaf value: P2 won, so from P1's perspective it's -1
        backup([parent, child], value=-1.0)

        # Child's raw Q is -1 (bad for P1), but from P2's view it's +1
        assert child.q_value == pytest.approx(-1.0)
        assert _q_from_parent(child, parent_player="P2") == pytest.approx(1.0)

        # Parent also correctly accumulated +1
        assert parent.q_value == pytest.approx(1.0)

    def test_losing_child_negative_q_for_loser(self):
        """Parent (P1) has a child where P2 wins → _q_from_parent is negative."""
        parent = _make_node("P1")
        child = _make_node("P2")

        # From P2's perspective, P2 won: value = +1
        backup([parent, child], value=1.0)

        # From P1's perspective, child Q should be negative (P2 winning is bad for P1)
        assert _q_from_parent(child, parent_player="P1") == pytest.approx(-1.0)
        assert parent.q_value == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# select_child: prefers winning moves
# ---------------------------------------------------------------------------


class TestSelectChildPrefersWins:
    """select_child should prefer actions that lead to winning positions."""

    def test_winning_action_selected_over_neutral(self):
        """Given one child with Q=+1 (win) and one with Q=0, select the win."""
        root = _make_expanded_node("P1", [(0, 1), (1, 0)])

        # Create children manually
        win_child = MCTSNode(prior=0.5, current_player="P1", action=(0, 1))
        win_child.visit_count = 1
        win_child.value_sum = 1.0  # Q=+1 from P1's view (same player)

        neutral_child = MCTSNode(prior=0.5, current_player="P1", action=(1, 0))
        neutral_child.visit_count = 1
        neutral_child.value_sum = 0.0  # Q=0

        root.children = {(0, 1): win_child, (1, 0): neutral_child}
        root.value_estimate = 0.0

        chosen = select_child(root)
        assert chosen == (0, 1), "Should select the winning action"

    def test_winning_action_across_turn_boundary(self):
        """Parent P1, children are P2 (turn boundary).
        Child A: Q = -1.0 from P2's perspective (= P1 won) → +1.0 from P1's view.
        Child B: Q = +0.5 from P2's perspective (= P2 slightly ahead) → -0.5 from P1's view.
        select_child should pick A.
        """
        root = _make_expanded_node("P1", [(0, 1), (1, 0)])

        # Child A: P2 perspective Q=-1 means P1 won
        child_a = MCTSNode(prior=0.5, current_player="P2", action=(0, 1))
        child_a.visit_count = 1
        child_a.value_sum = -1.0

        # Child B: P2 perspective Q=+0.5 means P2 ahead
        child_b = MCTSNode(prior=0.5, current_player="P2", action=(1, 0))
        child_b.visit_count = 1
        child_b.value_sum = 0.5

        root.children = {(0, 1): child_a, (1, 0): child_b}
        root.value_estimate = 0.0

        chosen = select_child(root)
        assert chosen == (0, 1), "Should select action leading to P1 win"


# ---------------------------------------------------------------------------
# Accumulation: multiple backups don't corrupt
# ---------------------------------------------------------------------------


class TestMultipleBackups:
    """Multiple backups through the same tree accumulate correctly."""

    def test_two_backups_same_path_accumulate(self):
        """Two +1 backups through P1→P2 path: P2 gets +2, P1 gets -2."""
        root = _make_node("P1")
        child = _make_node("P2")
        path = [root, child]

        backup(path, value=1.0)
        backup(path, value=1.0)

        assert child.visit_count == 2
        assert child.value_sum == pytest.approx(2.0)
        assert child.q_value == pytest.approx(1.0)

        assert root.visit_count == 2
        assert root.value_sum == pytest.approx(-2.0)
        assert root.q_value == pytest.approx(-1.0)

    def test_mixed_outcomes_average_correctly(self):
        """P1→P1→P2 path: one +1 and one -0.5 from P2's perspective."""
        root = _make_node("P1")
        mid = _make_node("P1")
        leaf = _make_node("P2")
        path = [root, mid, leaf]

        backup(path, value=1.0)   # Good for P2
        backup(path, value=-0.5)  # Slightly bad for P2

        assert leaf.visit_count == 2
        assert leaf.value_sum == pytest.approx(0.5)
        assert leaf.q_value == pytest.approx(0.25)

        # P1 nodes: first backup → -1, second → +0.5 = -0.5 total
        assert root.q_value == pytest.approx(-0.25)
        assert mid.q_value == pytest.approx(-0.25)
