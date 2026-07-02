"""MCTS components for HeXO AlphaZero.

Implements:
  - MCTSNode: tree node with visit statistics (REQ-2)
  - backup: player-aware value propagation (REQ-3)
  - sigma / normalize_q / v_mix: Gumbel AlphaZero score helpers
  - select_child: Section 5 non-root sequential-halving selection (REQ-4)
  - evaluate_state: network evaluation of a game state (REQ-5)
  - gumbel_top_k: Gumbel noise-based top-k action sampling (REQ-6)
  - compute_improved_policy: improved policy via completed-Q + sigma (REQ-8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    """A node in the MCTS tree.

    Args:
        prior:          Network's prior probability p(a) for the action that
                        leads to this node.
        action:         (q, r) hex coordinate that created this node, or None
                        for the root.
        current_player: "P1" or "P2" — the player to move at this position.
        value_estimate: Network's scalar value prediction V(s) at this state.
        parent:         Parent node, or None for the root.
        game_state:     The hexo_rs.GameState at this node, or None if not yet
                        stored.  Using ``object | None`` avoids importing hexo_rs
                        at the type level.
    """

    prior: float
    action: tuple[int, int] | None = None
    current_player: str | None = None
    value_estimate: float = 0.0
    parent: MCTSNode | None = None

    # Statistics updated during search
    visit_count: int = 0
    value_sum: float = 0.0

    # Children: action → child MCTSNode (lazily populated on first visit)
    children: dict[tuple[int, int], MCTSNode] = field(default_factory=dict)

    # Game state at this node (set when the node is created during expansion)
    game_state: object | None = None

    # Lazy expansion: priors and coords stored after network eval,
    # children created on-demand when selected
    _child_priors: dict[tuple[int, int], float] = field(default_factory=dict)
    _expanded: bool = False

    @property
    def is_expanded(self) -> bool:
        """True if this node has been evaluated by the network."""
        return self._expanded

    def get_or_create_child(self, action: tuple[int, int]) -> "MCTSNode":
        """Get existing child or lazily create it on first access."""
        if action in self.children:
            return self.children[action]
        # Create the child node on demand
        if action not in self._child_priors:
            raise ValueError(f"Action {action} not in node's legal moves")
        child_state = self.game_state.clone()
        child_state.apply_move(action[0], action[1])
        child_node = MCTSNode(
            prior=self._child_priors[action],
            action=action,
            current_player=child_state.current_player(),
            parent=self,
        )
        child_node.game_state = child_state
        # Infer current_player for terminal states
        if child_node.current_player is None:
            child_node.current_player = (
                "P2" if self.current_player == "P1" else "P1"
            )
        self.children[action] = child_node
        return child_node

    @property
    def q_value(self) -> float:
        """Empirical mean value: value_sum / visit_count, or 0.0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# ---------------------------------------------------------------------------
# Player-aware backup (REQ-3)
# ---------------------------------------------------------------------------


def backup(path: list[MCTSNode], value: float) -> None:
    """Propagate *value* from the leaf (last in path) back to the root (first).

    Rules
    -----
    - Every node in *path* gets ``visit_count += 1``.
    - The leaf (``path[-1]``) receives ``value_sum += value``.
    - Walking up towards the root, when moving from a child to its parent:
        - If ``child.current_player != parent.current_player``: negate value.
        - If same player (mid-turn two-placement), do not negate.

    Args:
        path:  List of nodes from root (index 0) to leaf (index -1).
        value: Value from the network at the leaf position (from leaf's
               perspective / current player at leaf).
    """
    # We propagate backwards: leaf gets the raw value, then walk up flipping
    # sign whenever the player changes.
    current_value = value

    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        node.visit_count += 1
        node.value_sum += current_value

        # Prepare value for the parent of this node (one step higher)
        if i > 0:
            parent = path[i - 1]
            if node.current_player != parent.current_player:
                current_value = -current_value
            # else: same player — mid-turn, value carries unchanged


# ---------------------------------------------------------------------------
# Pure scoring helpers
# ---------------------------------------------------------------------------


def sigma(
    q: float,
    max_child_visits: int,
    c_visit: int = 50,
    c_scale: float = 1.0,
) -> float:
    """Score transformation used in Gumbel AlphaZero (Danihelka et al., 2022).

    sigma(q) = (c_visit + max_child_visits) * c_scale * q

    Args:
        q:                Normalized Q-value in [0, 1].
        max_child_visits: max_{b} N(b) over the siblings being scored.
        c_visit:          Visit count baseline (default 50).
        c_scale:          Scale factor (default 1.0).

    Returns:
        Scaled score as a float.
    """
    return (c_visit + max_child_visits) * c_scale * q


def normalize_q(
    q_values: dict[tuple[int, int], float],
    q_min: float,
    q_max: float,
) -> dict[tuple[int, int], float]:
    """Min-max normalize a dict of Q-values.

    If ``q_min == q_max`` (all equal or single value), every entry maps to 0.0.

    Args:
        q_values: Mapping from action to Q-value.
        q_min:    Minimum Q (should be computed over *visited* children only).
        q_max:    Maximum Q (should be computed over *visited* children only).

    Returns:
        New dict with the same keys and values normalized to [0, 1].
    """
    if not q_values:
        return {}
    if not (q_max > q_min):
        return {a: 0.0 for a in q_values}
    span = q_max - q_min
    return {a: max(0.0, min(1.0, (q - q_min) / span)) for a, q in q_values.items()}


def _q_from_parent(child: MCTSNode, parent_player: str) -> float:
    """Return child's Q-value from the PARENT's perspective.

    Child nodes store Q from their own perspective (the next player to move).
    When parent and child have different players, the Q must be negated.
    When same player (mid-turn in HeXO's 2-placement turns), Q is used as-is.
    """
    q = child.q_value
    if child.current_player != parent_player:
        return -q
    return q


def v_mix(
    value_estimate: float,
    children: dict[tuple[int, int], MCTSNode],
    parent_player: str | None = None,
) -> float:
    """Mixed value combining network estimate with empirical Q (Eq. 33).

    v_mix = (v_hat + total_N / sum_visited_prior * sum_visited(pi(a)*Q(a)))
            / (1 + total_N)

    When no children have been visited (total_N == 0), returns ``value_estimate``.

    Args:
        value_estimate: Network's scalar V(s) for the parent node.
        children:       Mapping from action to child MCTSNode.

    Returns:
        Mixed value estimate as a float.
    """
    total_n = sum(c.visit_count for c in children.values())
    if total_n == 0:
        return value_estimate

    visited = [c for c in children.values() if c.visit_count > 0]
    sum_visited_prior = sum(c.prior for c in visited)
    if sum_visited_prior < 1e-8:
        return value_estimate
    # Q-values must be from the parent's perspective for correct mixing
    if parent_player is not None:
        sum_pi_q = sum(c.prior * _q_from_parent(c, parent_player) for c in visited)
    else:
        sum_pi_q = sum(c.prior * c.q_value for c in visited)

    numerator = value_estimate + (total_n / sum_visited_prior) * sum_pi_q
    return numerator / (1 + total_n)


# ---------------------------------------------------------------------------
# Section 5 non-root selection (REQ-4)
# ---------------------------------------------------------------------------


def select_child(
    node: MCTSNode,
    c_visit: int = 50,
    c_scale: float = 1.0,
) -> tuple[int, int]:
    """Select the best child action from *node* using Gumbel AlphaZero Eq. 7.

    Algorithm (non-root selection, Section 5):
    1.  Compute completedQ for each child:
        - Visited children: use child.q_value (empirical mean).
        - Unvisited children: use v_mix(node.value_estimate, node.children).
    2.  Normalize all completedQ values with min-max over *visited* Q only
        (clip unvisited at same scale).
    3.  For each child compute:
          pi'(a) = softmax( log(prior_a) + sigma(normalized_Q_a) )
    4.  Select argmax_a [ pi'(a) - N(a) / (1 + sum_b N(b)) ]

    Args:
        node:    Parent node whose children we select among.
        c_visit: Visit count baseline for sigma (default 50).
        c_scale: Scale factor for sigma (default 1.0).

    Returns:
        The (q, r) action key identifying the chosen child.
    """
    if not node.is_expanded:
        raise ValueError("Cannot select child from unexpanded node")

    # All known actions: union of realized children + unrealized priors
    all_actions = list(node._child_priors.keys())
    if not all_actions:
        raise ValueError("Node has no legal actions")

    # Step 1: completedQ — from the PARENT's perspective
    parent_player = node.current_player
    vmix_val = v_mix(node.value_estimate, node.children, parent_player)
    completed_q: dict[tuple[int, int], float] = {}
    for a in all_actions:
        if a in node.children and node.children[a].visit_count > 0:
            completed_q[a] = _q_from_parent(node.children[a], parent_player)
        else:
            completed_q[a] = vmix_val

    # Step 2: normalize using visited Q only (from parent's perspective)
    visited_qs = [
        _q_from_parent(node.children[a], parent_player)
        for a in all_actions
        if a in node.children and node.children[a].visit_count > 0
    ]
    if len(visited_qs) >= 1:
        q_min = min(visited_qs)
        q_max = max(visited_qs)
    else:
        q_min = q_max = 0.0

    norm_q = normalize_q(completed_q, q_min=q_min, q_max=q_max)

    # Step 3: sigma scores and pi'
    max_visits = max(
        (node.children[a].visit_count for a in all_actions if a in node.children),
        default=0,
    )
    logits_list = torch.tensor([
        math.log(node._child_priors[a] + 1e-8)
        + sigma(norm_q[a], max_child_visits=max_visits, c_visit=c_visit, c_scale=c_scale)
        for a in all_actions
    ])
    pi_prime = F.softmax(logits_list, dim=-1)

    # Step 4: selection criterion
    total_n = sum(
        node.children[a].visit_count for a in all_actions if a in node.children
    )
    scores = [
        float(pp) - (node.children[a].visit_count if a in node.children else 0) / (1 + total_n)
        for pp, a in zip(pi_prime, all_actions)
    ]

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return all_actions[best_idx]


# ---------------------------------------------------------------------------
# Network evaluation (REQ-5)
# ---------------------------------------------------------------------------


def evaluate_state(
    network: "torch.nn.Module",
    game: object,
    device: "torch.device",
    graph_fn=None,
) -> tuple[torch.Tensor, float, list[tuple[int, int]]]:
    """Evaluate a game state with the given network.

    Calls ``graph_fn(game)`` (default: ``game_to_graph``) to build a PyG Data
    object, moves tensors to *device*, runs the network under
    ``torch.no_grad()``, and returns results on CPU.

    The function does NOT toggle model mode; the caller is responsible for
    calling ``network.eval()`` before invoking this function.

    Args:
        network: A HeXONet instance (or any callable matching the same
                 ``forward(x, edge_index, legal_mask)`` signature).
        game:    A ``hexo_rs.GameState`` that is not terminal.
        device:  Torch device to run inference on.
        graph_fn: Callable to convert game state to PyG Data.
                  Defaults to ``game_to_graph`` if None.

    Returns:
        A 3-tuple ``(logits, value, coords)`` where:

        - ``logits`` — 1-D CPU tensor of length ``num_legal``, detached,
          no gradient.
        - ``value`` — Python float in [-1, 1].
        - ``coords`` — list of ``(q, r)`` int tuples, one per legal move,
          sorted by ``(q, r)`` matching the order produced by
          ``game_to_graph``.
    """
    if graph_fn is None:
        from hexo_a0.graph import game_to_graph  # local import to avoid circular dep
        graph_fn = game_to_graph

    data = graph_fn(game)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    legal_mask = data.legal_mask.to(device)

    stone_mask = data.stone_mask.to(device)

    # Pass edge_attr to the model when present (axis graphs)
    edge_attr = getattr(data, "edge_attr", None)
    kwargs = {}
    if edge_attr is not None:
        kwargs["edge_attr"] = edge_attr.to(device)

    with torch.no_grad():
        policy_logits, value_tensor = network.forward(
            x, edge_index, legal_mask, stone_mask, **kwargs
        )

    logits = policy_logits.detach().cpu()
    value = float(value_tensor.item())

    # Extract (q, r) coords for legal nodes; data.coords[legal_mask] already
    # sorted because game_to_graph places legal nodes sorted by (q,r).
    legal_coords_tensor = data.coords[data.legal_mask]  # shape (num_legal, 2)
    coords = [(int(c[0].item()), int(c[1].item())) for c in legal_coords_tensor]

    return logits, value, coords


# ---------------------------------------------------------------------------
# Gumbel top-k sampling (REQ-6)
# ---------------------------------------------------------------------------


def gumbel_top_k(
    logits: torch.Tensor,
    m: int,
    rng: "torch.Generator | None" = None,
) -> tuple[list[int], torch.Tensor]:
    """Sample Gumbel noise and return top-min(m, len(logits)) action indices.

    Algorithm:
    1. Sample ``n = len(logits)`` Gumbel(0, 1) variables.
    2. Compute ``scores = gumbels + logits``.
    3. Return the indices of the top-``min(m, n)`` scores together with the
       full Gumbel sample array.

    Args:
        logits: 1-D tensor of raw network logits (not log-softmax).
        m:      Number of candidates to return.
        rng:    Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        ``(candidate_indices, gumbel_samples)`` where:

        - ``candidate_indices`` — list of int indices into *logits* for the
          top-``k`` actions, sorted in descending order of
          ``(gumbel + logit)`` score.
        - ``gumbel_samples`` — 1-D float tensor of length ``len(logits)``
          containing the sampled Gumbel values for all actions.
    """
    n = logits.shape[0]
    k = min(m, n)

    # Sample uniform U ~ Uniform(0, 1) then apply Gumbel transform: -log(-log(U))
    if rng is not None:
        u = torch.rand(n, generator=rng, dtype=torch.float32)
    else:
        u = torch.rand(n, dtype=torch.float32)

    # Clamp to avoid log(0)
    u = u.clamp(min=1e-20, max=1.0 - 1e-20)
    gumbels = -torch.log(-torch.log(u))

    scores = gumbels + logits.float()
    _, top_indices = torch.topk(scores, k=k)
    candidate_indices = [int(i.item()) for i in top_indices]

    return candidate_indices, gumbels


# ---------------------------------------------------------------------------
# Improved policy computation (REQ-8)
# ---------------------------------------------------------------------------


def compute_improved_policy(
    logits: torch.Tensor,
    coords: list[tuple[int, int]],
    root_node: MCTSNode,
    c_visit: int = 50,
    c_scale: float = 1.0,
) -> torch.Tensor:
    """Compute the improved policy distribution from a completed search.

    For each legal action (indexed by position in *logits* / *coords*):

    1. Determine ``completedQ``:
       - If the action's child in ``root_node.children`` has
         ``visit_count > 0``: use ``child.q_value``.
       - Otherwise (unvisited child or action not in children at all): use
         ``v_mix(root_node.value_estimate, root_node.children)``.
    2. Normalize completedQ to [0, 1] using min-max over **visited** children
       Q-values only (via ``normalize_q``).
    3. Compute ``sigma(norm_q, max_child_visits)`` for each action, where
       ``max_child_visits`` is the maximum visit count across all children.
    4. Compute ``log_scores = logits + sigma_values``.
    5. Return ``softmax(log_scores)`` as a probability distribution.

    Actions absent from ``root_node.children`` receive completedQ = v_mix.
    Logit indexing is aligned to *coords*, which must match the output of
    ``evaluate_state``.

    Args:
        logits:     1-D tensor of raw network logits, one per legal move.
        coords:     List of ``(q, r)`` tuples corresponding to each logit index.
        root_node:  Root ``MCTSNode`` after search.
        c_visit:    Visit-count baseline for sigma (default 50).
        c_scale:    Scale factor for sigma (default 1.0).

    Returns:
        1-D probability tensor of shape ``(len(logits),)`` summing to 1.
    """
    parent_player = root_node.current_player
    vmix_val = v_mix(root_node.value_estimate, root_node.children, parent_player)

    # Step 1: gather completedQ for every action — from ROOT's perspective
    completed_q_dict: dict[tuple[int, int], float] = {}
    for coord in coords:
        child = root_node.children.get(coord)
        if child is not None and child.visit_count > 0:
            completed_q_dict[coord] = _q_from_parent(child, parent_player)
        else:
            completed_q_dict[coord] = vmix_val

    # Step 2: min-max normalize over visited children Q only (from root's perspective)
    visited_qs = [
        _q_from_parent(child, parent_player)
        for child in root_node.children.values()
        if child.visit_count > 0
    ]
    if len(visited_qs) >= 1:
        q_min = min(visited_qs)
        q_max = max(visited_qs)
    else:
        q_min = q_max = 0.0

    norm_q_dict = normalize_q(completed_q_dict, q_min=q_min, q_max=q_max)

    # Step 3: compute sigma for each action
    max_child_visits = max(
        (child.visit_count for child in root_node.children.values()),
        default=0,
    )

    sigma_values = torch.tensor(
        [sigma(norm_q_dict[coord], max_child_visits, c_visit=c_visit, c_scale=c_scale)
         for coord in coords],
        dtype=torch.float32,
    )

    # Step 4 & 5: log_scores then softmax
    log_scores = logits.float() + sigma_values

    policy = F.softmax(log_scores, dim=-1)

    return policy


# ---------------------------------------------------------------------------
# Single MCTS simulation (REQ-11)
# ---------------------------------------------------------------------------


def simulate(
    root: MCTSNode,
    network: "torch.nn.Module",
    device: "torch.device",
    forced_action: "tuple[int, int] | None" = None,
    c_visit: int = 50,
    c_scale: float = 1.0,
    graph_fn=None,
) -> None:
    """Run one MCTS simulation from *root*, expanding and backing up one leaf.

    Phases
    ------
    1. **Selection** — walk down the tree using ``select_child`` until reaching
       a node that has no children (an unexpanded leaf).  If *forced_action* is
       provided the first step always goes to ``root.children[forced_action]``
       instead of using ``select_child``.
    2. **Expansion** — if the leaf's game state is non-terminal, call
       ``evaluate_state`` to get network priors/value, create child nodes, and
       store the value on the leaf.  If terminal, compute the game outcome.
    3. **Backup** — call ``backup(path, value)`` to propagate the value from
       the leaf back to the root.

    Args:
        root:          Root ``MCTSNode`` (must already have ``game_state`` set
                       and children populated from a prior expansion).
        network:       Evaluation network (HeXONet or compatible).
        device:        Torch device for inference.
        forced_action: Optional action to force as the first step instead of
                       using ``select_child`` at the root.  Used by
                       ``sequential_halving`` to steer simulations.
        c_visit:       Visit count baseline for sigma (default 50).
        c_scale:       Scale factor for sigma (default 1.0).
    """
    # ------------------------------------------------------------------
    # Phase 1: Selection
    # ------------------------------------------------------------------
    path: list[MCTSNode] = [root]
    node = root

    # First step: forced or via select_child
    if forced_action is not None and forced_action in node._child_priors:
        node = node.get_or_create_child(forced_action)
        path.append(node)
    elif node.is_expanded:
        action = select_child(node, c_visit=c_visit, c_scale=c_scale)
        node = node.get_or_create_child(action)
        path.append(node)

    # Continue selection until we reach an unexpanded node
    while node.is_expanded:
        action = select_child(node, c_visit=c_visit, c_scale=c_scale)
        node = node.get_or_create_child(action)
        path.append(node)

    leaf = node

    # ------------------------------------------------------------------
    # Phase 2: Expansion
    # ------------------------------------------------------------------
    if leaf.game_state is None:
        raise RuntimeError("MCTSNode has no game_state — tree construction bug")

    if leaf.game_state.is_terminal():
        winner = leaf.game_state.winner()
        if winner is None:
            value = 0.0
        elif leaf.parent is not None and winner == leaf.parent.current_player:
            value = -1.0
        else:
            value = 1.0
        if leaf.parent is not None and leaf.current_player is None:
            parent_player = leaf.parent.current_player
            leaf.current_player = "P2" if parent_player == "P1" else "P1"
    else:
        # Lazy expansion: evaluate with network, store priors, but
        # DON'T create child nodes — they're created on demand
        logits, value, coords = evaluate_state(network, leaf.game_state, device, graph_fn=graph_fn)
        priors = F.softmax(logits, dim=-1)
        leaf._child_priors = {
            coord: float(priors[i]) for i, coord in enumerate(coords)
        }
        leaf._expanded = True
        leaf.value_estimate = value

    # ------------------------------------------------------------------
    # Phase 2 (cont.): Backup
    # ------------------------------------------------------------------
    backup(path, value)


# ---------------------------------------------------------------------------
# Batched simulation
# ---------------------------------------------------------------------------


def simulate_batch(
    root: MCTSNode,
    forced_actions: list[tuple[int, int]],
    network: "torch.nn.Module",
    device: "torch.device",
    c_visit: int = 50,
    c_scale: float = 1.0,
    graph_fn=None,
) -> None:
    """Run one simulation per forced action, batching all leaf evaluations.

    Instead of N sequential simulate() calls (each doing one forward pass),
    this does:
      1. Select a leaf for each forced action (CPU, sequential but fast)
      2. Batch all non-terminal leaf evaluations into ONE forward_batch call
      3. Expand all leaves and backup all paths

    This turns N forward passes into 1, giving ~Nx speedup on GPU.

    Args:
        root:           Root MCTSNode (already expanded).
        forced_actions: List of (q, r) actions — one simulation per action.
        network:        Evaluation network.
        device:         Torch device.
        c_visit:        Sigma parameter.
        c_scale:        Sigma parameter.
        graph_fn:       Graph builder function. Defaults to game_to_graph.
    """
    if graph_fn is None:
        from hexo_a0.graph import game_to_graph
        graph_fn = game_to_graph
    from torch_geometric.data import Batch

    if not forced_actions:
        return

    # --- Phase 1: Select a leaf for each forced action ---
    paths: list[list[MCTSNode]] = []
    leaves: list[MCTSNode] = []

    for action in forced_actions:
        path = [root]
        node = root

        # Forced first step
        if action in node._child_priors:
            node = node.get_or_create_child(action)
            path.append(node)

        # Continue selection until unexpanded leaf
        while node.is_expanded:
            child_action = select_child(node, c_visit=c_visit, c_scale=c_scale)
            node = node.get_or_create_child(child_action)
            path.append(node)

        paths.append(path)
        leaves.append(node)

    # --- Phase 2: Separate terminal vs non-terminal leaves ---
    terminal_indices: list[int] = []
    terminal_values: list[float] = []
    expand_indices: list[int] = []
    expand_leaves: list[MCTSNode] = []

    for i, leaf in enumerate(leaves):
        if leaf.game_state is None:
            raise RuntimeError("MCTSNode has no game_state — tree construction bug")

        if leaf.game_state.is_terminal():
            winner = leaf.game_state.winner()
            if winner is None:
                val = 0.0
            elif leaf.parent is not None and winner == leaf.parent.current_player:
                val = -1.0
            else:
                val = 1.0
            if leaf.parent is not None and leaf.current_player is None:
                parent_player = leaf.parent.current_player
                leaf.current_player = "P2" if parent_player == "P1" else "P1"
            terminal_indices.append(i)
            terminal_values.append(val)
        else:
            expand_indices.append(i)
            expand_leaves.append(leaf)

    # --- Phase 3: Batch evaluate all non-terminal leaves ---
    if expand_leaves:
        import logging as _log
        import time as _time
        _t0 = _time.perf_counter()
        data_list = [graph_fn(leaf.game_state) for leaf in expand_leaves]
        _t1 = _time.perf_counter()
        batch = Batch.from_data_list(data_list).to(device)
        _t2 = _time.perf_counter()

        with torch.no_grad():
            policy_logits_list, values = network.forward_batch(batch)
        _t3 = _time.perf_counter()

        _log.getLogger(__name__).debug(
            "  batch %d leaves: graph=%.1fms batch=%.1fms forward=%.1fms",
            len(expand_leaves),
            (_t1-_t0)*1000, (_t2-_t1)*1000, (_t3-_t2)*1000,
        )

        _t4 = _time.perf_counter()
        for j, (leaf, data) in enumerate(zip(expand_leaves, data_list)):
            logits_j = policy_logits_list[j]
            value_j = float(values[j].item())
            coords_j = [
                (int(c[0].item()), int(c[1].item()))
                for c in data.coords[data.legal_mask]
            ]

            # Lazy expansion: store priors, don't create child nodes
            priors = F.softmax(logits_j, dim=-1)
            leaf._child_priors = {
                coord: float(priors[k]) for k, coord in enumerate(coords_j)
            }
            leaf._expanded = True
            leaf.value_estimate = value_j

            # Backup
            idx = expand_indices[j]
            backup(paths[idx], value_j)

        _t5 = _time.perf_counter()
        _log.getLogger(__name__).debug(
            "  expand+backup: %.1fms (%.1fms per leaf)",
            (_t5-_t4)*1000, (_t5-_t4)*1000/max(len(expand_leaves),1),
        )

    # --- Phase 4: Backup terminal leaves ---
    for i, val in zip(terminal_indices, terminal_values):
        backup(paths[i], val)


# ---------------------------------------------------------------------------
# Sequential Halving (REQ-7)
# ---------------------------------------------------------------------------


def sequential_halving(
    root: MCTSNode,
    candidate_indices: list[int],
    candidate_gumbels: "torch.Tensor",
    coords: list[tuple[int, int]],
    logits: "torch.Tensor",
    network: "torch.nn.Module",
    config: object,
    device: "torch.device",
    graph_fn=None,
) -> list[int]:
    """Allocate simulations across candidates using Sequential Halving.

    Each phase distributes ``sims_per_action`` simulations to every surviving
    candidate by forcing the first move to that candidate's child.  After each
    phase the bottom half (by Gumbel score) is eliminated.  Halving stops when
    the budget is exhausted or only one candidate remains.

    Elimination uses the full Gumbel score:
        g(a) + logits(a) + sigma(normalized_Q(a), max_child_visits, c_visit, c_scale)
    where Q-values are min-max normalized over all visited root children.

    The per-phase simulation budget uses the ORIGINAL ``n_simulations``
    (not a decrementing counter), per the paper's Figure 1 formula:
        sims_per_action = max(1, floor(n / (ceil(log2(m)) * num_considered)))

    Args:
        root:              Root ``MCTSNode`` (already expanded; children exist
                           for every candidate action).
        candidate_indices: Indices into *coords* for the ``m`` top-k candidates
                           selected by Gumbel Top-k.
        candidate_gumbels: Full Gumbel sample tensor (length = len(coords)).
        coords:            List of ``(q, r)`` tuples produced by
                           ``evaluate_state``; maps index → action.
        logits:            Raw network logits at the root (1-D tensor, same
                           length as *coords*).
        network:           Evaluation network.
        config:            Object with ``n_simulations``, ``c_visit``, ``c_scale``.
        device:            Torch device.

    Returns:
        List of surviving candidate indices (subset of *candidate_indices*).

    Edge cases (AC-7f, AC-7g):
        - If ``config.n_simulations == 0``: return *candidate_indices* unchanged.
        - If ``len(candidate_indices) <= 1``: return *candidate_indices* unchanged.
    """
    n_simulations: int = config.n_simulations
    c_visit: int = config.c_visit
    c_scale: float = config.c_scale

    # AC-7f: zero budget — skip entirely
    if n_simulations == 0:
        return candidate_indices

    # AC-7g: single candidate — no halving needed
    if len(candidate_indices) <= 1:
        return candidate_indices

    num_phases = math.ceil(math.log2(len(candidate_indices)))
    remaining = list(candidate_indices)
    total_sims_used = 0

    for _phase in range(num_phases):
        if total_sims_used >= n_simulations or len(remaining) <= 1:
            break

        # Paper formula: sims_per_action = max(1, floor(n / (ceil(log2(m)) * num_considered)))
        sims_per_action = max(1, math.floor(n_simulations / (num_phases * len(remaining))))

        # Run sims_per_action rounds. Each round batches one sim per candidate.
        for _sim_round in range(sims_per_action):
            if total_sims_used >= n_simulations:
                break
            # Collect actions for this round (one per surviving candidate)
            batch_actions = []
            for idx in remaining:
                if total_sims_used >= n_simulations:
                    break
                batch_actions.append(coords[idx])
                total_sims_used += 1
            # Batch all forced simulations into one forward pass
            if batch_actions:
                simulate_batch(root, batch_actions, network, device,
                               c_visit=c_visit, c_scale=c_scale, graph_fn=graph_fn)

        # Eliminate bottom half by Gumbel score:
        # g(a) + logits(a) + sigma(normalized_Q(a))
        # Q-values from ROOT's perspective, normalized over ALL visited root children
        root_player = root.current_player
        visited_qs = [
            _q_from_parent(child, root_player)
            for child in root.children.values()
            if child.visit_count > 0
        ]
        if visited_qs:
            q_min = min(visited_qs)
            q_max = max(visited_qs)
        else:
            q_min = q_max = 0.0

        max_child_visits = max(
            (child.visit_count for child in root.children.values()),
            default=0,
        )

        vmix_val = v_mix(root.value_estimate, root.children, root_player)

        def _gumbel_score(idx: int) -> float:
            action = coords[idx]
            child = root.children.get(action)
            q_val = _q_from_parent(child, root_player) if (child is not None and child.visit_count > 0) else vmix_val
            if q_max > q_min:
                norm_q_val = max(0.0, min(1.0, (q_val - q_min) / (q_max - q_min)))
            else:
                norm_q_val = 0.0
            return (
                float(candidate_gumbels[idx].item())
                + float(logits[idx].item())
                + sigma(norm_q_val, max_child_visits, c_visit=c_visit, c_scale=c_scale)
            )

        remaining.sort(key=_gumbel_score, reverse=True)
        keep = math.ceil(len(remaining) / 2)
        remaining = remaining[:keep]

    return remaining


# ---------------------------------------------------------------------------
# Gumbel MCTS entry point (REQ-10)
# ---------------------------------------------------------------------------


def gumbel_mcts(
    game: object,
    network: "torch.nn.Module",
    config: object,
    device: "torch.device",
    rng: "torch.Generator | None" = None,
    graph_fn=None,
) -> tuple[tuple[int, int], torch.Tensor]:
    """Run Gumbel MCTS from a game state. Returns (action, improved_policy).

    Args:
        game:    hexo_rs.GameState (non-terminal).
        network: HeXONet model (must be in eval mode).
        config:  TrainingConfig with fields: n_simulations, m_actions,
                 c_visit, c_scale.
        device:  Torch device.
        rng:     Optional torch.Generator for reproducible Gumbel sampling.
        graph_fn: Graph builder function. Defaults to game_to_graph.

    Returns:
        (action_coord, improved_policy) where:
        - action_coord is a (q, r) tuple of the chosen move.
        - improved_policy is a 1-D tensor over all legal moves, summing to 1.

    Raises:
        ValueError if game is terminal.
    """
    if game.is_terminal():
        raise ValueError("Cannot run MCTS on terminal state")

    # Step 2: evaluate the root state
    logits, value, coords = evaluate_state(network, game, device, graph_fn=graph_fn)

    if not coords:
        raise ValueError("No legal moves in non-terminal state")

    # Step 3: build root node
    root = MCTSNode(
        prior=1.0,
        current_player=game.current_player(),
        value_estimate=value,
    )
    root.game_state = game

    # Step 4: lazy-expand root — store priors, children created on demand
    priors = F.softmax(logits, dim=-1)
    root._child_priors = {
        coord: float(priors[i]) for i, coord in enumerate(coords)
    }
    root._expanded = True

    # Step 5: Gumbel top-k sampling
    candidate_indices, gumbel_samples = gumbel_top_k(logits, config.m_actions, rng)

    # Step 6: sequential halving — allocates simulations across candidates
    surviving_indices = sequential_halving(
        root, candidate_indices, gumbel_samples, coords, logits,
        network, config, device, graph_fn=graph_fn,
    )

    # Step 7: compute improved policy over all legal moves
    improved_policy = compute_improved_policy(
        logits, coords, root, config.c_visit, config.c_scale
    )

    # Step 8: select action from survivors using Gumbel + logits + sigma(Q)
    # Q-values from ROOT's perspective, normalized over ALL visited root children.
    root_player = root.current_player
    visited_qs = [
        _q_from_parent(child, root_player)
        for child in root.children.values()
        if child.visit_count > 0
    ]
    if visited_qs:
        q_min = min(visited_qs)
        q_max = max(visited_qs)
    else:
        q_min = q_max = 0.0

    max_child_visits = max(
        (child.visit_count for child in root.children.values()),
        default=0,
    )

    vmix_val = v_mix(root.value_estimate, root.children, root_player)

    best_score: float = float("-inf")
    selected_coord: tuple[int, int] = coords[surviving_indices[0]]

    for idx in surviving_indices:
        coord = coords[idx]
        child = root.children.get(coord)
        q_val = _q_from_parent(child, root_player) if (child is not None and child.visit_count > 0) else vmix_val

        # Normalize Q using min-max computed from ALL visited root children
        if q_max > q_min:
            norm_q_val = max(0.0, min(1.0, (q_val - q_min) / (q_max - q_min)))
        else:
            norm_q_val = 0.0

        score = (
            float(gumbel_samples[idx].item())
            + float(logits[idx].item())
            + sigma(norm_q_val, max_child_visits, c_visit=config.c_visit, c_scale=config.c_scale)
        )
        if score > best_score:
            best_score = score
            selected_coord = coord

    # Step 9: return
    return selected_coord, improved_policy
