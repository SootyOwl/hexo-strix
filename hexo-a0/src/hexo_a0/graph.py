"""Graph construction: convert a hexo_rs.GameState to a PyTorch Geometric Data object.

Uses Rust-accelerated `hexo_rs.game_to_graph_raw()` for the heavy computation
(feature building, edge construction, hex distance), with a thin Python wrapper
to create the PyG Data object from the raw arrays.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def game_to_graph(
    game, *, threat_features: bool = False, relative_stones: bool = False
) -> Data:
    """Convert a GameState to a PyG Data object.

    Node ordering:
      - Placed stones first, sorted by coordinate tuple (q, r)
      - Empty legal cells next, sorted by coordinate tuple (q, r)

    Node features, absolute layout (8-dim float32, or 12-dim with
    threat_features: dims 8-11 = own/opp max clean-window line count,
    own/opp threat-axis count, relative to side to move):
      [0:3] Stone-type one-hot: [1,0,0]=P1, [0,1,0]=P2, [0,0,1]=empty
      [3]   Current player: +1.0 if P1's turn, -1.0 if P2's turn (global)
      [4]   Moves remaining: moves_remaining_this_turn() / 2.0 (global)
      [5]   Normalised q coordinate (relative to stone centroid)
      [6]   Normalised r coordinate (relative to stone centroid)
      [7]   Inverse distance to nearest stone: 1/d (0 for stones themselves)
      [8:12] (optional) Threat encoding: [own_max_line, opp_max_line,
            own_threat_axes, opp_threat_axes], normalised by win_length
            (lines) and 3 (axes); own = current player to move.

    With relative_stones=True the stone one-hot becomes side-to-move
    relative and the to_move dim is dropped (7-dim base, 11 with threat
    features; for terminal states side-to-move is treated as P2):
      [0:3] Stone-type one-hot: [1,0,0]=own (current player), [0,1,0]=opp,
            [0,0,1]=empty
      [3]   Moves remaining; [4] norm q; [5] norm r; [6] inv stone dist
      [7:11] (optional) Threat encoding as above.

    Args:
        game: A hexo_rs.GameState (must not be terminal).
        threat_features: Append 4 threat-encoding node features.
        relative_stones: Side-to-move-relative stone one-hot, to_move dropped.

    Returns:
        PyG Data with attributes: x, edge_index, legal_mask, stone_mask, coords.

    Raises:
        ValueError: if the game is in a terminal state.
    """
    import hexo_rs

    return _raw_to_data(
        hexo_rs.game_to_graph_raw(game, threat_features, relative_stones)
    )


def _raw_to_data(g: dict) -> Data:
    """Convert a raw graph dict from Rust to a PyG Data object."""
    n = g["num_nodes"]
    x = torch.tensor(g["features"], dtype=torch.float32).reshape(n, -1)
    edge_src = g["edge_src"]
    edge_dst = g["edge_dst"]
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.int64)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64)
    legal_mask = torch.tensor(g["legal_mask"], dtype=torch.bool)
    stone_mask = torch.tensor(g["stone_mask"], dtype=torch.bool)
    coords = torch.tensor(g["coords"], dtype=torch.int32).reshape(n, 2)
    return Data(
        x=x,
        edge_index=edge_index,
        legal_mask=legal_mask,
        stone_mask=stone_mask,
        coords=coords,
    )


def game_to_axis_graph(
    game,
    *,
    prune_empty_edges: bool = False,
    threat_features: bool = False,
    relative_stones: bool = False,
) -> Data:
    """Convert a GameState to a PyG Data object using axis-window graph.

    Uses Rust-accelerated `hexo_rs.game_to_axis_graph_raw()` for the heavy
    computation (feature building, edge construction with axis edge attributes).

    The returned Data includes `edge_attr` of shape (E, 5) encoding
    axis-window information for each edge.

    Node features: 8-dim float32 absolute layout, 7-dim with relative_stones
    (side-to-move-relative stone one-hot, to_move dropped), +4 threat dims
    with threat_features. See `game_to_graph` for the full feature schema.

    Args:
        game: A hexo_rs.GameState (must not be terminal).
        prune_empty_edges: Drop empty→empty edges to reduce graph size.
        threat_features: Append 4 threat-encoding node features.
        relative_stones: Side-to-move-relative stone one-hot, to_move dropped.

    Returns:
        PyG Data with: x, edge_index, edge_attr, legal_mask, stone_mask, coords.

    Raises:
        ValueError: if the game is in a terminal state.
    """
    import hexo_rs

    return _raw_to_axis_data(
        hexo_rs.game_to_axis_graph_raw(
            game, prune_empty_edges, threat_features, relative_stones
        )
    )


def _raw_to_axis_data(g: dict) -> Data:
    """Convert a raw axis-graph dict from Rust to a PyG Data object."""
    n = g["num_nodes"]
    x = torch.tensor(g["features"], dtype=torch.float32).reshape(n, -1)
    edge_src = g["edge_src"]
    edge_dst = g["edge_dst"]
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.int64)
        E = len(edge_src)
        edge_attr = torch.tensor(g["edge_attr"], dtype=torch.float32).reshape(E, 5)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.int64)
        edge_attr = torch.zeros((0, 5), dtype=torch.float32)
    legal_mask = torch.tensor(g["legal_mask"], dtype=torch.bool)
    stone_mask = torch.tensor(g["stone_mask"], dtype=torch.bool)
    coords = torch.tensor(g["coords"], dtype=torch.int32).reshape(n, 2)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        legal_mask=legal_mask,
        stone_mask=stone_mask,
        coords=coords,
    )


def graph_fn_from_model_config(model_config):
    """Return a callable ``game_state -> Data`` matching the model's graph config.

    Single source of truth for graph-construction flags (graph_type,
    prune_empty_edges, threat_features, relative_stone_encoding). Used by BOTH
    self-play generation and training-time reconstruction so the two cannot
    drift — a reconstructed graph is identical to one built at generation.
    """
    graph_type = model_config.graph_type
    prune = model_config.prune_empty_edges
    threat = getattr(model_config, "threat_features", False)
    rel = getattr(model_config, "relative_stone_encoding", False)
    if graph_type == "axis":
        return lambda g: game_to_axis_graph(
            g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
    return lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel)


def graph_batch_fn_from_model_config(model_config):
    """Return a callable ``list[game_state] -> list[Data]`` matching the model's
    graph config. Rayon-parallel + GIL-released (the batch builders use py.detach),
    so reconstructing a whole training batch in one call is ~10x faster than
    per-example graph_fn and doesn't block the main thread. Returns Data objects,
    so downstream augmentation/collation is unchanged."""
    graph_type = model_config.graph_type
    prune = model_config.prune_empty_edges
    threat = getattr(model_config, "threat_features", False)
    rel = getattr(model_config, "relative_stone_encoding", False)
    if graph_type == "axis":
        return lambda gs: game_to_axis_graph_batch(
            gs, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
    return lambda gs: game_to_graph_batch(gs, threat_features=threat, relative_stones=rel)


def game_to_axis_graph_batch(
    games: list,
    *,
    prune_empty_edges: bool = False,
    threat_features: bool = False,
    relative_stones: bool = False,
) -> list[Data]:
    """Convert a batch of GameStates to axis-graph PyG Data objects.

    Uses rayon for parallel graph construction across all states.
    Node features: 8-dim float32 absolute layout, 7-dim with relative_stones
    (to_move dropped), +4 threat dims with threat_features (see
    `game_to_graph` for the full schema).
    """
    import hexo_rs

    raw_graphs = hexo_rs.game_to_axis_graph_batch(
        games, prune_empty_edges, threat_features, relative_stones
    )
    return [_raw_to_axis_data(g) for g in raw_graphs]


def axis_states_to_batch(
    games: list,
    *,
    prune_empty_edges: bool = False,
    threat_features: bool = False,
    relative_stones: bool = False,
    device="cpu",
):
    """Collate GameStates into flat axis batch tensors via the Rust bytes binding.

    Bypasses per-graph Data objects and PyG collation entirely: Rust builds the
    graphs (rayon, GIL released), collates them flat, and hands back native-endian
    byte buffers — one ``torch.frombuffer`` memcpy per field.

    Returns (batch, aux): ``batch`` is a namespace with the PyG-Batch attributes
    ``HeXONet._forward_batch_core`` consumes (x, edge_index, edge_attr,
    legal_mask, batch, num_graphs); ``aux`` carries the Rust-precomputed index
    tensors (legal_idx, stone_idx, stone_batch — skip the GPU nonzero syncs),
    per-graph legal_counts, and per-node coords (N, 2).
    """
    import hexo_rs

    raw = hexo_rs.game_states_to_axis_batch_bytes(
        games, prune_empty_edges, threat_features, relative_stones
    )
    return _raw_axis_bytes_to_ns(raw, device)


def _raw_axis_bytes_to_ns(raw: dict, device):
    """Parse a Rust axis-batch byte dict into ``(batch, aux)`` namespaces.

    Shared by `axis_states_to_batch` and `augment_axis_states_to_batch` so the
    byte→tensor field layout (one ``torch.frombuffer`` memcpy per field, the
    PyG-Batch attributes ``_forward_batch_core`` consumes, and the
    Rust-precomputed index tensors) lives in exactly one place.
    """
    from types import SimpleNamespace

    import torch

    def t(key: str, dtype):
        buf = raw[key]
        if len(buf) == 0:
            return torch.empty(0, dtype=dtype, device=device)
        return torch.frombuffer(bytearray(buf), dtype=dtype).to(device)

    batch = SimpleNamespace(
        x=t("x", torch.float32).view(-1, raw["n_feat"]),
        edge_index=torch.stack([t("edge_src", torch.int64), t("edge_dst", torch.int64)]),
        edge_attr=t("edge_attr", torch.float32).view(-1, 5),
        legal_mask=t("legal_mask", torch.bool),
        batch=t("batch", torch.int64),
        num_graphs=raw["num_graphs"],
    )
    aux = SimpleNamespace(
        legal_idx=t("legal_idx", torch.int64),
        stone_idx=t("stone_idx", torch.int64),
        stone_batch=t("stone_batch", torch.int64),
        legal_counts=t("legal_counts", torch.int64),
        coords=t("coords", torch.int32).view(-1, 2),
    )
    return batch, aux


def augment_axis_states_to_batch(
    games: list,
    transform_indices: list,
    *,
    prune_empty_edges: bool = False,
    threat_features: bool = False,
    relative_stones: bool = False,
    device="cpu",
):
    """Collate + D6-augment GameStates into flat axis batch tensors via Rust.

    Single GIL-released Rust call (`augment_axis_states_to_batch_bytes`) that
    builds each state's axis graph, applies the D6 transform selected by
    ``transform_indices[i]`` (0 = identity, 1..11 = non-identity transforms),
    re-sorts to canonical node order, and collates flat — handing back the same
    native-endian byte buffers as `axis_states_to_batch` PLUS per-state
    permutations.

    Returns (batch, aux, permutations):
      - ``batch`` / ``aux`` mirror `axis_states_to_batch` exactly (the namespace
        `HeXONet._forward_batch_core` consumes + the Rust-precomputed index
        tensors).
      - ``permutations`` is a list of per-state lists where
        ``permutation[new_legal_idx] = old_legal_idx`` — reorder a policy target
        with ``new_policy = old_policy[permutation]``.
    """
    import hexo_rs

    raw = hexo_rs.augment_axis_states_to_batch_bytes(
        games, transform_indices, prune_empty_edges, threat_features, relative_stones
    )
    batch, aux = _raw_axis_bytes_to_ns(raw, device)
    return batch, aux, raw["permutations"]


def augment_axis_states_to_per_graph(
    games: list,
    transform_indices: list,
    *,
    prune_empty_edges: bool = False,
    threat_features: bool = False,
    relative_stones: bool = False,
):
    """Augment a batch of states in one Rust call, sliced into per-graph parts.

    Mirrors `augment_axis_states_to_batch` (single GIL-released call) but slices
    the flat collated result back into one lightweight namespace per state, each
    exposing the attributes a single-graph `_forward_batch_core` consumes
    (x, edge_index, edge_attr, legal_mask, batch, num_graphs=1) plus the
    Rust-precomputed per-graph index tensors (legal_idx, stone_idx, stone_batch,
    all local to the slice). This lets the training loop keep its per-example
    edge-budget micro-batching and node/edge instrumentation unchanged while
    paying only one batched augment call per batch.

    Returns (parts, permutations):
      - ``parts`` is a list of per-graph namespaces in state order.
      - ``permutations[i]`` reorders state i's policy: ``new = old[perm]``.
    """
    from types import SimpleNamespace

    import torch

    batch, aux, perms = augment_axis_states_to_batch(
        games,
        transform_indices,
        prune_empty_edges=prune_empty_edges,
        threat_features=threat_features,
        relative_stones=relative_stones,
    )
    n_graphs = batch.num_graphs
    b = batch.batch
    ei = batch.edge_index
    # Per-node graph id and a node-id base per graph (nodes are contiguous and
    # grouped by graph in the collated layout, edges never cross graphs).
    node_counts = torch.bincount(b, minlength=n_graphs)
    node_offsets = torch.zeros(n_graphs + 1, dtype=torch.long)
    node_offsets[1:] = torch.cumsum(node_counts, 0)
    edge_graph = b[ei[0]] if ei.shape[1] > 0 else torch.empty(0, dtype=torch.long)

    parts = []
    for g in range(n_graphs):
        n0, n1 = int(node_offsets[g]), int(node_offsets[g + 1])
        node_sl = slice(n0, n1)
        emask = edge_graph == g
        local_ei = ei[:, emask] - n0
        local_legal = batch.legal_mask[node_sl]
        # Rebase the Rust-precomputed global legal_idx/stone_idx to local node
        # ids (rather than re-running nonzero per graph) — keeps the "skip the
        # GPU nonzero syncs" property and treats legal and stone indices the
        # same way.
        legal_in_g = (aux.legal_idx >= n0) & (aux.legal_idx < n1)
        local_legal_idx = aux.legal_idx[legal_in_g] - n0
        stone_in_g = (aux.stone_idx >= n0) & (aux.stone_idx < n1)
        local_stone_idx = aux.stone_idx[stone_in_g] - n0
        ns = SimpleNamespace(
            x=batch.x[node_sl],
            edge_index=local_ei,
            edge_attr=batch.edge_attr[emask],
            legal_mask=local_legal,
            batch=torch.zeros(n1 - n0, dtype=torch.long),
            num_graphs=1,
            legal_idx=local_legal_idx,
            stone_idx=local_stone_idx,
            stone_batch=torch.zeros(local_stone_idx.shape[0], dtype=torch.long),
        )
        parts.append(ns)
    return parts, perms


def random_augment(
    data: Data,
    policy_target: "torch.Tensor",
    transform_index: "int | None" = None,
) -> tuple[Data, "torch.Tensor"]:
    """Apply a random non-identity D6 transform to a training example.

    Works for both hex-adjacency graphs and axis-window graphs, in both
    feature layouts (absolute 8/12-dim, relative 7/11-dim — inferred from
    x's width). Transforms coords, recomputes the normalised coord features
    (dims 5,6 absolute / 4,5 relative),
    re-sorts nodes to maintain (q,r) ordering, and reorders policy.
    For axis graphs, also remaps edge_attr axis one-hots and sign-flips distance.

    Args:
        data: PyG Data with x, edge_index, legal_mask, stone_mask, coords.
              May optionally have edge_attr (axis graphs).
        policy_target: 1-D policy tensor aligned with legal nodes in sorted order.
        transform_index: Test/parity seam. When ``None`` (default, production
            behaviour) a uniformly random non-identity transform is chosen.
            When an int in 1..11, forces that specific transform — index ``k``
            here corresponds to ``D6_TRANSFORMS[k]`` in the Rust batch path
            (i.e. ``_transforms[k - 1]`` of the local 11-element list).

    Returns:
        (transformed_data, reordered_policy) tuple.
    """

    import random as _random

    has_edge_attr = getattr(data, "edge_attr", None) is not None

    # D6 transforms on (q, r) — skip identity (index 0).
    # Each entry: (transform_fn, axis_perm, axis_signs)
    # axis_perm/signs describe how WIN_AXES [(1,0),(0,1),(1,-1)] are remapped.
    # axis_perm[old_axis] = new_axis, axis_signs[old_axis] = ±1 (distance sign flip)
    _transforms = [
        (lambda q, r: (-r, q + r),     [1, 2, 0], [ 1, -1,  1]),  # rot60
        (lambda q, r: (-q - r, q),     [2, 0, 1], [-1, -1,  1]),  # rot120
        (lambda q, r: (-q, -r),        [0, 1, 2], [-1, -1, -1]),  # rot180
        (lambda q, r: (r, -q - r),     [1, 2, 0], [-1,  1, -1]),  # rot240
        (lambda q, r: (q + r, -q),     [2, 0, 1], [ 1,  1, -1]),  # rot300
        (lambda q, r: (r, q),          [1, 0, 2], [ 1,  1, -1]),  # reflect
        (lambda q, r: (-q, q + r),     [2, 1, 0], [-1,  1, -1]),  # ref+rot60
        (lambda q, r: (-q - r, r),     [0, 2, 1], [-1, -1, -1]),  # ref+rot120
        (lambda q, r: (-r, -q),        [1, 0, 2], [-1, -1,  1]),  # ref+rot180
        (lambda q, r: (q, -q - r),     [2, 1, 0], [ 1, -1,  1]),  # ref+rot240
        (lambda q, r: (q + r, -r),     [0, 2, 1], [ 1,  1,  1]),  # ref+rot300
    ]
    if transform_index is None:
        choice = _random.choice(_transforms)
    else:
        # transform_index k (1..11) ↔ D6_TRANSFORMS[k] ↔ _transforms[k - 1].
        choice = _transforms[transform_index - 1]
    transform, axis_perm, axis_signs = choice

    coords = data.coords  # (N, 2) int32
    n_total = coords.shape[0]
    # For axis graphs, last node is dummy — exclude from coord transform
    has_dummy = has_edge_attr
    n_real = n_total - 1 if has_dummy else n_total

    q, r = coords[:n_real, 0], coords[:n_real, 1]
    new_q, new_r = transform(q, r)
    if has_dummy:
        # Dummy stays at (0, 0)
        new_coords = torch.zeros_like(coords)
        new_coords[:n_real, 0] = new_q
        new_coords[:n_real, 1] = new_r
    else:
        new_coords = torch.stack([new_q, new_r], dim=1)

    # Sort all nodes by (q, r) to maintain canonical ordering.
    # Stones come first, then legal moves — sort within each group.
    stone_mask = data.stone_mask
    n_stones = stone_mask[:n_real].sum().item()

    # Sort stones by new coords
    stone_coords = new_coords[:n_stones]
    stone_order = torch.argsort(stone_coords[:, 0] * 10000 + stone_coords[:, 1])

    # Sort legal moves by new coords
    n_legal = n_real - n_stones
    legal_coords = new_coords[n_stones:n_real]
    legal_order = torch.argsort(legal_coords[:, 0] * 10000 + legal_coords[:, 1])

    # Combined permutation: stones first, then legal, then dummy (if present)
    perm = torch.cat([stone_order, legal_order + n_stones])
    if has_dummy:
        perm = torch.cat([perm, torch.tensor([n_real])])

    # Apply permutation to all node attributes
    new_x = data.x[perm].clone()
    new_coords = new_coords[perm]
    new_legal_mask = data.legal_mask[perm]
    new_stone_mask = data.stone_mask[perm]

    # Recompute normalised coord features. Index depends on the feature
    # layout: absolute (8 or 12 dims) has them at 5,6; relative_stones
    # (7 or 11 dims) drops to_move, shifting them to 4,5.
    coord_q_idx = 5 if new_x.shape[1] in (8, 12) else 4
    if n_stones > 0:
        sc = new_coords[:n_stones].float()
        centroid_q = sc[:, 0].mean()
        centroid_r = sc[:, 1].mean()
        spread = max(
            (sc[:, 0] - centroid_q).abs().max().item(),
            (sc[:, 1] - centroid_r).abs().max().item(),
            1.0,
        )
    else:
        centroid_q, centroid_r, spread = 0.0, 0.0, 1.0
    # Recompute over real nodes only. The axis-graph dummy node (last, when
    # has_dummy) is a sentinel pinned at coords (0,0) with zero normalised
    # coord features at generation time; recomputing it relative to the stone
    # centroid would push it off (0,0) and — since the dummy participates in
    # message passing — perturb real-node embeddings, diverging from the
    # Rust-built generation graph. Slice [:n_real] to leave the dummy untouched.
    new_x[:n_real, coord_q_idx] = (new_coords[:n_real, 0].float() - centroid_q) / spread
    new_x[:n_real, coord_q_idx + 1] = (new_coords[:n_real, 1].float() - centroid_r) / spread

    # Remap edge indices through the permutation.
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm))
    new_edge_index = inv_perm[data.edge_index]

    # For axis graphs, remap edge_attr: permute axis one-hots and sign-flip distance
    new_edge_attr = None
    if has_edge_attr:
        ea = data.edge_attr.clone()  # (E, 5): [axis0_oh, axis1_oh, axis2_oh, signed_dist, same_color]
        # Permute axis one-hot columns
        new_ea = torch.zeros_like(ea)
        for old_ax in range(3):
            new_ea[:, axis_perm[old_ax]] = ea[:, old_ax]
            # Sign-flip distance for edges on this axis
            mask = ea[:, old_ax] > 0.5  # this axis is active
            if axis_signs[old_ax] == -1:
                new_ea[mask, 3] = -ea[mask, 3]
            else:
                new_ea[mask, 3] = ea[mask, 3]
        # same_color is invariant
        new_ea[:, 4] = ea[:, 4]
        new_edge_attr = new_ea

    kwargs = dict(
        x=new_x,
        edge_index=new_edge_index,
        legal_mask=new_legal_mask,
        stone_mask=new_stone_mask,
        coords=new_coords,
    )
    if new_edge_attr is not None:
        kwargs["edge_attr"] = new_edge_attr
    new_data = Data(**kwargs)

    # Reorder policy to match new legal node ordering
    reordered_policy = policy_target[legal_order]

    # Invariant: policy length must match legal_mask count
    n_legal_out = int(new_legal_mask.sum().item())
    if reordered_policy.shape[0] != n_legal_out:
        raise ValueError(
            f"random_augment: reordered policy length {reordered_policy.shape[0]} "
            f"!= legal_mask count {n_legal_out} "
            f"(n_total={n_total}, n_real={n_real}, n_stones={n_stones}, "
            f"n_legal={n_legal}, input_policy={policy_target.shape[0]}, "
            f"input_legal_mask_sum={int(data.legal_mask.sum().item())})"
        )

    return new_data, reordered_policy


def game_to_graph_batch(
    games: list, *, threat_features: bool = False, relative_stones: bool = False
) -> list[Data]:
    """Convert a batch of GameStates to PyG Data objects using parallel Rust.

    Uses rayon for parallel graph construction across all states.
    Node features: 8-dim float32 absolute layout, 7-dim with relative_stones
    (to_move dropped), +4 threat dims with threat_features (see
    `game_to_graph` for the full schema).
    """
    import hexo_rs

    raw_graphs = hexo_rs.game_to_graph_batch(games, threat_features, relative_stones)
    return [_raw_to_data(g) for g in raw_graphs]


def reconstruct_game_state(data: Data, config) -> "object":
    """Reconstruct a hexo_rs.GameState from a graph Data object + GameConfig.

    Used to recover game_state for legacy TrainingExamples that were
    pickled before PyGameState gained pickle support (and so silently
    dropped game_state). Safe within a single curriculum stage: the
    replay buffer is cleared on stage transitions, so every live example
    was produced under the current stage's GameConfig.

    Works for both hex-adjacency (`game_to_graph`) and axis-window
    (`game_to_axis_graph`) graphs — both use the same node feature schema
    (8-dim, or 12-dim with threat_features; see `game_to_graph`).

    Relative-encoding graphs (7/11-dim) drop the to_move feature, so the
    side to move is recovered from the stone count instead: every real game
    starts with the pre-seeded P1 origin stone and alternates two-placement
    turns with P2 placing first, so with `m = stone_count − 1` placements
    made, P1 is to move iff `m % 4 ∈ {2, 3}` and
    `moves_remaining = 2 − (m % 2)`. The parity-derived moves_remaining is
    cross-checked against the stored moves_remaining feature; a mismatch
    means the position did not come from an origin-seeded alternating game
    (e.g. a synthetic fixture) and reconstruction refuses loudly.

    Args:
        data: PyG Data with attributes `x` (n, 7/8/11/12), `coords` (n, 2),
              `stone_mask` (n,). Absolute feature schema (see `game_to_graph`):
                [0:3] stone-type one-hot: [1,0,0]=P1, [0,1,0]=P2, [0,0,1]=empty
                [3]   current player: +1.0 if P1's turn, -1.0 if P2's turn
                [4]   moves_remaining_this_turn / 2.0
              Relative schema: [0:3] = own/opp/empty (own = side to move),
              [3] = moves_remaining_this_turn / 2.0, no to_move dim.
        config: hexo_rs.GameConfig describing the position's rules.

    Returns:
        hexo_rs.GameState reconstructed via GameState.from_state(...).

    Raises:
        ValueError: if features encode an invalid moves_remaining (not 1 or 2),
                    a relative graph's stone-count parity contradicts its
                    moves_remaining feature, or the data is otherwise
                    inconsistent.
    """
    import hexo_rs

    x = data.x
    coords = data.coords
    stone_mask = data.stone_mask

    if x.shape[0] == 0:
        raise ValueError("reconstruct_game_state: empty graph cannot encode a game")

    relative = x.shape[1] in (7, 11)
    stone_indices = torch.nonzero(stone_mask, as_tuple=False).flatten()

    if relative:
        # Side to move from stone-count parity (origin pre-seeded P1, P2
        # places first, two placements per turn).
        placements = len(stone_indices) - 1
        if placements < 0:
            raise ValueError(
                "reconstruct_game_state: relative graph has no stones; every "
                "real position contains at least the pre-seeded origin stone."
            )
        current_player = "P1" if (placements % 4) // 2 == 1 else "P2"
        moves_remaining_parity = 2 - (placements % 2)
        moves_feat = float(x[0, 3].item())
    else:
        cur_feat = float(x[0, 3].item())
        current_player = "P1" if cur_feat > 0 else "P2"
        moves_remaining_parity = None
        moves_feat = float(x[0, 4].item())

    moves_remaining = int(round(moves_feat * 2.0))
    if moves_remaining not in (1, 2):
        raise ValueError(
            f"reconstruct_game_state: moves_remaining decoded to {moves_remaining} "
            f"from feature value {moves_feat}; must be 1 or 2."
        )
    if moves_remaining_parity is not None and moves_remaining != moves_remaining_parity:
        raise ValueError(
            f"reconstruct_game_state: relative graph stone-count parity implies "
            f"moves_remaining={moves_remaining_parity} but the feature encodes "
            f"{moves_remaining}; position is not from an origin-seeded "
            "alternating game, so side-to-move cannot be derived."
        )

    # Player from stone-type one-hot (cols 0..3). For axis graphs there may be
    # a dummy node — but the dummy is not in stone_mask, so it's harmless.
    opponent = "P2" if current_player == "P1" else "P1"
    stones: list[tuple[tuple[int, int], str]] = []
    for i in stone_indices.tolist():
        # argmax over the 3 stone-type columns picks the player. Column 2
        # (empty) should be ~0 for stones, but argmax is robust to float32
        # noise either way. Absolute: 0=P1, 1=P2. Relative: 0=own, 1=opp.
        cls = int(torch.argmax(x[i, 0:3]).item())
        if cls == 0:
            player = current_player if relative else "P1"
        elif cls == 1:
            player = opponent if relative else "P2"
        else:
            raise ValueError(
                f"reconstruct_game_state: stone node {i} has stone-type one-hot "
                f"{x[i, 0:3].tolist()} which decodes to 'empty'."
            )
        q = int(coords[i, 0].item())
        r = int(coords[i, 1].item())
        stones.append(((q, r), player))

    return hexo_rs.GameState.from_state(
        stones,
        current_player,
        moves_remaining,
        config=config,
    )
