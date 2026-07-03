"""Axis-relational message-passing conv for the axis-window hex graph.

The three hex axes are encoded as an edge *type* (a partition of the edge
set), NOT as an absolute one-hot edge feature. A single GINE conv block is
applied to each axis relation with **tied weights**, and the three relation
outputs are combined by a permutation-symmetric SUM. This makes the module
EXACTLY invariant to any relabelling of the 3 axis labels, so no axis-perm
data augmentation is needed to teach the network that the axes are
interchangeable.

Structure (see ``AxisRelationalConv.forward``)::

    a_k = Conv_shared(x, edges[type==k], dist_embed(dist[type==k]))  k in 0..2
    g   = Conv_global(x, global_edges)          # optional, its OWN weights
    x'  = node_update([x, a_0 + a_1 + a_2 + g]) # symmetric SUM over axes

``Conv_shared`` is ONE ``GINEConv`` instance reused across all 3 axes (tied
weights — the crux). ``dist_embed`` is a learned per-hop embedding table
indexed by the unsigned hop distance (1..window); both directed copies of an
edge (i->j and j->i) carry the same unsigned distance. The optional global /
dummy relation is a 4th edge type with its own untied branch and its own
learned edge embedding, and is not part of the axis sum, so it never breaks
the axis-permutation symmetry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GINEConv


def _make_gine(in_dim: int, out_dim: int, mlp_hidden: int, edge_dim: int) -> GINEConv:
    """Build a GINEConv whose update MLP maps ``in_dim -> out_dim``.

    ``GINEConv`` computes ``mlp((1+eps) * x_i + sum_j relu(x_j + W_e e_ij))``
    with ``aggr='add'`` and projects the edge feature from ``edge_dim`` to
    ``in_dim`` via an internal linear (``W_e``). ``in_dim`` is inferred from
    the MLP's first ``Linear``.
    """
    mlp = nn.Sequential(
        nn.Linear(in_dim, mlp_hidden),
        nn.ReLU(),
        nn.Linear(mlp_hidden, out_dim),
    )
    return GINEConv(mlp, train_eps=True, edge_dim=edge_dim)


class AxisRelationalConv(nn.Module):
    """Tied-weight, axis-permutation-invariant conv over the hex axis graph.

    Args:
        in_dim:         Input node feature dimension.
        out_dim:        Output node feature dimension.
        window:         Max unsigned hop distance; the per-hop embedding table
                        has ``window`` rows indexed by ``dist - 1`` for
                        ``dist in 1..window``.
        num_axes:       Number of axis relations sharing tied weights
                        (default 3, the hex axes).
        use_global:     If True, add a separate global/dummy relation with its
                        own untied branch and its own learned edge embedding.
        dist_embed_dim: Width of the per-hop edge embedding fed to the convs.
                        Defaults to ``in_dim``.
        mlp_hidden:     Hidden width of each GINE update MLP. Defaults to
                        ``out_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        window: int,
        num_axes: int = 3,
        use_global: bool = True,
        dist_embed_dim: int | None = None,
        mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if num_axes < 1:
            raise ValueError(f"num_axes must be >= 1, got {num_axes}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.window = window
        self.num_axes = num_axes
        self.use_global = use_global

        edge_dim = dist_embed_dim if dist_embed_dim is not None else in_dim
        hidden = mlp_hidden if mlp_hidden is not None else out_dim

        # Per-hop distance embedding (unsigned): rows 0..window-1 <-> dist 1..window.
        self.dist_embed = nn.Embedding(window, edge_dim)

        # ONE shared GINE block reused across all axis relations (tied weights).
        self.axis_conv = _make_gine(in_dim, out_dim, hidden, edge_dim)

        if use_global:
            # Separate branch with its OWN weights and its own learned edge
            # feature (broadcast to every global edge).
            self.global_conv = _make_gine(in_dim, out_dim, hidden, edge_dim)
            self.global_edge_embed = nn.Parameter(torch.randn(edge_dim) * 0.1)

        # Combine the (invariant) residual node features with the summed,
        # permutation-symmetric relational messages.
        self.node_update = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_dist: Tensor,
        global_edge_index: Tensor | None = None,
    ) -> Tensor:
        """Compute updated node embeddings.

        Args:
            x:                 Node features, shape ``(N, in_dim)``.
            edge_index:        Axis edges, shape ``(2, E)`` (long).
            edge_type:         Per-edge axis label in ``[0, num_axes)``,
                               shape ``(E,)`` (long).
            edge_dist:         Per-edge unsigned hop distance in
                               ``[1, window]``, shape ``(E,)`` (long).
            global_edge_index: Optional global-relation edges, shape
                               ``(2, E_g)`` (long). Only used when the module
                               was constructed with ``use_global=True``.

        Returns:
            Node embeddings of shape ``(N, out_dim)``.
        """
        # One embedding lookup for all axis edges; masked per relation below.
        dist_feat = self.dist_embed(edge_dist.long() - 1)  # (E, edge_dim)

        # Symmetric SUM over the axis relations with TIED weights. Because
        # GINEConv uses additive aggregation and the same weights are applied
        # to every relation, relabelling the axes only permutes the summands,
        # leaving the sum unchanged (exact axis-permutation invariance).
        agg = self.axis_conv.nn[-1].bias.new_zeros(x.size(0), self.out_dim)
        for k in range(self.num_axes):
            mask = edge_type == k
            ei_k = edge_index[:, mask]
            ef_k = dist_feat[mask]
            agg = agg + self.axis_conv(x, ei_k, ef_k)

        # Optional global/dummy relation: separate untied branch, its own
        # learned edge feature, NOT part of the axis sum -> cannot affect the
        # axis-permutation symmetry.
        if self.use_global and global_edge_index is not None and global_edge_index.numel() > 0:
            g_feat = self.global_edge_embed.unsqueeze(0).expand(
                global_edge_index.size(1), -1
            )
            agg = agg + self.global_conv(x, global_edge_index, g_feat)

        # Combine residual node features with the invariant message sum.
        return self.node_update(torch.cat([x, agg], dim=-1))
