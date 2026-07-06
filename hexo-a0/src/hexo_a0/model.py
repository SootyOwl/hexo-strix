"""GNN Representation Network for HeXO AlphaZero."""

import logging
import re

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GINEConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from hexo_a0.axis_conv import AxisRelationalConv
from hexo_a0.config import ModelConfig, node_feature_dim, legacy_lean_columns

logger = logging.getLogger(__name__)


def legacy_edges_to_lean(edge_index: Tensor, edge_attr: Tensor):
    """Split a legacy axis ``edge_attr`` (E×5 = [axis0, axis1, axis2,
    signed_dist, src_player]) into lean relational inputs: axis edges
    (``edge_type`` in {0,1,2} + unsigned ``edge_dist``) plus a separate
    ``global_edge_index`` for the dummy edges (all-zero axis one-hot).

    Matches ``graph.legacy_to_lean`` / the native Rust lean builder, so a
    relational model fed a legacy self-play/eval graph produces exactly the
    lean result. Native-lean callers pass ``edge_type`` directly and never
    reach this (e.g. training), so it is free for them.
    """
    axis_oh = edge_attr[:, 0:3]
    is_axis = axis_oh.abs().sum(dim=1) > 0.5
    edge_type = axis_oh.argmax(dim=1)[is_axis]
    edge_dist = edge_attr[is_axis, 3].abs().round().long().clamp(min=1)
    return edge_index[:, is_axis], edge_type, edge_dist, edge_index[:, ~is_axis]

# Set of JK modes that go through PyG's JumpingKnowledge module rather than
# the legacy scalar-weighted "sum" path. "cat" additionally changes the
# downstream head input dimension to L * hidden_dim.
_JK_PYG_MODES = ("cat", "max", "lstm")
_JK_VALID_MODES = ("sum",) + _JK_PYG_MODES

# Layer index detector for smart-load grafting. Matches both raw and
# torch.compile-prefixed (`_orig_mod.`) state-dict keys.
_CONV_INDEX_RE = re.compile(r"representation\.convs\.(\d+)\.")


class DedupGINEConv(GINEConv):
    """GINEConv whose per-layer edge linear runs on UNIQUE edge_attr rows.

    Axis-graph edge features are structural (axis one-hot x signed distance x
    window feature): ~13k edges carry only ~91 distinct rows on a real r=8
    board, and `self.lin` (H->H, per layer, per edge) dominates edge-side
    FLOPs. Pass `edge_attr_unique` + `edge_inverse` (from torch.unique on the
    RAW 5-dim attrs, computed once per forward) to project the unique table
    and gather; omit them to get stock GINEConv behavior. State-dict keys are
    unchanged (`lin.*`, `nn.*`) so checkpoints/export stay compatible.
    Equivalent to stock up to GEMM tiling (allclose, not bitwise).
    """

    def forward(self, x, edge_index, edge_attr=None, size=None,
                *, edge_attr_unique=None, edge_inverse=None):
        if edge_attr_unique is None:
            return super().forward(x, edge_index, edge_attr=edge_attr, size=size)
        pre = self.lin(edge_attr_unique).index_select(0, edge_inverse)
        self._dedup_pre_lined = True
        try:
            return super().forward(x, edge_index, edge_attr=pre, size=size)
        finally:
            self._dedup_pre_lined = False

    def message(self, x_j, edge_attr):
        if edge_attr is not None and getattr(self, "_dedup_pre_lined", False):
            return (x_j + edge_attr).relu()
        return super().message(x_j, edge_attr)


class RepresentationNetwork(nn.Module):
    """GATv2Conv-based representation network.

    Encodes a graph observation into per-node embeddings of shape
    (N, hidden_dim) using multi-head attention with residual connections,
    layer normalisation, and optional dropout.

    Args:
        config: ModelConfig instance holding all hyperparameters.

    Raises:
        ValueError: If config.hidden_dim is not divisible by config.num_heads.
    """

    def __init__(self, config: ModelConfig, graph_type: str = "hex") -> None:
        super().__init__()

        if graph_type not in ("hex", "axis"):
            raise ValueError(f"graph_type must be 'hex' or 'axis', got {graph_type!r}")

        # Axis-relational backbone: the 3 hex axes are an edge TYPE partition
        # consumed by tied-weight AxisRelationalConv layers (exactly
        # axis-permutation invariant), instead of the absolute axis one-hot
        # edge feature fed to GINE/GATv2. Gated so the legacy path is untouched.
        self.axis_relational = bool(getattr(config, "axis_relational", False))

        conv_type = getattr(config, "conv_type", "gatv2")
        if not self.axis_relational:
            if conv_type not in ("gatv2", "gine"):
                raise ValueError(
                    f"conv_type must be 'gatv2' or 'gine', got {conv_type!r}"
                )

            if conv_type == "gatv2" and config.hidden_dim % config.num_heads != 0:
                raise ValueError(
                    f"hidden_dim ({config.hidden_dim}) must be divisible by "
                    f"num_heads ({config.num_heads})."
                )

        self.hidden_dim = config.hidden_dim

        # Input projection: node features -> hidden_dim. Base layout is
        # 8-dim absolute or 7-dim with relative_stone_encoding (own/opp
        # stones, to_move dropped); threat_features appends 4 more dims
        # (valid widths: 7, 8, 11, 12).
        in_dim = node_feature_dim(config)
        self.input_proj = nn.Linear(in_dim, config.hidden_dim)

        # Edge feature handling. Relational mode does NOT create edge_proj: it
        # consumes edge_type/edge_dist/global_edge_index directly in each conv.
        if self.axis_relational:
            self.axis_window = int(getattr(config, "axis_window", 8))
            # Legacy→lean node-column map (self-play/eval feed legacy graphs);
            # None when the lean node schema equals legacy (no reduction).
            _cols = legacy_lean_columns(config)
            if _cols is not None:
                self.register_buffer(
                    "_lean_cols", torch.tensor(_cols, dtype=torch.long),
                    persistent=False,
                )
            else:
                self._lean_cols = None
        elif graph_type == "axis":
            self.edge_proj = nn.Linear(5, config.hidden_dim)

        # Stacked conv layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(config.num_layers):
            if self.axis_relational:
                conv = AxisRelationalConv(
                    config.hidden_dim, config.hidden_dim,
                    window=self.axis_window, num_axes=3, use_global=True,
                )
            else:
                conv = self._make_conv(config, conv_type, graph_type)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(config.hidden_dim))

        # Optional LayerScale: per-channel learnable gate on each layer's
        # conv contribution before residual addition. Init=1.0 preserves
        # no-LayerScale behaviour exactly. Grafted layers are reinit'd
        # to ~0 by `graft_state_dict`.
        self.use_layer_scale = bool(getattr(config, "use_layer_scale", False))
        if self.use_layer_scale:
            self.layer_scales = nn.ParameterList([
                nn.Parameter(torch.ones(config.hidden_dim))
                for _ in range(config.num_layers)
            ])

        # Optional Jumping Knowledge aggregation over per-layer outputs.
        # Four modes:
        #   - "sum"  (legacy): scalar softmax weighting per layer,
        #     init=zeros → uniform 1/L weighting. Output dim = H. Backwards-
        #     compatible with the original `gine-mini-jk` checkpoints.
        #   - "cat"  : concatenate per-layer outputs along feature dim. Output
        #     dim = L * H; heads must be sized accordingly.
        #   - "max"  : per-channel max-pool over layers. Output dim = H.
        #   - "lstm" : bi-LSTM attention over per-layer outputs (PyG default
        #     when ``channels`` is given). Output dim = H.
        # See docs/research/2026-05-13-jknet-axis-multiscale-hypothesis.md.
        self.use_jk = bool(getattr(config, "use_jk", False))
        self.jk_mode = str(getattr(config, "jk_mode", "sum"))
        if self.use_jk:
            if self.jk_mode not in _JK_VALID_MODES:
                raise ValueError(
                    f"jk_mode must be one of {_JK_VALID_MODES!r}, "
                    f"got {self.jk_mode!r}."
                )
            if self.jk_mode == "sum":
                # Legacy scalar-weighted sum (init=0 → uniform 1/L softmax).
                self.jk_weights = nn.Parameter(torch.zeros(config.num_layers))
            else:
                # PyG's JumpingKnowledge handles cat/max/lstm natively.
                self.jk = JumpingKnowledge(
                    mode=self.jk_mode,
                    channels=config.hidden_dim,
                    num_layers=config.num_layers,
                )

        self.pre_norm = config.pre_norm

        # Output dimensionality of the representation network as seen by
        # the downstream heads. For "cat" JK this is L * H, otherwise H.
        self.output_dim = self._compute_output_dim(config)

        # Pre-norm: add a final norm after the last layer (since the last
        # conv output isn't normalised before being passed to the heads).
        # For "sum"/"max"/"lstm" and use_jk=False this is a single
        # ``LayerNorm(hidden_dim)`` applied to the (N, H) aggregated output,
        # bit-equivalent to the original pre-JK behaviour.
        # For "cat" mode we deliberately apply the SAME ``LayerNorm(H)`` to
        # each per-layer ``h_i`` independently BEFORE the cat aggregation
        # (i.e. ``cat(norm(h_0), ..., norm(h_{L-1}))``). This keeps the
        # parameter shape at (H,), so the no-JK checkpoint's ``final_norm``
        # weights load directly. It also preserves forward equivalence after
        # head-grafting: the last H columns of the grafted head matrix
        # multiply ``norm(h_{L-1})`` exactly as the no-JK head did, and the
        # zeroed columns wipe out the other ``norm(h_i)`` slots — see
        # ``graft_heads_for_jk_cat``.
        if self.pre_norm:
            self.final_norm = nn.LayerNorm(config.hidden_dim)

        self.activation = nn.ReLU()
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        )

    @staticmethod
    def _compute_output_dim(config: ModelConfig) -> int:
        """Dimensionality of the representation network's output.

        ``cat`` mode concatenates per-layer outputs, so the heads see
        ``L * hidden_dim``. All other modes (and use_jk=False) collapse
        back to ``hidden_dim``.
        """
        if not bool(getattr(config, "use_jk", False)):
            return config.hidden_dim
        mode = str(getattr(config, "jk_mode", "sum"))
        if mode == "cat":
            return config.num_layers * config.hidden_dim
        return config.hidden_dim

    @staticmethod
    def _make_conv(config: ModelConfig, conv_type: str, graph_type: str) -> nn.Module:
        hidden = config.hidden_dim
        edge_dim = hidden if graph_type == "axis" else None

        if conv_type == "gatv2":
            head_dim = hidden // config.num_heads
            kwargs: dict = {}
            if graph_type == "axis":
                kwargs["edge_dim"] = edge_dim
            else:
                kwargs["add_self_loops"] = False
            return GATv2Conv(
                in_channels=hidden, out_channels=head_dim,
                heads=config.num_heads, concat=True, dropout=0.0,
                **kwargs,
            )

        if conv_type == "gine":
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            return GINEConv(mlp, edge_dim=edge_dim)

        raise ValueError(f"Unknown conv_type: {conv_type!r}")

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        edge_type: Tensor | None = None,
        edge_dist: Tensor | None = None,
        global_edge_index: Tensor | None = None,
    ) -> Tensor:
        """Compute node embeddings.

        Args:
            x:                 Node feature matrix of shape (N, node_features).
            edge_index:        Graph connectivity of shape (2, E).
            edge_attr:         Edge feature matrix of shape (E, 5), or None.
                               Legacy (non-axis_relational) mode only.
            edge_type:         Per-edge axis label in {0,1,2}, shape (E,).
                               Axis-relational mode only.
            edge_dist:         Per-edge unsigned hop distance, shape (E,).
                               Axis-relational mode only; clamped into
                               [1, axis_window] before the conv.
            global_edge_index: Optional global-relation edges, shape (2, E_g).
                               Axis-relational mode only.

        Returns:
            Node embedding matrix of shape (N, hidden_dim).
        """
        # A relational model handed a LEGACY graph (edge_attr present, no
        # edge_type) — as self-play leaf inference and the eval collate path
        # still produce — converts to lean inputs on the fly: drop the leaky
        # node columns and split edge_attr into edge_type/edge_dist/global.
        # Native-lean callers (training) pass edge_type and skip this entirely,
        # so it costs the training loop nothing.
        if self.axis_relational and edge_type is None and edge_attr is not None:
            if self._lean_cols is not None:
                x = x.index_select(1, self._lean_cols)
            edge_index, edge_type, edge_dist, global_edge_index = (
                legacy_edges_to_lean(edge_index, edge_attr)
            )
            edge_attr = None

        x = self.input_proj(x)  # (N, hidden_dim)

        # Edge inputs. Relational mode consumes edge_type/edge_dist/global;
        # legacy mode projects edge_attr once and reuses it across layers.
        projected_edge_attr = None
        if self.axis_relational:
            if edge_type is None or edge_dist is None:
                raise ValueError(
                    "Axis-relational model requires lean edge_type+edge_dist or "
                    "a legacy edge_attr to convert, but received neither. Ensure "
                    "the graph builder matches model.axis_relational=True."
                )
            # Defensive clamp into the per-hop embedding table's valid range.
            edge_dist = edge_dist.long().clamp(1, self.axis_window)
        else:
            if hasattr(self, "edge_proj"):
                if edge_attr is None:
                    raise ValueError(
                        "Axis-graph model requires edge_attr but received None. "
                        "Ensure the graph builder matches the model's graph_type."
                    )
                projected_edge_attr = self.edge_proj(edge_attr)  # (E, hidden_dim)
            elif edge_attr is not None:
                raise ValueError(
                    "Hex-graph model received unexpected edge_attr. "
                    "Ensure the graph builder matches the model's graph_type."
                )

        hs: list[Tensor] = [] if self.use_jk else None  # type: ignore[assignment]
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            if self.pre_norm:
                x = norm(x)
                if self.axis_relational:
                    x = conv(x, edge_index, edge_type, edge_dist, global_edge_index)
                else:
                    x = conv(x, edge_index, edge_attr=projected_edge_attr)
                if self.use_layer_scale:
                    x = self.layer_scales[i] * x
                x = x + residual
                x = self.activation(x)
                x = self.dropout(x)
            else:
                if self.axis_relational:
                    x = conv(x, edge_index, edge_type, edge_dist, global_edge_index)
                else:
                    x = conv(x, edge_index, edge_attr=projected_edge_attr)
                if self.use_layer_scale:
                    x = self.layer_scales[i] * x
                x = x + residual
                x = norm(x)
                x = self.activation(x)
                x = self.dropout(x)
            if self.use_jk:
                hs.append(x)

        if self.use_jk:
            if self.jk_mode == "sum":
                stacked = torch.stack(hs, dim=0)                  # (L, N, H)
                w = self.jk_weights.softmax(dim=0).view(-1, 1, 1)
                x = (w * stacked).sum(dim=0)                      # (N, H)
                if self.pre_norm:
                    x = self.final_norm(x)
            elif self.jk_mode == "cat":
                # Apply final_norm (shape H) to each h_i first, then cat.
                # This keeps the no-JK final_norm shape (H) so its trained
                # weights load directly, AND preserves grafted-head forward
                # equivalence at step 0.
                if self.pre_norm:
                    hs = [self.final_norm(h) for h in hs]
                x = self.jk(hs)                                   # (N, L*H)
            else:
                # max / lstm: aggregate first → (N, H), then norm.
                x = self.jk(hs)
                if self.pre_norm:
                    x = self.final_norm(x)
        else:
            if self.pre_norm:
                x = self.final_norm(x)

        return x


class PolicyHead(nn.Module):
    """MLP policy head that scores legal moves from node embeddings.

    Args:
        hidden_dim:    Dimensionality of incoming node embeddings.
        policy_hidden: Hidden size of the intermediate MLP layer.
    """

    def __init__(self, hidden_dim: int, policy_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, 1),
        )

    def forward(self, embeddings: Tensor, legal_mask: Tensor) -> Tensor:
        """Compute logits for legal moves only.

        Accepts either a boolean mask or a 1-D index tensor. Index tensors
        avoid GPU→CPU sync from aten::nonzero that boolean indexing triggers.

        Args:
            embeddings: Node embedding matrix of shape (N, hidden_dim).
            legal_mask: Boolean tensor of shape (N,) or 1-D long index tensor.

        Returns:
            1-D logit tensor of shape (num_legal,).
        """
        if legal_mask.dtype == torch.bool:
            legal_embeddings = embeddings[legal_mask]
        else:
            legal_embeddings = torch.index_select(embeddings, 0, legal_mask)
        logits = self.mlp(legal_embeddings)         # (num_legal, 1)
        return logits.squeeze(-1)                   # (num_legal,)


class ValueHead(nn.Module):
    """Stone-pooled value head that outputs a scalar game value.

    Pools over stone nodes only (not empty legal-move nodes), so the value
    estimate is based on the actual board position, not diluted by hundreds
    of empty candidate cells.

    Args:
        hidden_dim:   Dimensionality of incoming node embeddings.
        value_hidden: Hidden size of the intermediate MLP layer.
    """

    def __init__(self, hidden_dim: int, value_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, embeddings: Tensor, stone_mask: Tensor | None = None) -> Tensor:
        """Compute a scalar value estimate via mean pooling over stones.

        Accepts either a boolean mask or a 1-D index tensor. Index tensors
        avoid GPU→CPU sync from aten::nonzero that boolean indexing triggers.

        Args:
            embeddings: Node embedding matrix of shape (N, hidden_dim).
            stone_mask: Boolean tensor (N,), 1-D long index tensor, or None.

        Returns:
            0-D scalar tensor in [-1, 1].
        """
        if stone_mask is not None:
            if stone_mask.dtype == torch.bool:
                if stone_mask.any():
                    pooled = embeddings[stone_mask].mean(dim=0)
                else:
                    pooled = embeddings.mean(dim=0)
            else:
                # Index tensor — no nonzero sync
                pooled = torch.index_select(embeddings, 0, stone_mask).mean(dim=0)
        else:
            pooled = embeddings.mean(dim=0)
        out = self.mlp(pooled)            # (1,)
        return out.squeeze(-1)            # scalar 0-D tensor


class HeXONet(nn.Module):
    """Full AlphaZero-style network for HeXO: representation + policy + value.

    Args:
        config: ModelConfig instance holding all hyperparameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.representation = RepresentationNetwork(config, graph_type=config.graph_type)
        # Heads see ``representation.output_dim`` — this is ``L * hidden_dim``
        # for jk_mode="cat" and ``hidden_dim`` otherwise. Centralising via the
        # representation property keeps cat-mode head sizing in one place.
        head_in_dim = self.representation.output_dim
        self.policy_head = PolicyHead(head_in_dim, config.policy_hidden)
        self.value_head = ValueHead(head_in_dim, config.value_hidden)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        legal_mask: Tensor,
        stone_mask: Tensor | None = None,
        *,
        edge_attr: Tensor | None = None,
        edge_type: Tensor | None = None,
        edge_dist: Tensor | None = None,
        global_edge_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Convenience single-graph forward.

        Wraps the inputs into a single-element PyG Batch, delegates to
        forward_batch, and unpacks the result.

        Args:
            x:                 Node feature matrix of shape (N, node_features).
            edge_index:        Graph connectivity of shape (2, E).
            legal_mask:        Boolean tensor of shape (N,).
            stone_mask:        Boolean tensor of shape (N,); True for stones.
            edge_attr:         Edge feature matrix of shape (E, 5), or None.
                               Legacy (non-axis_relational) mode only.
            edge_type:         Per-edge axis label in {0,1,2}, shape (E,).
                               Axis-relational mode only.
            edge_dist:         Per-edge unsigned hop distance, shape (E,).
                               Axis-relational mode only.
            global_edge_index: Optional global-relation edges, shape (2, E_g).
                               Axis-relational mode only.

        Returns:
            (policy_logits, value) where policy_logits has shape (num_legal,)
            and value is a 0-D scalar tensor.
        """
        data = Data(x=x, edge_index=edge_index, legal_mask=legal_mask)
        if stone_mask is not None:
            data.stone_mask = stone_mask
        if edge_attr is not None:
            data.edge_attr = edge_attr
        if edge_type is not None:
            data.edge_type = edge_type
        if edge_dist is not None:
            data.edge_dist = edge_dist
        if global_edge_index is not None:
            data.global_edge_index = global_edge_index
        batch = Batch.from_data_list([data])
        policy_list, values = self.forward_batch(batch)
        return policy_list[0], values[0]

    def _forward_batch_core(
        self,
        batch: Batch,
        *,
        legal_idx: Tensor | None = None,
        stone_idx: Tensor | None = None,
        stone_batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Core batched forward: returns flat logits + counts + values (all GPU).

        When ``legal_idx``, ``stone_idx``, and ``stone_batch`` are provided,
        skips the GPU→CPU-sync ``nonzero()`` calls — required for
        torch.compile graph-break-free tracing.

        Returns:
            (all_logits, legal_counts, values_tensor) where all_logits is
            flat (total_legal,), legal_counts is (num_graphs,) long, and
            values_tensor is (num_graphs,).
        """
        if self.representation.axis_relational:
            # Edges never cross subgraph boundaries and PyG applies node-index
            # offsets to both edge_index and global_edge_index, so the axis
            # message passing is correct per-subgraph — batching is transparent.
            # edge_attr is passed too so a legacy collate batch (eval path)
            # auto-converts; native-lean batches (training) carry edge_type.
            embeddings = self.representation(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                edge_type=getattr(batch, "edge_type", None),
                edge_dist=getattr(batch, "edge_dist", None),
                global_edge_index=getattr(batch, "global_edge_index", None),
            )
        else:
            edge_attr = getattr(batch, "edge_attr", None)
            embeddings = self.representation(batch.x, batch.edge_index, edge_attr)
        num_graphs = batch.num_graphs
        device = embeddings.device

        # --- Policy: run MLP on ALL legal nodes at once ---
        if legal_idx is None:
            legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
        all_legal_embeddings = torch.index_select(embeddings, 0, legal_idx)
        all_logits = self.policy_head.mlp(all_legal_embeddings).squeeze(-1)

        # Legal counts per graph (stays on GPU)
        legal_int = batch.legal_mask.long()
        legal_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        legal_counts.scatter_add_(0, batch.batch, legal_int)

        # --- Value: scatter_mean over stone nodes, then MLP all at once ---
        if stone_idx is None:
            stone_mask = batch.stone_mask if hasattr(batch, "stone_mask") and batch.stone_mask is not None else ~batch.legal_mask
            stone_idx = stone_mask.nonzero(as_tuple=False).squeeze(1)
        if stone_batch is None:
            stone_batch = torch.index_select(batch.batch, 0, stone_idx)

        stone_embs = torch.index_select(embeddings, 0, stone_idx)

        pooled = torch.zeros(num_graphs, embeddings.shape[1], device=device)
        stone_counts = torch.zeros(num_graphs, 1, device=device)
        pooled.scatter_add_(0, stone_batch.unsqueeze(1).expand_as(stone_embs), stone_embs)
        stone_counts.scatter_add_(0, stone_batch.unsqueeze(1), torch.ones_like(stone_batch.unsqueeze(1), dtype=torch.float))
        stone_counts = stone_counts.clamp(min=1)
        pooled = pooled / stone_counts

        values_tensor = self.value_head.mlp(pooled).squeeze(-1)

        return all_logits, legal_counts, values_tensor

    def forward_batch(
        self, batch: Batch
    ) -> tuple[list[Tensor], Tensor]:
        """Primary batched forward pass.

        Vectorised: runs the GNN, policy MLP, and value MLP on the full
        batch in single kernel launches, then splits per-graph outputs.

        Args:
            batch: A PyG Batch containing graph data with a ``legal_mask``
                   and optional ``stone_mask`` attribute per graph.

        Returns:
            A tuple (list_of_policy_logits, values_tensor) where
            list_of_policy_logits[i] has shape (num_legal_i,) and
            values_tensor has shape (num_graphs,).
        """
        all_logits, legal_counts, values_tensor = self._forward_batch_core(batch)
        policy_list = list(all_logits.split(legal_counts.tolist()))
        return policy_list, values_tensor


def _zero_init_gine_mlp_final_linear(conv: nn.Module) -> bool:
    """Zero-init the final ``nn.Linear`` of a GINEConv MLP if present.

    Returns True if a Linear was found and zero'd. PyG GINEConv stores
    the MLP under ``conv.nn``; we walk it to find the last Linear so the
    layer's conv contribution is exactly 0 at the moment of grafting.
    GATv2Conv has no analogous "output" linear, so this is a no-op there.
    """
    nn_attr = getattr(conv, "nn", None)
    if nn_attr is None:
        return False
    last_linear: nn.Linear | None = None
    for m in nn_attr.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        return False
    with torch.no_grad():
        last_linear.weight.zero_()
        if last_linear.bias is not None:
            last_linear.bias.zero_()
    return True


def graft_state_dict(
    model: nn.Module,
    ckpt_sd: dict,
    *,
    layer_scale_graft_init: float = 1e-4,
) -> tuple[list[str], list[str], list[int]]:
    """Load a state_dict into ``model``, auto-grafting missing conv layers.

    The checkpoint may have been trained with fewer conv layers than the
    current model (depth extension) or without LayerScale at all. The
    grafting rules:

    - **Legacy layer** (its conv weights ARE in ``ckpt_sd``):
      preserve trained behaviour. If the model has LayerScale, set
      ``layer_scales[i] = 1.0`` *before* loading. The checkpoint may
      then overwrite this (newer LayerScale-aware checkpoints) or leave
      it (older pre-LayerScale checkpoints) — either way the result is
      mathematically equivalent to the trained model.

    - **Grafted layer** (its conv weights are NOT in ``ckpt_sd``):
      identity-at-graft. Zero-init the GINE MLP's final linear (if
      applicable) so ``conv_out == 0``, and set
      ``layer_scales[i] = layer_scale_graft_init`` (default 1e-4) so
      the layer is gated to near-zero contribution. ``load_state_dict``
      with ``strict=False`` will not overwrite these because the keys
      are not in the checkpoint.

    Args:
        model:                    Target model. May be a HeXONet or
                                  any nn.Module exposing
                                  ``representation.convs`` and
                                  optionally ``representation.layer_scales``.
        ckpt_sd:                  Source state dict. Caller is expected
                                  to have already stripped any
                                  ``_orig_mod.`` prefix from torch.compile.
        layer_scale_graft_init:   Init value for grafted layers' LayerScale.
                                  Default 1e-4. With the GINE MLP final linear
                                  zero-initialized, ``layer_scale * conv_out =
                                  1e-4 * 0 = 0`` exactly in fp32 — forward is
                                  still bit-exact identity at step 0. The
                                  non-zero scale matters at step 1: it lets
                                  gradient flow into conv_3 weights via the
                                  chain rule (``∂L/∂conv ∝ scale``), which
                                  Adam normalizes to a useful step direction.
                                  Setting this to 0.0 produces the
                                  double-zero-init dead-stuck failure
                                  (2026-05-09 incident): both ``∂L/∂scale`` and
                                  ``∂L/∂conv`` evaluate to exactly zero, so
                                  layer 3 never trains.

    Returns:
        ``(missing_keys, unexpected_keys, grafted_layer_indices)``.
    """
    rep = getattr(model, "representation", model)

    # Detect existing conv indices in the checkpoint
    indices = set()
    for k in ckpt_sd.keys():
        m = _CONV_INDEX_RE.search(k)
        if m:
            indices.add(int(m.group(1)))
    num_existing = (max(indices) + 1) if indices else 0
    has_layer_scale = hasattr(rep, "layer_scales")
    current_num = len(rep.convs)

    grafted: list[int] = []
    for i in range(current_num):
        if i < num_existing:
            # Legacy layer: preserve bit-equivalence.
            if has_layer_scale:
                with torch.no_grad():
                    rep.layer_scales[i].fill_(1.0)
        else:
            # Grafted layer: identity-at-init.
            grafted.append(i)
            _zero_init_gine_mlp_final_linear(rep.convs[i])
            if has_layer_scale:
                with torch.no_grad():
                    rep.layer_scales[i].fill_(layer_scale_graft_init)

    missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
    return list(missing), list(unexpected), grafted


def graft_heads_for_jk_cat(
    model: nn.Module,
    ckpt_sd: dict,
    *,
    hidden_dim: int,
    num_layers: int,
) -> bool:
    """Graft no-JK head weights into a jk_mode='cat' model in-place.

    When switching a trained checkpoint from ``use_jk=False`` (or
    ``jk_mode='sum'``) to ``jk_mode='cat'``, the first ``Linear`` of each
    head changes shape from ``Linear(hidden_dim, *)`` to
    ``Linear(L * hidden_dim, *)``. To preserve forward equivalence at the
    switch point, we want the new layer to behave identically to the old
    one when fed ``concat(h_0, ..., h_{L-1})``.

    For ``jk_mode='cat'``, PyG's ``JumpingKnowledge('cat')`` concatenates
    ``[h_0, h_1, ..., h_{L-1}]`` along the feature dim. The pre-JK model
    was trained on ``h_{L-1}`` only. So at the moment of grafting we set:

        new_weight[:, :(L-1)*H]   = 0
        new_weight[:, (L-1)*H:]   = old_weight     # = the h_{L-1} columns
        new_bias                  = old_bias

    With this layout, at step 0 the forward output is bit-identical to the
    no-JK model (the zero columns multiply h_0..h_{L-2} to nothing).
    Gradients then flow into those columns over training.

    The function mutates ``ckpt_sd`` (the dict passed by the caller) so
    that the subsequent ``load_state_dict`` lifts the grafted weights in.
    Returns ``True`` if at least one head was grafted, ``False`` if the
    checkpoint already matches the model (no graft needed).

    Args:
        model:       Target HeXONet (must have ``jk_mode='cat'``).
        ckpt_sd:     Checkpoint state dict (caller has stripped any
                     ``_orig_mod.`` prefix already). Mutated in-place:
                     entries for the grafted Linears are replaced with
                     the expanded weight + original bias.
        hidden_dim:  Per-layer hidden dimension ``H`` in the trained
                     model.
        num_layers:  Number of conv layers ``L`` in the new model. The
                     trained checkpoint must have been produced with the
                     same ``L`` (the head graft expands width, not depth).

    Returns:
        True when a graft was applied, False if both heads were already
        the right shape (e.g. the checkpoint was itself produced from a
        cat-mode run and no expansion is needed).
    """
    cat_dim = num_layers * hidden_dim
    keys = (
        "policy_head.mlp.0.weight",
        "policy_head.mlp.0.bias",
        "value_head.mlp.0.weight",
        "value_head.mlp.0.bias",
    )
    # Sanity: all four keys must be present (every checkpoint of HeXONet
    # has both heads).
    missing = [k for k in keys if k not in ckpt_sd]
    if missing:
        raise ValueError(
            "graft_heads_for_jk_cat: required head keys missing from "
            f"checkpoint: {missing}. Are these heads-with-MLP checkpoints?"
        )

    grafted = False
    for head_name in ("policy_head", "value_head"):
        w_key = f"{head_name}.mlp.0.weight"
        ckpt_w = ckpt_sd[w_key]
        old_in = ckpt_w.shape[1]
        if old_in == cat_dim:
            # Already at cat-dim — nothing to do for this head.
            continue
        if old_in != hidden_dim:
            raise ValueError(
                f"graft_heads_for_jk_cat: checkpoint {w_key} has in_features "
                f"{old_in}, expected hidden_dim={hidden_dim} (pre-JK) or "
                f"L*hidden_dim={cat_dim} (already cat-mode). Cannot graft."
            )
        # Zero-init a fresh (out_features, L*hidden_dim) tensor, then drop
        # the old weights into the last hidden_dim columns (= h_{L-1}).
        out_features = ckpt_w.shape[0]
        new_w = torch.zeros(
            out_features, cat_dim,
            dtype=ckpt_w.dtype, device=ckpt_w.device,
        )
        new_w[:, (num_layers - 1) * hidden_dim :] = ckpt_w
        ckpt_sd[w_key] = new_w
        # Bias is unchanged: shape (out_features,) regardless of input dim.
        grafted = True

    return grafted


def graft_input_proj_for_threat_features(ckpt_sd: dict, model_config=None) -> bool:
    """Graft an 8-feature input projection into a threat_features=True model.

    The 4 threat features are appended after the original 8, so the old
    weights occupy columns 0..8 and columns 8..12 are zero-initialised: at
    step 0 the forward output is bit-identical to the 8-dim model (the zero
    columns multiply the new features to nothing). Gradients then flow into
    those columns over training. Mutates ``ckpt_sd`` in place.

    The bias needs no reshape: its shape ``(H,)`` depends only on the output
    dimension, not the input width, so it is left untouched (its presence is
    checked only as a checkpoint-sanity guard).

    ``model_config`` (ModelConfig or its dict form, optional) is a
    belt-and-braces guard: grafting is refused when it enables
    ``relative_stone_encoding`` — the own/opp remap is input-dependent, not
    a column widening, so relative encoding requires a from-scratch run.

    Returns True when a graft was applied, False if the checkpoint is
    already 12-dim. Raises on any other input width.
    """
    if model_config is not None:
        relative = (
            model_config.get("relative_stone_encoding", False)
            if isinstance(model_config, dict)
            else getattr(model_config, "relative_stone_encoding", False)
        )
        if relative:
            raise ValueError(
                "graft_input_proj_for_threat_features: refusing to graft — "
                "model.relative_stone_encoding=true. The own/opp stone remap "
                "is input-dependent (not a column widening), so an "
                "absolute-encoding checkpoint cannot be grafted into a "
                "relative-encoding model; relative_stone_encoding requires a "
                "from-scratch run."
            )
    w_key = "representation.input_proj.weight"
    b_key = "representation.input_proj.bias"
    # Sanity: both input_proj keys must be present (every HeXONet checkpoint
    # has them).
    missing = [k for k in (w_key, b_key) if k not in ckpt_sd]
    if missing:
        raise ValueError(
            "graft_input_proj_for_threat_features: required input_proj keys "
            f"missing from checkpoint: {missing}. Is this a HeXONet checkpoint?"
        )
    w = ckpt_sd[w_key]
    if w.shape[1] == 12:
        return False
    if w.shape[1] in (7, 11):
        raise ValueError(
            f"graft_input_proj_for_threat_features: {w_key} has in_features "
            f"{w.shape[1]}, which is a relative_stone_encoding width. "
            "Relative-encoding models take threat features from scratch; "
            "the input-proj graft only widens absolute 8-dim checkpoints."
        )
    if w.shape[1] != 8:
        raise ValueError(
            f"graft_input_proj_for_threat_features: {w_key} has in_features "
            f"{w.shape[1]}, expected 8 (pre-threat) or 12 (already grafted)"
        )
    new_w = torch.zeros(w.shape[0], 12, dtype=w.dtype, device=w.device)
    new_w[:, :8] = w
    ckpt_sd[w_key] = new_w
    return True


def split_param_groups(
    model: nn.Module,
    weight_decay: float,
    layer_scale_weight_decay: float = 0.0,
) -> list[dict]:
    """Split model parameters into decay / no-decay groups for AdamW.

    No-decay parameters (per standard transformer/CNN best practice):
      - All 1-D tensors (biases, LayerNorm gain).
      - GINEConv's learnable ``eps`` scalar.
      - LayerScale parameters (named ``...layer_scales.{i}``), unless
        ``layer_scale_weight_decay > 0`` in which case they get their
        own group with that coefficient.

    All other parameters (Linear weights, attention matrices, conv MLP
    weights) receive ``weight_decay``.

    Returns a list of param-group dicts suitable for
    ``torch.optim.AdamW(param_groups, ...)``. Two groups when
    ``layer_scale_weight_decay == 0`` (default; layer_scales live in
    no-decay), three groups when ``> 0`` (layer_scales get their own).
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    layer_scale_params: list[nn.Parameter] = []
    decouple_layer_scales = layer_scale_weight_decay > 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "layer_scales" in name:
            if decouple_layer_scales:
                layer_scale_params.append(p)
            else:
                no_decay.append(p)
        elif p.ndim <= 1 or name.endswith(".eps"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups: list[dict] = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if decouple_layer_scales:
        groups.append(
            {"params": layer_scale_params, "weight_decay": layer_scale_weight_decay}
        )
    return groups
