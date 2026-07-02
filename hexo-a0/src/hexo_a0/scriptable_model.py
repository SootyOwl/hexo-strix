"""TorchScript-compatible HeXO model without PyG dependencies.

Replaces GATv2Conv with a pure-PyTorch implementation so the entire
model can be exported via torch.jit.script for use in Rust (tch-rs).

The forward pass takes raw tensors (no PyG Batch object):
  - x: (N, 8) node features
  - edge_index: (2, E) edges
  - legal_mask: (N,) bool
  - stone_mask: (N,) bool
  - batch: (N,) int — graph index per node
  - num_graphs: int
  - edge_attr: (E, 5) float — edge features (axis graphs), or empty tensor (hex)
"""

from typing import Final, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GATv2Layer(nn.Module):
    """Pure-PyTorch GATv2 attention layer (multi-head, concat mode).

    Implements:
        h_src = W_l @ x[src]
        h_dst = W_r @ x[dst]
        e = LeakyReLU(h_src + h_dst [+ edge_feat]) @ a   (per head)
        alpha = softmax(e, index=dst)
        out[dst] += alpha * h_src           (scatter_add)

    When ``edge_dim > 0``, edge features are projected via ``lin_edge`` and
    added into the attention computation, matching PyG's GATv2Conv with
    ``edge_dim``.  Self-loop edge features are filled with the mean of
    incoming edge features per node (matching PyG's ``fill_value='mean'``).
    """

    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        num_heads: int,
        edge_dim: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.edge_dim = edge_dim
        # W_l and W_r: project into (num_heads, head_dim)
        self.lin_l = nn.Linear(in_channels, num_heads * head_dim, bias=True)
        self.lin_r = nn.Linear(in_channels, num_heads * head_dim, bias=True)
        # Attention vector per head
        self.att = nn.Parameter(torch.empty(1, num_heads, head_dim))
        nn.init.xavier_uniform_(self.att)
        # Output bias (matches PyG's GATv2Conv)
        self.bias = nn.Parameter(torch.zeros(num_heads * head_dim))
        # Edge feature projection (matches PyG's lin_edge, bias=False).
        # Always created for TorchScript compatibility; unused when edge_dim=0.
        actual_edge_in = edge_dim if edge_dim > 0 else 1
        self.lin_edge = nn.Linear(actual_edge_in, num_heads * head_dim, bias=False)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = torch.zeros(0)
    ) -> Tensor:
        N = x.shape[0]
        H = self.num_heads
        D = self.head_dim
        has_edge_attr: bool = edge_attr.numel() > 0 and self.edge_dim > 0

        # Initialize edge_feat for TorchScript (must be defined on all paths)
        if has_edge_attr:
            E_total = edge_index.shape[1] + N  # edges + self-loops (added below)
        else:
            E_total = edge_index.shape[1]  # self-loops already in edge_index
        edge_feat = torch.zeros(E_total, H, D, device=x.device, dtype=x.dtype)

        # --- Self-loop edge features (fill_value='mean') ---
        # PyG first removes existing self-loops, then adds new ones with
        # edge features = mean of incoming edge features per destination node.
        # We match this: compute mean of incoming edge_attr per dst, append.
        if has_edge_attr:
            dst_orig = edge_index[1]
            # Remove existing self-loops before computing mean
            not_self_loop = edge_index[0] != edge_index[1]
            clean_dst = dst_orig[not_self_loop]
            clean_attr = edge_attr[not_self_loop]
            clean_edge_index = edge_index[:, not_self_loop]

            # Compute mean edge_attr per destination node for self-loop fill
            attr_sum = torch.zeros(N, edge_attr.shape[1], device=x.device, dtype=edge_attr.dtype)
            attr_count = torch.zeros(N, 1, device=x.device, dtype=edge_attr.dtype)
            attr_sum.scatter_add_(0, clean_dst.unsqueeze(1).expand_as(clean_attr), clean_attr)
            attr_count.scatter_add_(
                0, clean_dst.unsqueeze(1),
                torch.ones_like(clean_dst.unsqueeze(1), dtype=edge_attr.dtype),
            )
            attr_count = attr_count.clamp(min=1)
            self_loop_attr = attr_sum / attr_count  # (N, edge_dim)

            # Append self-loops
            self_loops = torch.arange(N, device=x.device, dtype=edge_index.dtype)
            self_loops = self_loops.unsqueeze(0).expand(2, -1)
            edge_index = torch.cat([clean_edge_index, self_loops], dim=1)
            all_edge_attr = torch.cat([clean_attr, self_loop_attr], dim=0)

            # Project edge features: (E+N, H, D)
            edge_feat = self.lin_edge(all_edge_attr).view(-1, H, D)
        # else: hex mode — self-loops pre-baked in Rust, no action needed

        src, dst = edge_index[0], edge_index[1]

        # Project ALL nodes first, then index per-edge (matches PyG order)
        x_l = self.lin_l(x).view(N, H, D)
        x_r = self.lin_r(x).view(N, H, D)
        h_src = x_l[src]  # (E+N, H, D)
        h_dst = x_r[dst]  # (E+N, H, D)

        # Attention scores: (E, H)
        attn_input = h_src + h_dst
        if has_edge_attr:
            attn_input = attn_input + edge_feat
        e = F.leaky_relu(attn_input, negative_slope=0.2)
        e = (e * self.att).sum(dim=-1)  # (E, H)

        # Softmax over incoming edges per destination node
        # Use scatter-based softmax
        e_max = torch.zeros(N, H, device=x.device, dtype=e.dtype)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce="amax", include_self=False)
        e = e - e_max[dst]
        e_exp = e.exp()
        e_sum = torch.zeros(N, H, device=x.device, dtype=e.dtype)
        e_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(e_exp), e_exp)
        alpha = e_exp / (e_sum[dst] + 1e-16)  # (E, H)

        # Weighted aggregation: scatter_add over destination
        weighted = alpha.unsqueeze(-1) * h_src  # (E, H, D)
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(weighted), weighted)

        return out.reshape(N, H * D) + self.bias  # (N, hidden_dim)


class GINELayer(nn.Module):
    """Pure-PyTorch GINEConv layer (TorchScript-compatible).

    Implements: h_j' = MLP(x_j + edge_attr_jk) for each edge, then
    out_i = sum(h_j' for j in neighbors of i) + (1+eps) * x_i

    Matches PyG's GINEConv with a 2-layer MLP.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp_0 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_1 = nn.Linear(hidden_dim, hidden_dim)
        # Edge feature projection (GINEConv projects edge_attr to match x dim)
        actual_edge_in = edge_dim if edge_dim > 0 else 1
        self.lin_edge = nn.Linear(actual_edge_in, hidden_dim, bias=True)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = torch.zeros(0)
    ) -> Tensor:
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        has_edge_attr: bool = edge_attr.numel() > 0 and self.edge_dim > 0

        # Message: relu(x_j + lin(edge_attr)) — matches PyG GINEConv.message
        msg = x[src]
        if has_edge_attr:
            msg = msg + self.lin_edge(edge_attr)
        msg = F.relu(msg)

        # Aggregate: scatter_add over destinations
        out = torch.zeros(N, self.hidden_dim, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)

        # Add self with learnable epsilon
        out = out + (1.0 + self.eps) * x

        # MLP on aggregated result — matches PyG's self.nn(out)
        out = self.mlp_0(out)
        out = F.relu(out)
        out = self.mlp_1(out)
        return out


class ScriptableHeXONet(nn.Module):
    """TorchScript-compatible HeXO network.

    Same architecture as HeXONet but uses pure-PyTorch layers instead of
    PyG conv layers. Can be exported via torch.jit.script.

    When ``graph_type="axis"``, an ``edge_proj`` layer projects raw 5-dim
    edge features to ``hidden_dim``, and each conv layer receives them.
    """

    # Final-typed flags so torch.jit.script can constant-fold the branch.
    # The forward pass collects per-layer outputs only when ``use_jk`` is on,
    # so the no-JK path stays bit-identical to the pre-JK forward.
    use_jk: Final[bool]
    jk_mode_id: Final[int]  # 0=sum, 1=cat, 2=max; lstm is unsupported here

    def __init__(
        self,
        node_features: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 9,
        num_heads: int = 8,
        policy_hidden: int = 128,
        value_hidden: int = 128,
        graph_type: str = "hex",
        pre_norm: bool = True,
        conv_type: str = "gatv2",
        use_layer_scale: bool = False,
        use_jk: bool = False,
        jk_mode: str = "sum",
    ):
        super().__init__()
        # ScriptableHeXONet supports sum/cat/max. lstm is intentionally out of
        # scope: it adds an LSTM module with internal state which makes the
        # forward pass less TorchScript-friendly and is not currently exercised
        # by training experiments.
        _jk_mode_map = {"sum": 0, "cat": 1, "max": 2}
        if jk_mode not in _jk_mode_map:
            raise ValueError(
                f"ScriptableHeXONet supports jk_mode in {list(_jk_mode_map)!r}, "
                f"got {jk_mode!r}"
            )
        self.jk_mode_id = _jk_mode_map[jk_mode]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.graph_type = graph_type
        self.pre_norm = pre_norm
        self.conv_type = conv_type
        self.use_layer_scale = use_layer_scale
        self.use_jk = use_jk
        # Head input dim: L*H for cat mode, H otherwise.
        head_in_dim = num_layers * hidden_dim if (use_jk and jk_mode == "cat") else hidden_dim
        self.head_in_dim = head_in_dim

        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)

        # Edge feature projection for axis graphs.
        # Always created for TorchScript compatibility; unused for hex.
        edge_dim = 0
        if graph_type == "axis":
            self.edge_proj = nn.Linear(5, hidden_dim)
            edge_dim = hidden_dim
        else:
            self.edge_proj = nn.Linear(1, 1)  # dummy, unused for hex

        # Conv layers with residual + LayerNorm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gine":
                self.convs.append(GINELayer(hidden_dim, edge_dim=edge_dim))
            else:
                head_dim = hidden_dim // num_heads
                self.convs.append(GATv2Layer(hidden_dim, head_dim, num_heads, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Optional LayerScale: per-channel learnable gate on each layer's
        # conv contribution. Default init=1.0 preserves no-LayerScale
        # behaviour. For TorchScript compatibility we always allocate the
        # ParameterList; ``use_layer_scale`` toggles whether it is applied
        # in forward().
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim))
            for _ in range(num_layers)
        ])

        # Jumping Knowledge weights: a learned softmax over conv-layer outputs.
        # Always allocated (like ``layer_scales``) for TorchScript shape
        # stability. ``use_jk`` gates whether they are *applied* in forward.
        # Zero init → softmax → uniform 1/L weighting when use_jk=True.
        self.jk_weights = nn.Parameter(torch.zeros(num_layers))

        # Pre-norm: final norm after last layer. For cat mode the norm is
        # applied per-layer to each h_i BEFORE cat (matches HeXONet), so
        # the shape is always (hidden_dim,) regardless of JK mode.
        if pre_norm:
            self.final_norm = nn.LayerNorm(hidden_dim)
        else:
            self.final_norm = nn.Identity()

        # Policy head MLP
        self.policy_mlp = nn.Sequential(
            nn.Linear(head_in_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, 1),
        )

        # Value head MLP
        self.value_mlp = nn.Sequential(
            nn.Linear(head_in_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        legal_mask: Tensor,
        stone_mask: Tensor,
        batch: Tensor,
        num_graphs: int,
        edge_attr: Tensor = torch.zeros(0),
        legal_idx: Tensor = torch.zeros(0, dtype=torch.long),
        stone_idx: Tensor = torch.zeros(0, dtype=torch.long),
        stone_batch: Tensor = torch.zeros(0, dtype=torch.long),
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass on a batched graph.

        Args:
            x: (N, 8) node features
            edge_index: (2, E) graph connectivity
            legal_mask: (N,) bool — True for legal move nodes
            stone_mask: (N,) bool — True for stone nodes
            batch: (N,) long — graph index per node
            num_graphs: number of graphs in the batch
            edge_attr: optional edge features for axis graphs
            legal_idx: precomputed indices of legal nodes (avoids nonzero)
            stone_idx: precomputed indices of stone nodes (avoids nonzero)
            stone_batch: precomputed batch assignments for stone nodes

        Returns:
            (all_logits, legal_counts, values) where:
              - all_logits: (total_legal,) — concatenated policy logits
              - legal_counts: (num_graphs,) — count of legal moves per graph
              - values: (num_graphs,) — value predictions
        """
        # Representation
        h = self.input_proj(x)

        # Project edge features once for axis graphs; reuse across all layers
        projected_edge_attr = torch.zeros(0)
        if edge_attr.numel() > 0 and self.graph_type == "axis":
            projected_edge_attr = self.edge_proj(edge_attr)

        # Per-layer post-residual outputs collected only when JK is on.
        # Annotated List[Tensor] so TorchScript can type-check the .append.
        hs: List[Tensor] = []
        for conv, norm, layer_scale in zip(self.convs, self.norms, self.layer_scales):
            residual = h
            if self.pre_norm:
                h = norm(h)
                h = conv(h, edge_index, projected_edge_attr)
                if self.use_layer_scale:
                    h = layer_scale * h
                h = h + residual
                h = F.relu(h)
            else:
                h = conv(h, edge_index, projected_edge_attr)
                if self.use_layer_scale:
                    h = layer_scale * h
                h = h + residual
                h = norm(h)
                h = F.relu(h)
            if self.use_jk:
                hs.append(h)

        # Jumping Knowledge aggregation. ``use_jk`` Final[bool] lets
        # torch.jit.script constant-fold the entire branch away when JK is
        # off, so behaviour is bit-identical to the pre-JK forward.
        if self.use_jk:
            if self.jk_mode_id == 0:
                # sum: learned scalar softmax over layers (legacy mode).
                stacked = torch.stack(hs, dim=0)              # (L, N, H)
                w = self.jk_weights.softmax(dim=0).view(-1, 1, 1)
                h = (w * stacked).sum(dim=0)                  # (N, H)
                h = self.final_norm(h)
            elif self.jk_mode_id == 1:
                # cat: apply final_norm to each h_i first, then concat →
                # (N, L*H). Mirrors HeXONet so grafted heads stay equivalent.
                normed: List[Tensor] = []
                for h_i in hs:
                    normed.append(self.final_norm(h_i))
                h = torch.cat(normed, dim=-1)
            else:
                # max: per-channel max over layers → (N, H), then norm.
                h = torch.stack(hs, dim=-1).max(dim=-1)[0]
                h = self.final_norm(h)
        else:
            h = self.final_norm(h)

        # Policy: run MLP on all legal nodes
        if legal_idx.numel() > 0:
            all_logits = self.policy_mlp(h.index_select(0, legal_idx)).squeeze(-1)
        else:
            all_logits = self.policy_mlp(h[legal_mask]).squeeze(-1)

        # Legal counts per graph
        legal_int = legal_mask.long()
        legal_counts = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
        legal_counts.scatter_add_(0, batch, legal_int)

        # Value: scatter_mean over stone nodes
        if stone_idx.numel() > 0 and stone_batch.numel() > 0:
            stone_embs = h.index_select(0, stone_idx)
            sb = stone_batch
        else:
            stone_embs = h[stone_mask]
            sb = batch[stone_mask]

        pooled = torch.zeros(num_graphs, self.head_in_dim, device=x.device, dtype=h.dtype)
        stone_counts = torch.zeros(num_graphs, 1, device=x.device, dtype=h.dtype)
        pooled.scatter_add_(0, sb.unsqueeze(1).expand_as(stone_embs), stone_embs)
        stone_counts.scatter_add_(
            0, sb.unsqueeze(1),
            torch.ones_like(sb.unsqueeze(1), dtype=h.dtype),
        )
        stone_counts = stone_counts.clamp(min=1)
        pooled = pooled / stone_counts

        values = self.value_mlp(pooled).squeeze(-1)

        return all_logits, legal_counts, values


def load_from_hexonet(scriptable: ScriptableHeXONet, hexonet_state_dict: dict) -> None:
    """Load weights from a HeXONet (PyG) state dict into a ScriptableHeXONet.

    Maps PyG conv parameter names to our pure-PyTorch layer names.
    GATv2Conv: lin_l, lin_r, att, bias, lin_edge → same names in GATv2Layer.
    GINEConv: nn.0.weight/bias, nn.2.weight/bias, lin.weight → mlp_0/mlp_1/lin_edge.
    """
    mapping = {}
    for key in hexonet_state_dict:
        new_key = key
        # representation.input_proj.* → input_proj.*
        new_key = new_key.replace("representation.input_proj.", "input_proj.")
        # representation.edge_proj.* → edge_proj.* (axis graphs)
        new_key = new_key.replace("representation.edge_proj.", "edge_proj.")
        # representation.convs.N.* → convs.N.*
        new_key = new_key.replace("representation.convs.", "convs.")
        # representation.norms.N.* → norms.N.*
        new_key = new_key.replace("representation.norms.", "norms.")
        # representation.layer_scales.N → layer_scales.N
        new_key = new_key.replace("representation.layer_scales.", "layer_scales.")
        # representation.jk_weights → jk_weights (Jumping Knowledge softmax)
        new_key = new_key.replace("representation.jk_weights", "jk_weights")
        # representation.final_norm.* → final_norm.* (pre-norm only)
        new_key = new_key.replace("representation.final_norm.", "final_norm.")
        # policy_head.mlp.* → policy_mlp.*
        new_key = new_key.replace("policy_head.mlp.", "policy_mlp.")
        # value_head.mlp.* → value_mlp.*
        new_key = new_key.replace("value_head.mlp.", "value_mlp.")
        # GINEConv-specific: PyG's GINEConv stores the MLP as "nn" and
        # edge projection as "lin"
        # nn.0.weight → mlp_0.weight, nn.2.weight → mlp_1.weight
        new_key = new_key.replace(".nn.0.", ".mlp_0.")
        new_key = new_key.replace(".nn.2.", ".mlp_1.")
        # GINEConv's edge projection: lin.weight → lin_edge.weight
        # (careful not to match lin_l/lin_r from GATv2)
        if ".lin." in new_key:
            new_key = new_key.replace(".lin.", ".lin_edge.")
        mapping[key] = new_key

    new_sd = {}
    for old_key, param in hexonet_state_dict.items():
        new_key = mapping[old_key]
        if new_key in dict(scriptable.named_parameters()) or new_key in dict(scriptable.named_buffers()):
            new_sd[new_key] = param

    missing, unexpected = scriptable.load_state_dict(new_sd, strict=False)
    # edge_proj and lin_edge are dummy layers in hex models (TorchScript requires
    # them to exist but they're never used). layer_scales are always allocated
    # but missing from pre-LayerScale checkpoints — keep their default 1.0 init.
    real_missing = [
        k for k in missing
        if "edge_proj" not in k
        and "lin_edge" not in k
        and "final_norm" not in k
        and "layer_scales" not in k
        and "jk_weights" not in k
    ]
    if real_missing:
        raise RuntimeError(
            f"load_from_hexonet: {len(real_missing)} required parameter(s) "
            f"missing after key translation. The checkpoint likely has an "
            f"unrecognized top-level layout or a graph_type/conv_type mismatch. "
            f"Missing: {real_missing}"
        )
    if unexpected:
        print(f"Unexpected keys: {unexpected}")


def export_torchscript(
    hexonet_state_dict: dict,
    config,
    output_path: str,
    bf16: bool = False,
    fp16: bool = False,
) -> None:
    """Export a HeXONet to TorchScript via ScriptableHeXONet.

    Args:
        hexonet_state_dict: State dict from HeXONet (with _orig_mod. stripped).
        config: ModelConfig dataclass with architecture parameters.
        output_path: Where to save the .pt TorchScript file.
        bf16: If True, convert model to bfloat16.
        fp16: If True, convert model to float16.
    """
    from hexo_a0.config import node_feature_dim

    model = ScriptableHeXONet(
        node_features=node_feature_dim(config),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        policy_hidden=config.policy_hidden,
        value_hidden=config.value_hidden,
        graph_type=config.graph_type,
        pre_norm=config.pre_norm,
        conv_type=getattr(config, "conv_type", "gatv2"),
        use_layer_scale=getattr(config, "use_layer_scale", False),
        use_jk=getattr(config, "use_jk", False),
        jk_mode=getattr(config, "jk_mode", "sum"),
    )
    load_from_hexonet(model, hexonet_state_dict)
    model.eval()
    if bf16:
        model = model.to(torch.bfloat16)
    elif fp16:
        model = model.to(torch.float16)

    scripted = torch.jit.script(model)
    # Write to a temp file then rename atomically to avoid Rust
    # self-play threads reading a half-written model.
    import tempfile, os
    dir_name = os.path.dirname(output_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".pt.tmp")
    os.close(fd)
    scripted.save(tmp_path)
    os.replace(tmp_path, output_path)
    import logging as _logging
    _logging.getLogger("hexo_a0.scriptable_model").debug("Saved TorchScript model to %s", output_path)


