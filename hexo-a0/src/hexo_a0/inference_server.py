"""Inference server for HeXO AlphaZero self-play.

Spawned as a subprocess by the Rust self-play binary. Loads a model
checkpoint, torch.compile's it, and serves inference requests over
stdin/stdout using a binary protocol.

Wire protocol: all little-endian, 6-byte header (u32 magic + u8 version + u8 msg_type).
See task spec for full message definitions.

Protocol v2 added a ``node_dim: u8`` field to the forward-message body header
(after ``has_edge_attr``) so non-8-dim node features (e.g. 12-dim threat
features) survive the wire. The Rust self-play binary spawns this server from
the same checkout, so no rolling version compat is needed — version mismatch
is a hard error.
"""

import argparse
import os
import struct
import sys
import time as _time
from pathlib import Path

import torch
from torch import Tensor

MAGIC = 0x48583034
VERSION = 2
MSG_FORWARD = 0x01
MSG_RELOAD = 0x02
MSG_SHUTDOWN = 0xFF

HEADER_SIZE = 6  # u32 + u8 + u8
HEADER_FMT = "<IBB"


def _install_parent_death_watchdog() -> None:
    """Make sure this server dies when the Rust self-play binary dies.

    Without this, abnormal parent termination (SIGKILL, panic, or a parent of
    the parent — like a test harness — getting cancelled) reparents us to PID
    1 and we'd happily keep occupying GPU memory forever.

    Two layers of defence:

    1. **Linux prctl(PR_SET_PDEATHSIG)**: kernel-level signal delivery the
       moment the *real* parent dies. Near-zero latency.
    2. **ppid poll**: portable fallback that catches the race where the parent
       was already dead before we called prctl, and the macOS/non-Linux case
       where prctl isn't available.

    Both paths call ``os._exit(0)`` (not ``sys.exit``) to skip Python
    finalisers — torch/HIP cleanup can hang once the parent's gone.
    """
    import signal
    import threading

    def _hard_exit(*_args):
        os._exit(0)

    # Install the hard-exit handler first so any kernel-delivered SIGTERM
    # (from prctl below, or from someone running `kill <pid>`) does the right
    # thing instead of being intercepted by Python's default raise-SystemExit.
    try:
        signal.signal(signal.SIGTERM, _hard_exit)
    except Exception:
        pass  # signals unavailable on this platform; ppid watchdog still helps.

    # Layer 1: kernel-delivered SIGTERM when our real parent dies (Linux only).
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            PR_SET_PDEATHSIG = 1
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        except Exception:
            pass  # missing libc / not supported; fall through to layer 2.

    # If we were *already* orphaned before prctl had a chance to fire (parent
    # died between fork and exec), bail out immediately.
    if os.getppid() == 1:
        os._exit(0)

    # Layer 2: portable ppid poll. Two-second cadence is plenty — the cost of
    # an extra two seconds of GPU memory occupancy on parent death is small
    # compared to the cost of a forgotten orphan running indefinitely.
    def _ppid_watchdog():
        while True:
            _time.sleep(2.0)
            if os.getppid() == 1:
                os._exit(0)

    threading.Thread(target=_ppid_watchdog, daemon=True).start()


def _check_protocol_header(magic: int, ver: int) -> None:
    """Validate the 6-byte message header's magic + version fields.

    Raises ``ValueError`` with a clear message on mismatch. Version mismatch
    means the Rust binary and this server come from different checkouts —
    a deployment bug, not something to paper over.
    """
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})")
    if ver != VERSION:
        raise ValueError(
            f"unsupported protocol version {ver} (this server speaks v{VERSION}; "
            "the Rust self_play binary and hexo_a0 must come from the same checkout)"
        )


def _read_exact(stream, n: int) -> bytearray:
    """Read exactly n bytes from stream, raise EOFError on short read."""
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        nbytes = stream.readinto(view[pos:])
        if not nbytes:
            raise EOFError(f"Expected {n} bytes, got {pos}")
        pos += nbytes
    return buf


def _write_forward_response(
    stream, logits: Tensor, legal_counts: Tensor, values: Tensor
) -> None:
    """Write a forward response to the binary stream."""
    total_legal = logits.shape[0]
    num_graphs = values.shape[0]

    buf = bytearray()
    buf.extend(struct.pack(HEADER_FMT, MAGIC, VERSION, MSG_FORWARD))
    buf.extend(struct.pack("<II", total_legal, num_graphs))
    buf.extend(logits.cpu().float().numpy().tobytes())
    buf.extend(legal_counts.cpu().int().numpy().tobytes())
    buf.extend(values.cpu().float().numpy().tobytes())
    stream.write(bytes(buf))
    stream.flush()


def _write_reload_ack(stream, success: bool) -> None:
    """Write a reload ACK to the binary stream."""
    buf = bytearray()
    buf.extend(struct.pack(HEADER_FMT, MAGIC, VERSION, MSG_RELOAD))
    buf.extend(struct.pack("<B", 1 if success else 0))
    stream.write(bytes(buf))
    stream.flush()


def _load_model(args) -> torch.nn.Module:
    """Build and load a ScriptableHeXONet from checkpoint."""
    from hexo_a0.scriptable_model import ScriptableHeXONet, load_from_hexonet

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # The lean/relational schema is authoritative from the checkpoint's embedded
    # model_config (training writes it into model_selfplay.pt), so self-play
    # needs no extra CLI plumbing — the server auto-configures from the ckpt.
    # CLI flags remain as a fallback for standalone/test invocations.
    emb = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    if isinstance(emb, dict):
        for _k in ("axis_relational", "axis_window", "compact_stone_onehot",
                   "node_coords", "moves_scope", "relative_stone_encoding",
                   "threat_features", "value_bins", "value_bin_min",
                   "value_bin_max"):
            if _k in emb:
                setattr(args, _k, emb[_k])

    # In axis_relational mode the model's input_proj is the LEAN width: the
    # wire graph arrives legacy (args.node_dim, e.g. 11) and the shim reduces it
    # to lean before input_proj. Size the model to the lean width accordingly.
    node_features = args.node_dim
    if getattr(args, "axis_relational", False):
        from types import SimpleNamespace
        from hexo_a0.config import node_feature_dim
        node_features = node_feature_dim(SimpleNamespace(
            relative_stone_encoding=getattr(args, "relative_stone_encoding", False),
            threat_features=getattr(args, "threat_features", False),
            compact_stone_onehot=getattr(args, "compact_stone_onehot", False),
            node_coords=getattr(args, "node_coords", True),
            moves_scope=getattr(args, "moves_scope", "node"),
        ))

    model = ScriptableHeXONet(
        node_features=node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        policy_hidden=args.policy_hidden,
        value_hidden=args.value_hidden,
        graph_type=args.graph_type,
        pre_norm=not args.no_pre_norm,
        conv_type=args.conv_type,
        use_jk=getattr(args, "use_jk", False),
        jk_mode=getattr(args, "jk_mode", "sum"),
        axis_relational=getattr(args, "axis_relational", False),
        axis_window=getattr(args, "axis_window", 8),
        relative_stone_encoding=getattr(args, "relative_stone_encoding", False),
        threat_features=getattr(args, "threat_features", False),
        compact_stone_onehot=getattr(args, "compact_stone_onehot", False),
        node_coords=getattr(args, "node_coords", True),
        moves_scope=getattr(args, "moves_scope", "node"),
        value_bins=getattr(args, "value_bins", 0),
        value_bin_min=getattr(args, "value_bin_min", -1.0),
        value_bin_max=getattr(args, "value_bin_max", 1.0),
    )

    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    # Strip _orig_mod. prefixes that torch.compile adds
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned[new_key] = v

    # Try direct load first; fall back to HeXONet key mapping
    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError:
        load_from_hexonet(model, cleaned)

    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    # Use bfloat16 on CUDA
    if device.type == "cuda":
        model = model.to(torch.bfloat16)

    if not args.no_compile:
        compile_kwargs = {}
        if getattr(args, "dynamic_compile", False) or os.environ.get("HEXO_DYNAMIC_COMPILE") == "1":
            compile_kwargs["dynamic"] = True
        # The axis-relational encoder now unifies all edges into fixed-shape
        # scatter buckets (no per-axis boolean masking / nonzero split), so it
        # compiles under fullgraph=True like the legacy path — no graph breaks,
        # no ~3x self-play launch overhead. (The heads' index_select path is
        # taken because _prepare_tensors passes legal_idx/stone_idx.)
        model = torch.compile(model, fullgraph=True, **compile_kwargs)

    return model


def _warmup(
    model: torch.nn.Module, device: torch.device, graph_type: str, node_dim: int = 8
) -> None:
    """Run a warmup forward pass to trigger JIT compilation."""
    n, e = 4, 6
    x = torch.randn(n, node_dim, device=device)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long, device=device
    )
    legal_mask = torch.ones(n, dtype=torch.bool, device=device)
    stone_mask = torch.ones(n, dtype=torch.bool, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)

    if device.type == "cuda":
        x = x.to(torch.bfloat16)

    edge_attr = torch.zeros(0, device=device)
    if graph_type == "axis":
        edge_attr = torch.randn(e, 5, device=device)
        if device.type == "cuda":
            edge_attr = edge_attr.to(torch.bfloat16)

    with torch.no_grad():
        model(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)


def _read_forward_body(stdin, expected_node_dim: int) -> dict:
    """Read forward request body into memory (header already consumed).

    ``expected_node_dim`` is the model's node-feature width (``--node-dim``).
    A wire/model mismatch fails here with a clear error instead of surfacing
    later as an obscure GNN shape error.
    """
    body_header = _read_exact(stdin, 14)
    total_nodes, total_edges, num_graphs = struct.unpack_from("<III", body_header, 0)
    has_edge_attr = body_header[12]
    node_dim = body_header[13]
    if node_dim != expected_node_dim:
        raise ValueError(
            f"wire node_dim {node_dim} != server --node-dim {expected_node_dim}"
        )

    features_bytes = _read_exact(stdin, total_nodes * node_dim * 4)
    edge_src_bytes = _read_exact(stdin, total_edges * 8)
    edge_dst_bytes = _read_exact(stdin, total_edges * 8)
    edge_attr_bytes = _read_exact(stdin, total_edges * 5 * 4) if has_edge_attr else None
    legal_mask_bytes = _read_exact(stdin, total_nodes)
    stone_mask_bytes = _read_exact(stdin, total_nodes)
    batch_bytes = _read_exact(stdin, total_nodes * 4)

    return {
        "total_nodes": total_nodes, "total_edges": total_edges,
        "num_graphs": num_graphs, "has_edge_attr": has_edge_attr,
        "node_dim": node_dim,
        "features": features_bytes, "edge_src": edge_src_bytes,
        "edge_dst": edge_dst_bytes, "edge_attr": edge_attr_bytes,
        "legal_mask": legal_mask_bytes, "stone_mask": stone_mask_bytes,
        "batch": batch_bytes,
    }


# Static buckets ported verbatim from hexo-rs/hexo-mcts/src/inference.rs as a
# fallback when --static-buckets is given. The adaptive bucketizer (default)
# learns its own from the live size distribution.
NODE_BUCKETS = (4096, 16384, 32768, 49152, 65536, 131072, 196608)
EDGE_BUCKETS = (98304, 393216, 786432, 1572864, 3145728, 4718592)
EDGE_BUCKETS_PRUNED = (16384, 65536, 131072, 196608, 262144, 393216, 524288, 1048576)
_BUCKET_OVERFLOWS = 0


def pick_bucket(n: int, buckets: tuple) -> int:
    """Return the smallest bucket >= n; on overflow, pad to a multiple of the largest.

    Mirrors ``pick_bucket`` in ``hexo-rs/hexo-mcts/src/inference.rs``. Sustained
    overflows mean the bucket set is undersized — first five overflows always
    warn, then every 100th.
    """
    global _BUCKET_OVERFLOWS
    for b in buckets:
        if n <= b:
            return b
    largest = buckets[-1]
    padded = -(-n // largest) * largest  # ceil-div * largest
    _BUCKET_OVERFLOWS += 1
    c = _BUCKET_OVERFLOWS
    if c <= 5 or c % 100 == 0:
        sys.stderr.write(
            f"[pick_bucket] WARNING: size {n} exceeds top bucket {largest}, "
            f"padding to {padded} (overflow count: {c}). Consider enlarging buckets.\n"
        )
        sys.stderr.flush()
    return padded


class _AdaptiveBuckets:
    """Observe-then-lock bucketizer with auto re-lock on workload drift.

    Behaviour:

    - ``observe(n)`` is called on EVERY batch (even after lock) and keeps the
      last ``ring_size`` observations in a deque.
    - First lock fires once ``observe_batches`` samples have accumulated:
      buckets are chosen at uniform quantiles of the observed distribution,
      padded by ``headroom_pct``, rounded up to ``quantum``.
    - ``pick(n)`` returns the smallest fitting bucket. Overflows (size above
      the top bucket) pad to ``ceil(n / largest) * largest``.
    - If the post-lock overflow rate exceeds ``relock_threshold`` over at
      least ``observe_batches`` batches, the bucketizer re-picks buckets from
      the current ring buffer. This handles drift (e.g. early-game-only
      observation → games-now-end-game distribution) without manual tuning.
      Limited to ``max_relocks`` total to bound torch.compile re-trace cost.
    """

    def __init__(
        self,
        max_buckets: int,
        headroom_pct: float,
        quantum: int,
        observe_batches: int,
        name: str,
        ring_size: int = 5000,
        relock_threshold: float = 0.10,
        max_relocks: int = 4,
    ) -> None:
        import collections
        self.max_buckets = max_buckets
        self.headroom_pct = headroom_pct
        self.quantum = quantum
        self.observe_batches = max(1, observe_batches)
        self.name = name
        self.observed: "collections.deque[int]" = collections.deque(
            maxlen=max(ring_size, observe_batches),
        )
        self.buckets: list[int] = []
        self.is_ready = False
        self.relock_threshold = relock_threshold
        self.max_relocks = max_relocks
        self._relocks_done = 0
        self._batches_since_lock = 0
        self._overflows_since_lock = 0

    def observe(self, n: int) -> bool:
        """Record one observation; return ``is_ready`` afterwards."""
        self.observed.append(n)
        if not self.is_ready and len(self.observed) >= self.observe_batches:
            self._lock(is_relock=False)
        return self.is_ready

    def _quantum_round_up(self, n: int) -> int:
        return -(-n // self.quantum) * self.quantum

    def _bucket_for(self, n: int) -> int:
        """Bucket lookup with overflow-pad fallback (no side effects)."""
        for b in self.buckets:
            if n <= b:
                return b
        return -(-n // self.buckets[-1]) * self.buckets[-1]

    def _lock(self, *, is_relock: bool) -> None:
        sizes = sorted(self.observed)
        n_obs = len(sizes)
        chosen: set[int] = set()
        # Uniform quantile bins — each bucket k covers an equal share of the
        # observed distribution; rounded up to quantum.
        for k in range(1, self.max_buckets + 1):
            idx = min(n_obs - 1, max(0, int(k * n_obs / self.max_buckets) - 1))
            target = int(sizes[idx] * (1.0 + self.headroom_pct))
            chosen.add(self._quantum_round_up(target))
        # Guarantee a bucket >= the largest observed size.
        chosen.add(self._quantum_round_up(int(sizes[-1] * (1.0 + self.headroom_pct))))
        self.buckets = sorted(chosen)
        self.is_ready = True
        if is_relock:
            self._relocks_done += 1
        self._batches_since_lock = 0
        self._overflows_since_lock = 0
        # Overhead diagnostic against the observed distribution.
        total_real = sum(sizes)
        total_padded = sum(self._bucket_for(s) for s in sizes)
        overhead_pct = (total_padded - total_real) / total_real * 100.0 if total_real else 0.0
        verb = "RE-LOCKED" if is_relock else "LOCKED"
        suffix = f" (relock #{self._relocks_done}/{self.max_relocks})" if is_relock else ""
        sys.stderr.write(
            f"[adaptive_buckets:{self.name}] {verb} from {n_obs} observations: "
            f"buckets={self.buckets} mean_overhead={overhead_pct:.1f}%{suffix}\n"
        )
        sys.stderr.flush()

    def pick(self, n: int) -> int:
        assert self.is_ready, "pick() called before _lock()"
        self._batches_since_lock += 1
        import bisect
        idx = bisect.bisect_left(self.buckets, n)
        if idx < len(self.buckets):
            return self.buckets[idx]
        largest = self.buckets[-1]
        padded = -(-n // largest) * largest
        self._overflows_since_lock += 1
        if (
            self._relocks_done < self.max_relocks
            and self._batches_since_lock >= self.observe_batches
            and self._overflows_since_lock / self._batches_since_lock >= self.relock_threshold
        ):
            self._lock(is_relock=True)
        return padded


def _prepare_tensors(
    body: dict,
    device: torch.device,
    transfer_stream: "torch.cuda.Stream | None",
    padded: bool = False,
    node_buckets: "_AdaptiveBuckets | None" = None,
    edge_buckets: "_AdaptiveBuckets | None" = None,
) -> tuple[tuple, int]:
    """Deserialize a forward request body into GPU tensors.

    When *transfer_stream* is not None, all H2D copies are issued on that
    stream so they can overlap with a forward pass running on the default
    stream.  The caller must synchronize *transfer_stream* before using the
    returned tensors for compute.

    When ``padded`` is True, a "ghost" graph (all-zero features, false masks,
    self-loops only) is appended at batch index ``real_num_graphs`` so that
    ``total_nodes`` and ``total_edges`` round up to one of a small fixed set of
    bucket sizes — stabilising the CUDA / HIP caching allocator and letting
    ``torch.compile`` reuse a single specialised kernel across batches.

    Returns ``(tensor_tuple, real_num_graphs)``. Callers must slice
    ``values`` and ``legal_counts`` outputs to ``real_num_graphs`` before
    responding; ``all_logits`` is already correctly sized because the ghost
    contributes 0 legal positions.
    """
    real_nodes = body["total_nodes"]
    real_edges = body["total_edges"]
    real_num_graphs = body["num_graphs"]
    node_dim = body["node_dim"]

    if padded:
        if node_buckets is not None:
            target_nodes = node_buckets.pick(real_nodes + 1)
            target_edges = edge_buckets.pick(real_edges + 1)
        else:
            target_nodes = pick_bucket(real_nodes + 1, NODE_BUCKETS)
            # Same heuristic as Rust: pruned axis graphs have edge/node ratio
            # ~3.5-8x, unpruned ~20-28x; switch bucket set on observed ratio.
            edge_ratio = (real_edges // real_nodes) if real_nodes > 0 else 0
            edge_bucket_set = EDGE_BUCKETS_PRUNED if edge_ratio <= 10 else EDGE_BUCKETS
            target_edges = pick_bucket(real_edges + 1, edge_bucket_set)
    else:
        target_nodes = real_nodes
        target_edges = real_edges

    ctx = torch.cuda.stream(transfer_stream) if transfer_stream is not None else _nullctx()

    with ctx:
        # Note: we avoid pin_memory() here. Pinned allocations grow
        # monotonically (never returned to OS) and with variable-sized
        # batches the pinned pool slowly balloons. The double-buffered
        # transfer stream already hides most of the H2D latency.
        real_features = torch.frombuffer(body["features"], dtype=torch.float32).reshape(real_nodes, node_dim)
        real_edge_src = torch.frombuffer(body["edge_src"], dtype=torch.int64)
        real_edge_dst = torch.frombuffer(body["edge_dst"], dtype=torch.int64)
        real_legal = torch.frombuffer(body["legal_mask"], dtype=torch.uint8).to(dtype=torch.bool)
        real_stone = torch.frombuffer(body["stone_mask"], dtype=torch.uint8).to(dtype=torch.bool)
        real_batch = torch.frombuffer(body["batch"], dtype=torch.int32).to(dtype=torch.long)
        if body["has_edge_attr"] and body["edge_attr"] is not None:
            real_edge_attr = torch.frombuffer(body["edge_attr"], dtype=torch.float32).reshape(real_edges, 5)
        else:
            real_edge_attr = None

        if padded:
            ghost_nodes = target_nodes - real_nodes
            ghost_edges = target_edges - real_edges
            assert ghost_nodes >= 1 and ghost_edges >= 1

            features_cpu = torch.zeros((target_nodes, node_dim), dtype=torch.float32)
            features_cpu[:real_nodes] = real_features
            features = features_cpu.to(device)

            edge_src_cpu = torch.empty(target_edges, dtype=torch.int64)
            edge_dst_cpu = torch.empty(target_edges, dtype=torch.int64)
            edge_src_cpu[:real_edges] = real_edge_src
            edge_dst_cpu[:real_edges] = real_edge_dst
            # Spread ghost edges as self-loops across the ghost-node range so no
            # single ghost node accumulates a degenerate in-degree.
            ghost_self_loops = (
                torch.arange(ghost_edges, dtype=torch.int64) % ghost_nodes + real_nodes
            )
            edge_src_cpu[real_edges:] = ghost_self_loops
            edge_dst_cpu[real_edges:] = ghost_self_loops
            edge_index = torch.stack([edge_src_cpu, edge_dst_cpu], dim=0).to(device)

            legal_cpu = torch.zeros(target_nodes, dtype=torch.bool)
            legal_cpu[:real_nodes] = real_legal
            legal_mask = legal_cpu.to(device)

            stone_cpu = torch.zeros(target_nodes, dtype=torch.bool)
            stone_cpu[:real_nodes] = real_stone
            stone_mask = stone_cpu.to(device)

            batch_cpu = torch.empty(target_nodes, dtype=torch.long)
            batch_cpu[:real_nodes] = real_batch
            batch_cpu[real_nodes:] = real_num_graphs  # ghost batch index
            batch_vec = batch_cpu.to(device)

            if real_edge_attr is not None:
                ea_cpu = torch.zeros((target_edges, 5), dtype=torch.float32)
                ea_cpu[:real_edges] = real_edge_attr
                edge_attr = ea_cpu.to(device)
            else:
                edge_attr = torch.zeros(0, device=device)

            num_graphs = real_num_graphs + 1
        else:
            features = real_features.to(device)
            edge_index = torch.stack([real_edge_src, real_edge_dst], dim=0).to(device)
            legal_mask = real_legal.to(device)
            stone_mask = real_stone.to(device)
            batch_vec = real_batch.to(device)
            edge_attr = (
                real_edge_attr.to(device) if real_edge_attr is not None
                else torch.zeros(0, device=device)
            )
            num_graphs = real_num_graphs

        if device.type == "cuda":
            features = features.to(torch.bfloat16)
            if edge_attr.numel() > 0:
                edge_attr = edge_attr.to(torch.bfloat16)

        # Compute index tensors over the FULL (possibly padded) masks. Ghost
        # nodes are False in legal_mask/stone_mask, so they are naturally
        # excluded — same effect as Rust's "iterate real_nodes only".
        legal_idx = torch.where(legal_mask)[0]
        stone_idx = torch.where(stone_mask)[0]
        stone_batch = batch_vec[stone_idx]

    tensors = (features, edge_index, legal_mask, stone_mask, batch_vec,
               num_graphs, edge_attr, legal_idx, stone_idx, stone_batch)
    return tensors, real_num_graphs


class _nullctx:
    """No-op context manager for CPU path."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


def _forward_and_dispatch(
    tensors: tuple, model: torch.nn.Module,
    transfer_stream: "torch.cuda.Stream | None",
    writer_queue: "queue.Queue",
    real_num_graphs: int,
) -> tuple[float, float, float]:
    """Wait for H2D transfer, launch forward, hand off response to writer.

    Records a CUDA event right after the model launch and enqueues
    ``("forward", event, logits, legal_counts, values)`` for the writer
    thread, which event-waits and then serialises + writes the response.

    Returns ``(stream_sync_ms, launch_ms, dispatch_ms)``. Note that
    ``launch_ms`` measures the model's *Python-side* call time only —
    actual GPU compute happens asynchronously and is paid for by the
    writer thread (or by the next iteration's stream_sync).
    """
    on_cuda = transfer_stream is not None

    t_sync_start = _time.perf_counter()
    if on_cuda:
        torch.cuda.current_stream().wait_stream(transfer_stream)
    t_sync_end = _time.perf_counter()

    with torch.no_grad():
        all_logits, legal_counts, values = model(*tensors)

    # If the inputs were ghost-padded, the last bin of legal_counts/values
    # belongs to the ghost — drop it before responding. ``all_logits`` is
    # already correctly sized because the ghost contributes 0 legal positions.
    if legal_counts.shape[0] != real_num_graphs:
        legal_counts = legal_counts[:real_num_graphs]
        values = values[:real_num_graphs]

    event = torch.cuda.Event() if on_cuda else None
    if event is not None:
        event.record()
    t_launch_end = _time.perf_counter()

    writer_queue.put(("forward", event, all_logits, legal_counts, values))
    t_dispatch_end = _time.perf_counter()

    return (
        (t_sync_end - t_sync_start) * 1000.0,
        (t_launch_end - t_sync_end) * 1000.0,
        (t_dispatch_end - t_launch_end) * 1000.0,
    )




def main() -> None:
    # Configure the CUDA/HIP caching allocator BEFORE the device context is
    # initialised. This inference subprocess shares the GPU with the training
    # process; the allocator config (gc threshold + split-size, optional
    # expandable_segments) is the real safety net against cross-process OOM and
    # replaces the old per-forward empty_cache(). Must run before any
    # torch.cuda.* / .to(cuda) below — the env var is read once at first device
    # touch.
    from hexo_a0.gpu_memory import configure_cuda_alloc
    configure_cuda_alloc()

    # Isolate our torch.compile/inductor cache from the trainer's. We inherit
    # the trainer's env (TORCHINDUCTOR_CACHE_DIR=.../train) when spawned by the
    # Rust self-play binary; a shared kernel cache between the two compiling
    # processes on the one APU is the leading suspect for the 2026-06-02 compile
    # NaN. FORCE our own dir (override the inherited trainer value);
    # HEXO_INDUCTOR_CACHE_DIR_INFER overrides if explicitly set.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.environ.get(
        "HEXO_INDUCTOR_CACHE_DIR_INFER", "/tmp/torchinductor_hexo/infer"
    )

    # Bind us to the parent's lifetime before doing anything heavyweight.
    # A torch.compile pass can take minutes; if the parent dies during that
    # window we want to die too, not finish compiling against a dead parent
    # and then start serving requests no one will read.
    _install_parent_death_watchdog()

    parser = argparse.ArgumentParser(description="HeXO inference server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--policy-hidden", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--graph-type", default="hex", choices=["hex", "axis"])
    parser.add_argument(
        "--node-dim", type=int, default=8,
        help="Node-feature dimension of the model/checkpoint: 8 absolute, "
             "7 relative stone encoding, +4 with threat features (12/11). "
             "Drives model construction and warmup input width; "
             "must match the node_dim the Rust side sends in v2 forward "
             "messages.",
    )
    parser.add_argument("--conv-type", default="gatv2", choices=["gatv2", "gine"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-pre-norm", action="store_true")
    parser.add_argument(
        "--use-jk", action="store_true",
        help="Enable Jumping Knowledge aggregation in the inference model. "
             "Must match the training config used to produce the checkpoint.",
    )
    parser.add_argument(
        "--jk-mode", default="sum", choices=["sum", "cat", "max"],
        help="JK aggregation mode (only honored when --use-jk is set). "
             "'lstm' is not supported in ScriptableHeXONet — use --jk-mode "
             "sum/cat/max for self-play.",
    )
    # --- D6-invariant lean schema (must match the training config) ---
    parser.add_argument(
        "--axis-relational", action="store_true",
        help="Use the D6-invariant axis-relational encoder (edge-type partition "
             "+ tied-weight AxisRelationalConv). The server converts the legacy "
             "wire graph to lean inputs internally; must match the checkpoint.",
    )
    parser.add_argument(
        "--axis-window", type=int, default=8,
        help="Max unsigned hop for the axis-relational distance embedding "
             "(only with --axis-relational; must be >= win_length-1).",
    )
    parser.add_argument(
        "--relative-stone-encoding", action="store_true",
        help="Relative (own/opp) stone encoding — needed for the lean-column map.",
    )
    parser.add_argument(
        "--threat-features", action="store_true",
        help="Threat node features present — needed for the lean-column map.",
    )
    parser.add_argument(
        "--compact-stone-onehot", action="store_true",
        help="Lean schema: the redundant 'empty' stone one-hot dim is dropped.",
    )
    parser.add_argument(
        "--no-node-coords", dest="node_coords", action="store_false", default=True,
        help="Lean schema: norm-q/r node coordinates are dropped.",
    )
    parser.add_argument(
        "--moves-scope", default="node", choices=["node", "graph"],
        help="moves-remaining scope in the lean node schema.",
    )
    # --- Distributional (binned) value head (must match the checkpoint) ---
    parser.add_argument(
        "--value-bins", type=int, default=0,
        help="Number of value bins for the distributional (C51-style) value "
             "head. 0 = scalar tanh head (legacy). Normally auto-configured "
             "from the checkpoint's embedded model_config.",
    )
    parser.add_argument(
        "--value-bin-min", type=float, default=-1.0,
        help="Lower edge of the value-bin center grid (only with --value-bins).",
    )
    parser.add_argument(
        "--value-bin-max", type=float, default=1.0,
        help="Upper edge of the value-bin center grid (only with --value-bins).",
    )
    parser.add_argument(
        "--dynamic-compile", action="store_true",
        help="Pass dynamic=True to torch.compile so Dynamo traces with "
             "symbolic shapes from the start (no per-shape specialisation). "
             "Trades per-call kernel speed for zero recompile cost on shape "
             "changes — usually a wash or regression for stable workloads.",
    )
    parser.add_argument(
        "--padded-inference", action="store_true",
        help="Pad inputs to bucket shapes (mirrors Rust forward_graphs_padded) "
             "so torch.compile sees a small fixed set of shapes.",
    )
    parser.add_argument(
        "--static-buckets", action="store_true",
        help="Use the static NODE_BUCKETS/EDGE_BUCKETS from Rust instead of "
             "the adaptive bucketizer (which learns the bucket set from live "
             "size distribution). Only relevant with --padded-inference.",
    )
    parser.add_argument(
        "--max-buckets", type=int, default=6,
        help="Adaptive bucketizer: cap on distinct bucket sizes per dimension "
             "(node, edge). When exceeded, closest-adjacent pair is merged.",
    )
    parser.add_argument(
        "--bucket-headroom-pct", type=float, default=0.10,
        help="Adaptive bucketizer: padding headroom above each chosen bucket "
             "size (default 10%%).",
    )
    parser.add_argument(
        "--bucket-observe-batches", type=int, default=1000,
        help="Adaptive bucketizer: minimum number of batches to observe "
             "before locking bucket sizes (default 1000 ~= 15s of self-play; "
             "needs to be long enough to cover end-of-game positions, not just "
             "early-game ones). Bucketizer auto re-locks if the post-lock "
             "overflow rate exceeds --bucket-relock-threshold, so under-sizing "
             "is self-correcting. Use 1 in tests.",
    )
    parser.add_argument(
        "--bucket-relock-threshold", type=float, default=0.10,
        help="Adaptive bucketizer: fraction of post-lock batches that must "
             "overflow the top bucket before triggering a re-lock from the "
             "ring buffer (default 10%%). Capped at 4 re-locks per run.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model = _load_model(args)

    # Warmup
    _warmup(model, device, args.graph_type, args.node_dim)

    # Redirect text-mode stdout to stderr so that any stray print(),
    # library warning, or unhandled traceback goes to the log pipe
    # instead of corrupting the binary protocol on stdout.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    sys.stdout = sys.stderr

    # Signal ready
    sys.stderr.write("READY\n")
    sys.stderr.flush()

    # Use a background thread to pre-read the next request while the GPU
    # is processing the current one. This overlaps pipe I/O with GPU compute.
    import queue
    import threading

    request_queue: queue.Queue = queue.Queue(maxsize=1)
    # Writer queue: ("forward", event, logits, legal_counts, values) or
    # ("reload_ack", success_bool) or None (shutdown). Bounded so a slow
    # writer applies backpressure rather than letting GPU outputs pile up.
    # Sized to allow the main dispatch thread to launch several batches ahead
    # of the writer's event-sync — at 4 the main thread blocks every time the
    # writer falls one batch behind. 16 gives ~300ms of in-flight forwards.
    writer_queue: queue.Queue = queue.Queue(maxsize=16)
    writer_perf = {"count": 0, "total_ms": 0.0, "wait_ms": 0.0}
    writer_perf_lock = threading.Lock()

    def _writer_thread() -> None:
        """Serialise model outputs and write them to stdout in order."""
        while True:
            item = writer_queue.get()
            if item is None:
                return
            kind = item[0]
            t0 = _time.perf_counter()
            if kind == "forward":
                _, event, logits, legal_counts, values = item
                if event is not None:
                    event.synchronize()
                t_synced = _time.perf_counter()
                _write_forward_response(stdout, logits, legal_counts, values)
                t_done = _time.perf_counter()
                with writer_perf_lock:
                    writer_perf["count"] += 1
                    writer_perf["wait_ms"] += (t_synced - t0) * 1000.0
                    writer_perf["total_ms"] += (t_done - t0) * 1000.0
            elif kind == "reload_ack":
                _write_reload_ack(stdout, success=item[1])
            else:
                sys.stderr.write(f"Unknown writer queue item: {kind!r}\n")

    def _reader_thread():
        """Read requests from stdin and enqueue them."""
        while True:
            try:
                header = _read_exact(stdin, HEADER_SIZE)
            except EOFError:
                request_queue.put(None)
                return
            magic, ver, msg_type = struct.unpack(HEADER_FMT, header)
            try:
                _check_protocol_header(magic, ver)
            except ValueError as e:
                sys.stderr.write(f"protocol error: {e}\n")
                sys.stderr.flush()
                request_queue.put(None)
                return
            if msg_type == MSG_SHUTDOWN:
                request_queue.put(None)
                return
            # Pre-read the full request body into memory
            if msg_type == MSG_FORWARD:
                try:
                    body = _read_forward_body(stdin, args.node_dim)
                except ValueError as e:
                    # Unrecoverable mid-stream (framing is lost); die loudly.
                    sys.stderr.write(f"protocol error: {e}\n")
                    sys.stderr.flush()
                    request_queue.put(None)
                    return
                request_queue.put(("forward", body))
            elif msg_type == MSG_RELOAD:
                path_len = struct.unpack("<I", _read_exact(stdin, 4))[0]
                path_bytes = _read_exact(stdin, path_len)
                request_queue.put(("reload", path_bytes.decode("utf-8")))
            else:
                request_queue.put(None)
                return

    reader = threading.Thread(target=_reader_thread, daemon=True)
    reader.start()
    writer = threading.Thread(target=_writer_thread, daemon=True)
    writer.start()

    # Main loop — double-buffered: prepare batch N+1's tensors on a transfer
    # stream while batch N's forward pass runs on the default (compute) stream.
    #
    # Perf logging is a 100-batch rolling window: every accumulator below is
    # reset after each `[perf]` line, so the printed averages reflect only
    # the most recent window (not torch.compile cold start + every batch
    # ever). `_perf_count` is cumulative — it drives the `batches=N` line so
    # users can still see total progress.
    _perf_count = 0
    _perf_window_total_ms = 0.0
    _perf_window_gpu_ms = 0.0
    _perf_window_sync_ms = 0.0
    _perf_window_fwd_ms = 0.0
    _perf_window_write_ms = 0.0
    _perf_window_graphs = 0
    _perf_window_nodes = 0

    transfer_stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    # Pressure-triggered empty_cache(): variable-shape inference makes the HIP
    # caching allocator reserve a block per distinct (node,edge) size and never
    # give it back (gc_threshold is inert on unified memory), so the reserve
    # creeps until the kernel OOM-kills self-play after hours. BUT clearing on a
    # batch cadence is a ~3x throughput regression on this unified-memory APU:
    # empty_cache() returns GTT to the driver and forces EVERY process (this
    # worker + the trainer) to re-fault pages back in. So instead of clearing on
    # a fixed schedule, clear ONLY when the allocator reserve is genuinely high
    # (a real leak), checked cheaply (memory_reserved(), no sync) every N batches.
    # In steady state (reserve ~1-2 GB, tens of GB free) it NEVER fires => zero
    # churn; it only kicks in if the reserve balloons past the threshold.
    #   HEXO_CACHE_CLEAR_RESERVED_MB : clear when reserved exceeds this (default
    #     8192; <=0 disables clearing entirely)
    #   HEXO_CACHE_CLEAR_CHECK_EVERY : how often to test the threshold (default 128)
    _clear_threshold_mb = float(os.environ.get("HEXO_CACHE_CLEAR_RESERVED_MB", "8192"))
    _clear_check_every = int(os.environ.get("HEXO_CACHE_CLEAR_CHECK_EVERY", "128"))
    _clear_check = 0
    _clear_on_cuda = device.type == "cuda" and _clear_threshold_mb > 0

    # Adaptive bucketizer (default when --padded-inference is set).
    if args.padded_inference and not args.static_buckets:
        node_bucketizer = _AdaptiveBuckets(
            max_buckets=args.max_buckets,
            headroom_pct=args.bucket_headroom_pct,
            quantum=4096,
            observe_batches=args.bucket_observe_batches,
            name="nodes",
            relock_threshold=args.bucket_relock_threshold,
        )
        edge_bucketizer = _AdaptiveBuckets(
            max_buckets=args.max_buckets,
            headroom_pct=args.bucket_headroom_pct,
            quantum=16384,
            observe_batches=args.bucket_observe_batches,
            name="edges",
            relock_threshold=args.bucket_relock_threshold,
        )
    else:
        node_bucketizer = None
        edge_bucketizer = None

    def _handle_reload(path: str) -> None:
        nonlocal model
        try:
            args.checkpoint = path
            model = _load_model(args)
            _warmup(model, device, args.graph_type, args.node_dim)
            print(f"Model reloaded from {path}", file=sys.stderr, flush=True)
            writer_queue.put(("reload_ack", True))
        except Exception as e:
            print(f"Reload failed: {e}", file=sys.stderr, flush=True)
            writer_queue.put(("reload_ack", False))

    def _next_forward_body():
        """Block until we get a forward request, handling reloads inline."""
        while True:
            item = request_queue.get()
            if item is None:
                return None
            if item[0] == "reload":
                _handle_reload(item[1])
                continue
            return item[1]  # forward body

    # State for double-buffering: when not None, tensors for the next
    # batch have already been prepared on the transfer stream.
    prefetched_body = None
    prefetched_tensors = None
    prefetched_real_n = 0

    _perf_window_prepare_ms = 0.0
    PERF_WINDOW = 100

    while True:
        t0 = _time.perf_counter()

        if prefetched_tensors is not None:
            # Tensors were pre-prepared during the previous forward pass
            body = prefetched_body
            tensors = prefetched_tensors
            real_n = prefetched_real_n
            prefetched_body = None
            prefetched_tensors = None
            t_prepared = t0  # prepare cost was hidden in the previous iteration
        else:
            body = _next_forward_body()
            if body is None:
                break
            if node_bucketizer is not None:
                node_bucketizer.observe(body["total_nodes"])
                edge_bucketizer.observe(body["total_edges"])
            effective_padded = (
                args.padded_inference
                and (node_bucketizer is None or node_bucketizer.is_ready)
            )
            tensors, real_n = _prepare_tensors(
                body, device, transfer_stream, padded=effective_padded,
                node_buckets=node_bucketizer, edge_buckets=edge_bucketizer,
            )
            t_prepared = _time.perf_counter()

        # Launch forward on the default stream and hand the outputs off
        # to the writer thread (which event-waits + writes to stdout).
        sync_ms, fwd_ms, write_ms = _forward_and_dispatch(
            tensors, model, transfer_stream, writer_queue, real_n,
        )

        # Drop our local refs; the writer thread holds its own refs until
        # it's done with these tensors.
        del tensors
        t_forward = _time.perf_counter()

        # While the response is being written, pre-fetch the next batch
        # and start H2D transfer on the transfer stream.
        next_body = _next_forward_body()
        if next_body is not None:
            if node_bucketizer is not None:
                node_bucketizer.observe(next_body["total_nodes"])
                edge_bucketizer.observe(next_body["total_edges"])
            next_effective_padded = (
                args.padded_inference
                and (node_bucketizer is None or node_bucketizer.is_ready)
            )
            prefetched_tensors, prefetched_real_n = _prepare_tensors(
                next_body, device, transfer_stream, padded=next_effective_padded,
                node_buckets=node_bucketizer, edge_buckets=edge_bucketizer,
            )
            prefetched_body = next_body

        t_done = _time.perf_counter()

        # Perf tracking (100-batch rolling window; reset after each print).
        #   prepare:  tensor deserialization + H2D (0 when prefetched)
        #   forward:  model forward + response write
        #   prefetch: preparing next batch (hidden overlap)
        _perf_count += 1
        _perf_window_prepare_ms += (t_prepared - t0) * 1000
        _perf_window_gpu_ms += (t_forward - t_prepared) * 1000
        _perf_window_sync_ms += sync_ms
        _perf_window_fwd_ms += fwd_ms
        _perf_window_write_ms += write_ms  # now ~dispatch time, write happens on writer thread
        _perf_window_total_ms += (t_forward - t0) * 1000  # prepare + dispatch (excludes prefetch)
        _perf_window_graphs += body["num_graphs"]
        _perf_window_nodes += body["total_nodes"]
        if _perf_count % PERF_WINDOW == 0:
            n = PERF_WINDOW
            # Drain + reset writer thread's counters under the lock so the
            # window stat there matches the main-loop window.
            with writer_perf_lock:
                w_count = writer_perf["count"]
                w_total = writer_perf["total_ms"]
                w_wait = writer_perf["wait_ms"]
                writer_perf["count"] = 0
                writer_perf["total_ms"] = 0.0
                writer_perf["wait_ms"] = 0.0
            w_total_avg = w_total / w_count if w_count else 0.0
            w_wait_avg = w_wait / w_count if w_count else 0.0
            w_io_avg = w_total_avg - w_wait_avg
            bucket_suffix = ""
            if node_bucketizer is not None:
                bucket_suffix = (
                    f" nodes={node_bucketizer.buckets}"
                    f" edges={edge_bucketizer.buckets}"
                )
            print(
                f"[perf] batches={_perf_count} "
                f"last_total={_perf_window_total_ms/n:.1f}ms "
                f"last_prepare={_perf_window_prepare_ms/n:.1f}ms "
                f"last_dispatch={_perf_window_gpu_ms/n:.1f}ms "
                f"(sync={_perf_window_sync_ms/n:.2f} launch={_perf_window_fwd_ms/n:.2f} put={_perf_window_write_ms/n:.2f}) "
                f"writer={w_total_avg:.2f}ms (event_wait={w_wait_avg:.2f} io={w_io_avg:.2f}) "
                f"last_graphs={_perf_window_graphs/n:.1f} "
                f"last_nodes={_perf_window_nodes/n:.0f}"
                f"{bucket_suffix}",
                file=sys.stderr, flush=True,
            )
            # Reset the rolling window so the next line is independent.
            _perf_window_total_ms = 0.0
            _perf_window_prepare_ms = 0.0
            _perf_window_gpu_ms = 0.0
            _perf_window_sync_ms = 0.0
            _perf_window_fwd_ms = 0.0
            _perf_window_write_ms = 0.0
            _perf_window_graphs = 0
            _perf_window_nodes = 0

        # Return the allocator's unused reserve to the system ONLY under real
        # memory pressure (see above) — never on a fixed cadence, which churns
        # GTT and slows every process on this unified-memory APU.
        if _clear_on_cuda:
            _clear_check += 1
            if _clear_check >= _clear_check_every:
                _clear_check = 0
                if torch.cuda.memory_reserved() / 1e6 > _clear_threshold_mb:
                    torch.cuda.empty_cache()

        if next_body is None:
            break

    # Drain and stop the writer cleanly so any in-flight responses get out.
    writer_queue.put(None)
    writer.join(timeout=5.0)


if __name__ == "__main__":
    main()
