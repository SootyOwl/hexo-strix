"""Unit tests for the v2 inference wire protocol (no subprocess, no GPU).

Builds forward-message bodies in pure Python bytes and checks that
``_read_forward_body`` / ``_prepare_tensors`` honour the node-feature-dim
field added in protocol v2, and that version mismatches are rejected loudly.
"""

import io
import struct

import numpy as np
import pytest
import torch

from hexo_a0.inference_server import (
    MAGIC,
    VERSION,
    _check_protocol_header,
    _prepare_tensors,
    _read_forward_body,
)

# Byte offset of the node_dim field within the forward-message body header
# (after total_nodes u32 | total_edges u32 | num_graphs u32 | has_edge_attr u8).
NODE_DIM_OFFSET = 13


def _build_forward_body(
    node_dim: int,
    n_nodes: int = 4,
    n_edges: int = 6,
    num_graphs: int = 1,
    edge_attr: bool = False,
) -> tuple[bytes, np.ndarray]:
    """Build a forward-message BODY (the 6-byte outer header is consumed by
    the reader thread before ``_read_forward_body`` runs, so it is omitted).

    Returns ``(body_bytes, features)`` so callers can compare round-tripped
    feature values.
    """
    features = np.random.randn(n_nodes, node_dim).astype(np.float32)
    edge_src = np.arange(n_edges, dtype=np.int64) % n_nodes
    edge_dst = (np.arange(n_edges, dtype=np.int64) + 1) % n_nodes
    legal_mask = np.ones(n_nodes, dtype=np.uint8)
    stone_mask = np.zeros(n_nodes, dtype=np.uint8)
    batch = np.zeros(n_nodes, dtype=np.int32)

    buf = bytearray()
    buf.extend(struct.pack("<III", n_nodes, n_edges, num_graphs))
    buf.extend(struct.pack("<BB", 1 if edge_attr else 0, node_dim))
    buf.extend(features.tobytes())
    buf.extend(edge_src.tobytes())
    buf.extend(edge_dst.tobytes())
    if edge_attr:
        buf.extend(np.random.randn(n_edges, 5).astype(np.float32).tobytes())
    buf.extend(legal_mask.tobytes())
    buf.extend(stone_mask.tobytes())
    buf.extend(batch.tobytes())
    return bytes(buf), features


@pytest.mark.parametrize("node_dim", [8, 12])
@pytest.mark.parametrize("edge_attr", [False, True], ids=["plain", "edge_attr"])
def test_read_forward_body_node_dim(node_dim: int, edge_attr: bool):
    """The features section is sized by the header's node_dim field."""
    body_bytes, features = _build_forward_body(node_dim, edge_attr=edge_attr)
    stream = io.BytesIO(body_bytes)

    body = _read_forward_body(stream, expected_node_dim=node_dim)

    assert body["node_dim"] == node_dim
    assert body["total_nodes"] == 4
    assert body["total_edges"] == 6
    assert body["num_graphs"] == 1
    assert len(body["features"]) == 4 * node_dim * 4
    decoded = np.frombuffer(bytes(body["features"]), dtype=np.float32)
    np.testing.assert_array_equal(decoded.reshape(4, node_dim), features)
    # The whole body must be consumed — a sizing bug leaves trailing bytes
    # (which would corrupt the NEXT message's framing).
    assert stream.read() == b""


def test_read_forward_body_rejects_mismatched_node_dim():
    """A wire dim that disagrees with the server's --node-dim must fail loudly,
    not surface later as an obscure GNN shape error."""
    body_bytes, _ = _build_forward_body(12)
    with pytest.raises(
        ValueError, match=r"wire node_dim 12 != server --node-dim 8"
    ):
        _read_forward_body(io.BytesIO(body_bytes), expected_node_dim=8)


def test_read_forward_body_rejects_zero_node_dim():
    body_bytes, _ = _build_forward_body(8)
    # Patch the node_dim byte to 0 (corrupt/legacy-v1 framing).
    corrupted = bytearray(body_bytes)
    corrupted[NODE_DIM_OFFSET] = 0
    with pytest.raises(ValueError, match="node_dim"):
        _read_forward_body(io.BytesIO(bytes(corrupted)), expected_node_dim=8)


@pytest.mark.parametrize("node_dim", [8, 12])
def test_prepare_tensors_reshapes_to_node_dim(node_dim: int):
    """_prepare_tensors must reshape features to (N, node_dim), unpadded CPU."""
    body_bytes, features = _build_forward_body(node_dim)
    body = _read_forward_body(io.BytesIO(body_bytes), expected_node_dim=node_dim)

    tensors, real_n = _prepare_tensors(body, torch.device("cpu"), None, padded=False)
    feat_tensor = tensors[0]

    assert real_n == 1
    assert feat_tensor.shape == (4, node_dim)
    np.testing.assert_array_equal(feat_tensor.numpy(), features)


def test_check_protocol_header_accepts_current_version():
    _check_protocol_header(MAGIC, VERSION)  # must not raise


def test_check_protocol_header_rejects_wrong_version():
    with pytest.raises(ValueError, match=r"version 1\b"):
        _check_protocol_header(MAGIC, 1)


def test_check_protocol_header_rejects_bad_magic():
    with pytest.raises(ValueError, match="magic"):
        _check_protocol_header(0xDEADBEEF, VERSION)
