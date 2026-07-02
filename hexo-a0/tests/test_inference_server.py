"""Round-trip test for the inference server binary protocol."""

import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

MAGIC = 0x48583034
VERSION = 2
MSG_FORWARD = 0x01
MSG_RELOAD = 0x02
MSG_SHUTDOWN = 0xFF


def _write_header(buf: bytearray, msg_type: int) -> None:
    buf.extend(struct.pack("<IBB", MAGIC, VERSION, msg_type))


def _build_forward_request(
    features: np.ndarray,       # (N, node_dim) float32
    edge_src: np.ndarray,       # (E,) int64
    edge_dst: np.ndarray,       # (E,) int64
    legal_mask: np.ndarray,     # (N,) uint8
    stone_mask: np.ndarray,     # (N,) uint8
    batch: np.ndarray,          # (N,) int32
    num_graphs: int,
    edge_attr: np.ndarray | None = None,  # (E, 5) float32
) -> bytes:
    total_nodes = features.shape[0]
    total_edges = edge_src.shape[0]
    has_edge_attr = 1 if edge_attr is not None else 0
    node_dim = features.shape[1]

    buf = bytearray()
    _write_header(buf, MSG_FORWARD)
    buf.extend(struct.pack("<III", total_nodes, total_edges, num_graphs))
    buf.extend(struct.pack("<BB", has_edge_attr, node_dim))
    buf.extend(features.astype(np.float32).tobytes())
    buf.extend(edge_src.astype(np.int64).tobytes())
    buf.extend(edge_dst.astype(np.int64).tobytes())
    if edge_attr is not None:
        buf.extend(edge_attr.astype(np.float32).tobytes())
    buf.extend(legal_mask.astype(np.uint8).tobytes())
    buf.extend(stone_mask.astype(np.uint8).tobytes())
    buf.extend(batch.astype(np.int32).tobytes())
    return bytes(buf)


def _build_shutdown() -> bytes:
    buf = bytearray()
    _write_header(buf, MSG_SHUTDOWN)
    return bytes(buf)


def _parse_forward_response(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse forward response, return (logits, legal_counts, values)."""
    offset = 0
    magic, ver, msg_type = struct.unpack_from("<IBB", data, offset)
    offset += 6
    assert magic == MAGIC
    assert ver == VERSION
    assert msg_type == MSG_FORWARD

    total_legal, num_graphs = struct.unpack_from("<II", data, offset)
    offset += 8

    logits = np.frombuffer(data, dtype=np.float32, count=total_legal, offset=offset)
    offset += total_legal * 4

    legal_counts = np.frombuffer(data, dtype=np.int32, count=num_graphs, offset=offset)
    offset += num_graphs * 4

    values = np.frombuffer(data, dtype=np.float32, count=num_graphs, offset=offset)
    offset += num_graphs * 4

    return logits, legal_counts, values


def _make_chain_graph(n_nodes: int, graph_id: int, node_dim: int = 8):
    """Make a simple chain graph with n_nodes nodes.

    All nodes are stones (stone_mask=1), and all are legal moves (legal_mask=1).
    Edges form a chain: 0-1, 1-2, ..., (n-2)-(n-1), plus reverse.
    """
    features = np.random.randn(n_nodes, node_dim).astype(np.float32)
    # Mark all as stones and legal
    legal_mask = np.ones(n_nodes, dtype=np.uint8)
    stone_mask = np.ones(n_nodes, dtype=np.uint8)
    batch = np.full(n_nodes, graph_id, dtype=np.int32)

    # Chain edges (bidirectional) + self-loops
    src_list = []
    dst_list = []
    for i in range(n_nodes - 1):
        src_list.extend([i, i + 1])
        dst_list.extend([i + 1, i])
    # Self-loops
    for i in range(n_nodes):
        src_list.append(i)
        dst_list.append(i)
    edge_src = np.array(src_list, dtype=np.int64)
    edge_dst = np.array(dst_list, dtype=np.int64)

    return features, edge_src, edge_dst, legal_mask, stone_mask, batch


@pytest.mark.timeout(30)
@pytest.mark.parametrize("node_dim", [8, 12], ids=["dim8", "dim12"])
@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
def test_inference_server_round_trip(padded: bool, node_dim: int):
    """Spawn inference server, send 2-graph forward request, verify response.

    Runs in both unpadded and padded modes — in padded mode the server appends
    a ghost graph to round inputs up to a bucket, then slices it off before
    responding. The response shape/contents must be identical either way.

    node_dim=12 exercises the protocol-v2 node-feature-dim field end to end
    (threat-feature graphs); the server is told via --node-dim.
    """
    from hexo_a0.scriptable_model import ScriptableHeXONet

    # 1. Create and save a small model checkpoint
    model = ScriptableHeXONet(
        node_features=node_dim,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        policy_hidden=16,
        value_hidden=16,
        graph_type="hex",
        conv_type="gatv2",
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
        torch.save({"model": model.state_dict()}, ckpt_path)

    try:
        # 2. Spawn the inference server
        cmd = [
            sys.executable, "-m", "hexo_a0.inference_server",
            "--checkpoint", ckpt_path,
            "--hidden-dim", "32",
            "--num-layers", "2",
            "--num-heads", "4",
            "--policy-hidden", "16",
            "--value-hidden", "16",
            "--graph-type", "hex",
            "--conv-type", "gatv2",
            "--device", "cpu",
            "--no-compile",
            "--node-dim", str(node_dim),
        ]
        if padded:
            cmd.extend(["--padded-inference", "--bucket-observe-batches", "1"])
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 3. Wait for READY on stderr (skip HIP/ROCm noise lines)
        deadline = time.time() + 30
        while time.time() < deadline:
            ready_line = proc.stderr.readline()
            if b"READY" in ready_line:
                break
        else:
            raise TimeoutError("Python server didn't send READY within 30s")

        # 4. Build 2 graphs with 5 nodes each
        n_nodes = 5
        g0 = _make_chain_graph(n_nodes, graph_id=0, node_dim=node_dim)
        g1 = _make_chain_graph(n_nodes, graph_id=1, node_dim=node_dim)

        # Concatenate graphs (offset edge indices for graph 1)
        features = np.concatenate([g0[0], g1[0]], axis=0)
        edge_src = np.concatenate([g0[1], g1[1] + n_nodes])
        edge_dst = np.concatenate([g0[2], g1[2] + n_nodes])
        legal_mask = np.concatenate([g0[3], g1[3]])
        stone_mask = np.concatenate([g0[4], g1[4]])
        batch = np.concatenate([g0[5], g1[5]])

        request = _build_forward_request(
            features, edge_src, edge_dst, legal_mask, stone_mask, batch,
            num_graphs=2,
        )

        # 5. Send forward request
        proc.stdin.write(request)
        proc.stdin.flush()

        # 6. Read response header
        header = _read_exact(proc.stdout, 6)
        magic, ver, msg_type = struct.unpack("<IBB", header)
        assert magic == MAGIC
        assert ver == VERSION
        assert msg_type == MSG_FORWARD

        # Read response body
        body_header = _read_exact(proc.stdout, 8)
        total_legal, num_graphs = struct.unpack("<II", body_header)
        assert num_graphs == 2
        # All nodes are legal, so total_legal should be 10
        assert total_legal == 10

        logits_data = _read_exact(proc.stdout, total_legal * 4)
        legal_counts_data = _read_exact(proc.stdout, num_graphs * 4)
        values_data = _read_exact(proc.stdout, num_graphs * 4)

        logits = np.frombuffer(logits_data, dtype=np.float32)
        legal_counts = np.frombuffer(legal_counts_data, dtype=np.int32)
        values = np.frombuffer(values_data, dtype=np.float32)

        # Verify shapes
        assert logits.shape == (10,)
        assert legal_counts.shape == (2,)
        assert values.shape == (2,)

        # Verify legal counts
        assert legal_counts[0] == 5
        assert legal_counts[1] == 5

        # Verify finiteness
        assert np.all(np.isfinite(logits)), f"Non-finite logits: {logits}"
        assert np.all(np.isfinite(values)), f"Non-finite values: {values}"

        # Values should be in [-1, 1] (tanh output)
        assert np.all(values >= -1.0) and np.all(values <= 1.0), f"Values out of range: {values}"

        # 7. Send shutdown
        proc.stdin.write(_build_shutdown())
        proc.stdin.flush()

        proc.wait(timeout=5)
        assert proc.returncode == 0

    finally:
        os.unlink(ckpt_path)


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes from a stream."""
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError(f"Expected {n} bytes, got {len(data)}")
        data += chunk
    return data


@pytest.mark.timeout(60)
def test_inference_server_dies_when_parent_dies():
    """Regression for the orphaned-server leak.

    Spawn a 'launcher' subprocess that itself spawns the inference server,
    then SIGKILL the launcher. The inference server should detect the orphan
    state via prctl(PR_SET_PDEATHSIG) or the ppid poll and self-exit within a
    few seconds — not keep running and squat on the GPU.
    """
    import signal
    from hexo_a0.scriptable_model import ScriptableHeXONet

    # 1. Make a tiny checkpoint the server can load fast.
    model = ScriptableHeXONet(
        hidden_dim=16, num_layers=1, num_heads=2,
        policy_hidden=8, value_hidden=8,
        graph_type="hex", conv_type="gatv2",
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
        torch.save({"model": model.state_dict()}, ckpt_path)

    try:
        # 2. Launcher: a bash process whose only job is to start the server
        # and then sit indefinitely. We'll SIGKILL the launcher (not the
        # server) to simulate abnormal parent death.
        server_cmd = (
            f"exec {sys.executable} -m hexo_a0.inference_server "
            f"--checkpoint {ckpt_path} "
            f"--hidden-dim 16 --num-layers 1 --num-heads 2 "
            f"--policy-hidden 8 --value-hidden 8 "
            f"--graph-type hex --conv-type gatv2 "
            f"--device cpu --no-compile"
        )
        # Bash trick: spawn server in background, capture its PID, then sleep
        # forever. Parent dying takes the server with it (via prctl/ppid).
        launcher_script = f"{server_cmd} & echo $! ; wait"
        launcher = subprocess.Popen(
            ["bash", "-c", launcher_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # detach so SIGKILL only hits the launcher
        )

        # 3. First line of launcher stdout = server PID.
        deadline = time.time() + 30
        server_pid: int | None = None
        while time.time() < deadline and server_pid is None:
            line = launcher.stdout.readline()
            if line:
                try:
                    server_pid = int(line.strip())
                except ValueError:
                    pass
        assert server_pid is not None, "launcher never reported server PID"

        # 4. Wait for the server to finish initialising (READY on stderr).
        # We're reading the *launcher's* stderr, which is the server's stderr.
        deadline = time.time() + 30
        while time.time() < deadline:
            line = launcher.stderr.readline()
            if not line:
                break
            if b"READY" in line:
                break
        else:
            launcher.kill()
            raise TimeoutError("server didn't reach READY in 30s")

        # 5. Confirm server is alive.
        os.kill(server_pid, 0)  # raises OSError if process not found

        # 6. SIGKILL the launcher. The server is now an orphan.
        launcher.kill()
        launcher.wait(timeout=5)

        # 7. Server should self-exit within ~5s (prctl fires immediately;
        # ppid poll is 2s cadence, so 5s is a safe upper bound).
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                os.kill(server_pid, 0)
                time.sleep(0.2)
            except (OSError, ProcessLookupError):
                return  # server exited as expected
        # If we got here, the server is still alive. Clean up and fail loud.
        try:
            os.kill(server_pid, signal.SIGKILL)
        except OSError:
            pass
        raise AssertionError(
            f"server pid {server_pid} did not exit within 8s of parent death"
        )

    finally:
        Path(ckpt_path).unlink(missing_ok=True)
