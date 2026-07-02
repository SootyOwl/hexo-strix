"""Tests for the new graph-based JSONL format.

The Rust self-play binary writes pre-built graph data per position instead of
moves + policy. The Python side deserializes directly into TrainingExamples
without replaying the game.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_graph, game_to_axis_graph
from hexo_a0.self_play import TrainingExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_jsonl_line(
    game_config=None,
    graph_type="hex",
    n_moves=3,
) -> dict:
    """Build a new-format JSONL dict by playing a short game and serializing
    the graph data at each position.

    This simulates what the Rust binary will produce.
    """
    if game_config is None:
        game_config = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    game = hexo_rs.GameState(game_config)
    graph_fn = game_to_axis_graph if graph_type == "axis" else game_to_graph

    examples = []
    moves_played = 0
    winner = "draw"

    while not game.is_terminal() and moves_played < n_moves:
        data = graph_fn(game)
        legal_moves = game.legal_moves()
        n_legal = len(legal_moves)

        # Uniform policy
        policy = [1.0 / n_legal] * n_legal

        player = game.current_player()  # "P1" or "P2"

        # Build the example dict (what Rust will write)
        ex = {
            "features": data.x.flatten().tolist(),
            "edge_src": data.edge_index[0].tolist() if data.edge_index.numel() > 0 else [],
            "edge_dst": data.edge_index[1].tolist() if data.edge_index.numel() > 0 else [],
            "legal_mask": data.legal_mask.tolist(),
            "stone_mask": data.stone_mask.tolist(),
            "coords": data.coords.flatten().tolist(),
            "num_nodes": data.x.shape[0],
            "policy": policy,
            "value": 0.0,  # draw
        }
        if graph_type == "axis" and hasattr(data, "edge_attr") and data.edge_attr is not None:
            ex["edge_attr"] = data.edge_attr.flatten().tolist()

        examples.append(ex)

        # Play first legal move
        move = legal_moves[0]
        game.apply_move(move[0], move[1])
        moves_played += 1

    return {"length": moves_played, "examples": examples}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphFormatParsing:
    """Test that _trajectories_to_examples handles the new graph-based format."""

    def _make_trainer(self, graph_type="hex"):
        """Create a minimal Trainer for testing _trajectories_to_examples."""
        from hexo_a0.model import HeXONet
        from hexo_a0.trainer import Trainer
        from hexo_a0.config import FullConfig, TrainingConfig

        conv = "gatv2" if graph_type == "hex" else "gine"
        mc = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type=graph_type, conv_type=conv, prune_empty_edges=False)
        cfg = FullConfig(model=mc, training=TrainingConfig(
            batch_size=8, buffer_capacity=100, edge_budget=0, grad_accumulation=False,
            lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0,
            target_reuse=0.0, draw_value=0.0, augment_symmetries=False,
        ))
        gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        model = HeXONet(mc)
        trainer = Trainer(model, cfg, gc, torch.device("cpu"))
        return trainer

    def test_new_format_produces_training_examples(self):
        """New-format JSONL should produce TrainingExample objects."""
        trainer = self._make_trainer()
        line = _make_graph_jsonl_line(n_moves=3)
        examples = trainer._trajectories_to_examples([line])
        assert len(examples) == 3
        for ex in examples:
            assert isinstance(ex, TrainingExample)

    def test_new_format_data_has_correct_attributes(self):
        """Rebuilt graph should have x, edge_index, legal_mask, stone_mask, coords."""
        from hexo_a0.graph import graph_fn_from_model_config
        trainer = self._make_trainer()
        line = _make_graph_jsonl_line(n_moves=1)
        examples = trainer._trajectories_to_examples([line])
        assert examples[0].data is None
        gfn = graph_fn_from_model_config(trainer.model_config)
        data = gfn(examples[0].game_state)
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "legal_mask")
        assert hasattr(data, "stone_mask")
        assert hasattr(data, "coords")
        assert data.x.shape[1] == 8
        assert data.edge_index.shape[0] == 2
        assert data.coords.shape[1] == 2

    def test_new_format_policy_is_tensor(self):
        """Policy target should be a float32 tensor."""
        trainer = self._make_trainer()
        line = _make_graph_jsonl_line(n_moves=1)
        examples = trainer._trajectories_to_examples([line])
        assert examples[0].policy_target.dtype == torch.float32
        assert examples[0].policy_target.ndim == 1

    def test_new_format_value_target(self):
        """Value target should be preserved from JSONL."""
        trainer = self._make_trainer()
        line = _make_graph_jsonl_line(n_moves=2)
        # Manually set value targets to check they're preserved
        line["examples"][0]["value"] = 1.0
        line["examples"][1]["value"] = -1.0
        examples = trainer._trajectories_to_examples([line])
        assert examples[0].value_target == 1.0
        assert examples[1].value_target == -1.0

    def test_axis_graph_format(self):
        """Axis graph format should include edge_attr in the rebuilt Data."""
        from hexo_a0.graph import graph_fn_from_model_config
        trainer = self._make_trainer(graph_type="axis")
        gfn = graph_fn_from_model_config(trainer.model_config)
        line = _make_graph_jsonl_line(graph_type="axis", n_moves=2)
        examples = trainer._trajectories_to_examples([line])
        for ex in examples:
            data = gfn(ex.game_state)
            assert hasattr(data, "edge_attr")
            assert data.edge_attr is not None
            assert data.edge_attr.shape[1] == 5

    def test_game_length_reported(self):
        """The 'length' field should be present for stats tracking."""
        line = _make_graph_jsonl_line(n_moves=5)
        assert line["length"] == 5

    def test_empty_examples_list(self):
        """A game with no examples should produce nothing."""
        trainer = self._make_trainer()
        line = {"length": 0, "examples": []}
        examples = trainer._trajectories_to_examples([line])
        assert len(examples) == 0

def _build_trainer(graph_type="axis"):
    """Standalone trainer builder mirroring TestGraphFormatParsing._make_trainer."""
    from hexo_a0.model import HeXONet
    from hexo_a0.trainer import Trainer
    from hexo_a0.config import FullConfig, TrainingConfig

    conv = "gatv2" if graph_type == "hex" else "gine"
    mc = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type=graph_type, conv_type=conv, prune_empty_edges=False)
    cfg = FullConfig(model=mc, training=TrainingConfig(
        batch_size=8, buffer_capacity=100, edge_budget=0, grad_accumulation=False,
        lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0,
        target_reuse=0.0, draw_value=0.0, augment_symmetries=False,
    ))
    gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
    model = HeXONet(mc)
    return Trainer(model, cfg, gc, torch.device("cpu"))


def test_parse_graph_examples_stores_game_state():
    """_parse_graph_examples must derive and store game_state (no graph data)."""
    trainer = _build_trainer(graph_type="axis")
    game_data = _make_graph_jsonl_line(graph_type="axis", n_moves=3)
    examples = trainer._parse_graph_examples(game_data)
    assert examples
    from hexo_a0.graph import graph_fn_from_model_config
    gfn = graph_fn_from_model_config(trainer.model_config)
    for ex in examples:
        assert ex.game_state is not None
        assert ex.data is None
        rebuilt = gfn(ex.game_state)
        assert ex.policy_target.shape[0] == int(rebuilt.legal_mask.sum())


class TestBinaryReader:
    """Test the length-prefixed binary reader."""

    def _make_binary_record(
        self, n_moves=2, graph_type="hex", magic=b"HX03", node_dim=8,
        window_stats=None, game_dict=None,
    ):
        """Build a binary record from a graph JSONL line (simulating Rust output).

        ``magic`` selects the envelope version: HX03 (no sample_weight),
        HX04 (adds per-example sample_weight), HX05 (adds a node_dim byte to
        the envelope), HX06 (adds 12B window telemetry after node_dim).
        ``node_dim`` > 8 zero-pads each node's 8-dim features out to the
        requested width (HX05/HX06 only).
        ``window_stats`` is a 4-tuple (in_window_moves, gated_moves,
        mass_removed_sum, acted_deficit_sum); required when magic=b"HX06".
        ``game_dict`` is an optional pre-built source dict (from
        ``_make_graph_jsonl_line``); when supplied, ``n_moves`` and
        ``graph_type`` are ignored for data generation so that the caller
        can use the same dict for both packing and comparison.
        """
        import struct
        game = game_dict if game_dict is not None else _make_graph_jsonl_line(n_moves=n_moves, graph_type=graph_type)
        examples = game["examples"]
        # Build record body (everything after magic+size header):
        # num_examples (u32 LE) + winner (i8) + move_count (u32 LE)
        # + node_dim (u8, HX05/HX06 only)
        # + window telemetry (12B, HX06 only) + examples
        winner_str = game.get("winner", "draw")
        winner_byte = 1 if winner_str == "P1" else (-1 if winner_str == "P2" else 0)
        body = (
            struct.pack("<I", len(examples))
            + struct.pack("<b", winner_byte)
            + struct.pack("<I", n_moves)
        )
        if magic in (b"HX05", b"HX06"):
            body += struct.pack("B", node_dim)
        if magic == b"HX06":
            assert window_stats is not None, "window_stats required for HX06"
            body += struct.pack("<HHff", *window_stats)
        for ex in examples:
            n = ex["num_nodes"]
            e = len(ex["edge_src"])
            nl = len(ex["policy"])
            has_ea = "edge_attr" in ex
            body += struct.pack("<HHH", n, e, nl)
            body += struct.pack("B", int(has_ea))
            body += struct.pack("<f", ex["value"])
            if magic in (b"HX04", b"HX05", b"HX06"):
                body += struct.pack("<f", ex.get("sample_weight", 1.0))
            features = ex["features"]
            if node_dim != 8:
                # Zero-pad each node's 8 features to node_dim (simulates the
                # threat-feature columns without rebuilding graphs).
                features = []
                for i in range(n):
                    features.extend(ex["features"][i * 8 : (i + 1) * 8])
                    features.extend([0.0] * (node_dim - 8))
            for f in features:
                body += struct.pack("<f", f)
            for s in ex["edge_src"]:
                body += struct.pack("<H", s)
            for d in ex["edge_dst"]:
                body += struct.pack("<H", d)
            if has_ea:
                for a in ex["edge_attr"]:
                    body += struct.pack("<f", a)
            for b in ex["legal_mask"]:
                body += struct.pack("B", int(b))
            for b in ex["stone_mask"]:
                body += struct.pack("B", int(b))
            for c in ex["coords"]:
                body += struct.pack("<h", c)
            for p in ex["policy"]:
                body += struct.pack("<f", p)
        # magic + size + body
        return magic + struct.pack("<I", len(body)) + body

    def test_read_binary_records(self):
        """Should read length-prefixed binary records."""
        import io
        from hexo_a0.trainer import Trainer

        record = self._make_binary_record(n_moves=2)
        # Write two records
        fh = io.BytesIO(record + record)

        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)
        assert len(batch) == 2
        assert batch[0]["length"] == 2
        assert len(batch[0]["examples"]) == 2

    def test_read_hx05_record_node_dim_12(self):
        """HX05 with node_dim=12: features parse at 12 floats per node."""
        import io
        from hexo_a0.trainer import Trainer

        record = self._make_binary_record(n_moves=2, magic=b"HX05", node_dim=12)
        fh = io.BytesIO(record)
        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)
        assert len(batch) == 1
        for ex in batch[0]["examples"]:
            assert len(ex["features"]) == ex["num_nodes"] * 12
            # First 8 dims of node 0 survive; the padded threat dims are zero.
            assert ex["features"][8:12] == [0.0, 0.0, 0.0, 0.0]

    def test_read_hx05_record_node_dim_8(self):
        """HX05 with node_dim=8 parses identically to HX04 content-wise."""
        import io
        from hexo_a0.trainer import Trainer

        record = self._make_binary_record(n_moves=1, magic=b"HX05", node_dim=8)
        fh = io.BytesIO(record)
        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)
        assert len(batch) == 1
        ex = batch[0]["examples"][0]
        assert len(ex["features"]) == ex["num_nodes"] * 8

    def test_read_hx04_record_still_8dim(self):
        """HX04 records (pre-node_dim) parse with the implied 8-dim width."""
        import io
        from hexo_a0.trainer import Trainer

        record = self._make_binary_record(n_moves=1, magic=b"HX04")
        fh = io.BytesIO(record)
        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)
        assert len(batch) == 1
        ex = batch[0]["examples"][0]
        assert len(ex["features"]) == ex["num_nodes"] * 8
        assert ex["sample_weight"] == 1.0

    def test_incomplete_record_seeks_back(self):
        """Should handle partial writes gracefully."""
        import io
        from hexo_a0.trainer import Trainer

        record = self._make_binary_record(n_moves=1)
        # Truncate the record
        fh = io.BytesIO(record[:20])

        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)
        assert len(batch) == 0
        assert fh.tell() == 0

    def test_read_hx06_record_with_window_stats(self):
        """Should parse HX06 window telemetry fields and the examples after them."""
        import io
        from hexo_a0.trainer import Trainer
        record = self._make_binary_record(
            n_moves=2, magic=b"HX06",
            window_stats=(7, 3, 0.8125, 0.4375),
        )
        fh = io.BytesIO(record)
        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)
        assert len(batch) == 1
        ws = batch[0]["window_stats"]
        assert ws["in_window_moves"] == 7
        assert ws["gated_moves"] == 3
        assert abs(ws["mass_removed_sum"] - 0.8125) < 1e-6
        assert abs(ws["acted_deficit_sum"] - 0.4375) < 1e-6
        assert batch[0]["length"] == 2

    def test_hx05_record_has_no_window_stats(self):
        """Should report window_stats=None for HX05 (gate-off) records.

        The parser contract is key-present-with-None (not key-absent), so
        we assert both that the key exists and that its value is None.
        """
        import io
        from hexo_a0.trainer import Trainer
        # Explicit HX05 (the helper's default magic is HX03).
        record = self._make_binary_record(n_moves=1, magic=b"HX05")
        fh = io.BytesIO(record)
        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)
        assert "window_stats" in batch[0] and batch[0]["window_stats"] is None

    def test_hx06_and_hx05_interleaved(self):
        """Should read mixed HX06/HX05 streams in order without desync."""
        import io
        from hexo_a0.trainer import Trainer
        r6 = self._make_binary_record(n_moves=1, magic=b"HX06",
                                      window_stats=(2, 1, 0.5, 0.25))
        r5 = self._make_binary_record(n_moves=1, magic=b"HX05")
        fh = io.BytesIO(r6 + r5 + r6)
        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)
        assert len(batch) == 3
        assert batch[0]["window_stats"] is not None
        assert batch[1].get("window_stats") is None
        assert batch[2]["window_stats"] is not None

    def test_binary_roundtrip_matches_json(self):
        """Binary-parsed examples should match JSON-parsed examples."""
        import io
        from hexo_a0.trainer import Trainer
        from hexo_a0.model import HeXONet
        from hexo_a0.config import ModelConfig
        from hexo_a0.config import FullConfig

        gc = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        game_dict = _make_graph_jsonl_line(gc, n_moves=3)

        # Parse from JSON dict
        from hexo_a0.config import TrainingConfig
        mc = ModelConfig(hidden_dim=16, num_layers=1, num_heads=1, graph_type="hex", conv_type="gatv2")
        cfg = FullConfig(model=mc, training=TrainingConfig(
            batch_size=8, buffer_capacity=100, edge_budget=0, grad_accumulation=False,
            lr_schedule="constant", lr_warmup_steps=0, total_train_steps=0,
            target_reuse=0.0, draw_value=0.0, augment_symmetries=False,
        ))
        model = HeXONet(mc)
        trainer = Trainer(model, cfg, gc, torch.device("cpu"))
        json_examples = trainer._trajectories_to_examples([game_dict])

        # Parse from binary
        record = self._make_binary_record(n_moves=3)
        fh = io.BytesIO(record)
        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)
        bin_examples = trainer._trajectories_to_examples(batch)

        from hexo_a0.graph import graph_fn_from_model_config
        gfn = graph_fn_from_model_config(trainer.model_config)
        assert len(json_examples) == len(bin_examples)
        for je, be in zip(json_examples, bin_examples):
            jd, bd = gfn(je.game_state), gfn(be.game_state)
            torch.testing.assert_close(jd.x, bd.x, atol=1e-6, rtol=1e-5)
            assert torch.equal(jd.edge_index, bd.edge_index)
            torch.testing.assert_close(je.policy_target, be.policy_target)
            assert je.value_target == be.value_target

    def test_hx06_node_dim12_stride_safety(self):
        """HX06 + node_dim=12 (threat features): window_stats parse correctly
        AND example content is intact after the 12-byte envelope block.

        This is the historical stride-bug failure mode: a silent misalignment
        after an envelope-size change would corrupt feature tensors for every
        example in the record.  We verify:
          - window_stats round-trips exactly
          - features length == num_nodes * 12 per example
          - policy values match what the helper wrote
          - value and sample_weight fields are exact
        """
        import io
        import struct
        from hexo_a0.trainer import Trainer

        ws_in = (11, 5, 0.625, 0.1875)  # (in_window, gated, mass_sum, deficit_sum)
        # Build the source dict ONCE; pass it into the record builder so the
        # comparison below uses exactly the same data that was packed — no
        # separate _make_graph_jsonl_line call that relies on determinism.
        source_game = _make_graph_jsonl_line(n_moves=3)
        # Build an HX06 record with node_dim=12 and 3 examples so we can
        # verify that alignment holds across multiple example boundaries.
        record = self._make_binary_record(
            magic=b"HX06", node_dim=12,
            window_stats=ws_in,
            game_dict=source_game,
        )
        fh = io.BytesIO(record)
        batch: list[dict] = []
        Trainer._read_binary_records(fh, batch, 10)

        assert len(batch) == 1, "Expected exactly one record"
        game = batch[0]

        # --- window_stats ---
        ws = game["window_stats"]
        assert ws is not None, "HX06 must carry window_stats"
        assert ws["in_window_moves"] == ws_in[0]
        assert ws["gated_moves"] == ws_in[1]
        assert abs(ws["mass_removed_sum"] - ws_in[2]) < 1e-6
        assert abs(ws["acted_deficit_sum"] - ws_in[3]) < 1e-6

        # --- example content integrity (guards stride misalignment) ---
        assert len(game["examples"]) == 3, "Expected 3 examples"

        src_examples = source_game["examples"]

        for i, (ex, src) in enumerate(zip(game["examples"], src_examples)):
            n = ex["num_nodes"]
            # Feature stride must be node_dim=12 wide
            assert len(ex["features"]) == n * 12, (
                f"example {i}: features length {len(ex['features'])} != {n} * 12"
            )
            # Padded (threat) dims are zero; real dims match the source
            for node_idx in range(n):
                real = ex["features"][node_idx * 12 : node_idx * 12 + 8]
                pad  = ex["features"][node_idx * 12 + 8 : node_idx * 12 + 12]
                assert pad == [0.0, 0.0, 0.0, 0.0], (
                    f"example {i} node {node_idx}: padding not zero: {pad}"
                )
                src_real = src["features"][node_idx * 8 : (node_idx + 1) * 8]
                assert real == src_real, (
                    f"example {i} node {node_idx}: features mismatch after stride change"
                )
            # Policy must round-trip (written as f32; relative tolerance 1e-6)
            assert len(ex["policy"]) == len(src["policy"]), (
                f"example {i}: policy length mismatch"
            )
            for j, (p_got, p_src) in enumerate(zip(ex["policy"], src["policy"])):
                assert abs(p_got - p_src) < 1e-5, (
                    f"example {i} policy[{j}]: {p_got} vs {p_src}"
                )
            # Value and sample_weight exact (f32 round-trip)
            assert abs(ex["value"] - src["value"]) < 1e-6, (
                f"example {i}: value mismatch"
            )
            assert abs(ex.get("sample_weight", 1.0) - src.get("sample_weight", 1.0)) < 1e-6, (
                f"example {i}: sample_weight mismatch"
            )
