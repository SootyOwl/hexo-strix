//! Frame parse/serialize roundtrip + a full forward through a tiny model,
//! plus a cross-implementation pin that spawns the real `hexo-infer-server`
//! binary and asserts bitwise equality with the in-process forward pass.

use hexo_infer::protocol::{
    parse_forward_body, read_header, MsgType, MAGIC, VERSION,
};
use hexo_infer::server::split_batch;
use hexo_infer::InferModel;
use hexo_rs::axis_graph::{game_to_axis_graph_raw_opts, AxisGraphData};
use hexo_rs::hexo_engine::{GameConfig, GameState};
use std::io::{Read, Write};
use std::process::{Command, Stdio};

/// Test helper mirroring `inference_subprocess.rs` lines 194-271: the FORWARD
/// body (everything after the 6-byte message header) is
/// `<III>` (total_nodes, total_edges, num_graphs) + `has_edge_attr` u8 +
/// `node_dim` u8, then features (f32) + edge_src (i64) + edge_dst (i64) +
/// [edge_attr (E×5 f32)] + legal_mask (u8) + stone_mask (u8) + batch (i32).
#[allow(clippy::too_many_arguments)]
fn build_forward_body(
    node_dim: u8,
    features: &[f32],
    edge_src: &[i64],
    edge_dst: &[i64],
    edge_attr: Option<&[f32]>,
    legal_mask: &[u8],
    stone_mask: &[u8],
    batch: &[i32],
) -> Vec<u8> {
    let total_nodes = (features.len() / node_dim as usize) as u32;
    let total_edges = edge_src.len() as u32;
    let num_graphs = batch.iter().copied().max().map_or(0, |m| m + 1) as u32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&total_nodes.to_le_bytes());
    buf.extend_from_slice(&total_edges.to_le_bytes());
    buf.extend_from_slice(&num_graphs.to_le_bytes());
    buf.push(edge_attr.is_some() as u8);
    buf.push(node_dim);
    for f in features {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    for s in edge_src {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    for d in edge_dst {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    if let Some(ea) = edge_attr {
        for f in ea {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
    buf.extend_from_slice(legal_mask);
    buf.extend_from_slice(stone_mask);
    for b in batch {
        buf.extend_from_slice(&b.to_le_bytes());
    }
    buf
}

#[test]
fn header_roundtrip() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.push(VERSION);
    buf.push(0x01);
    let (ver, ty) = read_header(&mut &buf[..]).unwrap();
    assert_eq!(ver, VERSION);
    assert_eq!(ty, MsgType::Forward);
}

#[test]
fn forward_body_splits_into_per_graph_axis_data() {
    let node_dim = 3usize;
    let features: Vec<f32> = (0..(5 * node_dim)).map(|i| i as f32).collect(); // 2 + 3 nodes
    let edge_src: Vec<i64> = vec![0, 2, 4]; // one edge in g0, two in g1
    let edge_dst: Vec<i64> = vec![1, 3, 3];
    let edge_attr: Vec<f32> = vec![0.0; 3 * 5];
    let legal_mask = vec![1u8, 0, 1, 1, 0];
    let stone_mask = vec![0u8, 1, 0, 0, 1];
    let batch: Vec<i32> = vec![0, 0, 1, 1, 1];

    let body = build_forward_body(
        node_dim as u8,
        &features,
        &edge_src,
        &edge_dst,
        Some(&edge_attr),
        &legal_mask,
        &stone_mask,
        &batch,
    );
    let parsed = parse_forward_body(&body).unwrap();
    let graphs = split_batch(&parsed).unwrap();
    assert_eq!(graphs.len(), 2);
    assert_eq!(graphs[0].num_nodes, 2);
    assert_eq!(graphs[1].num_nodes, 3);
    assert_eq!(graphs[1].edge_src, vec![0, 2]); // offsets removed
    assert_eq!(graphs[1].edge_dst, vec![1, 1]);
    assert_eq!(graphs[0].legal_mask, vec![true, false]);
}

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Collate per-graph [`AxisGraphData`] into a FORWARD body exactly as the
/// client (`inference_subprocess.rs`) does: node offsets applied to edge
/// indices, arrays concatenated in wire order.
fn collate(graphs: &[AxisGraphData], node_dim: u8) -> Vec<u8> {
    let mut features = Vec::new();
    let mut edge_src = Vec::new();
    let mut edge_dst = Vec::new();
    let mut edge_attr = Vec::new();
    let mut legal = Vec::new();
    let mut stone = Vec::new();
    let mut batch = Vec::new();
    let mut off: i64 = 0;
    for (gi, g) in graphs.iter().enumerate() {
        features.extend_from_slice(&g.features);
        for &s in &g.edge_src {
            edge_src.push(s + off);
        }
        for &d in &g.edge_dst {
            edge_dst.push(d + off);
        }
        edge_attr.extend_from_slice(&g.edge_attr);
        legal.extend(g.legal_mask.iter().map(|&b| b as u8));
        stone.extend(g.stone_mask.iter().map(|&b| b as u8));
        batch.extend(std::iter::repeat(gi as i32).take(g.num_nodes));
        off += g.num_nodes as i64;
    }
    build_forward_body(
        node_dim, &features, &edge_src, &edge_dst, Some(&edge_attr), &legal, &stone, &batch,
    )
}

fn read_exact_vec(r: &mut impl Read, n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).expect("short read from server");
    buf
}

fn u32_le(r: &mut impl Read) -> u32 {
    let b = read_exact_vec(r, 4);
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The load-bearing test: spawn the REAL `hexo-infer-server` binary, send one
/// real FORWARD frame, and assert the response is well-formed and bitwise
/// identical to the in-process `InferModel::forward` on the same split graphs.
#[test]
fn cross_impl_forward_matches_in_process_bitwise() {
    let st_path = fixtures_dir().join("tiny.safetensors");
    let model = InferModel::from_safetensors(&std::fs::read(&st_path).unwrap()).unwrap();
    let cfg = model.config();
    let node_dim = cfg.node_dim as u8;

    // Two distinct game states -> two axis graphs (tiny config: relative + threat).
    let build_graph = |moves: &[(i32, i32)]| -> AxisGraphData {
        let mut game = GameState::with_config(GameConfig {
            win_length: 6,
            placement_radius: 4,
            max_moves: 300,
        });
        for &m in moves {
            game.apply_move(m).unwrap();
        }
        game_to_axis_graph_raw_opts(
            &game,
            cfg.prune_empty_edges,
            cfg.threat_features,
            cfg.relative_stones,
        )
    };
    let graphs = vec![
        build_graph(&[(1, 2), (0, -2), (2, 0), (-1, 1)]),
        build_graph(&[(1, 2), (0, -2), (2, 0), (-1, 1), (3, -1), (0, 3)]),
    ];
    let num_graphs = graphs.len();
    let body = collate(&graphs, node_dim);

    // Spawn the real server binary (same crate => CARGO_BIN_EXE is defined).
    let mut child = Command::new(env!("CARGO_BIN_EXE_hexo-infer-server"))
        .arg("--checkpoint")
        .arg(&st_path)
        // Extra --model-* flags the spawner would forward, all ignored:
        .args(["--hidden-dim", "16", "--num-layers", "2", "--graph-type", "axis"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn hexo-infer-server");

    // Drain stderr; signal on READY.
    let stderr = child.stderr.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(stderr);
        let mut line = String::new();
        let mut sent = false;
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if !sent && line.trim() == "READY" {
                        let _ = tx.send(());
                        sent = true;
                    }
                }
                Err(_) => break,
            }
        }
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .expect("server never sent READY");

    // Write one FORWARD frame (6-byte header + body).
    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(&MAGIC.to_le_bytes()).unwrap();
        stdin.write_all(&[VERSION, 0x01]).unwrap();
        stdin.write_all(&body).unwrap();
        stdin.flush().unwrap();
    }

    // Read + validate the response frame.
    let stdout = child.stdout.as_mut().unwrap();
    let hdr = read_exact_vec(stdout, 6);
    assert_eq!(u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]), MAGIC);
    assert_eq!(hdr[4], VERSION);
    assert_eq!(hdr[5], 0x01, "response must be a FORWARD frame");

    let total_legal = u32_le(stdout) as usize;
    let resp_graphs = u32_le(stdout) as usize;
    assert_eq!(resp_graphs, num_graphs);

    let logits_bytes = read_exact_vec(stdout, total_legal * 4);
    let logits: Vec<f32> = logits_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let counts_bytes = read_exact_vec(stdout, resp_graphs * 4);
    let legal_counts: Vec<i32> = counts_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let values_bytes = read_exact_vec(stdout, resp_graphs * 4);
    let values: Vec<f32> = values_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Well-formedness: values per graph, and logits count == sum of legal counts.
    assert_eq!(values.len(), num_graphs);
    assert_eq!(
        legal_counts.iter().map(|&c| c as usize).sum::<usize>(),
        total_legal
    );

    // Bitwise equivalence vs in-process forward on the same graphs.
    let mut off = 0usize;
    for (gi, g) in graphs.iter().enumerate() {
        let (exp_logits, exp_value) = model.forward(g);
        assert_eq!(legal_counts[gi] as usize, exp_logits.len(), "graph {gi} legal count");
        for (j, &e) in exp_logits.iter().enumerate() {
            assert_eq!(
                logits[off + j].to_bits(),
                e.to_bits(),
                "graph {gi} logit {j} bit mismatch"
            );
        }
        assert_eq!(
            values[gi].to_bits(),
            exp_value.to_bits(),
            "graph {gi} value bit mismatch"
        );
        off += exp_logits.len();
    }

    // Clean shutdown.
    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(&MAGIC.to_le_bytes()).unwrap();
        stdin.write_all(&[VERSION, 0xFF]).unwrap();
        stdin.flush().unwrap();
    }
    let _ = child.wait();
}

fn write_reload(stdin: &mut impl Write, path: &str) {
    stdin.write_all(&MAGIC.to_le_bytes()).unwrap();
    stdin.write_all(&[VERSION, 0x02]).unwrap();
    let path_bytes = path.as_bytes();
    stdin
        .write_all(&(path_bytes.len() as u32).to_le_bytes())
        .unwrap();
    stdin.write_all(path_bytes).unwrap();
    stdin.flush().unwrap();
}

fn write_forward(stdin: &mut impl Write, body: &[u8]) {
    stdin.write_all(&MAGIC.to_le_bytes()).unwrap();
    stdin.write_all(&[VERSION, 0x01]).unwrap();
    stdin.write_all(body).unwrap();
    stdin.flush().unwrap();
}

/// Read one RELOAD ACK frame (header already validated by caller via
/// `read_exact_vec`/manual checks); returns the `success` byte.
fn read_reload_ack(stdout: &mut impl Read) -> u8 {
    let hdr = read_exact_vec(stdout, 6);
    assert_eq!(u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]), MAGIC);
    assert_eq!(hdr[4], VERSION);
    assert_eq!(hdr[5], 0x02, "response must be a RELOAD frame");
    read_exact_vec(stdout, 1)[0]
}

/// Read one FORWARD response frame and return (logits, legal_counts, values).
fn read_forward_response(stdout: &mut impl Read) -> (Vec<f32>, Vec<i32>, Vec<f32>) {
    let hdr = read_exact_vec(stdout, 6);
    assert_eq!(u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]), MAGIC);
    assert_eq!(hdr[4], VERSION);
    assert_eq!(hdr[5], 0x01, "response must be a FORWARD frame");

    let total_legal = u32_le(stdout) as usize;
    let num_graphs = u32_le(stdout) as usize;
    let logits_bytes = read_exact_vec(stdout, total_legal * 4);
    let logits: Vec<f32> = logits_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let counts_bytes = read_exact_vec(stdout, num_graphs * 4);
    let legal_counts: Vec<i32> = counts_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let values_bytes = read_exact_vec(stdout, num_graphs * 4);
    let values: Vec<f32> = values_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (logits, legal_counts, values)
}

/// RELOAD e2e: a RELOAD with a nonexistent path must ACK failure (success=0)
/// and leave the previously-loaded model serving FORWARD correctly; a
/// follow-up RELOAD with the original valid fixture path must ACK success
/// and FORWARD must keep working.
#[test]
fn reload_bad_path_then_good_path_keeps_serving() {
    let st_path = fixtures_dir().join("tiny.safetensors");
    let model = InferModel::from_safetensors(&std::fs::read(&st_path).unwrap()).unwrap();
    let cfg = model.config();
    let node_dim = cfg.node_dim as u8;

    let build_graph = |moves: &[(i32, i32)]| -> AxisGraphData {
        let mut game = GameState::with_config(GameConfig {
            win_length: 6,
            placement_radius: 4,
            max_moves: 300,
        });
        for &m in moves {
            game.apply_move(m).unwrap();
        }
        game_to_axis_graph_raw_opts(
            &game,
            cfg.prune_empty_edges,
            cfg.threat_features,
            cfg.relative_stones,
        )
    };
    let graphs = vec![build_graph(&[(1, 2), (0, -2), (2, 0), (-1, 1)])];
    let body = collate(&graphs, node_dim);

    let mut child = Command::new(env!("CARGO_BIN_EXE_hexo-infer-server"))
        .arg("--checkpoint")
        .arg(&st_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn hexo-infer-server");

    let stderr = child.stderr.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(stderr);
        let mut line = String::new();
        let mut sent = false;
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if !sent && line.trim() == "READY" {
                        let _ = tx.send(());
                        sent = true;
                    }
                }
                Err(_) => break,
            }
        }
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .expect("server never sent READY");

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();

    // 1. RELOAD with a nonexistent path -> ACK failure.
    write_reload(&mut stdin, "/nonexistent/path/does-not-exist.safetensors");
    let ack1 = read_reload_ack(&mut stdout);
    assert_eq!(ack1, 0, "reload from a nonexistent path must ACK failure");

    // 2. FORWARD still works (old model kept) — compare bitwise against the
    // in-process model, same as the cross-impl pin above.
    write_forward(&mut stdin, &body);
    let (logits, legal_counts, values) = read_forward_response(&mut stdout);
    let (exp_logits, exp_value) = model.forward(&graphs[0]);
    assert_eq!(legal_counts, vec![exp_logits.len() as i32]);
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].to_bits(), exp_value.to_bits());
    for (j, &e) in exp_logits.iter().enumerate() {
        assert_eq!(logits[j].to_bits(), e.to_bits(), "logit {j} bit mismatch after failed reload");
    }

    // 3. RELOAD with the original valid fixture path -> ACK success.
    write_reload(&mut stdin, st_path.to_str().unwrap());
    let ack2 = read_reload_ack(&mut stdout);
    assert_eq!(ack2, 1, "reload from the original valid path must ACK success");

    // 4. FORWARD still works after the successful reload.
    write_forward(&mut stdin, &body);
    let (logits2, legal_counts2, values2) = read_forward_response(&mut stdout);
    assert_eq!(legal_counts2, vec![exp_logits.len() as i32]);
    assert_eq!(values2.len(), 1);
    assert_eq!(values2[0].to_bits(), exp_value.to_bits());
    for (j, &e) in exp_logits.iter().enumerate() {
        assert_eq!(logits2[j].to_bits(), e.to_bits(), "logit {j} bit mismatch after successful reload");
    }

    // Clean shutdown.
    write_all_shutdown(&mut stdin);
    let _ = child.wait();
}

fn write_all_shutdown(stdin: &mut impl Write) {
    stdin.write_all(&MAGIC.to_le_bytes()).unwrap();
    stdin.write_all(&[VERSION, 0xFF]).unwrap();
    stdin.flush().unwrap();
}
