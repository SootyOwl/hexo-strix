//! HX04 v2 inference server: load a checkpoint, then serve FORWARD / RELOAD /
//! SHUTDOWN requests over stdin/stdout. A pure-Rust drop-in for
//! `python -m hexo_a0.inference_server`.

use std::io::{self, BufWriter, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use hexo_rs::axis_graph::AxisGraphData;

use crate::forward::InferModel;
use crate::protocol::{
    self, parse_forward_body, read_header, read_reload_path, write_forward_response,
    write_reload_ack, ForwardBody, MsgType,
};

/// Validate that the wire `node_dim` a FORWARD body was built with matches the
/// currently-loaded model's expected node-feature width.
///
/// Mirrors the Python server's `_read_forward_body(expected_node_dim=...)`
/// check (`hexo_a0/inference_server.py` ~L278-281): a client using a
/// different graph encoding (e.g. relative+threat vs legacy absolute) than
/// the loaded checkpoint must be rejected here, loudly, rather than let
/// `ops::linear`'s unchecked row-stride slicing read the feature buffer with
/// the wrong stride (silent wrong logits in release builds, or an
/// out-of-bounds panic).
pub fn check_node_dim(body: &ForwardBody, model_node_dim: usize) -> Result<(), String> {
    if body.node_dim != model_node_dim {
        return Err(format!(
            "wire node_dim {} != model node_dim {} (client's graph encoder does not match the loaded checkpoint)",
            body.node_dim, model_node_dim
        ));
    }
    Ok(())
}

/// Split a collated FORWARD batch into per-graph [`AxisGraphData`], undoing the
/// node-offset collation the client applied. Only the fields the forward pass
/// reads (features, edges, edge_attr, masks, num_nodes) are populated; the
/// relational + coords fields are left empty (never read by `forward`).
pub fn split_batch(b: &ForwardBody) -> Result<Vec<AxisGraphData>, String> {
    let total_nodes = b.total_nodes();
    let node_dim = b.node_dim;

    // Per-graph node counts + offsets. batch must be non-decreasing and cover
    // exactly graphs 0..num_graphs contiguously (collation groups nodes by
    // graph in order).
    let mut counts = vec![0usize; b.num_graphs];
    let mut prev = -1i32;
    for (i, &g) in b.batch.iter().enumerate() {
        if g < 0 || g as usize >= b.num_graphs {
            return Err(format!("batch[{i}]={g} out of range 0..{}", b.num_graphs));
        }
        if g < prev {
            return Err(format!(
                "batch not non-decreasing at node {i}: {g} < {prev} (graphs must be contiguous)"
            ));
        }
        prev = g;
        counts[g as usize] += 1;
    }
    let mut offsets = vec![0usize; b.num_graphs];
    let mut acc = 0usize;
    for g in 0..b.num_graphs {
        offsets[g] = acc;
        acc += counts[g];
    }
    debug_assert_eq!(acc, total_nodes);

    // Allocate per-graph builders.
    let mut graphs: Vec<AxisGraphData> = (0..b.num_graphs)
        .map(|g| {
            let n = counts[g];
            let off = offsets[g];
            AxisGraphData {
                features: b.features[off * node_dim..(off + n) * node_dim].to_vec(),
                edge_src: Vec::new(),
                edge_dst: Vec::new(),
                edge_attr: Vec::new(),
                edge_type: Vec::new(),
                edge_dist: Vec::new(),
                global_edge_src: Vec::new(),
                global_edge_dst: Vec::new(),
                legal_mask: b.legal_mask[off..off + n].iter().map(|&m| m != 0).collect(),
                stone_mask: b.stone_mask[off..off + n].iter().map(|&m| m != 0).collect(),
                coords: Vec::new(),
                num_nodes: n,
            }
        })
        .collect();

    // Distribute edges to their owning graph (identified by the source node's
    // batch), subtracting the graph's node offset to restore local indices.
    for k in 0..b.total_edges() {
        let src = b.edge_src[k];
        let dst = b.edge_dst[k];
        if src < 0 || src as usize >= total_nodes {
            return Err(format!("edge {k} src {src} out of range"));
        }
        if dst < 0 || dst as usize >= total_nodes {
            return Err(format!("edge {k} dst {dst} out of range"));
        }
        let g = b.batch[src as usize] as usize;
        if b.batch[dst as usize] as usize != g {
            return Err(format!(
                "edge {k} crosses graphs: src in {g}, dst in {}",
                b.batch[dst as usize]
            ));
        }
        let off = offsets[g] as i64;
        let gr = &mut graphs[g];
        gr.edge_src.push(src - off);
        gr.edge_dst.push(dst - off);
        if let Some(ref ea) = b.edge_attr {
            gr.edge_attr.extend_from_slice(&ea[k * 5..(k + 1) * 5]);
        }
    }

    Ok(graphs)
}

/// Evaluate a batch of split graphs. Mirrors `InferModel::eval_states`: graphs
/// are independent, so they are chunked across OS threads with bitwise-identical,
/// in-order results.
fn eval_graphs(model: &InferModel, graphs: &[AxisGraphData]) -> (Vec<Vec<f32>>, Vec<f32>) {
    if graphs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(graphs.len());
    if threads > 1 {
        let chunk = graphs.len().div_ceil(threads);
        let per_chunk: Vec<Vec<(Vec<f32>, f32)>> = std::thread::scope(|s| {
            let handles: Vec<_> = graphs
                .chunks(chunk)
                .map(|c| s.spawn(move || c.iter().map(|g| model.forward(g)).collect()))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        return per_chunk.into_iter().flatten().unzip();
    }
    graphs.iter().map(|g| model.forward(g)).unzip()
}

/// Derive the safetensors path the pure-Rust model loads from a `--checkpoint`
/// argument: a `.pt` path swaps its extension to `.safetensors`; a
/// `.safetensors` path is taken as-is; anything else also gets `.safetensors`
/// appended by extension-swap (best effort).
pub fn derive_safetensors(checkpoint: &Path) -> PathBuf {
    if checkpoint.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        checkpoint.to_path_buf()
    } else {
        checkpoint.with_extension("safetensors")
    }
}

/// Log the embedded checkpoint metadata (train_steps + source) to stderr —
/// the staleness observability contract for champion-swap freshness.
fn log_metadata(model: &InferModel, loaded_from: &Path) {
    let cfg = model.config();
    let train_steps = serde_json::from_str::<serde_json::Value>(&cfg.metadata_json)
        .ok()
        .and_then(|v| {
            v.get("train_steps")
                .map(|x| x.as_str().map(String::from).unwrap_or_else(|| x.to_string()))
        })
        .unwrap_or_else(|| "?".to_string());
    eprintln!(
        "hexo-infer-server: loaded {} (source={}, train_steps={})",
        loaded_from.display(),
        cfg.source_checkpoint,
        train_steps
    );
}

fn load(model_path: &Path) -> io::Result<InferModel> {
    let st_path = derive_safetensors(model_path);
    let bytes = std::fs::read(&st_path).map_err(|e| {
        io::Error::new(e.kind(), format!("reading {}: {e}", st_path.display()))
    })?;
    let model = InferModel::from_safetensors(&bytes).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("loading {}: {e}", st_path.display()),
        )
    })?;
    log_metadata(&model, &st_path);
    Ok(model)
}

/// Serve requests on stdin/stdout until SHUTDOWN or EOF.
pub fn serve(model_path: &Path) -> io::Result<()> {
    let mut model = load(model_path)?;

    // Announce readiness the moment weights are loaded — the client blocks on
    // this exact line on stderr before sending any request.
    eprintln!("READY");

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    loop {
        let (ver, ty) = match read_header(&mut reader) {
            Ok(h) => h,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // Clean EOF (parent dropped stdin) — treat as shutdown.
                return Ok(());
            }
            Err(e) => {
                eprintln!("hexo-infer-server: protocol error: {e}");
                return Ok(());
            }
        };
        if ver != protocol::VERSION {
            eprintln!(
                "hexo-infer-server: unsupported protocol version {ver} (this server speaks v{})",
                protocol::VERSION
            );
            return Ok(());
        }
        match ty {
            MsgType::Forward => {
                let body = read_forward_body(&mut reader)?;
                let parsed = parse_forward_body(&body).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("forward body: {e}"))
                })?;
                if let Err(e) = check_node_dim(&parsed, model.config().node_dim) {
                    // Matches the Python server's observable behavior for this
                    // case: log loudly and exit without writing a response, so
                    // the client sees a dead pipe instead of a bogus/garbage
                    // forward reply or an indefinite hang.
                    eprintln!("hexo-infer-server: protocol error: {e}");
                    return Ok(());
                }
                let graphs = split_batch(&parsed).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("split_batch: {e}"))
                })?;
                let (logits, values) = eval_graphs(&model, &graphs);
                write_forward_response(&mut writer, &logits, &values)?;
            }
            MsgType::Reload => {
                let path = read_reload_path(&mut reader)?;
                match load(Path::new(&path)) {
                    Ok(m) => {
                        model = m;
                        eprintln!("hexo-infer-server: reloaded from {path}");
                        write_reload_ack(&mut writer, true)?;
                    }
                    Err(e) => {
                        eprintln!("hexo-infer-server: reload failed: {e}");
                        write_reload_ack(&mut writer, false)?;
                    }
                }
            }
            MsgType::Shutdown => return Ok(()),
        }
    }
}

/// Read a full FORWARD request body from the stream: the 14-byte body header
/// (which carries the array sizes) followed by the arrays. Returns the raw
/// bytes (header + arrays) ready for [`parse_forward_body`].
fn read_forward_body(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut head = [0u8; 14];
    r.read_exact(&mut head)?;
    let total_nodes = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
    let total_edges = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
    let has_edge_attr = head[12] != 0;
    let node_dim = head[13] as usize;

    let feat_bytes = total_nodes * node_dim * 4;
    let edge_idx_bytes = total_edges * 8 * 2;
    let edge_attr_bytes = if has_edge_attr { total_edges * 5 * 4 } else { 0 };
    let mask_bytes = total_nodes * 2;
    let batch_bytes = total_nodes * 4;
    let rest = feat_bytes + edge_idx_bytes + edge_attr_bytes + mask_bytes + batch_bytes;

    let mut body = vec![0u8; 14 + rest];
    body[..14].copy_from_slice(&head);
    r.read_exact(&mut body[14..])?;
    Ok(body)
}

/// CLI entry point. Accepts `--checkpoint <path>` and IGNORES all other args
/// (the spawner forwards `--model-*` arch flags the pure-Rust server does not
/// need — arch metadata is embedded in the safetensors).
pub fn run_cli() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut checkpoint: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--checkpoint" {
            if i + 1 >= args.len() {
                eprintln!("hexo-infer-server: --checkpoint requires a value");
                return ExitCode::from(2);
            }
            checkpoint = Some(args[i + 1].clone());
            i += 2;
        } else if let Some(v) = a.strip_prefix("--checkpoint=") {
            checkpoint = Some(v.to_string());
            i += 1;
        } else {
            // Ignore every other argument (values and flags alike).
            i += 1;
        }
    }

    let checkpoint = match checkpoint {
        Some(c) => c,
        None => {
            eprintln!("hexo-infer-server: --checkpoint is required");
            return ExitCode::from(2);
        }
    };

    match serve(Path::new(&checkpoint)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hexo-infer-server: fatal: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod node_dim_guard_tests {
    use super::*;

    fn dummy_body(node_dim: usize) -> ForwardBody {
        ForwardBody {
            num_graphs: 0,
            node_dim,
            features: Vec::new(),
            edge_src: Vec::new(),
            edge_dst: Vec::new(),
            edge_attr: None,
            legal_mask: Vec::new(),
            stone_mask: Vec::new(),
            batch: Vec::new(),
        }
    }

    #[test]
    fn matching_node_dim_is_ok() {
        assert!(check_node_dim(&dummy_body(11), 11).is_ok());
    }

    #[test]
    fn mismatched_node_dim_is_rejected() {
        let err = check_node_dim(&dummy_body(8), 11).expect_err("must reject mismatch");
        assert!(err.contains('8'), "error should mention wire node_dim: {err}");
        assert!(err.contains("11"), "error should mention model node_dim: {err}");
    }
}
