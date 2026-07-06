//! HX04 v2 wire protocol — framing + (de)serialization.
//!
//! Byte-for-byte mirror of the client in
//! `hexo-rs/hexo-mcts/src/inference_subprocess.rs` and the Python server it
//! replaces (`hexo-a0/src/hexo_a0/inference_server.py`). All integers are
//! little-endian.
//!
//! Message header (6 bytes): `<IBB>` = MAGIC u32, VERSION u8, msg_type u8.
//!
//! FORWARD request body (after the header): `<III>` (total_nodes,
//! total_edges, num_graphs) + `has_edge_attr` u8 + `node_dim` u8, then
//! features (N×node_dim f32) + edge_src (E i64) + edge_dst (E i64) +
//! [edge_attr (E×5 f32) if has_edge_attr] + legal_mask (N u8) +
//! stone_mask (N u8) + batch (N i32).
//!
//! FORWARD response: header(FORWARD) + `<II>` (total_legal, num_graphs) +
//! logits (total_legal f32) + legal_counts (num_graphs i32) +
//! values (num_graphs f32).
//!
//! RELOAD request: header(RELOAD) + path_len u32 + path bytes (utf-8).
//! RELOAD ACK: header(RELOAD) + success u8.
//!
//! SHUTDOWN request: header(SHUTDOWN); no response.

use std::io::{self, Read, Write};

pub const MAGIC: u32 = 0x4858_3034; // "HX04"
pub const VERSION: u8 = 2;

pub const MSG_FORWARD: u8 = 0x01;
pub const MSG_RELOAD: u8 = 0x02;
pub const MSG_SHUTDOWN: u8 = 0xFF;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MsgType {
    Forward,
    Reload,
    Shutdown,
}

/// Read + validate the 6-byte message header. Returns `(version, msg_type)`.
///
/// Returns an `UnexpectedEof` error on a clean stream close (0 bytes before
/// the header) — callers treat that as a shutdown signal. A wrong magic or an
/// unknown message type is an `InvalidData` error (framing is lost).
pub fn read_header(r: &mut impl Read) -> io::Result<(u8, MsgType)> {
    let mut hdr = [0u8; 6];
    r.read_exact(&mut hdr)?;
    let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
    if magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})"),
        ));
    }
    let ver = hdr[4];
    let ty = match hdr[5] {
        MSG_FORWARD => MsgType::Forward,
        MSG_RELOAD => MsgType::Reload,
        MSG_SHUTDOWN => MsgType::Shutdown,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown message type 0x{other:02X}"),
            ));
        }
    };
    Ok((ver, ty))
}

/// Write the 6-byte message header for `ty`.
pub fn write_header(w: &mut impl Write, ty: MsgType) -> io::Result<()> {
    let type_byte = match ty {
        MsgType::Forward => MSG_FORWARD,
        MsgType::Reload => MSG_RELOAD,
        MsgType::Shutdown => MSG_SHUTDOWN,
    };
    w.write_all(&MAGIC.to_le_bytes())?;
    w.write_all(&[VERSION, type_byte])?;
    Ok(())
}

/// Parsed FORWARD request body (collated batch, node offsets already applied
/// to edge indices — exactly what the client wrote).
pub struct ForwardBody {
    pub num_graphs: usize,
    pub node_dim: usize,
    /// N×node_dim row-major f32.
    pub features: Vec<f32>,
    /// E i64 (global node indices, node offsets applied at collation).
    pub edge_src: Vec<i64>,
    /// E i64.
    pub edge_dst: Vec<i64>,
    /// E×5 row-major f32 (present iff `has_edge_attr`).
    pub edge_attr: Option<Vec<f32>>,
    /// N u8 (0/1).
    pub legal_mask: Vec<u8>,
    /// N u8 (0/1).
    pub stone_mask: Vec<u8>,
    /// N i32, node -> graph index.
    pub batch: Vec<i32>,
}

impl ForwardBody {
    pub fn total_nodes(&self) -> usize {
        self.batch.len()
    }
    pub fn total_edges(&self) -> usize {
        self.edge_src.len()
    }
}

fn take<'a>(body: &'a [u8], pos: &mut usize, n: usize, what: &str) -> Result<&'a [u8], String> {
    let end = pos
        .checked_add(n)
        .ok_or_else(|| format!("length overflow reading {what}"))?;
    if end > body.len() {
        return Err(format!(
            "truncated body reading {what}: need {n} bytes at offset {pos}, have {}",
            body.len().saturating_sub(*pos)
        ));
    }
    let slice = &body[*pos..end];
    *pos = end;
    Ok(slice)
}

fn read_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i64_vec(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_i32_vec(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Parse a FORWARD request body (everything after the 6-byte message header).
pub fn parse_forward_body(body: &[u8]) -> Result<ForwardBody, String> {
    let mut pos = 0usize;
    let head = take(body, &mut pos, 14, "body header")?;
    let total_nodes = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
    let total_edges = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
    let num_graphs = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    let has_edge_attr = head[12] != 0;
    let node_dim = head[13] as usize;

    let features = read_f32_vec(take(body, &mut pos, total_nodes * node_dim * 4, "features")?);
    let edge_src = read_i64_vec(take(body, &mut pos, total_edges * 8, "edge_src")?);
    let edge_dst = read_i64_vec(take(body, &mut pos, total_edges * 8, "edge_dst")?);
    let edge_attr = if has_edge_attr {
        Some(read_f32_vec(take(
            body,
            &mut pos,
            total_edges * 5 * 4,
            "edge_attr",
        )?))
    } else {
        None
    };
    let legal_mask = take(body, &mut pos, total_nodes, "legal_mask")?.to_vec();
    let stone_mask = take(body, &mut pos, total_nodes, "stone_mask")?.to_vec();
    let batch = read_i32_vec(take(body, &mut pos, total_nodes * 4, "batch")?);

    if batch.len() != total_nodes {
        return Err(format!(
            "batch length {} != total_nodes {total_nodes}",
            batch.len()
        ));
    }
    Ok(ForwardBody {
        num_graphs,
        node_dim,
        features,
        edge_src,
        edge_dst,
        edge_attr,
        legal_mask,
        stone_mask,
        batch,
    })
}

/// Write a FORWARD response frame.
///
/// `per_graph_logits[g]` is graph g's logits in `legal_moves()` order;
/// `values[g]` is graph g's value. `legal_counts` on the wire is
/// `per_graph_logits[g].len()` per graph (i32).
pub fn write_forward_response(
    w: &mut impl Write,
    per_graph_logits: &[Vec<f32>],
    values: &[f32],
) -> io::Result<()> {
    let num_graphs = per_graph_logits.len();
    let total_legal: usize = per_graph_logits.iter().map(|l| l.len()).sum();

    let mut buf =
        Vec::with_capacity(6 + 8 + total_legal * 4 + num_graphs * 4 + num_graphs * 4);
    write_header(&mut buf, MsgType::Forward)?;
    buf.extend_from_slice(&(total_legal as u32).to_le_bytes());
    buf.extend_from_slice(&(num_graphs as u32).to_le_bytes());
    for logits in per_graph_logits {
        for &v in logits {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    for logits in per_graph_logits {
        buf.extend_from_slice(&(logits.len() as i32).to_le_bytes());
    }
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&buf)?;
    w.flush()
}

/// Read a RELOAD request payload (path_len u32 + utf-8 path). The 6-byte
/// message header has already been consumed.
pub fn read_reload_path(r: &mut impl Read) -> io::Result<String> {
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut path_bytes = vec![0u8; len];
    r.read_exact(&mut path_bytes)?;
    String::from_utf8(path_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("non-utf8 reload path: {e}")))
}

/// Write a RELOAD ACK frame: header(RELOAD) + success u8 (1 = ok, 0 = fail).
pub fn write_reload_ack(w: &mut impl Write, success: bool) -> io::Result<()> {
    let mut buf = Vec::with_capacity(7);
    write_header(&mut buf, MsgType::Reload)?;
    buf.push(success as u8);
    w.write_all(&buf)?;
    w.flush()
}
