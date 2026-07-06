//! `SubprocessModel` — spawns a Python inference subprocess and communicates
//! via a binary protocol over stdin/stdout.

use rustc_hash::FxHashMap as HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write as _};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, SystemTime};

/// Per-request inference payloads are several MB (features + edges for a
/// 60k-node batch ≈ 5 MB). Linux's default pipe buffer is 64 KB, forcing
/// ~80 read/write cycles per request. Bumping to 1 MB cuts this to ~5
/// cycles and removes the bulk of pipe-blocking overhead.
const TARGET_PIPE_SIZE: libc::c_int = 1 << 20; // 1 MB

fn try_resize_pipe(raw_fd: libc::c_int, label: &str) {
    // SAFETY: raw_fd is the kernel-side pipe FD owned by the child handle;
    // F_SETPIPE_SZ is the documented resize op. Kernel caps at
    // /proc/sys/fs/pipe-max-size; on failure we just log and keep the
    // default — never fatal.
    let rc = unsafe { libc::fcntl(raw_fd, libc::F_SETPIPE_SZ, TARGET_PIPE_SIZE) };
    if rc < 0 {
        let err = std::io::Error::last_os_error();
        eprintln!("warning: failed to resize {label} pipe to {TARGET_PIPE_SIZE} bytes: {err}");
    } else {
        eprintln!("inference subprocess {label} pipe sized to {rc} bytes");
    }
}

use hexo_engine::types::Coord;

use crate::graph_tensors::GraphTensors;

const MAGIC: u32 = 0x48583034;
/// Protocol version. v2 added a `node_dim: u8` field to the forward-message
/// header (after `has_edge_attr`) so non-8-dim node features (e.g. 12-dim
/// threat features) survive the wire. Both sides always come from the same
/// checkout (the binary spawns the server), so no rolling compat is needed.
const VERSION: u8 = 2;
const MSG_FORWARD: u8 = 0x01;
const MSG_RELOAD: u8 = 0x02;
const MSG_SHUTDOWN: u8 = 0xFF;

pub struct SubprocessModel {
    child: Child,
    #[allow(dead_code)]
    model_args: Vec<String>,
    #[allow(dead_code)]
    python_bin: String,
    model_mtime: Option<SystemTime>,
    stderr_handle: Option<std::thread::JoinHandle<()>>,
}

/// Build the `Command` used to spawn the inference subprocess, without
/// touching stdio/env — those are attached by the caller.
///
/// - `inference_bin = None`: spawns `python_bin -m hexo_a0.inference_server
///   --checkpoint <checkpoint> <extra_args...>` (the historical Python
///   server).
/// - `inference_bin = Some(bin)`: spawns `<bin> --checkpoint <checkpoint>
///   <extra_args...>` (no `-m`) — the native `hexo-infer-server` drop-in.
///   The extra args are forwarded unchanged; the Rust server ignores
///   whatever it doesn't need.
pub fn spawn_command(
    python_bin: &str,
    inference_bin: Option<&Path>,
    checkpoint: &Path,
    extra_args: &[String],
) -> Command {
    let mut cmd = match inference_bin {
        Some(bin) => {
            let mut cmd = Command::new(bin);
            cmd.arg("--checkpoint").arg(checkpoint);
            cmd
        }
        None => {
            let mut cmd = Command::new(python_bin);
            cmd.args(["-m", "hexo_a0.inference_server", "--checkpoint"])
                .arg(checkpoint);
            cmd
        }
    };
    cmd.args(extra_args);
    cmd
}

impl SubprocessModel {
    /// Spawn the inference subprocess (stderr tagged `[python]`).
    ///
    /// Waits up to 600 seconds for the "READY" signal on stderr (torch.compile
    /// can take minutes on first run).
    pub fn spawn(
        python_bin: &str,
        inference_bin: Option<&Path>,
        model_path: &str,
        model_args: &[String],
    ) -> Result<Self, String> {
        Self::spawn_labeled(python_bin, inference_bin, model_path, model_args, "python")
    }

    /// Like [`spawn`](Self::spawn) but tags every stderr line `[{label}]`
    /// instead of `[python]`, so logs from multiple concurrent inference
    /// workers (e.g. `python w0`, `python w1`) are distinguishable. Keep
    /// `python` in the label so existing log greps still match.
    pub fn spawn_labeled(
        python_bin: &str,
        inference_bin: Option<&Path>,
        model_path: &str,
        model_args: &[String],
        label: &str,
    ) -> Result<Self, String> {
        let startup_label = label.to_string();
        let drain_label = label.to_string();
        let mut child = spawn_command(python_bin, inference_bin, Path::new(model_path), model_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn inference subprocess: {e}"))?;

        // Bump both pipes to 1 MB before any traffic — per-request payloads
        // are multi-MB (features+edges for ~60k-node batches), so the 64 KB
        // default forces dozens of write/read syscalls per request.
        if let Some(stdin) = child.stdin.as_ref() {
            try_resize_pipe(stdin.as_raw_fd(), "rust→python (stdin)");
        }
        if let Some(stdout) = child.stdout.as_ref() {
            try_resize_pipe(stdout.as_raw_fd(), "python→rust (stdout)");
        }

        let stderr = child.stderr.take().expect("stderr was piped");

        // Wait for READY on stderr using thread + channel pattern.
        let (tx, rx) = mpsc::channel();
        let startup_thread = std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,  // EOF
                    Ok(_) => {
                        let trimmed = line.trim();
                        eprintln!("[{startup_label}] {trimmed}");
                        if trimmed == "READY" {
                            let _ = tx.send(Ok(reader));
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("stderr read error: {e}")));
                        return;
                    }
                }
            }
            let _ = tx.send(Err("subprocess exited before sending READY".into()));
        });

        let reader = match rx.recv_timeout(Duration::from_secs(600)) {
            Ok(Ok(reader)) => reader,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                // Timeout — kill the child and the thread
                let _ = child.kill();
                let _ = startup_thread.join();
                return Err("timed out waiting for READY from inference subprocess (600s)".into());
            }
        };

        // Spawn stderr drain thread
        let stderr_handle = std::thread::spawn(move || {
            let mut reader = reader;
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => eprintln!("[{drain_label}] {}", line.trim()),
                    Err(_) => break,
                }
            }
        });

        let model_mtime = std::fs::metadata(model_path)
            .and_then(|m| m.modified())
            .ok();

        Ok(SubprocessModel {
            child,
            model_args: model_args.to_vec(),
            python_bin: python_bin.to_string(),
            model_mtime,
            stderr_handle: Some(stderr_handle),
        })
    }

    /// Send a batch of graphs for inference and return (policy, value) results.
    ///
    /// Policy is returned as a map from `Coord` to logit for each graph.
    pub fn forward_graphs(
        &mut self,
        graphs: Vec<GraphTensors>,
    ) -> Result<(Vec<HashMap<Coord, f64>>, Vec<f64>), String> {
        if !self.is_alive() {
            return Err("inference subprocess is not running".into());
        }

        let num_graphs = graphs.len() as u32;
        let mut total_nodes: u32 = 0;
        let mut total_edges: u32 = 0;
        let has_edge_attr = graphs.first().map_or(false, |g| g.edge_attr.is_some());

        for g in &graphs {
            total_nodes += g.num_nodes as u32;
            total_edges += g.num_edges as u32;
        }

        // Node-feature dim, derived per batch from the graph tensors. Batch
        // uniformity is guaranteed upstream (all graphs in a request come from
        // the same builder config); the debug_assert catches drift in tests.
        let node_dim = graphs
            .iter()
            .find(|g| g.num_nodes > 0)
            .map_or(8, |g| g.features.len() / g.num_nodes);
        debug_assert!(
            graphs.iter().all(|g| g.features.len() == g.num_nodes * node_dim),
            "node feature dim must be uniform across the batch"
        );

        // --- Build request into a single contiguous buffer ---
        //
        // Layout: header (20 bytes) + features (N×node_dim×4) + edge_src (E×8) +
        //         edge_dst (E×8) + [edge_attr (E×5×4)] + legal_mask (N) +
        //         stone_mask (N) + batch (N×4)
        let feat_bytes = total_nodes as usize * node_dim * 4;
        let edge_idx_bytes = total_edges as usize * 8 * 2; // src + dst
        let edge_attr_bytes = if has_edge_attr { total_edges as usize * 5 * 4 } else { 0 };
        let mask_bytes = total_nodes as usize * 2; // legal + stone
        let batch_bytes = total_nodes as usize * 4;
        let buf_size = 20 + feat_bytes + edge_idx_bytes + edge_attr_bytes + mask_bytes + batch_bytes;

        let mut buf = Vec::with_capacity(buf_size);

        // Header (20 bytes)
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.push(VERSION);
        buf.push(MSG_FORWARD);
        buf.extend_from_slice(&total_nodes.to_le_bytes());
        buf.extend_from_slice(&total_edges.to_le_bytes());
        buf.extend_from_slice(&num_graphs.to_le_bytes());
        buf.push(has_edge_attr as u8);
        buf.push(u8::try_from(node_dim).expect("node_dim exceeds u8"));

        // Features: copy each graph's feature slab as raw bytes
        for g in &graphs {
            buf.extend_from_slice(as_u8_slice(&g.features));
        }

        // Edge src with node offsets applied
        let mut node_offset: i64 = 0;
        for g in &graphs {
            for &src in &g.edge_src {
                buf.extend_from_slice(&(src + node_offset).to_le_bytes());
            }
            node_offset += g.num_nodes as i64;
        }

        // Edge dst with node offsets applied
        node_offset = 0;
        for g in &graphs {
            for &dst in &g.edge_dst {
                buf.extend_from_slice(&(dst + node_offset).to_le_bytes());
            }
            node_offset += g.num_nodes as i64;
        }

        // Edge attr (if present) — raw byte copy, no per-graph offset
        if has_edge_attr {
            for g in &graphs {
                if let Some(ref ea) = g.edge_attr {
                    buf.extend_from_slice(as_u8_slice(ea));
                }
            }
        }

        // Legal mask + stone mask — pack bools as single bytes
        for g in &graphs {
            for &m in &g.legal_mask {
                buf.push(m as u8);
            }
        }
        for g in &graphs {
            for &m in &g.stone_mask {
                buf.push(m as u8);
            }
        }

        // Batch indices
        for (batch_idx, g) in graphs.iter().enumerate() {
            let idx = batch_idx as i32;
            let idx_bytes = idx.to_le_bytes();
            for _ in 0..g.num_nodes {
                buf.extend_from_slice(&idx_bytes);
            }
        }

        debug_assert_eq!(buf.len(), buf_size);

        // Single write + flush
        let stdin = self.child.stdin.as_mut().expect("stdin was piped");
        stdin.write_all(&buf).map_err(|e| format!("write error: {e}"))?;
        stdin.flush().map_err(|e| format!("flush error: {e}"))?;

        // --- Read response from stdout ---
        let stdout = self.child.stdout.as_mut().expect("stdout was piped");

        let resp_magic = read_u32_le(stdout)?;
        if resp_magic != MAGIC {
            return Err(format!("bad magic in response: 0x{resp_magic:08X}"));
        }
        let resp_ver = read_u8(stdout)?;
        if resp_ver != VERSION {
            return Err(format!("bad version in response: {resp_ver}"));
        }
        let resp_type = read_u8(stdout)?;
        if resp_type != MSG_FORWARD {
            return Err(format!("unexpected response type: 0x{resp_type:02X}"));
        }

        let total_legal = read_u32_le(stdout)? as usize;
        let resp_num_graphs = read_u32_le(stdout)? as usize;
        if resp_num_graphs != graphs.len() {
            return Err(format!(
                "graph count mismatch: sent {}, got {resp_num_graphs}",
                graphs.len()
            ));
        }

        // Logits for all legal moves
        let mut logits = vec![0.0f32; total_legal];
        read_f32_slice(stdout, &mut logits)?;

        // Legal counts per graph
        let mut legal_counts = vec![0i32; resp_num_graphs];
        read_i32_slice(stdout, &mut legal_counts)?;

        // Values per graph
        let mut values = vec![0.0f32; resp_num_graphs];
        read_f32_slice(stdout, &mut values)?;

        // Map logits back to coordinates
        let mut policies = Vec::with_capacity(resp_num_graphs);
        let mut logit_offset = 0usize;
        for (i, g) in graphs.iter().enumerate() {
            let count = legal_counts[i] as usize;
            let mut policy = HashMap::with_capacity_and_hasher(count, Default::default());
            for j in 0..count {
                let coord = g.legal_coords[j];
                policy.insert(coord, logits[logit_offset + j] as f64);
            }
            logit_offset += count;
            policies.push(policy);
        }

        let values_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();

        Ok((policies, values_f64))
    }

    /// Try to reload the model checkpoint if the file has been modified.
    /// Returns true if a reload was performed and acknowledged.
    pub fn try_reload(&mut self, path: &str) -> bool {
        let new_mtime = match std::fs::metadata(path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return false,
        };

        if self.model_mtime == Some(new_mtime) {
            return false;
        }

        if !self.is_alive() {
            return false;
        }

        // Send reload message
        let stdin = match self.child.stdin.as_mut() {
            Some(s) => s,
            None => return false,
        };
        let mut w = BufWriter::new(stdin);
        let path_bytes = path.as_bytes();
        let path_len = path_bytes.len() as u32;

        if w.write_all(&MAGIC.to_le_bytes()).is_err()
            || w.write_all(&[VERSION, MSG_RELOAD]).is_err()
            || w.write_all(&path_len.to_le_bytes()).is_err()
            || w.write_all(path_bytes).is_err()
            || w.flush().is_err()
        {
            return false;
        }

        // Read ACK
        let stdout = match self.child.stdout.as_mut() {
            Some(s) => s,
            None => return false,
        };

        let magic = match read_u32_le(stdout) {
            Ok(m) => m,
            Err(_) => return false,
        };
        if magic != MAGIC {
            return false;
        }
        let ver = match read_u8(stdout) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if ver != VERSION {
            return false;
        }
        let msg_type = match read_u8(stdout) {
            Ok(t) => t,
            Err(_) => return false,
        };
        if msg_type != MSG_RELOAD {
            return false;
        }
        let success = match read_u8(stdout) {
            Ok(s) => s,
            Err(_) => return false,
        };

        if success != 0 {
            self.model_mtime = Some(new_mtime);
            true
        } else {
            false
        }
    }

    /// Check if the subprocess is still alive.
    fn is_alive(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(Some(_)) => false,  // exited
            Ok(None) => true,      // still running
            Err(_) => false,
        }
    }
}

impl Drop for SubprocessModel {
    fn drop(&mut self) {
        // Send shutdown message
        if let Some(stdin) = self.child.stdin.as_mut() {
            let mut w = BufWriter::new(stdin);
            let _ = w.write_all(&MAGIC.to_le_bytes());
            let _ = w.write_all(&[VERSION, MSG_SHUTDOWN]);
            let _ = w.flush();
        }
        // Drop stdin to signal EOF
        self.child.stdin.take();

        // Wait up to 5 seconds for graceful exit
        let start = std::time::Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if start.elapsed() > Duration::from_secs(5) {
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) => break,
            }
        }

        // Join the stderr drain thread
        if let Some(handle) = self.stderr_handle.take() {
            let _ = handle.join();
        }
    }
}

// --- Wire-format reading helpers ---

fn read_exact(r: &mut impl std::io::Read, buf: &mut [u8]) -> Result<(), String> {
    r.read_exact(buf).map_err(|e| format!("read error: {e}"))
}

fn read_u8(r: &mut impl std::io::Read) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    read_exact(r, &mut buf)?;
    Ok(buf[0])
}

fn read_u32_le(r: &mut impl std::io::Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    read_exact(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_slice(r: &mut impl std::io::Read, out: &mut [f32]) -> Result<(), String> {
    // Read as raw bytes, then convert
    let byte_len = out.len() * 4;
    let mut bytes = vec![0u8; byte_len];
    read_exact(r, &mut bytes)?;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

fn read_i32_slice(r: &mut impl std::io::Read, out: &mut [i32]) -> Result<(), String> {
    let byte_len = out.len() * 4;
    let mut bytes = vec![0u8; byte_len];
    read_exact(r, &mut bytes)?;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = i32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

/// Reinterpret a `&[f32]` as `&[u8]` for zero-copy writes.
/// Safe on all platforms (f32 has alignment ≥ u8).
fn as_u8_slice(slice: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4)
    }
}

#[cfg(test)]
mod spawn_command_tests {
    use super::*;

    fn args_of(cmd: &Command) -> Vec<String> {
        cmd.get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn python_path_spawns_module_with_no_inference_bin() {
        let checkpoint = Path::new("/tmp/model.pt");
        let extra = vec!["--graph-type".to_string(), "axis".to_string()];
        let cmd = spawn_command("python3", None, checkpoint, &extra);

        assert_eq!(cmd.get_program(), "python3");
        let args = args_of(&cmd);
        assert_eq!(
            args,
            vec![
                "-m",
                "hexo_a0.inference_server",
                "--checkpoint",
                "/tmp/model.pt",
                "--graph-type",
                "axis",
            ]
        );
    }

    #[test]
    fn inference_bin_spawns_native_binary_directly() {
        let checkpoint = Path::new("/tmp/model.pt");
        let extra = vec!["--graph-type".to_string(), "axis".to_string()];
        let bin = Path::new("/usr/local/bin/hexo-infer-server");
        let cmd = spawn_command("python3", Some(bin), checkpoint, &extra);

        assert_eq!(cmd.get_program(), bin.as_os_str());
        let args = args_of(&cmd);
        assert_eq!(
            args,
            vec!["--checkpoint", "/tmp/model.pt", "--graph-type", "axis"]
        );
    }

    #[test]
    fn inference_bin_omits_python_module_flag() {
        let checkpoint = Path::new("/tmp/model.pt");
        let bin = Path::new("/usr/local/bin/hexo-infer-server");
        let cmd = spawn_command("python3", Some(bin), checkpoint, &[]);

        let args = args_of(&cmd);
        assert!(!args.iter().any(|a| a == "-m"));
        assert!(!args.iter().any(|a| a == "hexo_a0.inference_server"));
    }
}
