//! TorchScript model inference for self-play.
//!
//! Loads a TorchScript-exported HeXO model and evaluates game states
//! entirely in Rust via tch-rs (libtorch bindings). No Python needed.

#![cfg(feature = "torch")]

use rustc_hash::FxHashMap as HashMap;
use std::sync::Mutex;

use tch::{CModule, Device, Kind, Tensor};

use hexo_engine::types::Coord;
use hexo_engine::GameState;

use crate::graph_tensors::{GraphTensors, build_graph_tensors, build_axis_graph_tensors};

// Re-export GraphType from graph_tensors so existing consumers keep working.
pub use crate::graph_tensors::GraphType;

/// Lightweight (total_nodes, total_edges) histogram shared across all
/// `forward_graphs*` call sites. Every `SHAPE_LOG_INTERVAL` batches we dump
/// a compact summary to stderr and clear the buffer. This is a diagnostic
/// to right-size bucket boundaries for `forward_graphs_padded`.
///
/// Disabled by default; call `enable_shape_hist()` to turn on.
pub(crate) static SHAPE_HIST: Mutex<Vec<(u32, u32)>> = Mutex::new(Vec::new());
static SHAPE_HIST_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
const SHAPE_LOG_INTERVAL: usize = 100;

/// Enable shape histogram logging to stderr.
pub fn enable_shape_hist() {
    SHAPE_HIST_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn record_shape(total_nodes: usize, total_edges: usize) {
    if !SHAPE_HIST_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let mut buf = match SHAPE_HIST.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    buf.push((total_nodes as u32, total_edges as u32));
    if buf.len() >= SHAPE_LOG_INTERVAL {
        let mut nodes: Vec<u32> = buf.iter().map(|&(n, _)| n).collect();
        let mut edges: Vec<u32> = buf.iter().map(|&(_, e)| e).collect();
        nodes.sort_unstable();
        edges.sort_unstable();
        let n = nodes.len();
        let pct = |v: &[u32], p: f64| v[((n as f64 - 1.0) * p) as usize];
        eprintln!(
            "[shape-hist] count={} nodes(min/p50/p90/p99/max)={}/{}/{}/{}/{} edges(min/p50/p90/p99/max)={}/{}/{}/{}/{}",
            n,
            nodes[0], pct(&nodes, 0.5), pct(&nodes, 0.9), pct(&nodes, 0.99), nodes[n - 1],
            edges[0], pct(&edges, 0.5), pct(&edges, 0.9), pct(&edges, 0.99), edges[n - 1],
        );
        buf.clear();
    }
}

/// A loaded TorchScript model for inference.
pub struct TorchModel {
    model: CModule,
    device: Device,
    /// If the model uses a reduced-precision dtype (bf16 or fp16),
    /// input tensors are cast to match before the forward pass.
    float_kind: Kind,
    graph_type: GraphType,
    /// Whether the model's forward() accepts precomputed index tensors
    /// (legal_idx, stone_idx, stone_batch) as arguments 8-10.
    /// Auto-detected on first forward call: if 10-arg call fails with an
    /// argument-count error, falls back to 7-arg mode for all subsequent calls.
    /// New models (11 args incl. self) eliminate GPU→CPU nonzero syncs.
    supports_index_tensors: std::sync::atomic::AtomicU8,  // 0=unknown, 1=yes, 2=no
}

impl TorchModel {
    /// Load a TorchScript model from a file.
    ///
    /// Auto-detects reduced-precision models by checking the first parameter's dtype.
    pub fn load(path: &str, device: Device) -> Result<Self, tch::TchError> {
        Self::load_with_graph_type(path, device, GraphType::Hex)
    }

    /// Load a TorchScript model with an explicit graph type.
    pub fn load_with_graph_type(
        path: &str,
        device: Device,
        graph_type: GraphType,
    ) -> Result<Self, tch::TchError> {
        let model = CModule::load_on_device(path, device)?;
        let float_kind = model
            .named_parameters()
            .map(|params: Vec<(String, Tensor)>| {
                params.first().map(|(_, t)| t.kind()).unwrap_or(Kind::Float)
            })
            .unwrap_or(Kind::Float);
        match float_kind {
            Kind::BFloat16 => eprintln!("Model loaded in bfloat16 mode"),
            Kind::Half => eprintln!("Model loaded in float16 mode"),
            _ => {}
        }
        eprintln!("Model graph_type: {:?}", graph_type);
        Ok(TorchModel {
            model, device, float_kind, graph_type,
            supports_index_tensors: std::sync::atomic::AtomicU8::new(0),
        })
    }

    /// Build graph tensors for a batch of game states.
    pub fn build_graphs(&self, states: &[GameState]) -> Vec<GraphTensors> {
        match self.graph_type {
            GraphType::Hex => states.iter().map(|s| build_graph_tensors(s)).collect(),
            GraphType::Axis => states.iter().map(|s| build_axis_graph_tensors(s)).collect(),
        }
    }

    /// Evaluate a batch of game states.
    ///
    /// Returns (logits_per_state, values) where:
    /// - logits_per_state[i] maps Coord → logit for state i
    /// - values[i] is the scalar value estimate
    pub fn evaluate(&self, states: &[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let graphs = self.build_graphs(states);
        self.forward_graphs(graphs)
    }

    /// Run model forward pass on pre-built graph tensors.
    pub fn forward_graphs(&self, graphs: Vec<GraphTensors>) -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let n = graphs.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        // Concatenate into batched tensors
        let mut total_nodes = 0usize;
        let mut total_edges = 0usize;
        for g in &graphs {
            total_nodes += g.num_nodes;
            total_edges += g.num_edges;
        }
        record_shape(total_nodes, total_edges);

        // Node feature dim: 7/8 base (relative/absolute stones), +4 with
        // threat features. Derive from the actual graphs so non-8-dim batches
        // reshape correctly.
        let fdim = graphs[0].features.len() / graphs[0].num_nodes.max(1);
        debug_assert!(
            graphs.iter().all(|g| g.features.len() / g.num_nodes.max(1) == fdim),
            "batch contains graphs with mixed feature dims"
        );

        // Features: (total_nodes, fdim)
        let mut all_features: Vec<f32> = Vec::with_capacity(total_nodes * fdim);
        let mut all_edge_src: Vec<i64> = Vec::with_capacity(total_edges);
        let mut all_edge_dst: Vec<i64> = Vec::with_capacity(total_edges);
        let mut all_edge_attr: Vec<f32> = Vec::new();
        let mut all_legal_mask: Vec<bool> = Vec::with_capacity(total_nodes);
        let mut all_stone_mask: Vec<bool> = Vec::with_capacity(total_nodes);
        let mut all_batch: Vec<i64> = Vec::with_capacity(total_nodes);

        let mut node_offset: i64 = 0;
        for (gi, g) in graphs.iter().enumerate() {
            all_features.extend_from_slice(&g.features);
            // Offset edge indices
            for &s in &g.edge_src {
                all_edge_src.push(s + node_offset);
            }
            for &d in &g.edge_dst {
                all_edge_dst.push(d + node_offset);
            }
            // Batch edge_attr (axis graphs only)
            if let Some(ref ea) = g.edge_attr {
                all_edge_attr.extend_from_slice(ea);
            }
            all_legal_mask.extend_from_slice(&g.legal_mask);
            all_stone_mask.extend_from_slice(&g.stone_mask);
            for _ in 0..g.num_nodes {
                all_batch.push(gi as i64);
            }
            node_offset += g.num_nodes as i64;
        }

        // Create tensors on device (convert to bf16 if model is bf16)
        let mut x = Tensor::from_slice(&all_features)
            .reshape([total_nodes as i64, fdim as i64])
            .to_device(self.device);
        if self.float_kind != Kind::Float {
            x = x.to_kind(self.float_kind);
        }
        let edge_index = Tensor::from_slice2(&[&all_edge_src, &all_edge_dst])
            .to_device(self.device);
        let legal_mask = Tensor::from_slice(
            &all_legal_mask.iter().map(|&b| b as i8).collect::<Vec<i8>>(),
        ).to_kind(Kind::Bool).to_device(self.device);
        let stone_mask = Tensor::from_slice(
            &all_stone_mask.iter().map(|&b| b as i8).collect::<Vec<i8>>(),
        ).to_kind(Kind::Bool).to_device(self.device);
        let batch_tensor = Tensor::from_slice(&all_batch).to_device(self.device);
        let num_graphs = n as i64;

        // Build IValue list — axis models get edge_attr as 7th argument
        let mut ivalues = vec![
            tch::IValue::Tensor(x),
            tch::IValue::Tensor(edge_index),
            tch::IValue::Tensor(legal_mask),
            tch::IValue::Tensor(stone_mask),
            tch::IValue::Tensor(batch_tensor),
            tch::IValue::Int(num_graphs),
        ];
        match self.graph_type {
            GraphType::Axis => {
                let mut edge_attr_tensor = Tensor::from_slice(&all_edge_attr)
                    .reshape([total_edges as i64, 5])
                    .to_device(self.device);
                if self.float_kind != Kind::Float {
                    edge_attr_tensor = edge_attr_tensor.to_kind(self.float_kind);
                }
                ivalues.push(tch::IValue::Tensor(edge_attr_tensor));
            }
            GraphType::Hex => {
                // Pass empty tensor sentinel for hex (TorchScript default arg)
                let empty = Tensor::zeros([0], (Kind::Float, self.device));
                ivalues.push(tch::IValue::Tensor(empty));
            }
        }

        // Precompute index tensors to eliminate GPU→CPU nonzero syncs.
        // Only appended if the model supports them (auto-detected on first call).
        let idx_support = self.supports_index_tensors.load(std::sync::atomic::Ordering::Relaxed);
        if idx_support != 2 {
            let mut legal_idx: Vec<i64> = Vec::new();
            let mut stone_idx: Vec<i64> = Vec::new();
            let mut stone_batch_vec: Vec<i64> = Vec::new();
            for i in 0..total_nodes {
                if all_legal_mask[i] {
                    legal_idx.push(i as i64);
                }
                if all_stone_mask[i] {
                    stone_idx.push(i as i64);
                    stone_batch_vec.push(all_batch[i]);
                }
            }
            ivalues.push(tch::IValue::Tensor(Tensor::from_slice(&legal_idx).to_device(self.device)));
            ivalues.push(tch::IValue::Tensor(Tensor::from_slice(&stone_idx).to_device(self.device)));
            ivalues.push(tch::IValue::Tensor(Tensor::from_slice(&stone_batch_vec).to_device(self.device)));
        }

        // Run model via IValue interface (num_graphs is int, not Tensor)
        let _guard = tch::no_grad_guard();
        let result = match self.model.forward_is(&ivalues) {
            Ok(r) => {
                if idx_support == 0 {
                    eprintln!("Model supports index tensors — GPU nonzero syncs eliminated");
                    self.supports_index_tensors.store(1, std::sync::atomic::Ordering::Relaxed);
                }
                r
            }
            Err(e) if idx_support == 0 && format!("{e}").contains("Expected at most") => {
                // Old model without index tensor params — retry without them
                eprintln!("Model does not support index tensors, using boolean masks (re-export to upgrade)");
                self.supports_index_tensors.store(2, std::sync::atomic::Ordering::Relaxed);
                ivalues.truncate(ivalues.len() - 3);
                match self.model.forward_is(&ivalues) {
                    Ok(r) => r,
                    Err(e2) => {
                        eprintln!("Model forward failed on retry: {e2}");
                        let dummy_logits = (0..n).map(|_| HashMap::default()).collect();
                        return (dummy_logits, vec![0.0; n]);
                    }
                }
            }
            Err(e) => {
                eprintln!("Model forward failed: {e}");
                let dummy_logits = (0..n).map(|_| HashMap::default()).collect();
                return (dummy_logits, vec![0.0; n]);
            }
        };

        // Parse tuple result — return dummy on unexpected format
        let (all_logits_tensor, legal_counts_tensor, values_tensor) = match result {
            tch::IValue::Tuple(ref parts) if parts.len() == 3 => {
                match (&parts[0], &parts[1], &parts[2]) {
                    (tch::IValue::Tensor(l), tch::IValue::Tensor(c), tch::IValue::Tensor(v)) => {
                        (l.shallow_clone(), c.shallow_clone(), v.shallow_clone())
                    }
                    _ => {
                        eprintln!("Unexpected tensor types in model output");
                        let dummy = (0..n).map(|_| HashMap::default()).collect();
                        return (dummy, vec![0.0; n]);
                    }
                }
            }
            _ => {
                eprintln!("Unexpected model output format");
                let dummy = (0..n).map(|_| HashMap::default()).collect();
                return (dummy, vec![0.0; n]);
            }
        };

        // Extract values
        let values_vec: Vec<f64> = match Vec::<f64>::try_from(values_tensor.to_device(Device::Cpu)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to extract values: {e}");
                return ((0..n).map(|_| HashMap::default()).collect(), vec![0.0; n]);
            }
        };

        // Extract logits per graph
        let counts_vec: Vec<i64> = match Vec::<i64>::try_from(legal_counts_tensor.to_device(Device::Cpu)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to extract counts: {e}");
                return ((0..n).map(|_| HashMap::default()).collect(), vec![0.0; n]);
            }
        };
        let all_logits_vec: Vec<f64> = match Vec::<f64>::try_from(all_logits_tensor.to_device(Device::Cpu)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to extract logits: {e}");
                return ((0..n).map(|_| HashMap::default()).collect(), vec![0.0; n]);
            }
        };

        let mut logits_maps = Vec::with_capacity(n);
        let mut offset = 0usize;
        for (gi, g) in graphs.iter().enumerate() {
            let count = counts_vec[gi] as usize;
            let logits_slice = &all_logits_vec[offset..offset + count];
            let legal_coords = &g.legal_coords;
            let map: HashMap<Coord, f64> = legal_coords
                .iter()
                .zip(logits_slice)
                .map(|(&c, &l)| (c, l))
                .collect();
            logits_maps.push(map);
            offset += count;
        }

        (logits_maps, values_vec)
    }

    /// Bucket-padded variant of `forward_graphs`.
    ///
    /// Pads `total_nodes` and `total_edges` to a small set of fixed buckets
    /// via a "ghost" graph appended at batch index `n`. This stabilises the
    /// HIP caching allocator (which fragments badly when input shapes vary
    /// every call). The ghost graph has all-zero features, false legal/stone
    /// masks, and self-loops only — so it contributes 0 logits, its value
    /// is discarded, and message passing cannot touch real nodes.
    pub fn forward_graphs_padded(
        &self,
        graphs: Vec<GraphTensors>,
    ) -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let n = graphs.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        let mut real_nodes = 0usize;
        let mut real_edges = 0usize;
        for g in &graphs {
            real_nodes += g.num_nodes;
            real_edges += g.num_edges;
        }
        record_shape(real_nodes, real_edges);

        let target_nodes = pick_bucket(real_nodes + 1, NODE_BUCKETS);
        // Pick edge bucket set based on observed edge/node ratio:
        // unpruned axis graphs have ~20-28× ratio, pruned ~3.5-8×.
        let edge_ratio = if real_nodes > 0 { real_edges / real_nodes } else { 0 };
        let edge_buckets = if edge_ratio <= 10 { EDGE_BUCKETS_PRUNED } else { EDGE_BUCKETS };
        let target_edges = pick_bucket(real_edges + 1, edge_buckets);
        let ghost_nodes = target_nodes - real_nodes;
        let ghost_edges = target_edges - real_edges;
        debug_assert!(ghost_nodes >= 1);
        debug_assert!(ghost_edges >= 1);

        // Node feature dim: 7/8 base (relative/absolute stones), +4 with
        // threat features (ghost-node padding features stay all-zero either way).
        let fdim = graphs[0].features.len() / graphs[0].num_nodes.max(1);
        debug_assert!(
            graphs.iter().all(|g| g.features.len() / g.num_nodes.max(1) == fdim),
            "batch contains graphs with mixed feature dims"
        );
        let mut all_features: Vec<f32> = vec![0.0; target_nodes * fdim];
        let mut all_edge_src: Vec<i64> = Vec::with_capacity(target_edges);
        let mut all_edge_dst: Vec<i64> = Vec::with_capacity(target_edges);
        let edge_attr_dim = 5usize;
        let mut all_edge_attr: Vec<f32> = if matches!(self.graph_type, GraphType::Axis) {
            vec![0.0; target_edges * edge_attr_dim]
        } else {
            Vec::new()
        };
        let mut all_legal_mask: Vec<bool> = vec![false; target_nodes];
        let mut all_stone_mask: Vec<bool> = vec![false; target_nodes];
        let mut all_batch: Vec<i64> = vec![0; target_nodes];

        let mut node_offset: usize = 0;
        let mut edge_offset: usize = 0;
        for (gi, g) in graphs.iter().enumerate() {
            let f_dst =
                &mut all_features[node_offset * fdim..(node_offset + g.num_nodes) * fdim];
            f_dst.copy_from_slice(&g.features);
            for &s in &g.edge_src {
                all_edge_src.push(s + node_offset as i64);
            }
            for &d in &g.edge_dst {
                all_edge_dst.push(d + node_offset as i64);
            }
            if let Some(ref ea) = g.edge_attr {
                let ea_dst = &mut all_edge_attr
                    [edge_offset * edge_attr_dim..(edge_offset + g.num_edges) * edge_attr_dim];
                ea_dst.copy_from_slice(ea);
            }
            all_legal_mask[node_offset..node_offset + g.num_nodes]
                .copy_from_slice(&g.legal_mask);
            all_stone_mask[node_offset..node_offset + g.num_nodes]
                .copy_from_slice(&g.stone_mask);
            for slot in &mut all_batch[node_offset..node_offset + g.num_nodes] {
                *slot = gi as i64;
            }
            node_offset += g.num_nodes;
            edge_offset += g.num_edges;
        }
        debug_assert_eq!(node_offset, real_nodes);
        debug_assert_eq!(edge_offset, real_edges);

        let ghost_batch_idx = n as i64;
        for slot in &mut all_batch[real_nodes..target_nodes] {
            *slot = ghost_batch_idx;
        }
        for k in 0..ghost_edges {
            let node = real_nodes + (k % ghost_nodes);
            all_edge_src.push(node as i64);
            all_edge_dst.push(node as i64);
        }
        debug_assert_eq!(all_edge_src.len(), target_edges);
        debug_assert_eq!(all_edge_dst.len(), target_edges);

        let mut x = Tensor::from_slice(&all_features)
            .reshape([target_nodes as i64, fdim as i64])
            .to_device(self.device);
        if self.float_kind != Kind::Float {
            x = x.to_kind(self.float_kind);
        }
        let edge_index =
            Tensor::from_slice2(&[&all_edge_src, &all_edge_dst]).to_device(self.device);
        let legal_mask = Tensor::from_slice(
            &all_legal_mask.iter().map(|&b| b as i8).collect::<Vec<i8>>(),
        )
        .to_kind(Kind::Bool)
        .to_device(self.device);
        let stone_mask = Tensor::from_slice(
            &all_stone_mask.iter().map(|&b| b as i8).collect::<Vec<i8>>(),
        )
        .to_kind(Kind::Bool)
        .to_device(self.device);
        let batch_tensor = Tensor::from_slice(&all_batch).to_device(self.device);
        let num_graphs_with_ghost = (n + 1) as i64;

        let mut ivalues = vec![
            tch::IValue::Tensor(x),
            tch::IValue::Tensor(edge_index),
            tch::IValue::Tensor(legal_mask),
            tch::IValue::Tensor(stone_mask),
            tch::IValue::Tensor(batch_tensor),
            tch::IValue::Int(num_graphs_with_ghost),
        ];
        match self.graph_type {
            GraphType::Axis => {
                let mut edge_attr_tensor = Tensor::from_slice(&all_edge_attr)
                    .reshape([target_edges as i64, edge_attr_dim as i64])
                    .to_device(self.device);
                if self.float_kind != Kind::Float {
                    edge_attr_tensor = edge_attr_tensor.to_kind(self.float_kind);
                }
                ivalues.push(tch::IValue::Tensor(edge_attr_tensor));
            }
            GraphType::Hex => {
                let empty = Tensor::zeros([0], (Kind::Float, self.device));
                ivalues.push(tch::IValue::Tensor(empty));
            }
        }

        // Precompute index tensors (only over real nodes, not ghost padding)
        let idx_support = self.supports_index_tensors.load(std::sync::atomic::Ordering::Relaxed);
        if idx_support != 2 {
            let mut legal_idx: Vec<i64> = Vec::new();
            let mut stone_idx: Vec<i64> = Vec::new();
            let mut stone_batch_vec: Vec<i64> = Vec::new();
            for i in 0..real_nodes {
                if all_legal_mask[i] {
                    legal_idx.push(i as i64);
                }
                if all_stone_mask[i] {
                    stone_idx.push(i as i64);
                    stone_batch_vec.push(all_batch[i]);
                }
            }
            ivalues.push(tch::IValue::Tensor(Tensor::from_slice(&legal_idx).to_device(self.device)));
            ivalues.push(tch::IValue::Tensor(Tensor::from_slice(&stone_idx).to_device(self.device)));
            ivalues.push(tch::IValue::Tensor(Tensor::from_slice(&stone_batch_vec).to_device(self.device)));
        }

        let _guard = tch::no_grad_guard();
        let result = match self.model.forward_is(&ivalues) {
            Ok(r) => {
                if idx_support == 0 {
                    self.supports_index_tensors.store(1, std::sync::atomic::Ordering::Relaxed);
                }
                r
            }
            Err(e) if idx_support == 0 && format!("{e}").contains("Expected at most") => {
                eprintln!("Model does not support index tensors, using boolean masks (re-export to upgrade)");
                self.supports_index_tensors.store(2, std::sync::atomic::Ordering::Relaxed);
                ivalues.truncate(ivalues.len() - 3);
                match self.model.forward_is(&ivalues) {
                    Ok(r) => r,
                    Err(e2) => {
                        eprintln!("Padded model forward failed on retry: {e2}");
                        let dummy_logits = (0..n).map(|_| HashMap::default()).collect();
                        return (dummy_logits, vec![0.0; n]);
                    }
                }
            }
            Err(e) => {
                eprintln!("Padded model forward failed: {e}");
                let dummy_logits = (0..n).map(|_| HashMap::default()).collect();
                return (dummy_logits, vec![0.0; n]);
            }
        };

        let (all_logits_tensor, legal_counts_tensor, values_tensor) = match result {
            tch::IValue::Tuple(ref parts) if parts.len() == 3 => match (&parts[0], &parts[1], &parts[2]) {
                (tch::IValue::Tensor(l), tch::IValue::Tensor(c), tch::IValue::Tensor(v)) => {
                    (l.shallow_clone(), c.shallow_clone(), v.shallow_clone())
                }
                _ => {
                    eprintln!("Unexpected tensor types in padded model output");
                    let dummy = (0..n).map(|_| HashMap::default()).collect();
                    return (dummy, vec![0.0; n]);
                }
            },
            _ => {
                eprintln!("Unexpected padded model output format");
                let dummy = (0..n).map(|_| HashMap::default()).collect();
                return (dummy, vec![0.0; n]);
            }
        };

        let values_vec_full: Vec<f64> =
            match Vec::<f64>::try_from(values_tensor.to_device(Device::Cpu)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Failed to extract padded values: {e}");
                    return ((0..n).map(|_| HashMap::default()).collect(), vec![0.0; n]);
                }
            };
        let values_vec: Vec<f64> = values_vec_full[..n].to_vec();

        let counts_vec: Vec<i64> =
            match Vec::<i64>::try_from(legal_counts_tensor.to_device(Device::Cpu)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Failed to extract padded counts: {e}");
                    return ((0..n).map(|_| HashMap::default()).collect(), vec![0.0; n]);
                }
            };
        debug_assert_eq!(counts_vec.len(), n + 1);
        debug_assert_eq!(counts_vec[n], 0, "ghost graph must contribute 0 legal moves");

        let all_logits_vec: Vec<f64> =
            match Vec::<f64>::try_from(all_logits_tensor.to_device(Device::Cpu)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Failed to extract padded logits: {e}");
                    return ((0..n).map(|_| HashMap::default()).collect(), vec![0.0; n]);
                }
            };

        let mut logits_maps = Vec::with_capacity(n);
        let mut offset = 0usize;
        for (gi, g) in graphs.iter().enumerate() {
            let count = counts_vec[gi] as usize;
            let logits_slice = &all_logits_vec[offset..offset + count];
            let map: HashMap<Coord, f64> = g
                .legal_coords
                .iter()
                .zip(logits_slice)
                .map(|(&c, &l)| (c, l))
                .collect();
            logits_maps.push(map);
            offset += count;
        }

        (logits_maps, values_vec)
    }
}

/// Node-count buckets for `forward_graphs_padded`. Powers of two give
/// predictable allocator behaviour and only ~7 distinct shapes across the
/// full range observed in self-play.
// Sized from observed self-play distributions: S1 5-5 warm-started from a
// 4-3 checkpoint shows p50≈18k, p99≈22k nodes. Top buckets provide headroom
// for later-stage boards (S2 6-6, S3 6-8) and the inevitable drift toward
// longer games as policies learn to defend. Keep bucket count small so the
// caching allocator sees at most ~6 distinct node shapes.
const NODE_BUCKETS: &[usize] = &[4096, 16384, 32768, 65536, 131072, 196608];

/// Edge-count buckets for `forward_graphs_padded`.
// Edge/node ratio is ~20-28× for unpruned axis graphs (3 axes × 2 directions
// × up to win_length-1 steps per direction, plus global-dummy-node edges).
// Each bucket is sized at ~24× the corresponding node bucket to leave slack.
const EDGE_BUCKETS: &[usize] = &[98304, 393216, 786432, 1572864, 3145728, 4718592];

/// Edge-count buckets for pruned axis graphs (empty→empty edges removed).
// Edge/node ratio drops to ~3.5-8× when empty→empty edges are pruned.
// Buckets sized at ~10× the corresponding node bucket for headroom.
const EDGE_BUCKETS_PRUNED: &[usize] = &[16384, 65536, 131072, 262144, 524288, 1048576];

/// Counts how many times `pick_bucket` has exceeded its top bucket and
/// fallen through to the `div_ceil` path. Each overflow introduces a new
/// tensor shape to the caching allocator, so a sustained stream of them
/// means the padding strategy is no longer doing its job and buckets need
/// to be resized.
static BUCKET_OVERFLOWS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[inline]
fn pick_bucket(n: usize, buckets: &[usize]) -> usize {
    for &b in buckets {
        if n <= b {
            return b;
        }
    }
    let largest = *buckets.last().unwrap();
    let padded = n.div_ceil(largest) * largest;
    // Rate-limited warning: first 5 overflows always log, then every 100th.
    let c = BUCKET_OVERFLOWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if c <= 5 || c.is_multiple_of(100) {
        eprintln!(
            "[pick_bucket] WARNING: size {} exceeds top bucket {}, padding to {} \
             (overflow count: {}). Consider enlarging buckets.",
            n, largest, padded, c,
        );
    }
    padded
}
