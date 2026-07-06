//! The HeXONet forward pass over `AxisGraphData`, ported from the eager PyTorch
//! oracle (`hexo-a0/src/hexo_a0/model.py`, parity-pinned by tests/parity.rs).
//!
//! Spec (all f32, LayerNorm eps 1e-5):
//!   x = input_proj(X);  e = edge_proj(A)   (edge projection ONCE, shared by layers)
//!   per layer (pre-norm residual):
//!     h    = LN(x)
//!     m_k  = relu(h[src_k] + lin_i(e_k))         GINE message
//!     agg  = Σ_{k: dst_k=v} m_k                  sum at edge DST
//!     z    = (1+eps_i)*h + agg                   (eps ≡ 0 in practice: no train_eps)
//!     c    = nn2_i(relu(nn0_i(z)))
//!     x    = relu(c + x)                         residual add THEN relu
//!   rep = JK-cat: concat_i LN_final(h_i)  |  no-JK: LN_final(x)
//!   policy: per-legal-node MLP -> logits (legal graph order == legal_moves() order)
//!   value: mean-pool rep over stone nodes (zero stones -> zeros), MLP, tanh

use crate::ops;
use crate::weights::{InferConfig, InferError, ModelWeights};
use hexo_rs::axis_graph::AxisGraphData;

pub struct InferModel {
    pub weights: ModelWeights,
}

impl InferModel {
    pub fn from_safetensors(bytes: &[u8]) -> Result<InferModel, InferError> {
        Ok(InferModel { weights: ModelWeights::from_safetensors(bytes)? })
    }

    pub fn config(&self) -> &InferConfig {
        &self.weights.config
    }

    pub fn source_checkpoint(&self) -> &str {
        self.weights.source_checkpoint()
    }

    /// Forward pass: (logits in legal_moves() order, value in [-1,1] from to_move's view).
    pub fn forward(&self, g: &AxisGraphData) -> (Vec<f32>, f32) {
        let w = &self.weights;
        let cfg = &w.config;
        let h = cfg.hidden_dim;
        let n = g.num_nodes;
        let e = g.edge_src.len();
        debug_assert_eq!(g.features.len(), n * cfg.node_dim, "node feature stride");
        debug_assert_eq!(g.edge_attr.len(), e * 5, "edge attr stride");

        // Projections
        let mut x = Vec::new();
        ops::linear(&g.features, n, &w.input_proj_w, &w.input_proj_b, cfg.node_dim, h, &mut x);

        // Edge-attr dedupe: edge features are structural (axis one-hot ×
        // signed distance × window feature), so the E edges carry only
        // O(100) distinct rows (91 on a real r=8 midgame vs ~13k edges),
        // while the per-layer edge transform below is ~85% of forward FLOPs.
        // Run edge_proj and each layer's lin over UNIQUE rows and gather by
        // index per edge — bitwise identical to the per-edge computation
        // (`linear` is row-pure; pinned by
        // unique_row_gather_matches_per_row_linear_bitwise and the forward
        // fingerprint test).
        let mut row_index: rustc_hash::FxHashMap<[u32; 5], u32> =
            rustc_hash::FxHashMap::default();
        let mut unique_attr: Vec<f32> = Vec::new();
        let mut edge_row: Vec<u32> = Vec::with_capacity(e);
        for k in 0..e {
            let a = &g.edge_attr[k * 5..(k + 1) * 5];
            let key = [a[0].to_bits(), a[1].to_bits(), a[2].to_bits(), a[3].to_bits(), a[4].to_bits()];
            let next = (unique_attr.len() / 5) as u32;
            let idx = *row_index.entry(key).or_insert_with(|| {
                unique_attr.extend_from_slice(a);
                next
            });
            edge_row.push(idx);
        }
        let u = unique_attr.len() / 5;
        let mut eproj = Vec::new();
        ops::linear(&unique_attr, u, &w.edge_proj_w, &w.edge_proj_b, 5, h, &mut eproj);

        // Layers
        let mut hs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_layers);
        let mut normed = Vec::new();
        let mut lin_e = Vec::new();
        let mut z = vec![0.0f32; n * h];
        let mut c1 = Vec::new();
        let mut c2 = Vec::new();
        for layer in &w.layers {
            ops::layer_norm(&x, n, h, &layer.norm_w, &layer.norm_b, &mut normed);
            // Over unique edge-attr rows (u ≪ e), gathered via edge_row below.
            ops::linear(&eproj, u, &layer.lin_w, &layer.lin_b, h, h, &mut lin_e);
            // agg[dst] += relu(normed[src] + lin_e[k])  — accumulate straight into z
            // after seeding z with (1+eps)*normed.
            let scale = 1.0 + layer.eps;
            for i in 0..n * h {
                z[i] = scale * normed[i];
            }
            for k in 0..e {
                let s = g.edge_src[k] as usize;
                let d = g.edge_dst[k] as usize;
                let nk = &normed[s * h..(s + 1) * h];
                let er = edge_row[k] as usize;
                let ek = &lin_e[er * h..(er + 1) * h];
                let zd = &mut z[d * h..(d + 1) * h];
                for j in 0..h {
                    let m = nk[j] + ek[j];
                    if m > 0.0 {
                        zd[j] += m;
                    }
                }
            }
            ops::linear(&z, n, &layer.nn0_w, &layer.nn0_b, h, h, &mut c1);
            ops::relu_inplace(&mut c1);
            ops::linear(&c1, n, &layer.nn2_w, &layer.nn2_b, h, h, &mut c2);
            for i in 0..n * h {
                x[i] = (c2[i] + x[i]).max(0.0); // residual add THEN relu (pre-norm branch)
            }
            if cfg.use_jk_cat {
                hs.push(x.clone());
            }
        }

        // Representation: JK-cat of final_norm(h_i) in layer order, or final_norm(x).
        let d = if cfg.use_jk_cat { cfg.num_layers * h } else { h };
        let mut rep = vec![0.0f32; n * d];
        if cfg.use_jk_cat {
            for (li, hi) in hs.iter().enumerate() {
                ops::layer_norm(hi, n, h, &w.final_norm_w, &w.final_norm_b, &mut normed);
                for node in 0..n {
                    rep[node * d + li * h..node * d + (li + 1) * h]
                        .copy_from_slice(&normed[node * h..(node + 1) * h]);
                }
            }
        } else {
            ops::layer_norm(&x, n, h, &w.final_norm_w, &w.final_norm_b, &mut rep);
        }

        // Policy head: gather legal rows (graph order == legal_moves() order), MLP -> 1.
        let legal_rows: Vec<usize> = (0..n).filter(|&i| g.legal_mask[i]).collect();
        let nl = legal_rows.len();
        let mut legal_rep = vec![0.0f32; nl * d];
        for (r, &node) in legal_rows.iter().enumerate() {
            legal_rep[r * d..(r + 1) * d].copy_from_slice(&rep[node * d..(node + 1) * d]);
        }
        let mut p1 = Vec::new();
        ops::linear(&legal_rep, nl, &w.policy0_w, &w.policy0_b, d, cfg.policy_hidden, &mut p1);
        ops::relu_inplace(&mut p1);
        let mut logits = Vec::new();
        ops::linear(&p1, nl, &w.policy2_w, &w.policy2_b, cfg.policy_hidden, 1, &mut logits);

        // Value head: mean-pool rep over STONE nodes; zero stones -> zeros (matches
        // _forward_batch_core's scatter_add + clamp(min=1), NOT mean-over-all).
        let mut pooled = vec![0.0f32; d];
        let n_stones = g.stone_mask.iter().filter(|&&b| b).count();
        if n_stones > 0 {
            for node in 0..n {
                if g.stone_mask[node] {
                    for j in 0..d {
                        pooled[j] += rep[node * d + j];
                    }
                }
            }
            let inv = 1.0 / n_stones as f32;
            for v in pooled.iter_mut() {
                *v *= inv;
            }
        }
        let mut v1 = Vec::new();
        ops::linear(&pooled, 1, &w.value0_w, &w.value0_b, d, cfg.value_hidden, &mut v1);
        ops::relu_inplace(&mut v1);
        let mut v2 = Vec::new();
        ops::linear(&v1, 1, &w.value2_w, &w.value2_b, cfg.value_hidden, 1, &mut v2);
        let value = v2[0].tanh();

        (logits, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_rs::axis_graph::game_to_axis_graph_raw_opts;
    use hexo_rs::hexo_engine::{GameConfig, GameState};

    fn tiny_model() -> InferModel {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.safetensors");
        InferModel::from_safetensors(&std::fs::read(p).unwrap()).unwrap()
    }

    /// Bit-level pin of the full forward, captured from the pre-dedupe
    /// per-edge implementation. The edge-attr dedupe must be EXACTLY
    /// behavior-preserving (same input row through the same linear ⇒ same
    /// output row, gathered), so any changed bit here is a bug, not drift.
    #[test]
    fn forward_bits_fingerprint_pinned() {
        let model = tiny_model();
        let mut game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
        for m in [(1, 2), (0, -2), (2, 0), (-1, 1), (3, -1), (0, 3), (-2, 0), (1, -3)] {
            game.apply_move(m).unwrap();
        }
        let cfg = model.config();
        let g = game_to_axis_graph_raw_opts(&game, cfg.prune_empty_edges, cfg.threat_features, cfg.relative_stones);
        let (logits, value) = model.forward(&g);
        let mut fp: u64 = logits.len() as u64;
        for v in logits.iter().chain(std::iter::once(&value)) {
            fp = fp.wrapping_mul(0x100000001b3).wrapping_add(v.to_bits() as u64);
        }
        assert_eq!(fp, 2037059312983329231, "forward output bits changed");
    }

    /// The dedupe's core soundness property, pinned against kernel changes:
    /// `linear` must be row-pure (each output row a function of its input row
    /// only), so unique-rows-then-gather is bitwise equal to per-edge rows.
    #[test]
    fn unique_row_gather_matches_per_row_linear_bitwise() {
        let model = tiny_model();
        let w = &model.weights;
        let mut game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
        for m in [(1, 2), (0, -2), (2, 0), (-1, 1)] {
            game.apply_move(m).unwrap();
        }
        let cfg = model.config();
        let g = game_to_axis_graph_raw_opts(&game, cfg.prune_empty_edges, cfg.threat_features, cfg.relative_stones);
        let e = g.edge_src.len();
        let h = cfg.hidden_dim;
        assert!(e > 0);

        // Per-edge reference.
        let mut full = Vec::new();
        ops::linear(&g.edge_attr, e, &w.edge_proj_w, &w.edge_proj_b, 5, h, &mut full);

        // Unique rows + gather.
        let mut index: rustc_hash::FxHashMap<[u32; 5], u32> = rustc_hash::FxHashMap::default();
        let mut unique: Vec<f32> = Vec::new();
        let mut row_of: Vec<u32> = Vec::with_capacity(e);
        for k in 0..e {
            let a = &g.edge_attr[k * 5..(k + 1) * 5];
            let key = [a[0].to_bits(), a[1].to_bits(), a[2].to_bits(), a[3].to_bits(), a[4].to_bits()];
            let next = (unique.len() / 5) as u32;
            let idx = *index.entry(key).or_insert_with(|| {
                unique.extend_from_slice(a);
                next
            });
            row_of.push(idx);
        }
        let u = unique.len() / 5;
        assert!(u < e, "fixture must contain duplicate edge_attr rows");
        let mut uproj = Vec::new();
        ops::linear(&unique, u, &w.edge_proj_w, &w.edge_proj_b, 5, h, &mut uproj);

        for k in 0..e {
            let a = &full[k * h..(k + 1) * h];
            let b = &uproj[row_of[k] as usize * h..(row_of[k] as usize + 1) * h];
            for j in 0..h {
                assert_eq!(a[j].to_bits(), b[j].to_bits(), "edge {k} dim {j}");
            }
        }
    }

    #[test]
    fn forward_shapes_sane() {
        let model = tiny_model();
        let game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
        let cfg = model.config();
        let g = game_to_axis_graph_raw_opts(&game, cfg.prune_empty_edges, cfg.threat_features, cfg.relative_stones);
        let (logits, value) = model.forward(&g);
        assert_eq!(logits.len(), game.legal_moves().len());
        assert!((-1.0..=1.0).contains(&value));
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}