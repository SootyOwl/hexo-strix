//! The `gumbel_mcts` evaluator: batch of states -> (per-state coord->logit maps, values).
//!
//! This is the exact closure body `gumbel_mcts` expects:
//!   `FnMut(&[GameState]) -> (Vec<FxHashMap<Coord, f64>>, Vec<f64>)`
//! Logits are RAW (the search softmaxes internally); values are side-to-move-relative.

use crate::forward::InferModel;
use hexo_rs::axis_graph::game_to_axis_graph_raw_opts;
use hexo_rs::hexo_engine::types::Coord;
use hexo_rs::hexo_engine::GameState;
use rustc_hash::FxHashMap;

impl InferModel {
    /// Evaluate one state: graph build -> forward -> coord->logit map + value.
    pub fn eval_one(&self, state: &GameState) -> (FxHashMap<Coord, f64>, f64) {
        let cfg = self.config();
        let g = game_to_axis_graph_raw_opts(
            state,
            cfg.prune_empty_edges,
            cfg.threat_features,
            cfg.relative_stones,
        );
        let (logits, value) = self.forward(&g);
        let legal = state.legal_moves();
        debug_assert_eq!(legal.len(), logits.len());
        let mut map = FxHashMap::default();
        for (coord, logit) in legal.iter().zip(logits.iter()) {
            map.insert(*coord, *logit as f64);
        }
        (map, value as f64)
    }

    /// Evaluate a batch of states. On native, states are chunked across OS
    /// threads (states are independent, so results are bitwise identical to
    /// the serial path and returned in order). On wasm the loop stays serial.
    pub fn eval_states(&self, states: &[GameState]) -> (Vec<FxHashMap<Coord, f64>>, Vec<f64>) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(states.len());
            if threads > 1 {
                let chunk = states.len().div_ceil(threads);
                let per_chunk: Vec<Vec<(FxHashMap<Coord, f64>, f64)>> = std::thread::scope(|s| {
                    let handles: Vec<_> = states
                        .chunks(chunk)
                        .map(|c| s.spawn(move || c.iter().map(|st| self.eval_one(st)).collect()))
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                return per_chunk.into_iter().flatten().unzip();
            }
        }
        states.iter().map(|st| self.eval_one(st)).unzip()
    }
}