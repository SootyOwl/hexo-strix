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
    /// Evaluate a batch of states one at a time (wasm is single-threaded; the
    /// serial `gumbel_mcts` path sends small batches).
    pub fn eval_states(&self, states: &[GameState]) -> (Vec<FxHashMap<Coord, f64>>, Vec<f64>) {
        let cfg = self.config();
        let mut logit_maps = Vec::with_capacity(states.len());
        let mut values = Vec::with_capacity(states.len());
        for state in states {
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
            logit_maps.push(map);
            values.push(value as f64);
        }
        (logit_maps, values)
    }
}