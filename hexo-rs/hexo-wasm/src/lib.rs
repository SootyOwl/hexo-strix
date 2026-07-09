//! wasm-bindgen WebWorker API for the strixbot HeXO player.
//!
//! `StrixBot` wraps hexo-infer (pure-Rust GNN forward pass) + the existing
//! `gumbel_mcts` search. Everything heavy is target-independent in `position.rs`
//! (natively tested); this file is only the FFI shell.
//!
//! Panic policy: `validate_position` pre-empts every known `from_state` panic, and
//! a `catch_unwind` fence turns any *unanticipated* panic in the search path into a
//! clean `Err` instead of unwinding across the FFI boundary (default wasm32 panic
//! strategy is unwind; do NOT set panic=abort).

pub mod position;
pub mod solver;

use hexo_infer::{InferError, InferModel};
use std::panic::{catch_unwind, AssertUnwindSafe};
use wasm_bindgen::prelude::*;

// Orphan rule forbids `impl From<InferError> for JsError` (both foreign types);
// a helper conversion does the same job.
fn js_err(e: InferError) -> JsError {
    JsError::new(&e.to_string())
}

fn fenced<T>(f: impl FnOnce() -> Result<T, InferError>) -> Result<T, JsError> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r.map_err(js_err),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            Err(JsError::new(&format!("internal error: {msg}")))
        }
    }
}

#[wasm_bindgen]
pub struct StrixBot {
    model: InferModel,
}

#[wasm_bindgen]
impl StrixBot {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8]) -> Result<StrixBot, JsError> {
        Ok(StrixBot {
            model: InferModel::from_safetensors(weights).map_err(js_err)?,
        })
    }

    /// Search for the best move. Position JSON per the crate README schema.
    /// NOTE: with disable_gumbel_noise + dirichlet off the search consumes NO
    /// randomness — the bot is fully deterministic and `seed` is currently INERT
    /// (reserved for future variety modes; documented in the README).
    pub fn best_move(&self, position_json: &str, sims: u32, m_actions: u32, seed: u64) -> Result<String, JsError> {
        fenced(|| position::best_move_impl(&self.model, position_json, sims, m_actions, seed))
    }

    /// Raw net eval, no search: {"value": f, "policy": [{q,r,p}...]} (softmaxed prior).
    /// `policy` here is the RAW prior — best_move's `policy` is the search-improved
    /// distribution; same shape, different numbers.
    pub fn evaluate(&self, position_json: &str) -> Result<String, JsError> {
        fenced(|| position::evaluate_impl(&self.model, position_json))
    }

    /// Model metadata passthrough: model_config, train_steps, source_checkpoint,
    /// and (if present) the training game_config — clients should mirror it in
    /// their position `config`.
    pub fn model_info(&self) -> String {
        self.model.config().metadata_json.clone()
    }
}