//! The `hexo_rs` Python extension module.
//!
//! Re-registers everything from `hexo_rs::python` (the hexo-mcts crate) and
//! adds the hexo-infer-backed native inference classes. This glue lives in its
//! own crate because hexo-infer depends on hexo-mcts, so hexo-mcts cannot
//! reference hexo-infer without a dependency cycle.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use hexo_infer::InferModel;
use hexo_rs::hexo_engine::GameState;
use hexo_rs::mcts::gumbel_mcts::gumbel_mcts;
use hexo_rs::python::{PyGameState, PyMCTSConfig};

/// Native (no-torch) inference model over safetensors weights.
///
/// The forward pass and the full Gumbel search run in Rust with the GIL
/// released; batched leaf evals fan out over OS threads.
#[pyclass(name = "InferModel", frozen)]
pub struct PyInferModel {
    inner: Arc<InferModel>,
}

#[pymethods]
impl PyInferModel {
    #[new]
    fn new(weights: Vec<u8>) -> PyResult<Self> {
        InferModel::from_safetensors(&weights)
            .map(|m| PyInferModel { inner: Arc::new(m) })
            .map_err(|e| PyValueError::new_err(format!("hexo-infer load failed: {e}")))
    }

    /// Raw safetensors metadata as a JSON string (format, model_config,
    /// game_config, train_steps, source_checkpoint).
    fn metadata_json(&self) -> String {
        self.inner.config().metadata_json.clone()
    }

    /// Raw network eval of one state: (dict {(q, r): logit over legal moves},
    /// value in [-1, 1] from the side-to-move's perspective).
    fn forward(&self, py: Python<'_>, state: &PyGameState) -> (HashMap<(i32, i32), f64>, f64) {
        let model = self.inner.clone();
        let st = state.inner.clone();
        py.detach(move || {
            let (map, value) = model.eval_one(&st);
            (map.into_iter().collect(), value)
        })
    }

    /// Native-eval twin of `hexo_rs.gumbel_mcts_with_diagnostics`: identical
    /// return tuple, but the search never re-enters Python.
    #[pyo3(signature = (game, config, seed=None, forced_candidates=None))]
    #[allow(clippy::type_complexity)]
    fn mcts_with_diagnostics(
        &self,
        py: Python<'_>,
        game: &PyGameState,
        config: &PyMCTSConfig,
        seed: Option<u64>,
        forced_candidates: Option<Vec<(i32, i32)>>,
    ) -> PyResult<(
        (i32, i32),
        Vec<f64>,
        Vec<u32>,
        Vec<f64>,
        Vec<f64>,
        Vec<usize>,
        Vec<(i32, i32)>,
    )> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let model = self.inner.clone();
        let g = game.inner.clone();
        let cfg = config.inner.clone();
        let result = py
            .detach(move || {
                let mut rng = match seed {
                    Some(s) => ChaCha8Rng::seed_from_u64(s),
                    None => ChaCha8Rng::from_os_rng(),
                };
                let mut eval = |states: &[GameState]| model.eval_states(states);
                gumbel_mcts(&g, &cfg, &mut rng, forced_candidates.as_deref(), &mut eval)
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            result.action,
            result.improved_policy,
            result.visit_counts,
            result.per_child_q,
            result.per_child_prior,
            result.candidate_indices,
            result.chosen_action_forced_candidates,
        ))
    }
}

#[pymodule]
#[pyo3(name = "hexo_rs")]
fn hexo_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ::hexo_rs::python::register(m)?;
    m.add_class::<PyInferModel>()?;
    Ok(())
}
