//! Dependency-light pure-Rust inference for HeXO GNN checkpoints.
//!
//! Reimplements the `HeXONet` forward pass (GINE + JK-cat + heads) over
//! [`hexo_mcts::axis_graph::AxisGraphData`] without libtorch/Python, so it can
//! compile to `wasm32-unknown-unknown` for in-browser/WebWorker play. Weights load
//! from a safetensors file exported by `hexo-a0 export`.

pub mod evaluator;
pub mod forward;
pub mod ops;
#[cfg(not(target_arch = "wasm32"))]
pub mod protocol;
#[cfg(not(target_arch = "wasm32"))]
pub mod server;
pub mod weights;

pub use forward::InferModel;
pub use weights::{InferConfig, InferError, ModelWeights};