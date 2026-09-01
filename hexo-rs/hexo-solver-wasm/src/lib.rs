//! Standalone solver-only WASM API for the HeXO forced-win prover.
//!
//! This package deliberately contains ONLY the combinatorial solver:
//! `hexo-engine` + `hexo-solver` + `wasm-bindgen`. No neural network, no
//! `hexo-infer`, no `hexo-mcts`, no weights download — a `StrixSolver`
//! instance carries no model and works fully offline.
//!
//! The wrapper source is shared verbatim with `hexo-wasm` (the full bot
//! package) via a path include, so both packages expose byte-identical
//! solver semantics and there is a single source of truth for the API.
//! `solver.rs` is a top-level module in both crates, so its `super::` test
//! imports resolve identically.

#[path = "../../hexo-wasm/src/solver.rs"]
pub mod solver;
