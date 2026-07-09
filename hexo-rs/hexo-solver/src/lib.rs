//! Fully-forcing (VCF) threat-space solver for HeXO + deep proof-search drivers.
//!
//! Pure combinatorial search with no neural-net, MCTS, or PyO3 dependencies.

pub mod forcing;
pub mod position;
pub mod prover;

pub use position::{
    SolverEngine, SolverPosition, solve_from_position, solve_wide_from_position,
};
