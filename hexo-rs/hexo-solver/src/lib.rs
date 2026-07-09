//! Fully-forcing (VCF) threat-space solver for HeXO + deep proof-search drivers.
//!
//! Pure combinatorial search with no neural-net, MCTS, or PyO3 dependencies.

pub mod forcing;
pub mod position;
pub mod prover;

pub use forcing::DefenseAnalysis;
pub use position::{
    SolverEngine, SolverPosition, is_game_valid_board, solve_defense_from_position,
    solve_from_position, solve_wide_from_position,
};
