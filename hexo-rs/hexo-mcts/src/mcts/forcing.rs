//! Thin re-export shim. The solver core lives in `hexo_solver::forcing`.
//! Public function signatures (taking `&GameState`) are preserved for the
//! PyO3 bindings and the MCTS depth-N shortcut.

pub use hexo_solver::forcing::*;
