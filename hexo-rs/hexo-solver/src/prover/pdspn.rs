//! PDS-PN driver — the two-level proof-number search of Winands, Uiterwijk & van
//! den Herik (2002), "PDS-PN: A New Proof-Number Search Algorithm".
//!
//! ## The paper's algorithm
//!
//! - **Level 1 (PDS):** a depth-first Proof-number and Disproof-number Search
//!   (Nagai) with multiple-iterative-deepening over proof/disproof thresholds,
//!   backed by a transposition table. It is not memory-bound (depth-first) but is
//!   comparatively slow because of delayed evaluation.
//! - **Level 2 (PN):** when level-1 must expand a node that is *not yet in the TT*,
//!   it runs a bounded, in-memory best-first PN search from that node and stores
//!   only the resulting root numbers back into the TT (PN²-style memory discipline:
//!   the level-2 tree is discarded). This gives level-1 far better than the `1/1`
//!   leaf heuristic while staying inside a fixed memory budget.
//!
//! ## This implementation
//!
//! Level 1 is the shared depth-first proof-number engine in [`super::dfpn`] (the
//! same `MID` with `1 + ε` thresholds, 2-way TT, and identical kernel node
//! semantics, so PDS-PN and df-pn agree by construction). PDS-PN adds the level-2
//! step: the first descent into any frontier node runs a bounded [`PnSearch`]
//! (capped at `--pn2-nodes`, default 50 000; tree discarded) to seed its numbers.
//!
//! **Parameter choices (the paper leaves these open; exact tuning is not
//! required):**
//! - The paper sizes each level-2 search by a logistic growth function `f(x)` of
//!   the level-1 tree size, with tuned `(a, b)` (best `(450K, 300K)`). We instead
//!   use a *fixed* per-frontier cap `--pn2-nodes` (default 50 000). A fixed cap is
//!   simpler, deterministic, and adequate for this VCF domain where wins are
//!   narrow; the growth function's purpose (bounding total memory) is served here
//!   by discarding every level-2 tree immediately.
//! - The paper's TwoBig TT is realised as the shared 2-way set-associative table.
//! - Threshold discipline follows Nagai/df-pn (`1 + ε`, ε = 0.25) rather than the
//!   plain `+1` increments in the paper's NegaPDS pseudocode — a strictly faster,
//!   well-established refinement (Pawlewicz & Lew 2007).
//!
//! Verdict discipline, budgets, deadline, cancel, and PV reconstruction are shared
//! with df-pn; a PDS-PN WIN carries a machine-checkable PV.

use super::io::Position;
use super::{Ctl, DriverResult, ProverConfig, dfpn};

pub fn solve(pos: &Position, cfg: &ProverConfig, ctl: &Ctl) -> DriverResult {
    dfpn::solve_mode(pos, cfg, ctl, true)
}
