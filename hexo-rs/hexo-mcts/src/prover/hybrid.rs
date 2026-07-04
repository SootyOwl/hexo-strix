//! Line-guided hybrid verifier — a Rust port of the validated 2026-07-04 Python
//! driver (v3), generalized to multiple composer lines.
//!
//! The insight: deep community compositions are unprovable by unguided search
//! within any affordable budget, but the composer *tells* us the attacker's
//! mainline. The hybrid uses those P1 turns as spine candidates and only has to
//! (a) enumerate the exact defender min-cover replies at each attacker turn and
//! (b) discharge off-line branches with a bounded kernel leaf solve. Transposition
//! merging (a shared proven-position set) collapses the rest.
//!
//! At each attacker (P1) node:
//!   a. proven-set (Zobrist) hit → proven;
//!   b. leaf solve with the kernel solver at `--leaf-budget` → proven;
//!   c. else try each spine candidate: apply it, and via the shared kernel
//!      classification require the move to be forcing (`B ≥ 2`) with no immediate
//!      defender win — `B ≥ 3` is proven at once, `B == 2` recurses into *every*
//!      min-cover reply, and all replies proven ⇒ node proven;
//!   d. else escalate the leaf solve to `--leaf-budget-max`;
//!   e. else record an `unverified` branch and keep going (the whole tree is
//!      verified; every failure is reported, not just the first).
//!
//! Verdict: WIN if the root is proven, else UNVERIFIED with the open-branch list.
//! The hybrid never emits NO — it cannot prove the absence of a forced win.

use super::io::{Line, Position, Stats, UnverifiedBranch, Verdict};
use super::kernel::{AndEval, KernelCtx};
use super::{Ctl, DriverResult, ProverConfig};
use crate::mcts::forcing::{CellSet2, Outcome, solve_limited};
use hexo_engine::types::{Coord, Player};
use rustc_hash::FxHashSet;
use std::time::Instant;

/// Group a line's post-position suffix into the attacker's successive turns.
///
/// The position's stone multiset must be a prefix of the line's move log (the same
/// check the Python driver used, order-insensitive since the board is a set). The
/// suffix is then split into turns by consecutive same-player runs (HeXO plays two
/// placements per turn; the first attacker turn may be a single placement when the
/// position is mid-turn), and the attacker's runs are returned in order.
fn attacker_turns_of_line(line: &Line, pos: &Position) -> Option<Vec<CellSet2>> {
    let n = pos.stones.len();
    if line.moves.len() < n {
        return None;
    }
    // Prefix check: the first `n` line moves are exactly the position's stones
    // (order-insensitive — the board is a set). `Player` isn't `Ord`, so key it.
    let key = |&(c, p): &(Coord, Player)| (c, matches!(p, Player::P2) as u8);
    let mut a: Vec<(Coord, u8)> = line.moves[..n].iter().map(key).collect();
    let mut b: Vec<(Coord, u8)> = pos.stones.iter().map(key).collect();
    a.sort_unstable();
    b.sort_unstable();
    if a != b {
        return None;
    }
    let suffix = &line.moves[n..];
    let mut turns: Vec<CellSet2> = Vec::new();
    let mut i = 0;
    while i < suffix.len() {
        let (_, player) = suffix[i];
        // Consume the consecutive same-player run (one turn = up to 2 placements).
        let mut cells: Vec<Coord> = Vec::new();
        while i < suffix.len() && suffix[i].1 == player && cells.len() < 2 {
            cells.push(suffix[i].0);
            i += 1;
        }
        if player == pos.attacker {
            turns.push(CellSet2::from_cells(&cells));
        }
    }
    Some(turns)
}

struct Hybrid<'a> {
    k: KernelCtx,
    /// spine[i] = deduped candidate attacker turns at attacker-depth i (across all
    /// lines).
    spine: Vec<Vec<CellSet2>>,
    proven: FxHashSet<u64>,
    unverified: Vec<UnverifiedBranch>,
    path: Vec<Coord>,
    ctl: &'a Ctl,
    depth_cap: u8,
    leaf_budget: u64,
    leaf_budget_max: u64,
    max_moves: u32,
    wide: bool,
    /// Total node visits (calls to `verify`).
    visits: u64,
    /// Total bounded leaf-solve CALLS (both budget tiers) — a cost measure.
    leaf_calls: u64,
    /// Nodes CLOSED by a kernel leaf solve (comparable to the Python baseline's
    /// "solver-proven leaf branches").
    leaf_proven: u64,
    /// Nodes CLOSED by following a spine candidate (comparable to the baseline's
    /// "line-steps").
    line_proven: u64,
}

impl<'a> Hybrid<'a> {
    /// A bounded kernel leaf solve of the current attacker-to-move position.
    fn leaf_solve(&mut self, placements: u8, budget: u64) -> bool {
        self.leaf_calls += 1;
        let game = self.k.game_state(placements, self.max_moves);
        matches!(
            solve_limited(&game, self.depth_cap, budget, self.wide, self.ctl.forcing_limits()),
            Outcome::Win(_)
        )
    }

    /// Verify that the attacker forces a win from the current (attacker-to-move)
    /// position. `depth` is the attacker-turn index into the spine. Returns whether
    /// the node is proven; records open branches under `self.unverified`.
    fn verify(&mut self, depth: usize, placements: u8) -> bool {
        self.visits += 1;
        if self.ctl.expired() {
            // Out of time: report this node as unverified rather than fabricate.
            self.unverified.push(UnverifiedBranch { path: self.path.clone(), note: "deadline".into() });
            return false;
        }
        let h = self.k.hash();
        if self.proven.contains(&h) {
            return true;
        }
        let uv_start = self.unverified.len();
        // (b) bounded leaf solve.
        if self.leaf_solve(placements, self.leaf_budget) {
            self.leaf_proven += 1;
            self.proven.insert(h);
            return true;
        }
        // (c) spine-guided candidates.
        if let Some(cands) = self.spine.get(depth) {
            let cands = cands.clone();
            for cand in &cands {
                if !self.k.cells_empty(cand) {
                    continue; // candidate cell already occupied — not playable here
                }
                self.k.place_attacker(cand);
                for &c in cand.cells() {
                    self.path.push(c);
                }
                let proved = match self.k.and_eval() {
                    AndEval::AttackerWin => true, // B >= 3: unstoppable
                    AndEval::Loss => false,       // not forcing / defender wins first
                    AndEval::Covers(covers) => {
                        // Recurse into EVERY min-cover reply; all must be proven.
                        // Keep going after a failure so all open branches surface.
                        let mut all = true;
                        for cov in &covers {
                            self.k.place_defender(cov);
                            for &c in cov.cells() {
                                self.path.push(c);
                            }
                            let sub = self.verify(depth + 1, 2);
                            for _ in cov.cells() {
                                self.path.pop();
                            }
                            self.k.unplace(cov);
                            if !sub {
                                all = false;
                            }
                        }
                        all
                    }
                };
                for _ in cand.cells() {
                    self.path.pop();
                }
                self.k.unplace(cand);
                if proved {
                    // Discard any deeper branches recorded by earlier failed
                    // candidate attempts for this same node — a sibling candidate
                    // closed it, so those failures were spurious.
                    self.unverified.truncate(uv_start);
                    self.line_proven += 1;
                    self.proven.insert(h);
                    return true;
                }
            }
        }
        // (d) escalated leaf solve.
        if self.leaf_budget_max > self.leaf_budget && self.leaf_solve(placements, self.leaf_budget_max) {
            self.unverified.truncate(uv_start);
            self.leaf_proven += 1;
            self.proven.insert(h);
            return true;
        }
        // (e) unproven. If no deeper branch pinpointed the failure (e.g. no spine
        // candidate was even applicable here), record this node itself.
        if self.unverified.len() == uv_start {
            self.unverified.push(UnverifiedBranch {
                path: self.path.clone(),
                note: "off-line-unproven".into(),
            });
        }
        false
    }
}

pub fn verify(pos: &Position, lines: &[Line], cfg: &ProverConfig, ctl: &Ctl) -> DriverResult {
    let wl = pos.config.win_length;
    if !KernelCtx::supported(wl) {
        return DriverResult::new(Verdict::Unverified);
    }
    // Build the spine from every line whose prefix matches the position.
    let mut per_depth: Vec<Vec<CellSet2>> = Vec::new();
    let mut main_line_pv: Vec<Coord> = Vec::new();
    for line in lines {
        if let Some(turns) = attacker_turns_of_line(line, pos) {
            if main_line_pv.is_empty() {
                // Best-effort PV: the input mainline's suffix (attacker + defender).
                let n = pos.stones.len();
                main_line_pv = line.moves[n..].iter().map(|(c, _)| *c).collect();
            }
            for (i, t) in turns.into_iter().enumerate() {
                if per_depth.len() <= i {
                    per_depth.push(Vec::new());
                }
                if !per_depth[i].contains(&t) {
                    per_depth[i].push(t);
                }
            }
        }
    }

    let ctx = match KernelCtx::new_wide(&pos.stones, pos.attacker, wl, pos.config.placement_radius, cfg.wide) {
        Some(c) => c,
        None => return DriverResult::new(Verdict::Unverified),
    };
    let mut h = Hybrid {
        k: ctx,
        spine: per_depth,
        proven: FxHashSet::default(),
        unverified: Vec::new(),
        path: Vec::new(),
        ctl,
        depth_cap: cfg.depth_cap,
        leaf_budget: cfg.leaf_budget,
        leaf_budget_max: cfg.leaf_budget_max,
        max_moves: pos.config.max_moves,
        wide: cfg.wide,
        visits: 0,
        leaf_calls: 0,
        leaf_proven: 0,
        line_proven: 0,
    };

    let t = Instant::now();
    let proven = h.verify(0, pos.placements_remaining);
    let elapsed = t.elapsed().as_secs_f64();

    let mut res = DriverResult::new(if proven { Verdict::Win } else { Verdict::Unverified });
    if proven {
        res.pv = main_line_pv;
    } else {
        res.unverified = h.unverified.clone();
    }
    // Report proven-by-category counts (comparable to the Python baseline's
    // "solver-proven leaf branches" / "line-steps"); `nodes` = total node visits,
    // `tt_hits` = raw leaf-solve CALL count. See the `Stats` field docs in io.rs
    // for the hybrid-specific meanings.
    res.stats = Stats {
        nodes: h.visits,
        tt_hits: h.leaf_calls,
        tt_bytes: 0,
        elapsed_s: elapsed,
        leaf_solves: h.leaf_proven,
        line_steps: h.line_proven,
    };
    res
}
