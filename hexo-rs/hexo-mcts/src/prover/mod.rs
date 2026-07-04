//! HeXO deep prover — a standalone research CLI running several proof-search
//! drivers over the validated VCF kernel (`crate::mcts::forcing`).
//!
//! Drivers: `idtt` (the production iterative-deepening solver, baseline), `dfpn`
//! (Nagai depth-first proof-number search), `pdspn` (Winands et al. two-level
//! PDS-PN), and `hybrid` (line-guided verification). A `race` portfolio runs
//! several drivers on OS threads and takes the first definitive verdict.
//!
//! All drivers share one rule set through [`kernel::KernelCtx`]; see that module.
//! This module owns the shared config/verdict types, the `idtt` wrapper, and the
//! top-level dispatch. No PyO3 surface, no production-path changes.

pub mod dfpn;
#[cfg(test)]
mod fixtures;
pub mod hybrid;
pub mod io;
pub(crate) mod kernel;
pub mod pdspn;
pub mod pn;
pub mod portfolio;

use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::types::Coord;
use io::{Line, Position, Report, Stats, UnverifiedBranch, Verdict};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Which proof-search driver to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DriverKind {
    Idtt,
    Dfpn,
    Pdspn,
    Hybrid,
    Race,
}

impl DriverKind {
    pub fn name(self) -> &'static str {
        match self {
            DriverKind::Idtt => "idtt",
            DriverKind::Dfpn => "dfpn",
            DriverKind::Pdspn => "pdspn",
            DriverKind::Hybrid => "hybrid",
            DriverKind::Race => "race",
        }
    }
    pub fn parse(s: &str) -> Option<DriverKind> {
        Some(match s {
            "idtt" => DriverKind::Idtt,
            "dfpn" => DriverKind::Dfpn,
            "pdspn" => DriverKind::Pdspn,
            "hybrid" => DriverKind::Hybrid,
            "race" => DriverKind::Race,
            _ => return None,
        })
    }
}

/// Search-strategy configuration shared by every driver.
#[derive(Clone)]
pub struct ProverConfig {
    pub driver: DriverKind,
    /// Attacker-turn generator width. `true` enables the experimental wide-partner
    /// knob (a strict superset of the tight generator; off by default).
    pub wide: bool,
    pub depth_cap: u8,
    pub node_budget: u64,
    pub tt_mb: usize,
    pub pn2_nodes: u64,
    pub leaf_budget: u64,
    pub leaf_budget_max: u64,
    /// Wall-clock limit in seconds; 0 disables (no deadline).
    pub time_limit_s: f64,
    pub race_set: Vec<DriverKind>,
}

impl Default for ProverConfig {
    fn default() -> Self {
        ProverConfig {
            driver: DriverKind::Dfpn,
            wide: false,
            depth_cap: 40,
            node_budget: 20_000_000,
            tt_mb: 512,
            pn2_nodes: 50_000,
            leaf_budget: 1_000_000,
            leaf_budget_max: 10_000_000,
            time_limit_s: 1800.0,
            race_set: vec![DriverKind::Idtt, DriverKind::Dfpn, DriverKind::Pdspn],
        }
    }
}

impl ProverConfig {
    /// The width string for reports.
    pub fn width_str(&self) -> &'static str {
        if self.wide { "wide" } else { "tight" }
    }
}

/// Cooperative search control: a wall-clock deadline plus a cancel flag a racing
/// thread can raise. Every driver checks [`Ctl::expired`] at its budget cadence.
#[derive(Clone)]
pub struct Ctl {
    pub deadline: Option<Instant>,
    pub cancel: Arc<AtomicBool>,
}

impl Ctl {
    pub fn new(time_limit_s: f64) -> Ctl {
        let deadline = if time_limit_s > 0.0 {
            Some(Instant::now() + std::time::Duration::from_secs_f64(time_limit_s))
        } else {
            None
        };
        Ctl { deadline, cancel: Arc::new(AtomicBool::new(false)) }
    }

    /// True once the deadline has passed or another thread cancelled the search.
    #[inline]
    pub fn expired(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
            || self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// The `forcing::Limits` view of this control, for the `idtt` driver.
    pub(crate) fn forcing_limits(&self) -> crate::mcts::forcing::Limits {
        crate::mcts::forcing::Limits {
            deadline: self.deadline,
            cancel: Some(Arc::clone(&self.cancel)),
        }
    }
}

/// A single driver's outcome, assembled into a [`Report`] by [`run`].
#[derive(Clone, Debug)]
pub struct DriverResult {
    pub verdict: Verdict,
    pub depth: Option<u8>,
    pub pv: Vec<Coord>,
    pub stats: Stats,
    pub unverified: Vec<UnverifiedBranch>,
}

impl DriverResult {
    pub fn new(verdict: Verdict) -> DriverResult {
        DriverResult { verdict, depth: None, pv: Vec::new(), stats: Stats::default(), unverified: Vec::new() }
    }
}

impl Position {
    /// Build the engine `GameState` for this position (used by the `idtt` driver,
    /// which calls the production solver directly).
    pub fn to_game(&self) -> GameState {
        let cfg = GameConfig {
            win_length: self.config.win_length,
            placement_radius: self.config.placement_radius,
            max_moves: self.config.max_moves,
        };
        GameState::from_state(&self.stones, self.attacker, self.placements_remaining, cfg)
    }
}

/// The `idtt` baseline: the production iterative-deepening + TT solver, with an
/// added wall-clock deadline and race-cancel hook (both no-ops when unset). It
/// proves the *shortest* win (its depth is authoritative); `nodes` is not exposed
/// by the production solver, so it stays 0 in the stats.
pub fn idtt(pos: &Position, cfg: &ProverConfig, ctl: &Ctl) -> DriverResult {
    use crate::mcts::forcing::{Outcome, solve_limited};
    let game = pos.to_game();
    let t = Instant::now();
    let outcome = solve_limited(&game, cfg.depth_cap, cfg.node_budget, cfg.wide, ctl.forcing_limits());
    let elapsed = t.elapsed().as_secs_f64();
    let mut r = match outcome {
        Outcome::Win(w) => {
            let mut r = DriverResult::new(Verdict::Win);
            r.depth = Some(w.depth);
            r.pv = w.pv;
            r
        }
        Outcome::No => DriverResult::new(Verdict::No),
        Outcome::BudgetExceeded => DriverResult::new(Verdict::BudgetExceeded),
    };
    r.stats.elapsed_s = elapsed;
    r
}

/// Run a single (non-race) driver.
fn run_one(driver: DriverKind, pos: &Position, lines: &[Line], cfg: &ProverConfig, ctl: &Ctl) -> DriverResult {
    match driver {
        DriverKind::Idtt => idtt(pos, cfg, ctl),
        DriverKind::Dfpn => dfpn::solve(pos, cfg, ctl),
        DriverKind::Pdspn => pdspn::solve(pos, cfg, ctl),
        DriverKind::Hybrid => hybrid::verify(pos, lines, cfg, ctl),
        DriverKind::Race => unreachable!("race is dispatched by run(), not run_one()"),
    }
}

/// Top-level entry: dispatch the configured driver (or the portfolio race) and
/// assemble the JSON report.
pub fn run(pos: &Position, lines: &[Line], cfg: &ProverConfig) -> Report {
    // Default driver selection: hybrid if lines were provided, else dfpn — but the
    // caller (CLI) sets `cfg.driver` explicitly; this is only a safety net.
    let driver = cfg.driver;
    if driver == DriverKind::Race {
        return portfolio::race(pos, lines, cfg);
    }
    let ctl = Ctl::new(cfg.time_limit_s);
    let res = run_one(driver, pos, lines, cfg, &ctl);
    Report {
        verdict: res.verdict,
        depth: res.depth,
        pv: res.pv,
        driver: driver.name().to_string(),
        width: cfg.width_str().to_string(),
        stats: res.stats,
        race: Vec::new(),
        unverified: res.unverified,
    }
}
