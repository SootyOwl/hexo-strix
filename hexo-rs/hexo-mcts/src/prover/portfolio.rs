//! Portfolio racing: run several drivers concurrently on OS threads and take the
//! first *definitive* verdict (WIN or NO). Each driver is single-threaded and
//! deterministic on its own; the race shares one [`Ctl`] whose cancel flag the
//! winner raises, so the losers stop at their next budget check. Which driver wins
//! is timing-dependent (documented and accepted); the reported verdict is not.

use super::io::{Line, Position, RaceEntry, Report};
use super::{Ctl, DriverKind, ProverConfig, run_one};
use std::time::Instant;

pub fn race(pos: &Position, lines: &[Line], cfg: &ProverConfig) -> Report {
    // Assemble the driver set: the configured unguided racers, plus the hybrid
    // only when composer lines were supplied (it needs them).
    let mut drivers: Vec<DriverKind> = cfg
        .race_set
        .iter()
        .copied()
        .filter(|d| *d != DriverKind::Race)
        .collect();
    if drivers.is_empty() {
        drivers = vec![DriverKind::Idtt, DriverKind::Dfpn, DriverKind::Pdspn];
    }
    if !lines.is_empty() && !drivers.contains(&DriverKind::Hybrid) {
        drivers.push(DriverKind::Hybrid);
    }

    let ctl = Ctl::new(cfg.time_limit_s);
    let start = Instant::now();

    // Scoped threads borrow the shared inputs without cloning them.
    let results: Vec<(DriverKind, super::DriverResult, f64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = drivers
            .iter()
            .map(|&driver| {
                let ctl = ctl.clone();
                scope.spawn(move || {
                    let t = Instant::now();
                    let res = run_one(driver, pos, lines, cfg, &ctl);
                    // A definitive result cancels the rest of the field.
                    if res.verdict.is_definitive() {
                        ctl.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    (driver, res, t.elapsed().as_secs_f64())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("prover driver thread panicked")).collect()
    });
    let _ = start;

    let race: Vec<RaceEntry> = results
        .iter()
        .map(|(d, r, e)| RaceEntry {
            driver: d.name().to_string(),
            verdict: r.verdict,
            elapsed_s: *e,
            nodes: r.stats.nodes,
        })
        .collect();

    // Winner = the definitive result that finished first; failing that, the best
    // non-definitive by priority (WIN not expected here, then UNVERIFIED, then
    // BUDGET_EXCEEDED). WIN/NO are the only definitive verdicts.
    let winner = results
        .iter()
        .filter(|(_, r, _)| r.verdict.is_definitive())
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .or_else(|| {
            results.iter().min_by_key(|(_, r, _)| verdict_rank(r.verdict))
        });

    match winner {
        Some((driver, res, _)) => Report {
            verdict: res.verdict,
            depth: res.depth,
            pv: res.pv.clone(),
            driver: driver.name().to_string(),
            width: cfg.width_str().to_string(),
            stats: res.stats,
            race,
            unverified: res.unverified.clone(),
        },
        None => Report {
            verdict: super::io::Verdict::BudgetExceeded,
            depth: None,
            pv: Vec::new(),
            driver: "race".to_string(),
            width: cfg.width_str().to_string(),
            stats: super::io::Stats::default(),
            race,
            unverified: Vec::new(),
        },
    }
}

/// Priority for picking a non-definitive fallback winner (lower = preferred).
fn verdict_rank(v: super::io::Verdict) -> u8 {
    use super::io::Verdict::*;
    match v {
        Win => 0,
        No => 1,
        Unverified => 2,
        BudgetExceeded => 3,
    }
}
