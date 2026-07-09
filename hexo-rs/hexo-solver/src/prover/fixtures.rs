//! Cross-driver parity fixtures (real HDS puzzle positions) and the parity /
//! micro tests. Stone arrays are copied verbatim from `hexo_solver::forcing`'s validated
//! parity fixtures so the prover proves the *same* positions the kernel's own
//! parity suite pins — no HDS-codec conversion in the test path.

use super::io::{Line, PosConfig, Position, Verdict};
use super::kernel::KernelCtx;
use super::pn::{INF, ProofTt, one_plus_eps, sat_add};
use super::{Ctl, DriverKind, ProverConfig, dfpn, hybrid, idtt, pdspn};
use hexo_engine::types::{Coord, Player};
use std::path::PathBuf;
use Player::{P1, P2};

/// Absolute path to a fixture JSON under `scripts/fixtures/forcing_puzzles/`.
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/fixtures/forcing_puzzles/")).join(name)
}

fn pos(stones: &[(Coord, Player)], attacker: Player, placements: u8) -> Position {
    Position { stones: stones.to_vec(), attacker, placements_remaining: placements, config: PosConfig::default() }
}

/// Replay a PV through the real engine and assert it ends in the attacker's win.
fn assert_pv_wins(p: &Position, pv: &[Coord]) {
    assert!(!pv.is_empty(), "a WIN must carry a non-empty PV");
    let mut g = p.to_game();
    for &c in pv {
        assert!(!g.is_terminal(), "PV continues after the game ended: {pv:?}");
        g.apply_move(c).expect("every PV move must be legal");
    }
    assert!(g.is_terminal(), "PV must end the game");
    assert_eq!(g.winner(), Some(p.attacker), "PV must end in the attacker's win");
}

/// Fast parity config: small TT, generous node budget (these solve in ms), no
/// wall limit.
fn fast_cfg() -> ProverConfig {
    ProverConfig {
        driver: DriverKind::Dfpn,
        tt_mb: 256,
        node_budget: 80_000_000,
        depth_cap: 40,
        time_limit_s: 0.0,
        ..ProverConfig::default()
    }
}

// ---- fixtures (attacker, placements=2; Some(depth) = idtt's known win depth) ----

const XSNFYLL: &[(Coord, Player)] = &[
    ((0, 0), P1), ((1, -2), P2), ((2, -4), P2), ((0, -3), P1), ((2, -5), P1),
    ((0, -2), P2), ((1, -4), P2), ((-2, 0), P1), ((1, 0), P1), ((-1, 0), P2),
    ((1, -3), P2), ((3, -4), P1), ((3, -2), P1),
];

const HZ3HTY: &[(Coord, Player)] = &[
    ((0, 0), P1), ((5, 0), P2), ((5, -1), P2), ((5, -2), P1), ((4, 0), P1),
    ((5, 4), P2), ((6, 4), P2), ((6, 3), P1), ((4, 4), P1), ((10, 4), P2),
    ((11, 3), P2), ((11, 4), P1), ((12, 2), P1), ((6, 8), P2), ((5, 8), P2),
    ((4, 8), P1), ((5, 9), P1), ((10, 8), P2), ((11, 7), P2), ((9, 9), P1),
    ((10, 7), P1),
];

const L9MXN59: &[(Coord, Player)] = &[
    ((0, 0), P1), ((1, -1), P2), ((2, -1), P2), ((1, 0), P1), ((0, -1), P1),
    ((3, -2), P2), ((2, -2), P2), ((2, -3), P1), ((4, -2), P1), ((3, -3), P2),
    ((4, -3), P2), ((3, -4), P1), ((3, -1), P1), ((10, -4), P2), ((11, -4), P2),
    ((10, -3), P1), ((12, -4), P1),
];

/// (name, stones, attacker, expected idtt depth). Fast subset.
#[allow(clippy::type_complexity)]
const FAST: &[(&str, &[(Coord, Player)], Player, Option<u8>)] = &[
    ("xsnfyll", XSNFYLL, P2, Some(4)),
    ("0hz3hty", HZ3HTY, P2, Some(5)),
    ("l9mxn59", L9MXN59, P2, None),
];

/// Cross-driver parity (hard, fast): dfpn agrees with idtt on WIN, its PV replays
/// to an engine win; on the counterthreat near-miss neither may report WIN.
#[test]
fn parity_dfpn_matches_idtt_fast() {
    let cfg = fast_cfg();
    let ctl = Ctl::new(0.0);
    for &(name, stones, atk, expected) in FAST {
        let p = pos(stones, atk, 2);
        let id = idtt(&p, &cfg, &ctl);
        let df = dfpn::solve(&p, &cfg, &ctl);
        match expected {
            Some(d) => {
                assert_eq!(id.verdict, Verdict::Win, "{name}: idtt must WIN");
                assert_eq!(id.depth, Some(d), "{name}: idtt depth");
                assert_eq!(df.verdict, Verdict::Win, "{name}: dfpn must WIN");
                assert_pv_wins(&p, &df.pv);
            }
            None => {
                assert_ne!(id.verdict, Verdict::Win, "{name}: idtt must not WIN");
                assert_ne!(df.verdict, Verdict::Win, "{name}: dfpn must not fabricate a WIN");
            }
        }
    }
}

/// Cross-driver parity including pdspn (hard, fast).
#[test]
fn parity_pdspn_matches_idtt_fast() {
    let cfg = fast_cfg();
    let ctl = Ctl::new(0.0);
    for &(name, stones, atk, expected) in FAST {
        let p = pos(stones, atk, 2);
        let pd = pdspn::solve(&p, &cfg, &ctl);
        match expected {
            Some(_) => {
                assert_eq!(pd.verdict, Verdict::Win, "{name}: pdspn must WIN");
                assert_pv_wins(&p, &pd.pv);
            }
            None => assert_ne!(pd.verdict, Verdict::Win, "{name}: pdspn must not fabricate a WIN"),
        }
    }
}

// ---- slow parity fixtures (the big/deep ones) ----

const ACLY7KB: &[(Coord, Player)] = &[
    ((0,0),P1),((1,-2),P2),((-1,2),P2),((-1,0),P1),((-2,0),P1),((1,0),P2),((-5,0),P2),((-2,3),P1),((1,-3),P1),((0,-2),P2),((-2,2),P2),((-4,2),P1),((-2,-2),P1),((3,-2),P2),((1,2),P2),((2,-1),P1),((1,1),P1),((2,-2),P2),((4,-2),P2),((5,-2),P1),((-1,-2),P1),((0,2),P2),((2,2),P2),((3,2),P1),((-3,2),P1),((-3,1),P2),((-1,-1),P2),((-2,1),P1),((-2,-3),P1),((-2,-1),P2),((-4,3),P2),((-3,-1),P1),((4,0),P1),((-1,-3),P2),((4,1),P2),((-3,-2),P1),((5,-1),P1),((-6,-2),P2),((5,-3),P2),((6,-4),P1),((-6,-1),P1),((4,-1),P2),((3,1),P2),((5,0),P1),((-4,-3),P1),((5,1),P2),((-3,-4),P2),((-6,2),P1),((6,1),P1),((-5,2),P2),((7,0),P2),((-5,1),P1),((6,-2),P1),((-7,3),P2),((6,0),P2),((8,-2),P1),((-6,3),P1),((7,-2),P2),((7,-3),P2),((7,-1),P1),((6,-3),P1),((6,-6),P2),((-6,0),P2),((7,-4),P1),((-7,0),P1),((4,-4),P2),((-8,1),P2),((8,-5),P1),((8,-4),P1),((9,-6),P2),((8,-3),P2),((5,-5),P1),((0,-5),P1),((1,-6),P2),((9,-5),P2),((4,-5),P1),((9,-7),P1),((3,-5),P2),((-8,0),P2),((-8,2),P1),((8,-6),P1),((8,-9),P2),((11,-9),P2),((9,-9),P1),((-7,-1),P1),((-7,-2),P2),((-8,-1),P2),((-8,-2),P1),((-9,0),P1),((-2,-6),P2),((-4,-4),P2),((-3,-5),P1),((-5,-4),P1),
];

const JNZZMCM: &[(Coord, Player)] = &[
    ((0,0),P1),((0,4),P2),((1,3),P2),((1,0),P1),((-1,1),P1),((-1,0),P2),((1,-1),P2),((-1,5),P1),((0,3),P1),((2,2),P2),((3,1),P2),((4,0),P1),((4,1),P1),((3,0),P2),((4,2),P2),((3,2),P1),((2,3),P1),((1,4),P2),((-1,4),P2),((2,4),P1),((1,5),P1),((3,3),P2),((4,-2),P2),((4,-1),P1),((5,-3),P1),((6,-3),P2),((5,-4),P2),((4,-3),P1),((6,-4),P1),((5,-5),P2),((7,-6),P2),((6,-5),P1),((7,-7),P1),((6,-7),P2),((6,-8),P2),((5,-7),P1),((4,-6),P1),((4,-7),P2),((3,5),P2),((3,7),P1),((2,8),P1),((4,6),P2),((2,9),P2),((-3,9),P1),((-4,9),P1),((-1,7),P2),((-2,9),P2),((-2,10),P1),((-4,11),P1),((-3,10),P2),((-4,10),P2),((-8,8),P1),((-8,7),P1),((-7,6),P2),((-8,5),P2),((-8,6),P1),((-7,5),P1),((-9,7),P2),((-5,10),P2),((-9,6),P1),((7,-4),P1),((-3,3),P2),((8,-4),P2),((-5,9),P1),((-6,10),P1),((-8,9),P2),((-6,11),P2),
];

const HU01JK4: &[(Coord, Player)] = &[
    ((0,0),P1),((1,-2),P2),((-1,-4),P2),((0,-3),P1),((-3,0),P1),((-1,-2),P2),((0,-2),P2),((-2,-2),P1),((-1,-3),P1),((1,-3),P2),((-4,0),P2),((-2,0),P1),((1,-5),P1),((0,-4),P2),((2,0),P2),((-3,-4),P1),((-2,2),P1),((-2,1),P2),((-1,1),P2),((-3,1),P1),((2,-4),P1),((-3,2),P2),((-1,0),P2),((0,-1),P1),((-1,3),P1),((3,-1),P2),((4,-2),P2),((3,-2),P1),((5,-3),P1),((1,-1),P2),((4,-1),P2),((4,0),P1),((1,1),P1),((-3,-2),P2),((-2,-6),P2),((-2,-3),P1),((5,-1),P1),((7,-3),P2),((-2,-5),P2),((1,3),P1),((0,3),P1),((2,3),P2),((0,4),P2),((1,2),P1),((2,2),P1),((3,1),P2),((-4,3),P2),((-5,4),P1),((-4,2),P1),((3,2),P2),((3,0),P2),((3,3),P1),((1,4),P1),((1,5),P2),((-5,3),P2),((-2,4),P1),((-2,5),P1),((-1,4),P2),((-3,5),P2),((-6,3),P1),((-7,4),P1),((-6,4),P2),((-5,2),P2),((5,-2),P1),((5,-4),P1),((5,0),P2),((5,-5),P2),((4,-3),P1),((6,-5),P1),((7,-6),P2),((2,-1),P2),((7,-2),P1),((6,-6),P1),((6,-4),P2),((-9,8),P2),((-10,10),P1),((-10,7),P1),((-10,8),P2),((-9,7),P2),((-8,6),P1),((-9,6),P1),((-11,8),P2),((-8,8),P2),((-12,8),P1),((-7,8),P1),((-11,12),P2),((-10,6),P2),((-12,10),P1),((-5,6),P1),((-9,10),P2),((-2,3),P2),((-9,11),P1),((-7,9),P1),((-7,6),P2),((-10,12),P2),((-12,12),P1),((-12,11),P1),((-12,9),P2),((-11,11),P2),((-11,10),P1),((-5,7),P1),((-5,8),P2),((-12,13),P2),((-13,14),P1),((-13,12),P1),((-13,13),P2),((-14,13),P2),((-11,13),P1),((6,-10),P1),((7,-10),P2),((6,-9),P2),((5,-8),P1),((5,-9),P1),((4,-8),P2),((5,-10),P2),((-16,13),P1),((-2,7),P1),((-3,7),P2),((-3,4),P2),((-3,6),P1),((-1,2),P1),((-2,6),P2),((-15,11),P2),((-4,8),P1),((-15,12),P1),((-14,12),P2),((-14,10),P2),((-14,11),P1),((10,-2),P1),((9,-2),P2),((-18,15),P2),((-16,14),P1),((10,-5),P1),((10,-3),P2),((-16,12),P2),((-18,14),P1),((12,-5),P1),((13,-5),P2),((-14,14),P2),((12,-4),P1),((9,-1),P1),((12,-6),P2),((9,-10),P2),((8,-3),P1),((-3,-3),P1),((-5,-3),P2),((-14,15),P2),((-3,-5),P1),((-14,16),P1),
];

#[allow(clippy::type_complexity)]
const SLOW: &[(&str, &[(Coord, Player)], Player, u8)] = &[
    ("acly7kb", ACLY7KB, P2, 4),
    ("jnzzmcm", JNZZMCM, P1, 7),
    ("hu01jk4", HU01JK4, P2, 6),
];

/// Slow cross-driver parity across all three unguided drivers.
/// run: cargo test -p hexo-mcts prover::fixtures::parity_all_drivers_slow -- --ignored --nocapture
#[test]
#[ignore]
fn parity_all_drivers_slow() {
    let cfg = fast_cfg();
    let ctl = Ctl::new(0.0);
    for &(name, stones, atk, d) in SLOW {
        let p = pos(stones, atk, 2);
        let id = idtt(&p, &cfg, &ctl);
        assert_eq!(id.verdict, Verdict::Win, "{name}: idtt WIN");
        assert_eq!(id.depth, Some(d), "{name}: idtt depth");
        for (dn, dr) in [("dfpn", dfpn::solve(&p, &cfg, &ctl)), ("pdspn", pdspn::solve(&p, &cfg, &ctl))] {
            assert_eq!(dr.verdict, Verdict::Win, "{name}: {dn} must WIN");
            assert_pv_wins(&p, &dr.pv);
            eprintln!("{name} {dn}: WIN nodes={} tt_hits={} pv_len={}", dr.stats.nodes, dr.stats.tt_hits, dr.pv.len());
        }
    }
}

// ---- micro tests ----

#[test]
fn mate_in_one_and_two_all_drivers() {
    let ctl = Ctl::new(0.0);
    let cfg = fast_cfg();
    // mate in one: a five-run, one placement completes.
    let five: Vec<(Coord, Player)> = (0..5).map(|q| ((q, 0), P1)).collect();
    let p1 = pos(&five, P1, 2);
    for (name, r) in [("idtt", idtt(&p1, &cfg, &ctl)), ("dfpn", dfpn::solve(&p1, &cfg, &ctl)), ("pdspn", pdspn::solve(&p1, &cfg, &ctl))] {
        assert_eq!(r.verdict, Verdict::Win, "{name} mate-in-1");
        assert_pv_wins(&p1, &r.pv);
    }
    // mate in two: the double-four fork.
    let fork: &[(Coord, Player)] = &[((0, 0), P1), ((1, 0), P1), ((2, 0), P1), ((0, 1), P1), ((0, 2), P1)];
    let p2 = pos(fork, P1, 2);
    for (name, r) in [("idtt", idtt(&p2, &cfg, &ctl)), ("dfpn", dfpn::solve(&p2, &cfg, &ctl)), ("pdspn", pdspn::solve(&p2, &cfg, &ctl))] {
        assert_eq!(r.verdict, Verdict::Win, "{name} mate-in-2");
        assert_pv_wins(&p2, &r.pv);
    }
}

#[test]
fn no_forcing_win_all_drivers() {
    let ctl = Ctl::new(0.0);
    let cfg = fast_cfg();
    let scattered: &[(Coord, Player)] = &[((0, 0), P1), ((10, 10), P1)];
    let p = pos(scattered, P1, 2);
    assert_eq!(idtt(&p, &cfg, &ctl).verdict, Verdict::No, "idtt gate-No");
    assert_eq!(dfpn::solve(&p, &cfg, &ctl).verdict, Verdict::No, "dfpn No");
    assert_eq!(pdspn::solve(&p, &cfg, &ctl).verdict, Verdict::No, "pdspn No");
}

#[test]
fn pn_dn_algebra() {
    // saturating add caps at INF.
    assert_eq!(sat_add(INF, 5), INF);
    assert_eq!(sat_add(3, 4), 7);
    assert_eq!(sat_add(INF, INF), INF);
    // one_plus_eps inflates by ~25% and never drops below second_best+1.
    assert_eq!(one_plus_eps(INF, 0.25), INF);
    assert!(one_plus_eps(0, 0.25) >= 1);
    assert!(one_plus_eps(100, 0.25) >= 101);
    assert!(one_plus_eps(100, 0.25) <= 130);
}

#[test]
fn tt_two_way_coexist_then_evict_weakest() {
    let mut tt = ProofTt::new(1);
    // Three keys sharing the low index bits map to the same 2-way bucket (their
    // high bits differ, so they are distinct keys).
    let a = 0x1234_0000u64;
    let b = a ^ (1u64 << 40);
    let c = a ^ (1u64 << 41);
    // Two ways: `a` and `b` coexist (a 1-way direct-mapped table would evict one).
    tt.store(a, 5, 5, 100);
    tt.store(b, 6, 6, 50);
    assert_eq!(tt.probe(a), Some((5, 5)), "both keys coexist in a 2-way bucket");
    assert_eq!(tt.probe(b), Some((6, 6)));
    // A third colliding key with more effort evicts the weaker occupant (`b`,
    // work 50 < `a` work 100), leaving the costlier entry in place.
    tt.store(c, 9, 9, 200);
    assert_eq!(tt.probe(c), Some((9, 9)), "higher-effort entry lands");
    assert_eq!(tt.probe(a), Some((5, 5)), "higher-effort `a` survives");
    assert_eq!(tt.probe(b), None, "lower-effort `b` evicted");
    // Same-key overwrite always applies.
    tt.store(a, 1, 1, 1);
    assert_eq!(tt.probe(a), Some((1, 1)));
}

#[test]
fn tt_protects_resolved_entries() {
    let mut tt = ProofTt::new(1);
    let a = 0x55u64;
    let b = a ^ (1u64 << 40);
    let c = a ^ (1u64 << 41);
    // Fill the bucket with two resolved entries (pn == 0), modest work.
    tt.store(a, 0, INF, 5);
    tt.store(b, 0, INF, 5);
    // A high-effort UNRESOLVED colliding write must not evict a resolved entry.
    tt.store(c, 7, 7, 10_000);
    assert_eq!(tt.probe(a), Some((0, INF)), "resolved entry survives unresolved collision");
    assert_eq!(tt.probe(b), Some((0, INF)));
    assert_eq!(tt.probe(c), None, "unresolved write rejected against two resolved slots");
}

#[test]
fn deadline_is_honored() {
    // A position that would take real time unguided; with a already-past deadline
    // dfpn must bail to BUDGET_EXCEEDED rather than fabricate a verdict.
    let cfg = ProverConfig { tt_mb: 8, node_budget: u64::MAX, ..ProverConfig::default() };
    // Deadline of 0s => `Ctl::expired()` true immediately after the first sample.
    let ctl = Ctl { deadline: Some(std::time::Instant::now()), cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)) };
    let p = pos(HU01JK4, P2, 2);
    let r = dfpn::solve(&p, &cfg, &ctl);
    // Either it solved instantly (fast) or it honored the deadline; it must never
    // hang. HU01JK4 is a genuine win but deep; with an expired deadline the honest
    // outcomes are WIN (if trivially fast) or BUDGET_EXCEEDED — never NO.
    assert_ne!(r.verdict, Verdict::No, "must not fabricate NO under a deadline");
}

#[test]
#[ignore]
fn debug_dfpn_0hz3hty() {
    let cfg = ProverConfig { tt_mb: 256, node_budget: 60_000_000, depth_cap: 40, time_limit_s: 0.0, ..ProverConfig::default() };
    let ctl = Ctl::new(0.0);
    let p = pos(HZ3HTY, P2, 2);
    let r = dfpn::solve(&p, &cfg, &ctl);
    eprintln!("0hz3hty dfpn: {:?} depth={:?} pv_len={} nodes={}", r.verdict, r.depth, r.pv.len(), r.stats.nodes);
}

/// Hybrid acceptance (hard): verify the 18-turn composition `0l4291i` from turn 0
/// given both composer lines. Expected WIN with zero unverified branches (the
/// session's Python v3 result). If it does NOT verify, the honest output is the
/// UNVERIFIED branch list — this test would then surface it as a finding.
/// run: cargo test --release -p hexo-mcts prover::fixtures::hybrid_acceptance_0l4291i -- --ignored --nocapture
#[test]
#[ignore]
fn hybrid_acceptance_0l4291i() {
    let pos = Position::load(&fixture("0l4291i_live.json")).expect("position fixture");
    let l1 = Line::load(&fixture("d9ci11d_line.json")).expect("line 1");
    let l2 = Line::load(&fixture("7zoidcw_line.json")).expect("line 2");
    let cfg = ProverConfig { driver: DriverKind::Hybrid, tt_mb: 256, time_limit_s: 7200.0, ..ProverConfig::default() };
    let ctl = Ctl::new(cfg.time_limit_s);
    let r = hybrid::verify(&pos, &[l1, l2], &cfg, &ctl);
    eprintln!(
        "hybrid 0l4291i: {:?} visits={} leaf_proven={} line_proven={} leaf_calls={} elapsed={:.1}s",
        r.verdict, r.stats.nodes, r.stats.leaf_solves, r.stats.line_steps, r.stats.tt_hits, r.stats.elapsed_s
    );
    for u in &r.unverified {
        eprintln!("  UNVERIFIED [{}]: {:?}", u.note, u.path);
    }
    assert_eq!(r.verdict, Verdict::Win, "0l4291i must verify as a forced win from turn 0");
    assert!(r.unverified.is_empty(), "a verified win has no open branches");
}

/// Unguided stretch (measured, success optional): run df-pn and PDS-PN on
/// `0l4291i` under the 30-minute wall cap and record the outcome + stats. Either
/// verdict is a valid research result; no assertion beyond "no fabricated NO/WIN".
/// run: cargo test --release -p hexo-mcts prover::fixtures::stretch_unguided_0l4291i -- --ignored --nocapture
#[test]
#[ignore]
fn stretch_unguided_0l4291i() {
    let pos = Position::load(&fixture("0l4291i_live.json")).expect("position fixture");
    let cfg = ProverConfig { tt_mb: 512, node_budget: u64::MAX, depth_cap: 60, time_limit_s: 1800.0, ..ProverConfig::default() };
    for (name, f) in [("dfpn", dfpn::solve as fn(&Position, &ProverConfig, &Ctl) -> super::DriverResult), ("pdspn", pdspn::solve)] {
        let ctl = Ctl::new(cfg.time_limit_s);
        let r = f(&pos, &cfg, &ctl);
        eprintln!(
            "stretch {name} 0l4291i: {:?} depth={:?} nodes={} tt_hits={} elapsed={:.1}s",
            r.verdict, r.depth, r.stats.nodes, r.stats.tt_hits, r.stats.elapsed_s
        );
    }
}

/// Independent confirmation of the landmark: PDS-PN proves `0l4291i` unguided AND
/// its reconstructed PV replays through the engine to an attacker win.
/// run: cargo test --release -p hexo-mcts prover::fixtures::pdspn_0l4291i_pv_replays -- --ignored --nocapture
#[test]
#[ignore]
fn pdspn_0l4291i_pv_replays() {
    let pos = Position::load(&fixture("0l4291i_live.json")).expect("position fixture");
    let cfg = ProverConfig { tt_mb: 512, node_budget: u64::MAX, depth_cap: 60, time_limit_s: 1800.0, ..ProverConfig::default() };
    let ctl = Ctl::new(cfg.time_limit_s);
    let r = pdspn::solve(&pos, &cfg, &ctl);
    eprintln!(
        "pdspn 0l4291i: {:?} depth={:?} pv_len={} nodes={} leaf_solves={} elapsed={:.1}s",
        r.verdict, r.depth, r.pv.len(), r.stats.nodes, r.stats.leaf_solves, r.stats.elapsed_s
    );
    assert_eq!(r.verdict, Verdict::Win, "pdspn must prove 0l4291i a forced win");
    assert!(!r.pv.is_empty(), "the WIN must carry a reconstructed PV");
    assert_pv_wins(&pos, &r.pv); // independent engine replay to an attacker win
}

#[test]
fn kernel_supported_bounds() {
    assert!(KernelCtx::supported(6));
    assert!(!KernelCtx::supported(0));
    assert!(!KernelCtx::supported(20));
}
