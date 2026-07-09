//! Arbitrary-position loader for the solver. Not required to be a reachable
//! game position — supports boards that could not arise in a real game.
//!
//! [`SolverPosition`] mirrors [`prover::io::Position`] but is the public,
//! WASM-facing type. [`solve_from_position`] dispatches across [`SolverEngine`]
//! (Idtt/Pns/Dfpn/Pdspn):
//!
//! - **Idtt** (the default, production iterative-deepening solver) takes a
//!   *board-level* path: it builds a [`forcing::SolverBoard`] directly from the
//!   stones (via [`SolverPosition::to_solver_board`]) and calls
//!   [`forcing::solve_from_board`]. This never constructs a `GameState`, so it
//!   works for **fully arbitrary** boards — including ones missing the
//!   `(0,0,P1)` origin stone that `GameState::from_state` requires.
//! - **Dfpn / Pdspn / Pns** (the deep proof-search drivers) build their shared
//!   [`prover::kernel::KernelCtx`] from the stones directly (so the *search* is
//!   arbitrary-board safe), but their PV reconstruction routes through
//!   [`prover::io::Position::to_game`] → `GameState::from_state`, which seeds an
//!   origin `(0,0,P1)` stone. For a position lacking that origin the resulting
//!   `GameState` does not match the input stones (and a `(0,0,P2)` stone would
//!   panic on the collision). Deep provers therefore require a *game-valid*
//!   position (the `(0,0,P1)` origin present); for a position that is not
//!   game-valid they return [`Outcome::BudgetExceeded`] — an honest give-up, never
//!   a bogus verdict. Use Idtt for arbitrary boards.

use crate::forcing::{self, DefenseAnalysis, Limits, Outcome, SolverBoard, MAX_WL};
use crate::prover;
use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::types::{Coord, Player};

/// An arbitrary board state for the solver. Not required to be a reachable game
/// position: the stones may form any configuration (including ones missing the
/// `(0,0,P1)` origin stone a real game always starts with).
///
/// `to_move` is the attacker (the side whose forced win we test).
/// `moves_remaining` is the attacker's placements left this turn (1 or 2).
#[derive(Clone, Debug)]
pub struct SolverPosition {
    pub win_length: u8,
    pub placement_radius: i32,
    pub max_moves: u32,
    pub to_move: Player,
    pub moves_remaining: u8,
    pub stones: Vec<(Coord, Player)>,
}

/// Which solver engine to run. Idtt is the production default and the only engine
/// that supports fully arbitrary boards; the deep proof-search drivers require a
/// game-valid position (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolverEngine {
    #[default]
    Idtt,
    Pns,
    Dfpn,
    Pdspn,
}

impl SolverPosition {
    /// Build a `SolverBoard` directly from the stones, replicating the pre-sizing
    /// + reach-tracking setup that `forcing::solve_ex` does for a `GameState`:
    /// reserve the bounding box (avoids incremental grid regrowth), enable O(1)
    /// reach tracking in the tight regime before bulk-loading so counts build
    /// incrementally, then place every stone. Works on any coords — no origin
    /// stone required.
    pub fn to_solver_board(&self) -> SolverBoard {
        let mut board = SolverBoard::new();
        if let Some(&(c0, _)) = self.stones.first() {
            let (mut lo, mut hi) = (c0, c0);
            for &(c, _) in &self.stones {
                lo = (lo.0.min(c.0), lo.1.min(c.1));
                hi = (hi.0.max(c.0), hi.1.max(c.1));
            }
            board.reserve(lo, hi);
        }
        if self.placement_radius < self.win_length as i32 - 1 {
            board.enable_reach(self.placement_radius);
        }
        for &(c, p) in &self.stones {
            board.place(c, p);
        }
        board
    }

    /// Build the deep-prover [`prover::io::Position`] (same shape as `Self`).
    fn to_prover_position(&self) -> prover::io::Position {
        prover::io::Position {
            stones: self.stones.clone(),
            attacker: self.to_move,
            placements_remaining: self.moves_remaining,
            config: prover::io::PosConfig {
                win_length: self.win_length,
                placement_radius: self.placement_radius,
                max_moves: self.max_moves,
            },
        }
    }
}

/// Strict game-validity gate for the deep-prover branch. The deep provers
/// reconstruct their PV through `prover::io::Position::to_game()` →
/// `GameState::from_state`, which seeds an `(0,0,P1)` origin stone and then
/// `Board::place`s every stone — panicking (`CellOccupied`) if a stone collides
/// with the seeded origin or with another stone. This gate rejects the inputs
/// that would panic there, so the deep-prover branch can never crash on
/// untrusted `SolverPosition` input. Requires ALL of:
///
/// 1. `(0,0,P1)` is present (positive origin — the condition for the PV path to
///    reproduce the input stones).
/// 2. No coord appears more than once (uniqueness — `Board::place` panics on
///    duplicates, e.g. `((5,5),P1)` + `((5,5),P2)`).
/// 3. No `(0,0,P2)` present (the contradictory origin — `(0,0)` may only be P1).
pub fn is_game_valid_board(stones: &[(Coord, Player)]) -> bool {
    use std::collections::HashSet;
    let mut has_origin = false;
    let mut coords: HashSet<Coord> = HashSet::with_capacity(stones.len());
    for &(c, p) in stones {
        if c == (0, 0) {
            if p == Player::P2 {
                return false; // contradictory origin: (0,0) may only be P1
            }
            has_origin = true;
        }
        if !coords.insert(c) {
            return false; // duplicate coord — Board::place would panic
        }
    }
    has_origin
}

/// Defensive guard against pathological coordinate spreads, mirroring the
/// pre-sizing block in `forcing::solve_ex` (lines ~1204-1218) and
/// `prover::kernel::KernelCtx::new_wide`. A pathological spread (e.g.
/// `((i32::MAX, 0), P1)`) would blow up the dense grid (`grow` panics past
/// `MAX_GRID_CELLS`) or overflow grid arithmetic before that assert could fire.
/// Real games are spatially bounded near the origin; if either bound trips we
/// give up honestly instead of aborting/OOMing. Returns `false` if the stones'
/// bounding box is too large to allocate.
fn spread_is_valid(stones: &[(Coord, Player)]) -> bool {
    const MAX_COORD: i32 = 1 << 30;
    let mut it = stones.iter();
    let &(c0, _) = match it.next() {
        Some(x) => x,
        None => return true, // empty board — no spread to check
    };
    let (mut lo, mut hi) = (c0, c0);
    for &(c, _) in it {
        lo = (lo.0.min(c.0), lo.1.min(c.1));
        hi = (hi.0.max(c.0), hi.1.max(c.1));
    }
    if lo.0 < -MAX_COORD || lo.1 < -MAX_COORD || hi.0 > MAX_COORD || hi.1 > MAX_COORD {
        return false;
    }
    let (span_q, span_r) = (
        hi.0 as i64 - lo.0 as i64 + 6 * forcing::GRID_PAD as i64,
        hi.1 as i64 - lo.1 as i64 + 6 * forcing::GRID_PAD as i64,
    );
    span_q.saturating_mul(span_r) <= forcing::MAX_GRID_CELLS
}

/// Solve from an arbitrary position using the selected engine.
///
/// `depth_cap` bounds the iterative-deepening / search depth; `node_budget` bounds
/// the search nodes (or TT expansions). See the module docs for the
/// engine-specific arbitrary-board caveats.
pub fn solve_from_position(
    position: &SolverPosition,
    engine: SolverEngine,
    depth_cap: u8,
    node_budget: u64,
) -> Outcome {
    dispatch(position, engine, depth_cap, node_budget, false)
}

/// Like [`solve_from_position`] but with the experimental wide-partner generator
/// knob enabled (a strict superset of the tight generator: a `solve` win is never
/// lost, only possibly found at the same or a shallower depth). Maps to
/// `forcing::solve_wide` for Idtt and to `ProverConfig { wide: true }` for the deep
/// drivers.
pub fn solve_wide_from_position(
    position: &SolverPosition,
    engine: SolverEngine,
    depth_cap: u8,
    node_budget: u64,
) -> Outcome {
    dispatch(position, engine, depth_cap, node_budget, true)
}

/// Defensive analysis for the side to move (see [`forcing::solve_defense`]):
/// detects the opponent's flipped-perspective forcing threat and reports which
/// placements refute it (killers / pair anchors / best-delay fallback).
///
/// Unlike [`solve_from_position`] (which dispatches across engines), defense
/// analysis always uses the forcing `solve` (idtt) internally — the real
/// `forcing::solve_defense` stacks many budget-bound sub-solves and the forcing
/// solver is the only engine that supports the candidate-verification pattern.
/// The `engine` selection from the caller is therefore inert for defense.
///
/// Requires a **game-valid** position: the analysis builds a `GameState` via
/// `GameState::from_state` (which seeds the `(0,0,P1)` origin and panics on
/// duplicate/contradictory coords — see [`is_game_valid_board`]). For a position
/// that is not game-valid, returns `None` (an honest give-up that conflates with
/// "no threat found"; the WASM layer gates with [`is_game_valid_board`] and
/// returns a clear `Err` for invalid positions, so the WASM API never conflates).
///
/// `time_limit` bounds the whole analysis on native (partial results on expiry);
/// on wasm32 the deadline is skipped (`Instant::now()` panics there) and
/// `node_budget` is the sole bound.
pub fn solve_defense_from_position(
    position: &SolverPosition,
    depth_cap: u8,
    node_budget: u64,
    time_limit: std::time::Duration,
) -> Option<DefenseAnalysis> {
    // Defense builds a GameState (GameState::from_state seeds (0,0,P1) and panics
    // on duplicate/contradictory coords). Gate on game-validity; invalid → None.
    if !is_game_valid_board(&position.stones) {
        return None;
    }
    let cfg = GameConfig {
        win_length: position.win_length,
        placement_radius: position.placement_radius,
        max_moves: position.max_moves,
    };
    let game = GameState::from_state(
        &position.stones,
        position.to_move,
        position.moves_remaining,
        cfg,
    );
    forcing::solve_defense(&game, depth_cap, node_budget, time_limit)
}

fn dispatch(
    position: &SolverPosition,
    engine: SolverEngine,
    depth_cap: u8,
    node_budget: u64,
    wide: bool,
) -> Outcome {
    // The strip-scan buffers are stack-sized for win_length in 1..=MAX_WL; anything
    // else cannot be analyzed — report an honest give-up, never a bogus `No`.
    if !(1..=MAX_WL).contains(&(position.win_length as usize)) {
        return Outcome::BudgetExceeded;
    }
    // moves_remaining must be 1 or 2 (a turn has at most two placements).
    if !(1..=2).contains(&position.moves_remaining) {
        return Outcome::BudgetExceeded;
    }
    // Pathological-spread guard (mirrors forcing::solve_ex + KernelCtx::new_wide):
    // protects the idtt board-level path (and is a harmless belt-and-suspenders
    // check for the deep provers, which re-guard in KernelCtx). A pathological
    // coordinate spread would blow up the dense grid or overflow grid arithmetic;
    // give up honestly instead of aborting/OOMing.
    if !spread_is_valid(&position.stones) {
        return Outcome::BudgetExceeded;
    }

    match engine {
        SolverEngine::Idtt => {
            let mut board = position.to_solver_board();
            forcing::solve_from_board(
                &mut board,
                position.to_move,
                position.moves_remaining,
                position.win_length,
                position.placement_radius,
                depth_cap,
                node_budget,
                wide,
                Limits::default(),
            )
        }
        SolverEngine::Dfpn | SolverEngine::Pdspn | SolverEngine::Pns => {
            // Deep provers: see the module-level caveat. The search itself (KernelCtx
            // from stones) is arbitrary-board safe, but the PV reconstruction uses
            // `to_game()` which requires the `(0,0,P1)` origin and panics on a
            // contradictory `(0,0,P2)` or any duplicate coord. Restrict to game-valid
            // positions; idtt handles arbitrary boards.
            if !is_game_valid_board(&position.stones) {
                return Outcome::BudgetExceeded;
            }
            let prover_pos = position.to_prover_position();
            let cfg = prover_config(engine, depth_cap, node_budget, wide);
            let ctl = prover::Ctl::new(0.0);
            let result = match engine {
                SolverEngine::Dfpn => prover::dfpn::solve(&prover_pos, &cfg, &ctl),
                SolverEngine::Pdspn => prover::pdspn::solve(&prover_pos, &cfg, &ctl),
                // Pns has no standalone driver entry point; run the shared best-first
                // `PnSearch` (the same level-2 engine PDS-PN uses) as a pure PN search.
                SolverEngine::Pns => pns_solve(&prover_pos, &cfg, &ctl),
                _ => unreachable!(),
            };
            driver_to_outcome(result)
        }
    }
}

/// Build a [`prover::ProverConfig`] for one deep-prover run from the public
/// dispatch params.
fn prover_config(engine: SolverEngine, depth_cap: u8, node_budget: u64, wide: bool) -> prover::ProverConfig {
    let driver = match engine {
        SolverEngine::Dfpn => prover::DriverKind::Dfpn,
        SolverEngine::Pdspn => prover::DriverKind::Pdspn,
        // Pns reuses the dfpn DriverKind slot only so the config is well-formed; the
        // actual dispatch above never reads `cfg.driver` for Pns (pns_solve ignores it).
        SolverEngine::Pns => prover::DriverKind::Dfpn,
        _ => prover::DriverKind::Idtt,
    };
    prover::ProverConfig {
        driver,
        wide,
        depth_cap,
        node_budget,
        tt_mb: 512,
        pn2_nodes: 50_000,
        leaf_budget: 1_000_000,
        leaf_budget_max: 10_000_000,
        time_limit_s: 0.0,
        race_set: Vec::new(),
    }
}

/// Map a deep-prover [`prover::DriverResult`] to a forcing [`Outcome`].
///
/// `Verdict::Win` carries an optional `depth` and `pv`; we reconstruct
/// `ForcingWin` from them. `Unverified` (the hybrid driver's open-branch result)
/// is treated as `BudgetExceeded` — it is not a definitive proof. For a Win with no
/// depth/pv (defensive: should not happen for dfpn/pdspn, and never for our Pns
/// wrapper which doesn't reconstruct a PV) we synthesize a minimal `ForcingWin`
/// with `depth = 0` and an empty PV rather than panic; the caller can detect the
/// empty PV if it needs one.
///
/// Note: `ForcingWin.first_move` is only meaningful when `pv` is non-empty. For
/// engines that return a verdict only (Pns), `pv` is empty and `first_move` is a
/// `(0,0)` sentinel — callers must check `!pv.is_empty()` before using
/// `first_move`.
fn driver_to_outcome(r: prover::DriverResult) -> Outcome {
    use prover::io::Verdict;
    match r.verdict {
        Verdict::Win => {
            let depth = r.depth.unwrap_or(0);
            let first_move = r.pv.first().copied().unwrap_or((0, 0));
            Outcome::Win(forcing::ForcingWin {
                depth,
                first_move,
                pv: r.pv,
            })
        }
        Verdict::No => Outcome::No,
        Verdict::Unverified | Verdict::BudgetExceeded => Outcome::BudgetExceeded,
    }
}

/// Pure best-first proof-number search (the `PnSearch` engine PDS-PN uses at
/// level 2), run as a standalone driver. Returns only a verdict: a PN search does
/// not reconstruct a principal variation, so for a `Win` the `DriverResult` has
/// `depth = None` and an empty `pv` — `driver_to_outcome` then synthesizes a
/// `ForcingWin` with `depth = 0` (the WASM caller that needs a PV should use Idtt,
/// which reconstructs it; Pns is a verdict-only cross-check).
fn pns_solve(pos: &prover::io::Position, cfg: &prover::ProverConfig, ctl: &prover::Ctl) -> prover::DriverResult {
    use prover::io::Verdict;
    use prover::kernel::{KernelCtx, Node};
    use prover::pn::PnSearch;

    let wl = pos.config.win_length;
    if !KernelCtx::supported(wl) {
        return prover::DriverResult::new(Verdict::BudgetExceeded);
    }
    let mut k = match KernelCtx::new_wide(
        &pos.stones,
        pos.attacker,
        wl,
        pos.config.placement_radius,
        cfg.wide,
    ) {
        Some(c) => c,
        None => return prover::DriverResult::new(Verdict::BudgetExceeded),
    };
    let root = Node::Or { placements: pos.placements_remaining };
    let mut search = PnSearch::new(cfg.pn2_nodes);
    // `Instant::now()` traps at runtime on wasm32-unknown-unknown (no monotonic
    // clock in std for that target). The elapsed value is only used for the
    // `stats.elapsed_s` report (not correctness), so on wasm we report 0.0.
    #[cfg(not(target_arch = "wasm32"))]
    let t = std::time::Instant::now();
    let (pn, dn) = search.search(&mut k, root);
    #[cfg(not(target_arch = "wasm32"))]
    let elapsed = t.elapsed().as_secs_f64();
    #[cfg(target_arch = "wasm32")]
    let elapsed: f64 = 0.0;

    // Verdict discipline mirrors dfpn: only a genuinely resolved root is WIN/NO.
    let verdict = if pn == 0 {
        Verdict::Win
    } else if dn == 0 && !ctl.expired() {
        Verdict::No
    } else {
        Verdict::BudgetExceeded
    };
    let mut res = prover::DriverResult::new(verdict);
    res.stats.elapsed_s = elapsed;
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::types::Player;

    /// A position with an obvious forced win for P1: three in a row along the q-axis
    /// at win_length 4 (one completion away), P2 scattered and not blocking. Includes
    /// the `(0,0,P1)` origin so deep provers (which require a game-valid position
    /// for their PV path) can also run on it.
    fn tiny_win_position() -> SolverPosition {
        SolverPosition {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![
                ((0, 0), Player::P1),
                ((1, 0), Player::P1),
                ((2, 0), Player::P1),
                // P2 stones scattered, not blocking the q-axis run.
                ((5, 5), Player::P2),
            ],
        }
    }

    /// A position with the origin omitted — an arbitrary board that could not arise
    /// in a real game. Idtt must still solve it; deep provers must give up honestly.
    fn arbitrary_no_origin_position() -> SolverPosition {
        SolverPosition {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![
                ((3, 0), Player::P1),
                ((4, 0), Player::P1),
                ((5, 0), Player::P1),
                ((8, 8), Player::P2),
            ],
        }
    }

    #[test]
    fn idtt_finds_win_from_position() {
        let pos = tiny_win_position();
        let outcome = solve_from_position(&pos, SolverEngine::Idtt, 6, 20_000);
        assert!(
            outcome.is_win(),
            "idtt should find a forced win for the three-in-a-row position, got {outcome:?}"
        );
        if let Outcome::Win(w) = &outcome {
            assert!(w.depth >= 1, "a win must have positive depth");
            assert!(!w.pv.is_empty(), "a win must carry a principal variation");
            assert_eq!(w.first_move, w.pv[0], "first_move must equal pv[0]");
        }
    }

    #[test]
    fn idtt_wide_finds_win_from_position() {
        let pos = tiny_win_position();
        let outcome = solve_wide_from_position(&pos, SolverEngine::Idtt, 6, 20_000);
        assert!(outcome.is_win(), "idtt-wide should find the win, got {outcome:?}");
    }

    #[test]
    fn dfpn_agrees_with_idtt() {
        let pos = tiny_win_position();
        let idtt = solve_from_position(&pos, SolverEngine::Idtt, 6, 20_000);
        let dfpn = solve_from_position(&pos, SolverEngine::Dfpn, 6, 20_000);
        // Both should agree on Win/No (depths may differ).
        assert_eq!(idtt.is_win(), dfpn.is_win(), "dfpn must agree with idtt");
        if idtt.is_win() {
            assert!(dfpn.is_win());
        }
    }

    #[test]
    fn pdspn_agrees_with_idtt() {
        let pos = tiny_win_position();
        let idtt = solve_from_position(&pos, SolverEngine::Idtt, 6, 20_000);
        let pdspn = solve_from_position(&pos, SolverEngine::Pdspn, 6, 20_000);
        assert_eq!(idtt.is_win(), pdspn.is_win(), "pdspn must agree with idtt");
    }

    #[test]
    fn pns_agrees_with_idtt() {
        let pos = tiny_win_position();
        let idtt = solve_from_position(&pos, SolverEngine::Idtt, 6, 20_000);
        let pns = solve_from_position(&pos, SolverEngine::Pns, 6, 20_000);
        assert_eq!(idtt.is_win(), pns.is_win(), "pns must agree with idtt");
    }

    #[test]
    fn idtt_solves_arbitrary_no_origin_position() {
        // A board that could not arise in a real game (no (0,0,P1) origin stone).
        // Idtt takes the board-level path and must still solve it.
        let pos = arbitrary_no_origin_position();
        let outcome = solve_from_position(&pos, SolverEngine::Idtt, 6, 20_000);
        assert!(outcome.is_win(), "idtt must solve the arbitrary no-origin board");
    }

    #[test]
    fn deep_provers_give_up_honestly_on_arbitrary_no_origin() {
        // Deep provers require the (0,0,P1) origin for their PV path; without it they
        // must return BudgetExceeded (an honest give-up), never panic or fabricate.
        let pos = arbitrary_no_origin_position();
        let dfpn = solve_from_position(&pos, SolverEngine::Dfpn, 6, 20_000);
        let pdspn = solve_from_position(&pos, SolverEngine::Pdspn, 6, 20_000);
        let pns = solve_from_position(&pos, SolverEngine::Pns, 6, 20_000);
        assert!(
            matches!(dfpn, Outcome::BudgetExceeded),
            "dfpn must give up honestly without origin, got {dfpn:?}"
        );
        assert!(
            matches!(pdspn, Outcome::BudgetExceeded),
            "pdspn must give up honestly without origin, got {pdspn:?}"
        );
        assert!(
            matches!(pns, Outcome::BudgetExceeded),
            "pns must give up honestly without origin, got {pns:?}"
        );
    }

    #[test]
    fn invalid_win_length_gives_up() {
        let mut pos = tiny_win_position();
        pos.win_length = 0;
        assert!(matches!(
            solve_from_position(&pos, SolverEngine::Idtt, 6, 20_000),
            Outcome::BudgetExceeded
        ));
        pos.win_length = MAX_WL as u8 + 1;
        assert!(matches!(
            solve_from_position(&pos, SolverEngine::Idtt, 6, 20_000),
            Outcome::BudgetExceeded
        ));
    }

    #[test]
    fn no_forcing_win_reports_no() {
        // P2 has no completion nearby and no forcing line at win_length 6 with a
        // wide radius; P1 to move but stones placed so neither side threatens.
        // Use a sparse draw-ish position: a single P1 stone and a single P2 stone
        // far apart at win_length 6, radius 2 (tight). The soundness gate should
        // short-circuit to No (no four-threat, no completion within placements).
        let pos = SolverPosition {
            win_length: 6,
            placement_radius: 2,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![((0, 0), Player::P1), ((10, 10), Player::P2)],
        };
        let outcome = solve_from_position(&pos, SolverEngine::Idtt, 6, 20_000);
        assert!(
            matches!(outcome, Outcome::No) || matches!(outcome, Outcome::BudgetExceeded),
            "a non-forcing position should be No or BudgetExceeded, got {outcome:?}"
        );
    }

    // --- Task 4 review-fix tests: panic prevention on untrusted input ---

    #[test]
    fn dfpn_does_not_panic_on_contradictory_origin() {
        // (0,0,P1) AND (0,0,P2) present — a contradictory origin. GameState::from_state
        // seeds (0,0,P1) then Board::place((0,0,P2)) would panic with CellOccupied on
        // the deep-prover PV reconstruction path. The strict gate must catch this and
        // return BudgetExceeded, never panic. Belt-and-suspenders: wrap in
        // catch_unwind to assert no unwind.
        let pos = SolverPosition {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![
                ((0, 0), Player::P1),
                ((0, 0), Player::P2), // contradictory origin
                ((1, 0), Player::P1),
                ((2, 0), Player::P1),
                ((5, 5), Player::P2),
            ],
        };
        let outcome = std::panic::catch_unwind(|| {
            solve_from_position(&pos, SolverEngine::Dfpn, 6, 20_000)
        });
        assert!(outcome.is_ok(), "dfpn must not panic on contradictory origin");
        assert!(
            matches!(outcome.unwrap(), Outcome::BudgetExceeded),
            "dfpn must give up honestly on contradictory origin"
        );
    }

    #[test]
    fn dfpn_does_not_panic_on_duplicate_coord() {
        // (5,5) appears twice (P1 and P2) — Board::place panics with CellOccupied on
        // the deep-prover PV reconstruction path. The uniqueness check must catch
        // this and return BudgetExceeded, never panic.
        let pos = SolverPosition {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![
                ((0, 0), Player::P1),
                ((1, 0), Player::P1),
                ((2, 0), Player::P1),
                ((5, 5), Player::P1), // duplicate coord (different player)
                ((5, 5), Player::P2),
            ],
        };
        let outcome = std::panic::catch_unwind(|| {
            solve_from_position(&pos, SolverEngine::Dfpn, 6, 20_000)
        });
        assert!(outcome.is_ok(), "dfpn must not panic on duplicate coord");
        assert!(
            matches!(outcome.unwrap(), Outcome::BudgetExceeded),
            "dfpn must give up honestly on duplicate coord"
        );
    }

    #[test]
    fn idtt_does_not_panic_on_pathological_spread() {
        // Coordinates near the i32 extremes would blow up the dense grid (grow
        // panics past MAX_GRID_CELLS) or overflow grid arithmetic. The spread guard
        // must return BudgetExceeded, never abort/OOM. Origin included so the deep
        // provers are not rejected for origin reasons — this isolates the spread guard.
        let pos = SolverPosition {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![
                ((i32::MAX, 0), Player::P1),
                ((0, i32::MAX), Player::P2),
                ((0, 0), Player::P1),
            ],
        };
        let outcome = std::panic::catch_unwind(|| {
            solve_from_position(&pos, SolverEngine::Idtt, 6, 20_000)
        });
        assert!(outcome.is_ok(), "idtt must not panic on pathological spread");
        assert!(
            matches!(outcome.unwrap(), Outcome::BudgetExceeded),
            "idtt must give up honestly on pathological spread"
        );
    }
}
