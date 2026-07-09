//! wasm-bindgen export for the standalone VCF solver.
//!
//! Pure combinatorial search (no model weights) wired to the `hexo-solver`
//! crate. Exposes [`StrixSolver`] with `solve` / `solve_wide` / `solve_threat`
//! (Task 5) and a `solve_defense` stub (Task 6 fills it in).
//!
//! Timing note: `std::time::Instant::now()` panics at runtime on
//! `wasm32-unknown-unknown` (no monotonic clock in std for that target). We
//! gate it to native targets and report `time_ms = 0.0` on wasm to avoid a
//! runtime panic. A future switch to the `web-time` crate would restore
//! timing on wasm without a cfg gate.

use hexo_engine::types::{Coord, Player as EnginePlayer};
use hexo_solver::forcing::{ForcingWin, Outcome};
use hexo_solver::{
    is_game_valid_board, solve_defense_from_position, solve_from_position,
    solve_wide_from_position, SolverEngine, SolverPosition,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Player {
    P1 = 1,
    P2 = 2,
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverEngineEnum {
    Idtt = 0,
    Pns = 1,
    Dfpn = 2,
    Pdspn = 3,
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoordW {
    pub q: i32,
    pub r: i32,
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Clone, Debug)]
pub struct Stone {
    pub coord: CoordW,
    pub player: Player,
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Clone, Debug)]
pub struct Position {
    pub win_length: u8,
    pub placement_radius: i32,
    pub max_moves: u32,
    pub to_move: Player,
    pub moves_remaining: u8,
    pub stones: Vec<Stone>,
}

#[wasm_bindgen]
pub struct SolverLimits {
    pub depth_cap: u8,
    pub node_budget: u64,
    pub engine: SolverEngineEnum,
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveKind {
    Win = 0,
    No = 1,
    BudgetExceeded = 2,
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Clone, Debug)]
pub struct Turn {
    pub turn: u32,
    pub player: Player,
    pub cells: Vec<CoordW>,
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Clone, Debug)]
pub struct SolveOutcome {
    pub kind: SolveKind,
    pub depth: u8,
    pub nodes: u64,
    pub time_ms: f64,
    pub pv: Vec<Turn>,
}

// --- Task 6: defense analysis export ---

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefenseKind {
    /// The opponent has a proven forcing threat (the mover has a defense to find).
    ThreatFound = 0,
    /// No proven opponent threat — nothing to defend. NOTE: this conflates "no
    /// threat" with "budget exceeded during the threat check" (`solve_defense`
    /// returns `None` for both); the caller cannot distinguish them from this
    /// enum alone. A larger `node_budget` disambiguates on re-query.
    NoThreat = 1,
    /// Budget exceeded during the threat check. Currently UNREACHABLE: the
    /// `solve_defense` mapping produces only `ThreatFound`/`NoThreat` (the
    /// underlying `solve_defense` returns `None` for both no-threat and
    /// budget-exceeded, which maps to `NoThreat`). Reserved for a future revision
    /// where `solve_defense` distinguishes budget-exceeded from no-threat.
    BudgetExceeded = 2,
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairAnchor {
    pub first: CoordW,
    pub second: CoordW,
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Debug)]
pub struct DefenseOutcome {
    /// `ThreatFound` if the opponent has a proven threat, else `NoThreat`.
    pub kind: DefenseKind,
    /// Total search nodes across the threat solve + all candidate re-solves.
    /// `DefenseAnalysis` does not currently aggregate node counts, so this is 0
    /// (a future enhancement could propagate the sum).
    pub nodes: u64,
    /// Wall-clock time of the whole analysis (ms). `0.0` on wasm (no monotonic
    /// clock — see the module-level `Instant::now()` note).
    pub time_ms: f64,
    /// The opponent's threat as a `SolveOutcome` (`kind: Win`), chunked into
    /// turns the same way as `solve`: the threat is a fresh 2-placement turn
    /// for the opponent, so the first turn has 2 cells. `None` when there is no
    /// proven threat. `depth` is the real threat depth (captured in
    /// `DefenseAnalysis::threat_depth`); `nodes` is 0 (the sub-solve's node
    /// count is not surfaced by `solve_defense`).
    pub threat: Option<SolveOutcome>,
    /// Single placements after which the threat is no longer provable (or which
    /// end the game outright).
    pub killers: Vec<CoordW>,
    /// `(first, second)` placement pairs that jointly refute the threat.
    /// Searched only when the mover has 2 placements left and `killers` is empty.
    pub pair_anchors: Vec<PairAnchor>,
    /// Max-delay fallback: the surviving candidate whose re-proven threat PV is
    /// longest. `None` when the threat PV offers no legal candidate.
    pub best_delay: Option<CoordW>,
}

#[wasm_bindgen]
pub struct StrixSolver;

#[wasm_bindgen]
impl StrixSolver {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        StrixSolver
    }

    pub fn solve(
        &self,
        position: &Position,
        limits: &SolverLimits,
    ) -> Result<SolveOutcome, JsError> {
        Self::solve_inner(position, limits, false)
    }

    pub fn solve_wide(
        &self,
        position: &Position,
        limits: &SolverLimits,
    ) -> Result<SolveOutcome, JsError> {
        Self::solve_inner(position, limits, true)
    }

    /// Opponent's forcing win if left unblocked: flip `to_move` and solve.
    /// Does NOT call `forcing::solve_threat` (GameState-based); the flip +
    /// re-solve IS the threat semantics.
    pub fn solve_threat(
        &self,
        position: &Position,
        limits: &SolverLimits,
    ) -> Result<SolveOutcome, JsError> {
        let mut flipped = position.clone();
        flipped.to_move = match flipped.to_move {
            Player::P1 => Player::P2,
            Player::P2 => Player::P1,
        };
        Self::solve_inner(&flipped, limits, false)
    }

    /// Defensive analysis for the side to move: detects the opponent's
    /// flipped-perspective forcing threat and reports which placements refute it
    /// (killers / pair anchors / best-delay fallback). Wraps
    /// `hexo_solver::solve_defense_from_position`, which delegates to the
    /// battle-tested `forcing::solve_defense`.
    ///
    /// `limits.engine` is INERT for defense: the analysis always uses the forcing
    /// (idtt) solver internally (the candidate-verification pattern is
    /// forcing-only). `depth_cap` and `node_budget` are honored. The time limit
    /// is a fixed 10s default on native (the `SolverLimits` surface has no time
    /// field); on wasm the deadline is skipped and `node_budget` is the sole
    /// bound.
    ///
    /// Requires a GAME-VALID position (the `(0,0,P1)` origin present, no
    /// duplicate/contradictory coords — `GameState::from_state` seeds the origin
    /// and panics otherwise). For an invalid position returns `Err(JsError)`
    /// with a clear message; defense analysis is inherently about a game
    /// position's turn, so requiring game-validity is consistent with the deep
    /// provers (Task 4).
    pub fn solve_defense(
        &self,
        position: &Position,
        limits: &SolverLimits,
    ) -> Result<DefenseOutcome, JsError> {
        let t0 = instant_now();
        let pos = convert_position(position)?;
        if !is_game_valid_board(&pos.stones) {
            return Err(JsError::new(
                "solve_defense requires a game-valid position: the (0,0,P1) origin \
                 stone must be present, with no duplicate or contradictory coords",
            ));
        }
        let analysis = solve_defense_from_position(
            &pos,
            limits.depth_cap,
            limits.node_budget,
            std::time::Duration::from_secs(10),
        );
        let time_ms = elapsed_ms(t0);
        // `solve_defense` returns None for BOTH "no threat" and "budget exceeded
        // during the threat check" — there is no distinct BudgetExceeded case
        // (see forcing.rs: Outcome::No | Outcome::BudgetExceeded => return None).
        // Map None → NoThreat (documented conflation).
        match analysis {
            None => Ok(DefenseOutcome {
                kind: DefenseKind::NoThreat,
                nodes: 0,
                time_ms,
                threat: None,
                killers: Vec::new(),
                pair_anchors: Vec::new(),
                best_delay: None,
            }),
            Some(a) => {
                // The threat is the opponent's fresh 2-placement turn: chunk with
                // moves_remaining=2, attacker = opponent of `to_move`.
                let threat_attacker = opponent(position.to_move);
                let threat = if a.threat_pv.is_empty() {
                    None
                } else {
                    Some(SolveOutcome {
                        kind: SolveKind::Win,
                        depth: a.threat_depth,
                        nodes: 0, // solve_defense does not surface sub-solve node counts
                        time_ms: 0.0, // the threat sub-solve's own time is not isolated
                        pv: chunk_pv(&a.threat_pv, 2, threat_attacker),
                    })
                };
                Ok(DefenseOutcome {
                    kind: if threat.is_some() {
                        DefenseKind::ThreatFound
                    } else {
                        DefenseKind::NoThreat
                    },
                    nodes: 0,
                    time_ms,
                    threat,
                    killers: a.killers.iter().map(|&(q, r)| CoordW { q, r }).collect(),
                    pair_anchors: a
                        .pair_anchors
                        .iter()
                        .map(|&(a, b)| PairAnchor {
                            first: CoordW { q: a.0, r: a.1 },
                            second: CoordW { q: b.0, r: b.1 },
                        })
                        .collect(),
                    best_delay: a.best_delay.map(|(q, r)| CoordW { q, r }),
                })
            }
        }
    }
}

impl StrixSolver {
    fn solve_inner(
        position: &Position,
        limits: &SolverLimits,
        wide: bool,
    ) -> Result<SolveOutcome, JsError> {
        let t0 = instant_now();
        let pos = convert_position(position)?;
        let engine = match limits.engine {
            SolverEngineEnum::Idtt => SolverEngine::Idtt,
            SolverEngineEnum::Pns => SolverEngine::Pns,
            SolverEngineEnum::Dfpn => SolverEngine::Dfpn,
            SolverEngineEnum::Pdspn => SolverEngine::Pdspn,
        };
        let outcome = if wide {
            solve_wide_from_position(&pos, engine, limits.depth_cap, limits.node_budget)
        } else {
            solve_from_position(&pos, engine, limits.depth_cap, limits.node_budget)
        };
        let time_ms = elapsed_ms(t0);
        Ok(convert_outcome(
            &outcome,
            time_ms,
            position.moves_remaining,
            position.to_move,
        ))
    }
}

/// `Instant::now()` shim: returns a handle on native, `None` on wasm (where
/// `Instant::now()` panics). `elapsed_ms(None)` reports `0.0`.
#[cfg(not(target_arch = "wasm32"))]
fn instant_now() -> Option<std::time::Instant> {
    Some(std::time::Instant::now())
}

#[cfg(target_arch = "wasm32")]
fn instant_now() -> Option<std::time::Instant> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn elapsed_ms(t0: Option<std::time::Instant>) -> f64 {
    t0.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0)
}

#[cfg(target_arch = "wasm32")]
fn elapsed_ms(_t0: Option<std::time::Instant>) -> f64 {
    0.0
}

fn convert_position(pos: &Position) -> Result<SolverPosition, JsError> {
    let to_move = match pos.to_move {
        Player::P1 => EnginePlayer::P1,
        Player::P2 => EnginePlayer::P2,
    };
    let stones: Vec<_> = pos
        .stones
        .iter()
        .map(|s| {
            (
                (s.coord.q, s.coord.r),
                match s.player {
                    Player::P1 => EnginePlayer::P1,
                    Player::P2 => EnginePlayer::P2,
                },
            )
        })
        .collect();
    Ok(SolverPosition {
        win_length: pos.win_length,
        placement_radius: pos.placement_radius,
        max_moves: pos.max_moves,
        to_move,
        moves_remaining: pos.moves_remaining,
        stones,
    })
}

/// Chunk the flat pv into turn-grouped `Turn` objects.
/// - Turn 0: attacker, `moves_remaining` cells
/// - Turn 1: defender, 2 cells (min-cover)
/// - Turn 2: attacker, 2 cells (fresh turn, always 2 placements)
/// - Turn 3: defender, 2 cells
/// - ... alternate, every turn after the first has 2 cells.
fn chunk_pv(pv: &[Coord], moves_remaining: u8, attacker: Player) -> Vec<Turn> {
    if pv.is_empty() {
        return Vec::new();
    }
    let mut turns = Vec::new();
    let mut idx = 0;
    let mut turn_num = 0u32;
    let mut current_player = attacker;
    // First attacker turn: moves_remaining cells.
    let first_len = moves_remaining.min(pv.len() as u8) as usize;
    turns.push(Turn {
        turn: turn_num,
        player: current_player,
        cells: pv[idx..idx + first_len]
            .iter()
            .map(|&(q, r)| CoordW { q, r })
            .collect(),
    });
    idx += first_len;
    turn_num += 1;
    current_player = opponent(current_player);
    // Subsequent turns: 2 cells each, alternating.
    while idx < pv.len() {
        let len = 2.min(pv.len() - idx);
        turns.push(Turn {
            turn: turn_num,
            player: current_player,
            cells: pv[idx..idx + len]
                .iter()
                .map(|&(q, r)| CoordW { q, r })
                .collect(),
        });
        idx += len;
        turn_num += 1;
        current_player = opponent(current_player);
    }
    turns
}

fn opponent(p: Player) -> Player {
    match p {
        Player::P1 => Player::P2,
        Player::P2 => Player::P1,
    }
}

fn convert_outcome(
    outcome: &Outcome,
    time_ms: f64,
    moves_remaining: u8,
    attacker: Player,
) -> SolveOutcome {
    match outcome {
        Outcome::Win(w) => SolveOutcome {
            kind: SolveKind::Win,
            depth: w.depth,
            nodes: 0, // idtt does not report nodes; deep provers do via stats (future)
            time_ms,
            pv: chunk_pv(&w.pv, moves_remaining, attacker),
        },
        Outcome::No => SolveOutcome {
            kind: SolveKind::No,
            depth: 0,
            nodes: 0,
            time_ms,
            pv: Vec::new(),
        },
        Outcome::BudgetExceeded => SolveOutcome {
            kind: SolveKind::BudgetExceeded,
            depth: 0,
            nodes: 0,
            time_ms,
            pv: Vec::new(),
        },
    }
}

// `ForcingWin` is imported for the `Outcome::Win(ForcingWin)` variant match
// in `convert_outcome`; the binding uses field access, not the type name, but
// the import documents the variant's payload type.
#[allow(unused_imports)]
use ForcingWin as _;

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_position() -> Position {
        Position {
            win_length: 6,
            placement_radius: 8,
            max_moves: 300,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: Vec::new(),
        }
    }

    fn limits_idtt(depth: u8, budget: u64) -> SolverLimits {
        SolverLimits {
            depth_cap: depth,
            node_budget: budget,
            engine: SolverEngineEnum::Idtt,
        }
    }

    /// An empty board has no forcing line: the solver must report `No` (or
    /// `BudgetExceeded`), never `Win`.
    #[test]
    fn solve_returns_no_on_empty_board() {
        let solver = StrixSolver::new();
        let pos = empty_position();
        let out = solver.solve(&pos, &limits_idtt(6, 5_000)).unwrap();
        assert!(
            matches!(out.kind, SolveKind::No | SolveKind::BudgetExceeded),
            "empty board must not be a win, got {:?}",
            out.kind
        );
    }

    /// Three-in-a-row at win_length 4 is a forced win for P1 (one completion away).
    #[test]
    fn solve_finds_win_on_three_in_a_row() {
        let solver = StrixSolver::new();
        let pos = Position {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![
                Stone {
                    coord: CoordW { q: 0, r: 0 },
                    player: Player::P1,
                },
                Stone {
                    coord: CoordW { q: 1, r: 0 },
                    player: Player::P1,
                },
                Stone {
                    coord: CoordW { q: 2, r: 0 },
                    player: Player::P1,
                },
                Stone {
                    coord: CoordW { q: 5, r: 5 },
                    player: Player::P2,
                },
            ],
        };
        let out = solver.solve(&pos, &limits_idtt(6, 20_000)).unwrap();
        assert!(matches!(out.kind, SolveKind::Win), "expected a win, got {:?} (depth {}, pv_len {})",
            out.kind, out.depth, out.pv.len());
        assert!(out.depth >= 1, "a win must have positive depth");
        assert!(!out.pv.is_empty(), "a win must carry a chunked pv");
        // First turn is the attacker with moves_remaining (2) cells.
        assert_eq!(out.pv[0].turn, 0, "first turn index is 0");
        assert_eq!(out.pv[0].player, Player::P1, "first turn is the attacker");
        assert!(out.pv[0].cells.len() <= 2, "first turn has at most moves_remaining cells");
    }

    /// `solve_threat` flips `to_move` and re-solves: on a position where P1 has
    /// three-in-a-row (P1's win), the threat from P2's perspective is P1's win.
    #[test]
    fn solve_threat_flips_to_move() {
        let solver = StrixSolver::new();
        let pos = Position {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P2,
            moves_remaining: 2,
            stones: vec![
                Stone {
                    coord: CoordW { q: 0, r: 0 },
                    player: Player::P1,
                },
                Stone {
                    coord: CoordW { q: 1, r: 0 },
                    player: Player::P1,
                },
                Stone {
                    coord: CoordW { q: 2, r: 0 },
                    player: Player::P1,
                },
                Stone {
                    coord: CoordW { q: 5, r: 5 },
                    player: Player::P2,
                },
            ],
        };
        // P2 to move: no forcing win for P2.
        let out = solver.solve(&pos, &limits_idtt(6, 20_000)).unwrap();
        assert!(
            matches!(out.kind, SolveKind::No | SolveKind::BudgetExceeded),
            "P2 has no forcing win here, got {:?}",
            out.kind
        );
        // Threat (flip to P1): P1 has the forcing win.
        let threat = solver.solve_threat(&pos, &limits_idtt(6, 20_000)).unwrap();
        assert!(matches!(threat.kind, SolveKind::Win), "threat should be P1's win, got {:?}", threat.kind);
    }

    /// `solve_wide` is a superset of `solve`: a solve win is never lost.
    #[test]
    fn solve_wide_finds_win_too() {
        let solver = StrixSolver::new();
        let pos = Position {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![
                Stone {
                    coord: CoordW { q: 0, r: 0 },
                    player: Player::P1,
                },
                Stone {
                    coord: CoordW { q: 1, r: 0 },
                    player: Player::P1,
                },
                Stone {
                    coord: CoordW { q: 2, r: 0 },
                    player: Player::P1,
                },
                Stone {
                    coord: CoordW { q: 5, r: 5 },
                    player: Player::P2,
                },
            ],
        };
        let out = solver.solve_wide(&pos, &limits_idtt(6, 20_000)).unwrap();
        assert!(matches!(out.kind, SolveKind::Win), "wide should also find the win, got {:?}", out.kind);
    }

    /// `solve_defense` on a position where the opponent (P1) has a proven
    /// forcing threat against the mover (P2) reports `ThreatFound` with a
    /// chunked threat PV. Uses the three-in-a-row-at-win_length-4 shape: P1
    /// has three along the q-axis, P2 to move — P1's flipped fresh-turn solve is
    /// a win, so the defense finds a threat. Killers/pairs/best_delay are
    /// whatever the analysis reports (not asserted beyond non-panic + kind).
    #[test]
    fn solve_defense_finds_threat() {
        let solver = StrixSolver::new();
        let pos = Position {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P2,
            moves_remaining: 2,
            stones: vec![
                Stone { coord: CoordW { q: 0, r: 0 }, player: Player::P1 },
                Stone { coord: CoordW { q: 1, r: 0 }, player: Player::P1 },
                Stone { coord: CoordW { q: 2, r: 0 }, player: Player::P1 },
                Stone { coord: CoordW { q: 5, r: 5 }, player: Player::P2 },
            ],
        };
        let out = solver.solve_defense(&pos, &limits_idtt(6, 20_000)).unwrap();
        assert_eq!(out.kind, DefenseKind::ThreatFound, "P1's three-in-a-row is a threat");
        let threat = out.threat.as_ref().expect("ThreatFound must carry a threat");
        assert!(matches!(threat.kind, SolveKind::Win), "threat must be a Win");
        assert!(threat.depth >= 1, "threat depth must be positive");
        assert!(!threat.pv.is_empty(), "threat must carry a chunked pv");
        // The threat is P1's fresh 2-placement turn: first turn is P1 with 2 cells.
        assert_eq!(threat.pv[0].turn, 0);
        assert_eq!(threat.pv[0].player, Player::P1, "threat attacker is P1 (opponent)");
        assert!(threat.pv[0].cells.len() <= 2);
    }

    /// `solve_defense` on a position with no opponent threat reports `NoThreat`
    /// (a single P1 stone and a single P2 stone far apart, win_length 6, tight
    /// radius — no forcing line for either side).
    #[test]
    fn solve_defense_no_threat() {
        let solver = StrixSolver::new();
        let pos = Position {
            win_length: 6,
            placement_radius: 2,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![
                Stone { coord: CoordW { q: 0, r: 0 }, player: Player::P1 },
                Stone { coord: CoordW { q: 10, r: 10 }, player: Player::P2 },
            ],
        };
        let out = solver.solve_defense(&pos, &limits_idtt(6, 20_000)).unwrap();
        assert_eq!(out.kind, DefenseKind::NoThreat, "no opponent threat expected");
        assert!(out.threat.is_none(), "NoThreat carries no threat");
        assert!(out.killers.is_empty(), "no threat → no killers");
        assert!(out.pair_anchors.is_empty(), "no threat → no pair anchors");
        assert!(out.best_delay.is_none(), "no threat → no best_delay");
    }

    /// `solve_defense` rejects a non-game-valid position (missing the (0,0,P1)
    /// origin). On native the `Err(JsError)` path can't be exercised directly
    /// (`JsError::new` is a wasm-bindgen imported function that panics on
    /// non-wasm targets), so this test verifies the gate the WASM layer relies
    /// on: `is_game_valid_board` returns false for the no-origin board. The
    /// actual `Err` return is covered by the wasm32-gated test below.
    #[test]
    fn solve_defense_rejects_invalid_position() {
        let pos = Position {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            // No (0,0,P1) origin — not game-valid.
            stones: vec![
                Stone { coord: CoordW { q: 3, r: 0 }, player: Player::P1 },
                Stone { coord: CoordW { q: 4, r: 0 }, player: Player::P1 },
                Stone { coord: CoordW { q: 8, r: 8 }, player: Player::P2 },
            ],
        };
        let solver_pos = convert_position(&pos).unwrap();
        assert!(
            !is_game_valid_board(&solver_pos.stones),
            "no-origin board must fail the game-validity gate"
        );
    }

    /// On wasm32, `solve_defense` returns a clear `Err` for a non-game-valid
    /// position (the gate fires before any `GameState::from_state` panic). Gated
    /// to wasm32 because `JsError::new()` is a wasm-bindgen imported function
    /// that panics on non-wasm targets.
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn solve_defense_returns_err_on_invalid_position() {
        let solver = StrixSolver::new();
        let pos = Position {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
            to_move: Player::P1,
            moves_remaining: 2,
            stones: vec![
                Stone { coord: CoordW { q: 3, r: 0 }, player: Player::P1 },
                Stone { coord: CoordW { q: 4, r: 0 }, player: Player::P1 },
                Stone { coord: CoordW { q: 8, r: 8 }, player: Player::P2 },
            ],
        };
        let res = solver.solve_defense(&pos, &limits_idtt(6, 5_000));
        assert!(res.is_err(), "non-game-valid position must return Err");
        let dbg = format!("{:?}", res.unwrap_err());
        assert!(
            dbg.contains("game-valid"),
            "error must mention game-validity, got {dbg:?}"
        );
    }

    /// `chunk_pv` chunks a flat pv into alternating turns: first attacker turn
    /// has `moves_remaining` cells, every subsequent turn has 2.
    #[test]
    fn chunk_pv_first_turn_has_moves_remaining() {
        // 5-cell pv, moves_remaining=2, attacker=P1
        let pv: Vec<Coord> = vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)];
        let turns = chunk_pv(&pv, 2, Player::P1);
        assert_eq!(turns.len(), 3, "5 cells -> 2 + 2 + 1 = 3 turns");
        assert_eq!(turns[0].cells.len(), 2, "turn 0: moves_remaining (2) cells");
        assert_eq!(turns[0].player, Player::P1, "turn 0: attacker");
        assert_eq!(turns[1].cells.len(), 2, "turn 1: 2 cells (defender)");
        assert_eq!(turns[1].player, Player::P2, "turn 1: defender");
        assert_eq!(turns[2].cells.len(), 1, "turn 2: 1 remaining cell (attacker)");
        assert_eq!(turns[2].player, Player::P1, "turn 2: attacker");
    }

    /// `chunk_pv` with an empty pv returns an empty vec.
    #[test]
    fn chunk_pv_empty_returns_empty() {
        let turns = chunk_pv(&[], 2, Player::P1);
        assert!(turns.is_empty());
    }

    /// `chunk_pv` with moves_remaining=1 puts 1 cell in the first turn.
    #[test]
    fn chunk_pv_single_placement_first_turn() {
        let pv: Vec<Coord> = vec![(0, 0), (1, 0), (2, 0), (3, 0)];
        let turns = chunk_pv(&pv, 1, Player::P2);
        assert_eq!(turns[0].cells.len(), 1, "turn 0: 1 cell (moves_remaining=1)");
        assert_eq!(turns[0].player, Player::P2, "turn 0: attacker (P2)");
        assert_eq!(turns[1].cells.len(), 2, "turn 1: 2 cells (defender)");
        assert_eq!(turns[1].player, Player::P1, "turn 1: defender (P1)");
        assert_eq!(turns[2].cells.len(), 1, "turn 2: 1 remaining cell (attacker)");
    }
}