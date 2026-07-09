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
    solve_from_position, solve_wide_from_position, SolverEngine, SolverPosition,
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
#[derive(Debug)]
pub struct SolveOutcome {
    pub kind: SolveKind,
    pub depth: u8,
    pub nodes: u64,
    pub time_ms: f64,
    pub pv: Vec<Turn>,
}

// --- Task 6 stub types (surface complete; implementation deferred) ---

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefenseKind {
    /// A single placement that refutes the threat (or ends the game).
    Killer = 0,
    /// A pair of placements that jointly refute the threat.
    Pair = 1,
    /// Max-delay fallback: the surviving candidate with the longest re-proven PV.
    Delay = 2,
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
    pub kind: DefenseKind,
    pub threat_pv: Vec<CoordW>,
    pub killers: Vec<CoordW>,
    pub pair_anchors: Vec<PairAnchor>,
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

    /// Task 6 fills this in; the types are defined so the WASM surface is stable.
    pub fn solve_defense(
        &self,
        _position: &Position,
        _limits: &SolverLimits,
    ) -> Result<DefenseOutcome, JsError> {
        Err(JsError::new("solve_defense not yet implemented (Task 6)"))
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

    /// `solve_defense` is a Task 6 stub: must return an error, not panic.
    /// Gated to wasm32 because JsError::new() is a wasm-bindgen imported
    /// function that panics on non-wasm targets.
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn solve_defense_returns_not_implemented() {
        let solver = StrixSolver::new();
        let pos = empty_position();
        let res = solver.solve_defense(&pos, &limits_idtt(6, 5_000));
        assert!(res.is_err(), "solve_defense must return Err (Task 6 stub)");
        // JsError doesn't impl Display in wasm-bindgen 0.2; inspect via Debug.
        let dbg = format!("{:?}", res.unwrap_err());
        assert!(
            dbg.contains("not yet implemented"),
            "error must mention not-yet-implemented, got {dbg:?}"
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