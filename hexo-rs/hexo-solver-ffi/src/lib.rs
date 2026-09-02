//! C ABI JSON wrapper for the standalone forcing solver.
//!
//! This crate intentionally exposes a tiny, stable boundary for JVM callers:
//! UTF-8 JSON in, heap-allocated UTF-8 JSON out. Kotlin/JNA must release returned
//! strings with [`hexo_free_string`].
//!
//! Honesty contract (mirrors the wasm surface): `hexo_solve_defense_json` never
//! reports `no_threat` when the threat check merely ran out of budget — that
//! case is `budget_exceeded`. Defensive responses distinguish globally verified
//! refutations (`pair_anchors`) from exact immediate covers whose deeper status
//! is open (`tactical_pairs`), expose initiative-taking pairs
//! (`counter_threats`), and list budget-starved candidates (`unresolved`).

use hexo_engine::types::Player;
use hexo_solver::forcing::Outcome;
use hexo_solver::{
    DefenseVerdict, SolverEngine, SolverPosition, is_game_valid_board,
    solve_defense_verdict_from_position, solve_from_position, solve_wide_from_position,
};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct SolveRequest {
    win_length: u8,
    placement_radius: i32,
    max_moves: u32,
    to_move: u8,
    moves_remaining: u8,
    depth_cap: u8,
    node_budget: u64,
    #[serde(default)]
    engine: SolverEngineRequest,
    #[serde(default)]
    wide: bool,
    #[serde(default)]
    time_limit_ms: Option<u64>,
    stones: Vec<StoneRequest>,
}

/// Solver engine selected by the JSON API. Lowercase names match the public
/// engine names used by the other wrappers; omission remains backward
/// compatible and selects the production IDTT engine.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum SolverEngineRequest {
    #[default]
    Idtt,
    Pns,
    Dfpn,
    Pdspn,
}

impl From<SolverEngineRequest> for SolverEngine {
    fn from(engine: SolverEngineRequest) -> Self {
        match engine {
            SolverEngineRequest::Idtt => SolverEngine::Idtt,
            SolverEngineRequest::Pns => SolverEngine::Pns,
            SolverEngineRequest::Dfpn => SolverEngine::Dfpn,
            SolverEngineRequest::Pdspn => SolverEngine::Pdspn,
        }
    }
}

#[derive(Debug, Deserialize)]
struct StoneRequest {
    q: i32,
    r: i32,
    player: u8,
}

#[derive(Debug, Serialize)]
struct SolveResponse {
    kind: &'static str,
    depth: u8,
    pv: Vec<TurnResponse>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct CoordResponse {
    q: i32,
    r: i32,
}

#[derive(Debug, Serialize)]
struct TurnResponse {
    turn: u32,
    player: u8,
    cells: Vec<CoordResponse>,
}

#[derive(Debug, Serialize)]
struct DefenseResponse {
    kind: &'static str,
    threat: Option<ThreatResponse>,
    killers: Vec<CoordResponse>,
    pair_anchors: Vec<PairAnchorResponse>,
    /// Pair anchors that refute by taking the initiative (the opponent must
    /// answer the mover's new forcing threat first).
    counter_threats: Vec<PairAnchorResponse>,
    /// Exact minimum covers of the immediate attack whose deeper whole-line
    /// verification was inconclusive: they stop the current line but are NOT
    /// global refutations.
    tactical_pairs: Vec<PairAnchorResponse>,
    /// Candidate replies whose deeper re-check exhausted the work budget.
    unresolved: Vec<CoordResponse>,
    /// Whether the wide (threat + quiet-builder) generator was used.
    wide: bool,
    best_delay: Option<CoordResponse>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ThreatResponse {
    depth: u8,
    pv: Vec<TurnResponse>,
}

#[derive(Debug, Serialize)]
struct PairAnchorResponse {
    first: CoordResponse,
    second: CoordResponse,
}

/// Solve a position from a JSON request.
///
/// # Safety
///
/// `input` must either be null or point to a valid NUL-terminated C string for
/// the duration of the call. The returned pointer must be freed exactly once
/// with [`hexo_free_string`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_solve_json(input: *const c_char) -> *mut c_char {
    let response = match catch_unwind(AssertUnwindSafe(|| solve_json(input))) {
        Ok(response) => response,
        Err(panic) => error_json(&format!("internal panic: {}", panic_message(panic))),
    };
    into_c_string(response)
}

/// Run defensive analysis for the side to move from a JSON request.
///
/// Response `kind` is one of `no_threat`, `budget_exceeded`, `threat_found`, or
/// `error`. `budget_exceeded` means the threat check ran out of work budget —
/// retry with a larger budget; it is never a safety verdict.

/// # Safety
///
/// `input` must either be null or point to a valid NUL-terminated C string for
/// the duration of the call. The returned pointer must be freed exactly once
/// with [`hexo_free_string`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_solve_defense_json(input: *const c_char) -> *mut c_char {
    let response = match catch_unwind(AssertUnwindSafe(|| solve_defense_json(input))) {
        Ok(response) => response,
        Err(panic) => defense_error_json(&format!("internal panic: {}", panic_message(panic))),
    };
    into_c_string(response)
}

/// Free a string returned by [`hexo_solve_json`] or [`hexo_solve_defense_json`].
///
/// # Safety
///
/// `ptr` must be null or a pointer previously returned by [`hexo_solve_json`]
/// that has not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

fn solve_json(input: *const c_char) -> String {
    let request = match parse_request(input) {
        Ok(request) => request,
        Err(err) => return error_json(&err),
    };

    match solve_request(request) {
        Ok(response) => to_json(&response),
        Err(err) => error_json(&err),
    }
}

fn solve_defense_json(input: *const c_char) -> String {
    let request = match parse_request(input) {
        Ok(request) => request,
        Err(err) => return defense_error_json(&err),
    };

    match defense_request(request) {
        Ok(response) => to_json(&response),
        Err(err) => defense_error_json(&err),
    }
}

fn parse_request(input: *const c_char) -> Result<SolveRequest, String> {
    if input.is_null() {
        return Err("input pointer is null".to_owned());
    }

    let input = unsafe { CStr::from_ptr(input) }
        .to_str()
        .map_err(|_| "input is not valid utf-8".to_owned())?;

    serde_json::from_str::<SolveRequest>(input)
        .map_err(|err| format!("invalid request json: {err}"))
}

fn solve_request(request: SolveRequest) -> Result<SolveResponse, String> {
    let wide = request.wide;
    let engine = request.engine.into();
    let depth_cap = request.depth_cap;
    let node_budget = request.node_budget;
    let moves_remaining = request.moves_remaining;
    let attacker = request.to_move;
    let position = request.into_position()?;

    let outcome = if wide {
        solve_wide_from_position(&position, engine, depth_cap, node_budget)
    } else {
        solve_from_position(&position, engine, depth_cap, node_budget)
    };

    Ok(match outcome {
        Outcome::Win(win) => SolveResponse {
            kind: "win",
            depth: win.depth,
            pv: chunk_pv(win.pv, moves_remaining, attacker),
            error: None,
        },
        Outcome::No => SolveResponse {
            kind: "no",
            depth: 0,
            pv: Vec::new(),
            error: None,
        },
        Outcome::BudgetExceeded => SolveResponse {
            kind: "budget_exceeded",
            depth: 0,
            pv: Vec::new(),
            error: None,
        },
    })
}

fn defense_request(request: SolveRequest) -> Result<DefenseResponse, String> {
    let wide = request.wide;
    let depth_cap = request.depth_cap;
    let node_budget = request.node_budget;
    let time_limit = Duration::from_millis(request.time_limit_ms.unwrap_or(10_000));
    let threat_attacker = opponent_u8(request.to_move)?;
    let position = request.into_position()?;

    if !is_game_valid_board(&position.stones) {
        return Err(
            "solve_defense requires a game-valid position: the (0,0,P1) origin stone must be present, with no duplicate or contradictory coords"
                .to_owned(),
        );
    }
    // Defense builds a GameState internally, which materializes the full move
    // neighborhood for the given placement radius. Keep the same native guard
    // as the wasm surface so untrusted callers cannot force an oversized
    // allocation.
    if !(1..=64).contains(&position.placement_radius) {
        return Err("solve_defense requires 1 <= placement_radius <= 64".to_owned());
    }

    // The verdict API is load-bearing here: a starved threat check reports
    // `budget_exceeded`, never the dangerous `no_threat` (the legacy
    // Option-returning wrapper conflated the two).
    match solve_defense_verdict_from_position(&position, depth_cap, node_budget, time_limit, wide) {
        DefenseVerdict::NoThreat => Ok(DefenseResponse {
            kind: "no_threat",
            threat: None,
            killers: Vec::new(),
            pair_anchors: Vec::new(),
            counter_threats: Vec::new(),
            tactical_pairs: Vec::new(),
            unresolved: Vec::new(),
            wide,
            best_delay: None,
            error: None,
        }),
        DefenseVerdict::BudgetExceeded => Ok(DefenseResponse {
            kind: "budget_exceeded",
            threat: None,
            killers: Vec::new(),
            pair_anchors: Vec::new(),
            counter_threats: Vec::new(),
            tactical_pairs: Vec::new(),
            unresolved: Vec::new(),
            wide,
            best_delay: None,
            error: None,
        }),
        DefenseVerdict::Threat(analysis) => Ok(DefenseResponse {
            kind: "threat_found",
            threat: Some(ThreatResponse {
                depth: analysis.threat_depth,
                pv: chunk_pv(analysis.threat_pv, 2, threat_attacker),
            }),
            killers: coords_response(analysis.killers),
            pair_anchors: pairs_response(analysis.pair_anchors),
            counter_threats: pairs_response(analysis.counter_threats),
            tactical_pairs: pairs_response(analysis.tactical_pairs),
            unresolved: coords_response(analysis.unresolved),
            wide,
            best_delay: analysis.best_delay.map(coord_response),
            error: None,
        }),
    }
}

impl SolveRequest {
    fn into_position(self) -> Result<SolverPosition, String> {
        let to_move = parse_player(self.to_move, "to_move")?;
        let stones = self
            .stones
            .into_iter()
            .map(|stone| {
                Ok((
                    (stone.q, stone.r),
                    parse_player(stone.player, "stone.player")?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(SolverPosition {
            win_length: self.win_length,
            placement_radius: self.placement_radius,
            max_moves: self.max_moves,
            to_move,
            moves_remaining: self.moves_remaining,
            stones,
        })
    }
}

fn parse_player(value: u8, field: &str) -> Result<Player, String> {
    match value {
        1 => Ok(Player::P1),
        2 => Ok(Player::P2),
        _ => Err(format!("{field} must be 1 or 2")),
    }
}

fn coord_response((q, r): (i32, i32)) -> CoordResponse {
    CoordResponse { q, r }
}

fn coords_response(coords: Vec<(i32, i32)>) -> Vec<CoordResponse> {
    coords.into_iter().map(coord_response).collect()
}

fn pairs_response(pairs: Vec<((i32, i32), (i32, i32))>) -> Vec<PairAnchorResponse> {
    pairs
        .into_iter()
        .map(|(first, second)| PairAnchorResponse {
            first: coord_response(first),
            second: coord_response(second),
        })
        .collect()
}

/// Chunk the flat solver PV into wasm-compatible turn records.
///
/// Turn 0 belongs to the attacker and has `moves_remaining` cells. Every later
/// turn alternates player and has up to 2 cells.
fn chunk_pv(pv: Vec<(i32, i32)>, moves_remaining: u8, attacker: u8) -> Vec<TurnResponse> {
    if pv.is_empty() {
        return Vec::new();
    }

    let mut turns = Vec::new();
    let mut idx = 0usize;
    let mut turn = 0u32;
    let mut player = attacker;

    let first_len = (moves_remaining as usize).min(pv.len());
    turns.push(TurnResponse {
        turn,
        player,
        cells: coords_response(pv[idx..idx + first_len].to_vec()),
    });
    idx += first_len;
    turn += 1;
    player = opponent_u8(player).expect("validated player");

    while idx < pv.len() {
        let len = 2.min(pv.len() - idx);
        turns.push(TurnResponse {
            turn,
            player,
            cells: coords_response(pv[idx..idx + len].to_vec()),
        });
        idx += len;
        turn += 1;
        player = opponent_u8(player).expect("validated player");
    }

    turns
}

fn opponent_u8(player: u8) -> Result<u8, String> {
    match player {
        1 => Ok(2),
        2 => Ok(1),
        _ => Err("to_move must be 1 or 2".to_owned()),
    }
}

fn error_json(message: &str) -> String {
    to_json(&SolveResponse {
        kind: "error",
        depth: 0,
        pv: Vec::new(),
        error: Some(message.to_owned()),
    })
}

fn defense_error_json(message: &str) -> String {
    to_json(&DefenseResponse {
        kind: "error",
        threat: None,
        killers: Vec::new(),
        pair_anchors: Vec::new(),
        counter_threats: Vec::new(),
        tactical_pairs: Vec::new(),
        unresolved: Vec::new(),
        wide: false,
        best_delay: None,
        error: Some(message.to_owned()),
    })
}

fn to_json<T: Serialize>(response: &T) -> String {
    serde_json::to_string(response).unwrap_or_else(|err| {
        format!(r#"{{"kind":"error","error":"response serialization failed: {err}"}}"#)
    })
}

fn into_c_string(response: String) -> *mut c_char {
    let sanitized = response.replace('\0', "\\u0000");
    CString::new(sanitized)
        .expect("NUL bytes were sanitized")
        .into_raw()
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn call_json(raw: &str) -> String {
        let input = CString::new(raw).expect("test json has no NUL bytes");
        let ptr = unsafe { hexo_solve_json(input.as_ptr()) };
        assert!(!ptr.is_null());
        let out = unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .expect("responses are utf-8")
            .to_owned();
        unsafe { hexo_free_string(ptr) };
        out
    }

    fn call_defense_json(raw: &str) -> String {
        let input = CString::new(raw).expect("test json has no NUL bytes");
        let ptr = unsafe { hexo_solve_defense_json(input.as_ptr()) };
        assert!(!ptr.is_null());
        let out = unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .expect("responses are utf-8")
            .to_owned();
        unsafe { hexo_free_string(ptr) };
        out
    }

    const WIN_REQUEST: &str = r#"{"win_length":6,"placement_radius":8,"max_moves":400,"to_move":1,"moves_remaining":2,"depth_cap":8,"node_budget":20000,"stones":[[0,0,1],[1,0,1],[2,0,1],[3,0,1],[5,5,2]]}"#;

    const QIET_REQUEST_PREFIX: &str = r#"{"win_length":6,"placement_radius":8,"max_moves":400,"to_move":1,"moves_remaining":2,"depth_cap":12,"node_budget":100000,"wide":true,"stones":"#;

    #[test]
    fn solve_reports_win_with_turn_chunks() {
        let response: serde_json::Value = serde_json::from_str(&call_json(WIN_REQUEST)).unwrap();
        assert_eq!(response["kind"], "win");
        assert_eq!(response["error"], serde_json::Value::Null);
        let turns = response["pv"].as_array().unwrap();
        assert!(!turns.is_empty());
        assert_eq!(turns[0]["player"], 1);
        assert!(turns[0]["cells"].as_array().unwrap().len() <= 2);
    }

    #[test]
    fn invalid_request_is_a_clean_error() {
        let response: serde_json::Value = serde_json::from_str(&call_json("{}")).unwrap();
        assert_eq!(response["kind"], "error");
        assert!(response["error"].as_str().unwrap().contains("win_length"));
    }

    #[test]
    fn request_accepts_every_solver_engine_and_defaults_to_idtt() {
        let request = CString::new(WIN_REQUEST).unwrap();
        let parsed = parse_request(request.as_ptr()).unwrap();
        assert!(matches!(parsed.engine, SolverEngineRequest::Idtt));

        for (name, expected) in [
            ("idtt", SolverEngineRequest::Idtt),
            ("pns", SolverEngineRequest::Pns),
            ("dfpn", SolverEngineRequest::Dfpn),
            ("pdspn", SolverEngineRequest::Pdspn),
        ] {
            let raw = WIN_REQUEST.replacen(
                r#""depth_cap":8"#,
                &format!(r#""engine":"{name}","depth_cap":8"#),
                1,
            );
            let request = CString::new(raw).unwrap();
            let parsed = parse_request(request.as_ptr()).unwrap();
            assert!(std::mem::discriminant(&parsed.engine) == std::mem::discriminant(&expected));
        }
    }

    #[test]
    fn unknown_solver_engine_is_a_clean_error() {
        let raw =
            WIN_REQUEST.replacen(r#""depth_cap":8"#, r#""engine":"unknown","depth_cap":8"#, 1);
        let response: serde_json::Value = serde_json::from_str(&call_json(&raw)).unwrap();
        assert_eq!(response["kind"], "error");
        assert!(
            response["error"]
                .as_str()
                .unwrap()
                .contains("unknown variant")
        );
    }

    #[test]
    fn defense_reports_honest_budget_exceeded() {
        // The qiet position has a real threat whose sighting needs more than
        // one node. The legacy wrapper conflated that starved check with
        // NoThreat; the FFI must say budget_exceeded instead.
        let replay: serde_json::Value = serde_json::from_str(include_str!(
            "../../../scripts/fixtures/forcing_puzzles/qietby7_17_line.json"
        ))
        .unwrap();
        let stones: Vec<[i32; 3]> = replay["moves"]
            .as_array()
            .unwrap()
            .iter()
            .take(31)
            .map(|m| {
                [
                    m[0].as_i64().unwrap() as i32,
                    m[1].as_i64().unwrap() as i32,
                    if m[2] == "P1" { 1 } else { 2 },
                ]
            })
            .collect();
        let request = serde_json::json!({
            "win_length": 6,
            "placement_radius": 8,
            "max_moves": 400,
            "to_move": 1,
            "moves_remaining": 2,
            "depth_cap": 40,
            "node_budget": 1,
            "stones": stones,
        });
        let response: serde_json::Value =
            serde_json::from_str(&call_defense_json(&request.to_string())).unwrap();
        assert_eq!(response["kind"], "budget_exceeded");
        assert_eq!(response["error"], serde_json::Value::Null);
    }

    #[test]
    fn defense_exposes_initiative_evidence() {
        let replay: serde_json::Value = serde_json::from_str(include_str!(
            "../../../scripts/fixtures/forcing_puzzles/qietby7_17_line.json"
        ))
        .unwrap();
        let stones: Vec<[i32; 3]> = replay["moves"]
            .as_array()
            .unwrap()
            .iter()
            .take(31)
            .map(|m| {
                [
                    m[0].as_i64().unwrap() as i32,
                    m[1].as_i64().unwrap() as i32,
                    if m[2] == "P1" { 1 } else { 2 },
                ]
            })
            .collect();
        let mut request = String::from(QIET_REQUEST_PREFIX);
        request.push_str(&serde_json::to_string(&stones).unwrap());
        request.push('}');
        let response: serde_json::Value =
            serde_json::from_str(&call_defense_json(&request)).unwrap();
        assert_eq!(response["kind"], "threat_found");
        assert_eq!(response["wide"], true);
        let counter = response["counter_threats"].as_array().unwrap();
        assert!(counter.iter().any(|pair| pair["first"]["q"] == 2
            && pair["first"]["r"] == 0
            && pair["second"]["q"] == 3
            && pair["second"]["r"] == 0));
        // The honesty fields are always present, even when empty.
        assert!(response["tactical_pairs"].is_array());
        assert!(response["unresolved"].is_array());
    }
}
