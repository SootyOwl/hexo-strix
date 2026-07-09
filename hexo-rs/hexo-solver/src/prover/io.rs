//! JSON position / line / report formats for the deep-prover CLI (serde).
//!
//! Positions use the established snapshot format already emitted elsewhere in the
//! project; lines are ordered full game logs from the seed. All network fetching
//! and URL rendering is handled by the thin Python wrapper, never here.

use hexo_engine::types::{Coord, Player};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A prover verdict. `WIN`/`NO` are definitive; `UNVERIFIED` means the hybrid
/// left open branches; `BUDGET_EXCEEDED` means a node budget or wall deadline was
/// hit before resolution. A verdict is never fabricated: an unresolved search
/// reports `BUDGET_EXCEEDED`, an unverified hybrid reports `UNVERIFIED`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Verdict {
    #[serde(rename = "WIN")]
    Win,
    #[serde(rename = "NO")]
    No,
    #[serde(rename = "UNVERIFIED")]
    Unverified,
    #[serde(rename = "BUDGET_EXCEEDED")]
    BudgetExceeded,
}

impl Verdict {
    /// A definitive proof result — the only kind that ends a portfolio race.
    pub fn is_definitive(self) -> bool {
        matches!(self, Verdict::Win | Verdict::No)
    }
}

fn parse_player(s: &str) -> Result<Player, String> {
    match s {
        "P1" => Ok(Player::P1),
        "P2" => Ok(Player::P2),
        other => Err(format!("bad player tag {other:?} (want \"P1\" or \"P2\")")),
    }
}

fn player_tag(p: Player) -> &'static str {
    match p {
        Player::P1 => "P1",
        Player::P2 => "P2",
    }
}

/// Game config for a position; defaults to full hexo (6, 8, 300).
#[derive(Clone, Copy, Debug)]
pub struct PosConfig {
    pub win_length: u8,
    pub placement_radius: i32,
    pub max_moves: u32,
}

impl Default for PosConfig {
    fn default() -> Self {
        PosConfig { win_length: 6, placement_radius: 8, max_moves: 300 }
    }
}

/// A parsed position: the stones on the board, whose forced win we test, and the
/// placements the attacker has left this turn.
#[derive(Clone, Debug)]
pub struct Position {
    pub stones: Vec<(Coord, Player)>,
    pub attacker: Player,
    pub placements_remaining: u8,
    pub config: PosConfig,
}

/// A full game log from the seed, players tagged in play order.
#[derive(Clone, Debug)]
pub struct Line {
    pub moves: Vec<(Coord, Player)>,
}

fn parse_move(v: &serde_json::Value) -> Result<(Coord, Player), String> {
    let arr = v.as_array().ok_or("move must be a [q, r, player] array")?;
    if arr.len() != 3 {
        return Err(format!("move must have 3 elements, got {}", arr.len()));
    }
    let q = arr[0].as_i64().ok_or("q must be an integer")? as i32;
    let r = arr[1].as_i64().ok_or("r must be an integer")? as i32;
    let p = parse_player(arr[2].as_str().ok_or("player must be a string")?)?;
    Ok(((q, r), p))
}

impl Position {
    pub fn from_json(text: &str) -> Result<Position, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let stones = v["stones"]
            .as_array()
            .ok_or("`stones` must be an array")?
            .iter()
            .map(parse_move)
            .collect::<Result<Vec<_>, _>>()?;
        let attacker = parse_player(
            v["attacker"].as_str().ok_or("`attacker` must be a string")?,
        )?;
        let placements_remaining =
            v["placements_remaining"].as_u64().ok_or("`placements_remaining` must be an int")? as u8;
        let config = if v.get("config").is_some() && !v["config"].is_null() {
            let c = &v["config"];
            let d = PosConfig::default();
            PosConfig {
                win_length: c["win_length"].as_u64().map(|x| x as u8).unwrap_or(d.win_length),
                placement_radius: c["placement_radius"]
                    .as_i64()
                    .map(|x| x as i32)
                    .unwrap_or(d.placement_radius),
                max_moves: c["max_moves"].as_u64().map(|x| x as u32).unwrap_or(d.max_moves),
            }
        } else {
            PosConfig::default()
        };
        Ok(Position { stones, attacker, placements_remaining, config })
    }

    pub fn load(path: &Path) -> Result<Position, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        Position::from_json(&text)
            .map_err(|e| format!("parsing {}: {e}", path.display()))
    }
}

impl Line {
    pub fn from_json(text: &str) -> Result<Line, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let moves = v["moves"]
            .as_array()
            .ok_or("`moves` must be an array")?
            .iter()
            .map(parse_move)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Line { moves })
    }

    pub fn load(path: &Path) -> Result<Line, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        Line::from_json(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
    }
}

/// Per-search counters. Absent/not-applicable fields serialize as 0.
///
/// Meanings are driver-specific:
/// - **dfpn / pdspn:** `nodes` = MID expansions, `tt_hits` = TT probe hits,
///   `tt_bytes` = TT size, `leaf_solves` = level-2 PN searches (pdspn only).
/// - **idtt:** only `elapsed_s` (the production solver exposes no node count).
/// - **hybrid:** `nodes` = total node visits; `leaf_solves` = nodes CLOSED by a
///   kernel leaf solve; `line_steps` = nodes CLOSED by following a spine candidate
///   (these two are directly comparable to the Python baseline's "solver-proven
///   leaf branches" / "line-steps"); `tt_hits` = raw leaf-solve CALL count (cost).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Stats {
    pub nodes: u64,
    pub tt_hits: u64,
    pub tt_bytes: u64,
    pub elapsed_s: f64,
    pub leaf_solves: u64,
    pub line_steps: u64,
}

/// One driver's outcome in a portfolio race.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RaceEntry {
    pub driver: String,
    pub verdict: Verdict,
    pub elapsed_s: f64,
    pub nodes: u64,
}

/// An open branch the hybrid could not close.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnverifiedBranch {
    pub path: Vec<Coord>,
    pub note: String,
}

/// The report written to `--out` and summarized to stdout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub verdict: Verdict,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<u8>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pv: Vec<Coord>,
    pub driver: String,
    pub width: String,
    pub stats: Stats,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub race: Vec<RaceEntry>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unverified: Vec<UnverifiedBranch>,
}

impl Report {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Report serializes")
    }
}

/// Render a player tag for diagnostics.
pub fn tag(p: Player) -> &'static str {
    player_tag(p)
}
