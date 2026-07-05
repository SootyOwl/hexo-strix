//! Target-independent core: position JSON parsing, validation, search, serialization.
//!
//! Everything here is plain Rust (no wasm-bindgen types) so it is natively testable;
//! `lib.rs` wraps it for the wasm FFI boundary.

use hexo_infer::{InferError, InferModel};
use hexo_rs::hexo_engine::types::{Coord, Player};
use hexo_rs::hexo_engine::{GameConfig, GameState};
use hexo_rs::mcts::gumbel_mcts::gumbel_mcts;
use hexo_rs::mcts::MCTSConfig;
use rand::SeedableRng;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct PositionJson {
    pub config: ConfigJson,
    pub stones: Vec<StoneJson>,
    pub to_move: u8,
    pub moves_remaining: u8,
}

#[derive(Deserialize)]
pub struct ConfigJson {
    pub win_length: u8,
    pub placement_radius: i32,
    pub max_moves: u32,
}

#[derive(Deserialize)]
pub struct StoneJson {
    pub q: i32,
    pub r: i32,
    pub player: u8,
}

fn err(msg: impl Into<String>) -> InferError {
    InferError::UnsupportedConfig(msg.into())
}

fn player_of(n: u8) -> Result<Player, InferError> {
    match n {
        1 => Ok(Player::P1),
        2 => Ok(Player::P2),
        other => Err(err(format!("player must be 1 or 2, got {other}"))),
    }
}

fn hex_distance_origin(q: i32, r: i32) -> i32 {
    // axial hex distance from (0,0)
    (q.abs() + r.abs() + (q + r).abs()) / 2
}

/// Validate every `from_state` precondition + game-rule sanity, returning `Err`
/// instead of letting `from_state`/`Board::place` panic across the FFI boundary.
/// Returns the constructed (non-terminal, playable) GameState.
pub fn validate_and_build(pos: &PositionJson, model: &InferModel) -> Result<GameState, InferError> {
    let c = &pos.config;
    if c.win_length < 2 {
        return Err(err("win_length must be >= 2"));
    }
    if c.placement_radius < 1 {
        return Err(err("placement_radius must be >= 1"));
    }
    if c.max_moves < 1 {
        return Err(err("max_moves must be >= 1"));
    }
    if !(pos.moves_remaining == 1 || pos.moves_remaining == 2) {
        return Err(err(format!("moves_remaining must be 1 or 2, got {}", pos.moves_remaining)));
    }
    let to_move = player_of(pos.to_move)?;

    // Training-config cross-check: a position larger than the model was trained on is
    // out-of-distribution garbage, not just a weaker eval. The training game_config
    // travels in the safetensors metadata.
    if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&model.config().metadata_json) {
        if let Some(gc) = meta.get("game_config").and_then(|s| s.as_str()) {
            if let Ok(tgc) = serde_json::from_str::<serde_json::Value>(gc) {
                if let Some(twl) = tgc.get("win_length").and_then(|v| v.as_u64()) {
                    if u64::from(c.win_length) > twl {
                        return Err(err(format!(
                            "position win_length {} exceeds the model's training win_length {twl}",
                            c.win_length
                        )));
                    }
                }
                if let Some(tr) = tgc.get("placement_radius").and_then(|v| v.as_i64()) {
                    if i64::from(c.placement_radius) > tr {
                        return Err(err(format!(
                            "position placement_radius {} exceeds the model's training radius {tr}",
                            c.placement_radius
                        )));
                    }
                }
            }
        }
    }

    // Stones: player range, duplicates, radius, origin contract.
    let mut stones: Vec<(Coord, Player)> = Vec::with_capacity(pos.stones.len());
    let mut seen = std::collections::HashSet::with_capacity(pos.stones.len());
    let mut origin_ok = false;
    for s in &pos.stones {
        let p = player_of(s.player)?;
        let coord: Coord = (s.q, s.r);
        if !seen.insert(coord) {
            return Err(err(format!("duplicate stone at ({}, {})", s.q, s.r)));
        }
        if coord == (0, 0) {
            // from_state's placement loop skips (0,0) ONLY for P1; origin-as-P2
            // collides with the pre-seeded origin stone and would panic.
            if p != Player::P1 {
                return Err(err("origin (0,0) must be P1 (the engine pre-seeds it)"));
            }
            origin_ok = true;
        }
        if hex_distance_origin(s.q, s.r) > c.placement_radius {
            return Err(err(format!(
                "stone ({}, {}) is outside placement_radius {}",
                s.q, s.r, c.placement_radius
            )));
        }
        stones.push((coord, p));
    }
    // Origin contract (PINNED): require (0,0,P1) present — from_state's
    // move_count = stones.len()-1 assumes the origin is counted in the list.
    if !origin_ok {
        return Err(err("origin stone (0,0,P1) required (the engine pre-seeds P1 at origin)"));
    }
    // Draw exhaustion.
    if stones.len() as u32 - 1 >= c.max_moves {
        return Err(err(format!(
            "position has {} moves but max_moves is {} (game over by draw rule)",
            stones.len() - 1,
            c.max_moves
        )));
    }

    let game = GameState::from_state(
        &stones,
        to_move,
        pos.moves_remaining,
        GameConfig {
            win_length: c.win_length,
            placement_radius: c.placement_radius,
            max_moves: c.max_moves,
        },
    );

    // Terminal detection: from_state runs NO win check (winner hard-coded None) —
    // recompute via the engine's has_winner.
    if let Some(w) = game.has_winner() {
        return Err(err(format!("position is terminal: {w:?} has a winning line")));
    }
    if game.legal_moves().is_empty() {
        return Err(err("position has no legal moves"));
    }
    Ok(game)
}

/// Production search config. NEVER bare `..Default::default()`: Default zeroes
/// c_visit/c_scale (σ(Q) collapses to 0 → silently weak bot) and leaves gumbel
/// noise ON (self-play exploration). Matches serving/analysis.py:143-145.
pub fn search_config(sims: u32, m_actions: u32) -> MCTSConfig {
    MCTSConfig {
        n_simulations: sims,
        m_actions: m_actions as usize,
        c_visit: 50,
        c_scale: 1.0,
        disable_gumbel_noise: true,
        // Remaining fields are correct as zero: root_dirichlet_alpha 0,
        // root_dirichlet_fraction 0, virtual_loss 0, forced_candidate_capture_k 0
        // (self-play-only p2 feature). If MCTSConfig gains a field, audit its
        // Default before relying on this.
        ..Default::default()
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

fn policy_json(coords: &[Coord], probs: &[f64]) -> serde_json::Value {
    // Descending by p, ties broken by coord ascending (documented contract).
    let mut entries: Vec<(Coord, f64)> = coords.iter().copied().zip(probs.iter().copied()).collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    serde_json::Value::Array(
        entries
            .into_iter()
            .map(|((q, r), p)| serde_json::json!({"q": q, "r": r, "p": p}))
            .collect(),
    )
}

/// Full search: parse, validate, run gumbel MCTS, serialize the response.
/// `value` = network value-head output for the root (to_move's perspective).
pub fn best_move_impl(
    model: &InferModel,
    position_json: &str,
    sims: u32,
    m_actions: u32,
    seed: u64,
) -> Result<String, InferError> {
    let pos: PositionJson = serde_json::from_str(position_json)
        .map_err(|e| err(format!("bad position JSON: {e}")))?;
    let game = validate_and_build(&pos, model)?;

    // Root value: one direct forward (MCTSResult has no root-value field).
    let (_, values) = model.eval_states(std::slice::from_ref(&game));
    let value = values[0];

    let config = search_config(sims, m_actions);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut eval = |states: &[GameState]| model.eval_states(states);
    let result = gumbel_mcts(&game, &config, &mut rng, None, &mut eval)?;

    Ok(serde_json::json!({
        "move": {"q": result.action.0, "r": result.action.1},
        "value": value,
        "policy": policy_json(&result.coords, &result.improved_policy),
    })
    .to_string())
}

/// Raw net eval (no search): softmaxed prior + value.
pub fn evaluate_impl(model: &InferModel, position_json: &str) -> Result<String, InferError> {
    let pos: PositionJson = serde_json::from_str(position_json)
        .map_err(|e| err(format!("bad position JSON: {e}")))?;
    let game = validate_and_build(&pos, model)?;
    let legal = game.legal_moves();
    let (maps, values) = model.eval_states(std::slice::from_ref(&game));
    let logits: Vec<f64> = legal.iter().map(|c| maps[0][c]).collect();
    let probs = softmax(&logits);
    Ok(serde_json::json!({
        "value": values[0],
        "policy": policy_json(&legal, &probs),
    })
    .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> InferModel {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../hexo-infer/tests/fixtures/tiny.safetensors");
        InferModel::from_safetensors(&std::fs::read(p).unwrap()).unwrap()
    }

    fn pos(stones: &str, to_move: u8, mr: u8) -> String {
        format!(
            r#"{{"config":{{"win_length":6,"placement_radius":4,"max_moves":300}},"stones":[{stones}],"to_move":{to_move},"moves_remaining":{mr}}}"#
        )
    }

    const ORIGIN: &str = r#"{"q":0,"r":0,"player":1}"#;

    #[test]
    fn search_config_is_not_degenerate() {
        let c = search_config(64, 16);
        assert_eq!(c.c_visit, 50);
        assert_eq!(c.c_scale, 1.0);
        assert!(c.disable_gumbel_noise);
    }

    #[test]
    fn valid_position_returns_legal_move() {
        let model = tiny_model();
        let p = pos(&format!(r#"{ORIGIN},{{"q":1,"r":0,"player":2}}"#), 2, 1);
        let out = best_move_impl(&model, &p, 32, 8, 42).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(v["move"]["q"].is_i64() && v["move"]["r"].is_i64());
        let psum: f64 = v["policy"].as_array().unwrap().iter().map(|e| e["p"].as_f64().unwrap()).sum();
        assert!((psum - 1.0).abs() < 1e-6, "best_move policy sums to {psum}");
    }

    #[test]
    fn deterministic_and_seed_inert() {
        let model = tiny_model();
        let p = pos(ORIGIN, 2, 2);
        let a = best_move_impl(&model, &p, 32, 8, 42).unwrap();
        assert_eq!(a, best_move_impl(&model, &p, 32, 8, 42).unwrap(), "same seed repeats");
        assert_eq!(a, best_move_impl(&model, &p, 32, 8, 99).unwrap(), "seed is inert (noise off)");
    }

    #[test]
    fn evaluate_matches_fixture_value_and_sums() {
        let model = tiny_model();
        let fx: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../hexo-infer/tests/fixtures/tiny_initial.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let p = pos(ORIGIN, 2, 2);
        let out: serde_json::Value = serde_json::from_str(&evaluate_impl(&model, &p).unwrap()).unwrap();
        let expected = fx["expected"]["value"].as_f64().unwrap();
        assert!(
            (out["value"].as_f64().unwrap() - expected).abs() < 1e-4,
            "evaluate value {} vs fixture {expected}",
            out["value"]
        );
        let psum: f64 = out["policy"].as_array().unwrap().iter().map(|e| e["p"].as_f64().unwrap()).sum();
        assert!((psum - 1.0).abs() < 1e-6);
        // Descending order.
        let ps: Vec<f64> = out["policy"].as_array().unwrap().iter().map(|e| e["p"].as_f64().unwrap()).collect();
        assert!(ps.windows(2).all(|w| w[0] >= w[1]), "policy must be sorted descending");
    }

    #[test]
    fn validation_rejects_not_panics() {
        let model = tiny_model();
        // (rule, position JSON, expected error substring)
        let cases = [
            ("mr=0", pos(ORIGIN, 2, 0), "moves_remaining"),
            ("mr=3", pos(ORIGIN, 2, 3), "moves_remaining"),
            ("to_move=3", pos(ORIGIN, 3, 2), "player"),
            ("player=0", pos(r#"{"q":0,"r":0,"player":1},{"q":1,"r":0,"player":0}"#, 2, 2), "player"),
            ("dup", pos(r#"{"q":0,"r":0,"player":1},{"q":1,"r":0,"player":2},{"q":1,"r":0,"player":2}"#, 1, 2), "duplicate"),
            ("origin-as-P2", pos(r#"{"q":0,"r":0,"player":2}"#, 1, 2), "origin"),
            ("origin-missing", pos(r#"{"q":1,"r":0,"player":2}"#, 1, 2), "origin"),
            ("out-of-radius", pos(&format!(r#"{ORIGIN},{{"q":9,"r":0,"player":2}}"#), 1, 2), "radius"),
            ("bad-config", r#"{"config":{"win_length":1,"placement_radius":4,"max_moves":300},"stones":[{"q":0,"r":0,"player":1}],"to_move":2,"moves_remaining":2}"#.to_string(), "win_length"),
            ("garbage-json", "{not json".to_string(), "JSON"),
        ];
        for (name, p, want) in cases {
            let e = best_move_impl(&model, &p, 8, 4, 0).unwrap_err().to_string();
            assert!(e.contains(want), "{name}: error {e:?} missing {want:?}");
        }
    }

    #[test]
    fn terminal_position_rejected_via_win_recompute() {
        let model = tiny_model();
        // P2 has 6-in-a-row along r-axis (win_length 6). from_state won't flag it;
        // has_winner must.
        let stones = r#"{"q":0,"r":0,"player":1},
            {"q":1,"r":0,"player":2},{"q":1,"r":1,"player":2},{"q":1,"r":2,"player":2},
            {"q":1,"r":-1,"player":2},{"q":1,"r":-2,"player":2},{"q":1,"r":-3,"player":2},
            {"q":0,"r":1,"player":1},{"q":0,"r":2,"player":1},{"q":0,"r":-1,"player":1},{"q":2,"r":0,"player":1}"#;
        let p = pos(stones, 1, 2);
        let e = best_move_impl(&model, &p, 8, 4, 0).unwrap_err().to_string();
        assert!(e.contains("terminal"), "expected terminal rejection, got {e:?}");
    }

    #[test]
    fn reconstruction_equivalence_initial_and_midturn() {
        use hexo_rs::axis_graph::game_to_axis_graph_raw_opts;
        // Case 1: initial position, fresh vs from_state.
        let cfg = GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 };
        let fresh = GameState::with_config(cfg);
        let recon = GameState::from_state(&[((0, 0), Player::P1)], Player::P2, 2, cfg);
        for (a, b, label) in [(&fresh, &recon, "initial")] {
            let ga = game_to_axis_graph_raw_opts(a, true, true, true);
            let gb = game_to_axis_graph_raw_opts(b, true, true, true);
            assert_eq!(ga.features, gb.features, "{label}: features");
            assert_eq!(ga.edge_src, gb.edge_src, "{label}: edge_src");
            assert_eq!(ga.edge_attr, gb.edge_attr, "{label}: edge_attr");
            assert_eq!(ga.legal_mask, gb.legal_mask, "{label}: legal_mask");
            assert_eq!(ga.stone_mask, gb.stone_mask, "{label}: stone_mask");
        }
        // Case 2: mid-turn (9 plies -> moves_remaining==1) — the only coverage that
        // from_state sets turn/move_count right for states the wasm API receives.
        let mut game = GameState::with_config(cfg);
        for _ in 0..9 {
            let mv = game.legal_moves()[3];
            game.apply_move(mv).unwrap();
        }
        assert_eq!(game.moves_remaining_this_turn(), 1);
        let recon = GameState::from_state(
            &game.placed_stones(),
            game.current_player().unwrap(),
            game.moves_remaining_this_turn(),
            cfg,
        );
        let ga = game_to_axis_graph_raw_opts(&game, true, true, true);
        let gb = game_to_axis_graph_raw_opts(&recon, true, true, true);
        assert_eq!(ga.features, gb.features, "midturn: features");
        assert_eq!(ga.edge_src, gb.edge_src, "midturn: edge_src");
        assert_eq!(ga.edge_attr, gb.edge_attr, "midturn: edge_attr");
        assert_eq!(ga.legal_mask, gb.legal_mask, "midturn: legal_mask");
        assert_eq!(ga.stone_mask, gb.stone_mask, "midturn: stone_mask");
    }
}