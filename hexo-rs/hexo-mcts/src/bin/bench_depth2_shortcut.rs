//! Microbenchmark: depth-2 forced-win shortcut for HeXO MCTS root.
//!
//! Compares two extensions to the existing `gumbel_mcts.rs:122-141`
//! immediate-win shortcut:
//!
//!   - **candidate-only**: scan only the Gumbel-top-K candidate set
//!     (size m_actions) for both depth-1 and depth-2 forced wins.
//!   - **all-legal**: scan every legal move at the root for the same.
//!
//! Positions are extracted directly from a `games_*.bin` (HX04) emitted
//! by the running self-play loop, so the benchmark exercises realistic
//! mid/late-game distributions rather than random play (random play
//! almost never produces own_4 patterns, defeating the purpose).
//!
//! Usage:
//!     bench_depth2_shortcut <games.bin> [--max=N]
//!
//! Output: per-position-bucket wall-clock cost, hit-rate divergence, and
//! projected per-MCTS-call overhead.

use hexo_engine::{Coord, GameConfig, GameState, Player};
use hexo_rs::mcts::MCTSConfig;
use hexo_rs::mcts::gumbel_mcts::gumbel_mcts;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashMap as HashMap;
use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::time::Instant;

const M_ACTIONS: usize = 16;
const ITERS_PER_POSITION: usize = 5;
const MAGIC_HX04: &[u8; 4] = b"HX04";
const MAGIC_HX03: &[u8; 4] = b"HX03";

#[derive(Debug, Clone)]
struct Position {
    stones: Vec<(Coord, Player)>,
    current_player: Player,
    moves_remaining: u8,
    /// Legal-cell coordinates aligned with `policy` (sorted graph order).
    legal_coords: Vec<Coord>,
    /// Improved policy from MCTS (one entry per legal cell). Used as a proxy
    /// for the network's prior on each legal move when generating realistic
    /// Gumbel-top-K candidate sets — improved policy is more concentrated than
    /// raw network logits so this is an *optimistic upper bound* on
    /// candidate-only catch rate. Real raw-logit candidates would catch fewer.
    policy: Vec<f32>,
}

#[inline]
fn shortcut_candidate_only(
    game: &GameState,
    coords: &[Coord],
    candidate_indices: &[usize],
) -> Option<Coord> {
    let me = game.current_player();
    if me.is_none() {
        return None;
    }

    for &idx in candidate_indices {
        let coord = coords[idx];
        let mut g = game.clone();
        if g.apply_move(coord).is_ok() && g.is_terminal() && g.winner() == me {
            return Some(coord);
        }
    }

    // Depth-2 only meaningful at the start of a 2-placement turn.
    if game.moves_remaining_this_turn() < 2 {
        return None;
    }

    for &idx in candidate_indices {
        let m1 = coords[idx];
        let mut g1 = game.clone();
        if g1.apply_move(m1).is_err() {
            continue;
        }
        if g1.is_terminal() {
            continue;
        }
        if g1.current_player() != me {
            continue;
        }
        for m2 in g1.legal_moves() {
            let mut g2 = g1.clone();
            if g2.apply_move(m2).is_ok() && g2.is_terminal() && g2.winner() == me {
                return Some(m1);
            }
        }
    }

    None
}

/// Step 1 of the production optimisation: axis-pruned m2 search with the
/// engine's cached legal_moves_set as the legality guard. m2 is constrained
/// to cells on the same WIN_AXIS as m1 within (win_length - 1) cells.
#[inline]
fn shortcut_candidate_axis_pruned(
    game: &GameState,
    coords: &[Coord],
    candidate_indices: &[usize],
) -> Option<Coord> {
    use hexo_engine::types::WIN_AXES;
    let me = game.current_player();
    if me.is_none() {
        return None;
    }

    for &idx in candidate_indices {
        let coord = coords[idx];
        let mut g = game.clone();
        if g.apply_move(coord).is_ok() && g.is_terminal() && g.winner() == me {
            return Some(coord);
        }
    }

    if game.moves_remaining_this_turn() < 2 {
        return None;
    }

    let max_dist = (game.config().win_length - 1) as i32;

    for &idx in candidate_indices {
        let m1 = coords[idx];
        let mut g1 = game.clone();
        if g1.apply_move(m1).is_err() {
            continue;
        }
        if g1.is_terminal() {
            continue;
        }
        if g1.current_player() != me {
            continue;
        }
        let legal = g1.legal_moves_set();
        let (q1, r1) = m1;
        for &(dq, dr) in &WIN_AXES {
            for sign in [1i32, -1] {
                for d in 1..=max_dist {
                    let m2 = (q1 + sign * d * dq, r1 + sign * d * dr);
                    if !legal.contains(&m2) {
                        continue;
                    }
                    let mut g2 = g1.clone();
                    if g2.apply_move(m2).is_ok() && g2.is_terminal() && g2.winner() == me {
                        return Some(m1);
                    }
                }
            }
        }
    }
    None
}

/// Step 2 of the production optimisation: step 1 + a per-m1 pre-check on BOTH
/// Phase A (depth-1) and Phase B (depth-2). Counts own stones along each
/// WIN_AXIS within (win_length - 1) cells of m1; Phase A needs ≥ (win_length - 1),
/// Phase B needs ≥ (win_length - 2). Mirrors the production gumbel_mcts.rs.
#[inline]
fn shortcut_candidate_pre_checked(
    game: &GameState,
    coords: &[Coord],
    candidate_indices: &[usize],
) -> Option<Coord> {
    use hexo_engine::types::WIN_AXES;
    let me = game.current_player();
    let me_player = match me {
        Some(p) => p,
        None => return None,
    };

    let max_dist = (game.config().win_length - 1) as i32;
    let min_own_d1 = (game.config().win_length as usize).saturating_sub(1);
    let min_own_d2 = (game.config().win_length as usize).saturating_sub(2);
    let stones_ref = game.stones();

    // Phase A with pre-check
    for &idx in candidate_indices {
        let m1 = coords[idx];
        let (q1, r1) = m1;
        let mut axis_viable = false;
        for &(dq, dr) in &WIN_AXES {
            let mut count = 0usize;
            for d in 1..=max_dist {
                if stones_ref.get(&(q1 + d * dq, r1 + d * dr)) == Some(&me_player) {
                    count += 1;
                }
                if stones_ref.get(&(q1 - d * dq, r1 - d * dr)) == Some(&me_player) {
                    count += 1;
                }
            }
            if count >= min_own_d1 {
                axis_viable = true;
                break;
            }
        }
        if !axis_viable {
            continue;
        }
        let mut g = game.clone();
        if g.apply_move(m1).is_ok() && g.is_terminal() && g.winner() == me {
            return Some(m1);
        }
    }

    if game.moves_remaining_this_turn() < 2 {
        return None;
    }

    for &idx in candidate_indices {
        let m1 = coords[idx];
        let (q1, r1) = m1;
        let mut axis_viable = false;
        for &(dq, dr) in &WIN_AXES {
            let mut count = 0usize;
            for d in 1..=max_dist {
                if stones_ref.get(&(q1 + d * dq, r1 + d * dr)) == Some(&me_player) {
                    count += 1;
                }
                if stones_ref.get(&(q1 - d * dq, r1 - d * dr)) == Some(&me_player) {
                    count += 1;
                }
            }
            if count >= min_own_d2 {
                axis_viable = true;
                break;
            }
        }
        if !axis_viable {
            continue;
        }

        let mut g1 = game.clone();
        if g1.apply_move(m1).is_err() {
            continue;
        }
        if g1.is_terminal() {
            continue;
        }
        if g1.current_player() != me {
            continue;
        }
        let legal = g1.legal_moves_set();
        for &(dq, dr) in &WIN_AXES {
            for sign in [1i32, -1] {
                for d in 1..=max_dist {
                    let m2 = (q1 + sign * d * dq, r1 + sign * d * dr);
                    if !legal.contains(&m2) {
                        continue;
                    }
                    let mut g2 = g1.clone();
                    if g2.apply_move(m2).is_ok() && g2.is_terminal() && g2.winner() == me {
                        return Some(m1);
                    }
                }
            }
        }
    }
    None
}

#[inline]
fn shortcut_all_legal(game: &GameState) -> Option<Coord> {
    let me = game.current_player();
    if me.is_none() {
        return None;
    }
    let legal = game.legal_moves();

    for &m1 in &legal {
        let mut g = game.clone();
        if g.apply_move(m1).is_ok() && g.is_terminal() && g.winner() == me {
            return Some(m1);
        }
    }

    if game.moves_remaining_this_turn() < 2 {
        return None;
    }

    for &m1 in &legal {
        let mut g1 = game.clone();
        if g1.apply_move(m1).is_err() {
            continue;
        }
        if g1.is_terminal() {
            continue;
        }
        if g1.current_player() != me {
            continue;
        }
        for m2 in g1.legal_moves() {
            let mut g2 = g1.clone();
            if g2.apply_move(m2).is_ok() && g2.is_terminal() && g2.winner() == me {
                return Some(m1);
            }
        }
    }

    None
}

fn random_candidates(n: usize, m: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let m = m.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    indices.truncate(m);
    indices
}

/// Gumbel-top-K sampling: returns the indices of the top `m` by
/// `logit + Gumbel(0, 1)`. Mirrors the production sampler in
/// `mcts/halving.rs` but is independent so this bench runs without
/// the gumbel_mcts feature stack.
fn gumbel_top_k(logits: &[f32], m: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let m = m.min(logits.len());
    let mut scores: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            // Gumbel(0, 1) = -log(-log(U)) for U ~ Uniform(0, 1).
            let u: f32 = rng.random_range(1e-30_f32..1.0_f32);
            let g = -(-u.ln()).ln();
            (i, l + g)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(m);
    scores.into_iter().map(|(i, _)| i).collect()
}

/// Convert a probability distribution to logits via log(p + eps).
/// MCTS-improved policy is used as a stand-in for raw network logits.
/// This is OPTIMISTIC for candidate-only catch rate (improved policy is more
/// concentrated on terminal-leading moves than raw logits would be).
fn policy_to_logits(policy: &[f32]) -> Vec<f32> {
    const EPS: f32 = 1e-9;
    policy.iter().map(|&p| (p + EPS).ln()).collect()
}

/// Map a candidate index in `policy` (legal-cell graph order) to a coord-space
/// index (sorted lexicographic by GameState::legal_moves()). Returns None if
/// the policy-coord doesn't match any sorted coord (shouldn't happen).
fn map_legal_indices(
    policy_legal: &[Coord],
    sorted_legal: &[Coord],
    candidate_in_policy: &[usize],
) -> Vec<usize> {
    let mut result = Vec::with_capacity(candidate_in_policy.len());
    for &i in candidate_in_policy {
        let coord = policy_legal[i];
        if let Some(pos) = sorted_legal.iter().position(|&c| c == coord) {
            result.push(pos);
        }
    }
    result
}

// --- HX04 / HX03 reader ---

fn read_u16_le(buf: &[u8], offset: &mut usize) -> u16 {
    let v = u16::from_le_bytes(buf[*offset..*offset + 2].try_into().unwrap());
    *offset += 2;
    v
}

fn read_u32_le(buf: &[u8], offset: &mut usize) -> u32 {
    let v = u32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    v
}

fn read_i16_le(buf: &[u8], offset: &mut usize) -> i16 {
    let v = i16::from_le_bytes(buf[*offset..*offset + 2].try_into().unwrap());
    *offset += 2;
    v
}

fn read_f32_le(buf: &[u8], offset: &mut usize) -> f32 {
    let v = f32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    v
}

fn parse_example(buf: &[u8], offset: &mut usize, has_sample_weight: bool) -> Position {
    let num_nodes = read_u16_le(buf, offset) as usize;
    let num_edges = read_u16_le(buf, offset) as usize;
    let num_legal = read_u16_le(buf, offset) as usize;
    let has_ea = buf[*offset] != 0;
    *offset += 1;
    *offset += 4; // value f32
    if has_sample_weight {
        *offset += 4; // sample_weight f32
    }

    // Features: num_nodes * 8 f32
    let features_offset = *offset;
    *offset += num_nodes * 8 * 4;

    // Edges: u16 src + u16 dst
    *offset += num_edges * 2 * 2;

    // Edge attr (axis graph): num_edges * 5 * 4 f32
    if has_ea {
        *offset += num_edges * 5 * 4;
    }

    // Masks: legal + stone (u8 each)
    let legal_mask_offset = *offset;
    *offset += num_nodes; // legal mask
    *offset += num_nodes; // stone mask

    // Coords (i16 LE, q,r per node)
    let coords_offset = *offset;
    *offset += num_nodes * 2 * 2;

    // Policy: num_legal f32
    let policy_offset = *offset;
    *offset += num_legal * 4;

    // Decode features into stones + global player/moves features.
    // Also collect the legal_coords (in graph order) to align with policy.
    let mut stones: Vec<(Coord, Player)> = Vec::new();
    let mut legal_coords: Vec<Coord> = Vec::with_capacity(num_legal);
    let mut player_feat: f32 = 0.0;
    let mut moves_feat: f32 = 0.0;
    for i in 0..num_nodes {
        let feat_base = features_offset + i * 8 * 4;
        let mut local_offset = feat_base;
        let p1 = read_f32_le(buf, &mut local_offset);
        let p2 = read_f32_le(buf, &mut local_offset);
        let _empty = read_f32_le(buf, &mut local_offset);
        let pf = read_f32_le(buf, &mut local_offset);
        let mf = read_f32_le(buf, &mut local_offset);
        if i == 0 {
            player_feat = pf;
            moves_feat = mf;
        }

        let coord_offset = coords_offset + i * 2 * 2;
        let mut local_co = coord_offset;
        let q = read_i16_le(buf, &mut local_co) as i32;
        let r = read_i16_le(buf, &mut local_co) as i32;

        if p1 > 0.5 {
            stones.push(((q, r), Player::P1));
        } else if p2 > 0.5 {
            stones.push(((q, r), Player::P2));
        }

        let is_legal = buf[legal_mask_offset + i] != 0;
        if is_legal {
            legal_coords.push((q, r));
        }
    }

    let mut policy: Vec<f32> = Vec::with_capacity(num_legal);
    for k in 0..num_legal {
        let mut o = policy_offset + k * 4;
        policy.push(read_f32_le(buf, &mut o));
    }

    let current_player = if player_feat > 0.0 { Player::P1 } else { Player::P2 };
    let moves_remaining = (moves_feat * 2.0).round() as u8;

    Position {
        stones,
        current_player,
        moves_remaining,
        legal_coords,
        policy,
    }
}

fn read_positions(path: &str, max_positions: usize) -> Vec<Position> {
    let mut file = File::open(path).expect("open games.bin");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read");

    let mut positions = Vec::new();
    let mut cursor = 0;
    while cursor + 8 < buf.len() && positions.len() < max_positions {
        let magic = &buf[cursor..cursor + 4];
        let has_sample_weight = if magic == MAGIC_HX04 {
            true
        } else if magic == MAGIC_HX03 {
            false
        } else {
            panic!("unknown magic at offset {cursor}: {magic:?}");
        };
        cursor += 4;
        let record_size = read_u32_le(&buf, &mut cursor) as usize;
        let record_end = cursor + record_size;
        let num_examples = read_u32_le(&buf, &mut cursor);
        cursor += 1; // winner
        cursor += 4; // move_count
        for _ in 0..num_examples {
            if positions.len() >= max_positions {
                break;
            }
            let pos = parse_example(&buf, &mut cursor, has_sample_weight);
            positions.push(pos);
        }
        cursor = record_end;
    }
    positions
}

#[derive(Default, Debug)]
struct BucketStats {
    label: &'static str,
    n: usize,
    cand_total_ns: u128,
    axis_total_ns: u128,
    prechk_total_ns: u128,
    all_total_ns: u128,
    iters_per_pos: u128,
    cand_d1: usize,
    cand_d2: usize,
    all_d1: usize,
    all_d2: usize,
    missed_by_cand: usize,
    legal_sum: usize,
}

fn classify_hit(game: &GameState, c: Coord) -> u8 {
    let mut g = game.clone();
    g.apply_move(c).ok();
    if g.is_terminal() {
        1
    } else {
        2
    }
}

#[derive(Copy, Clone, Debug)]
enum CandidateMode {
    Random,
    Gumbel,
}

fn run_bench(positions: &[Position], config: GameConfig, mode: CandidateMode) -> Vec<BucketStats> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut early = BucketStats {
        label: "early(<30 stones)",
        ..Default::default()
    };
    let mut mid = BucketStats {
        label: "mid(30-79 stones)",
        ..Default::default()
    };
    let mut late = BucketStats {
        label: "late(>=80 stones)",
        ..Default::default()
    };
    let mut all = BucketStats {
        label: "ALL",
        ..Default::default()
    };

    for pos in positions {
        let game = GameState::from_state(
            &pos.stones,
            pos.current_player,
            pos.moves_remaining,
            config,
        );
        let coords = game.legal_moves();
        if coords.is_empty() {
            continue;
        }
        let cand: Vec<usize> = match mode {
            CandidateMode::Random => random_candidates(coords.len(), M_ACTIONS, &mut rng),
            CandidateMode::Gumbel => {
                // policy is aligned with pos.legal_coords (graph order); Gumbel-top-K
                // picks indices into that, which we then remap to the sorted-legal
                // index space the shortcut iterates over.
                let logits = policy_to_logits(&pos.policy);
                let cand_in_policy = gumbel_top_k(&logits, M_ACTIONS, &mut rng);
                map_legal_indices(&pos.legal_coords, &coords, &cand_in_policy)
            }
        };

        // Time candidate-only (baseline: pre-axis-pruning, pre-pre-check)
        let t = Instant::now();
        let mut last_cand: Option<Coord> = None;
        for _ in 0..ITERS_PER_POSITION {
            last_cand = shortcut_candidate_only(&game, &coords, &cand);
            black_box(&last_cand);
        }
        let cand_ns = t.elapsed().as_nanos();

        // Time axis-pruned candidate (step-1 optimisation: bounded m2 + HashSet guard)
        let t = Instant::now();
        let mut last_axis: Option<Coord> = None;
        for _ in 0..ITERS_PER_POSITION {
            last_axis = shortcut_candidate_axis_pruned(&game, &coords, &cand);
            black_box(&last_axis);
        }
        let axis_ns = t.elapsed().as_nanos();

        // Time pre-checked candidate (step-2 optimisation: + per-m1 own-stone pre-check)
        let t = Instant::now();
        let mut last_prechk: Option<Coord> = None;
        for _ in 0..ITERS_PER_POSITION {
            last_prechk = shortcut_candidate_pre_checked(&game, &coords, &cand);
            black_box(&last_prechk);
        }
        let prechk_ns = t.elapsed().as_nanos();

        // Time all-legal
        let t = Instant::now();
        let mut last_all: Option<Coord> = None;
        for _ in 0..ITERS_PER_POSITION {
            last_all = shortcut_all_legal(&game);
            black_box(&last_all);
        }
        let all_ns = t.elapsed().as_nanos();

        // Correctness: all candidate variants must return identical results.
        if last_cand != last_axis || last_cand != last_prechk {
            panic!(
                "variant divergence: cand={last_cand:?} axis={last_axis:?} prechk={last_prechk:?} on pos with {} stones",
                pos.stones.len()
            );
        }

        let bucket = if pos.stones.len() < 30 {
            &mut early
        } else if pos.stones.len() < 80 {
            &mut mid
        } else {
            &mut late
        };

        for s in [bucket, &mut all] {
            s.n += 1;
            s.cand_total_ns += cand_ns;
            s.axis_total_ns += axis_ns;
            s.prechk_total_ns += prechk_ns;
            s.all_total_ns += all_ns;
            s.iters_per_pos = ITERS_PER_POSITION as u128;
            s.legal_sum += coords.len();
            if let Some(c) = last_cand {
                if classify_hit(&game, c) == 1 {
                    s.cand_d1 += 1;
                } else {
                    s.cand_d2 += 1;
                }
            }
            if let Some(c) = last_all {
                if classify_hit(&game, c) == 1 {
                    s.all_d1 += 1;
                } else {
                    s.all_d2 += 1;
                }
            }
            if last_all.is_some() && last_cand.is_none() {
                s.missed_by_cand += 1;
            }
        }
    }

    vec![early, mid, late, all]
}

/// Run full gumbel_mcts on each position with a uniform-prior dummy eval.
/// Designed for `perf record` / flamegraph profiling of the MCTS body
/// (selection / simulation / virtual-loss / backup / softmax) — eval is
/// near-free so the profile shows CPU work in the tree search, not in the
/// learned policy network.
fn run_mcts_bench(
    positions: &[Position],
    config: GameConfig,
    n_simulations: u32,
    virtual_loss: f64,
) {
    let mcts_config = MCTSConfig {
        n_simulations,
        m_actions: M_ACTIONS,
        c_visit: 50,
        c_scale: 1.0,
        virtual_loss,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let mut dummy_eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let mut logits_list = Vec::with_capacity(states.len());
        let mut values = Vec::with_capacity(states.len());
        for s in states {
            let legal = s.legal_moves();
            let mut m = HashMap::with_capacity_and_hasher(legal.len(), Default::default());
            for c in legal {
                m.insert(c, 0.0);
            }
            logits_list.push(m);
            values.push(0.0);
        }
        (logits_list, values)
    };

    let mut total_calls: u64 = 0;
    let mut skipped_terminal: u64 = 0;
    let mut skipped_shortcut: u64 = 0;
    let start = Instant::now();
    for pos in positions {
        let game = GameState::from_state(&pos.stones, pos.current_player, pos.moves_remaining, config);
        if game.is_terminal() {
            skipped_terminal += 1;
            continue;
        }
        let result = match gumbel_mcts(&game, &mcts_config, &mut rng, None, &mut dummy_eval) {
            Ok(r) => r,
            Err(_) => continue,
        };
        // Detect whether the call exited via the one-hot shortcut path.
        let one_hot = result.improved_policy.iter().any(|&p| p > 0.999);
        if one_hot {
            skipped_shortcut += 1;
        }
        black_box(&result);
        total_calls += 1;
    }
    let elapsed = start.elapsed();
    let n_body = total_calls.saturating_sub(skipped_shortcut);
    let per_call_us = elapsed.as_micros() as f64 / total_calls.max(1) as f64;
    let per_body_us = if n_body > 0 {
        // Subtract estimated shortcut-path time using the bench's prechk
        // measurement (~3 μs); approximate, but small relative to body cost.
        let body_total_us = elapsed.as_micros() as f64 - (skipped_shortcut as f64 * 3.5);
        body_total_us / n_body as f64
    } else {
        0.0
    };

    println!();
    println!("MCTS bench results:");
    println!("  positions input:        {}", positions.len());
    println!("  skipped (terminal):     {}", skipped_terminal);
    println!("  calls completed:        {}", total_calls);
    println!("  - via shortcut (~one-hot): {}", skipped_shortcut);
    println!("  - via MCTS body:        {}", n_body);
    println!("  wall-clock total:       {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  μs / call (all):        {:.1}", per_call_us);
    println!("  μs / call (body only):  {:.1}  (approx; shortcut subtracted)", per_body_us);
    println!("  n_simulations:          {}", n_simulations);
    println!("  virtual_loss:           {}", virtual_loss);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <games.bin> [--max=N] [--radius=R] [--win=W] [--mcts] [--sims=N] [--vl=F]", args[0]);
        std::process::exit(1);
    }
    let path = &args[1];
    let mut max_positions: usize = 50_000;
    let mut radius: i32 = 2;
    let mut win_length: u8 = 6;
    let mut mode = CandidateMode::Gumbel;
    let mut do_mcts = false;
    let mut n_simulations: u32 = 128;
    let mut virtual_loss: f64 = 1.0;
    for a in &args[2..] {
        if let Some(v) = a.strip_prefix("--max=") {
            max_positions = v.parse().expect("--max value");
        } else if let Some(v) = a.strip_prefix("--radius=") {
            radius = v.parse().expect("--radius value");
        } else if let Some(v) = a.strip_prefix("--win=") {
            win_length = v.parse().expect("--win value");
        } else if let Some(v) = a.strip_prefix("--candidates=") {
            mode = match v {
                "random" => CandidateMode::Random,
                "gumbel" => CandidateMode::Gumbel,
                other => panic!("--candidates must be random|gumbel (got {other})"),
            };
        } else if a == "--mcts" {
            do_mcts = true;
        } else if let Some(v) = a.strip_prefix("--sims=") {
            n_simulations = v.parse().expect("--sims value");
        } else if let Some(v) = a.strip_prefix("--vl=") {
            virtual_loss = v.parse().expect("--vl value");
        }
    }
    let _ = mode; // unused in mcts mode

    println!(
        "Loading positions from {path} (max={max_positions}, radius={radius}, win={win_length}, candidates={mode:?})"
    );
    let positions = read_positions(path, max_positions);
    println!("Loaded {} positions", positions.len());

    let config = GameConfig {
        win_length,
        placement_radius: radius,
        max_moves: 200,
    };

    if do_mcts {
        run_mcts_bench(&positions, config, n_simulations, virtual_loss);
        return;
    }

    let buckets = run_bench(&positions, config, mode);

    println!();
    println!(
        "{:<22} {:>6} {:>7} {:>11} {:>11} {:>11} {:>11} {:>6} {:>6}",
        "bucket", "n", "L_mean", "cand_ns", "axis_ns", "prechk_ns", "all_ns", "spd_a", "spd_p",
    );
    for s in &buckets {
        if s.n == 0 {
            continue;
        }
        let denom = (s.n as u128 * s.iters_per_pos) as f64;
        let cand_mean = s.cand_total_ns as f64 / denom;
        let axis_mean = s.axis_total_ns as f64 / denom;
        let prechk_mean = s.prechk_total_ns as f64 / denom;
        let all_mean = s.all_total_ns as f64 / denom;
        let l_mean = s.legal_sum as f64 / s.n as f64;
        let spd_axis = cand_mean / axis_mean.max(1.0);
        let spd_prechk = cand_mean / prechk_mean.max(1.0);
        println!(
            "{:<22} {:>6} {:>7.1} {:>11.0} {:>11.0} {:>11.0} {:>11.0} {:>6.2}x {:>5.2}x",
            s.label,
            s.n,
            l_mean,
            cand_mean,
            axis_mean,
            prechk_mean,
            all_mean,
            spd_axis,
            spd_prechk,
        );
    }
    println!();
    println!("spd_a = cand/axis speedup (step 1: axis-prune + HashSet guard)");
    println!("spd_p = cand/prechk speedup (step 1 + step 2: + per-m1 pre-check)");
}
