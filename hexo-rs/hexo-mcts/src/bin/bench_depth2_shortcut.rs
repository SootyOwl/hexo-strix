//! Microbenchmark: depth-2 forced-win shortcut for HeXO MCTS root.
//!
//! Compares two extensions to the existing `gumbel_mcts.rs:122-141`
//! immediate-win shortcut:
//!
//!   - **candidate-only**: scan only the Gumbel-top-K candidate set
//!     (size m_actions) for both depth-1 and depth-2 forced wins.
//!   - **all-legal**: scan every legal move at the root for the same.
//!
//! Positions are extracted directly from a `games_*.bin` emitted by the
//! running self-play loop (current wire format: HX07 board-state records;
//! legacy HX03/HX04 serialized-graph records also supported, auto-detected
//! by magic), so the benchmark exercises realistic mid/late-game
//! distributions rather than random play (random play almost never produces
//! own_4 patterns, defeating the purpose). Falls back to a synthetic
//! corpus when no real `games.bin` is found on disk.
//!
//! Usage:
//!     bench_depth2_shortcut [<games.bin>|synthetic] [--corpus=PATH] [--max=N] [--depth-cap=N] [--node-budget=N]
//!
//! Output: per-position-bucket wall-clock cost, hit-rate divergence, and
//! projected per-MCTS-call overhead.

use hexo_engine::{Coord, GameConfig, GameState, Player};
use hexo_rs::mcts::MCTSConfig;
use hexo_rs::mcts::forcing;
use hexo_rs::mcts::gumbel_mcts::gumbel_mcts;
use hexo_rs::mcts::gumbel_mcts::{SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET};
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

// --- HX07 reader (board-state records; current self-play wire format) ---
//
// HX07 envelope: [4B magic "HX07"][4B LE record_size][4B LE num_examples]
// [1B winner i8][4B LE move_count u32][1B has_window u8]
// [opt 12B window stats IFF has_window==1].
// Each example: [2B num_stones u16][1B current_player u8 (0=P1,1=P2)]
// [1B moves_remaining u8][2B num_legal u16][4B value f32][4B sample_weight f32]
// [num_stones*(2B q i16, 2B r i16, 1B player u8)][num_legal*4B policy f32].
// See `self_play.rs`'s `RECORD_MAGIC_HX07` docs (~line 1169) for the
// authoritative spec this mirrors. Only `stones`/`current_player`/
// `moves_remaining` are needed for the forcing-solve bench; value/
// sample_weight/policy are skipped over (cursor advanced past them).
const MAGIC_HX07: &[u8; 4] = b"HX07";

fn read_i8(buf: &[u8], offset: &mut usize) -> i8 {
    let v = buf[*offset] as i8;
    *offset += 1;
    v
}

fn parse_example_hx07(buf: &[u8], offset: &mut usize) -> Position {
    let num_stones = read_u16_le(buf, offset) as usize;
    let current_player_byte = buf[*offset];
    *offset += 1;
    let moves_remaining = buf[*offset];
    *offset += 1;
    let num_legal = read_u16_le(buf, offset) as usize;
    *offset += 4; // value f32
    *offset += 4; // sample_weight f32

    let mut stones: Vec<(Coord, Player)> = Vec::with_capacity(num_stones);
    for _ in 0..num_stones {
        let q = read_i16_le(buf, offset) as i32;
        let r = read_i16_le(buf, offset) as i32;
        let p = if buf[*offset] == 0 { Player::P1 } else { Player::P2 };
        *offset += 1;
        stones.push(((q, r), p));
    }

    // Policy floats aren't needed for the forcing-solve bench; skip.
    *offset += num_legal * 4;

    let current_player = if current_player_byte == 0 { Player::P1 } else { Player::P2 };

    Position {
        stones,
        current_player,
        moves_remaining,
        legal_coords: Vec::new(),
        policy: Vec::new(),
    }
}

fn read_positions_hx07(path: &str, max_positions: usize) -> Vec<Position> {
    let mut file = File::open(path).expect("open HX07 games.bin");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read");

    let mut positions = Vec::new();
    let mut cursor = 0;
    while cursor + 8 < buf.len() && positions.len() < max_positions {
        let magic = &buf[cursor..cursor + 4];
        if magic != MAGIC_HX07 {
            panic!("unknown magic at offset {cursor}: {magic:?} (expected HX07)");
        }
        cursor += 4;
        let record_size = read_u32_le(&buf, &mut cursor) as usize;
        let record_end = cursor + record_size;
        let num_examples = read_u32_le(&buf, &mut cursor);
        let _winner = read_i8(&buf, &mut cursor);
        cursor += 4; // move_count u32
        let has_window = buf[cursor] != 0;
        cursor += 1;
        if has_window {
            cursor += 12; // in_window(2) + gated(2) + mass_removed(4) + acted_deficit(4)
        }
        let mut examples_parsed = 0u32;
        for _ in 0..num_examples {
            if positions.len() >= max_positions {
                break;
            }
            let pos = parse_example_hx07(&buf, &mut cursor);
            positions.push(pos);
            examples_parsed += 1;
        }
        // If we parsed every example in this record (didn't bail early on
        // max_positions), the cursor must land exactly on record_end -- the
        // same exact-byte-consumption invariant `self_play.rs`'s HX07
        // round-trip test checks. A mismatch here would mean the per-example
        // offset math (parse_example_hx07) has drifted from the writer
        // (write_example_hx07), which -- unlike a magic mismatch -- would
        // NOT otherwise crash the loop (cursor is unconditionally reset to
        // record_end below), so it could silently corrupt multi-example
        // records without this check.
        if examples_parsed == num_examples {
            assert_eq!(
                cursor, record_end,
                "HX07 record parsed to offset {cursor} but expected record_end {record_end} \
                 (num_examples={num_examples}) -- parse_example_hx07 offset math has drifted \
                 from the HX07 writer"
            );
        }
        cursor = record_end;
    }
    positions
}

/// Sniff the first 4 bytes of a file to detect its magic, if it exists and
/// is readable. Returns `None` if the path doesn't exist or is too short.
fn sniff_magic(path: &str) -> Option<[u8; 4]> {
    let mut file = File::open(path).ok()?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).ok()?;
    Some(magic)
}

// --- Synthetic corpus (fallback when no real games.bin is on disk) ---
//
// The bench originally read positions out of a real self-play `games_*.bin`
// (HX03/HX04, a serialized-graph wire format). The self-play writer has since
// moved to HX07 (board-state records; see `self_play.rs`'s
// `RECORD_MAGIC_HX07` docs and `read_positions_hx07` above), and `main()`
// prefers a real HX07 corpus (default: `DEFAULT_HX07_CORPUS`) when present on
// disk. This synthetic generator remains as a fallback for environments where
// no real corpus is available (e.g. a fresh checkout with no self-play run
// yet): it plays games directly with `hexo_engine`, no NN involved, using a
// biased-random policy (mostly random, occasionally preferring cells that
// extend the mover's own axis runs) so that own-3/own-4 patterns -- and
// occasionally deeper forced-win shapes -- actually occur, unlike pure
// uniform random play. Only `stones` / `current_player` / `moves_remaining`
// are populated (`legal_coords` / `policy` are left empty): the
// `forcing::solve` measurement pass needs only board state, not a network
// prior.

/// Score a candidate cell by the most own stones already on some WIN_AXIS
/// through it within `win_length - 1` cells (a cheap proxy for "how much
/// this move builds toward a line"), used to bias synthetic self-play.
fn own_axis_score(game: &GameState, me: Player, c: Coord) -> i32 {
    use hexo_engine::types::WIN_AXES;
    let (q, r) = c;
    let max_dist = (game.config().win_length - 1) as i32;
    let stones_ref = game.stones();
    let mut best = 0i32;
    for &(dq, dr) in &WIN_AXES {
        let mut count = 0i32;
        for d in 1..=max_dist {
            if stones_ref.get(&(q + d * dq, r + d * dr)) == Some(&me) {
                count += 1;
            }
            if stones_ref.get(&(q - d * dq, r - d * dr)) == Some(&me) {
                count += 1;
            }
        }
        best = best.max(count);
    }
    best
}

/// Pick a move: with probability `1 - bias` uniformly at random over legal
/// moves; otherwise greedily (uniform tie-break) among the legal moves with
/// the highest `own_axis_score`. `bias` close to 1 produces more line-heavy,
/// tactical positions (own-3/4s, occasional forced wins); `bias` close to 0
/// produces quiet/scattered positions, similar to genuine early self-play.
fn choose_synthetic_move(game: &GameState, bias: f64, rng: &mut ChaCha8Rng) -> Coord {
    let legal = game.legal_moves();
    debug_assert!(!legal.is_empty());
    if rng.random::<f64>() >= bias {
        return legal[rng.random_range(0..legal.len())];
    }
    let me = game
        .current_player()
        .expect("non-terminal game guaranteed Some(current_player)");
    let mut best_score = i32::MIN;
    let mut best: Vec<Coord> = Vec::new();
    for &c in &legal {
        let score = own_axis_score(game, me, c);
        if score > best_score {
            best_score = score;
            best.clear();
            best.push(c);
        } else if score == best_score {
            best.push(c);
        }
    }
    best[rng.random_range(0..best.len())]
}

/// Play one synthetic self-play-like game to terminal/draw, recording a
/// `Position` snapshot at the start of every turn (`moves_remaining_this_turn()
/// == 2`) -- the only point `forcing::solve` is ever invoked in production
/// (`gumbel_mcts.rs`'s Step 3.5 guard). `legal_coords`/`policy` are left
/// empty; only board state is needed downstream.
fn play_synthetic_game(config: GameConfig, bias: f64, rng: &mut ChaCha8Rng) -> Vec<Position> {
    let mut game = GameState::with_config(config);
    let mut out = Vec::new();
    loop {
        if game.is_terminal() {
            break;
        }
        if game.moves_remaining_this_turn() == 2 {
            out.push(Position {
                stones: game.placed_stones(),
                current_player: game
                    .current_player()
                    .expect("non-terminal game guaranteed Some(current_player)"),
                moves_remaining: game.moves_remaining_this_turn(),
                legal_coords: Vec::new(),
                policy: Vec::new(),
            });
        }
        let mv = choose_synthetic_move(&game, bias, rng);
        if game.apply_move(mv).is_err() {
            break; // shouldn't happen: mv is drawn from game.legal_moves()
        }
    }
    out
}

/// Generate a synthetic corpus by playing biased-random games until
/// `max_positions` turn-start snapshots are collected (or `n_games` games
/// have been played, whichever comes first). Mixes three bias levels across
/// games (quiet / moderate / tactical) for a spread of position types.
fn generate_synthetic_corpus(
    config: GameConfig,
    n_games: usize,
    max_positions: usize,
) -> Vec<Position> {
    let mut positions = Vec::new();
    for g in 0..n_games {
        if positions.len() >= max_positions {
            break;
        }
        let bias = match g % 3 {
            0 => 0.25, // mostly random: quiet/scattered positions
            1 => 0.6,  // moderate: some line-building
            _ => 0.9,  // tactical: heavy line-building, more forced shapes
        };
        let mut rng = ChaCha8Rng::seed_from_u64(0xF0AC1A6 ^ (g as u64));
        positions.extend(play_synthetic_game(config, bias, &mut rng));
    }
    positions.truncate(max_positions);
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

// --- forcing::solve vs. exhaustive depth-1/2 scan ---

#[derive(Default)]
struct ForcingBenchStats {
    /// Positions actually measured (moves_remaining_this_turn() == 2 and
    /// non-terminal -- the only point the shortcut fires in production).
    n: usize,
    times_us: Vec<f64>,
    /// Same latencies, split by outcome. `no_or_exceeded_us` is the latency
    /// class that matters most for self-play throughput: in production,
    /// `forcing::solve` runs before every normal-search fallback, so on a
    /// non-winning root (`No` or `BudgetExceeded`) its cost is pure overhead
    /// added on top of the subsequent MCTS search -- unlike `Win`, which
    /// replaces the search entirely (net win, not overhead).
    win_us: Vec<f64>,
    no_or_exceeded_us: Vec<f64>,
    solve_win: usize,
    solve_no: usize,
    solve_budget_exceeded: usize,
    /// Wins found by the exhaustive depth-1/2 scan (`shortcut_all_legal`,
    /// unbounded by candidate set -- the strongest possible baseline for a
    /// depth <= 2 shortcut).
    old_win: usize,
    old_d1: usize,
    old_d2: usize,
    /// `forcing::solve` found `Win` on a position the old exhaustive scan
    /// also won -- expected, the common case.
    both_win: usize,
    /// Old scan found a win but `forcing::solve` did NOT -- a REGRESSION;
    /// should be 0. Kept separate from `solve_no`/`solve_budget_exceeded`
    /// tallies above for a direct superset check.
    regressions: usize,
    /// `forcing::solve` found a win the old exhaustive depth-1/2 scan could
    /// not (i.e. a genuinely deeper, multi-turn forced win) -- the new
    /// capability this solver adds.
    extra_deeper_wins: usize,
    win_depth_hist: HashMap<u8, usize>,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn run_forcing_bench(
    positions: &[Position],
    config: GameConfig,
    depth_cap: u8,
    node_budget: u64,
) -> ForcingBenchStats {
    let mut stats = ForcingBenchStats::default();
    for pos in positions {
        // Matches the production guard in gumbel_mcts.rs Step 3.5: the
        // solver only ever runs at the start of a turn.
        if pos.moves_remaining != 2 {
            continue;
        }
        let game = GameState::from_state(&pos.stones, pos.current_player, pos.moves_remaining, config);
        if game.is_terminal() {
            continue;
        }

        let old_hit = shortcut_all_legal(&game);

        let t = Instant::now();
        let outcome = forcing::solve(&game, depth_cap, node_budget);
        let elapsed_us = t.elapsed().as_nanos() as f64 / 1000.0;

        stats.n += 1;
        stats.times_us.push(elapsed_us);

        if let Some(c) = old_hit {
            stats.old_win += 1;
            if classify_hit(&game, c) == 1 {
                stats.old_d1 += 1;
            } else {
                stats.old_d2 += 1;
            }
        }

        match outcome {
            forcing::Outcome::Win(w) => {
                stats.solve_win += 1;
                stats.win_us.push(elapsed_us);
                *stats.win_depth_hist.entry(w.depth).or_insert(0) += 1;
                if old_hit.is_some() {
                    stats.both_win += 1;
                } else {
                    stats.extra_deeper_wins += 1;
                }
            }
            forcing::Outcome::No => {
                stats.solve_no += 1;
                stats.no_or_exceeded_us.push(elapsed_us);
                if old_hit.is_some() {
                    stats.regressions += 1;
                }
            }
            forcing::Outcome::BudgetExceeded => {
                stats.solve_budget_exceeded += 1;
                stats.no_or_exceeded_us.push(elapsed_us);
                if old_hit.is_some() {
                    stats.regressions += 1;
                }
            }
        }
    }
    stats
}

// --- tight vs wide generator A/B (analysis-screen wide-by-default sizing) ---

#[derive(Default)]
struct WideAbStats {
    n: usize,
    tight_win: usize,
    tight_no: usize,
    tight_be: usize,
    wide_win: usize,
    wide_no: usize,
    wide_be: usize,
    /// Wins only the wide generator sees (threat + quiet-builder turns) — the
    /// coverage the analysis screen gains.
    wide_only_wins: usize,
    /// Tight wins the wide run did NOT prove at the same node budget — the
    /// budget-starvation risk of a bigger tree at a fixed budget; the wide
    /// generator itself is a strict superset, so any count here is budget, not
    /// generation.
    tight_only_wins: usize,
    tight_us: Vec<f64>,
    wide_us: Vec<f64>,
    wide_win_us: Vec<f64>,
    wide_no_us: Vec<f64>,
    tight_no_us: Vec<f64>,
    wide_win_depth_hist: HashMap<u8, usize>,
}

fn run_wide_ab_bench(
    positions: &[Position],
    config: GameConfig,
    depth_cap: u8,
    node_budget: u64,
) -> WideAbStats {
    let mut stats = WideAbStats::default();
    for pos in positions {
        if pos.moves_remaining != 2 {
            continue;
        }
        let game = GameState::from_state(&pos.stones, pos.current_player, pos.moves_remaining, config);
        if game.is_terminal() {
            continue;
        }
        stats.n += 1;

        let t = Instant::now();
        let tight = forcing::solve(&game, depth_cap, node_budget);
        let tight_us = t.elapsed().as_nanos() as f64 / 1000.0;
        stats.tight_us.push(tight_us);

        let t = Instant::now();
        let wide = forcing::solve_wide(&game, depth_cap, node_budget);
        let wide_us = t.elapsed().as_nanos() as f64 / 1000.0;
        stats.wide_us.push(wide_us);

        match &tight {
            forcing::Outcome::Win(_) => stats.tight_win += 1,
            forcing::Outcome::No => {
                stats.tight_no += 1;
                stats.tight_no_us.push(tight_us);
            }
            forcing::Outcome::BudgetExceeded => {
                stats.tight_be += 1;
                stats.tight_no_us.push(tight_us);
            }
        }
        match &wide {
            forcing::Outcome::Win(w) => {
                stats.wide_win += 1;
                stats.wide_win_us.push(wide_us);
                *stats.wide_win_depth_hist.entry(w.depth).or_insert(0) += 1;
                if !tight.is_win() {
                    stats.wide_only_wins += 1;
                }
            }
            forcing::Outcome::No => {
                stats.wide_no += 1;
                stats.wide_no_us.push(wide_us);
            }
            forcing::Outcome::BudgetExceeded => {
                stats.wide_be += 1;
                stats.wide_no_us.push(wide_us);
            }
        }
        if tight.is_win() && !wide.is_win() {
            stats.tight_only_wins += 1;
        }
    }
    stats
}

fn print_wide_ab_report(stats: &WideAbStats, depth_cap: u8, node_budget: u64) {
    let dist = |v: &mut Vec<f64>| -> (f64, f64, f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
        (mean, percentile(v, 0.5), percentile(v, 0.99), v.last().copied().unwrap_or(0.0))
    };
    let mut s = WideAbStats {
        tight_us: stats.tight_us.clone(),
        wide_us: stats.wide_us.clone(),
        wide_win_us: stats.wide_win_us.clone(),
        wide_no_us: stats.wide_no_us.clone(),
        tight_no_us: stats.tight_no_us.clone(),
        ..Default::default()
    };
    println!();
    println!("=== tight vs wide forcing::solve A/B ===");
    println!(
        "depth_cap={depth_cap}  node_budget={node_budget}  positions (mr==2, non-terminal): {}",
        stats.n
    );
    println!(
        "  tight: Win={} No={} BudgetExceeded={}",
        stats.tight_win, stats.tight_no, stats.tight_be
    );
    println!(
        "  wide:  Win={} No={} BudgetExceeded={}",
        stats.wide_win, stats.wide_no, stats.wide_be
    );
    println!(
        "  wide-only wins: {}   tight-only wins (budget starvation at equal budget): {}",
        stats.wide_only_wins, stats.tight_only_wins
    );
    let (m, p50, p99, max) = dist(&mut s.tight_us);
    println!("  tight all:  mean={m:.0}us p50={p50:.0}us p99={p99:.0}us max={max:.0}us");
    let (m, p50, p99, max) = dist(&mut s.wide_us);
    println!("  wide all:   mean={m:.0}us p50={p50:.0}us p99={p99:.0}us max={max:.0}us");
    let (m, p50, p99, max) = dist(&mut s.tight_no_us);
    println!("  tight no:   mean={m:.0}us p50={p50:.0}us p99={p99:.0}us max={max:.0}us");
    let (m, p50, p99, max) = dist(&mut s.wide_no_us);
    println!("  wide no:    mean={m:.0}us p50={p50:.0}us p99={p99:.0}us max={max:.0}us");
    let (m, p50, p99, max) = dist(&mut s.wide_win_us);
    println!("  wide win:   mean={m:.0}us p50={p50:.0}us p99={p99:.0}us max={max:.0}us");
    let mut depths: Vec<&u8> = stats.wide_win_depth_hist.keys().collect();
    depths.sort();
    print!("  wide win depth histogram: ");
    for d in depths {
        print!("depth={}:{}  ", d, stats.wide_win_depth_hist[d]);
    }
    println!();
}

fn print_forcing_report(stats: &ForcingBenchStats, depth_cap: u8, node_budget: u64) {
    println!();
    println!("=== forcing::solve vs. exhaustive depth-1/2 scan ===");
    println!(
        "depth_cap={depth_cap}  node_budget={node_budget}  positions measured (moves_remaining==2, non-terminal): {}",
        stats.n
    );
    if stats.n == 0 {
        println!("(no eligible positions -- nothing to report)");
        return;
    }

    let mut sorted = stats.times_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = sorted.iter().sum();
    let mean = sum / sorted.len() as f64;
    let p50 = percentile(&sorted, 0.50);
    let p99 = percentile(&sorted, 0.99);
    let max = *sorted.last().unwrap();

    println!();
    println!("latency (us/position), ALL outcomes:");
    println!("  mean = {mean:.2}   p50 = {p50:.2}   p99 = {p99:.2}   max = {max:.2}");

    // Outcome-conditioned breakdown: `no_or_exceeded` is the class that
    // matters for self-play throughput (see field doc on
    // `ForcingBenchStats::no_or_exceeded_us`) -- it is pure overhead paid on
    // every non-winning root before the normal MCTS search runs.
    let mut win_sorted = stats.win_us.clone();
    win_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut noex_sorted = stats.no_or_exceeded_us.clone();
    noex_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !win_sorted.is_empty() {
        let m = win_sorted.iter().sum::<f64>() / win_sorted.len() as f64;
        println!(
            "  Win only        (n={}): mean={:.2}  p50={:.2}  p99={:.2}  max={:.2}",
            win_sorted.len(),
            m,
            percentile(&win_sorted, 0.50),
            percentile(&win_sorted, 0.99),
            *win_sorted.last().unwrap(),
        );
    }
    if !noex_sorted.is_empty() {
        let m = noex_sorted.iter().sum::<f64>() / noex_sorted.len() as f64;
        println!(
            "  No/BudgetExceeded (n={}, pure overhead): mean={:.2}  p50={:.2}  p99={:.2}  max={:.2}",
            noex_sorted.len(),
            m,
            percentile(&noex_sorted, 0.50),
            percentile(&noex_sorted, 0.99),
            *noex_sorted.last().unwrap(),
        );
    }

    println!();
    println!("outcomes:");
    println!(
        "  solve: Win={} ({:.2}%)  No={}  BudgetExceeded={}",
        stats.solve_win,
        100.0 * stats.solve_win as f64 / stats.n as f64,
        stats.solve_no,
        stats.solve_budget_exceeded,
    );
    println!(
        "  old exhaustive depth-1/2 scan: Win={} (d1={}, d2={})",
        stats.old_win, stats.old_d1, stats.old_d2
    );
    println!(
        "  both agree Win: {}   solve-only (deeper) wins: {}   REGRESSIONS (old won, solve didn't): {}",
        stats.both_win, stats.extra_deeper_wins, stats.regressions
    );
    println!(
        "  superset check: {}",
        if stats.regressions == 0 {
            "PASS -- solve finds a superset of the old scan's wins"
        } else {
            "FAIL -- solve MISSED wins the old exhaustive scan found (regression!)"
        }
    );

    let mut depths: Vec<&u8> = stats.win_depth_hist.keys().collect();
    depths.sort();
    print!("  win depth histogram: ");
    for d in depths {
        print!("depth={}:{}  ", d, stats.win_depth_hist[d]);
    }
    println!();
}

/// Default real self-play corpus (HX07, win_length=6/placement_radius=2 --
/// matches this const's own regime). Preserved permanent copy: the live
/// self-play `batches/` dir is ephemeral (the trainer consumes+deletes those
/// files), so the bench points at a stable copy under `bench-corpus/`
/// (git-ignored). Used when no positional path is given and the file is
/// present on disk; falls back to `synthetic` otherwise.
const DEFAULT_HX07_CORPUS: &str =
    "/var/home/tyto/Development/personal/hexo-strix/bench-corpus/games_lean-d6_wl6r2_0002.bin";

fn main() {
    let args: Vec<String> = env::args().collect();

    // Positional path (first non-`--` argument), a `--corpus=` flag, or the
    // default real HX07 corpus (if present) / `synthetic` (fallback). A real
    // corpus is always preferred over synthetic when available.
    let mut path: Option<String> = None;
    let mut max_positions: Option<usize> = None;
    let mut n_games: usize = 2_000;
    let mut radius: i32 = 2;
    let mut win_length: u8 = 6;
    let mut mode = CandidateMode::Gumbel;
    let mut do_mcts = false;
    let mut n_simulations: u32 = 128;
    let mut virtual_loss: f64 = 1.0;
    let mut depth_cap: u8 = SELF_PLAY_DEPTH_CAP;
    let mut node_budget: u64 = SELF_PLAY_NODE_BUDGET;
    let mut wide_ab = false;
    for a in &args[1..] {
        if let Some(v) = a.strip_prefix("--corpus=") {
            path = Some(v.to_string());
        } else if let Some(v) = a.strip_prefix("--max=") {
            max_positions = Some(v.parse().expect("--max value"));
        } else if let Some(v) = a.strip_prefix("--games=") {
            n_games = v.parse().expect("--games value");
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
        } else if a == "--wide-ab" {
            wide_ab = true;
        } else if let Some(v) = a.strip_prefix("--sims=") {
            n_simulations = v.parse().expect("--sims value");
        } else if let Some(v) = a.strip_prefix("--vl=") {
            virtual_loss = v.parse().expect("--vl value");
        } else if let Some(v) = a.strip_prefix("--depth-cap=") {
            depth_cap = v.parse().expect("--depth-cap value");
        } else if let Some(v) = a.strip_prefix("--node-budget=") {
            node_budget = v.parse().expect("--node-budget value");
        } else if !a.starts_with("--") {
            path = Some(a.clone());
        } else {
            eprintln!("unknown flag: {a}");
            std::process::exit(1);
        }
    }

    // Resolve the path: explicit arg wins; otherwise prefer the real HX07
    // corpus if it exists on disk; otherwise fall back to synthetic.
    let path = path.unwrap_or_else(|| {
        if std::path::Path::new(DEFAULT_HX07_CORPUS).exists() {
            DEFAULT_HX07_CORPUS.to_string()
        } else {
            "synthetic".to_string()
        }
    });
    let synthetic = path == "synthetic";
    // Default cap: 50_000 for a real games.bin (unchanged); for the
    // synthetic corpus, default to ~5000 turn-start positions (matches the
    // corpus size assumed by docs/superpowers/plans/2026-07-03-forcing-solver-rust-port.md
    // Task 11) unless the user overrides via --max.
    let max_positions = max_positions.unwrap_or(if synthetic { 5_000 } else { 50_000 });

    let config = GameConfig {
        win_length,
        placement_radius: radius,
        max_moves: 200,
    };

    // `has_policy_data`: only legacy HX03/HX04 records carry legal_coords/
    // policy (needed for the Gumbel-top-K candidate-scoped table below).
    // HX07 board-state records and the synthetic corpus leave those empty.
    let (positions, has_policy_data) = if synthetic {
        println!(
            "Generating synthetic corpus (no real corpus found at {DEFAULT_HX07_CORPUS}): \
             n_games={n_games}, max_positions={max_positions}, radius={radius}, win={win_length}"
        );
        let positions = generate_synthetic_corpus(config, n_games, max_positions);
        println!("Generated {} turn-start positions", positions.len());
        (positions, false)
    } else {
        // Auto-detect wire format by magic: HX07 (board-state, current
        // format) vs. legacy HX03/HX04 (serialized-graph format).
        let magic = sniff_magic(&path);
        let (positions, has_policy_data) = match magic.as_ref().map(|m| m.as_slice()) {
            Some(m) if m == MAGIC_HX07 => {
                println!(
                    "Loading HX07 positions from {path} (max={max_positions}, radius={radius}, win={win_length})"
                );
                (read_positions_hx07(&path, max_positions), false)
            }
            Some(_) => {
                println!(
                    "Loading legacy HX03/HX04 positions from {path} (max={max_positions}, radius={radius}, win={win_length}, candidates={mode:?})"
                );
                (read_positions(&path, max_positions), true)
            }
            None => panic!("could not open or sniff magic from {path}"),
        };
        println!("Loaded {} positions", positions.len());
        (positions, has_policy_data)
    };

    if do_mcts {
        run_mcts_bench(&positions, config, n_simulations, virtual_loss);
        return;
    }

    // Tight vs wide generator A/B (analysis-screen wide-by-default sizing):
    // run with the budget under consideration, e.g.
    //   --wide-ab --depth-cap=10 --node-budget=20000 --max=2000 (trajectory tier)
    if wide_ab {
        let stats = run_wide_ab_bench(&positions, config, depth_cap, node_budget);
        print_wide_ab_report(&stats, depth_cap, node_budget);
        return;
    }

    // New: forcing::solve at the self-play budget vs. the exhaustive
    // depth-1/2 scan. Works for both real (file) and synthetic corpora --
    // only needs board state, not policy.
    let forcing_stats = run_forcing_bench(&positions, config, depth_cap, node_budget);
    print_forcing_report(&forcing_stats, depth_cap, node_budget);

    // Old candidate-scoped comparison table: only meaningful with a real
    // policy (Gumbel-top-K candidates need it), so skip it for the
    // synthetic corpus and HX07 board-state corpora (legal_coords/policy
    // are empty in both).
    if !has_policy_data {
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
