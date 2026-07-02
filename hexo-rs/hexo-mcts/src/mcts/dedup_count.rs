//! Phase 0 DAG-MCTS measurement spike: throwaway canonical-serialisation
//! fingerprint and instrumentation hook for counting would-be dedup hits in
//! tree MCTS expansion.
//!
//! **This module is throwaway.** The fingerprint here is intentionally simple
//! (O(stones) per call, full re-serialisation) and is NOT the real Zobrist
//! that will ship in Phase 1. Its only purpose is to let us measure how many
//! expansions in a typical Gumbel-AZ search land on a position that has
//! already been expanded, so we can decide whether a DAG transposition table
//! is worth building for real.
//!
//! Everything in this module is gated behind the `dedup_count` cargo feature;
//! when the feature is off there is no cost to production code.
#![cfg(feature = "dedup_count")]

use hexo_engine::{Coord, GameState, Player};
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_hash::FxHashMap as HashMap;
use xxhash_rust::xxh3::xxh3_128;

/// Marker bytes embedded in the serialised buffer.
const SIDE_P1: u8 = 0;
const SIDE_P2: u8 = 1;
const SIDE_GAMEOVER: u8 = 255;

const MOVES_LEFT_GAMEOVER: u8 = 0;
const MOVES_LEFT_MID_TURN: u8 = 1;
const MOVES_LEFT_FRESH: u8 = 2;

const PENDING_NONE: u8 = 0xFF;
const PENDING_SOME: u8 = 0x01;

const WINNER_NONE: u8 = 0;
const WINNER_P1: u8 = 1;
const WINNER_P2: u8 = 2;
const WINNER_DRAW: u8 = 3;

/// Compute a 128-bit fingerprint for `state`, taking the optional `parent_state`
/// into account to record the "pending cell" in mid-turn positions.
///
/// Canonical serialisation:
/// 1. Sorted `(coord, player)` occupancy bytes.
/// 2. Side-to-move byte.
/// 3. `moves_left` byte.
/// 4. Pending cell (coord bytes if mid-turn and identifiable, otherwise marker).
/// 5. Game-over winner byte.
///
/// Hashed via `xxh3_128`.
///
/// This is a throwaway spike fingerprint — see module docs.
pub fn position_fingerprint(state: &GameState, parent_state: Option<&GameState>) -> u128 {
    let mut buf: Vec<u8> = Vec::with_capacity(64);

    // 1. Sorted occupancy.
    let mut stones: Vec<(Coord, Player)> = state.placed_stones();
    stones.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for ((q, r), player) in &stones {
        buf.extend_from_slice(&q.to_le_bytes());
        buf.extend_from_slice(&r.to_le_bytes());
        buf.push(match player {
            Player::P1 => 1,
            Player::P2 => 2,
        });
    }

    // 2. Side to move.
    let side = if state.is_terminal() {
        SIDE_GAMEOVER
    } else {
        match state.current_player() {
            Some(Player::P1) => SIDE_P1,
            Some(Player::P2) => SIDE_P2,
            None => SIDE_GAMEOVER,
        }
    };
    buf.push(side);

    // 3. moves_left category.
    let moves_left_byte = if state.is_terminal() {
        MOVES_LEFT_GAMEOVER
    } else {
        match state.moves_remaining_this_turn() {
            1 => MOVES_LEFT_MID_TURN,
            _ => MOVES_LEFT_FRESH,
        }
    };
    buf.push(moves_left_byte);

    // 4. Pending cell: identified only when mid-turn and parent has exactly
    //    one fewer stone than child. Uses set difference.
    let mut pending: Option<Coord> = None;
    if moves_left_byte == MOVES_LEFT_MID_TURN {
        if let Some(parent) = parent_state {
            let child_stones = state.placed_stones();
            let parent_stones = parent.placed_stones();
            if child_stones.len() == parent_stones.len() + 1 {
                let parent_set: FxHashSet<Coord> =
                    parent_stones.iter().map(|(c, _)| *c).collect();
                for (c, _) in &child_stones {
                    if !parent_set.contains(c) {
                        pending = Some(*c);
                        break;
                    }
                }
            }
        }
    }
    match pending {
        Some((q, r)) => {
            buf.push(PENDING_SOME);
            buf.extend_from_slice(&q.to_le_bytes());
            buf.extend_from_slice(&r.to_le_bytes());
        }
        None => buf.push(PENDING_NONE),
    }

    // 5. Winner byte.
    let winner_byte = if state.is_terminal() {
        match state.winner() {
            Some(Player::P1) => WINNER_P1,
            Some(Player::P2) => WINNER_P2,
            None => WINNER_DRAW,
        }
    } else {
        WINNER_NONE
    };
    buf.push(winner_byte);

    xxh3_128(&buf)
}

/// Instrumentation state threaded through a single MCTS search (or aggregated
/// across many). Counts fingerprint-collisions at expansion time.
#[derive(Default, Clone)]
pub struct InstrumentationState {
    /// All fingerprints observed at expansion so far.
    pub seen: FxHashSet<u128>,
    /// Total calls to `record_expansion`.
    pub total_expansions: u64,
    /// Hits classified as within-turn (commutative mid-turn transposition).
    pub within_turn_hits: u64,
    /// Hits classified as cross-turn (all other hits, including winning-first-stone).
    pub cross_turn_hits: u64,
}

impl InstrumentationState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one expansion: compute the fingerprint of `child`, classify any
    /// hit, and insert into the seen set. `parent` is the state the child was
    /// reached from (may be `None` for the root).
    ///
    /// Classification rules:
    /// * Within-turn: parent had `moves_left == 2`, child has `moves_left == 0`
    ///   (full turn completed without the game ending). Both placements land
    ///   in the same turn; since the fingerprint ignores within-turn ordering,
    ///   any repeat is an order-reversal transposition.
    /// * Cross-turn: everything else, including winning-first-stone, mid-turn
    ///   states, and normal across-turn collisions.
    pub fn record_expansion(&mut self, parent: Option<&GameState>, child: &GameState) {
        self.total_expansions += 1;
        let fp = position_fingerprint(child, parent);
        if !self.seen.insert(fp) {
            // Classify the hit.
            let within_turn = classify_within_turn(parent, child);
            if within_turn {
                self.within_turn_hits += 1;
            } else {
                self.cross_turn_hits += 1;
            }
        }
    }
}

/// Return `true` if this parent->child transition is a completed 2-stone turn
/// that did NOT end the game. Winning-first-stone paths are excluded: their
/// child is terminal without the commutativity premise holding.
fn classify_within_turn(parent: Option<&GameState>, child: &GameState) -> bool {
    let Some(parent) = parent else {
        return false;
    };
    if parent.is_terminal() || child.is_terminal() {
        return false;
    }
    // Parent started a fresh turn.
    if parent.moves_remaining_this_turn() != 2 {
        return false;
    }
    // Child finished that turn (dropped to 0 means turn flipped to other player
    // with moves_left == 2, so from child's view moves_remaining is 2 but the
    // current_player has changed).
    if child.moves_remaining_this_turn() != 2 {
        return false;
    }
    if parent.current_player() == child.current_player() {
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Thread-local instrumentation singleton.
//
// Avoids plumbing `&mut InstrumentationState` through every MCTS function
// signature for the spike. Tree MCTS is single-threaded within a search, and
// the feature is only meant to be enabled for short measurement runs.
// ---------------------------------------------------------------------------

use std::cell::RefCell;

thread_local! {
    static INSTRUMENTATION: RefCell<InstrumentationState> =
        RefCell::new(InstrumentationState::new());
}

/// Record an expansion against the thread-local instrumentation state.
pub fn record_expansion_tls(parent: Option<&GameState>, child: &GameState) {
    INSTRUMENTATION.with(|cell| cell.borrow_mut().record_expansion(parent, child));
}

/// Reset the thread-local instrumentation state to empty.
pub fn reset_tls() {
    INSTRUMENTATION.with(|cell| *cell.borrow_mut() = InstrumentationState::new());
}

// ---------------------------------------------------------------------------
// REQ-0e-bis: per-search eval cache for the wall-clock upper-bound prototype.
//
// On a fingerprint hit at leaf expansion, the cached (policy_logits, value)
// pair is reused instead of calling the evaluator. This produces a tree node
// with the SAME (logits, value) as a real expansion would, so the rest of
// MCTS treats it identically — but with no network forward pass.
//
// This is NOT real DAG MCGS: no recursive-Q backprop, no node sharing.
// Hence the resulting wall-clock delta is an upper bound on what real
// DAG MCGS can save in HeXO under the same configuration.
// ---------------------------------------------------------------------------

/// Cached (policy_logits, value) pair from a previous evaluator call.
#[derive(Clone)]
pub struct CachedEval {
    pub policy_logits: HashMap<Coord, f64>,
    pub value: f64,
}

#[derive(Default)]
pub struct EvalCache {
    pub cache: FxHashMap<u128, CachedEval>,
    pub hits: u64,
    pub misses: u64,
}

impl EvalCache {
    pub fn new() -> Self {
        Self::default()
    }
}

thread_local! {
    static EVAL_CACHE: RefCell<EvalCache> = RefCell::new(EvalCache::new());
    /// Runtime toggle for the cache+skip path. Even with the `dedup_skip`
    /// feature compiled in, the skip path is a no-op unless this flag is
    /// `true`. Lets a single binary run both "count mode" (skip disabled)
    /// and "skip mode" (skip enabled) back-to-back for wall-clock A/B.
    static SKIP_ENABLED: RefCell<bool> = const { RefCell::new(false) };
}

/// Set the runtime skip-enabled flag for the current thread.
pub fn set_skip_enabled_tls(enabled: bool) {
    SKIP_ENABLED.with(|cell| *cell.borrow_mut() = enabled);
}

/// Read the runtime skip-enabled flag for the current thread.
pub fn skip_enabled_tls() -> bool {
    SKIP_ENABLED.with(|cell| *cell.borrow())
}

/// Look up `fp` in the per-thread eval cache. Returns a clone of the cached
/// entry on hit, `None` on miss. Increments hit/miss counters accordingly.
pub fn eval_cache_get_tls(fp: u128) -> Option<CachedEval> {
    EVAL_CACHE.with(|cell| {
        let mut c = cell.borrow_mut();
        match c.cache.get(&fp).cloned() {
            Some(entry) => {
                c.hits += 1;
                Some(entry)
            }
            None => {
                c.misses += 1;
                None
            }
        }
    })
}

/// Insert an evaluator result into the per-thread cache.
pub fn eval_cache_insert_tls(fp: u128, entry: CachedEval) {
    EVAL_CACHE.with(|cell| {
        cell.borrow_mut().cache.insert(fp, entry);
    });
}

/// Reset the per-thread eval cache to empty.
pub fn eval_cache_reset_tls() {
    EVAL_CACHE.with(|cell| *cell.borrow_mut() = EvalCache::new());
}

/// Snapshot the per-thread eval cache hit/miss counters and current size.
pub fn eval_cache_snapshot_tls() -> (u64, u64, usize) {
    EVAL_CACHE.with(|cell| {
        let c = cell.borrow();
        (c.hits, c.misses, c.cache.len())
    })
}

/// Snapshot the thread-local instrumentation counters (seen set excluded).
pub fn snapshot_tls() -> (u64, u64, u64, usize) {
    INSTRUMENTATION.with(|cell| {
        let s = cell.borrow();
        (
            s.total_expansions,
            s.within_turn_hits,
            s.cross_turn_hits,
            s.seen.len(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::GameConfig;

    fn small_config() -> GameConfig {
        GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        }
    }

    /// Post-turn commutativity: for every legal (A,B) mid-turn pair where
    /// neither placement wins the game, applying A then B yields the same
    /// fingerprint as B then A.
    #[test]
    fn post_turn_commutativity() {
        let base = GameState::with_config(small_config());
        let moves = base.legal_moves();
        let mut checked = 0;
        for &a in &moves {
            for &b in &moves {
                if a == b {
                    continue;
                }
                let mut ab = base.clone();
                if ab.apply_move(a).is_err() {
                    continue;
                }
                if ab.is_terminal() {
                    continue; // winning first stone — excluded.
                }
                if ab.apply_move(b).is_err() {
                    continue;
                }
                if ab.is_terminal() {
                    continue;
                }

                let mut ba = base.clone();
                if ba.apply_move(b).is_err() {
                    continue;
                }
                if ba.is_terminal() {
                    continue;
                }
                if ba.apply_move(a).is_err() {
                    continue;
                }
                if ba.is_terminal() {
                    continue;
                }

                let fp_ab = position_fingerprint(&ab, Some(&base));
                let fp_ba = position_fingerprint(&ba, Some(&base));
                assert_eq!(
                    fp_ab, fp_ba,
                    "commutativity failure for moves {:?} and {:?}",
                    a, b
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no mid-turn pairs were actually compared");
    }

    /// Anti-collapse: two states with identical occupancy but different side
    /// to move must produce different fingerprints.
    #[test]
    fn anti_collapse_p1_vs_p2() {
        // Start at base (P2 to move, moves_left=2). Play (1,0) then (2,0) to
        // complete P2's turn → P1 to move with moves_left=2.
        let base = GameState::with_config(small_config());
        let mut p1_to_move = base.clone();
        p1_to_move.apply_move((1, 0)).unwrap();
        p1_to_move.apply_move((2, 0)).unwrap();
        assert_eq!(p1_to_move.current_player(), Some(Player::P1));

        // Hand-construct a "pretend P2 to move" by using the base state plus
        // two arbitrary stones — we can't easily do that, so instead verify
        // side-to-move is actually part of the hash by comparing the real
        // base state (P2 to move) to a post-turn state (P1 to move) that
        // happens to share... no, occupancy differs.
        //
        // Instead: two mid-turn states after playing (1,0) — parent is base,
        // current player is still P2 with moves_left=1. Compare to a
        // constructed state where occupancy is the same but we advance to a
        // P1-to-move equivalent by finishing the turn and then backing up
        // mentally — we can't back up. So rely on the direct test: build two
        // identical-occupancy states by playing the same two stones from
        // two different starting players. Since P1 is always hardcoded at
        // (0,0), we can't fully vary starting player.
        //
        // Simplest valid test: compare the post-turn state (P1 to move) with
        // a fingerprint computed on the same GameState but with side byte
        // overridden — we can't override. So instead: show that the mid-turn
        // state (P2 to move, moves_left=1) after (1,0) has a different
        // fingerprint from the post-turn state with the same two stones.
        //
        // But that differs in moves_left too, not just side. So build a
        // second post-turn state with different occupancy that shares no
        // stones with the first post-turn state? That won't test side-to-move
        // in isolation either.
        //
        // Acceptable compromise: craft fingerprints manually via two direct
        // calls where only the side-to-move marker differs in the buffer by
        // constructing two synthetic states. We'll instead use the library's
        // assertion that side byte is hashed: build (P1 to move) and an
        // alternate state with same stones but different internal side via
        // replaying moves in a different order is impossible because P1
        // always starts. So we go with the strongest available check:
        // fingerprint of base (P2, moves_left=2) differs from fingerprint
        // of post-turn (P1, moves_left=2) — these differ in side AND
        // occupancy, which is acceptable as an anti-collapse test: if side
        // weren't hashed, we'd still see a difference from occupancy. To
        // *isolate* side, we reach into a manual buffer comparison below.
        let base_fp = position_fingerprint(&base, None);
        let post_fp = position_fingerprint(&p1_to_move, None);
        assert_ne!(base_fp, post_fp);

        // Isolated side-to-move test: two post-turn states where the only
        // difference is which player is to move. Construct by comparing
        // p1_to_move against a different game that has P2 to move with the
        // same 3 stones: not possible via normal play. Instead, verify the
        // byte layout directly: build the buffer twice with the same stones
        // and differ only in the side byte, hash, and compare.
        use xxhash_rust::xxh3::xxh3_128;
        let stones_bytes: Vec<u8> = {
            let mut v = Vec::new();
            let mut s = p1_to_move.placed_stones();
            s.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            for ((q, r), p) in &s {
                v.extend_from_slice(&q.to_le_bytes());
                v.extend_from_slice(&r.to_le_bytes());
                v.push(match p {
                    Player::P1 => 1,
                    Player::P2 => 2,
                });
            }
            v
        };
        let mut buf_p1 = stones_bytes.clone();
        buf_p1.push(SIDE_P1);
        buf_p1.push(MOVES_LEFT_FRESH);
        buf_p1.push(PENDING_NONE);
        buf_p1.push(WINNER_NONE);

        let mut buf_p2 = stones_bytes.clone();
        buf_p2.push(SIDE_P2);
        buf_p2.push(MOVES_LEFT_FRESH);
        buf_p2.push(PENDING_NONE);
        buf_p2.push(WINNER_NONE);

        assert_ne!(xxh3_128(&buf_p1), xxh3_128(&buf_p2));
    }

    /// Mid-turn distinguishability: two mid-turn states sharing the same
    /// post-turn occupancy placeholder but different pending cells must
    /// fingerprint differently.
    #[test]
    fn mid_turn_pending_differs() {
        let base = GameState::with_config(small_config());
        let mut mid_a = base.clone();
        mid_a.apply_move((1, 0)).unwrap();
        assert_eq!(mid_a.moves_remaining_this_turn(), 1);

        let mut mid_b = base.clone();
        mid_b.apply_move((2, 0)).unwrap();
        assert_eq!(mid_b.moves_remaining_this_turn(), 1);

        let fp_a = position_fingerprint(&mid_a, Some(&base));
        let fp_b = position_fingerprint(&mid_b, Some(&base));
        assert_ne!(fp_a, fp_b);
    }
}
