//! Fully-forcing (VCF) threat-space solver. Pure Rust; no NN dependency.
//! Ports scripts/forcing_search_prototype.py (validated). See the design spec.
use hexo_engine::types::{Coord, Player};
use rustc_hash::{FxHashMap, FxHashSet};

const WIN_AXES: [(i32, i32); 3] = [(1, 0), (0, 1), (1, -1)];

#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic per-(coord, player) Zobrist value (no stored table; the board is
/// infinite so coords are unbounded).
#[inline]
fn zob(coord: Coord, player: Player) -> u64 {
    let (q, r) = coord;
    let p = matches!(player, Player::P2) as u64;
    splitmix64(
        (q as u64).wrapping_mul(0x100_0000_01B3)
            ^ ((r as u64).wrapping_mul(0x9E37_79B1)) << 1
            ^ (p << 63)
            ^ 0xD1B5_4A32_D192_ED03,
    )
}

/// Mutable board for make/unmake search — no per-node clone.
pub struct SolverBoard {
    pub stones: FxHashMap<Coord, Player>,
    pub hash: u64,
}

impl SolverBoard {
    pub fn new() -> Self {
        SolverBoard { stones: FxHashMap::default(), hash: 0 }
    }
    #[inline]
    pub fn place(&mut self, coord: Coord, player: Player) {
        debug_assert!(!self.stones.contains_key(&coord), "double-place corrupts Zobrist");
        self.stones.insert(coord, player);
        self.hash ^= zob(coord, player);
    }
    #[inline]
    pub fn remove(&mut self, coord: Coord) {
        if let Some(player) = self.stones.remove(&coord) {
            self.hash ^= zob(coord, player);
        }
    }
    #[inline]
    pub fn get(&self, coord: Coord) -> Option<Player> {
        self.stones.get(&coord).copied()
    }
}

/// Enemy-free length-wl windows through `player`'s stones holding >= wl-2 of them;
/// each returned as its sorted empty cells (size 1 or 2). Strip-scan per (stone, axis).
pub fn completions(board: &SolverBoard, player: Player, wl: u8) -> FxHashSet<Vec<Coord>> {
    let l = wl as i32;
    let mut result: FxHashSet<Vec<Coord>> = FxHashSet::default();
    let mut stones: Vec<Coord> = board.stones.iter()
        .filter(|&(_, &owner)| owner == player)
        .map(|(&c, _)| c)
        .collect();
    stones.sort_unstable();
    for (sq, sr) in stones {
        for &(dq, dr) in WIN_AXES.iter() {
            // strip of 2l-1 cells centered on the stone
            let coords: Vec<Coord> = (1 - l..l).map(|o| (sq + o * dq, sr + o * dr)).collect();
            let owners: Vec<Option<Player>> = coords.iter().map(|&c| board.get(c)).collect();
            for k in 0..(l as usize) { // windows owners[k..k+l], each covers the center
                let mut pc = 0u8;
                let mut empties: Vec<Coord> = Vec::with_capacity(2);
                let mut enemy = false;
                for j in k..k + l as usize {
                    match owners[j] {
                        None => empties.push(coords[j]),
                        Some(o) if o == player => pc += 1,
                        Some(_) => { enemy = true; break; }
                    }
                }
                if !enemy && (pc as i32) >= (wl as i32) - 2 && (pc as i32) <= (wl as i32) - 1 {
                    empties.sort_unstable();
                    result.insert(empties);
                }
            }
        }
    }
    result
}

/// (B, all size-B hitting sets) over the completion sets. Small inputs -> enumerate
/// k-subsets of the involved cells by increasing k.
pub fn min_covers(comps: &[Vec<Coord>]) -> (u8, Vec<Vec<Coord>>) {
    if comps.is_empty() { return (0, vec![vec![]]); }
    let mut universe: Vec<Coord> = comps.iter().flatten().copied().collect();
    universe.sort_unstable();
    universe.dedup();
    let n = universe.len();
    for k in 1..=n {
        let mut covers: Vec<Vec<Coord>> = Vec::new();
        // iterate all k-combinations of `universe`
        let mut idx: Vec<usize> = (0..k).collect();
        loop {
            let sub: Vec<Coord> = idx.iter().map(|&i| universe[i]).collect();
            let hits_all = comps.iter().all(|c| c.iter().any(|cell| sub.contains(cell)));
            if hits_all { covers.push(sub); }
            // advance combination
            let mut i = k;
            loop {
                if i == 0 { break; }
                i -= 1;
                if idx[i] != i + n - k { break; }
                if i == 0 { i = usize::MAX; break; }
            }
            if i == usize::MAX { break; }
            idx[i] += 1;
            for j in i + 1..k { idx[j] = idx[j - 1] + 1; }
        }
        if !covers.is_empty() { return (k as u8, covers); }
    }
    (n as u8, vec![universe])
}

/// Near-complete lines (enemy-free windows with wl-4..wl-3 stones), indexed by empty
/// cell — so a move's B is a table lookup, not a rescan.
pub fn threat_table(board: &SolverBoard, player: Player, wl: u8) -> FxHashMap<Coord, Vec<(Vec<Coord>, u8)>> {
    let l = wl as i32;
    let mut index: FxHashMap<Coord, Vec<(Vec<Coord>, u8)>> = FxHashMap::default();
    let mut seen: FxHashSet<Vec<Coord>> = FxHashSet::default();
    let mut stones: Vec<Coord> = board.stones.iter()
        .filter(|&(_, &owner)| owner == player)
        .map(|(&c, _)| c)
        .collect();
    stones.sort_unstable();
    for (sq, sr) in stones {
        for &(dq, dr) in WIN_AXES.iter() {
            let coords: Vec<Coord> = (1 - l..l).map(|o| (sq + o * dq, sr + o * dr)).collect();
            let owners: Vec<Option<Player>> = coords.iter().map(|&c| board.get(c)).collect();
            for k in 0..(l as usize) {
                let mut pc = 0u8; let mut empties: Vec<Coord> = Vec::new(); let mut enemy = false;
                for j in k..k + l as usize {
                    match owners[j] {
                        None => empties.push(coords[j]),
                        Some(o) if o == player => pc += 1,
                        Some(_) => { enemy = true; break; }
                    }
                }
                if enemy || !((pc as i32) >= (wl as i32) - 4 && (pc as i32) <= (wl as i32) - 3) { continue; }
                let window: Vec<Coord> = coords[k..k + l as usize].to_vec();
                if !seen.insert(window) { continue; }
                empties.sort_unstable();
                for &e in empties.iter() {
                    index.entry(e).or_default().push((empties.clone(), pc));
                }
            }
        }
    }
    index
}

/// B after playing `mv`, read from the threat table (no board scan).
pub fn move_b(index: &FxHashMap<Coord, Vec<(Vec<Coord>, u8)>>, mv: &[Coord], wl: u8) -> u8 {
    let mut comps: FxHashSet<Vec<Coord>> = FxHashSet::default();
    for &cell in mv {
        if let Some(entries) = index.get(&cell) {
            for (empties, pc) in entries {
                let filled = empties.iter().filter(|e| mv.contains(e)).count() as u8;
                if (*pc as i32) + (filled as i32) >= (wl as i32) - 2 {
                    let mut remaining: Vec<Coord> = empties.iter().filter(|e| !mv.contains(e)).copied().collect();
                    remaining.sort_unstable();
                    comps.insert(remaining);
                }
            }
        }
    }
    let v: Vec<Vec<Coord>> = comps.into_iter().collect();
    min_covers(&v).0
}

pub fn has_completion(board: &SolverBoard, player: Player, wl: u8, max_empties: u8) -> bool {
    completions(board, player, wl).iter().any(|c| c.len() as u8 <= max_empties)
}

/// Sound gate: can the attacker make a four this turn? (any enemy-free window with
/// >= wl-4 attacker stones). Equivalent to a non-empty threat table.
pub fn any_four_gate(board: &SolverBoard, attacker: Player, wl: u8) -> bool {
    !threat_table(board, attacker, wl).is_empty()
}

pub fn attacker_turns(board: &SolverBoard, attacker: Player, defender: Player, placements: u8, wl: u8) -> Vec<Vec<Coord>> {
    let index = threat_table(board, attacker, wl);
    let mut hot: Vec<Coord> = index.keys().copied().collect();
    hot.sort_unstable();
    let mut scored: Vec<(u8, Vec<Coord>)> = Vec::new();
    if placements == 1 {
        for &c in &hot {
            let b = move_b(&index, &[c], wl);
            if b >= 2 { scored.push((b, vec![c])); }
        }
    } else {
        // partners = hot ∪ defender-block cells (cells of defender completions)
        let mut partners: FxHashSet<Coord> = hot.iter().copied().collect();
        for comp in completions(board, defender, wl) {
            for cell in comp { partners.insert(cell); }
        }
        let mut partners_sorted: Vec<Coord> = partners.into_iter().collect();
        partners_sorted.sort_unstable();
        let mut seen: FxHashSet<(Coord, Coord)> = FxHashSet::default();
        for &h in &hot {
            for &a in &partners_sorted {
                if a == h { continue; }
                let mv = if h < a { (h, a) } else { (a, h) };
                if !seen.insert(mv) { continue; }
                let b = move_b(&index, &[mv.0, mv.1], wl);
                if b >= 2 { scored.push((b, vec![mv.0, mv.1])); }
            }
        }
    }
    scored.sort_by(|x, y| y.0.cmp(&x.0)); // best-first (higher B)
    scored.into_iter().map(|(_, m)| m).collect()
}

pub enum SolveResult { Win { depth: u8 }, No, BudgetExceeded }

struct SearchState {
    tt: FxHashMap<(u64, bool, u8, u8), (bool, bool)>, // (hash, is_or, placements, budget) -> (won, cutoff)
    nodes: u64,
    budget: u64,
    exceeded: bool,
    wl: u8,
    atk: Player,
    dfn: Player,
}

fn tick(s: &mut SearchState) -> bool {
    s.nodes += 1;
    if s.nodes > s.budget { s.exceeded = true; }
    s.exceeded
}

/// Returns (won, cutoff).
fn atk_within(board: &mut SolverBoard, placements: u8, budget: u8, s: &mut SearchState) -> (bool, bool) {
    if s.exceeded || tick(s) { return (false, false); }
    if has_completion(board, s.atk, s.wl, placements) { return (true, false); }
    if budget < 2 { return (false, true); }
    let key = (board.hash, true, placements, budget);
    if let Some(&v) = s.tt.get(&key) { return v; }
    let (mut won, mut cut) = (false, false);
    for mv in attacker_turns(board, s.atk, s.dfn, placements, s.wl) {
        for &c in &mv { board.place(c, s.atk); }
        let (w, subcut) = def_within(board, budget - 1, s);
        for &c in &mv { board.remove(c); }
        cut |= subcut;
        if w { won = true; break; }
    }
    let res = if won { (true, false) } else { (false, cut) };
    if !s.exceeded { s.tt.insert(key, res); }
    res
}

fn def_within(board: &mut SolverBoard, budget: u8, s: &mut SearchState) -> (bool, bool) {
    if s.exceeded || tick(s) { return (false, false); }
    if has_completion(board, s.dfn, s.wl, 2) { return (false, false); } // defender wins first
    let comps: Vec<Vec<Coord>> = completions(board, s.atk, s.wl).into_iter().collect();
    let (bnum, covers) = min_covers(&comps);
    if bnum < 2 { return (false, false); }
    if bnum >= 3 { return if budget >= 1 { (true, false) } else { (false, true) }; }
    let key = (board.hash, false, 0, budget);
    if let Some(&v) = s.tt.get(&key) { return v; }
    let mut result = (true, false);
    for cover in covers {
        for &c in &cover { board.place(c, s.dfn); }
        let (w, subcut) = atk_within(board, 2, budget, s);
        for &c in &cover { board.remove(c); }
        if !w { result = (false, subcut); break; }
    }
    if !s.exceeded { s.tt.insert(key, result); }
    result
}

pub fn solve_from(board: &mut SolverBoard, attacker: Player, defender: Player,
                  placements_remaining: u8, wl: u8, depth_cap: u8, node_budget: u64) -> SolveResult {
    // sound gate
    if !any_four_gate(board, attacker, wl) && !has_completion(board, attacker, wl, placements_remaining) {
        return SolveResult::No;
    }
    let mut s = SearchState { tt: FxHashMap::default(), nodes: 0, budget: node_budget,
        exceeded: false, wl, atk: attacker, dfn: defender };
    for depth in 1..=depth_cap {
        let (won, cut) = atk_within(board, placements_remaining, depth, &mut s);
        if won { return SolveResult::Win { depth }; }
        if s.exceeded { return SolveResult::BudgetExceeded; }
        if !cut { return SolveResult::No; }
    }
    SolveResult::BudgetExceeded
}

use hexo_engine::game::GameState;

pub struct ForcingWin { pub depth: u8, pub first_move: Coord, pub pv: Vec<Coord> }
pub enum Outcome { Win(ForcingWin), No, BudgetExceeded }

pub fn solve(game: &GameState, depth_cap: u8, node_budget: u64) -> Outcome {
    let attacker = match game.current_player() { Some(p) => p, None => return Outcome::No };
    let defender = attacker.opponent();
    let wl = game.config().win_length;
    let placements = game.moves_remaining_this_turn();
    let mut board = SolverBoard::new();
    for (&c, &p) in game.stones() { board.place(c, p); }
    match solve_from(&mut board, attacker, defender, placements, wl, depth_cap, node_budget) {
        SolveResult::No => Outcome::No,
        SolveResult::BudgetExceeded => Outcome::BudgetExceeded,
        SolveResult::Win { depth } => {
            // Compute on the pristine (post-solve_from, pre-extract_pv) board: extract_pv
            // permanently applies the winning line to `board`, so first_winning_move must
            // run first or it would be probing an already-won position.
            let fm = first_winning_move(&mut board, attacker, defender, placements, wl, depth, node_budget);
            let pv = extract_pv(&mut board, attacker, defender, placements, wl, depth, node_budget);
            match fm.or_else(|| pv.first().copied()) {
                Some(first) => Outcome::Win(ForcingWin { depth, first_move: first, pv }),
                // Should be impossible for a proven win, but never panic: fall back to
                // treating it as no forcing win found; the MCTS caller just runs normal search.
                None => Outcome::No,
            }
        }
    }
}

/// The attacker's first winning placement for a proven `Win{depth}`: an immediately
/// executable completion's first cell, else the first `attacker_turns` move whose
/// `def_within(depth-1, ..)` (fresh full-budget state) holds. Never panics; `None` only
/// if no such move exists (shouldn't happen for a proven win).
fn first_winning_move(board: &mut SolverBoard, attacker: Player, defender: Player,
                       placements: u8, wl: u8, depth: u8, node_budget: u64) -> Option<Coord> {
    for c in completions(board, attacker, wl) {
        if c.len() as u8 <= placements {
            return c.first().copied();
        }
    }
    for mv in attacker_turns(board, attacker, defender, placements, wl) {
        for &c in &mv { board.place(c, attacker); }
        let mut s = SearchState { tt: FxHashMap::default(), nodes: 0, budget: node_budget,
            exceeded: false, wl, atk: attacker, dfn: defender };
        let (w, _) = def_within(board, depth.saturating_sub(1), &mut s);
        for &c in &mv { board.remove(c); }
        if w { return mv.first().copied(); }
    }
    None
}

/// Port of principal_variation: returns the flat placement sequence of one winning line
/// (attacker turns interleaved with prolonging defender covers), ending in six-in-a-row.
fn extract_pv(board: &mut SolverBoard, attacker: Player, defender: Player,
              mut placements: u8, wl: u8, depth: u8, node_budget: u64) -> Vec<Coord> {
    let mut pv: Vec<Coord> = Vec::new();
    let mut remaining = depth;
    loop {
        // attacker to move: immediate completion?
        let mut fin: Option<Vec<Coord>> = None;
        for c in completions(board, attacker, wl) {
            if c.len() as u8 <= placements { fin = Some(c); break; }
        }
        if let Some(cells) = fin {
            for &c in &cells { board.place(c, attacker); pv.push(c); }
            break;
        }
        // find a winning forcing move
        let mut chosen: Option<Vec<Coord>> = None;
        for mv in attacker_turns(board, attacker, defender, placements, wl) {
            for &c in &mv { board.place(c, attacker); }
            let mut s = SearchState { tt: FxHashMap::default(), nodes: 0, budget: node_budget,
                exceeded: false, wl, atk: attacker, dfn: defender };
            let (w, _) = def_within(board, remaining - 1, &mut s);
            for &c in &mv { board.remove(c); }
            if w { chosen = Some(mv); break; }
        }
        let mv = match chosen { Some(m) => m, None => break };
        for &c in &mv { board.place(c, attacker); pv.push(c); }
        placements = 2;
        // defender: B>=3 terminal, or prolonging cover
        let comps: Vec<Vec<Coord>> = completions(board, attacker, wl).into_iter().collect();
        let (bnum, covers) = min_covers(&comps);
        if bnum >= 3 {
            // defender blocks two threat cells (futile), attacker completes next loop
            let mut cells: Vec<Coord> = comps.iter().flatten().copied().collect();
            cells.sort_unstable(); cells.dedup();
            for &c in cells.iter().take(2) { board.place(c, defender); pv.push(c); }
            remaining -= 1;
            continue;
        }
        // pick the cover that prolongs the win the most
        let mut best: Option<(u8, Vec<Coord>)> = None;
        for cover in covers {
            for &c in &cover { board.place(c, defender); }
            let mut sub: Option<u8> = None;
            for d in 1..remaining {
                let mut s = SearchState { tt: FxHashMap::default(), nodes: 0, budget: node_budget,
                    exceeded: false, wl, atk: attacker, dfn: defender };
                let (w, _) = atk_within(board, 2, d, &mut s);
                if w { sub = Some(d); break; }
            }
            for &c in &cover { board.remove(c); }
            if let Some(sd) = sub {
                if best.as_ref().map_or(true, |(bd, _)| sd > *bd) { best = Some((sd, cover)); }
            }
        }
        let (sd, cover) = match best { Some(b) => b, None => break };
        for &c in &cover { board.place(c, defender); pv.push(c); }
        remaining = sd;
    }
    pv
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::types::Player;

    #[test]
    fn completions_four_run_open_both_sides_has_three() {
        // __XXXX__ along the q-axis, no enemies (wl=6)
        let mut b = SolverBoard::new();
        for q in 0..4 { b.place((q, 0), Player::P1); }
        let comps = completions(&b, Player::P1, 6);
        let want: FxHashSet<Vec<Coord>> = [
            vec![(-2, 0), (-1, 0)],   // left
            vec![(-1, 0), (4, 0)],    // straddle
            vec![(4, 0), (5, 0)],     // right
        ].into_iter().collect();
        assert_eq!(comps, want);
    }

    #[test]
    fn min_covers_straddle_rejects_naive_block() {
        let comps = vec![
            vec![(-2, 0), (-1, 0)], vec![(-1, 0), (4, 0)], vec![(4, 0), (5, 0)],
        ];
        let (bnum, covers) = min_covers(&comps);
        assert_eq!(bnum, 2);
        let cset: FxHashSet<Vec<Coord>> = covers.into_iter().collect();
        let want: FxHashSet<Vec<Coord>> = [
            vec![(-2, 0), (4, 0)], vec![(-1, 0), (5, 0)], vec![(-1, 0), (4, 0)],
        ].into_iter().collect();
        assert_eq!(cset, want);
        assert!(!cset.contains(&vec![(-2, 0), (5, 0)])); // misses the straddle
    }

    #[test]
    fn boundary_b_values() {
        // 5-run open both ends -> B=2 unique cover
        let mut b = SolverBoard::new();
        for q in 0..5 { b.place((q, 0), Player::P1); }
        assert_eq!(min_covers(&sorted_vec(completions(&b, Player::P1, 6))).0, 2);
        // walled 4-run (one empty each side) -> B=1
        let mut w = SolverBoard::new();
        for q in 0..4 { w.place((q, 0), Player::P1); }
        w.place((-2, 0), Player::P2); w.place((5, 0), Player::P2);
        assert_eq!(min_covers(&sorted_vec(completions(&w, Player::P1, 6))).0, 1);
    }

    fn sorted_vec(s: FxHashSet<Vec<Coord>>) -> Vec<Vec<Coord>> { s.into_iter().collect() }

    #[test]
    fn move_b_matches_double_four_fork() {
        // two 3-runs sharing (0,0) on q and r axes; playing (3,0)+(0,3) -> B>=3
        let mut b = SolverBoard::new();
        for c in [(0,0),(1,0),(2,0),(0,1),(0,2)] { b.place(c, Player::P1); }
        let idx = threat_table(&b, Player::P1, 6);
        assert!(move_b(&idx, &[(3,0),(0,3)], 6) >= 3);
        // a lone extension is only B<=2
        assert!(move_b(&idx, &[(3,0),(9,9)], 6) <= 2);
    }

    #[test]
    fn attacker_turns_orders_double_four_first() {
        let mut b = SolverBoard::new();
        for c in [(0,0),(1,0),(2,0),(0,1),(0,2)] { b.place(c, Player::P1); }
        let moves = attacker_turns(&b, Player::P1, Player::P2, 2, 6);
        assert!(!moves.is_empty());
        // the top move creates B>=3
        let idx = threat_table(&b, Player::P1, 6);
        assert!(move_b(&idx, &moves[0], 6) >= 3);
    }

    #[test]
    fn any_four_gate_false_on_scattered() {
        let mut b = SolverBoard::new();
        b.place((0,0), Player::P1); b.place((10,10), Player::P1);
        assert!(!any_four_gate(&b, Player::P1, 6));
    }

    fn line(cells: &[Coord], p: Player) -> SolverBoard {
        let mut b = SolverBoard::new();
        for &c in cells { b.place(c, p); }
        b
    }

    #[test]
    fn mate_in_one() {
        let mut b = line(&[(0,0),(1,0),(2,0),(3,0),(4,0)], Player::P1);
        assert!(matches!(solve_from(&mut b, Player::P1, Player::P2, 2, 6, 40, 5_000_000), SolveResult::Win { depth: 1 }));
    }
    #[test]
    fn mate_in_two_fork() {
        let mut b = line(&[(0,0),(1,0),(2,0),(0,1),(0,2)], Player::P1);
        assert!(matches!(solve_from(&mut b, Player::P1, Player::P2, 2, 6, 40, 5_000_000), SolveResult::Win { depth: 2 }));
    }
    #[test]
    fn no_forcing_win() {
        let mut b = line(&[(0,0),(10,10)], Player::P1);
        assert!(matches!(solve_from(&mut b, Player::P1, Player::P2, 2, 6, 40, 5_000_000), SolveResult::No));
    }

    #[test]
    fn solve_from_gamestate_returns_pv_that_wins() {
        use hexo_engine::game::{GameState, GameConfig};
        let stones: Vec<(Coord, Player)> = [(0,0),(1,0),(2,0),(0,1),(0,2)]
            .into_iter().map(|c| (c, Player::P1)).collect();
        let game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);
        match solve(&game, 40, 5_000_000) {
            Outcome::Win(w) => {
                assert_eq!(w.depth, 2);
                // The position is symmetric: either run can be extended in either open
                // direction to form the double-four fork, so any of the four extension
                // cells is a valid winning first move (FIX 3 pins one deterministically,
                // but doesn't privilege a particular one).
                let fork_cells = [(0, 3), (0, -1), (3, 0), (-1, 0)];
                assert!(fork_cells.contains(&w.first_move),
                        "first_move {:?} not a fork cell", w.first_move);
                assert_eq!(w.pv.first(), Some(&w.first_move));
                assert!(w.pv.len() >= 3); // >=2 attacker fork cells + a completion
            }
            _ => panic!("expected Win"),
        }
    }

    #[test]
    fn zobrist_place_remove_is_reversible_and_order_independent() {
        let mut b = SolverBoard::new();
        let h0 = b.hash;
        b.place((0, 0), Player::P1);
        b.place((1, 0), Player::P2);
        assert_ne!(b.hash, h0);
        // order independence: same stones -> same hash
        let mut c = SolverBoard::new();
        c.place((1, 0), Player::P2);
        c.place((0, 0), Player::P1);
        assert_eq!(b.hash, c.hash);
        // remove reverts
        b.remove((0, 0));
        b.remove((1, 0));
        assert_eq!(b.hash, h0);
        assert!(b.stones.is_empty());
    }

    // FIX 2: wl-4/wl-3/wl-2 thresholds must not underflow (u8) for small win_length.
    #[test]
    fn wl4_and_wl2_do_not_panic() {
        let mut b4 = line(&[(0,0),(1,0),(2,0)], Player::P1);
        let _ = solve_from(&mut b4, Player::P1, Player::P2, 2, 4, 10, 100_000);
        let _ = completions(&b4, Player::P1, 4);
        let _ = threat_table(&b4, Player::P1, 4);

        let mut b2 = line(&[(0,0)], Player::P1);
        let _ = solve_from(&mut b2, Player::P1, Player::P2, 2, 2, 10, 100_000);
        let _ = completions(&b2, Player::P1, 2);
        let _ = threat_table(&b2, Player::P1, 2);
    }

    // FIX 3: candidate enumeration (board.stones / game.stones() hashmap iteration) must
    // not leak into first_move/PV selection non-determinism.
    #[test]
    fn first_move_is_deterministic() {
        use hexo_engine::game::{GameState, GameConfig};
        let stones: Vec<(Coord, Player)> = [(0,0),(1,0),(2,0),(0,1),(0,2)]
            .into_iter().map(|c| (c, Player::P1)).collect();
        let game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);
        let a = match solve(&game, 40, 5_000_000) {
            Outcome::Win(w) => w.first_move,
            _ => panic!("expected Win"),
        };
        let b = match solve(&game, 40, 5_000_000) {
            Outcome::Win(w) => w.first_move,
            _ => panic!("expected Win"),
        };
        assert_eq!(a, b, "first_move must be deterministic across calls");
    }
}
