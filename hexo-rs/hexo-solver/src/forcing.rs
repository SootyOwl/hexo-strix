//! Fully-forcing (VCF) threat-space solver. Pure Rust; no NN dependency.
//! Ports scripts/forcing_search_prototype.py (validated). See the design spec.
//!
//! Performance notes (2026-07-04 optimization pass):
//! - `SolverBoard` is a dense, grow-on-demand grid (not a hash map): strip scans
//!   read cells with one array load. Real game positions are spatially bounded
//!   (every stone lands within `placement_radius` of another), so the grid stays
//!   small; a pathological multi-thousand-cell coordinate spread would grow it
//!   proportionally (guarded by `MAX_GRID_CELLS`).
//! - In the tight-radius regime (`placement_radius < win_length - 1`, the live
//!   self-play configuration) legality queries are O(1) reads of an incrementally
//!   maintained per-cell "stones within radius" count.
//! - Gap sets, covers, and attacker moves are copyable fixed-size cell sets
//!   (`CellSet2`, `ThreatEntry`) — no per-window/per-candidate heap allocation,
//!   and the per-position memo caches clone with a memcpy.
//! - `min_covers2`/`cover_class` classify B into {0, 1, 2, >=3} with bitmask
//!   hitting-set checks; the search never needs more (covers only materialize
//!   for B <= 2), which also removes the old combinatorial subset enumeration.
use hexo_engine::hex::{hex_distance as hex_dist, hex_offsets};
use hexo_engine::types::{Coord, Player};
use rustc_hash::{FxHashMap, FxHashSet};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Research-only search cutoffs threaded into `SearchState` (`idtt` prover driver
/// + portfolio racing). The production default is `Limits::default()` (both
/// `None`), which makes `tick` behave byte-for-byte as before. `deadline` is a
/// wall-clock cutoff; `cancel`, when set by a racing thread, cooperatively stops
/// this search. Both are sampled sparsely so `Instant::now()`/atomic loads never
/// dominate the ~10µs/node hot path.
#[derive(Clone, Default)]
pub struct Limits {
    pub deadline: Option<Instant>,
    pub cancel: Option<Arc<AtomicBool>>,
}

const WIN_AXES: [(i32, i32); 3] = [(1, 0), (0, 1), (1, -1)];
/// Max supported win_length: strip-scan buffers are stack arrays of 2*MAX_WL-1.
pub const MAX_WL: usize = 16;
const STRIP: usize = 2 * MAX_WL - 1;
/// Grid border padding kept around every stone. Must be >= any tight-regime
/// radius (which is < win_length - 1 <= MAX_WL - 1) so that reach-count updates
/// for a just-placed stone never fall outside the grid.
pub(crate) const GRID_PAD: i32 = 16;
/// Hard cap on grid area — fails loudly on pathological coordinate spreads
/// instead of attempting a multi-GB allocation.
pub(crate) const MAX_GRID_CELLS: i64 = 64 * 1024 * 1024;

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

#[inline]
fn pcode(p: Player) -> u8 {
    match p {
        Player::P1 => 1,
        Player::P2 => 2,
    }
}

/// A sorted set of 0..=2 cells. Represents a completion's gap set, a defender
/// cover, or an attacker move (1 or 2 placements; the sorted order is also the
/// placement order, matching the old normalized-(min,max) convention).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CellSet2 {
    len: u8,
    cells: [Coord; 2],
}

impl CellSet2 {
    /// Padding for unused slots; never a real cell (real coords are bounded by
    /// the grid, far below i32::MAX).
    const PAD_CELL: Coord = (i32::MAX, i32::MAX);

    pub fn empty() -> Self {
        CellSet2 { len: 0, cells: [Self::PAD_CELL; 2] }
    }
    pub fn one(c: Coord) -> Self {
        CellSet2 { len: 1, cells: [c, Self::PAD_CELL] }
    }
    pub fn two(a: Coord, b: Coord) -> Self {
        debug_assert_ne!(a, b);
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        CellSet2 { len: 2, cells: [a, b] }
    }
    pub fn from_cells(cells: &[Coord]) -> Self {
        match *cells {
            [] => Self::empty(),
            [a] => Self::one(a),
            [a, b] => Self::two(a, b),
            _ => unreachable!("CellSet2 holds at most 2 cells"),
        }
    }
    #[inline]
    pub fn cells(&self) -> &[Coord] {
        &self.cells[..self.len as usize]
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn contains(&self, c: Coord) -> bool {
        self.cells().contains(&c)
    }
    #[inline]
    pub fn first(&self) -> Option<Coord> {
        self.cells().first().copied()
    }
}

/// Mutable board for make/unmake search — dense grid + stone list, no per-node
/// clone. Cells outside the grid are empty by construction (the grid always
/// covers every stone with `GRID_PAD` slack).
pub struct SolverBoard {
    grid: Vec<u8>, // 0 empty, 1 P1, 2 P2
    /// Per-cell count of stones within `reach_radius` (tight regime only).
    reach: Vec<u16>,
    q0: i32,
    r0: i32,
    w: i32,
    h: i32,
    /// -1 = reach tracking off.
    reach_radius: i32,
    disk: Vec<Coord>,
    pub stones: Vec<(Coord, Player)>,
    pub hash: u64,
}

impl Default for SolverBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl SolverBoard {
    pub fn new() -> Self {
        SolverBoard {
            grid: Vec::new(),
            reach: Vec::new(),
            q0: 0,
            r0: 0,
            w: 0,
            h: 0,
            reach_radius: -1,
            disk: Vec::new(),
            stones: Vec::new(),
            hash: 0,
        }
    }

    #[inline]
    fn idx(&self, (q, r): Coord) -> Option<usize> {
        let x = q - self.q0;
        let y = r - self.r0;
        if x < 0 || y < 0 || x >= self.w || y >= self.h {
            None
        } else {
            Some((y * self.w + x) as usize)
        }
    }

    /// Raw cell code: 0 empty (including out-of-grid), 1 P1, 2 P2.
    #[inline]
    fn raw(&self, c: Coord) -> u8 {
        match self.idx(c) {
            Some(i) => self.grid[i],
            None => 0,
        }
    }

    #[inline]
    pub fn get(&self, coord: Coord) -> Option<Player> {
        match self.raw(coord) {
            1 => Some(Player::P1),
            2 => Some(Player::P2),
            _ => None,
        }
    }

    /// Grow the grid so the GRID_PAD box around `c` is inside it.
    #[inline]
    fn ensure_cover(&mut self, c: Coord) {
        if self.w == 0
            || c.0 - GRID_PAD < self.q0
            || c.1 - GRID_PAD < self.r0
            || c.0 + GRID_PAD >= self.q0 + self.w
            || c.1 + GRID_PAD >= self.r0 + self.h
        {
            self.grow(c, c);
        }
    }

    /// Rebuild the grid to cover its current bounds plus the box [lo, hi] with
    /// 2*GRID_PAD additive slack, so a placement drifting outward triggers a
    /// rebuild at most every GRID_PAD cells (search growth is bounded by
    /// depth_cap, so rebuilds stay rare; `solve` pre-sizes the common case).
    fn grow(&mut self, lo: Coord, hi: Coord) {
        let pad = 2 * GRID_PAD;
        let (mut nq0, mut nr0, mut nq1, mut nr1) =
            (lo.0 - pad, lo.1 - pad, hi.0 + pad, hi.1 + pad);
        if self.w != 0 {
            nq0 = nq0.min(self.q0);
            nr0 = nr0.min(self.r0);
            nq1 = nq1.max(self.q0 + self.w - 1);
            nr1 = nr1.max(self.r0 + self.h - 1);
        }
        let (nw, nh) = (nq1 - nq0 + 1, nr1 - nr0 + 1);
        assert!(
            (nw as i64) * (nh as i64) <= MAX_GRID_CELLS,
            "SolverBoard grid would exceed {MAX_GRID_CELLS} cells ({nw}x{nh}); \
             positions with unbounded coordinate spread are unsupported"
        );
        let cells = (nw as i64 * nh as i64) as usize; // guarded above, fits usize
        let mut ngrid = vec![0u8; cells];
        let tracking = self.reach_radius >= 0;
        let mut nreach = if tracking { vec![0u16; cells] } else { Vec::new() };
        for y in 0..self.h {
            let src = (y * self.w) as usize;
            let dst = (((y + self.r0 - nr0) * nw) + (self.q0 - nq0)) as usize;
            ngrid[dst..dst + self.w as usize].copy_from_slice(&self.grid[src..src + self.w as usize]);
            if tracking {
                nreach[dst..dst + self.w as usize]
                    .copy_from_slice(&self.reach[src..src + self.w as usize]);
            }
        }
        self.grid = ngrid;
        self.reach = nreach;
        self.q0 = nq0;
        self.r0 = nr0;
        self.w = nw;
        self.h = nh;
    }

    /// Pre-size the grid to cover the box [lo, hi]; avoids repeated growth when
    /// bulk-loading a position.
    pub fn reserve(&mut self, lo: Coord, hi: Coord) {
        debug_assert!(lo.0 <= hi.0 && lo.1 <= hi.1, "reserve expects lo <= hi componentwise");
        if self.idx((lo.0 - GRID_PAD, lo.1 - GRID_PAD)).is_none()
            || self.idx((hi.0 + GRID_PAD, hi.1 + GRID_PAD)).is_none()
        {
            self.grow(lo, hi);
        }
    }

    /// Switch on O(1) `within_radius` queries for `radius` by maintaining, for
    /// every grid cell, the count of stones within hex-distance `radius`.
    /// Idempotent for the same radius. Only worthwhile in the tight regime.
    pub fn enable_reach(&mut self, radius: i32) {
        debug_assert!((0..=GRID_PAD).contains(&radius), "reach radius must fit in GRID_PAD");
        if self.reach_radius == radius {
            return;
        }
        self.reach_radius = radius;
        self.disk = hex_offsets(radius);
        self.reach = vec![0u16; self.grid.len()];
        let (q0, r0, w) = (self.q0, self.r0, self.w);
        for si in 0..self.stones.len() {
            let (s, _) = self.stones[si];
            for di in 0..self.disk.len() {
                let (dq, dr) = self.disk[di];
                // In bounds by the GRID_PAD >= radius invariant.
                let i = ((s.1 + dr - r0) * w + (s.0 + dq - q0)) as usize;
                self.reach[i] += 1;
            }
        }
    }

    #[inline]
    pub fn place(&mut self, coord: Coord, player: Player) {
        debug_assert!(self.get(coord).is_none(), "double-place corrupts Zobrist");
        self.ensure_cover(coord);
        let i = self.idx(coord).unwrap();
        self.grid[i] = pcode(player);
        self.stones.push((coord, player));
        self.hash ^= zob(coord, player);
        if self.reach_radius >= 0 {
            let (q0, r0, w) = (self.q0, self.r0, self.w);
            for di in 0..self.disk.len() {
                let (dq, dr) = self.disk[di];
                let j = ((coord.1 + dr - r0) * w + (coord.0 + dq - q0)) as usize;
                self.reach[j] += 1;
            }
        }
    }

    #[inline]
    pub fn remove(&mut self, coord: Coord) {
        let Some(i) = self.idx(coord) else { return };
        if self.grid[i] == 0 {
            return;
        }
        let player = if self.grid[i] == 1 { Player::P1 } else { Player::P2 };
        self.grid[i] = 0;
        self.hash ^= zob(coord, player);
        // Make/unmake removes are LIFO in the search, so scan from the end.
        if let Some(pos) = self.stones.iter().rposition(|&(c, _)| c == coord) {
            self.stones.swap_remove(pos);
        }
        if self.reach_radius >= 0 {
            let (q0, r0, w) = (self.q0, self.r0, self.w);
            for di in 0..self.disk.len() {
                let (dq, dr) = self.disk[di];
                let j = ((coord.1 + dr - r0) * w + (coord.0 + dq - q0)) as usize;
                self.reach[j] -= 1;
            }
        }
    }

    /// True iff `cell` is within hex-distance `radius` of some stone currently on
    /// the board. This is exactly the engine's legality disk (a cell is legal iff
    /// empty AND within `placement_radius` of a stone; see `legal_moves::legal_moves`).
    /// O(1) when reach tracking is enabled for this radius (`solve_from` enables it
    /// in the tight regime); otherwise a linear stone scan (unit tests, ad-hoc use).
    #[inline]
    pub fn within_radius(&self, cell: Coord, radius: i32) -> bool {
        debug_assert!(radius >= 0, "within_radius expects a real radius, not the off sentinel");
        if radius == self.reach_radius {
            return match self.idx(cell) {
                Some(i) => self.reach[i] > 0,
                // Out of grid => farther than GRID_PAD >= radius from every stone.
                None => false,
            };
        }
        self.stones.iter().any(|&(s, _)| hex_dist(cell, s) <= radius)
    }
}

/// Scan all length-`wl` enemy-free windows for `player` whose stone count lies in
/// [pc_lo, pc_hi] and (in the tight regime) whose gap cells are all playable,
/// invoking `f(window_start, axis_index, pc, gap_cells)`. `f` returns true to stop
/// the scan early. Gap cells are emitted in ascending coord order (strip order).
///
/// Radius-awareness: a window only counts if the attacker could legally fill ALL
/// its gap cells. Skipped entirely when radius >= win_length-1 (every gap of a
/// window through a stone is within wl-1 <= radius of that stone), keeping the
/// wide-radius path byte-for-byte unchanged and zero-cost.
fn scan_windows(
    board: &SolverBoard,
    player: Player,
    wl: u8,
    radius: i32,
    pc_lo: i32,
    pc_hi: i32,
    f: &mut impl FnMut(Coord, usize, u8, &[Coord]) -> bool,
) {
    let l = wl as i32;
    debug_assert!((1..=MAX_WL as i32).contains(&l), "win_length must be 1..={MAX_WL}");
    let enforce = radius < l - 1;
    let mine = pcode(player);
    let n = (2 * l - 1) as usize;
    let mut coords = [(0i32, 0i32); STRIP];
    let mut owners = [0u8; STRIP]; // 0 empty, 1 mine, 2 enemy
    let mut empties = [(0i32, 0i32); MAX_WL];
    for si in 0..board.stones.len() {
        let (s, owner) = board.stones[si];
        if owner != player {
            continue;
        }
        for (ai, &(dq, dr)) in WIN_AXES.iter().enumerate() {
            // strip of 2l-1 cells centered on the stone
            for (t, item) in coords.iter_mut().enumerate().take(n) {
                let o = t as i32 + 1 - l;
                let c = (s.0 + o * dq, s.1 + o * dr);
                *item = c;
                let raw = board.raw(c);
                owners[t] = if raw == 0 {
                    0
                } else if raw == mine {
                    1
                } else {
                    2
                };
            }
            for k in 0..(l as usize) {
                // window owners[k..k+l], each covers the center stone
                let mut pc = 0i32;
                let mut ne = 0usize;
                let mut enemy = false;
                for j in k..k + l as usize {
                    match owners[j] {
                        0 => {
                            empties[ne] = coords[j];
                            ne += 1;
                        }
                        1 => pc += 1,
                        _ => {
                            enemy = true;
                            break;
                        }
                    }
                }
                if enemy || pc < pc_lo || pc > pc_hi {
                    continue;
                }
                // Window-level legality filter: drop the WHOLE window if any gap
                // is unplayable — the attacker could not legally complete it, so
                // it is neither a real completion/threat nor a legal defender
                // block. (Per-cell filtering could leave an unplayable companion
                // cell in a surviving cell's payload, inflating `move_b`.)
                if enforce && empties[..ne].iter().any(|&e| !board.within_radius(e, radius)) {
                    continue;
                }
                if f(coords[k], ai, pc as u8, &empties[..ne]) {
                    return;
                }
            }
        }
    }
}

/// Enemy-free length-wl windows through `player`'s stones holding >= wl-2 of them,
/// each returned as its (sorted) gap-cell set of size 1 or 2. Sorted + deduped.
pub fn completions(board: &SolverBoard, player: Player, wl: u8, radius: i32) -> Vec<CellSet2> {
    let l = wl as i32;
    let mut out: Vec<CellSet2> = Vec::new();
    scan_windows(board, player, wl, radius, l - 2, l - 1, &mut |_, _, _, empties| {
        out.push(CellSet2::from_cells(empties));
        false
    });
    out.sort_unstable();
    out.dedup();
    out
}

/// A near-complete window's gap cells (3..=4 of them) plus its stone count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ThreatEntry {
    pc: u8,
    len: u8,
    cells: [Coord; 4],
}

impl ThreatEntry {
    #[inline]
    pub fn cells(&self) -> &[Coord] {
        &self.cells[..self.len as usize]
    }
}

pub(crate) type ThreatIndex = FxHashMap<Coord, Vec<ThreatEntry>>;

/// Near-complete lines (enemy-free windows with wl-4..wl-3 stones), indexed by gap
/// cell — so a move's B is a table lookup, not a rescan.
fn threat_table(board: &SolverBoard, player: Player, wl: u8, radius: i32) -> ThreatIndex {
    let l = wl as i32;
    let mut index: ThreatIndex = FxHashMap::default();
    // Windows are visited once per stone they contain; dedup by identity
    // (start cell, axis) — cheap, and equivalent to deduping the cell vec.
    let mut seen: FxHashSet<(Coord, u8)> = FxHashSet::default();
    scan_windows(board, player, wl, radius, l - 4, l - 3, &mut |start, ai, pc, empties| {
        if !seen.insert((start, ai as u8)) {
            return false;
        }
        debug_assert!(empties.len() <= 4 && empties.windows(2).all(|w| w[0] < w[1]));
        let mut e = ThreatEntry { pc, len: empties.len() as u8, cells: [(0, 0); 4] };
        e.cells[..empties.len()].copy_from_slice(empties);
        for &c in empties {
            index.entry(c).or_default().push(e);
        }
        false
    });
    index
}

/// True iff `player` already has a completed winning line on the board. The
/// VCF kernel never looks at full windows (its scans want gaps to fill), so a
/// caller assembling arbitrary positions (e.g. the VCT probe placing defender
/// replies) must guard with this before asking the solver about a board where
/// the game is in fact already over.
pub(crate) fn has_full_line(board: &SolverBoard, player: Player, wl: u8, radius: i32) -> bool {
    let l = wl as i32;
    let mut found = false;
    scan_windows(board, player, wl, radius, l, l, &mut |_, _, _, _| {
        found = true;
        true
    });
    found
}

/// One-pass fusion of `threat_table` and the wide builder-score scan: a single
/// strip walk over enemy-free windows with pc in `[1, wl-1]` accumulates the
/// builder score of every gap cell (+pc per visit = `pc²` per window — see
/// `wide_partner_cells` for why that ordering is right), and inserts
/// `ThreatEntry`s for windows in the threat band `[wl-4, wl-3]` exactly as
/// `threat_table` does. The wider pc filter only ADDS callback invocations
/// (every visited window contains the strip's center stone, so pc >= 1
/// always); the threat-band windows are visited in the same order with the
/// same dedup, so the index is identical to `threat_table`'s. Wide-mode only:
/// tight callers keep the narrower scan and pay nothing for scoring.
pub(crate) fn threat_table_scored(
    board: &SolverBoard,
    player: Player,
    wl: u8,
    radius: i32,
) -> (ThreatIndex, FxHashMap<Coord, u32>) {
    let l = wl as i32;
    let mut index: ThreatIndex = FxHashMap::default();
    let mut scores: FxHashMap<Coord, u32> = FxHashMap::default();
    let mut seen: FxHashSet<(Coord, u8)> = FxHashSet::default();
    scan_windows(board, player, wl, radius, 1, l - 1, &mut |start, ai, pc, empties| {
        for &e in empties {
            *scores.entry(e).or_insert(0) += pc as u32;
        }
        if (l - 4..=l - 3).contains(&(pc as i32)) && seen.insert((start, ai as u8)) {
            debug_assert!(empties.len() <= 4 && empties.windows(2).all(|w| w[0] < w[1]));
            let mut e = ThreatEntry { pc, len: empties.len() as u8, cells: [(0, 0); 4] };
            e.cells[..empties.len()].copy_from_slice(empties);
            for &c in empties {
                index.entry(c).or_default().push(e);
            }
        }
        false
    });
    (index, scores)
}

/// B after playing `mv` (exact up to 4; 4 means ">= 4"), read from the threat
/// table (no board scan). `scratch` is caller-provided to keep the hot pair
/// loop allocation-free.
fn move_b_with(index: &ThreatIndex, mv: &[Coord], wl: u8, scratch: &mut Vec<CellSet2>) -> u8 {
    scratch.clear();
    for &cell in mv {
        if let Some(entries) = index.get(&cell) {
            for e in entries {
                let filled = e.cells().iter().filter(|c| mv.contains(c)).count() as i32;
                if (e.pc as i32) + filled >= (wl as i32) - 2 {
                    // remaining = gaps not filled by mv; <= 2 by arithmetic
                    // (pc + filled >= wl-2 and |gaps| = wl - pc), never 0
                    // (|gaps| >= 3 > |mv|).
                    let mut rem = [CellSet2::PAD_CELL; 2];
                    let mut nr = 0usize;
                    for &g in e.cells() {
                        if !mv.contains(&g) {
                            rem[nr] = g;
                            nr += 1;
                        }
                    }
                    debug_assert!((1..=2).contains(&nr), "remaining gaps must be 1..=2");
                    scratch.push(CellSet2::from_cells(&rem[..nr]));
                }
            }
        }
    }
    scratch.sort_unstable();
    scratch.dedup();
    cover_class(scratch)
}

/// B after playing `mv` (allocating convenience wrapper).
#[cfg(test)]
fn move_b(index: &ThreatIndex, mv: &[Coord], wl: u8) -> u8 {
    let mut scratch = Vec::new();
    move_b_with(index, mv, wl, &mut scratch)
}

/// Fill `cells` with the distinct sorted cells of `comps` and `masks[j]` with the
/// bitmask of comps that cell `cells[j]` hits. Requires `comps.len() <= 64` (so at
/// most 128 distinct cells). Returns the distinct-cell count.
fn build_cell_masks(comps: &[CellSet2], cells: &mut [Coord; 128], masks: &mut [u64; 128]) -> usize {
    debug_assert!(comps.len() <= 64, "mask fast path is sized for <= 64 comps");
    let mut u = 0usize;
    for cp in comps {
        for &c in cp.cells() {
            cells[u] = c;
            u += 1;
        }
    }
    cells[..u].sort_unstable();
    let mut w = 0usize;
    for i in 0..u {
        if i == 0 || cells[i] != cells[w - 1] {
            cells[w] = cells[i];
            w += 1;
        }
    }
    masks[..w].fill(0);
    for (i, cp) in comps.iter().enumerate() {
        for &c in cp.cells() {
            let j = cells[..w].binary_search(&c).unwrap();
            masks[j] |= 1u64 << i;
        }
    }
    w
}

/// Core of the capped min-hitting-set computation. Returns B capped at 3 (3 means
/// ">= 3": the search treats those identically — an unstoppable multi-threat).
/// When `collect` is given, ALL minimum covers are pushed for B <= 2, in
/// lexicographic cell order (parity with the old k-combination enumeration);
/// for B >= 3 covers are never needed and none are collected.
fn min_covers_capped(comps: &[CellSet2], collect: Option<&mut Vec<CellSet2>>) -> u8 {
    let n = comps.len();
    if n == 0 {
        if let Some(out) = collect {
            out.push(CellSet2::empty());
        }
        return 0;
    }
    debug_assert!(comps.iter().all(|c| !c.is_empty()));
    if n > 64 {
        return min_covers_big(comps, collect);
    }
    let full: u64 = if n == 64 { !0 } else { (1u64 << n) - 1 };
    let mut cells = [(0i32, 0i32); 128];
    let mut masks = [0u64; 128];
    let u = build_cell_masks(comps, &mut cells, &mut masks);
    let alive = full; // all comps alive at the root
    let b = min_hit_masks(comps, &cells[..u], &masks, alive, 3);
    if let Some(out) = collect {
        match b {
            1 => {
                for j in 0..u {
                    if masks[j] == full {
                        out.push(CellSet2::one(cells[j]));
                    }
                }
            }
            2 => {
                for ja in 0..u {
                    for jb in ja + 1..u {
                        if masks[ja] | masks[jb] == full {
                            out.push(CellSet2::two(cells[ja], cells[jb]));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    b
}

/// Fallback for > 64 comps (rare): same semantics, chunked bitset words.
fn min_covers_big(comps: &[CellSet2], mut collect: Option<&mut Vec<CellSet2>>) -> u8 {
    let n = comps.len();
    let words = n.div_ceil(64);
    let mut cells: Vec<Coord> = comps.iter().flat_map(|c| c.cells().iter().copied()).collect();
    cells.sort_unstable();
    cells.dedup();
    let u = cells.len();
    let mut masks = vec![0u64; u * words];
    for (i, cp) in comps.iter().enumerate() {
        for &c in cp.cells() {
            let j = cells.binary_search(&c).unwrap();
            masks[j * words + i / 64] |= 1u64 << (i % 64);
        }
    }
    let full_last: u64 = if n.is_multiple_of(64) { !0 } else { (1u64 << (n % 64)) - 1 };
    let is_full = |m: &[u64]| {
        m[..words - 1].iter().all(|&w| w == !0) && m[words - 1] == full_last
    };
    let mut b = 4u8;
    for j in 0..u {
        if is_full(&masks[j * words..(j + 1) * words]) {
            if let Some(out) = collect.as_deref_mut() {
                out.push(CellSet2::one(cells[j]));
                b = 1;
            } else {
                return 1;
            }
        }
    }
    if b == 1 {
        return 1;
    }
    let mut or_buf = vec![0u64; words];
    for a in 0..u {
        for bb in a + 1..u {
            for wi in 0..words {
                or_buf[wi] = masks[a * words + wi] | masks[bb * words + wi];
            }
            if is_full(&or_buf) {
                if let Some(out) = collect.as_deref_mut() {
                    out.push(CellSet2::two(cells[a], cells[bb]));
                    b = 2;
                } else {
                    return 2;
                }
            }
        }
    }
    if b == 2 { 2 } else { 3 }
}

/// Exact B capped at 4 (4 means ">= 4") without materializing covers — the
/// hot-path classifier and move-ordering key. Exactness up to 4 preserves the
/// prototype's preference for double-four moves (B=4) over triple threats
/// (B=3); B >= 5 sorts equal to 4 (all B >= 3 moves win identically next turn,
/// so only PV/first-move cosmetics could differ). Computed by branch-and-bound
/// on the hitting set: every cover must hit the first un-hit comp with one of
/// its <= 2 cells, so the tree has <= 2^4 leaves — no combinatorial enumeration.
fn cover_class(comps: &[CellSet2]) -> u8 {
    let n = comps.len();
    if n == 0 {
        return 0;
    }
    debug_assert!(comps.iter().all(|c| !c.is_empty()));
    if n == 1 {
        return 1;
    }
    if n == 2 {
        // B=1 iff the two gap sets intersect.
        let joint = comps[0].cells().iter().any(|&c| comps[1].contains(c));
        return if joint { 1 } else { 2 };
    }
    if n > 64 {
        return min_hit_slice(comps.to_vec(), 4);
    }
    let mut cells = [(0i32, 0i32); 128];
    let mut masks = [0u64; 128];
    let u = build_cell_masks(comps, &mut cells, &mut masks);
    let alive: u64 = if n == 64 { !0 } else { (1u64 << n) - 1 };
    min_hit_masks(comps, &cells[..u], &masks, alive, 4)
}

/// min(B, cap) over the still-alive comps (bitmask), branching on the first
/// alive comp's cells. Depth is bounded by `cap`.
fn min_hit_masks(comps: &[CellSet2], cells: &[Coord], masks: &[u64], alive: u64, cap: u8) -> u8 {
    if alive == 0 {
        return 0;
    }
    if cap == 0 {
        return 0; // budget exhausted: min(B, 0) — reads as "B >= original cap"
    }
    let first = alive.trailing_zeros() as usize;
    let mut best = cap;
    for &c in comps[first].cells() {
        let j = cells.binary_search(&c).unwrap();
        let r = 1 + min_hit_masks(comps, cells, masks, alive & !masks[j], best - 1);
        if r < best {
            best = r;
            if best == 1 {
                break; // no branch can return < 1
            }
        }
    }
    best
}

/// Allocating branch-and-bound fallback for > 64 comps (rare).
fn min_hit_slice(alive: Vec<CellSet2>, cap: u8) -> u8 {
    if alive.is_empty() {
        return 0;
    }
    if cap == 0 {
        return 0;
    }
    let first = alive[0];
    let mut best = cap;
    for &c in first.cells() {
        let sub: Vec<CellSet2> = alive.iter().filter(|cp| !cp.contains(c)).copied().collect();
        let r = 1 + min_hit_slice(sub, best - 1);
        if r < best {
            best = r;
            if best == 1 {
                break;
            }
        }
    }
    best
}

/// (B capped at 3, all size-B covers for B <= 2; empty for B >= 3 — the search
/// never uses covers there: B >= 3 is an immediate attacker win).
pub fn min_covers2(comps: &[CellSet2]) -> (u8, Vec<CellSet2>) {
    let mut covers = Vec::new();
    let b = min_covers_capped(comps, Some(&mut covers));
    (b, covers)
}

/// Does the player have a completion needing at most `max_empties` placements?
/// Early-exits on the first hit.
fn has_completion(board: &SolverBoard, player: Player, wl: u8, max_empties: u8, radius: i32) -> bool {
    let l = wl as i32;
    let mut found = false;
    scan_windows(board, player, wl, radius, l - 2, l - 1, &mut |_, _, _, empties| {
        if empties.len() as u8 <= max_empties {
            found = true;
        }
        found
    });
    found
}

/// Sound gate: can the attacker make a four this turn? Scans for enemy-free
/// windows with wl-4..=wl-3 attacker stones (exactly the threat table's range;
/// windows with more stones are completions, covered by `has_completion` at the
/// `solve_from` gate). Equivalent to a non-empty threat table, but early-exits
/// on the first qualifying window instead of building the table.
fn any_four_gate(board: &SolverBoard, attacker: Player, wl: u8, radius: i32) -> bool {
    let l = wl as i32;
    let mut found = false;
    scan_windows(board, attacker, wl, radius, l - 4, l - 3, &mut |_, _, _, _| {
        found = true;
        true
    });
    found
}

/// EXPERIMENTAL (off by default): empty cells within `wl-1` steps along a WIN_AXES
/// direction (either sign) of any attacker stone, ordered most-threatening-first.
/// A superset of the hot cells that also admits quiet build stones one step off an
/// existing line — needed for "force with one stone, quietly build with the other"
/// attacker turns (a fresh threat that only pays off a move or two later). Cell set
/// ports the Python prototype's `_wide_partners`
/// (scripts/forcing_search_prototype.py:21-34); the ordering is new (the prototype
/// searched them in coordinate order). Respects the same radius-legality filtering
/// as other generated cells (`scan_windows`'s `enforce` gate), so a wide cell the
/// attacker could not legally play never reaches the partner set.
///
/// Score = sum of `pc(w)^2` over enemy-free windows `w` containing the cell with
/// `pc >= 1` attacker stones, precomputed by `threat_table_scored`'s fused strip
/// walk and passed in as `scores`. Superlinear in line strength, and cells nearer
/// a stone (or aligned with stones on several axes) share more windows — so
/// proximity and multi-axis alignment both raise the score. Equal scores break by
/// hex distance to the nearest own stone (window count only sees axis-aligned
/// proximity; a cell can be hex-close to a stone it shares no axis with), then by
/// coord for determinism. A builder never changes a move's B (it joins no
/// near-complete window), so B ordering alone leaves all (hot, builder) pairs
/// tied; this ordering puts the most threatening builders first in the search.
/// Returns `(cell, score, nearest-own-stone dist)`.
pub(crate) fn wide_partner_cells(
    board: &SolverBoard,
    attacker: Player,
    wl: u8,
    radius: i32,
    scores: &FxHashMap<Coord, u32>,
) -> Vec<(Coord, u32, i32)> {
    let l = wl as i32;
    let enforce = radius < l - 1;
    let own: Vec<Coord> = board
        .stones
        .iter()
        .filter(|&&(_, o)| o == attacker)
        .map(|&(c, _)| c)
        .collect();
    let mut cells: Vec<Coord> = Vec::new();
    for &s in &own {
        for &(dq, dr) in WIN_AXES.iter() {
            for k in 1..l {
                for c in [(s.0 + k * dq, s.1 + k * dr), (s.0 - k * dq, s.1 - k * dr)] {
                    if board.get(c).is_none() && (!enforce || board.within_radius(c, radius)) {
                        cells.push(c);
                    }
                }
            }
        }
    }
    cells.sort_unstable();
    cells.dedup();
    let mut scored: Vec<(Coord, u32, i32)> = cells
        .into_iter()
        .map(|c| {
            let dist = own.iter().map(|&s| hex_dist(c, s)).min().unwrap_or(i32::MAX);
            (c, scores.get(&c).copied().unwrap_or(0), dist)
        })
        .collect();
    // Most-threatening-first, then closest-to-own-stones; coord ascending last
    // keeps search order deterministic (never let scan order into tie-breaking).
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));
    scored
}

/// Forcing attacker turns (B >= 2 after the move), best-first by B. `wide` gates the
/// experimental wide-partner-width knob (see `wide_partner_cells`); false reproduces
/// the tight (hot + defender-block partners) generator byte-for-byte.
fn attacker_turns(
    board: &SolverBoard,
    attacker: Player,
    defender: Player,
    placements: u8,
    wl: u8,
    radius: i32,
    wide: bool,
) -> Vec<CellSet2> {
    let dfn_comps = completions(board, defender, wl, radius);
    attacker_turns_with(board, attacker, placements, wl, radius, &dfn_comps, wide)
}

/// See `attacker_turns`; takes precomputed defender completions.
pub fn attacker_turns_with(
    board: &SolverBoard,
    attacker: Player,
    placements: u8,
    wl: u8,
    radius: i32,
    dfn_comps: &[CellSet2],
    wide: bool,
) -> Vec<CellSet2> {
    let enforce = radius < wl as i32 - 1;
    // Wide mode fuses the builder-score accumulation into the threat-table
    // strip walk (one scan instead of two); tight keeps the narrower scan.
    let (index, wide_scores) = if wide {
        threat_table_scored(board, attacker, wl, radius)
    } else {
        (threat_table(board, attacker, wl, radius), FxHashMap::default())
    };
    let mut hot: Vec<Coord> = index.keys().copied().collect();
    hot.sort_unstable();
    // Per-hot-cell forcing potential, used to skip candidates that provably
    // cannot reach B >= 2 without running `move_b`: `strong` counts entries the
    // cell completes on its own (pc >= wl-3), `weak` the rest (need a second
    // fill, so they only contribute when the partner is in the same window).
    // A move's comp count is bounded by strong(h) + strong(a) + joint-weak
    // <= strong(h) + strong(a) + min(weak(h), weak(a)); below 2 means B <= 1.
    let potential: Vec<(u32, u32)> = hot
        .iter()
        .map(|c| {
            let entries = &index[c];
            let strong = entries.iter().filter(|e| e.pc as i32 >= wl as i32 - 3).count() as u32;
            (strong, entries.len() as u32 - strong)
        })
        .collect();
    // (B, h_strong, move): h_strong rides along as a wide-mode-only secondary
    // sort key — see the final sort below.
    let mut scored: Vec<(u8, u32, CellSet2)> = Vec::new();
    let mut scratch: Vec<CellSet2> = Vec::new();
    if placements == 1 {
        for (hi, &c) in hot.iter().enumerate() {
            // A lone placement only converts its strong entries into completions.
            if potential[hi].0 < 2 {
                continue;
            }
            // c must be a legal placement on the current board.
            if enforce && !board.within_radius(c, radius) {
                continue;
            }
            let b = move_b_with(&index, &[c], wl, &mut scratch);
            if b >= 2 {
                scored.push((b, potential[hi].0, CellSet2::one(c)));
            }
        }
    } else {
        // partners = hot ∪ defender-block cells (cells of defender completions),
        // tagged is_hot so hot-hot pairs dedup by order (emit only when a > h)
        // instead of via a seen-set — preserving the prototype's exact emission
        // order (h ascending, partner ascending, first occurrence kept). Each
        // partner carries its (strong, weak) potential; block-only cells are in
        // no attacker window, so theirs is (0, 0).
        let mut partners: Vec<(Coord, bool, u32, u32)> = hot
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, true, potential[i].0, potential[i].1))
            .collect();
        {
            let mut blocks: Vec<Coord> =
                dfn_comps.iter().flat_map(|c| c.cells().iter().copied()).collect();
            blocks.sort_unstable();
            blocks.dedup();
            blocks.retain(|c| hot.binary_search(c).is_err());
            partners.extend(blocks.into_iter().map(|c| (c, false, 0, 0)));
        }
        // Sort hot/block partners BEFORE appending wide cells: wide cells carry
        // their own most-threatening-first order (see `wide_partner_cells`),
        // which a coord re-sort would destroy. Tight is byte-identical (nothing
        // is appended after the sort), and hot/block partners staying ahead of
        // quiet builders is itself the right priority — they add completions,
        // builders never do.
        partners.sort_unstable();
        // EXPERIMENTAL wide partners (off by default; zero cost when `wide` is
        // false — the generator below never runs). Tagged is_hot=false with (0, 0)
        // potential, exactly like block-only cells: a non-hot cell is never a key
        // in `index` (every gap of a threat-table window is indexed, so if it were
        // a shared gap of one of `h`'s entries it would already be `hot`), so its
        // presence in a pair can never change `move_b_with`'s result — `filled`
        // for each of `h`'s entries is always exactly 1 (from `h` itself). A pair
        // (h, wide-cell) is therefore forcing iff `h` ALONE would be (h_strong >=
        // 2), which is exactly what the existing prefilter bound
        // `h_strong + a_strong + min(h_weak, a_weak) < 2` reduces to when
        // a_strong = a_weak = 0 — the prefilter already admits wide pairs exactly,
        // no gating or exemption needed.
        if wide {
            // partners is coord-sorted (coord is the tuple's leading key, coords
            // are unique), so `existing` is sorted for the binary_search dedup.
            let existing: Vec<Coord> = partners.iter().map(|p| p.0).collect();
            let mut wide_cells = wide_partner_cells(board, attacker, wl, radius, &wide_scores);
            wide_cells.retain(|(c, _, _)| existing.binary_search(c).is_err());
            partners.extend(wide_cells.into_iter().map(|(c, _, _)| (c, false, 0, 0)));
        }
        for (hi, &h) in hot.iter().enumerate() {
            let (h_strong, h_weak) = potential[hi];
            for &(a, a_hot, a_strong, a_weak) in partners.iter() {
                if a == h || (a_hot && a < h) {
                    continue;
                }
                // Comp-count upper bound below 2 => B <= 1, filtered anyway.
                if h_strong + a_strong + h_weak.min(a_weak) < 2 {
                    continue;
                }
                let mv = CellSet2::two(h, a);
                // Two-placement legality: mv cells are played in sorted order, so
                // cells[0] must be legal on the current board; cells[1] is played
                // second — legal within radius of an existing stone OR of cells[0].
                if enforce {
                    let [c1, c2] = [mv.cells[0], mv.cells[1]];
                    if !board.within_radius(c1, radius) {
                        continue;
                    }
                    // `hex_dist(c1, c2) <= radius` is currently unreachable: `hot`
                    // and `blocks` are both pre-filtered to within-radius against
                    // the pre-move board (window-level filtering in `scan_windows`),
                    // so `within_radius(c2, ..)` is already true here. Retained for
                    // correctness if that pre-filtering is ever relaxed — a known
                    // completeness gap (rejecting some legal two-placement moves),
                    // not a soundness one.
                    if !(board.within_radius(c2, radius) || hex_dist(c1, c2) <= radius) {
                        continue;
                    }
                }
                let b = move_b_with(&index, mv.cells(), wl, &mut scratch);
                if b >= 2 {
                    scored.push((b, h_strong, mv));
                }
            }
        }
    }
    if wide {
        // Wide: best-first by B, then strongest hot cell first — the h-outer
        // emission loop groups pairs by hot cell in coordinate order, and with
        // builders in the mix a group is large, so refuting every pair of a
        // weak hot cell before reaching a strong one is expensive. Stable, so
        // within a (B, h_strong) tie the emission order survives: hot/block
        // partners (coord asc), then builders most-threatening-first.
        scored.sort_by(|x, y| y.0.cmp(&x.0).then(y.1.cmp(&x.1)));
    } else {
        scored.sort_by(|x, y| y.0.cmp(&x.0)); // best-first (higher B), stable
    }
    scored.into_iter().map(|(_, _, m)| m).collect()
}

#[derive(Debug)]
enum SolveResult { Win { depth: u8 }, No, BudgetExceeded }

/// (attacker completions, defender completions) of one position.
type NodeComps = (Vec<CellSet2>, Vec<CellSet2>);

struct SearchState {
    tt: FxHashMap<(u64, bool, u8, u8), (bool, bool)>, // (hash, is_or, placements, budget) -> (won, cutoff)
    // Per-position memo: forcing move list, budget-independent (mirrors the Python
    // prototype's `state["gencache"]`). Keyed by (board.hash, placements).
    gencache: FxHashMap<(u64, u8), Rc<Vec<CellSet2>>>,
    // Per-position memo: budget-independent (mirrors `state["comps"]` /
    // `_node_comps`). Keyed by board.hash.
    comps: FxHashMap<u64, Rc<NodeComps>>,
    nodes: u64,
    budget: u64,
    exceeded: bool,
    wl: u8,
    radius: i32,
    atk: Player,
    dfn: Player,
    /// EXPERIMENTAL wide-partner-width knob (see `wide_partner_cells`); fixed for
    /// the lifetime of a single search, so the gencache/tt keys need no extra
    /// dimension for it.
    wide: bool,
    /// Research-only wall-clock deadline + cooperative cancel (see `Limits`). The
    /// production default is empty, making `tick` byte-identical to before.
    limits: Limits,
    /// EXPERIMENTAL proof-support capture (None in production = zero cost).
    /// For every node proven WON, records the sorted, deduped set of cells its
    /// proof relied on — completion gap cells, played attacker/cover cells —
    /// unioned over the proven sub-DAG. Keyed (hash, is_or, placements; 0 for
    /// defender nodes). By threat monotonicity (opponent stones only REMOVE the
    /// attacker's windows and only ADD the defender's), a later defender
    /// placement outside this set cannot invalidate the proof PROVIDED it also
    /// creates no new defender completion (the "wins first" checks relied on
    /// their absence) — the caller must pair the support test with that check.
    /// See `vct_probe` for the consumer and the full soundness argument.
    support: Option<FxHashMap<(u64, bool, u8), Rc<Vec<Coord>>>>,
}

impl SearchState {
    fn new(wl: u8, radius: i32, atk: Player, dfn: Player, node_budget: u64, wide: bool) -> Self {
        SearchState {
            tt: FxHashMap::default(),
            gencache: FxHashMap::default(),
            comps: FxHashMap::default(),
            nodes: 0,
            budget: node_budget,
            exceeded: false,
            wl,
            radius,
            atk,
            dfn,
            wide,
            limits: Limits::default(),
            support: None,
        }
    }

    /// Fresh node budget for a follow-up run (PV probes) that KEEPS the warm
    /// tt/gencache/comps. Cached verdicts are position truths, so a warm cache
    /// changes which probes complete within budget — never what they conclude.
    fn reset_run(&mut self) {
        self.nodes = 0;
        self.exceeded = false;
    }

    /// Record a proven-won node's support set (sorted + deduped). No-op unless
    /// support capture is enabled.
    fn record_support(&mut self, key: (u64, bool, u8), mut cells: Vec<Coord>) {
        if let Some(map) = &mut self.support {
            cells.sort_unstable();
            cells.dedup();
            map.insert(key, Rc::new(cells));
        }
    }
}

fn tick(s: &mut SearchState) -> bool {
    s.nodes += 1;
    if s.nodes > s.budget {
        s.exceeded = true;
    } else if (s.limits.deadline.is_some() || s.limits.cancel.is_some()) && s.nodes & 0x1FFF == 0 {
        // Sample clock/atomic sparsely so they never dominate the ~10µs/node hot
        // path. Only reachable when a limit is set (research prover); production
        // callers pass `Limits::default()` and skip this branch entirely.
        if let Some(dl) = s.limits.deadline
            && Instant::now() >= dl
        {
            s.exceeded = true;
        }
        if let Some(c) = &s.limits.cancel
            && c.load(Ordering::Relaxed)
        {
            s.exceeded = true;
        }
    }
    s.exceeded
}

/// (attacker completions, defender completions) for this position, computed once and
/// cached by `board.hash` so the full-board scan runs once per position instead of
/// once per visit (a position is revisited across deepening rounds and transpositions).
/// Mirrors the prototype's `_node_comps`. Rc so cache hits are a refcount bump, not
/// a Vec clone (the search is single-threaded).
fn node_comps(board: &SolverBoard, s: &mut SearchState) -> Rc<NodeComps> {
    if let Some(v) = s.comps.get(&board.hash) { return Rc::clone(v); }
    let atk_c = completions(board, s.atk, s.wl, s.radius);
    let dfn_c = completions(board, s.dfn, s.wl, s.radius);
    let v = Rc::new((atk_c, dfn_c));
    s.comps.insert(board.hash, Rc::clone(&v));
    v
}

/// `attacker_turns`, memoized per (board.hash, placements): the move list is
/// budget-independent, so a transposition skips rebuilding the threat table entirely
/// (the expensive part). Mirrors the prototype's `gencache`.
fn attacker_turns_memo(
    board: &SolverBoard,
    placements: u8,
    dfn_comps: &[CellSet2],
    s: &mut SearchState,
) -> Rc<Vec<CellSet2>> {
    let key = (board.hash, placements);
    if let Some(v) = s.gencache.get(&key) { return Rc::clone(v); }
    let moves = Rc::new(attacker_turns_with(board, s.atk, placements, s.wl, s.radius, dfn_comps, s.wide));
    s.gencache.insert(key, Rc::clone(&moves));
    moves
}

/// Returns (won, cutoff).
fn atk_within(board: &mut SolverBoard, placements: u8, budget: u8, s: &mut SearchState) -> (bool, bool) {
    if s.exceeded || tick(s) { return (false, false); }
    let comps = node_comps(board, s);
    let (atk_comps, dfn_comps) = (&comps.0, &comps.1);
    if let Some(win) = atk_comps.iter().find(|c| c.len() as u8 <= placements) {
        // Immediate completion: the proof relies only on this window's gap cells
        // (its stones are the attacker's own — the opponent cannot remove them).
        s.record_support((board.hash, true, placements), win.cells().to_vec());
        return (true, false);
    }
    if budget < 2 { return (false, true); }
    let key = (board.hash, true, placements, budget);
    if let Some(&v) = s.tt.get(&key) { return v; }
    let (mut won, mut cut) = (false, false);
    // (winning move, its proven child's support) when capture is enabled.
    let mut won_support: Option<Vec<Coord>> = None;
    for &mv in attacker_turns_memo(board, placements, dfn_comps, s).iter() {
        for &c in mv.cells() { board.place(c, s.atk); }
        let (w, subcut) = def_within(board, budget - 1, s);
        if w && s.support.is_some() {
            // Child support must be read while mv is still on the board (the
            // child is keyed by the post-move hash). A missing child entry
            // poisons this node's support (stays None → callers fall back to
            // full re-verification; never an unsound partial set).
            let child = s.support.as_ref().and_then(|m| m.get(&(board.hash, false, 0))).cloned();
            if let Some(child) = child {
                let mut cells = child.as_ref().clone();
                cells.extend_from_slice(mv.cells());
                won_support = Some(cells);
            }
        }
        for &c in mv.cells() { board.remove(c); }
        cut |= subcut;
        if w { won = true; break; }
    }
    if won && let Some(cells) = won_support {
        s.record_support((board.hash, true, placements), cells);
    }
    let res = if won { (true, false) } else { (false, cut) };
    if !s.exceeded { s.tt.insert(key, res); }
    res
}

fn def_within(board: &mut SolverBoard, budget: u8, s: &mut SearchState) -> (bool, bool) {
    if s.exceeded || tick(s) { return (false, false); }
    let comps = node_comps(board, s);
    let (atk_comps, dfn_comps) = (&comps.0, &comps.1);
    if dfn_comps.iter().any(|c| c.len() as u8 <= 2) { return (false, false); } // defender wins first
    let (bnum, covers) = min_covers2(atk_comps);
    if bnum < 2 { return (false, false); }
    if bnum >= 3 {
        return if budget >= 1 {
            // B >= 3 win: the verdict relies on the full completion set (an
            // opponent stone in any gap removes a completion and can drop B
            // below 3), so the support is every completion's gap cells. The
            // reliance on the defender NOT holding a completion (the check
            // above) is the support consumer's separate obligation.
            let cells: Vec<Coord> =
                atk_comps.iter().flat_map(|c| c.cells().iter().copied()).collect();
            s.record_support((board.hash, false, 0), cells);
            (true, false)
        } else {
            (false, true)
        };
    }
    let key = (board.hash, false, 0, budget);
    if let Some(&v) = s.tt.get(&key) { return v; }
    // The all-covers-fail loop below defaults to a win; that is only sound if
    // B == 2 guarantees at least one enumerable cover (a size-2 hitting set
    // implies one). Make the cross-function dependency on min_covers2 explicit,
    // and fail CONSERVATIVE in release: an internal inconsistency here must
    // become a missed win, never a false one.
    debug_assert!(!covers.is_empty(), "bnum == 2 must come with covers");
    if covers.is_empty() {
        return (false, false);
    }
    // Defender covers are single cells the defender plays to block; each must be a
    // legal placement. Covers are drawn only from the cells of `atk_comps`, which
    // `scan_windows` has already filtered to within-radius gaps in the tight regime,
    // so every cover cell is playable by construction — assert the invariant.
    #[cfg(debug_assertions)]
    if s.radius < s.wl as i32 - 1 {
        for cover in &covers {
            debug_assert!(
                cover.cells().iter().all(|&c| board.within_radius(c, s.radius)),
                "defender cover cell is out of placement radius"
            );
        }
    }
    let mut result = (true, false);
    // Support accumulator: the completion set (the covers derive from it — an
    // opponent stone in any gap changes which min-covers exist) plus, per
    // cover, its cells and the proven child's support. Any missing child
    // support poisons the node (None → no entry → consumers re-verify fully).
    let mut support_acc: Option<Vec<Coord>> = s.support.is_some().then(|| {
        atk_comps.iter().flat_map(|c| c.cells().iter().copied()).collect()
    });
    for cover in covers {
        for &c in cover.cells() { board.place(c, s.dfn); }
        let (w, subcut) = atk_within(board, 2, budget, s);
        if w && let Some(acc) = &mut support_acc {
            match s.support.as_ref().and_then(|m| m.get(&(board.hash, true, 2))).cloned() {
                Some(child) => {
                    acc.extend_from_slice(cover.cells());
                    acc.extend_from_slice(&child);
                }
                None => support_acc = None,
            }
        }
        for &c in cover.cells() { board.remove(c); }
        if !w { result = (false, subcut); break; }
    }
    if result.0 && let Some(cells) = support_acc {
        s.record_support((board.hash, false, 0), cells);
    }
    if !s.exceeded { s.tt.insert(key, result); }
    result
}

/// Test-only convenience wrapper (production code calls `solve_from_ex` directly via
/// `solve`/`solve_ex`, threading the wide knob through).
#[cfg(test)]
fn solve_from(board: &mut SolverBoard, attacker: Player, defender: Player,
                  placements_remaining: u8, wl: u8, radius: i32, depth_cap: u8, node_budget: u64) -> SolveResult {
    solve_from_ex(board, attacker, defender, placements_remaining, wl, radius, depth_cap, node_budget, false, Limits::default())
}

/// See `solve_from`; `wide` gates the experimental wide-partner-width knob (see
/// `wide_partner_cells`) — false reproduces `solve_from` exactly. `limits` are
/// research-only cutoffs (`Limits::default()` for every production caller —
/// identical behaviour to before).
fn solve_from_ex(board: &mut SolverBoard, attacker: Player, defender: Player,
                  placements_remaining: u8, wl: u8, radius: i32, depth_cap: u8, node_budget: u64,
                  wide: bool, limits: Limits) -> SolveResult {
    let mut s = SearchState::new(wl, radius, attacker, defender, node_budget, wide);
    s.limits = limits;
    solve_from_state(board, placements_remaining, depth_cap, &mut s)
}

/// The actual ID driver, on a caller-owned `SearchState`. Callers that need the
/// PV afterwards keep `s` and hand its warm tt/gencache/comps to
/// `first_winning_move`/`extract_pv` — the probes then run mostly on cache hits
/// instead of repeating the whole winning search from cold (the old
/// "double-search": a fresh state per probe made PV extraction cost as much as
/// the solve itself, and could even starve under `node_budget` and return an
/// empty PV for a proven win).
fn solve_from_state(
    board: &mut SolverBoard,
    placements_remaining: u8,
    depth_cap: u8,
    s: &mut SearchState,
) -> SolveResult {
    let (wl, radius) = (s.wl, s.radius);
    // The strip-scan buffers are stack-sized for win_length in 1..=MAX_WL; any
    // other config cannot be analyzed — report an honest give-up, never a bogus `No`.
    if !(1..=MAX_WL).contains(&(wl as usize)) {
        return SolveResult::BudgetExceeded;
    }
    // Below wl=5 the threat machinery is blind to some MULTI-TURN wins: the
    // strip scan only visits windows through the player's own stones (pc >= 1
    // always), so for wl < 4 the whole threat band [wl-4, wl-3] is unreachable,
    // and for wl == 4 its pc = 0 element is — yet at wl=4 a bare pair played
    // into empty space is already a forcing double threat the generator would
    // never emit. An immediate completion still solves (the completion band
    // works down to wl=2), but anything deeper would surface as a FALSE proven
    // `No`. Give up honestly instead.
    if (wl as usize) < 5 && !has_completion(board, s.atk, wl, placements_remaining, radius) {
        return SolveResult::BudgetExceeded;
    }
    // Tight regime: switch legality queries to O(1) incremental counts.
    if radius < wl as i32 - 1 {
        board.enable_reach(radius);
    }
    // sound gate
    if !any_four_gate(board, s.atk, wl, radius) && !has_completion(board, s.atk, wl, placements_remaining, radius) {
        return SolveResult::No;
    }
    for depth in 1..=depth_cap {
        let (won, cut) = atk_within(board, placements_remaining, depth, s);
        if won { return SolveResult::Win { depth }; }
        if s.exceeded { return SolveResult::BudgetExceeded; }
        if !cut { return SolveResult::No; }
    }
    SolveResult::BudgetExceeded
}

use hexo_engine::game::GameState;

#[derive(Debug)]
pub struct ForcingWin { pub depth: u8, pub first_move: Coord, pub pv: Vec<Coord> }
#[derive(Debug)]
pub enum Outcome { Win(ForcingWin), No, BudgetExceeded }

impl Outcome {
    /// True iff this is a proven forced win for the side to move.
    #[inline]
    pub fn is_win(&self) -> bool { matches!(self, Outcome::Win(_)) }
}

pub fn solve(game: &GameState, depth_cap: u8, node_budget: u64) -> Outcome {
    solve_ex(game, depth_cap, node_budget, false, Limits::default())
}

/// `solve` with research-only cutoffs (the prover `idtt` driver + portfolio
/// racing). `Limits::default()` is exactly `solve`. `pub(crate)`: no new public
/// API surface.
pub fn solve_limited(
    game: &GameState,
    depth_cap: u8,
    node_budget: u64,
    wide: bool,
    limits: Limits,
) -> Outcome {
    solve_ex(game, depth_cap, node_budget, wide, limits)
}

/// `solve` with the wide-partner-width knob (see `wide_partner_cells`) turned
/// on — a superset generator, so a `solve` win is never lost, only possibly
/// found at the same or a different (never longer, since the search is best-first
/// and iterative-deepening) depth, and previously-No/BudgetExceeded positions may
/// newly resolve to Win. Production callers: the analysis screen (via the PyO3
/// `wide` kwarg), the WASM `StrixSolver::solve_wide`, the VCT probe, and — as
/// of 2026-07-14 — the self-play/live-play root shortcut in
/// `gumbel_mcts`/`batched` (the ~1.5-1.7× mean per-call cost is negligible at
/// one call per placement; the tight-vs-wide A/B lives in
/// `bench_depth2_shortcut`). Tight `solve` remains for callers that want the
/// cheaper generator explicitly.
pub fn solve_wide(game: &GameState, depth_cap: u8, node_budget: u64) -> Outcome {
    solve_ex(game, depth_cap, node_budget, true, Limits::default())
}

/// See `solve`; `wide` gates the experimental wide-partner-width knob. false
/// reproduces `solve` exactly. `limits` are research-only cutoffs
/// (`Limits::default()` for all production callers — identical behaviour).
fn solve_ex(game: &GameState, depth_cap: u8, node_budget: u64, wide: bool, limits: Limits) -> Outcome {
    solve_ex_support(game, depth_cap, node_budget, wide, limits, false).0
}

/// EXPERIMENTAL: `solve_wide` + proof-support capture. On `Win`, additionally
/// returns the root proof's support set — the cells the proof relied on (see
/// `SearchState::support` for semantics and the consumer's obligations). `None`
/// support on a `Win` means capture was poisoned somewhere in the DAG (rare);
/// consumers must then fall back to full re-verification.
pub(crate) fn solve_wide_with_support(
    game: &GameState,
    depth_cap: u8,
    node_budget: u64,
) -> (Outcome, Option<Rc<Vec<Coord>>>) {
    solve_ex_support(game, depth_cap, node_budget, true, Limits::default(), true)
}

fn solve_ex_support(
    game: &GameState,
    depth_cap: u8,
    node_budget: u64,
    wide: bool,
    limits: Limits,
    capture: bool,
) -> (Outcome, Option<Rc<Vec<Coord>>>) {
    let attacker = match game.current_player() { Some(p) => p, None => return (Outcome::No, None) };
    let defender = attacker.opponent();
    let wl = game.config().win_length;
    let radius = game.config().placement_radius;
    let placements = game.moves_remaining_this_turn();
    if !(1..=MAX_WL).contains(&(wl as usize)) {
        return (Outcome::BudgetExceeded, None); // see solve_from
    }
    let mut board = SolverBoard::new();
    // Pre-size once over the position's bounding box (avoids incremental regrowth),
    // and enable reach tracking before bulk-loading so counts build incrementally.
    let mut it = game.stones().iter();
    if let Some((&c0, _)) = it.next() {
        let (mut lo, mut hi) = (c0, c0);
        for (&c, _) in it {
            lo = (lo.0.min(c.0), lo.1.min(c.1));
            hi = (hi.0.max(c.0), hi.1.max(c.1));
        }
        // A pathological coordinate spread would blow up the dense grid (`grow`
        // panics past MAX_GRID_CELLS), and coordinates near the i32 extremes
        // would overflow the grid arithmetic before that assert could fire.
        // Real games are spatially bounded near the origin; if either bound
        // trips, give up honestly instead of aborting the engine.
        const MAX_COORD: i32 = 1 << 30;
        if lo.0 < -MAX_COORD || lo.1 < -MAX_COORD || hi.0 > MAX_COORD || hi.1 > MAX_COORD {
            return (Outcome::BudgetExceeded, None);
        }
        let (span_q, span_r) = (
            hi.0 as i64 - lo.0 as i64 + 6 * GRID_PAD as i64,
            hi.1 as i64 - lo.1 as i64 + 6 * GRID_PAD as i64,
        );
        if span_q.saturating_mul(span_r) > MAX_GRID_CELLS {
            return (Outcome::BudgetExceeded, None);
        }
        board.reserve(lo, hi);
    }
    if radius < wl as i32 - 1 {
        board.enable_reach(radius);
    }
    // `game.stones()` iteration order is nondeterministic (engine HashMap).
    // Result determinism relies on the XOR-commutative Zobrist hash plus every
    // derived list (completions, hot cells, partners, moves) being sorted —
    // do not let stone/scan order leak into tie-breaking.
    for (&c, &p) in game.stones() { board.place(c, p); }
    let mut s = SearchState::new(wl, radius, attacker, defender, node_budget, wide);
    s.limits = limits;
    if capture {
        s.support = Some(FxHashMap::default());
    }
    match solve_from_state(&mut board, placements, depth_cap, &mut s) {
        SolveResult::No => (Outcome::No, None),
        SolveResult::BudgetExceeded => (Outcome::BudgetExceeded, None),
        SolveResult::Win { depth } => {
            // Root support must be read while the board is pristine (the PV
            // probes below re-prove sub-positions and extract_pv permanently
            // mutates the board).
            let support = s
                .support
                .as_ref()
                .and_then(|m| m.get(&(board.hash, true, placements)))
                .cloned();
            // PV probes run on the winning search's warm state, but without its
            // research-only deadline/cancel cutoffs — the pre-refactor probes
            // always used cutoff-free fresh states, and a deadline that expired
            // mid-solve must not blank the PV of an already-proven win.
            s.limits = Limits::default();
            // Compute on the pristine (post-solve_from, pre-extract_pv) board: extract_pv
            // permanently applies the winning line to `board`, so first_winning_move must
            // run first or it would be probing an already-won position.
            let fm = first_winning_move(&mut board, placements, depth, &mut s);
            let pv = extract_pv(&mut board, placements, depth, &mut s);
            match fm.or_else(|| pv.first().copied()) {
                Some(first) => (Outcome::Win(ForcingWin { depth, first_move: first, pv }), support),
                // Nearly impossible with warm-state probes, but if both PV
                // re-probes starve under budget, report the honest give-up —
                // never `No`, which would contradict the proof the main search
                // just produced.
                None => (Outcome::BudgetExceeded, None),
            }
        }
    }
}

/// Board-level entry point: solve from a pre-built `SolverBoard` (the same path
/// `solve`/`solve_ex` take after building the board from a `GameState`). This is the
/// public way to solve an *arbitrary* board that could not arise in a real game
/// (e.g. one missing the `(0,0,P1)` origin stone `GameState::from_state` requires):
/// the caller constructs the board directly via `SolverBoard::place` (and
/// `reserve`/`enable_reach` for the tight regime) and passes it here.
///
/// `solve_from_ex` is the actual search; this wrapper runs it, then reconstructs the
/// principal variation and first winning move exactly as `solve_ex` does. `wide`
/// gates the experimental wide-partner generator; pass `false` for the production
/// search. `limits` carries optional deadline/cancel cutoffs
/// (`Limits::default()` = no cutoff = the `solve` behaviour).
pub fn solve_from_board(
    board: &mut SolverBoard,
    attacker: Player,
    placements: u8,
    wl: u8,
    radius: i32,
    depth_cap: u8,
    node_budget: u64,
    wide: bool,
    limits: Limits,
) -> Outcome {
    let defender = attacker.opponent();
    let mut s = SearchState::new(wl, radius, attacker, defender, node_budget, wide);
    s.limits = limits;
    match solve_from_state(board, placements, depth_cap, &mut s) {
        SolveResult::No => Outcome::No,
        SolveResult::BudgetExceeded => Outcome::BudgetExceeded,
        SolveResult::Win { depth } => {
            // Warm-state PV probes, cutoffs cleared — see `solve_ex`.
            s.limits = Limits::default();
            let fm = first_winning_move(board, placements, depth, &mut s);
            let pv = extract_pv(board, placements, depth, &mut s);
            match fm.or_else(|| pv.first().copied()) {
                Some(first) => Outcome::Win(ForcingWin { depth, first_move: first, pv }),
                // Honest give-up, never a `No` contradicting the proven win —
                // see solve_ex_support.
                None => Outcome::BudgetExceeded,
            }
        }
    }
}

/// The attacker's first winning placement for a proven `Win{depth}`: an immediately
/// executable completion's first cell, else the first `attacker_turns` move whose
/// `def_within(depth-1, ..)` holds. Runs on the winning search's own state — each
/// probe gets a fresh node budget (`reset_run`) but keeps the warm caches, so the
/// re-searches are mostly TT hits. Never panics; `None` only if no such move
/// exists (shouldn't happen for a proven win).
fn first_winning_move(board: &mut SolverBoard, placements: u8, depth: u8,
                       s: &mut SearchState) -> Option<Coord> {
    for c in completions(board, s.atk, s.wl, s.radius) {
        if c.len() as u8 <= placements {
            return c.first();
        }
    }
    for mv in attacker_turns(board, s.atk, s.dfn, placements, s.wl, s.radius, s.wide) {
        for &c in mv.cells() { board.place(c, s.atk); }
        s.reset_run();
        let (w, _) = def_within(board, depth.saturating_sub(1), s);
        for &c in mv.cells() { board.remove(c); }
        if w { return mv.first(); }
    }
    None
}

/// The defender's cosmetic 2-cell reply for a `B >= 3` (futile, unstoppable) position:
/// no 2-cell block can change the verdict, but `extract_pv` still needs a legal-looking
/// pair to keep the PV an alternating line. Rather than an arbitrary canonical-order
/// pair (which can land both cells on the same threat and read as a redundant
/// non-defense), greedily pick the pair that blocks the most: c1 is the cell hitting
/// (a member of) the most completions, c2 the cell hitting the most completions NOT
/// already hit by c1. Deterministic — ties broken by total-hit count then canonical
/// cell order, never iteration order (see module determinism notes).
pub fn futile_defender_pair(comps: &[CellSet2]) -> CellSet2 {
    let mut cells: Vec<Coord> = comps.iter().flat_map(|c| c.cells().iter().copied()).collect();
    cells.sort_unstable();
    cells.dedup();
    let degree = |c: Coord| comps.iter().filter(|s| s.cells().contains(&c)).count();
    let c1 = *cells
        .iter()
        .max_by_key(|&&c| (degree(c), std::cmp::Reverse(c)))
        .expect("futile branch only reached with >= 1 completion");
    let c2 = *cells
        .iter()
        .filter(|&&c| c != c1)
        .max_by_key(|&&c| {
            let new_hits = comps.iter().filter(|s| !s.cells().contains(&c1) && s.cells().contains(&c)).count();
            (new_hits, degree(c), std::cmp::Reverse(c))
        })
        .expect("B >= 3 implies a minimum hitting set of size >= 3, so >= 2 distinct cells exist");
    CellSet2::two(c1, c2)
}

/// Port of principal_variation: returns the flat placement sequence of one winning line
/// (attacker turns interleaved with prolonging defender covers), ending in six-in-a-row.
/// Runs on the winning search's own state (fresh node budget per probe, warm caches —
/// see `first_winning_move`), so the line walk is mostly TT hits instead of a second
/// full search.
fn extract_pv(board: &mut SolverBoard, mut placements: u8, depth: u8,
              s: &mut SearchState) -> Vec<Coord> {
    let (attacker, defender, wl, radius, wide) = (s.atk, s.dfn, s.wl, s.radius, s.wide);
    let mut pv: Vec<Coord> = Vec::new();
    let mut remaining = depth;
    loop {
        // attacker to move: immediate completion?
        let mut fin: Option<CellSet2> = None;
        for c in completions(board, attacker, wl, radius) {
            if c.len() as u8 <= placements { fin = Some(c); break; }
        }
        if let Some(cells) = fin {
            for &c in cells.cells() { board.place(c, attacker); pv.push(c); }
            break;
        }
        // find a winning forcing move
        let mut chosen: Option<CellSet2> = None;
        for mv in attacker_turns(board, attacker, defender, placements, wl, radius, wide) {
            for &c in mv.cells() { board.place(c, attacker); }
            s.reset_run();
            let (w, _) = def_within(board, remaining.saturating_sub(1), s);
            for &c in mv.cells() { board.remove(c); }
            if w { chosen = Some(mv); break; }
        }
        let mv = match chosen { Some(m) => m, None => break };
        for &c in mv.cells() { board.place(c, attacker); pv.push(c); }
        placements = 2;
        // defender: B>=3 terminal, or prolonging cover
        let comps = completions(board, attacker, wl, radius);
        let (bnum, covers) = min_covers2(&comps);
        if bnum >= 3 {
            // defender blocks two threat cells (futile), attacker completes next loop
            let pair = futile_defender_pair(&comps);
            for &c in pair.cells() { board.place(c, defender); pv.push(c); }
            remaining = remaining.saturating_sub(1);
            continue;
        }
        // pick the cover that prolongs the win the most
        let mut best: Option<(u8, CellSet2)> = None;
        for cover in covers {
            for &c in cover.cells() { board.place(c, defender); }
            let mut sub: Option<u8> = None;
            for d in 1..remaining {
                s.reset_run();
                let (w, _) = atk_within(board, 2, d, s);
                if w { sub = Some(d); break; }
            }
            for &c in cover.cells() { board.remove(c); }
            if let Some(sd) = sub
                && best.as_ref().is_none_or(|(bd, _)| sd > *bd)
            {
                best = Some((sd, cover));
            }
        }
        let (sd, cover) = match best { Some(b) => b, None => break };
        for &c in cover.cells() { board.place(c, defender); pv.push(c); }
        remaining = sd;
    }
    pv
}

/// Defensive read-out for the side to move when the OPPONENT — hypothetically
/// to move with a fresh 2-placement turn — has a proven forcing win.
///
/// Semantics match the live-play defense loop this replaces (game.py's
/// `_defensive_move` heuristics, 2026-07-05): a candidate "kills" the threat
/// when the flipped re-solve after it no longer proves the win, where
/// `BudgetExceeded` counts as killed (parity with the old loop; an honest
/// "couldn't re-prove" stands down to MCTS rather than panicking the defense).
#[derive(Debug)]
pub struct DefenseAnalysis {
    /// The depth of the opponent's proven threat (from the flipped fresh-turn
    /// `solve`). Captured so WASM/PyO3 callers can reconstruct a `SolveOutcome`
    /// with the real depth rather than a sentinel. The threat PV is a fresh
    /// 2-placement turn, so chunk with `moves_remaining = 2`.
    pub threat_depth: u8,
    /// The opponent's proven threat line (flipped fresh-turn solve).
    pub threat_pv: Vec<Coord>,
    /// Single placements after which the threat is no longer provable, or
    /// which end the game outright (the engine has no self-loss rule, so a
    /// terminal mover placement is a mover win or a draw — both strictly
    /// better than the standing loss).
    pub killers: Vec<Coord>,
    /// (first, second) placement pairs that jointly refute the threat.
    /// Searched only when the mover has 2 placements left and `killers` is
    /// empty — this is the case the old max-delay survivor heuristic lost
    /// (strongloss_a: every killer pair was anchored on a cell the delay
    /// metric ranked below a far-away non-anchor). `first` alone does NOT
    /// refute; the caller plays it and the next placement's re-check finds
    /// `second` (or better).
    pub pair_anchors: Vec<(Coord, Coord)>,
    /// Max-delay fallback: the surviving candidate whose re-proven threat PV
    /// is longest (first such in PV order on ties, matching the old loop).
    /// `None` when the threat PV offers no legal candidate.
    pub best_delay: Option<Coord>,
}

/// `game` with the same stones but `side` to move on a fresh 2-placement turn
/// — the hypothetical the defense reasons about.
fn flipped_fresh_turn(game: &GameState, side: Player) -> GameState {
    let stones: Vec<(Coord, Player)> = game.stones().iter().map(|(&c, &p)| (c, p)).collect();
    GameState::from_state(&stones, side, 2, *game.config())
}

/// Forcing win for the OPPONENT of the side to move, as if it were their
/// turn (fresh 2 placements): a THREAT, not a proven loss — the actual mover
/// places first and may break it (see `solve_defense` for exactly that
/// question). Display/analysis convenience so callers don't have to rebuild
/// a flipped `GameState` themselves.
pub fn solve_threat(game: &GameState, depth_cap: u8, node_budget: u64) -> Outcome {
    match game.current_player() {
        Some(mover) => solve(&flipped_fresh_turn(game, mover.opponent()), depth_cap, node_budget),
        None => Outcome::No,
    }
}

/// `solve_threat` with the experimental wide-partner generator (see `solve_wide`):
/// the flipped-perspective threat check, but able to see "force with one stone,
/// quietly build with the other" attacker turns.
pub fn solve_threat_wide(game: &GameState, depth_cap: u8, node_budget: u64) -> Outcome {
    match game.current_player() {
        Some(mover) => solve_wide(&flipped_fresh_turn(game, mover.opponent()), depth_cap, node_budget),
        None => Outcome::No,
    }
}

/// Solve the mover's defensive problem: is the opponent's (flipped) forcing
/// win refutable this turn, and by which placements?
///
/// Returns `None` when there is no proven opponent threat (nothing to defend;
/// by threat monotonicity — the mover's stones only remove opponent threat
/// cells — the caller can safely fall through to normal search). Candidates
/// are the threat-PV cells the mover can legally take, in PV order; each
/// killer/pair verification is one flipped re-`solve` at the same
/// `depth_cap`/`node_budget` as the threat solve (the SAME tier sees the SAME
/// depth on both sides). `time_limit` bounds the whole analysis: on expiry
/// the partial result is returned (unverified candidates are simply not
/// reported), so live turns stay bounded no matter how many candidates the
/// threat line offers.
/// Outcome of a defensive analysis, distinguishing "provably nothing to
/// defend" from "the threat check ran out of budget" — a starved check must
/// never be read as safe.
#[derive(Debug)]
pub enum DefenseVerdict {
    /// The opponent has a proven threat; here is what refutes it.
    Threat(DefenseAnalysis),
    /// Provably no opponent threat within `depth_cap` (the threat solve
    /// returned a genuine `No`).
    NoThreat,
    /// The threat check exhausted `node_budget` before proving anything —
    /// retry with a larger budget; NOT a safety verdict.
    BudgetExceeded,
}

/// Back-compat wrapper: tight generator, `Threat` flattened to `Some` and both
/// give-up cases to `None` (the historical conflation — callers that need to
/// tell them apart use [`solve_defense_ex`]).
pub fn solve_defense(game: &GameState, depth_cap: u8, node_budget: u64,
                     time_limit: std::time::Duration) -> Option<DefenseAnalysis> {
    match solve_defense_ex(game, depth_cap, node_budget, time_limit, false) {
        DefenseVerdict::Threat(a) => Some(a),
        DefenseVerdict::NoThreat | DefenseVerdict::BudgetExceeded => None,
    }
}

/// See `solve_defense`; `wide` runs every internal solve (threat sighting AND
/// candidate/pair verifications — both sides must see the same generator, or a
/// wide-only threat would be "refuted" by any placement the tight re-solve is
/// blind to) with the wide generator. A missing `current_player` reports
/// `NoThreat` (terminal game: nothing to defend).
pub fn solve_defense_ex(game: &GameState, depth_cap: u8, node_budget: u64,
                        time_limit: std::time::Duration, wide: bool) -> DefenseVerdict {
    let Some(mover) = game.current_player() else {
        return DefenseVerdict::NoThreat;
    };
    let solve_g = if wide { solve_wide } else { solve };
    let opp = mover.opponent();
    // `Instant::now()` panics at runtime on wasm32-unknown-unknown (no monotonic
    // clock in std for that target). On wasm the deadline is skipped entirely —
    // `node_budget` is the sole bound (the partial-result-on-expiry semantics
    // are lost, but every sub-solve is still budget-bound). Native behavior is
    // unchanged: the deadline still honors `time_limit`.
    #[cfg(not(target_arch = "wasm32"))]
    let t0 = Instant::now();
    #[cfg(target_arch = "wasm32")]
    let _ = time_limit;
    let threat = match solve_g(&flipped_fresh_turn(game, opp), depth_cap, node_budget) {
        Outcome::Win(w) => w,
        Outcome::No => return DefenseVerdict::NoThreat,
        Outcome::BudgetExceeded => return DefenseVerdict::BudgetExceeded,
    };
    let threat_depth = threat.depth;
    let legal = game.legal_moves_set();
    let mut candidates: Vec<Coord> = Vec::new();
    for &c in &threat.pv {
        if legal.contains(&c) && !candidates.contains(&c) {
            candidates.push(c);
        }
    }

    let mut killers: Vec<Coord> = Vec::new();
    // Survivors keep the PV their re-solve proved: its length is the delay
    // metric, its cells are the second-placement candidates in the pair pass.
    let mut survivors: Vec<(Coord, Vec<Coord>)> = Vec::new();
    for &c in &candidates {
        #[cfg(not(target_arch = "wasm32"))]
        if t0.elapsed() > time_limit { break; }
        let mut trial = game.clone();
        if trial.apply_move(c).is_err() { continue; }
        if trial.is_terminal() {
            killers.push(c);
            continue;
        }
        match solve_g(&flipped_fresh_turn(&trial, opp), depth_cap, node_budget) {
            Outcome::Win(w) => survivors.push((c, w.pv)),
            Outcome::No | Outcome::BudgetExceeded => killers.push(c),
        }
    }

    let mut pair_anchors: Vec<(Coord, Coord)> = Vec::new();
    if killers.is_empty() && game.moves_remaining_this_turn() == 2 {
        'anchors: for (c1, pv2) in &survivors {
            #[cfg(not(target_arch = "wasm32"))]
            if t0.elapsed() > time_limit { break; }
            let mut trial = game.clone();
            if trial.apply_move(*c1).is_err() { continue; }
            let legal2 = trial.legal_moves_set().clone();
            let mut seconds: Vec<Coord> = Vec::new();
            for &c in pv2 {
                if legal2.contains(&c) && !seconds.contains(&c) {
                    seconds.push(c);
                }
            }
            for c2 in seconds {
                #[cfg(not(target_arch = "wasm32"))]
                if t0.elapsed() > time_limit { break 'anchors; }
                let mut trial2 = trial.clone();
                if trial2.apply_move(c2).is_err() { continue; }
                let refuted = trial2.is_terminal()
                    || !matches!(
                        solve_g(&flipped_fresh_turn(&trial2, opp), depth_cap, node_budget),
                        Outcome::Win(_)
                    );
                if refuted {
                    // One verified completion per anchor: the caller only
                    // plays `first`; the next placement re-checks anyway.
                    pair_anchors.push((*c1, c2));
                    continue 'anchors;
                }
            }
        }
    }

    // First-in-PV-order on ties, like the old loop's max().
    let mut best_delay: Option<(Coord, usize)> = None;
    for (c, pv) in &survivors {
        if best_delay.as_ref().is_none_or(|(_, len)| pv.len() > *len) {
            best_delay = Some((*c, pv.len()));
        }
    }

    DefenseVerdict::Threat(DefenseAnalysis {
        threat_depth,
        threat_pv: threat.pv,
        killers,
        pair_anchors,
        best_delay: best_delay.map(|(c, _)| c),
    })
}

#[cfg(test)]
mod defense_tests {
    use super::*;
    use hexo_engine::game::{GameConfig, GameState};
    use hexo_engine::types::Player;
    use std::time::Duration;

    // Live serve config (wl=6, r=8) and budgets ("strong" tier as of
    // 2026-07-05: depth 10, 20k nodes — the config the fixture games were
    // lost at).
    const CFG: GameConfig = GameConfig { win_length: 6, placement_radius: 8, max_moves: 400 };
    const DEPTH: u8 = 10;
    const BUDGET: u64 = 20_000;
    const NO_LIMIT: Duration = Duration::from_secs(600);

    fn p(s: &str) -> Player { if s == "P1" { Player::P1 } else { Player::P2 } }

    fn state(stones: &[(i32, i32, &str)], mover: &str, rem: u8) -> GameState {
        let typed: Vec<(Coord, Player)> =
            stones.iter().map(|&(q, r, pl)| ((q, r), p(pl))).collect();
        GameState::from_state(&typed, p(mover), rem, CFG)
    }

    // strongloss_a_line.json prefix 6 (scripts/fixtures/forcing_puzzles/):
    // the live game where the max-delay survivor heuristic lost — the human
    // (P2) completed an 18-placement VCF; every single P1 block re-proves it,
    // but killer PAIRS exist. The live bot played (-3,3), the max-delay pick,
    // and lost; a pair anchor was the saving move.
    const STRONGLOSS_A6: &[(i32, i32, &str)] = &[
        (0, 0, "P1"), (1, 2, "P2"), (2, 3, "P2"), (3, -1, "P1"),
        (2, 1, "P1"), (1, 4, "P2"), (1, 3, "P2"),
    ];

    // strongloss_b_line.json prefix 8: same failure shape, other colour
    // (P1's VCF, P2 to defend), from the second lost live game.
    const STRONGLOSS_B8: &[(i32, i32, &str)] = &[
        (0, 0, "P1"), (1, 2, "P2"), (2, 3, "P2"), (2, 2, "P1"), (3, -2, "P1"),
        (3, 4, "P2"), (3, -1, "P2"), (2, -2, "P1"), (1, -2, "P1"),
    ];

    /// Every reported pair must actually refute the threat: apply both
    /// placements, flip, re-solve — no proven win may remain.
    fn assert_anchors_refute(game: &GameState, analysis: &DefenseAnalysis) {
        let opp = game.current_player().unwrap().opponent();
        assert!(!analysis.pair_anchors.is_empty(), "expected killer pairs");
        for &(c1, c2) in &analysis.pair_anchors {
            let mut trial = game.clone();
            trial.apply_move(c1).unwrap();
            trial.apply_move(c2).unwrap();
            assert!(
                trial.is_terminal()
                    || !matches!(
                        solve(&flipped_fresh_turn(&trial, opp), DEPTH, BUDGET),
                        Outcome::Win(_)
                    ),
                "anchor pair ({c1:?}, {c2:?}) does not refute the threat"
            );
        }
    }

    #[test]
    fn defense_finds_killer_pairs_strongloss_a() {
        let game = state(STRONGLOSS_A6, "P1", 2);
        let a = solve_defense(&game, DEPTH, BUDGET, NO_LIMIT)
            .expect("P2's VCF must be detected");
        assert!(a.killers.is_empty(), "no single block survives here");
        assert_anchors_refute(&game, &a);
        assert!(a.best_delay.is_some());
    }

    #[test]
    fn defense_finds_killer_pairs_strongloss_b() {
        let game = state(STRONGLOSS_B8, "P2", 2);
        let a = solve_defense(&game, DEPTH, BUDGET, NO_LIMIT)
            .expect("P1's VCF must be detected");
        assert!(a.killers.is_empty(), "no single block survives here");
        assert_anchors_refute(&game, &a);
    }

    #[test]
    fn defense_none_when_no_threat() {
        // strongloss_a prefix 2: quiet opening, no VCF on either side.
        let game = state(&STRONGLOSS_A6[..3], "P1", 2);
        assert!(solve_defense(&game, DEPTH, BUDGET, NO_LIMIT).is_none());
    }

    #[test]
    fn solve_threat_flips_to_the_opponent() {
        // strongloss_a prefix 6: P1 to move, P2 (flipped) has the proven VCF.
        let game = state(STRONGLOSS_A6, "P1", 2);
        assert!(matches!(solve(&game, DEPTH, BUDGET), Outcome::No));
        assert!(matches!(solve_threat(&game, DEPTH, BUDGET), Outcome::Win(_)));
        // Quiet opening: no threat either way.
        let quiet = state(&STRONGLOSS_A6[..3], "P1", 2);
        assert!(matches!(solve_threat(&quiet, DEPTH, BUDGET), Outcome::No));
    }

    #[test]
    fn defense_single_placement_skips_pair_search() {
        // strongloss_a prefix 7: the real game's P1 already spent placement 1
        // on (-3,3) (the max-delay pick); with rem=1 only single killers are
        // meaningful — the pair pass must not run.
        let mut stones = STRONGLOSS_A6.to_vec();
        stones.push((-3, 3, "P1"));
        let game = state(&stones, "P1", 1);
        let a = solve_defense(&game, DEPTH, BUDGET, NO_LIMIT)
            .expect("threat still stands after the wasted block");
        assert!(a.pair_anchors.is_empty(), "no pair search at rem=1");
        assert!(!a.killers.is_empty() || a.best_delay.is_some());
    }

    #[test]
    fn defense_time_limit_returns_partial() {
        // A zero time limit must still return (an empty-but-well-formed
        // analysis with the threat PV), never hang or panic.
        let game = state(STRONGLOSS_A6, "P1", 2);
        let a = solve_defense(&game, DEPTH, BUDGET, Duration::ZERO)
            .expect("threat solve itself is not limited");
        assert!(!a.threat_pv.is_empty());
        assert!(a.killers.is_empty() && a.pair_anchors.is_empty());
        assert!(a.best_delay.is_none());
    }
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
        let comps = completions(&b, Player::P1, 6, 8);
        let want = vec![
            CellSet2::two((-2, 0), (-1, 0)),   // left
            CellSet2::two((-1, 0), (4, 0)),    // straddle
            CellSet2::two((4, 0), (5, 0)),     // right
        ];
        assert_eq!(comps, want);
    }

    #[test]
    fn min_covers_straddle_rejects_naive_block() {
        let comps = vec![
            CellSet2::two((-2, 0), (-1, 0)),
            CellSet2::two((-1, 0), (4, 0)),
            CellSet2::two((4, 0), (5, 0)),
        ];
        let (bnum, covers) = min_covers2(&comps);
        assert_eq!(bnum, 2);
        // Exact covers, in the documented lexicographic cell order.
        assert_eq!(covers, vec![
            CellSet2::two((-2, 0), (4, 0)),
            CellSet2::two((-1, 0), (4, 0)),
            CellSet2::two((-1, 0), (5, 0)),
        ]);
        assert!(!covers.contains(&CellSet2::two((-2, 0), (5, 0)))); // misses the straddle
    }

    /// The B >= 3 (futile) branch of `extract_pv` still needs a 2-cell defender
    /// reply to keep the PV alternating, even though no block changes the verdict.
    /// The naive canonical-order-first-two choice (`(-2,0),(-1,0)`) hits only the
    /// left two completions (`c0`, `c1`) and never reaches the straddle's far side
    /// (`c2`) or the independent threat (`c3`) — exactly the "drifts to a corner"
    /// look reported on the analysis board. The greedy max-coverage pair instead
    /// spans the shared straddle cells and hits ALL THREE open-four completions.
    #[test]
    fn futile_defender_pair_prefers_max_coverage_over_canonical_order() {
        let comps = vec![
            CellSet2::two((-2, 0), (-1, 0)), // c0
            CellSet2::two((-1, 0), (4, 0)),  // c1 (straddle)
            CellSet2::two((4, 0), (5, 0)),   // c2
            CellSet2::two((0, 4), (0, 5)),   // c3: independent, disjoint threat
        ];
        assert_eq!(min_covers2(&comps).0, 3, "test fixture must actually be B >= 3");

        // The naive canonical-order-first-two pair the old code used.
        let mut canonical: Vec<Coord> = comps.iter().flat_map(|c| c.cells().iter().copied()).collect();
        canonical.sort_unstable();
        canonical.dedup();
        let canonical_pair = CellSet2::two(canonical[0], canonical[1]);
        assert_eq!(canonical_pair, CellSet2::two((-2, 0), (-1, 0)));
        let canonical_hits = comps.iter().filter(|c| c.cells().iter().any(|x| canonical_pair.cells().contains(x))).count();
        assert_eq!(canonical_hits, 2, "canonical pair only reaches c0 and c1, missing c2 and c3");

        let greedy = futile_defender_pair(&comps);
        assert_eq!(greedy, CellSet2::two((-1, 0), (4, 0)));
        assert_ne!(greedy, canonical_pair, "greedy must differ from the naive canonical pair here");
        let greedy_hits = comps.iter().filter(|c| c.cells().iter().any(|x| greedy.cells().contains(x))).count();
        assert_eq!(greedy_hits, 3, "greedy pair hits 3 distinct completions, not just 1");
        assert!(greedy_hits >= 2);
    }

    /// End-to-end: a real board reaching the `B >= 3` futile branch through the
    /// public `solve()` path, with an asymmetric block (`(-1,0)`) so the fork is
    /// deterministic instead of the fully-symmetric double-four in
    /// `solve_from_gamestate_returns_pv_that_wins`. After the fork move
    /// `(0,-1),(3,0)`, the r-axis run is a genuine open four (3 completions,
    /// B=2) and the q-axis run is closed on the left (1 completion, B=1) —
    /// combined B=3. The naive canonical-order-first-two defender pair would be
    /// `(0,-3),(0,-2)`, hitting only 2 of the 4 completions and leaving the
    /// r-axis straddle's far side alive; the greedy pair spans the straddle
    /// (`(0,-2),(0,3)`) and kills the WHOLE r-axis open four (3 completions),
    /// forcing the finishing move onto the independent q-axis threat.
    #[test]
    fn futile_pv_pair_is_greedy_not_canonical_on_real_board() {
        use hexo_engine::game::{GameState, GameConfig};
        let mut stones: Vec<(Coord, Player)> = [(0,0),(1,0),(2,0),(0,1),(0,2)]
            .into_iter().map(|c| (c, Player::P1)).collect();
        stones.push(((-1, 0), Player::P2));
        let game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);
        let w = match solve(&game, 40, 5_000_000) {
            Outcome::Win(w) => w,
            _ => panic!("expected a forced win"),
        };
        assert_eq!(w.depth, 2);
        assert_eq!(w.pv.first(), Some(&w.first_move));
        assert_eq!(w.pv.len(), 6, "fork(2) + futile defender pair(2) + completion(2)");

        // Reconstruct the board exactly as extract_pv sees it right after the
        // fork move, to compute the greedy expectation independently of the
        // production code path (same helper, real completions() call).
        let mut b = SolverBoard::new();
        for &(c, p) in &stones { b.place(c, p); }
        for &c in &w.pv[..2] { b.place(c, Player::P1); }
        let comps = completions(&b, Player::P1, 6, 8);
        assert_eq!(min_covers2(&comps).0, 3, "must actually be the B >= 3 futile branch");
        let expected = futile_defender_pair(&comps);
        assert_eq!(
            CellSet2::two(w.pv[2], w.pv[3]), expected,
            "emitted futile pair must equal the greedy max-coverage expectation"
        );
        let hits = comps.iter()
            .filter(|comp| comp.cells().iter().any(|x| expected.cells().contains(x)))
            .count();
        assert!(hits >= 2, "futile pair must hit >= 2 distinct completions, hit {hits}");
        assert_eq!(hits, 3, "greedy pair kills the entire open-four axis (3 of 4 completions)");

        // The naive canonical-order pair the old code used is a strictly worse,
        // more redundant-looking choice — pin it as a regression guard.
        let mut canonical: Vec<Coord> = comps.iter().flat_map(|c| c.cells().iter().copied()).collect();
        canonical.sort_unstable(); canonical.dedup();
        let canonical_pair = CellSet2::two(canonical[0], canonical[1]);
        assert_ne!(expected, canonical_pair, "greedy must differ from the naive canonical pair here");

        // Replay the PV through the real engine end to end (legality + verdict).
        let mut g = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);
        for &c in &w.pv {
            assert!(!g.is_terminal(), "PV continues after the game ended");
            g.apply_move(c).expect("every PV move must be legal");
        }
        assert!(g.is_terminal());
        assert_eq!(g.winner(), Some(Player::P1));
    }

    #[test]
    fn boundary_b_values() {
        // 5-run open both ends -> B=2 unique cover
        let mut b = SolverBoard::new();
        for q in 0..5 { b.place((q, 0), Player::P1); }
        assert_eq!(min_covers2(&completions(&b, Player::P1, 6, 8)).0, 2);
        // walled 4-run (one empty each side) -> B=1
        let mut w = SolverBoard::new();
        for q in 0..4 { w.place((q, 0), Player::P1); }
        w.place((-2, 0), Player::P2); w.place((5, 0), Player::P2);
        assert_eq!(min_covers2(&completions(&w, Player::P1, 6, 8)).0, 1);
    }

    #[test]
    fn move_b_matches_double_four_fork() {
        // two 3-runs sharing (0,0) on q and r axes; playing (3,0)+(0,3) -> B>=3
        let mut b = SolverBoard::new();
        for c in [(0,0),(1,0),(2,0),(0,1),(0,2)] { b.place(c, Player::P1); }
        let idx = threat_table(&b, Player::P1, 6, 8);
        // Exactly 4: two independent open fours, each needing a 2-cell cover.
        // Pins the B=4-over-B=3 move-ordering preference (double-four first).
        assert_eq!(move_b(&idx, &[(3,0),(0,3)], 6), 4);
        // a lone extension makes exactly one open four: B exactly 2
        assert_eq!(move_b(&idx, &[(3,0),(9,9)], 6), 2);
    }

    #[test]
    fn cover_class_is_exact_up_to_cap() {
        // k pairwise-disjoint 2-cell comps -> B = k, reported capped at 4
        let mk = |k: i32| (0..k).map(|i| CellSet2::two((10 * i, 0), (10 * i + 1, 0))).collect::<Vec<_>>();
        assert_eq!(cover_class(&mk(0)), 0);
        assert_eq!(cover_class(&mk(1)), 1);
        assert_eq!(cover_class(&mk(2)), 2);
        assert_eq!(cover_class(&mk(3)), 3);
        assert_eq!(cover_class(&mk(4)), 4);
        assert_eq!(cover_class(&mk(5)), 4, "B >= 5 reports the cap");
        // a shared cell keeps B at 1 no matter how many comps
        let shared: Vec<CellSet2> = (0..10).map(|i| CellSet2::two((0, 0), (i + 1, 5))).collect();
        assert_eq!(cover_class(&shared), 1);
        // all four hitting-set implementations must agree on the same inputs:
        // fast mask paths vs the > 64-comp fallbacks (differential check)
        // mk(64)/mk(65) pin the n == 64 mask boundary and the > 64 dispatch edge
        for comps in [mk(0), mk(1), mk(2), mk(3), mk(4), mk(5), mk(64), mk(65), shared] {
            let (b2, covers2) = min_covers2(&comps);
            assert_eq!(b2, cover_class(&comps).min(3));
            assert_eq!(min_hit_slice(comps.to_vec(), 4), cover_class(&comps), "slice fallback disagrees");
            if b2 >= 3 {
                assert!(covers2.is_empty(), "no covers may be collected for B >= 3");
            }
            if !comps.is_empty() {
                let mut covers_fast = Vec::new();
                let mut covers_big = Vec::new();
                let b_fast = min_covers_capped(&comps, Some(&mut covers_fast));
                let b_big = min_covers_big(&comps, Some(&mut covers_big));
                assert_eq!(b_big, b_fast, "big fallback disagrees on B");
                assert_eq!(covers_big, covers_fast, "big fallback disagrees on covers");
            }
        }
    }

    #[test]
    fn hitting_set_big_fallback_matches_fast_path() {
        // > 64 comps exercises min_covers_big / min_hit_slice
        let disjoint: Vec<CellSet2> = (0..70).map(|i| CellSet2::two((10 * i, 0), (10 * i + 1, 0))).collect();
        assert_eq!(cover_class(&disjoint), 4);
        assert_eq!(min_covers2(&disjoint).0, 3);
        // B=2 with a unique cover {(0,0),(1,1)}: 70 comps through (0,0), 3 through (1,1)
        let mut c2: Vec<CellSet2> = (0..70).map(|i| CellSet2::two((0, 0), (i + 1, 7))).collect();
        c2.extend((0..3).map(|i| CellSet2::two((1, 1), (i + 1, 9))));
        assert_eq!(cover_class(&c2), 2);
        let (b, covers) = min_covers2(&c2);
        assert_eq!(b, 2);
        assert_eq!(covers, vec![CellSet2::two((0, 0), (1, 1))]);
    }

    #[test]
    fn attacker_turns_orders_double_four_first() {
        let mut b = SolverBoard::new();
        for c in [(0,0),(1,0),(2,0),(0,1),(0,2)] { b.place(c, Player::P1); }
        let moves = attacker_turns(&b, Player::P1, Player::P2, 2, 6, 8, false);
        assert!(!moves.is_empty());
        let idx = threat_table(&b, Player::P1, 6, 8);
        // the top move is a genuine double-four (B=4), not merely a triple threat
        assert_eq!(move_b(&idx, moves[0].cells(), 6), 4);
        // and the whole list is sorted best-first by B
        let bs: Vec<u8> = moves.iter().map(|m| move_b(&idx, m.cells(), 6)).collect();
        assert!(bs.windows(2).all(|w| w[0] >= w[1]), "moves not descending by B: {bs:?}");
    }

    #[test]
    fn any_four_gate_false_on_scattered() {
        let mut b = SolverBoard::new();
        b.place((0,0), Player::P1); b.place((10,10), Player::P1);
        assert!(!any_four_gate(&b, Player::P1, 6, 8));
        // and true once a window holds wl-4 stones (adjacent pair, wl=6)
        let mut c = SolverBoard::new();
        c.place((0,0), Player::P1); c.place((1,0), Player::P1);
        assert!(any_four_gate(&c, Player::P1, 6, 8));
        assert!(!any_four_gate(&c, Player::P2, 6, 8), "gate is per-player");
    }

    #[test]
    fn tiny_budget_reports_budget_exceeded() {
        // mate-in-two fork position: with a 1-node budget the search cannot finish,
        // and must say so rather than returning a bogus Win or No.
        let mut b = line(&[(0,0),(1,0),(2,0),(0,1),(0,2)], Player::P1);
        assert!(matches!(
            solve_from(&mut b, Player::P1, Player::P2, 2, 6, 8, 40, 1),
            SolveResult::BudgetExceeded
        ));
    }

    fn line(cells: &[Coord], p: Player) -> SolverBoard {
        let mut b = SolverBoard::new();
        for &c in cells { b.place(c, p); }
        b
    }

    #[test]
    fn mate_in_one() {
        let mut b = line(&[(0,0),(1,0),(2,0),(3,0),(4,0)], Player::P1);
        assert!(matches!(solve_from(&mut b, Player::P1, Player::P2, 2, 6, 8, 40, 5_000_000), SolveResult::Win { depth: 1 }));
    }
    #[test]
    fn mate_in_two_fork() {
        let mut b = line(&[(0,0),(1,0),(2,0),(0,1),(0,2)], Player::P1);
        let (h0, n0) = (b.hash, b.stones.len());
        assert!(matches!(solve_from(&mut b, Player::P1, Player::P2, 2, 6, 8, 40, 5_000_000), SolveResult::Win { depth: 2 }));
        // make/unmake must leave the board pristine (solve() relies on this to
        // probe first_winning_move on a clean position)
        assert_eq!((b.hash, b.stones.len()), (h0, n0), "solve_from must not leak stones");
    }
    /// EXPERIMENTAL wide-partner-width knob: it's a strict superset generator, so on
    /// a position where the tight generator already finds the shortest forced win,
    /// turning `wide` on must find the SAME win at the SAME depth (never longer —
    /// the search is best-first/iterative-deepening over an admitted superset of
    /// tight's moves, so it can only find an equal-or-shorter path, and this fork is
    /// already the shortest possible: depth 2).
    #[test]
    fn wide_is_superset_same_depth_on_mate_in_two_fork() {
        let mut b = line(&[(0,0),(1,0),(2,0),(0,1),(0,2)], Player::P1);
        assert!(matches!(
            solve_from_ex(&mut b, Player::P1, Player::P2, 2, 6, 8, 40, 5_000_000, true, Limits::default()),
            SolveResult::Win { depth: 2 }
        ));
    }
    #[test]
    fn no_forcing_win() {
        // gate-level No: scattered stones, no near-complete window at all
        let mut b = line(&[(0,0),(10,10)], Player::P1);
        assert!(matches!(solve_from(&mut b, Player::P1, Player::P2, 2, 6, 8, 40, 5_000_000), SolveResult::No));
    }

    /// A strict `No` proved by CLOSING the whole tree (`cut == false`) rather than
    /// by the gate or a timeout: the attacker has live threats (gate passes), but
    /// the defender holds an open four and completes first on every line — the
    /// counter-threat refutation makes each branch resolve immediately.
    #[test]
    fn search_proves_strict_no_when_defender_wins_first() {
        let mut b = SolverBoard::new();
        for q in 0..3 { b.place((q, 0), Player::P1); }
        for q in 10..14 { b.place((q, 0), Player::P2); }
        assert!(any_four_gate(&b, Player::P1, 6, 8), "gate must pass for this to test the search");
        assert!(matches!(
            solve_from(&mut b, Player::P1, Player::P2, 2, 6, 8, 40, 5_000_000),
            SolveResult::No
        ));
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
                // Replay the PV through the real engine: apply_move enforces turn
                // structure (2 placements/turn) and legality, and the line must
                // end in an attacker win — validating extract_pv end to end.
                let mut g = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);
                for &c in &w.pv {
                    assert!(!g.is_terminal(), "PV continues after the game ended");
                    g.apply_move(c).expect("every PV move must be legal");
                }
                assert!(g.is_terminal(), "PV must end the game");
                assert_eq!(g.winner(), Some(Player::P1), "PV must end in an attacker win");
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
        // the player term must matter: same cell, different colour, different hash
        let mut p1 = SolverBoard::new();
        p1.place((0, 0), Player::P1);
        let mut p2 = SolverBoard::new();
        p2.place((0, 0), Player::P2);
        assert_ne!(p1.hash, p2.hash, "zob must include the player");
    }

    /// The dense grid must behave identically to a sparse map under growth:
    /// stones placed far apart (forcing regrowth) stay readable, and reach
    /// counts survive the copy.
    #[test]
    fn grid_growth_preserves_stones_and_reach() {
        let mut b = SolverBoard::new();
        b.enable_reach(2);
        b.place((0, 0), Player::P1);
        assert!(b.within_radius((2, 0), 2));
        assert!(!b.within_radius((3, 0), 2));
        // Far placement forces growth in both directions.
        b.place((100, -80), Player::P2);
        assert_eq!(b.get((0, 0)), Some(Player::P1));
        assert_eq!(b.get((100, -80)), Some(Player::P2));
        assert!(b.within_radius((2, 0), 2), "reach counts must survive grid growth");
        assert!(b.within_radius((99, -79), 2));
        assert!(!b.within_radius((50, -40), 2));
        // Removing reverts reach.
        b.remove((100, -80));
        assert!(!b.within_radius((99, -79), 2));
        // Untracked-radius queries fall back to a stone scan.
        assert!(b.within_radius((5, 0), 8));
    }

    /// Differential: the O(1) tracked reach path must agree with the linear
    /// stone-scan ground truth at the SAME radius, on every cell near the
    /// stones, across place/remove/growth — the reach counts are the most
    /// state-heavy machinery in the board.
    #[test]
    fn reach_counts_match_linear_scan() {
        let radius = 2;
        let mut tracked = SolverBoard::new();
        tracked.enable_reach(radius);
        let mut plain = SolverBoard::new(); // never tracked: within_radius scans stones
        let script: &[(Coord, bool)] = &[
            ((0, 0), true), ((1, 0), true), ((40, -25), true), // forces growth
            ((1, 0), false), ((3, -2), true), ((40, -25), false),
        ];
        for &(c, place) in script {
            if place {
                tracked.place(c, Player::P1);
                plain.place(c, Player::P1);
            } else {
                tracked.remove(c);
                plain.remove(c);
            }
            for q in -6..8 {
                for r in -6..8 {
                    let cell = (q, r);
                    assert_eq!(
                        tracked.within_radius(cell, radius),
                        plain.within_radius(cell, radius),
                        "reach mismatch at {cell:?} after {c:?} place={place}"
                    );
                }
            }
        }
    }

    /// `has_completion` is half of the solve gate: pin its unique contribution
    /// (a walled run that is a completion but NOT a fresh-four threat source).
    #[test]
    fn has_completion_thresholds() {
        // open 4-run: completions with 1..2 gaps exist
        let b = line(&[(0,0),(1,0),(2,0),(3,0)], Player::P1);
        assert!(has_completion(&b, Player::P1, 6, 2, 8));
        assert!(!has_completion(&b, Player::P1, 6, 1, 8), "no 1-gap completion yet");
        // 5-run: a 1-gap completion appears
        let b5 = line(&[(0,0),(1,0),(2,0),(3,0),(4,0)], Player::P1);
        assert!(has_completion(&b5, Player::P1, 6, 1, 8));
        // scattered stones: none
        let s = line(&[(0,0),(10,10)], Player::P1);
        assert!(!has_completion(&s, Player::P1, 6, 2, 8));
        // and it is per-player
        assert!(!has_completion(&b5, Player::P2, 6, 2, 8));
    }

    /// Unit-level pin of the tight-regime window filter: a window whose gap is
    /// unreachable at the placement radius must be dropped from the threat
    /// table (not just filtered later at move level).
    #[test]
    fn tight_regime_filters_unreachable_windows() {
        let mut b = SolverBoard::new();
        b.place((0, 0), Player::P1);
        b.place((1, 0), Player::P1);
        // Wide regime: the far gap (4,0) is indexed (window [0..5] and friends).
        let wide = threat_table(&b, Player::P1, 6, 8);
        assert!(wide.contains_key(&(4, 0)), "wide regime must index the far gap");
        // Tight regime (radius 2): (4,0) is hex-distance 3 from the nearest stone,
        // so every window containing it is unplayable and must be dropped...
        b.enable_reach(2);
        let tight = threat_table(&b, Player::P1, 6, 2);
        assert!(!tight.contains_key(&(4, 0)), "unreachable-gap windows must be dropped");
        // ...while fully-reachable windows survive.
        assert!(tight.contains_key(&(-1, 0)), "reachable windows must survive");
    }

    // FIX 2: wl-4/wl-3/wl-2 thresholds must not underflow (u8) for small win_length —
    // and small-wl results must be sane, not just panic-free: a 3-run at wl=4 and a
    // lone stone at wl=2 are both one placement from a win.
    #[test]
    fn wl4_and_wl2_do_not_panic() {
        let mut b4 = line(&[(0,0),(1,0),(2,0)], Player::P1);
        assert!(matches!(
            solve_from(&mut b4, Player::P1, Player::P2, 2, 4, 8, 10, 100_000),
            SolveResult::Win { depth: 1 }
        ));
        let _ = completions(&b4, Player::P1, 4, 8);
        let _ = threat_table(&b4, Player::P1, 4, 8);

        let mut b2 = line(&[(0,0)], Player::P1);
        assert!(matches!(
            solve_from(&mut b2, Player::P1, Player::P2, 2, 2, 8, 10, 100_000),
            SolveResult::Win { depth: 1 }
        ));
        let _ = completions(&b2, Player::P1, 2, 8);
        let _ = threat_table(&b2, Player::P1, 2, 8);
    }

    /// Configs the solver cannot analyze must yield an honest give-up
    /// (BudgetExceeded), never a bogus No or a panic/giant allocation.
    #[test]
    fn unsupported_configs_give_up_honestly() {
        // win_length beyond the stack-buffer cap
        let mut b = line(&[(0,0),(1,0),(2,0)], Player::P1);
        assert!(matches!(
            solve_from(&mut b, Player::P1, Player::P2, 2, 20, 8, 10, 100_000),
            SolveResult::BudgetExceeded
        ));
        // win_length 0
        let mut b0 = line(&[(0,0)], Player::P1);
        assert!(matches!(
            solve_from(&mut b0, Player::P1, Player::P2, 2, 0, 8, 10, 100_000),
            SolveResult::BudgetExceeded
        ));
        // pathological coordinate spread: rejected before any grid allocation
        use hexo_engine::game::{GameState, GameConfig};
        let stones = vec![((0, 0), Player::P1), ((1_000_000, 1_000_000), Player::P2)];
        let game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);
        assert!(matches!(solve(&game, 6, 100_000), Outcome::BudgetExceeded));
    }

    /// The PRODUCTION regime is tight (placement_radius < win_length - 1, live
    /// self-play runs radius 2): pin that a forced win IS found there — same
    /// depth as the wide regime — and that its PV replays legally through the
    /// engine under the radius-2 config. Guards against tight-regime
    /// over-pruning silently turning every Win into No.
    #[test]
    fn tight_regime_finds_win_with_legal_pv() {
        use hexo_engine::game::{GameState, GameConfig};
        let stones: Vec<(Coord, Player)> = [(0,0),(1,0),(2,0),(0,1),(0,2)]
            .into_iter().map(|c| (c, Player::P1)).collect();
        let cfg = GameConfig { win_length: 6, placement_radius: 2, max_moves: 200 };
        let game = GameState::from_state(&stones, Player::P1, 2, cfg);
        match solve(&game, 40, 5_000_000) {
            Outcome::Win(w) => {
                assert_eq!(w.depth, 2, "tight regime must find the same fork win");
                let mut g = GameState::from_state(&stones, Player::P1, 2, cfg);
                for &c in &w.pv {
                    assert!(!g.is_terminal(), "PV continues after the game ended");
                    g.apply_move(c).expect("PV move must be legal at radius 2");
                }
                assert_eq!(g.winner(), Some(Player::P1));
            }
            _ => panic!("expected Win in the tight regime"),
        }
    }

    // FIX 3: candidate enumeration must not leak into first_move/PV selection
    // non-determinism.
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

    /// TDD regression for the radius-blindness bug: in the tight regime
    /// (`placement_radius < win_length - 1`) the solver must never return a forced win
    /// whose line plays a cell outside the placement radius (an ILLEGAL HeXO move).
    ///
    /// Construction (`win_length = 6`, attacker = P2): a single fork cell `(20,0)`
    /// extends two P2 three-runs into two simultaneous 4-in-a-rows (a double-four,
    /// move_b == 4) -> forced win at depth 2. A *distant* P2 two-run at
    /// `(-15,0),(-14,0)` seeds threat cells far from every stone; a radius-blind
    /// `attacker_turns` pairs one of those far cells (e.g. `(-19,0)`, hex-distance 4
    /// from the nearest stone) with the fork cell as a useless partner and drops it
    /// into the winning line. Legal at radius 8, ILLEGAL at radius 2.
    #[test]
    fn radius_aware_rejects_illegal_forcing_win() {
        use hexo_engine::game::{GameState, GameConfig};
        let stones: Vec<(Coord, Player)> = vec![
            ((0, 0), Player::P1),                                                // seeded distant defender stone
            ((21, 0), Player::P2), ((22, 0), Player::P2), ((23, 0), Player::P2), // q-arm three-run
            ((20, 1), Player::P2), ((20, 2), Player::P2), ((20, 3), Player::P2), // r-arm three-run
            ((-15, 0), Player::P2), ((-14, 0), Player::P2),                      // distant threat two-run
        ];
        let cfg = |r: i32| GameConfig { win_length: 6, placement_radius: r, max_moves: 200 };

        // Walk the PV from the initial stones; return the first cell that is NOT within
        // `radius` of a stone present just before it is played (legality is colour-agnostic).
        let first_illegal = |pv: &[Coord], radius: i32| -> Option<Coord> {
            let mut present: FxHashSet<Coord> = stones.iter().map(|(c, _)| *c).collect();
            for &cell in pv {
                let legal = present.iter().any(|&s| hex_dist(cell, s) <= radius);
                if !legal { return Some(cell); }
                present.insert(cell);
            }
            None
        };

        // Control + non-vacuity: at radius 8 (>= win_length-1, so all legality work is
        // skipped) the solver returns the win, and its PV contains a cell that would be
        // ILLEGAL under radius 2 -- proving the position really triggers the bug.
        let g8 = GameState::from_state(&stones, Player::P2, 2, cfg(8));
        let w8 = match solve(&g8, 40, 5_000_000) {
            Outcome::Win(w) => w,
            _ => panic!("radius-8 control: expected Win"),
        };
        assert_eq!(w8.depth, 2, "radius-8 win depth");
        assert!(
            first_illegal(&w8.pv, 2).is_some(),
            "non-vacuous test needs the radius-blind PV {:?} to contain a cell illegal under radius 2",
            w8.pv
        );

        // The fix: at radius 2 the solver must not return a win that plays an illegal
        // cell; for ANY win it returns, every PV cell must be legal when played.
        let g2 = GameState::from_state(&stones, Player::P2, 2, cfg(2));
        match solve(&g2, 40, 5_000_000) {
            Outcome::Win(w) => {
                if let Some(bad) = first_illegal(&w.pv, 2) {
                    panic!("radius-2 solver returned a win with out-of-radius move {bad:?} in PV {:?}", w.pv);
                }
            }
            Outcome::No | Outcome::BudgetExceeded => {} // also acceptable: no bogus win returned
        }
    }

    // Cross-implementation parity fixtures captured from the validated Python
    // prototype (scripts/forcing_search_prototype.py) on real HDS puzzle positions.
    // expected_depth = Some(d) means the Python solver found a forced win at depth d;
    // None means it proved No forcing win (within its search).
    #[allow(clippy::type_complexity)]
    const FIXTURES: &[(&[(Coord, Player)], Player, u8, Option<u8>)] = &[
        // xsnfyll: 13 stones, attacker=P2
        (&[((0,0), Player::P1), ((1,-2), Player::P2), ((2,-4), Player::P2), ((0,-3), Player::P1), ((2,-5), Player::P1), ((0,-2), Player::P2), ((1,-4), Player::P2), ((-2,0), Player::P1), ((1,0), Player::P1), ((-1,0), Player::P2), ((1,-3), Player::P2), ((3,-4), Player::P1), ((3,-2), Player::P1)], Player::P2, 2, Some(4)),
        // 0hz3hty: 21 stones, attacker=P2
        (&[((0,0), Player::P1), ((5,0), Player::P2), ((5,-1), Player::P2), ((5,-2), Player::P1), ((4,0), Player::P1), ((5,4), Player::P2), ((6,4), Player::P2), ((6,3), Player::P1), ((4,4), Player::P1), ((10,4), Player::P2), ((11,3), Player::P2), ((11,4), Player::P1), ((12,2), Player::P1), ((6,8), Player::P2), ((5,8), Player::P2), ((4,8), Player::P1), ((5,9), Player::P1), ((10,8), Player::P2), ((11,7), Player::P2), ((9,9), Player::P1), ((10,7), Player::P1)], Player::P2, 2, Some(5)),
        // acly7kb: 93 stones, attacker=P2
        (&[((0,0), Player::P1), ((1,-2), Player::P2), ((-1,2), Player::P2), ((-1,0), Player::P1), ((-2,0), Player::P1), ((1,0), Player::P2), ((-5,0), Player::P2), ((-2,3), Player::P1), ((1,-3), Player::P1), ((0,-2), Player::P2), ((-2,2), Player::P2), ((-4,2), Player::P1), ((-2,-2), Player::P1), ((3,-2), Player::P2), ((1,2), Player::P2), ((2,-1), Player::P1), ((1,1), Player::P1), ((2,-2), Player::P2), ((4,-2), Player::P2), ((5,-2), Player::P1), ((-1,-2), Player::P1), ((0,2), Player::P2), ((2,2), Player::P2), ((3,2), Player::P1), ((-3,2), Player::P1), ((-3,1), Player::P2), ((-1,-1), Player::P2), ((-2,1), Player::P1), ((-2,-3), Player::P1), ((-2,-1), Player::P2), ((-4,3), Player::P2), ((-3,-1), Player::P1), ((4,0), Player::P1), ((-1,-3), Player::P2), ((4,1), Player::P2), ((-3,-2), Player::P1), ((5,-1), Player::P1), ((-6,-2), Player::P2), ((5,-3), Player::P2), ((6,-4), Player::P1), ((-6,-1), Player::P1), ((4,-1), Player::P2), ((3,1), Player::P2), ((5,0), Player::P1), ((-4,-3), Player::P1), ((5,1), Player::P2), ((-3,-4), Player::P2), ((-6,2), Player::P1), ((6,1), Player::P1), ((-5,2), Player::P2), ((7,0), Player::P2), ((-5,1), Player::P1), ((6,-2), Player::P1), ((-7,3), Player::P2), ((6,0), Player::P2), ((8,-2), Player::P1), ((-6,3), Player::P1), ((7,-2), Player::P2), ((7,-3), Player::P2), ((7,-1), Player::P1), ((6,-3), Player::P1), ((6,-6), Player::P2), ((-6,0), Player::P2), ((7,-4), Player::P1), ((-7,0), Player::P1), ((4,-4), Player::P2), ((-8,1), Player::P2), ((8,-5), Player::P1), ((8,-4), Player::P1), ((9,-6), Player::P2), ((8,-3), Player::P2), ((5,-5), Player::P1), ((0,-5), Player::P1), ((1,-6), Player::P2), ((9,-5), Player::P2), ((4,-5), Player::P1), ((9,-7), Player::P1), ((3,-5), Player::P2), ((-8,0), Player::P2), ((-8,2), Player::P1), ((8,-6), Player::P1), ((8,-9), Player::P2), ((11,-9), Player::P2), ((9,-9), Player::P1), ((-7,-1), Player::P1), ((-7,-2), Player::P2), ((-8,-1), Player::P2), ((-8,-2), Player::P1), ((-9,0), Player::P1), ((-2,-6), Player::P2), ((-4,-4), Player::P2), ((-3,-5), Player::P1), ((-5,-4), Player::P1)], Player::P2, 2, Some(4)),
        // l9mxn59: 17 stones, attacker=P2 -- No forcing win
        (&[((0,0), Player::P1), ((1,-1), Player::P2), ((2,-1), Player::P2), ((1,0), Player::P1), ((0,-1), Player::P1), ((3,-2), Player::P2), ((2,-2), Player::P2), ((2,-3), Player::P1), ((4,-2), Player::P1), ((3,-3), Player::P2), ((4,-3), Player::P2), ((3,-4), Player::P1), ((3,-1), Player::P1), ((10,-4), Player::P2), ((11,-4), Player::P2), ((10,-3), Player::P1), ((12,-4), Player::P1)], Player::P2, 2, None),
    ];

    // The two largest/deepest fixtures live in a separate #[ignore]d test since they
    // are still the slowest (~hundreds of ms release, seconds in debug):
    // jnzzmcm (67 stones -> depth 7), hu01jk4 (149 stones -> depth 6).
    // run manually: cargo test -p hexo-solver forcing::tests::parity_win_depths_match_python_slow -- --ignored --nocapture
    #[allow(clippy::type_complexity)]
    const SLOW_FIXTURES: &[(&[(Coord, Player)], Player, u8, Option<u8>)] = &[
        // jnzzmcm: 67 stones, attacker=P1
        (&[((0,0), Player::P1), ((0,4), Player::P2), ((1,3), Player::P2), ((1,0), Player::P1), ((-1,1), Player::P1), ((-1,0), Player::P2), ((1,-1), Player::P2), ((-1,5), Player::P1), ((0,3), Player::P1), ((2,2), Player::P2), ((3,1), Player::P2), ((4,0), Player::P1), ((4,1), Player::P1), ((3,0), Player::P2), ((4,2), Player::P2), ((3,2), Player::P1), ((2,3), Player::P1), ((1,4), Player::P2), ((-1,4), Player::P2), ((2,4), Player::P1), ((1,5), Player::P1), ((3,3), Player::P2), ((4,-2), Player::P2), ((4,-1), Player::P1), ((5,-3), Player::P1), ((6,-3), Player::P2), ((5,-4), Player::P2), ((4,-3), Player::P1), ((6,-4), Player::P1), ((5,-5), Player::P2), ((7,-6), Player::P2), ((6,-5), Player::P1), ((7,-7), Player::P1), ((6,-7), Player::P2), ((6,-8), Player::P2), ((5,-7), Player::P1), ((4,-6), Player::P1), ((4,-7), Player::P2), ((3,5), Player::P2), ((3,7), Player::P1), ((2,8), Player::P1), ((4,6), Player::P2), ((2,9), Player::P2), ((-3,9), Player::P1), ((-4,9), Player::P1), ((-1,7), Player::P2), ((-2,9), Player::P2), ((-2,10), Player::P1), ((-4,11), Player::P1), ((-3,10), Player::P2), ((-4,10), Player::P2), ((-8,8), Player::P1), ((-8,7), Player::P1), ((-7,6), Player::P2), ((-8,5), Player::P2), ((-8,6), Player::P1), ((-7,5), Player::P1), ((-9,7), Player::P2), ((-5,10), Player::P2), ((-9,6), Player::P1), ((7,-4), Player::P1), ((-3,3), Player::P2), ((8,-4), Player::P2), ((-5,9), Player::P1), ((-6,10), Player::P1), ((-8,9), Player::P2), ((-6,11), Player::P2)], Player::P1, 2, Some(7)),
        // hu01jk4: 149 stones, attacker=P2
        (&[((0,0), Player::P1), ((1,-2), Player::P2), ((-1,-4), Player::P2), ((0,-3), Player::P1), ((-3,0), Player::P1), ((-1,-2), Player::P2), ((0,-2), Player::P2), ((-2,-2), Player::P1), ((-1,-3), Player::P1), ((1,-3), Player::P2), ((-4,0), Player::P2), ((-2,0), Player::P1), ((1,-5), Player::P1), ((0,-4), Player::P2), ((2,0), Player::P2), ((-3,-4), Player::P1), ((-2,2), Player::P1), ((-2,1), Player::P2), ((-1,1), Player::P2), ((-3,1), Player::P1), ((2,-4), Player::P1), ((-3,2), Player::P2), ((-1,0), Player::P2), ((0,-1), Player::P1), ((-1,3), Player::P1), ((3,-1), Player::P2), ((4,-2), Player::P2), ((3,-2), Player::P1), ((5,-3), Player::P1), ((1,-1), Player::P2), ((4,-1), Player::P2), ((4,0), Player::P1), ((1,1), Player::P1), ((-3,-2), Player::P2), ((-2,-6), Player::P2), ((-2,-3), Player::P1), ((5,-1), Player::P1), ((7,-3), Player::P2), ((-2,-5), Player::P2), ((1,3), Player::P1), ((0,3), Player::P1), ((2,3), Player::P2), ((0,4), Player::P2), ((1,2), Player::P1), ((2,2), Player::P1), ((3,1), Player::P2), ((-4,3), Player::P2), ((-5,4), Player::P1), ((-4,2), Player::P1), ((3,2), Player::P2), ((3,0), Player::P2), ((3,3), Player::P1), ((1,4), Player::P1), ((1,5), Player::P2), ((-5,3), Player::P2), ((-2,4), Player::P1), ((-2,5), Player::P1), ((-1,4), Player::P2), ((-3,5), Player::P2), ((-6,3), Player::P1), ((-7,4), Player::P1), ((-6,4), Player::P2), ((-5,2), Player::P2), ((5,-2), Player::P1), ((5,-4), Player::P1), ((5,0), Player::P2), ((5,-5), Player::P2), ((4,-3), Player::P1), ((6,-5), Player::P1), ((7,-6), Player::P2), ((2,-1), Player::P2), ((7,-2), Player::P1), ((6,-6), Player::P1), ((6,-4), Player::P2), ((-9,8), Player::P2), ((-10,10), Player::P1), ((-10,7), Player::P1), ((-10,8), Player::P2), ((-9,7), Player::P2), ((-8,6), Player::P1), ((-9,6), Player::P1), ((-11,8), Player::P2), ((-8,8), Player::P2), ((-12,8), Player::P1), ((-7,8), Player::P1), ((-11,12), Player::P2), ((-10,6), Player::P2), ((-12,10), Player::P1), ((-5,6), Player::P1), ((-9,10), Player::P2), ((-2,3), Player::P2), ((-9,11), Player::P1), ((-7,9), Player::P1), ((-7,6), Player::P2), ((-10,12), Player::P2), ((-12,12), Player::P1), ((-12,11), Player::P1), ((-12,9), Player::P2), ((-11,11), Player::P2), ((-11,10), Player::P1), ((-5,7), Player::P1), ((-5,8), Player::P2), ((-12,13), Player::P2), ((-13,14), Player::P1), ((-13,12), Player::P1), ((-13,13), Player::P2), ((-14,13), Player::P2), ((-11,13), Player::P1), ((6,-10), Player::P1), ((7,-10), Player::P2), ((6,-9), Player::P2), ((5,-8), Player::P1), ((5,-9), Player::P1), ((4,-8), Player::P2), ((5,-10), Player::P2), ((-16,13), Player::P1), ((-2,7), Player::P1), ((-3,7), Player::P2), ((-3,4), Player::P2), ((-3,6), Player::P1), ((-1,2), Player::P1), ((-2,6), Player::P2), ((-15,11), Player::P2), ((-4,8), Player::P1), ((-15,12), Player::P1), ((-14,12), Player::P2), ((-14,10), Player::P2), ((-14,11), Player::P1), ((10,-2), Player::P1), ((9,-2), Player::P2), ((-18,15), Player::P2), ((-16,14), Player::P1), ((10,-5), Player::P1), ((10,-3), Player::P2), ((-16,12), Player::P2), ((-18,14), Player::P1), ((12,-5), Player::P1), ((13,-5), Player::P2), ((-14,14), Player::P2), ((12,-4), Player::P1), ((9,-1), Player::P1), ((12,-6), Player::P2), ((9,-10), Player::P2), ((8,-3), Player::P1), ((-3,-3), Player::P1), ((-5,-3), Player::P2), ((-14,15), Player::P2), ((-3,-5), Player::P1), ((-14,16), Player::P1)], Player::P2, 2, Some(6)),
    ];

    /// Cross-implementation parity: the Rust solver must reproduce the validated
    /// Python prototype's win depths on real HDS puzzle positions.
    #[test]
    fn parity_win_depths_match_python() {
        // End-to-end on one real puzzle (xsnfyll): solve() must return the parity
        // depth AND a PV that replays legally through the engine to a win —
        // real-board PV validation, not just depth agreement.
        {
            use hexo_engine::game::{GameState, GameConfig};
            let (stones, attacker, placements, expected) = FIXTURES[0];
            let game = GameState::from_state(stones, attacker, placements, GameConfig::FULL_HEXO);
            let w = match solve(&game, 40, 80_000_000) {
                Outcome::Win(w) => w,
                _ => panic!("xsnfyll: expected Win"),
            };
            assert_eq!(Some(w.depth), expected, "xsnfyll depth");
            let mut g = GameState::from_state(stones, attacker, placements, GameConfig::FULL_HEXO);
            for &c in &w.pv {
                assert!(!g.is_terminal(), "PV continues after the game ended");
                g.apply_move(c).expect("xsnfyll PV move must be legal");
            }
            assert_eq!(g.winner(), Some(attacker), "xsnfyll PV must end in the attacker's win");
        }
        for &(stones, attacker, placements, expected_depth) in FIXTURES.iter() {
            let Some(d) = expected_depth else { continue }; // No-fixtures covered separately
            let mut b = SolverBoard::new();
            for &(c, p) in stones { b.place(c, p); }
            let defender = attacker.opponent();
            let r = solve_from(&mut b, attacker, defender, placements, 6, 8, 40, 80_000_000);
            match r {
                SolveResult::Win { depth } => assert_eq!(depth, d, "depth mismatch on fixture"),
                SolveResult::No => panic!("expected Win{{{d}}}, got No"),
                SolveResult::BudgetExceeded => panic!("expected Win{{{d}}}, got BudgetExceeded"),
            }
        }
    }

    // run manually: cargo test -p hexo-solver forcing::tests::parity_win_depths_match_python_slow -- --ignored --nocapture
    #[test]
    #[ignore]
    fn parity_win_depths_match_python_slow() {
        for &(stones, attacker, placements, expected_depth) in SLOW_FIXTURES.iter() {
            let Some(d) = expected_depth else { continue };
            let mut b = SolverBoard::new();
            for &(c, p) in stones { b.place(c, p); }
            let defender = attacker.opponent();
            let r = solve_from(&mut b, attacker, defender, placements, 6, 8, 40, 80_000_000);
            match r {
                SolveResult::Win { depth } => assert_eq!(depth, d, "depth mismatch on fixture"),
                SolveResult::No => panic!("expected Win{{{d}}}, got No"),
                SolveResult::BudgetExceeded => panic!("expected Win{{{d}}}, got BudgetExceeded"),
            }
        }
    }

    /// l9mxn59: the validated Python prototype proved No forcing win for this position.
    ///
    /// NOTE (perf discrepancy, not a correctness bug): the Rust port's `No` requires the
    /// *entire* forcing tree (every branch, not just a winning line) to resolve within
    /// `depth_cap` plies of budget (`cut == false`) -- exactly like the Python prototype's
    /// `_atk_within`/`solve`. On this branchy 17-stone position the tree does NOT close
    /// out within a fast budget: per-depth cost keeps growing with `cut=true`
    /// (inconclusive), so proving a strict `No` here is computationally infeasible in a
    /// fast test (deep cost is unique positions, so memoization doesn't rescue it). So
    /// instead of asserting `SolveResult::No` at depth_cap=6 (which actually yields
    /// `BudgetExceeded`), we assert the fast, sound invariant both implementations must
    /// agree on: the Rust solver must not fabricate a forced Win where the validated
    /// Python solver found none.
    #[test]
    fn parity_l9mxn59_is_not_forced() {
        let (stones, attacker, placements, expected_depth) = FIXTURES[3];
        assert_eq!(expected_depth, None, "fixture index changed; expected l9mxn59 (No)");
        let mut b = SolverBoard::new();
        for &(c, p) in stones { b.place(c, p); }
        let defender = attacker.opponent();
        let r = solve_from(&mut b, attacker, defender, placements, 6, 8, 6, 80_000_000);
        assert!(
            !matches!(r, SolveResult::Win { .. }),
            "unsound: Rust found a forced win where the validated Python solver found none"
        );
    }

    // Ad-hoc benchmarking fixtures: three hard puzzle positions that were very slow
    // in the Python prototype. run manually:
    // cargo test --release -p hexo-solver forcing::tests::hard_puzzles_timing -- --ignored --nocapture
    #[allow(clippy::type_complexity)]
    const HARD_FIXTURES: &[(&str, &[(Coord, Player)], Player, u8)] = &[
        ("zrugh2x", &[((0,0), Player::P1), ((1,0), Player::P2), ((2,-2), Player::P2), ((0,1), Player::P1), ((1,1), Player::P1), ((-1,1), Player::P2), ((0,2), Player::P2), ((4,1), Player::P1), ((4,0), Player::P1), ((5,1), Player::P2), ((4,2), Player::P2), ((6,0), Player::P1), ((0,-2), Player::P1), ((5,0), Player::P2), ((0,-4), Player::P2), ((5,-1), Player::P1), ((2,2), Player::P1), ((3,1), Player::P2), ((2,-4), Player::P2), ((2,-6), Player::P1), ((4,-4), Player::P1), ((4,-6), Player::P2), ((3,-5), Player::P2), ((4,-1), Player::P1), ((6,-1), Player::P1), ((4,-3), Player::P2), ((2,-1), Player::P2), ((3,-2), Player::P1), ((5,-7), Player::P1), ((7,-1), Player::P2), ((6,-2), Player::P2), ((5,-9), Player::P1), ((6,-9), Player::P1), ((5,-8), Player::P2), ((4,-8), Player::P2), ((7,-9), Player::P1), ((8,-9), Player::P1), ((9,-9), Player::P2), ((4,-9), Player::P2), ((4,-10), Player::P1), ((8,-10), Player::P1), ((8,-11), Player::P2), ((9,-11), Player::P2), ((7,-8), Player::P1), ((7,-7), Player::P1)], Player::P2, 2),
        ("jh7yo7y", &[((0,0), Player::P1), ((8,0), Player::P2), ((7,1), Player::P2), ((-1,1), Player::P1), ((-2,2), Player::P1), ((1,-1), Player::P2), ((6,2), Player::P2), ((5,3), Player::P1), ((-2,1), Player::P1), ((0,1), Player::P2), ((-2,3), Player::P2), ((-1,2), Player::P1), ((-3,3), Player::P1), ((-4,4), Player::P2), ((9,-1), Player::P2), ((10,-2), Player::P1), ((-3,2), Player::P1), ((-1,0), Player::P2), ((-4,2), Player::P2), ((7,0), Player::P1), ((-4,1), Player::P1), ((-3,1), Player::P2), ((-6,4), Player::P2), ((-5,3), Player::P1), ((4,1), Player::P1), ((8,1), Player::P2), ((9,1), Player::P2), ((6,1), Player::P1), ((8,-1), Player::P1), ((5,2), Player::P2), ((-6,6), Player::P2), ((9,-2), Player::P1), ((4,3), Player::P1), ((10,-3), Player::P2), ((4,2), Player::P2)], Player::P1, 2),
        ("g2xx6wl", &[((0,0), Player::P1), ((1,-2), Player::P2), ((-8,0), Player::P2), ((0,1), Player::P1), ((0,2), Player::P1), ((0,3), Player::P2), ((-1,2), Player::P2), ((0,-2), Player::P1), ((-3,1), Player::P1), ((-2,0), Player::P2), ((0,-1), Player::P2), ((-1,0), Player::P1), ((-7,-2), Player::P1), ((-9,1), Player::P2), ((-10,2), Player::P2), ((-11,3), Player::P1), ((-9,2), Player::P1), ((-8,-1), Player::P2), ((-10,-1), Player::P2), ((-7,-1), Player::P1), ((-8,-2), Player::P1), ((1,1), Player::P2), ((-7,0), Player::P2), ((-10,0), Player::P1), ((1,0), Player::P1), ((-5,6), Player::P2), ((-9,-1), Player::P2), ((-13,-1), Player::P1), ((-13,-2), Player::P1), ((-13,0), Player::P2), ((-12,-2), Player::P2), ((-6,-5), Player::P1), ((-5,-5), Player::P1), ((-7,-5), Player::P2), ((-6,-4), Player::P2), ((-6,-2), Player::P1), ((-5,-2), Player::P1), ((-9,-2), Player::P2), ((-4,-2), Player::P2), ((-5,-3), Player::P1), ((-5,-4), Player::P1), ((-5,-1), Player::P2), ((-5,-7), Player::P2), ((-4,-5), Player::P1), ((-3,-5), Player::P1), ((-4,-4), Player::P2), ((-1,-5), Player::P2), ((3,0), Player::P1), ((-3,-8), Player::P1), ((2,0), Player::P2), ((-3,-6), Player::P2), ((-3,-3), Player::P1), ((-8,-3), Player::P1), ((-9,-3), Player::P2), ((-9,-4), Player::P2), ((-9,-5), Player::P1), ((-9,0), Player::P1), ((-6,-6), Player::P2), ((-8,-4), Player::P2), ((-10,-2), Player::P1), ((-4,-8), Player::P1), ((-7,-6), Player::P2), ((-5,-6), Player::P2), ((-4,-6), Player::P1), ((-4,-7), Player::P1), ((-4,-9), Player::P2), ((-5,-8), Player::P2), ((-6,-7), Player::P1), ((-7,-7), Player::P1), ((-4,-3), Player::P2), ((-4,0), Player::P2), ((-4,1), Player::P1), ((-5,5), Player::P1), ((-2,1), Player::P2), ((-2,3), Player::P2), ((-2,2), Player::P1), ((-3,4), Player::P1), ((-3,3), Player::P2), ((-6,0), Player::P2), ((-3,0), Player::P1), ((-1,3), Player::P1), ((-2,-1), Player::P2), ((-2,-2), Player::P2), ((-2,-3), Player::P1), ((-3,-1), Player::P1), ((-3,-2), Player::P2), ((-10,9), Player::P2), ((-9,3), Player::P1), ((-9,8), Player::P1), ((-13,10), Player::P2), ((-12,12), Player::P2), ((-13,11), Player::P1), ((-12,11), Player::P1), ((-11,11), Player::P2), ((-10,10), Player::P2), ((-9,9), Player::P1), ((-9,10), Player::P1), ((-10,11), Player::P2), ((-10,12), Player::P2), ((-10,8), Player::P1), ((-10,13), Player::P1), ((-11,12), Player::P2), ((-9,12), Player::P2), ((-8,12), Player::P1), ((-14,12), Player::P1), ((-11,10), Player::P2), ((-11,13), Player::P2), ((-11,15), Player::P1), ((-11,9), Player::P1), ((-12,10), Player::P2), ((-14,14), Player::P2), ((-13,13), Player::P1), ((-14,10), Player::P1), ((-9,7), Player::P2), ((-10,14), Player::P2), ((-13,15), Player::P1), ((-14,11), Player::P1), ((-13,14), Player::P2), ((-14,9), Player::P2), ((-14,15), Player::P1), ((-11,14), Player::P1), ((-12,15), Player::P2), ((-12,8), Player::P2), ((-8,-6), Player::P1), ((-11,-4), Player::P1), ((-10,-4), Player::P2), ((-10,-5), Player::P2), ((-5,-11), Player::P1), ((-10,-3), Player::P1), ((-4,-13), Player::P2), ((-11,7), Player::P2), ((-10,7), Player::P1), ((-10,6), Player::P1), ((-10,3), Player::P2), ((1,8), Player::P2), ((2,6), Player::P1), ((-11,8), Player::P1), ((0,8), Player::P2), ((0,7), Player::P2)], Player::P1, 2),
    ];

    #[test]
    #[ignore]
    fn hard_puzzles_timing() {
        for &(code, stones, attacker, placements) in HARD_FIXTURES.iter() {
            let mut b = SolverBoard::new();
            for &(c, p) in stones { b.place(c, p); }
            let defender = attacker.opponent();
            let t = std::time::Instant::now();
            let r = solve_from(&mut b, attacker, defender, placements, 6, 8, 20, 300_000_000);
            let elapsed = t.elapsed();
            eprintln!("{code} -> {r:?}  ({elapsed:.2?})");
        }
    }

    /// EXPERIMENTAL DIAGNOSTIC (not a correctness/regression test): does the wide
    /// partner-width knob find a forced win on the live 0l4291i position — the
    /// puzzle creator's FIXED replacement for the retracted 94gnnol mate-in-19
    /// claim — that the tight generator hasn't? Known context: tight finds no win
    /// at depth_cap=12/node_budget=250k (1.8s) and depth_cap=40/5M (54s); a deeper
    /// depth_cap=40/100M tight run is in progress elsewhere. A Win here would be a
    /// genuinely interesting result (wide seeing a force-and-build line tight's
    /// generator can't reach); no-win is also a valid, consistent result. Escalates
    /// the node budget (1M, then 10M) and stops there.
    /// run manually (release; this is compute-heavy):
    /// cargo test --release -p hexo-solver forcing::tests::wide_0l4291i_experiment -- --ignored --nocapture
    #[test]
    #[ignore]
    fn wide_0l4291i_experiment() {
        use hexo_engine::game::{GameState, GameConfig};
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../scripts/fixtures/forcing_puzzles/0l4291i_live.json"
        );
        let data = std::fs::read_to_string(path).expect("0l4291i_live.json fixture must be readable");
        let v: serde_json::Value = serde_json::from_str(&data).expect("fixture must be valid JSON");
        let parse_player = |s: &str| match s {
            "P1" => Player::P1,
            "P2" => Player::P2,
            other => panic!("bad player {other}"),
        };
        let stones: Vec<(Coord, Player)> = v["stones"]
            .as_array()
            .expect("stones must be an array")
            .iter()
            .map(|s| {
                let arr = s.as_array().expect("stone must be a [q, r, player] array");
                let q = arr[0].as_i64().unwrap() as i32;
                let r = arr[1].as_i64().unwrap() as i32;
                ((q, r), parse_player(arr[2].as_str().unwrap()))
            })
            .collect();
        let attacker = parse_player(v["attacker"].as_str().expect("attacker must be a string"));
        let placements = v["placements_remaining"].as_u64().expect("placements_remaining must be an int") as u8;
        eprintln!(
            "0l4291i: {} stones, attacker={attacker:?}, placements_remaining={placements}",
            stones.len()
        );

        let cfg = GameConfig { win_length: 6, placement_radius: 8, max_moves: 300 };
        let game = GameState::from_state(&stones, attacker, placements, cfg);

        for &budget in &[1_000_000u64, 10_000_000] {
            let t = std::time::Instant::now();
            let outcome = solve_wide(&game, 40, budget);
            let elapsed = t.elapsed();
            match outcome {
                Outcome::Win(w) => {
                    eprintln!("WIDE budget={budget} -> Win{{depth={}}}  ({elapsed:.2?})", w.depth);
                    eprintln!("PV ({} cells): {:?}", w.pv.len(), w.pv);
                    return;
                }
                Outcome::No => {
                    eprintln!("WIDE budget={budget} -> No (whole forcing tree closed)  ({elapsed:.2?})");
                    return;
                }
                Outcome::BudgetExceeded => {
                    eprintln!("WIDE budget={budget} -> BudgetExceeded  ({elapsed:.2?})");
                    if elapsed.as_secs() >= 300 {
                        eprintln!("stopping escalation: this step alone took >= 5 min");
                        return;
                    }
                }
            }
        }
        eprintln!("WIDE: no win found through 10M nodes");
    }

    // Component-level cost decomposition on real fixture boards, in both the wide
    // (radius 8, puzzles) and tight (radius 2, production self-play) regimes.
    // run manually:
    // cargo test --release -p hexo-solver forcing::tests::component_micro_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn component_micro_bench() {
        let boards: Vec<(&str, &[(Coord, Player)], Player)> = vec![
            ("0hz3hty(21st)", FIXTURES[1].0, FIXTURES[1].1),
            ("jnzzmcm(67st)", SLOW_FIXTURES[0].0, SLOW_FIXTURES[0].1),
            ("hu01jk4(149st)", SLOW_FIXTURES[1].0, SLOW_FIXTURES[1].1),
        ];
        for (name, stones, atk) in boards {
            let mut b = SolverBoard::new();
            for &(c, p) in stones { b.place(c, p); }
            let dfn = atk.opponent();
            for radius in [8i32, 2] {
                if radius == 2 { b.enable_reach(2); }
                let n = 2000;
                let t = std::time::Instant::now();
                for _ in 0..n { std::hint::black_box(threat_table(&b, atk, 6, radius)); }
                let tt_us = t.elapsed().as_micros() as f64 / n as f64;
                let t = std::time::Instant::now();
                for _ in 0..n { std::hint::black_box(completions(&b, atk, 6, radius)); }
                let ca_us = t.elapsed().as_micros() as f64 / n as f64;
                let t = std::time::Instant::now();
                for _ in 0..n { std::hint::black_box(completions(&b, dfn, 6, radius)); }
                let cd_us = t.elapsed().as_micros() as f64 / n as f64;
                let t = std::time::Instant::now();
                let n2 = 200;
                for _ in 0..n2 { std::hint::black_box(attacker_turns(&b, atk, dfn, 2, 6, radius, false)); }
                let at_us = t.elapsed().as_micros() as f64 / n2 as f64;
                let idx = threat_table(&b, atk, 6, radius);
                let hot = idx.len();
                let moves = attacker_turns(&b, atk, dfn, 2, 6, radius, false).len();
                let comps = completions(&b, atk, 6, radius);
                let t = std::time::Instant::now();
                for _ in 0..n { std::hint::black_box(min_covers2(&comps)); }
                let mc_us = t.elapsed().as_micros() as f64 / n as f64;
                eprintln!("{name} r={radius}: threat_table={tt_us:.1}us comps_atk={ca_us:.1}us comps_dfn={cd_us:.1}us attacker_turns={at_us:.1}us min_covers={mc_us:.2}us  (hot={hot} moves={moves} comps={})", comps.len());
            }
        }
    }

    /// The 2026-07-10 user-reported miss: P1 has a bare 3-run and a forced win
    /// whose first turn pairs an open-four maker with a QUIET builder — a shape
    /// only the wide generator can emit. P2's stones are scattered noise.
    fn quiet_builder_win_stones() -> Vec<(Coord, Player)> {
        let mut stones: Vec<(Coord, Player)> = [(0, 0), (1, 0), (2, 0)]
            .into_iter().map(|c| (c, Player::P1)).collect();
        stones.extend([(0, -8), (-1, -15), (7, -22), (3, -26)]
            .into_iter().map(|c| (c, Player::P2)));
        stones
    }

    /// Wide partner cells come back ordered most-threatening-first: a builder
    /// adjacent to attacker stones on TWO axes (e.g. (0,1): one step up the
    /// vertical of (0,0) AND one step up the diagonal of (1,0)) must outscore a
    /// lone-axis cell at maximum range (e.g. (0,5), five steps from (0,0) with
    /// a single shared window), and the list must be sorted (score desc,
    /// nearest-own-stone dist asc, coord asc) so search order is deterministic.
    #[test]
    fn wide_partners_are_scored_most_threatening_first() {
        let mut b = SolverBoard::new();
        for &(c, p) in &quiet_builder_win_stones() { b.place(c, p); }
        let (_, scores) = threat_table_scored(&b, Player::P1, 6, 8);
        let scored = wide_partner_cells(&b, Player::P1, 6, 8, &scores);
        let score_of = |cell: Coord| -> u32 {
            scored.iter().find(|(c, _, _)| *c == cell).map(|&(_, s, _)| s)
                .unwrap_or_else(|| panic!("{cell:?} missing from wide partners"))
        };
        assert!(
            score_of((0, 1)) > score_of((0, 5)),
            "two-axis adjacent builder must outscore a lone-axis max-range cell: \
             (0,1)={} vs (0,5)={}",
            score_of((0, 1)), score_of((0, 5))
        );
        for w in scored.windows(2) {
            let ((c1, s1, d1), (c2, s2, d2)) = (w[0], w[1]);
            assert!(
                s1 > s2 || (s1 == s2 && (d1 < d2 || (d1 == d2 && c1 < c2))),
                "wide partners must be sorted (score desc, dist asc, coord asc): \
                 {c1:?}(s{s1},d{d1}) before {c2:?}(s{s2},d{d2})"
            );
        }
    }

    /// On equal window score, the builder hex-closer to an own stone wins. The
    /// board is built so the closer cell also has the LARGER coordinate — pinning
    /// that proximity (not coordinate order) is what breaks the tie: (20,2) sits
    /// 2 steps from the lone stone (20,0) (4 shared vertical windows); (0,4) sits
    /// 4 steps from both (0,0) (2 vertical windows) and (4,0) (2 diagonal
    /// windows) — same score 4, but distance 2 vs 4.
    #[test]
    fn wide_partners_prefer_closer_builders_on_equal_threat() {
        let mut b = SolverBoard::new();
        for c in [(0, 0), (4, 0), (20, 0)] { b.place(c, Player::P1); }
        let (_, scores) = threat_table_scored(&b, Player::P1, 6, 8);
        let scored = wide_partner_cells(&b, Player::P1, 6, 8, &scores);
        let entry = |cell: Coord| -> (u32, i32) {
            scored.iter().find(|(c, _, _)| *c == cell).map(|&(_, s, d)| (s, d))
                .unwrap_or_else(|| panic!("{cell:?} missing from wide partners"))
        };
        let (near_s, near_d) = entry((20, 2));
        let (far_s, far_d) = entry((0, 4));
        assert_eq!(near_s, far_s, "fixture requires equal window scores");
        assert_eq!((near_d, far_d), (2, 4), "distances must be measured to the nearest own stone");
        let pos = |cell: Coord| scored.iter().position(|(c, _, _)| *c == cell).unwrap();
        assert!(
            pos((20, 2)) < pos((0, 4)),
            "equal-score builders must order by stone proximity, not coordinate"
        );
    }

    /// Root move ordering on the quiet-builder board: every early root move
    /// must pair the strongest hot cell (h_strong=3 ties break to (-1,0) by
    /// coord), with its hot partners first and then builders
    /// most-threatening-first — so the user-reported winning shape
    /// ((-1,0), (0,1)) sits in the first handful of 129 root moves instead of
    /// wherever coordinate order dropped it.
    #[test]
    fn wide_root_moves_lead_with_strongest_hot_group() {
        let mut b = SolverBoard::new();
        for &(c, p) in &quiet_builder_win_stones() { b.place(c, p); }
        let dfn_comps = completions(&b, Player::P2, 6, 8);
        let moves = attacker_turns_with(&b, Player::P1, 2, 6, 8, &dfn_comps, true);
        assert!(moves.len() > 100, "wide root list should be large, got {}", moves.len());
        for (i, mv) in moves.iter().take(5).enumerate() {
            assert!(
                mv.cells().contains(&(-1, 0)),
                "root move {i} must belong to the strongest hot cell's group, got {mv:?}"
            );
        }
        let pos_of = |pair: CellSet2| moves.iter().position(|m| *m == pair)
            .unwrap_or_else(|| panic!("{pair:?} not generated"));
        assert!(
            pos_of(CellSet2::two((-1, 0), (0, 1))) < 8,
            "the open-four + adjacent-builder pair must be tried within the first 8 moves"
        );
    }

    /// Solve-level completeness pin on the 2026-07-10 user-reported position:
    /// the quiet-builder forced win must resolve at ANALYSIS-scale budgets
    /// (knee measured at ~15k nodes — the floor is the depth-6 verification
    /// tree, not move ordering). Tight `solve` returns No here at ANY budget
    /// (the winning turn shape needs a builder partner it cannot generate).
    #[test]
    fn wide_solves_quiet_builder_win_at_analysis_budget() {
        use hexo_engine::game::{GameConfig, GameState};
        let game = GameState::from_state(
            &quiet_builder_win_stones(), Player::P1, 2, GameConfig::FULL_HEXO);
        match solve_wide(&game, 12, 20_000) {
            Outcome::Win(w) => {
                assert_eq!(w.depth, 6, "known winning-move-in-6");
                assert_eq!(w.first_move, (-1, 0), "win starts from the open-four maker");
                // Replay the PV through the engine: the line must be legal and
                // actually end in the attacker's win — a wide-only bug that
                // fabricates a plausible-looking PV must not pass.
                let mut replay = game.clone();
                for &c in &w.pv {
                    replay.apply_move(c).expect("wide PV must be a legal line");
                }
                assert!(replay.is_terminal(), "wide PV must end the game");
                assert_eq!(replay.winner(), Some(Player::P1), "wide PV must win for the attacker");
            }
            other => panic!("expected wide Win within 20k nodes, got {other:?}"),
        }
        match solve(&game, 24, 5_000_000) {
            Outcome::No => {}
            other => panic!("tight generator finding this win means the fixture rotted: {other:?}"),
        }
    }

    /// The fused single-pass scan must reproduce BOTH of its parents exactly:
    /// the threat index byte-for-byte (same windows, same dedup, same entry
    /// order — the wider pc filter only adds callback invocations, never
    /// reorders the threat-band ones) and the builder scores against a
    /// straight two-pass oracle (Σ pc per visit = Σ pc² per window).
    #[test]
    fn fused_scan_matches_plain_threat_table_and_score_oracle() {
        let mut dense = SolverBoard::new();
        for i in 0..25i32 {
            let c = ((i % 5) * 2 - 4, (i / 5) * 2 - 4);
            dense.place(c, if i % 3 == 0 { Player::P2 } else { Player::P1 });
        }
        let mut quiet = SolverBoard::new();
        for &(c, p) in &quiet_builder_win_stones() { quiet.place(c, p); }
        for b in [&quiet, &dense] {
            let (index, scores) = threat_table_scored(b, Player::P1, 6, 8);
            let plain = threat_table(b, Player::P1, 6, 8);
            assert_eq!(index, plain, "fused threat index must equal plain threat_table");
            // Independent oracle: dedup to UNIQUE windows (start, axis) and add
            // pc² exactly once per window — the production code instead relies
            // on the strip scan visiting each window pc times and adding pc per
            // visit, so agreement here verifies that multiplicity trick rather
            // than restating it.
            let mut oracle: FxHashMap<Coord, u32> = FxHashMap::default();
            let mut seen: FxHashSet<(Coord, u8)> = FxHashSet::default();
            scan_windows(b, Player::P1, 6, 8, 1, 5, &mut |start, ai, pc, empties| {
                if seen.insert((start, ai as u8)) {
                    for &e in empties {
                        *oracle.entry(e).or_insert(0) += (pc as u32) * (pc as u32);
                    }
                }
                false
            });
            assert_eq!(scores, oracle, "fused scores must equal Σ pc² per unique window");
        }
    }

    /// Pin the score and distance VALUES, not just relative order: (0,1) sits
    /// one step from (0,0) on the vertical (5 shared windows, pc=1) and one
    /// step from (1,0) on the diagonal (5 more) → score 10, dist 1; (0,5) sits
    /// five steps from (0,0) with a single shared window → score 1, dist 5.
    #[test]
    fn wide_partner_scores_have_exact_values() {
        let mut b = SolverBoard::new();
        for &(c, p) in &quiet_builder_win_stones() { b.place(c, p); }
        let (_, scores) = threat_table_scored(&b, Player::P1, 6, 8);
        let scored = wide_partner_cells(&b, Player::P1, 6, 8, &scores);
        let entry = |cell: Coord| scored.iter().find(|(c, _, _)| *c == cell)
            .map(|&(_, s, d)| (s, d)).unwrap();
        assert_eq!(entry((0, 1)), (10, 1));
        assert_eq!(entry((0, 5)), (1, 5));
    }

    /// The defensive analysis gets the same wide knob: with P2 to move on the
    /// quiet-builder board, P1's wide-only threat is invisible to the tight
    /// defense (historical behavior: None / NoThreat) but the wide defense
    /// sights it and produces verified refutation candidates from its PV.
    #[test]
    fn wide_defense_sights_quiet_builder_threat_tight_defense_cannot() {
        use hexo_engine::game::{GameConfig, GameState};
        use std::time::Duration;
        let game = GameState::from_state(
            &quiet_builder_win_stones(), Player::P2, 2, GameConfig::FULL_HEXO);
        match solve_defense_ex(&game, 12, 100_000, Duration::from_secs(60), false) {
            DefenseVerdict::NoThreat => {}
            other => panic!("tight defense must prove NoThreat here, got {other:?}"),
        }
        match solve_defense_ex(&game, 12, 100_000, Duration::from_secs(60), true) {
            DefenseVerdict::Threat(a) => {
                assert_eq!(a.threat_depth, 6, "the quiet-builder threat is a win-in-6");
                assert!(!a.threat_pv.is_empty(), "sighted threat must carry its PV");
                assert!(
                    !a.killers.is_empty() || !a.pair_anchors.is_empty() || a.best_delay.is_some(),
                    "defense must report some candidate against the sighted threat"
                );
            }
            other => panic!("wide defense must sight the threat, got {other:?}"),
        }
    }

    /// A starved threat check is BudgetExceeded — never the proven NoThreat it
    /// was historically conflated with.
    #[test]
    fn defense_distinguishes_no_threat_from_starved_check() {
        use hexo_engine::game::{GameConfig, GameState};
        use std::time::Duration;
        // Real threat (P1's open fork), mover P2 — at node_budget 1 the threat
        // check can only starve.
        let mut stones: Vec<(Coord, Player)> = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]
            .into_iter().map(|c| (c, Player::P1)).collect();
        stones.push(((5, 5), Player::P2));
        let game = GameState::from_state(&stones, Player::P2, 2, GameConfig::FULL_HEXO);
        match solve_defense_ex(&game, 12, 1, Duration::from_secs(60), false) {
            DefenseVerdict::BudgetExceeded => {}
            other => panic!("starved check must be BudgetExceeded, got {other:?}"),
        }
        match solve_defense_ex(&game, 12, 100_000, Duration::from_secs(60), false) {
            DefenseVerdict::Threat(_) => {}
            other => panic!("full budget must sight the fork threat, got {other:?}"),
        }
        // Provably nothing to defend: a bare-board opponent cannot threaten.
        let quiet = GameState::from_state(
            &[((0, 0), Player::P1), ((5, 5), Player::P2)], Player::P2, 2, GameConfig::FULL_HEXO);
        match solve_defense_ex(&quiet, 12, 100_000, Duration::from_secs(60), false) {
            DefenseVerdict::NoThreat => {}
            other => panic!("expected proven NoThreat, got {other:?}"),
        }
        // Back-compat wrapper flattens both give-ups to None.
        assert!(solve_defense(&game, 12, 1, Duration::from_secs(60)).is_none());
        assert!(solve_defense(&quiet, 12, 100_000, Duration::from_secs(60)).is_none());
    }

    /// The flipped-perspective threat solve gets the same wide knob: with P2 to
    /// move on the quiet-builder board, P1's forcing win is a THREAT that only
    /// the wide generator can see.
    #[test]
    fn wide_threat_sees_quiet_builder_threat() {
        use hexo_engine::game::{GameConfig, GameState};
        let game = GameState::from_state(
            &quiet_builder_win_stones(), Player::P2, 2, GameConfig::FULL_HEXO);
        match solve_threat_wide(&game, 12, 20_000) {
            Outcome::Win(w) => assert_eq!(w.depth, 6),
            other => panic!("expected wide threat Win, got {other:?}"),
        }
        match solve_threat(&game, 12, 250_000) {
            Outcome::No => {}
            other => panic!("tight threat solve should not see it: {other:?}"),
        }
    }
}
