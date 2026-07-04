//! `pub(crate)` facade over the validated VCF kernel in `crate::mcts::forcing`.
//!
//! Every driver (df-pn, PDS-PN, hybrid) evaluates node terminal status and
//! generates children **only** through this facade, so cross-driver agreement is
//! a property of shared code, not of three parallel re-derivations. The `idtt`
//! baseline calls the production `solve`/`solve_deadline` directly; those walk the
//! very same `completions`/`attacker_turns_with`/`min_covers2` primitives this
//! facade exposes, so all four drivers share one rule set.
//!
//! ## The AND/OR tree (exactly the ID search's implicit tree)
//!
//! - **OR node** = attacker to move with `placements ∈ {1,2}` remaining. The
//!   attacker needs *one* forcing continuation to win.
//!     - win-now: an attacker completion executable within `placements` → WIN.
//!     - otherwise children = `attacker_turns` (each B≥2 forcing turn); an empty
//!       list is a definitive LOSS (no forcing continuation).
//! - **AND node** = defender to move, evaluated on the post-attacker-move board.
//!   The attacker needs *all* defender replies refuted.
//!     - defender has an immediate 2-placement completion → LOSS (counter-threat
//!       refutation: the defender wins first).
//!     - `B = min_covers(attacker completions)`: `B < 2` → LOSS (the move was not
//!       actually forcing); `B ≥ 3` → WIN (unstoppable multi-threat); `B == 2` →
//!       children = the exact min-cover set, each leading to a fresh OR node with
//!       `placements = 2`.
//!
//! This mirrors `forcing::atk_within` / `forcing::def_within` line for line; the
//! only thing the proof-number drivers change is *search order*, never the rules.
//!
//! ## DAG soundness
//!
//! Every attacker turn and every defender cover only ever *adds* stones to the
//! board, so a position's Zobrist hash is a monotone function of a growing stone
//! multiset: the state graph is acyclic and a node's game-theoretic value is
//! path-independent. Plain transposition-table reuse keyed on `(hash, kind,
//! placements)` is therefore sound with no graph-history-interaction handling.

use crate::mcts::forcing::{
    CellSet2, SolverBoard, attacker_turns_with, completions, futile_defender_pair, min_covers2,
    MAX_WL,
};
use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::types::{Coord, Player};
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// (attacker completions, defender completions) of one position.
type NodeComps = (Vec<CellSet2>, Vec<CellSet2>);

/// Result of classifying an OR (attacker-to-move) node.
pub(crate) enum OrEval {
    /// The attacker completes six-in-a-row within the remaining placements.
    WinNow,
    /// No forcing continuation exists — a definitive disproof of this node.
    Loss,
    /// Forcing attacker turns, best-first by threat count. Never empty.
    Moves(Rc<Vec<CellSet2>>),
}

/// Result of classifying an AND (defender-to-move) node, evaluated on the board
/// *after* the attacker's move has been applied.
pub(crate) enum AndEval {
    /// `B ≥ 3`: the defender cannot cover every threat — attacker wins.
    AttackerWin,
    /// Defender completes first, or the attacker's move left `B < 2` (not
    /// forcing). Either way this branch is a disproof for the attacker.
    Loss,
    /// `B == 2`: the exact minimum-cover set the defender must choose among.
    /// Every reply must be refuted for the attacker to win. Never empty.
    Covers(Vec<CellSet2>),
}

/// Which side is to move at a node, and (for OR nodes) how many placements remain.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Node {
    /// Attacker to move with `placements` remaining this turn (1 or 2).
    Or { placements: u8 },
    /// Defender to move (always a full fresh attacker turn follows).
    And,
}

impl Node {
    #[inline]
    pub(crate) fn is_or(self) -> bool {
        matches!(self, Node::Or { .. })
    }
    /// TT-key discriminant: `(is_or, placements)`. AND nodes fold to placements 0
    /// (the following OR always restores placements = 2).
    #[inline]
    pub(crate) fn tag(self) -> (bool, u8) {
        match self {
            Node::Or { placements } => (true, placements),
            Node::And => (false, 0),
        }
    }
}

/// The shared kernel context: one mutable board walked by make/unmake, plus the
/// two budget-independent per-position memos (mirrors `forcing::SearchState`'s
/// `comps`/`gencache`). Reused across a driver's whole search — sound because both
/// caches are keyed by position, not by search budget.
pub(crate) struct KernelCtx {
    board: SolverBoard,
    atk: Player,
    dfn: Player,
    wl: u8,
    radius: i32,
    wide: bool,
    comps: FxHashMap<u64, Rc<NodeComps>>,
    gencache: FxHashMap<(u64, u8), Rc<Vec<CellSet2>>>,
}

impl KernelCtx {
    /// True iff this `(win_length, ...)` is analyzable by the kernel (the
    /// strip-scan buffers are stack-sized for `win_length ∈ 1..=MAX_WL`).
    pub(crate) fn supported(wl: u8) -> bool {
        (1..=MAX_WL).contains(&(wl as usize))
    }

    /// Build the kernel context for a position, using exactly the board-setup path
    /// `forcing::solve` uses: reserve the bounding box, enable O(1) reach tracking
    /// in the tight regime, then bulk-load stones. Returns `None` for configs the
    /// kernel cannot analyze (unsupported `win_length`, pathological coordinate
    /// spread) — the caller reports an honest give-up, never a bogus verdict.
    pub(crate) fn new_wide(
        stones: &[(Coord, Player)],
        attacker: Player,
        wl: u8,
        radius: i32,
        wide: bool,
    ) -> Option<KernelCtx> {
        if !Self::supported(wl) {
            return None;
        }
        let mut board = SolverBoard::new();
        if let Some(&(c0, _)) = stones.first() {
            let (mut lo, mut hi) = (c0, c0);
            for &(c, _) in stones {
                lo = (lo.0.min(c.0), lo.1.min(c.1));
                hi = (hi.0.max(c.0), hi.1.max(c.1));
            }
            // Same bounds guard as `forcing::solve_ex`: reject pathological spreads
            // before any grid allocation rather than aborting on overflow.
            const MAX_COORD: i32 = 1 << 30;
            if lo.0 < -MAX_COORD || lo.1 < -MAX_COORD || hi.0 > MAX_COORD || hi.1 > MAX_COORD {
                return None;
            }
            const GRID_PAD: i64 = 16;
            const MAX_GRID_CELLS: i64 = 64 * 1024 * 1024;
            let span_q = hi.0 as i64 - lo.0 as i64 + 6 * GRID_PAD;
            let span_r = hi.1 as i64 - lo.1 as i64 + 6 * GRID_PAD;
            if span_q.saturating_mul(span_r) > MAX_GRID_CELLS {
                return None;
            }
            board.reserve(lo, hi);
        }
        if radius < wl as i32 - 1 {
            board.enable_reach(radius);
        }
        for &(c, p) in stones {
            board.place(c, p);
        }
        Some(KernelCtx {
            board,
            atk: attacker,
            dfn: attacker.opponent(),
            wl,
            radius,
            wide,
            comps: FxHashMap::default(),
            gencache: FxHashMap::default(),
        })
    }

    #[inline]
    pub(crate) fn hash(&self) -> u64 {
        self.board.hash
    }

    /// (attacker completions, defender completions), memoized by `board.hash`.
    fn node_comps(&mut self) -> Rc<NodeComps> {
        let h = self.board.hash;
        if let Some(v) = self.comps.get(&h) {
            return Rc::clone(v);
        }
        let atk_c = completions(&self.board, self.atk, self.wl, self.radius);
        let dfn_c = completions(&self.board, self.dfn, self.wl, self.radius);
        let v = Rc::new((atk_c, dfn_c));
        self.comps.insert(h, Rc::clone(&v));
        v
    }

    /// Classify an OR (attacker-to-move) node with `placements` remaining.
    pub(crate) fn or_eval(&mut self, placements: u8) -> OrEval {
        let comps = self.node_comps();
        if comps.0.iter().any(|c| c.len() as u8 <= placements) {
            return OrEval::WinNow;
        }
        let key = (self.board.hash, placements);
        let moves = if let Some(v) = self.gencache.get(&key) {
            Rc::clone(v)
        } else {
            let m = Rc::new(attacker_turns_with(
                &self.board,
                self.atk,
                placements,
                self.wl,
                self.radius,
                &comps.1,
                self.wide,
            ));
            self.gencache.insert(key, Rc::clone(&m));
            m
        };
        if moves.is_empty() {
            OrEval::Loss
        } else {
            OrEval::Moves(moves)
        }
    }

    /// Classify an AND (defender-to-move) node on the post-attacker-move board.
    pub(crate) fn and_eval(&mut self) -> AndEval {
        let comps = self.node_comps();
        // Counter-threat refutation: the defender completes first.
        if comps.1.iter().any(|c| c.len() as u8 <= 2) {
            return AndEval::Loss;
        }
        let (bnum, covers) = min_covers2(&comps.0);
        if bnum < 2 {
            AndEval::Loss
        } else if bnum >= 3 {
            AndEval::AttackerWin
        } else {
            AndEval::Covers(covers)
        }
    }

    /// The greedy cosmetic 2-cell defender reply for a `B ≥ 3` PV step (reuses the
    /// production `extract_pv` helper so proof-search PVs read identically).
    pub(crate) fn futile_pair(&mut self) -> CellSet2 {
        let comps = self.node_comps();
        futile_defender_pair(&comps.0)
    }

    /// The first attacker completion executable within `placements` (for win-now
    /// PV emission). `None` only if the node is not actually a win-now.
    pub(crate) fn win_now_cells(&mut self, placements: u8) -> Option<CellSet2> {
        let comps = self.node_comps();
        comps.0.iter().copied().find(|c| c.len() as u8 <= placements)
    }

    #[inline]
    pub(crate) fn place(&mut self, cells: &CellSet2, player: Player) {
        for &c in cells.cells() {
            self.board.place(c, player);
        }
    }

    #[inline]
    pub(crate) fn unplace(&mut self, cells: &CellSet2) {
        for &c in cells.cells() {
            self.board.remove(c);
        }
    }

    #[inline]
    pub(crate) fn place_attacker(&mut self, cells: &CellSet2) {
        self.place(cells, self.atk);
    }
    #[inline]
    pub(crate) fn place_defender(&mut self, cells: &CellSet2) {
        self.place(cells, self.dfn);
    }

    /// True iff every cell of `cells` is currently empty on the board (a spine
    /// candidate must be playable).
    pub(crate) fn cells_empty(&self, cells: &CellSet2) -> bool {
        cells.cells().iter().all(|&c| self.board.get(c).is_none())
    }

    /// Build an engine `GameState` for the current board with the attacker to move
    /// (used by the hybrid's leaf solves, which call the production kernel solver).
    pub(crate) fn game_state(&self, placements: u8, max_moves: u32) -> GameState {
        let cfg = GameConfig {
            win_length: self.wl,
            placement_radius: self.radius,
            max_moves,
        };
        GameState::from_state(&self.board.stones, self.atk, placements, cfg)
    }
}
