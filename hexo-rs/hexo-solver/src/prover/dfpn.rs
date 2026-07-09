//! Nagai-style depth-first proof-number search (df-pn) over the shared kernel.
//!
//! df-pn is a depth-first reformulation of best-first PN search: it walks the
//! AND/OR tree with a pair of thresholds `(thpn, thdn)` and only backs out of a
//! node once its proof or disproof number reaches its threshold, using a
//! transposition table to remember partial progress between re-visits (so it
//! behaves like best-first PN while keeping df memory discipline). We apply the
//! `1 + ε` threshold trick (Pawlewicz & Lew 2007) to cut TT re-expansions.
//!
//! Node semantics come entirely from [`KernelCtx`] (win-now, B≥3-unstoppable,
//! defender-first refutation, exact min-cover replies), so a df-pn WIN/NO is the
//! same predicate the `idtt` baseline proves — only the search order differs.
//!
//! **DAG soundness.** Positions only ever gain stones, so the state graph is
//! acyclic and every node's value is path-independent: plain TT reuse keyed on
//! `(hash, kind, placements)` needs no graph-history-interaction handling.
//!
//! **Termination / honesty.** df-pn relies on the TT to store increasing pn/dn
//! across re-visits; with a bounded, evicting TT there is a theoretical
//! re-expansion risk, so the node budget + wall deadline + a hard recursion-depth
//! guard are the backstop. On exhaustion the driver returns `BUDGET_EXCEEDED`; it
//! never fabricates WIN or NO.

use super::io::{Stats, Verdict};
use super::kernel::{AndEval, KernelCtx, Node, OrEval};
use super::pn::{INF, PnSearch, ProofTt, node_key, one_plus_eps, sat_add};
use super::{Ctl, DriverResult, ProverConfig};
use crate::forcing::CellSet2;
use hexo_engine::types::Coord;
use rustc_hash::FxHashSet;
use std::time::Instant;

const EPS: f64 = 0.25;
/// Descent-selection bias (proof-number units per move rank) at OR nodes. The
/// kernel emits attacker turns best-first (highest threat count); trusting that
/// order focuses the proof search. Only reorders exploration — node pn/dn stay
/// exact, so soundness is untouched. 0 = pure proof-number order.
const MOVE_RANK_BIAS: u32 = 0;
/// Hard recursion-depth ceiling: a single forcing line deeper than this trips
/// `BUDGET_EXCEEDED` rather than risk unbounded board growth / stack use. Real
/// wins are shallow (corpus depth ≤ 6; the deepest composition is ~18 turns ≈ 40
/// plies), so this only fires on pathological non-terminating forcing.
const MAX_PLY: u32 = 1024;

pub struct Dfpn<'a> {
    k: KernelCtx,
    tt: ProofTt,
    ctl: &'a Ctl,
    nodes: u64,
    budget: u64,
    exceeded: bool,
    /// Node keys that reached `pn == 0` (proved). Used only for PV reconstruction;
    /// bounded by the size of the proof DAG, not the whole search.
    proven: FxHashSet<u64>,
    /// Reusable scratch for one node's child (pn, dn) pairs (avoids per-expansion
    /// allocation). Only read before the recursive descend, so a nested `mid`
    /// re-using it is safe.
    kids: Vec<(u32, u32)>,
    max_depth: u32,
    /// PDS-PN mode: when true, the first descent into a frontier node runs a
    /// bounded level-2 best-first PN search to seed its `(pn, dn)` (PN²-style; the
    /// PN tree is discarded, only the node's numbers persist in the TT). df-pn
    /// leaves this off and uses the cheap `(1, n)` estimate.
    pds_mode: bool,
    pn: PnSearch,
    pn2_nodes: u64,
    /// Node keys already level-2-seeded (so PN runs at most once per node).
    pn_seeded: FxHashSet<u64>,
    leaf_solves: u64,
}

impl<'a> Dfpn<'a> {
    fn new(k: KernelCtx, cfg: &ProverConfig, ctl: &'a Ctl, pds_mode: bool) -> Dfpn<'a> {
        Dfpn {
            k,
            tt: ProofTt::new(cfg.tt_mb),
            ctl,
            nodes: 0,
            budget: cfg.node_budget,
            exceeded: false,
            proven: FxHashSet::default(),
            kids: Vec::new(),
            max_depth: 0,
            pds_mode,
            pn: PnSearch::new(cfg.pn2_nodes),
            pn2_nodes: cfg.pn2_nodes,
            pn_seeded: FxHashSet::default(),
            leaf_solves: 0,
        }
    }

    /// PDS-PN level-2 seed: on the first descent into `node` (board at its
    /// position), run a bounded best-first PN search and cache the resulting
    /// `(pn, dn)` in the TT. The PN tree itself is discarded (PN²-style memory
    /// discipline — only the node's numbers survive). No-op in df-pn mode or on a
    /// node already seeded/resolved.
    fn pn_seed(&mut self, node: Node) {
        if !self.pds_mode {
            return;
        }
        let key = node_key(self.k.hash(), node);
        if !self.pn_seeded.insert(key) {
            return;
        }
        if let Some((pn, dn)) = self.tt.probe(key)
            && (pn == 0 || dn == 0)
        {
            return; // already resolved
        }
        let (pn, dn) = self.pn.search(&mut self.k, node);
        self.leaf_solves += 1;
        if pn == 0 {
            self.proven.insert(key);
            // Export the level-2 subtree's proven positions so PV reconstruction
            // can descend through this frontier (the PN tree is otherwise discarded).
            self.pn.collect_proven(&mut self.proven);
        }
        // Store with high work so the (expensive) PN result is retained.
        self.tt.store(key, pn, dn, self.pn2_nodes.min(INF as u64) as u32);
    }

    /// The `(pn, dn)` of a child node whose position the board currently reflects,
    /// using **immediate evaluation with the "1 and n" initialization** (Allis;
    /// the paper's speed-critical heuristic): terminals resolve to `(0, INF)` /
    /// `(INF, 0)` at generation, and an unexpanded internal node is seeded from its
    /// branching factor — an OR (attacker) node to `(1, m_moves)`, an AND
    /// (defender) node to `(c_covers, 1)`. This gives df-pn tree-shape information
    /// immediately instead of the shapeless `(1, 1)`, which is the difference
    /// between diving into deep non-winning forcing lines and going straight at
    /// the shallow proof. The estimate is cached in the TT (work = 1) so it is
    /// computed once per position; a later `mid` on the node refines it.
    fn child_val(&mut self, node: Node) -> (u32, u32) {
        let key = node_key(self.k.hash(), node);
        if let Some(v) = self.tt.probe(key) {
            return v;
        }
        let (pn, dn, _terminal) = super::pn::eval_child(&mut self.k, node);
        if pn == 0 {
            self.proven.insert(key);
        }
        self.tt.store(key, pn, dn, 1);
        (pn, dn)
    }

    #[inline]
    fn tick(&mut self, depth: u32) -> bool {
        self.nodes += 1;
        if depth > self.max_depth {
            self.max_depth = depth;
        }
        if self.nodes > self.budget || depth > MAX_PLY {
            self.exceeded = true;
        } else if self.nodes & 0x1FFF == 0 && self.ctl.expired() {
            self.exceeded = true;
        }
        self.exceeded
    }

    /// The df-pn MID routine: expand the node whose position the board currently
    /// reflects, refining its `(pn, dn)` until one reaches its threshold, and
    /// return the node's numbers. Stores results in the TT.
    fn mid(&mut self, node: Node, thpn: u32, thdn: u32, depth: u32) -> (u32, u32) {
        if self.tick(depth) {
            return (1, 1); // bail; caller ignores the value once `exceeded`
        }
        let entry_nodes = self.nodes;
        let key = node_key(self.k.hash(), node);

        // Early TT cutoff (the paper's `lookUpTT` at the top of MID): if the stored
        // numbers already resolve the node or meet the caller's thresholds, return
        // them without re-expanding. Without this, `mid` re-explores proven/settled
        // subtrees on every revisit — the difference between a focused proof and a
        // brute-force sweep of the whole depth-limited tree.
        if let Some((pn, dn)) = self.tt.probe(key)
            && (pn == 0 || dn == 0 || pn >= thpn || dn >= thdn)
        {
            return (pn, dn);
        }

        // Delayed evaluation: classify (terminal or child list) on expansion.
        match node {
            Node::Or { placements } => match self.k.or_eval(placements) {
                OrEval::WinNow => {
                    self.proven.insert(key);
                    self.tt.store(key, 0, INF, 1);
                    (0, INF)
                }
                OrEval::Loss => {
                    self.tt.store(key, INF, 0, 1);
                    (INF, 0)
                }
                OrEval::Moves(mvs) => self.expand_or(key, &mvs, thpn, thdn, depth, entry_nodes),
            },
            Node::And => match self.k.and_eval() {
                AndEval::AttackerWin => {
                    self.proven.insert(key);
                    self.tt.store(key, 0, INF, 1);
                    (0, INF)
                }
                AndEval::Loss => {
                    self.tt.store(key, INF, 0, 1);
                    (INF, 0)
                }
                AndEval::Covers(covers) => {
                    self.expand_and(key, &covers, thpn, thdn, depth, entry_nodes)
                }
            },
        }
    }

    /// OR node (attacker): `pn = min` child pn, `dn = sum` child dn. Descend into
    /// the min-pn child (proof-number-first).
    fn expand_or(
        &mut self,
        key: u64,
        mvs: &[CellSet2],
        thpn: u32,
        thdn: u32,
        depth: u32,
        entry_nodes: u64,
    ) -> (u32, u32) {
        loop {
            if self.exceeded {
                break;
            }
            // Read children's current numbers from the TT (immediate "1 and n"
            // eval on first sight). `pn = min` child pn, `dn = sum` child dn.
            self.kids.clear();
            let mut pn = INF;
            let mut dn = 0u32;
            for mv in mvs.iter() {
                self.k.place_attacker(mv);
                let (cpn, cdn) = self.child_val(Node::And);
                self.k.unplace(mv);
                pn = pn.min(cpn);
                dn = sat_add(dn, cdn);
                self.kids.push((cpn, cdn));
            }
            let work = (self.nodes - entry_nodes).min(INF as u64) as u32;
            if pn == 0 {
                self.proven.insert(key);
                self.tt.store(key, 0, INF, work);
                return (0, INF);
            }
            if pn >= thpn || dn >= thdn {
                self.tt.store(key, pn, dn, work);
                return (pn, dn);
            }
            self.tt.store(key, pn, dn, work); // record progress before descending
            // Descent selection is **rank-biased** by `MOVE_RANK_BIAS * i`: the
            // kernel returns attacker turns best-first (highest threat count), and
            // in this VCF domain the winning move is almost always among the top
            // few, so trusting that order focuses the proof instead of sweeping the
            // whole depth-limited tree. The node's reported `pn`/`dn` stay the true
            // min/sum (soundness), so this only reorders exploration.
            //
            // The child's proof threshold is derived from the *biased* runner-up
            // score so it is consistent with selection: the selected child is
            // searched until its own biased score would lose the lead. (Deriving
            // the threshold from the unbiased second-best instead lets the selected
            // child start already over-threshold, so it returns without progress
            // and the parent spins — the bug this replaces.)
            let mut best = 0usize;
            let mut best_score = u32::MAX;
            let mut second_score = u32::MAX;
            for (i, &(cpn, _)) in self.kids.iter().enumerate() {
                let score = cpn.saturating_add((i as u32).saturating_mul(MOVE_RANK_BIAS));
                if score < best_score {
                    second_score = best_score;
                    best_score = score;
                    best = i;
                } else if score < second_score {
                    second_score = score;
                }
            }
            let best_child_dn = self.kids[best].1;
            // Convert the runner-up biased score back to a true-pn threshold for the
            // selected child (subtract its own rank bias); `one_plus_eps` inflates.
            let sel_bias = (best as u32).saturating_mul(MOVE_RANK_BIAS);
            let thresh_pn = second_score.saturating_sub(sel_bias);
            let child_thpn = thpn.min(one_plus_eps(thresh_pn, EPS));
            // Remaining disproof budget for the chosen child = thdn minus the
            // disproof numbers of its siblings.
            let child_thdn = thdn.saturating_sub(dn.saturating_sub(best_child_dn));
            let mv = mvs[best];
            self.k.place_attacker(&mv);
            self.pn_seed(Node::And);
            self.mid(Node::And, child_thpn, child_thdn, depth + 1);
            self.k.unplace(&mv);
        }
        self.tt.probe(key).unwrap_or((1, 1))
    }

    /// AND node (defender): `pn = sum` child pn, `dn = min` child dn. Descend into
    /// the min-dn child (disproof-number-first).
    fn expand_and(
        &mut self,
        key: u64,
        covers: &[CellSet2],
        thpn: u32,
        thdn: u32,
        depth: u32,
        entry_nodes: u64,
    ) -> (u32, u32) {
        loop {
            if self.exceeded {
                break;
            }
            let mut pn = 0u32;
            let mut dn = INF;
            let mut best = 0usize;
            let mut best_dn = INF;
            let mut second_dn = INF;
            let mut best_child_pn = 0u32;
            for (i, cov) in covers.iter().enumerate() {
                self.k.place_defender(cov);
                let (cpn, cdn) = self.child_val(Node::Or { placements: 2 });
                self.k.unplace(cov);
                pn = sat_add(pn, cpn);
                dn = dn.min(cdn);
                if cdn < best_dn {
                    second_dn = best_dn;
                    best_dn = cdn;
                    best = i;
                    best_child_pn = cpn;
                } else if cdn < second_dn {
                    second_dn = cdn;
                }
            }
            let work = (self.nodes - entry_nodes).min(INF as u64) as u32;
            if pn == 0 {
                self.proven.insert(key);
                self.tt.store(key, 0, INF, work);
                return (0, INF);
            }
            if pn >= thpn || dn >= thdn {
                self.tt.store(key, pn, dn, work);
                return (pn, dn);
            }
            self.tt.store(key, pn, dn, work);
            let child_thdn = thdn.min(one_plus_eps(second_dn, EPS));
            let child_thpn = thpn.saturating_sub(pn.saturating_sub(best_child_pn));
            let cov = covers[best];
            self.k.place_defender(&cov);
            self.pn_seed(Node::Or { placements: 2 });
            self.mid(Node::Or { placements: 2 }, child_thpn, child_thdn, depth + 1);
            self.k.unplace(&cov);
        }
        self.tt.probe(key).unwrap_or((1, 1))
    }

    /// Reconstruct one winning line by descending proven children. Mirrors the
    /// kernel's `extract_pv`: attacker turns interleaved with the prolonging /
    /// futile defender replies, ending in six-in-a-row. Board is mutated (the PV
    /// is applied); callers rebuild a fresh context afterwards if needed.
    fn build_pv(&mut self, root: Node) -> (Vec<Coord>, u8) {
        let mut pv: Vec<Coord> = Vec::new();
        let mut turns: u8 = 0;
        let mut node = root;
        loop {
            match node {
                Node::Or { placements } => {
                    if let Some(cells) = self.k.win_now_cells(placements) {
                        self.k.place_attacker(&cells);
                        for &c in cells.cells() {
                            pv.push(c);
                        }
                        turns = turns.saturating_add(1);
                        break;
                    }
                    let moves = match self.k.or_eval(placements) {
                        OrEval::Moves(m) => m,
                        _ => break,
                    };
                    let mut chosen: Option<CellSet2> = None;
                    for mv in moves.iter() {
                        self.k.place_attacker(mv);
                        let ck = node_key(self.k.hash(), Node::And);
                        if self.proven.contains(&ck) {
                            chosen = Some(*mv);
                            break; // leave placed
                        }
                        self.k.unplace(mv);
                    }
                    let mv = match chosen {
                        Some(m) => m,
                        None => break,
                    };
                    for &c in mv.cells() {
                        pv.push(c);
                    }
                    turns = turns.saturating_add(1);
                    node = Node::And;
                }
                Node::And => match self.k.and_eval() {
                    AndEval::AttackerWin => {
                        let pair = self.k.futile_pair();
                        self.k.place_defender(&pair);
                        for &c in pair.cells() {
                            pv.push(c);
                        }
                        node = Node::Or { placements: 2 };
                    }
                    AndEval::Covers(covers) => {
                        let mut chosen: Option<CellSet2> = None;
                        for cov in covers.iter() {
                            self.k.place_defender(cov);
                            let ck = node_key(self.k.hash(), Node::Or { placements: 2 });
                            if self.proven.contains(&ck) {
                                chosen = Some(*cov);
                                break;
                            }
                            self.k.unplace(cov);
                        }
                        let cov = match chosen {
                            Some(c) => c,
                            None => break,
                        };
                        for &c in cov.cells() {
                            pv.push(c);
                        }
                        node = Node::Or { placements: 2 };
                    }
                    AndEval::Loss => break,
                },
            }
        }
        (pv, turns)
    }

    fn stats(&self, elapsed: f64) -> Stats {
        Stats {
            nodes: self.nodes,
            tt_hits: self.tt.hits(),
            tt_bytes: self.tt.bytes(),
            elapsed_s: elapsed,
            leaf_solves: self.leaf_solves,
            line_steps: 0,
        }
    }
}

/// Run df-pn on a position.
pub fn solve(pos: &super::io::Position, cfg: &ProverConfig, ctl: &Ctl) -> DriverResult {
    solve_mode(pos, cfg, ctl, false)
}

/// Shared df-pn / PDS-PN driver. `pds_mode` enables the level-2 PN warm-start.
pub(crate) fn solve_mode(
    pos: &super::io::Position,
    cfg: &ProverConfig,
    ctl: &Ctl,
    pds_mode: bool,
) -> DriverResult {
    let wl = pos.config.win_length;
    if !KernelCtx::supported(wl) {
        return DriverResult::new(Verdict::BudgetExceeded);
    }
    let ctx = match KernelCtx::new_wide(
        &pos.stones,
        pos.attacker,
        wl,
        pos.config.placement_radius,
        cfg.wide,
    ) {
        Some(c) => c,
        None => return DriverResult::new(Verdict::BudgetExceeded),
    };
    let mut d = Dfpn::new(ctx, cfg, ctl, pds_mode);
    let root = Node::Or { placements: pos.placements_remaining };
    let t = Instant::now();
    let (pn, dn) = d.mid(root, INF, INF, 0);
    let elapsed = t.elapsed().as_secs_f64();
    if std::env::var("DFPN_DEBUG").is_ok() {
        eprintln!(
            "[dfpn] nodes={} pn={pn} dn={dn} max_depth={} tt_hits={} elapsed={elapsed:.2}s",
            d.nodes,
            d.max_depth,
            d.tt.hits()
        );
    }

    // Verdict discipline: only a genuinely resolved root yields WIN/NO.
    let verdict = if pn == 0 {
        Verdict::Win
    } else if dn == 0 && !d.exceeded {
        Verdict::No
    } else {
        Verdict::BudgetExceeded
    };

    let mut res = DriverResult::new(verdict);
    if verdict == Verdict::Win {
        // First try reconstructing the PV by descending proven children (the
        // spec's approach; exact for df-pn). In PDS-PN mode some winning nodes were
        // proven inside a *discarded* level-2 PN tree, so their children aren't in
        // the proven set and this walk can come up short — fall back to the kernel
        // solver's own PV, which is guaranteed valid for a genuinely forced win.
        if let Some(pv_ctx) = KernelCtx::new_wide(
            &pos.stones,
            pos.attacker,
            wl,
            pos.config.placement_radius,
            cfg.wide,
        ) {
            let proven = std::mem::take(&mut d.proven);
            d.k = pv_ctx;
            d.proven = proven;
            let (pv, turns) = d.build_pv(root);
            if pv_replays_win(pos, &pv) {
                res.pv = pv;
                res.depth = Some(turns);
            }
        }
        if res.pv.is_empty() {
            if let Some((pv, depth)) = kernel_pv(pos, cfg, ctl) {
                res.pv = pv;
                res.depth = Some(depth);
            }
        }
    }
    res.stats = d.stats(elapsed);
    res
}

/// Replay a candidate PV through the engine; true iff it legally ends in the
/// attacker's win. Guards the proof-search PV reconstruction.
fn pv_replays_win(pos: &super::io::Position, pv: &[Coord]) -> bool {
    if pv.is_empty() {
        return false;
    }
    let mut g = pos.to_game();
    for &c in pv {
        if g.is_terminal() || g.apply_move(c).is_err() {
            return false;
        }
    }
    g.is_terminal() && g.winner() == Some(pos.attacker)
}

/// Fallback PV for a proven win: the kernel solver's own principal variation
/// (same rules, guaranteed legal). Returns `(pv, depth)` or `None` if the solver
/// cannot reconstruct it within the configured budget (only possible for a
/// win too deep for the kernel — i.e. the optional deep-stretch regime). Honours
/// the caller's `Ctl` deadline/cancel so the fallback can never run past
/// `--time-limit` even under a very large `--node-budget`.
fn kernel_pv(pos: &super::io::Position, cfg: &ProverConfig, ctl: &Ctl) -> Option<(Vec<Coord>, u8)> {
    use crate::forcing::{Outcome, solve_limited};
    let game = pos.to_game();
    match solve_limited(&game, cfg.depth_cap, cfg.node_budget, cfg.wide, ctl.forcing_limits()) {
        Outcome::Win(w) => Some((w.pv, w.depth)),
        _ => None,
    }
}
