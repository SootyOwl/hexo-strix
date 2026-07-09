//! Shared proof-number machinery: the `INF` sentinel + saturating pn/dn algebra,
//! a byte-budgeted direct-mapped transposition table with effort-based
//! replacement, and the classic in-memory best-first PN search used as PDS-PN's
//! level-2 evaluator.
//!
//! Proof number `pn` = a lower bound on the leaves that must be proved to prove a
//! node; disproof number `dn` = the dual. At an OR node `pn = min` over children,
//! `dn = sum`; at an AND node they swap. A proved node is `(0, INF)`, a disproved
//! node `(INF, 0)`, an unexpanded unknown `(1, 1)`.

use super::kernel::{AndEval, KernelCtx, Node, OrEval};
use crate::forcing::CellSet2;
use rustc_hash::FxHashSet;

/// The proof/disproof "infinity". Kept well below `u32::MAX` so saturating sums of
/// several children never wrap; any pn/dn `>= INF` is treated as infinite.
pub(crate) const INF: u32 = 1 << 30;

/// Saturating add capped at `INF` (child pn/dn sums at OR-disproof / AND-proof).
#[inline]
pub(crate) fn sat_add(a: u32, b: u32) -> u32 {
    a.saturating_add(b).min(INF)
}

/// df-pn `1 + ε` threshold inflation (Pawlewicz & Lew 2007): search the current
/// best child until its number exceeds the runner-up by a factor `(1 + ε)`,
/// which sharply cuts transposition-table re-expansions vs. the plain `+1` rule.
#[inline]
pub(crate) fn one_plus_eps(second_best: u32, eps: f64) -> u32 {
    if second_best >= INF {
        return INF;
    }
    let inflated = ((second_best as f64 + 1.0) * (1.0 + eps)).ceil();
    if inflated >= INF as f64 { INF } else { inflated as u32 }
}

/// Immediate "1 and n" evaluation of a child node (board at its position):
/// terminals resolve to `(0, INF)` / `(INF, 0)`; internal nodes seed from their
/// branching factor. Returns `(pn, dn, is_terminal)`.
pub(crate) fn eval_child(k: &mut KernelCtx, node: Node) -> (u32, u32, bool) {
    match node {
        Node::Or { placements } => match k.or_eval(placements) {
            OrEval::WinNow => (0, INF, true),
            OrEval::Loss => (INF, 0, true),
            OrEval::Moves(m) => (1, (m.len() as u32).clamp(1, INF - 1), false),
        },
        Node::And => match k.and_eval() {
            AndEval::AttackerWin => (0, INF, true),
            AndEval::Loss => (INF, 0, true),
            AndEval::Covers(c) => ((c.len() as u32).clamp(1, INF - 1), 1, false),
        },
    }
}

#[inline]
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// A transposition-table key folding the board Zobrist hash together with the
/// node kind + placements, so an OR and AND node at the same board never collide.
#[inline]
pub(crate) fn node_key(hash: u64, node: Node) -> u64 {
    let (is_or, placements) = node.tag();
    let tag = ((placements as u64) << 1) | (is_or as u64);
    hash ^ mix(tag ^ 0xD1B5_4A32_D192_ED03)
}

#[derive(Clone, Copy)]
struct Slot {
    key: u64,
    pn: u32,
    dn: u32,
    /// Subtree effort (nodes expanded) that produced this entry; the replacement
    /// key — a colliding write keeps whichever entry cost more to compute.
    work: u32,
    occupied: bool,
}

/// A **2-way set-associative**, fixed-byte-budget transposition table (the paper's
/// "TwoBig" scheme). Correctness of the proof searches never depends on the TT (it
/// only accelerates re-visits), so a bounded evicting table is sound. But a 1-way
/// direct-mapped table has a sharp collision cliff: two hot keys that differ only
/// in a dropped index bit map to the same slot and evict each other every visit,
/// and losing a proven leaf freezes the parent's proof number at a non-progress
/// fixed point. Two ways per bucket let such a pair coexist, which is the
/// difference between converging in thousands of nodes and thrashing to the budget
/// on this domain. Replacement within a full bucket: never evict a resolved
/// (`pn==0 || dn==0`) entry for an unresolved one; otherwise evict the smaller
/// `work` (the cheaper-to-recompute subtree).
const WAYS: usize = 2;

pub(crate) struct ProofTt {
    slots: Vec<Slot>,
    /// Bucket-index mask (over `nbuckets`, a power of two).
    mask: usize,
    hits: u64,
    stored: u64,
}

impl ProofTt {
    /// Size to the largest power-of-two bucket count whose `WAYS` slots fit in
    /// `tt_mb` megabytes (minimum 1024 buckets so tiny configs still function).
    pub(crate) fn new(tt_mb: usize) -> ProofTt {
        let budget = tt_mb.max(1) * 1024 * 1024;
        let entry = std::mem::size_of::<Slot>() * WAYS;
        let raw = (budget / entry).max(1024);
        let nbuckets = (raw.next_power_of_two() / 2).max(1024);
        ProofTt {
            slots: vec![Slot { key: 0, pn: 0, dn: 0, work: 0, occupied: false }; nbuckets * WAYS],
            mask: nbuckets - 1,
            hits: 0,
            stored: 0,
        }
    }

    #[inline]
    fn base(&self, key: u64) -> usize {
        ((key as usize) & self.mask) * WAYS
    }

    /// Look up `(pn, dn)` for a node key; `None` on miss (unknown → `(1, 1)`).
    #[inline]
    pub(crate) fn probe(&mut self, key: u64) -> Option<(u32, u32)> {
        let b = self.base(key);
        for w in 0..WAYS {
            let s = self.slots[b + w];
            if s.occupied && s.key == key {
                self.hits += 1;
                return Some((s.pn, s.dn));
            }
        }
        None
    }

    /// Store `(pn, dn)` with the subtree `work` into the key's bucket: reuse the
    /// slot already holding this key, else an empty slot, else evict the bucket's
    /// weakest slot (an unresolved slot before a resolved one; then least `work`).
    #[inline]
    pub(crate) fn store(&mut self, key: u64, pn: u32, dn: u32, work: u32) {
        let b = self.base(key);
        // Same-key or empty slot: write there.
        let mut victim = b;
        let mut victim_rank = u64::MAX; // higher = more worth evicting
        for w in 0..WAYS {
            let s = self.slots[b + w];
            if !s.occupied || s.key == key {
                self.slots[b + w] = Slot { key, pn, dn, work, occupied: true };
                self.stored += 1;
                return;
            }
            // Eviction rank: unresolved entries (rank has high bit clear) are
            // evicted before resolved ones; within a class, smaller work first.
            let resolved = (s.pn == 0 || s.dn == 0) as u64;
            let rank = ((1 - resolved) << 32) | (u32::MAX - s.work.min(u32::MAX)) as u64;
            if rank > victim_rank || victim_rank == u64::MAX {
                victim_rank = rank;
                victim = b + w;
            }
        }
        // Only displace the chosen victim if the incoming entry is at least as
        // worth keeping: resolved always displaces unresolved; else compare work.
        let v = self.slots[victim];
        let incoming_resolved = pn == 0 || dn == 0;
        let victim_resolved = v.pn == 0 || v.dn == 0;
        let replace = match (incoming_resolved, victim_resolved) {
            (true, false) => true,
            (false, true) => false,
            _ => work >= v.work,
        };
        if replace {
            self.slots[victim] = Slot { key, pn, dn, work, occupied: true };
            self.stored += 1;
        }
    }

    pub(crate) fn hits(&self) -> u64 {
        self.hits
    }

    pub(crate) fn bytes(&self) -> u64 {
        (self.slots.len() * std::mem::size_of::<Slot>()) as u64
    }
}

// --------------------------------------------------------------------------
// Classic in-memory best-first PN search (PDS-PN level 2 / PN²-style evaluator).
// --------------------------------------------------------------------------

/// A node of the in-memory PN tree. Children are lazily generated on first
/// expansion; terminals carry `(0, INF)` / `(INF, 0)`.
struct PnNode {
    node: Node,
    /// The move (attacker turn or defender cover) that reaches this node from its
    /// parent; `None` at the root. Cells are (un)applied on the shared board as we
    /// walk up/down, so the board always reflects the current node.
    mv: Option<CellSet2>,
    pn: u32,
    dn: u32,
    expanded: bool,
    /// Indices into the arena; empty until expanded.
    children: Vec<usize>,
    parent: usize,
    terminal: bool,
    /// The node's `node_key` (board hash folded with kind/placements), captured
    /// when the board was at this node's position. Lets a proved level-2 subtree
    /// export its proven nodes into the level-1 proven-set for PV reconstruction.
    key: u64,
}

/// A bounded best-first PN search over the shared kernel. Returns refined
/// `(pn, dn)` estimates for a subposition, capped at `max_nodes` expansions
/// (PN²-style: the tree is discarded after the call). Used only to seed PDS-PN
/// level-1 leaf numbers with something better than the `(1, 1)` heuristic.
pub(crate) struct PnSearch {
    arena: Vec<PnNode>,
    max_nodes: u64,
    expansions: u64,
}

impl PnSearch {
    pub(crate) fn new(max_nodes: u64) -> PnSearch {
        PnSearch { arena: Vec::new(), max_nodes: max_nodes.max(1), expansions: 0 }
    }

    /// Run PN search from `root_node` (board in `k` must be at that position).
    /// Returns `(pn, dn)` of the root when it is (dis)proved or the node cap is
    /// hit. The board is restored to the root position on return.
    pub(crate) fn search(&mut self, k: &mut KernelCtx, root_node: Node) -> (u32, u32) {
        self.arena.clear();
        self.expansions = 0;
        self.arena.push(PnNode {
            node: root_node,
            mv: None,
            pn: 1,
            dn: 1,
            expanded: false,
            children: Vec::new(),
            parent: usize::MAX,
            terminal: false,
            key: node_key(k.hash(), root_node),
        });
        loop {
            let root = &self.arena[0];
            if root.pn == 0 || root.dn == 0 {
                break;
            }
            if self.expansions >= self.max_nodes {
                break;
            }
            // Descend to the most-proving node, applying moves to the board.
            let mut cur = 0usize;
            loop {
                let n = &self.arena[cur];
                if !n.expanded || n.children.is_empty() {
                    break;
                }
                // OR: follow min-pn child; AND: follow min-dn child.
                let is_or = n.node.is_or();
                let mut best = n.children[0];
                let mut best_val = u32::MAX;
                for &c in &n.children {
                    let cc = &self.arena[c];
                    let v = if is_or { cc.pn } else { cc.dn };
                    if v < best_val {
                        best_val = v;
                        best = c;
                    }
                }
                cur = best;
                let mv = self.arena[cur].mv;
                if let Some(mv) = mv {
                    // Apply the move that leads into `cur`. Attacker moves into AND
                    // children, defender covers into OR children.
                    let parent_is_or = self.arena[self.arena[cur].parent].node.is_or();
                    if parent_is_or {
                        k.place_attacker(&mv);
                    } else {
                        k.place_defender(&mv);
                    }
                }
            }
            // Expand `cur`.
            self.expand(k, cur);
            self.expansions += 1;
            // Back up, unwinding the board to the root.
            self.backup(cur);
            self.unwind(k, cur);
        }
        let (pn, dn) = (self.arena[0].pn, self.arena[0].dn);
        (pn, dn)
    }

    /// Export every proved (`pn == 0`) node's key from the current tree into
    /// `out`. Called after a `search` that proved its root, so the level-1 driver's
    /// proven-set gains the level-2 subtree's proven positions — which is what lets
    /// PV reconstruction descend through a node that PDS-PN closed via level-2 PN
    /// (whose tree is otherwise discarded). Sound: every such node is a genuine
    /// forced win under the shared kernel rules.
    pub(crate) fn collect_proven(&self, out: &mut FxHashSet<u64>) {
        for n in &self.arena {
            if n.pn == 0 {
                out.insert(n.key);
            }
        }
    }

    /// Apply the terminal test / child generation for node `cur` (board is at its
    /// position). Children are **immediately evaluated** with the "1 and n"
    /// initialization — each child is classified on generation, so terminals get
    /// `(0, INF)` / `(INF, 0)` at once and internal nodes are seeded from their
    /// branching factor rather than the shapeless `(1, 1)`. This is what makes a
    /// bounded PN search useful within `max_nodes`.
    fn expand(&mut self, k: &mut KernelCtx, cur: usize) {
        let node = self.arena[cur].node;
        self.arena[cur].expanded = true;
        // (child_node, move) list, and whether children are reached by an attacker
        // move (OR parent) or a defender cover (AND parent).
        let (child_node, moves, parent_is_or): (Node, Vec<CellSet2>, bool) = match node {
            Node::Or { placements } => match k.or_eval(placements) {
                OrEval::WinNow => return self.set_terminal(cur, 0, INF),
                OrEval::Loss => return self.set_terminal(cur, INF, 0),
                OrEval::Moves(mvs) => (Node::And, mvs.to_vec(), true),
            },
            Node::And => match k.and_eval() {
                AndEval::AttackerWin => return self.set_terminal(cur, 0, INF),
                AndEval::Loss => return self.set_terminal(cur, INF, 0),
                AndEval::Covers(covers) => (Node::Or { placements: 2 }, covers, false),
            },
        };
        for mv in moves {
            if parent_is_or {
                k.place_attacker(&mv);
            } else {
                k.place_defender(&mv);
            }
            let (pn, dn, terminal) = eval_child(k, child_node);
            let key = node_key(k.hash(), child_node);
            k.unplace(&mv);
            let idx = self.arena.len();
            self.arena.push(PnNode {
                node: child_node,
                mv: Some(mv),
                pn,
                dn,
                expanded: false,
                children: Vec::new(),
                parent: cur,
                terminal,
                key,
            });
            self.arena[cur].children.push(idx);
        }
        self.recompute(cur);
    }

    fn set_terminal(&mut self, cur: usize, pn: u32, dn: u32) {
        let n = &mut self.arena[cur];
        n.pn = pn;
        n.dn = dn;
        n.terminal = true;
    }

    /// Recompute a node's pn/dn from its children (OR: pn=min, dn=sum; AND swap).
    fn recompute(&mut self, cur: usize) {
        if self.arena[cur].terminal {
            return;
        }
        let is_or = self.arena[cur].node.is_or();
        let children = self.arena[cur].children.clone();
        if children.is_empty() {
            return;
        }
        let (mut min_pn, mut sum_pn) = (INF, 0u32);
        let (mut min_dn, mut sum_dn) = (INF, 0u32);
        for &c in &children {
            let cc = &self.arena[c];
            min_pn = min_pn.min(cc.pn);
            sum_pn = sat_add(sum_pn, cc.pn);
            min_dn = min_dn.min(cc.dn);
            sum_dn = sat_add(sum_dn, cc.dn);
        }
        let n = &mut self.arena[cur];
        if is_or {
            n.pn = min_pn;
            n.dn = sum_dn;
        } else {
            n.pn = sum_pn;
            n.dn = min_dn;
        }
    }

    /// Propagate updated numbers from `cur` up to the root.
    fn backup(&mut self, mut cur: usize) {
        loop {
            self.recompute(cur);
            let p = self.arena[cur].parent;
            if p == usize::MAX {
                break;
            }
            cur = p;
        }
    }

    /// Undo the board moves applied while descending to `cur`, back to the root.
    fn unwind(&mut self, k: &mut KernelCtx, mut cur: usize) {
        while self.arena[cur].parent != usize::MAX {
            let mv = self.arena[cur].mv;
            if let Some(mv) = mv {
                k.unplace(&mv);
            }
            cur = self.arena[cur].parent;
        }
    }
}
