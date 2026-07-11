//! EXPERIMENTAL quiet-first-turn VCT probe.
//!
//! Answers the class of position the VCF solver provably cannot (e.g. HDS
//! ymf9zzt): the attacker's FIRST turn is quiet — it creates no completion, so
//! no fully-forcing line exists — but one building turn may set up an
//! unstoppable VCF. The probe searches exactly one tempo loss:
//!
//! 1. Direct wide VCF ([`forcing::solve_wide`]) — if it already wins, done.
//! 2. Candidate quiet turns: pairs of the top builder cells, ordered by the
//!    wide generator's threat score ([`forcing::wide_partner_cells`]).
//! 3. Necessary-condition pruner: after the quiet turn, the attacker must own
//!    a latent VCF *if the defender passed*. A quiet turn that can't beat a
//!    passing defender beats nobody. Budget ladder: `No` is proven at any
//!    budget, so candidates screen cheaply and only ambiguous ones escalate.
//! 4. Bounded defender refutation: every defender reply pair drawn from a
//!    RELEVANCE ZONE (latent-VCF PV cells + the defender's own best builders +
//!    the attacker's best remaining builders) plus the anchor pairs
//!    `(zone cell, pass cell)` must still lose to a wide VCF.
//!    `No`/`BudgetExceeded` on any reply counts as a refutation — the probe
//!    never claims a win it could not re-prove.
//!
//! ## Proof reuse (threat monotonicity)
//!
//! Defender stones only REMOVE attacker windows and only ADD defender ones, so
//! a proven attacker proof on position X remains valid on X + extra defender
//! stones when those stones (a) touch none of the cells the proof relied on
//! (its SUPPORT — see `forcing::SearchState::support`) and (b) cannot combine
//! with existing defender stones AND the proof's own forced cover placements
//! into a defender completion at any node of the proof — the "defender wins
//! first" checks relied on their absence. (b) is enforced conservatively by
//! `defender_window_risk`: a reply cell in any attacker-free window whose
//! defender stones plus empty-support cells could reach `wl - 2` blocks reuse
//! (cover placements only ever land on support cells, so this bounds every
//! reachable proof node, not just the reply board).
//!
//! Two reuse levels: the pass-case proof (level 0), and the anchor `(d, pass)`
//! proofs (level 1) — anchors are legal defenses needing verification anyway,
//! and dropping the far-periphery pass stone from a position only helps the
//! attacker. Level 1 additionally requires the wide-radius regime
//! (`placement_radius >= wl - 1`): in the tight regime a proof move could be
//! legal only via the pass stone, so anchors cover nothing there.
//!
//! ## Soundness contract (read before trusting a `QuietWin`)
//!
//! A [`ProbeOutcome::QuietWin`] is a win against the BOUNDED defender model,
//! not a proof: a defender placement outside the relevance zone is assumed
//! irrelevant (the zone hypothesis — the same idea Connect6 proofs formalize
//! as relevance zones, here unproven). The zone used is returned in the result
//! so callers can display, audit, or widen it. `NoWin` is similarly "no win
//! found within this probe's horizon", never a refutation. Analysis display
//! must label probe verdicts distinctly from proven VCF wins.

use crate::forcing::{
    self, threat_table_scored, wide_partner_cells, ForcingWin, Outcome, SolverBoard, MAX_WL,
};
use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::hex::hex_distance as hex_dist;
use hexo_engine::types::{Coord, Player, WIN_AXES};
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct ProbeConfig {
    /// Top-K builder cells forming the quiet-turn candidate pool.
    pub builder_pool: usize,
    /// Cap on candidate quiet turns (pairs, best score-sum first).
    pub max_quiet_turns: usize,
    /// How many latent-VCF PV cells seed the relevance zone (the builder
    /// contributions below are additive, never truncated by this).
    pub pv_zone_cells: usize,
    /// How many of each side's top builder cells join the zone (defender
    /// counter-play + attacker denial).
    pub zone_builders: usize,
    /// Proof reuse (levels 0 and 1). Disable to brute-force every defense
    /// pair — the differential oracle for the reuse logic.
    pub proof_reuse: bool,
    /// Budget for the step-3 latent-VCF (threat) check.
    pub threat_depth: u8,
    pub threat_budget: u64,
    /// Budget for step-1 direct VCF and each step-4 verification re-solve.
    pub verify_depth: u8,
    pub verify_budget: u64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        ProbeConfig {
            builder_pool: 16,
            max_quiet_turns: 24,
            pv_zone_cells: 14,
            zone_builders: 4,
            proof_reuse: true,
            threat_depth: 12,
            threat_budget: 200_000,
            verify_depth: 12,
            verify_budget: 100_000,
        }
    }
}

/// A quiet turn the bounded defender model could not refute.
#[derive(Debug)]
pub struct QuietWin {
    pub quiet_turn: [Coord; 2],
    /// The latent VCF that materializes if the defender passes (its PV seeds
    /// the relevance zone).
    pub threat_win: ForcingWin,
    /// Every defender pair verified to lose (re-solved or covered by proof
    /// reuse): `C(|zone|, 2)` zone pairs plus, when a pass cell exists, the
    /// `|zone|` anchor pairs `(zone cell, pass)`. Always > 0 — a candidate
    /// with nothing to verify is skipped, never claimed.
    pub defenses_checked: usize,
    /// Pairs that actually ran a verification solve (anchors + uncovered).
    pub defenses_solved: usize,
    /// Pairs dismissed by the pass-case proof (level-0 reuse).
    pub defenses_skipped_by_proof: usize,
    /// Pairs dismissed by an anchor `(d, pass)` proof covering the partner
    /// cell (level-1 reuse).
    pub defenses_covered_by_anchor: usize,
    /// The relevance zone (the soundness assumption, exposed for audit).
    pub zone: Vec<Coord>,
}

/// Instrumentation: a quiet turn that survived the pass-check but was refuted
/// by a concrete zone defense — the data that tells us how tight the defender
/// model is in practice.
#[derive(Debug)]
pub struct NearMiss {
    pub quiet_turn: [Coord; 2],
    pub refuting_defense: [Coord; 2],
}

#[derive(Debug)]
pub enum ProbeOutcome {
    /// No tempo loss needed: a direct (wide) VCF win.
    VcfWin(ForcingWin),
    /// Win against the bounded defender model after one quiet turn.
    QuietWin(QuietWin),
    /// No win found within the probe's horizon (NOT a refutation).
    NoWin {
        candidates_tried: usize,
        threat_survivors: usize,
        near_misses: Vec<NearMiss>,
    },
}

/// Probe-entry validation mirroring `forcing::solve_ex`'s config and
/// coordinate-spread guards: the probe calls scan helpers directly (which only
/// `debug_assert` the config), so it needs the solver's `wl` bounds itself —
/// including the lower bound 4, below which the threat band is empty and the
/// whole threat machinery is blind. The spread margin covers the boards the
/// probe builds later: quiet/zone/pass cells stay within `placement_radius` of
/// existing stones, and latent-VCF PV cells can drift outward by up to
/// `wl - 1` per attacker turn over `threat_depth` turns.
fn probe_input_ok(stones: &[(Coord, Player)], config: &GameConfig, cfg: &ProbeConfig) -> bool {
    if !(4..=MAX_WL).contains(&(config.win_length as usize)) || config.placement_radius < 1 {
        return false;
    }
    const MAX_COORD: i32 = 1 << 30;
    let margin = config.placement_radius as i64
        + config.win_length as i64 * (cfg.threat_depth as i64 + 2);
    let mut it = stones.iter();
    let Some(&(c0, _)) = it.next() else { return true };
    let (mut lo, mut hi) = (c0, c0);
    for &(c, _) in it {
        lo = (lo.0.min(c.0), lo.1.min(c.1));
        hi = (hi.0.max(c.0), hi.1.max(c.1));
    }
    if lo.0 < -MAX_COORD || lo.1 < -MAX_COORD || hi.0 > MAX_COORD || hi.1 > MAX_COORD {
        return false;
    }
    let (span_q, span_r) = (
        hi.0 as i64 - lo.0 as i64 + 2 * margin + 6 * forcing::GRID_PAD as i64,
        hi.1 as i64 - lo.1 as i64 + 2 * margin + 6 * forcing::GRID_PAD as i64,
    );
    span_q.saturating_mul(span_r) <= forcing::MAX_GRID_CELLS
}

/// Build a `SolverBoard` the same way `forcing::solve_ex` does. Callers must
/// have passed `probe_input_ok` (spread guards live there).
fn board_of(stones: &[(Coord, Player)], config: &GameConfig) -> SolverBoard {
    let mut board = SolverBoard::new();
    if let Some(&(c0, _)) = stones.first() {
        let (mut lo, mut hi) = (c0, c0);
        for &(c, _) in stones {
            lo = (lo.0.min(c.0), lo.1.min(c.1));
            hi = (hi.0.max(c.0), hi.1.max(c.1));
        }
        board.reserve(lo, hi);
    }
    if config.placement_radius < config.win_length as i32 - 1 {
        board.enable_reach(config.placement_radius);
    }
    for &(c, p) in stones {
        board.place(c, p);
    }
    board
}

/// A legal "pass" cell: on the board's periphery (exactly `placement_radius`
/// from an existing stone, so always legal), as far from the attacker's stones
/// as the stone set allows, and not colliding with `avoid`. Models the
/// defender reply "one/both stones spent irrelevantly" without leaving the
/// legal-move set.
fn pass_cell(stones: &[(Coord, Player)], attacker: Player, config: &GameConfig, avoid: &[Coord]) -> Option<Coord> {
    let r = config.placement_radius;
    let atk: Vec<Coord> = stones.iter().filter(|&&(_, p)| p == attacker).map(|&(c, _)| c).collect();
    let occupied: Vec<Coord> = stones.iter().map(|&(c, _)| c).collect();
    let mut best: Option<(i32, Coord)> = None;
    for &(s, _) in stones {
        for d in [(r, 0), (-r, 0), (0, r), (0, -r), (r, -r), (-r, r)] {
            let c = (s.0 + d.0, s.1 + d.1);
            if occupied.contains(&c) || avoid.contains(&c) {
                continue;
            }
            let dist = atk.iter().map(|&a| hex_dist(c, a)).min().unwrap_or(i32::MAX);
            if best.map(|(bd, bc)| (dist, c) > (bd, bc)).unwrap_or(true) {
                best = Some((dist, c));
            }
        }
    }
    best.map(|(_, c)| c)
}

/// Defender relevance zone on the post-quiet-turn position: latent-VCF PV
/// cells first (direct blocks, the strongest replies; `cfg.pv_zone_cells`
/// bounds this portion only), then ALWAYS both sides' top builders
/// (`cfg.zone_builders` each: defender counter-play, attacker denial) — those
/// are never truncated away, or a long PV would silently reduce the defender
/// model to pure blocking. Deduped in that priority order. The pass cell is
/// returned SEPARATELY (`None` when no legal periphery cell exists — callers
/// must not infer it from the zone).
fn defender_zone(
    stones: &[(Coord, Player)],
    threat_win: &ForcingWin,
    attacker: Player,
    defender: Player,
    config: &GameConfig,
    cfg: &ProbeConfig,
) -> (Vec<Coord>, Option<Coord>) {
    let board = board_of(stones, config);
    let occupied: Vec<Coord> = stones.iter().map(|&(c, _)| c).collect();
    let mut zone: Vec<Coord> = Vec::new();
    let push = |c: Coord, zone: &mut Vec<Coord>| {
        if !occupied.contains(&c) && !zone.contains(&c) {
            zone.push(c);
        }
    };
    for &c in threat_win.pv.iter().take(cfg.pv_zone_cells) {
        push(c, &mut zone);
    }
    let (wl, r) = (config.win_length, config.placement_radius);
    let (_, dfn_scores) = threat_table_scored(&board, defender, wl, r);
    for (c, _, _) in wide_partner_cells(&board, defender, wl, r, &dfn_scores)
        .into_iter()
        .take(cfg.zone_builders)
    {
        push(c, &mut zone);
    }
    let (_, atk_scores) = threat_table_scored(&board, attacker, wl, r);
    for (c, _, _) in wide_partner_cells(&board, attacker, wl, r, &atk_scores)
        .into_iter()
        .take(cfg.zone_builders)
    {
        push(c, &mut zone);
    }
    let pass = pass_cell(stones, attacker, config, &zone);
    (zone, pass)
}

/// The deep-node half of the reuse condition: could a reply cell combine with
/// existing defender stones AND future proof cover placements (which land only
/// on support cells) into a defender completion at some node of the reused
/// proof? Counts every empty support cell in the window as a potential future
/// defender stone (over-rejection, safe). Windows holding an attacker stone on
/// `board_d` are excluded — that is sound, NOT a shortcut: stones only ever
/// accumulate during the proof, so a window blocked by an attacker stone now
/// is blocked at every node of the reused proof and can never become a
/// defender completion.
fn defender_window_risk(
    board_d: &SolverBoard,
    support: &[Coord],
    reply: &[Coord],
    attacker: Player,
    wl: u8,
) -> bool {
    let l = wl as i32;
    for &cell in reply {
        for &(dq, dr) in WIN_AXES.iter() {
            for k in 0..l {
                // window cells: cell + (o - k) * axis for o in 0..l
                let mut dfn = 0i32;
                let mut support_empties = 0i32;
                let mut blocked = false;
                for o in 0..l {
                    let c = (cell.0 + (o - k) * dq, cell.1 + (o - k) * dr);
                    match board_d.get(c) {
                        Some(p) if p == attacker => {
                            blocked = true;
                            break;
                        }
                        Some(_) => dfn += 1,
                        None => {
                            if support.binary_search(&c).is_ok() {
                                support_empties += 1;
                            }
                        }
                    }
                }
                // `l - 2` is LOAD-BEARING and coupled to the solver: a window
                // with wl-2 defender stones is a 2-gap completion, exactly what
                // def_within's "defender wins first" check (forcing.rs,
                // `dfn_comps.any(len <= 2)`) treats as a defender win. If that
                // threshold ever changes, this margin must change with it.
                if !blocked && dfn + support_empties >= l - 2 {
                    return true;
                }
            }
        }
    }
    false
}

/// Quiet-first-turn probe. See the module docs for semantics and the
/// soundness contract. Requires the attacker at a turn start
/// (`moves_remaining_this_turn() == 2`), a valid config, and a sane coordinate
/// spread; anything else returns `NoWin{0,..}`.
pub fn probe(game: &GameState, cfg: &ProbeConfig) -> ProbeOutcome {
    let no_win = |tried, survivors, misses| ProbeOutcome::NoWin {
        candidates_tried: tried,
        threat_survivors: survivors,
        near_misses: misses,
    };
    let Some(attacker) = game.current_player() else {
        return no_win(0, 0, Vec::new());
    };
    if game.moves_remaining_this_turn() != 2 || game.is_terminal() {
        return no_win(0, 0, Vec::new());
    }
    let defender = attacker.opponent();
    let config = *game.config();
    let base: Vec<(Coord, Player)> = game.stones().iter().map(|(&c, &p)| (c, p)).collect();
    if !probe_input_ok(&base, &config, cfg) {
        return no_win(0, 0, Vec::new());
    }
    // 1) Direct wide VCF: no tempo loss needed.
    if let Outcome::Win(w) = forcing::solve_wide(game, cfg.verify_depth, cfg.verify_budget) {
        return ProbeOutcome::VcfWin(w);
    }
    // Level-1 anchor reuse is only sound in the wide-radius regime: in the
    // tight regime a proof move could be legal solely via the pass stone.
    let (wl, r) = (config.win_length, config.placement_radius);
    let anchors_sound = r >= wl as i32 - 1;

    // 2) Quiet-turn candidates: pairs of top builders, best score-sum first.
    let board = board_of(&base, &config);
    let (_, scores) = threat_table_scored(&board, attacker, wl, r);
    let pool: Vec<(Coord, u32)> = wide_partner_cells(&board, attacker, wl, r, &scores)
        .into_iter()
        .take(cfg.builder_pool)
        .map(|(c, s, _)| (c, s))
        .collect();
    let mut pairs: Vec<(u32, Coord, Coord)> = Vec::new();
    for (i, &(a, sa)) in pool.iter().enumerate() {
        for &(b, sb) in pool.iter().skip(i + 1) {
            pairs.push((sa + sb, a.min(b), a.max(b)));
        }
    }
    pairs.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    pairs.truncate(cfg.max_quiet_turns);

    let mut threat_survivors = 0usize;
    let mut near_misses: Vec<NearMiss> = Vec::new();
    let tried = pairs.len();
    for &(_, a, b) in &pairs {
        let mut after_q = base.clone();
        after_q.push((a, attacker));
        after_q.push((b, attacker));
        // 3) Necessary condition: latent VCF if the defender passes, with a
        // budget ladder (dedup keeps tiny configured budgets from re-running
        // the identical solve). The check solves the attacker's fresh
        // 2-placement turn directly, capturing proof support for step 4.
        let after_q_game = GameState::from_state(&after_q, defender, 2, config);
        if after_q_game.is_terminal() {
            continue; // a quiet turn never wins on the spot; direct wins were step 1's job
        }
        let atk_game = GameState::from_state(&after_q, attacker, 2, config);
        let mut ladder = vec![
            (cfg.threat_budget / 40).max(1),
            (cfg.threat_budget / 5).max(1),
            cfg.threat_budget.max(1),
        ];
        ladder.dedup();
        let mut threat = None;
        for budget in ladder {
            match forcing::solve_wide_with_support(&atk_game, cfg.threat_depth, budget) {
                (Outcome::Win(w), sup) => {
                    threat = Some((w, sup));
                    break;
                }
                (Outcome::No, _) => break,         // proven: no budget changes a No
                (Outcome::BudgetExceeded, _) => {} // ambiguous: escalate
            }
        }
        let Some((tw, support)) = threat else { continue };
        threat_survivors += 1;
        let support = if cfg.proof_reuse { support } else { None };

        // 4) Bounded refutation: anchors first (their proofs power level-1
        // reuse), then every zone pair.
        let (zone, pass) = defender_zone(&after_q, &tw, attacker, defender, &config, cfg);
        let n_anchor_pairs = if pass.is_some() { zone.len() } else { 0 };
        let total = n_anchor_pairs + zone.len() * zone.len().saturating_sub(1) / 2;
        if total == 0 {
            continue; // nothing to verify — never claim a win over zero defenses
        }
        // The reuse condition, parameterized by whose proof is being reused.
        let reuse_covers = |sup: &[Coord], reply: &[Coord], board_d: &SolverBoard| -> bool {
            reply.iter().all(|c| sup.binary_search(c).is_err())
                && !defender_window_risk(board_d, sup, reply, attacker, wl)
        };
        let mut refuted: Option<[Coord; 2]> = None;
        let (mut solved, mut skipped, mut covered) = (0usize, 0usize, 0usize);
        // Phase A — anchors `(d, pass)`: legal zone defenses that need
        // verifying regardless; verified with support capture so their proofs
        // can cover phase-B pairs. An anchor whose support leaks onto the pass
        // cell covers nothing; poisoned captures (None) cover nothing.
        let mut anchors: Vec<(Coord, Rc<Vec<Coord>>)> = Vec::new();
        if let Some(p) = pass {
            'anchors: for &d in zone.iter() {
                let mut after_d = after_q.clone();
                after_d.push((d, defender));
                after_d.push((p, defender));
                let board_d = board_of(&after_d, &config);
                if forcing::has_full_line(&board_d, defender, wl, r) {
                    refuted = Some([d, p]);
                    break 'anchors;
                }
                solved += 1;
                let (out, sup) =
                    forcing::solve_wide_with_support(&after_d_game_of(&after_d, attacker, config), cfg.verify_depth, cfg.verify_budget);
                match out {
                    Outcome::Win(_) => {
                        if anchors_sound
                            && cfg.proof_reuse
                            && let Some(sup) = sup
                            && sup.binary_search(&p).is_err()
                        {
                            anchors.push((d, sup));
                        }
                    }
                    _ => {
                        refuted = Some([d, p]);
                        break 'anchors;
                    }
                }
            }
        }
        // Phase B — the zone pairs, cheapest disposal first: level-0
        // (pass-case proof), level-1 (either cell's anchor proof), else solve.
        if refuted.is_none() {
            'defense: for (i, &d1) in zone.iter().enumerate() {
                for &d2 in zone.iter().skip(i + 1) {
                    let mut after_d = after_q.clone();
                    after_d.push((d1, defender));
                    after_d.push((d2, defender));
                    let board_d = board_of(&after_d, &config);
                    if forcing::has_full_line(&board_d, defender, wl, r) {
                        refuted = Some([d1, d2]); // the reply itself completes a defender win
                        break 'defense;
                    }
                    if let Some(sup) = &support
                        && reuse_covers(sup, &[d1, d2], &board_d)
                    {
                        skipped += 1;
                        continue;
                    }
                    let by_anchor = anchors.iter().any(|(ac, sup)| {
                        let partner = if *ac == d1 { d2 } else if *ac == d2 { d1 } else { return false };
                        reuse_covers(sup, &[partner], &board_d)
                    });
                    if by_anchor {
                        covered += 1;
                        continue;
                    }
                    solved += 1;
                    match forcing::solve_wide(&after_d_game_of(&after_d, attacker, config), cfg.verify_depth, cfg.verify_budget) {
                        Outcome::Win(_) => {}
                        // No AND BudgetExceeded both refute: never claim a win
                        // we could not re-prove.
                        _ => {
                            refuted = Some([d1, d2]);
                            break 'defense;
                        }
                    }
                }
            }
        }
        match refuted {
            None => {
                debug_assert_eq!(solved + skipped + covered, total);
                return ProbeOutcome::QuietWin(QuietWin {
                    quiet_turn: [a, b],
                    threat_win: tw,
                    defenses_checked: solved + skipped + covered,
                    defenses_solved: solved,
                    defenses_skipped_by_proof: skipped,
                    defenses_covered_by_anchor: covered,
                    zone,
                });
            }
            Some(d) => {
                if near_misses.len() < 8 {
                    near_misses.push(NearMiss { quiet_turn: [a, b], refuting_defense: d });
                }
            }
        }
    }
    no_win(tried, threat_survivors, near_misses)
}

fn after_d_game_of(stones: &[(Coord, Player)], attacker: Player, config: GameConfig) -> GameState {
    GameState::from_state(stones, attacker, 2, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet_builder_game() -> GameState {
        // The 2026-07-10 wide-solver fixture: a direct wide VCF exists, so the
        // probe must short-circuit to VcfWin without spending a tempo.
        let mut stones: Vec<(Coord, Player)> = [(0, 0), (1, 0), (2, 0)]
            .into_iter().map(|c| (c, Player::P1)).collect();
        stones.extend([(0, -8), (-1, -15), (7, -22), (3, -26)]
            .into_iter().map(|c| (c, Player::P2)));
        GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO)
    }

    fn ymf9zzt_game() -> GameState {
        // HDS ymf9zzt "is forced win?": P1's three blocked on BOTH row ends —
        // provably no VCF at any depth; the quiet-first-turn question is open.
        let mut stones: Vec<(Coord, Player)> = [(0, 0), (-1, 0), (1, 0)]
            .into_iter().map(|c| (c, Player::P1)).collect();
        stones.extend([(-2, 0), (2, 0), (-10, 7), (9, -8)]
            .into_iter().map(|c| (c, Player::P2)));
        GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO)
    }

    #[test]
    fn probe_short_circuits_to_direct_vcf_win() {
        match probe(&quiet_builder_game(), &ProbeConfig::default()) {
            ProbeOutcome::VcfWin(w) => assert_eq!(w.depth, 6),
            other => panic!("expected VcfWin, got {other:?}"),
        }
    }

    /// The headline behaviour: ymf9zzt has no VCF, but the probe proves a
    /// quiet-turn win against the bounded defender model — with consistent
    /// accounting across solves, level-0 skips, and level-1 anchor coverage.
    #[test]
    fn probe_finds_ymf9zzt_quiet_win_with_consistent_accounting() {
        match probe(&ymf9zzt_game(), &ProbeConfig::default()) {
            ProbeOutcome::QuietWin(w) => {
                assert_eq!(w.quiet_turn, [(-1, 1), (0, 1)], "known winning quiet turn");
                assert_eq!(w.threat_win.depth, 4, "latent VCF is a win-in-4");
                let n = w.zone.len();
                assert!(n >= 2, "zone must offer real defenses, got {n}");
                // n + C(n,2) holds only when a pass cell exists (its absence
                // would drop the n anchor pairs) — so this also pins that the
                // open ymf9zzt board yields a pass cell.
                assert_eq!(
                    w.defenses_checked,
                    n + n * (n - 1) / 2,
                    "every anchor and zone pair must be accounted for"
                );
                assert!(w.defenses_solved > 0, "anchors always solve");
                assert!(w.defenses_covered_by_anchor > 0, "anchor proofs must cover pairs here");
                // The quiet turn and the latent VCF must be real: replay the
                // quiet turn (legal on the base board), then the threat PV on
                // the pass-case position — it must end in the attacker's win.
                let mut replay = ymf9zzt_game();
                for &c in &w.quiet_turn {
                    replay.apply_move(c).expect("quiet turn must be legal");
                }
                let after_q: Vec<(Coord, Player)> =
                    replay.stones().iter().map(|(&c, &p)| (c, p)).collect();
                let mut threat_replay =
                    GameState::from_state(&after_q, Player::P1, 2, *replay.config());
                for &c in &w.threat_win.pv {
                    threat_replay.apply_move(c).expect("threat PV must be a legal line");
                }
                assert!(threat_replay.is_terminal(), "threat PV must end the game");
                assert_eq!(threat_replay.winner(), Some(Player::P1));
            }
            other => panic!("expected QuietWin, got {other:?}"),
        }
    }

    /// Empirical weight for the zone hypothesis on the headline fixture: the
    /// QuietWin must survive a substantially larger defender model (more PV
    /// cells, twice the builders) with the same quiet turn.
    #[test]
    fn probe_ymf9zzt_win_survives_widened_zone() {
        let cfg = ProbeConfig {
            pv_zone_cells: 20,
            zone_builders: 8,
            ..ProbeConfig::default()
        };
        match probe(&ymf9zzt_game(), &cfg) {
            ProbeOutcome::QuietWin(w) => {
                assert_eq!(w.quiet_turn, [(-1, 1), (0, 1)]);
                assert!(w.zone.len() > 20, "widened zone must actually be larger");
            }
            other => panic!("QuietWin must survive a widened zone, got {other:?}"),
        }
    }

    /// Differential oracle for the reuse logic: brute-forcing every defense
    /// (proof_reuse = false) must reach the same verdict, from the same quiet
    /// turn, as the reuse-pruned run — and the pruned run must actually prune.
    #[test]
    fn probe_reuse_matches_brute_force() {
        let cfg_reuse = ProbeConfig::default();
        let cfg_brute = ProbeConfig { proof_reuse: false, ..ProbeConfig::default() };
        let (r1, r2) = (probe(&ymf9zzt_game(), &cfg_reuse), probe(&ymf9zzt_game(), &cfg_brute));
        match (r1, r2) {
            (ProbeOutcome::QuietWin(a), ProbeOutcome::QuietWin(b)) => {
                assert_eq!(a.quiet_turn, b.quiet_turn);
                assert_eq!(a.defenses_checked, b.defenses_checked);
                assert_eq!(b.defenses_solved, b.defenses_checked, "brute force solves everything");
                assert!(
                    a.defenses_solved < b.defenses_solved,
                    "reuse must prune: {} vs {}", a.defenses_solved, b.defenses_solved
                );
            }
            (a, b) => panic!("verdicts diverged: {a:?} vs {b:?}"),
        }
    }

    /// Refutation path: with a verification budget too small to re-prove
    /// anything, every candidate that survives the threat check is refuted at
    /// its first anchor defense (BudgetExceeded counts as refuted — the probe
    /// never over-claims), and the refuting defense is reported.
    #[test]
    fn probe_starved_verification_reports_near_misses_not_wins() {
        let cfg = ProbeConfig { verify_budget: 10, ..ProbeConfig::default() };
        match probe(&ymf9zzt_game(), &cfg) {
            ProbeOutcome::NoWin { threat_survivors, near_misses, .. } => {
                assert!(threat_survivors > 0, "threat checks still pass at full budget");
                assert!(!near_misses.is_empty(), "each survivor must record its refutation");
                for nm in &near_misses {
                    assert_ne!(nm.refuting_defense[0], nm.refuting_defense[1]);
                }
            }
            other => panic!("expected NoWin under starved verification, got {other:?}"),
        }
    }

    /// Config guards: values the solver would reject with an honest
    /// BudgetExceeded must not panic the probe's direct scan-helper calls.
    #[test]
    fn probe_rejects_invalid_configs_gracefully() {
        let stones: Vec<(Coord, Player)> = vec![((0, 0), Player::P1), ((5, 5), Player::P2)];
        for config in [
            GameConfig { win_length: 17, placement_radius: 8, max_moves: 200 }, // > MAX_WL
            GameConfig { win_length: 6, placement_radius: 0, max_moves: 200 },  // degenerate radius
        ] {
            let game = GameState::from_state(&stones, Player::P1, 2, config);
            match probe(&game, &ProbeConfig::default()) {
                ProbeOutcome::NoWin { candidates_tried: 0, .. } => {}
                other => panic!("expected guarded NoWin for {config:?}, got {other:?}"),
            }
        }
    }

    /// Zone construction: PV cells lead (capped), both sides' builders are
    /// never truncated away by a long PV, and the pass cell is returned
    /// separately, far from the attack.
    #[test]
    fn probe_zone_keeps_builders_and_separates_pass() {
        let stones: Vec<(Coord, Player)> = vec![
            ((0, 0), Player::P1), ((0, 1), Player::P1),
            ((5, 5), Player::P2),
        ];
        let long_pv: Vec<Coord> = (2..12).map(|k| (0, k)).collect();
        let tw = ForcingWin { depth: 4, first_move: (0, 2), pv: long_pv };
        let cfg = ProbeConfig { pv_zone_cells: 4, zone_builders: 2, ..ProbeConfig::default() };
        let (zone, pass) = defender_zone(&stones, &tw, Player::P1, Player::P2, &GameConfig::FULL_HEXO, &cfg);
        assert_eq!(&zone[..4], &[(0, 2), (0, 3), (0, 4), (0, 5)], "PV cells lead, capped at 4");
        assert!(
            zone.iter().any(|&c| hex_dist(c, (5, 5)) <= 5),
            "defender builders (near (5,5)) must survive a long PV: {zone:?}"
        );
        let pass = pass.expect("open board must offer a pass cell");
        assert!(!zone.contains(&pass), "pass is not a zone member");
        let dist_to_attacker = [(0, 0), (0, 1)].iter().map(|&a| hex_dist(pass, a)).min().unwrap();
        assert!(dist_to_attacker >= GameConfig::FULL_HEXO.placement_radius,
                "pass cell must be far from the attack, got {pass:?} at {dist_to_attacker}");
    }

    /// Full instrumentation dump for manual runs.
    #[test]
    #[ignore]
    fn probe_ymf9zzt_report() {
        let t = std::time::Instant::now();
        let out = probe(&ymf9zzt_game(), &ProbeConfig::default());
        eprintln!("ymf9zzt probe ({:.1}s): {out:#?}", t.elapsed().as_secs_f64());
    }
}
