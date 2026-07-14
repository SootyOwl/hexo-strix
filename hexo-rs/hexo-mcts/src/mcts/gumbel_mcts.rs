use rustc_hash::FxHashMap as HashMap;

use hexo_engine::{Coord, GameState};
use hexo_engine::types::WIN_AXES;
use rand::Rng;

use super::MCTSConfig;
use super::halving::{compute_improved_policy, gumbel_top_k, sequential_halving};
use super::node::MCTSNode;
use super::scoring::{QContext, sigma};
use super::simulate::{
    apply_virtual_loss, complete_simulation, revert_virtual_loss, simulate_select,
};
use super::forcing;

/// Depth cap (in attacker *turns*, not placements) for the bounded VCF
/// forcing-win shortcut run at every mr=2 root. Set high on purpose: a forced
/// win is the case worth paying for (decisive move + clean data + skips MCTS),
/// and iterative deepening finds shallow wins cheaply — `depth_cap` only sets
/// how deep we're *willing* to look; `SELF_PLAY_NODE_BUDGET` is the hard ceiling
/// on per-position effort (branchy no-win positions). The Task-11 bench sweep
/// (synthetic wl6/r2 corpus) showed depth drives the tail, but that corpus
/// overstates the branchy-no-win tax vs NN-guided play — so the REAL arbiter is
/// the pre-live self-play A/B (throughput on vs off). Dial via `node_budget`
/// (the effective tail lever at high depth) if the real tail bites.
pub const SELF_PLAY_DEPTH_CAP: u8 = 6;
/// Per-position node-budget cap for the same shortcut — the hard ceiling on
/// effort per root, independent of win/no-win. This, not `depth_cap`, is the
/// lever that bounds worst-case latency on branchy no-win positions once the
/// depth cap is high enough for the budget to bind. Provisional pending the
/// pre-live A/B (see `SELF_PLAY_DEPTH_CAP`).
///
/// `pub`: shared with `batched::batched_gumbel_mcts` (identical forced-win
/// shortcut applied per-game before the halving/eval rounds) and with
/// `bin/bench_depth2_shortcut`, which measures `forcing::solve` at exactly
/// this self-play budget (Task 11,
/// docs/superpowers/plans/2026-07-03-forcing-solver-rust-port.md).
pub const SELF_PLAY_NODE_BUDGET: u64 = 2_000;

/// Result of a Gumbel MCTS search.
pub struct MCTSResult {
    /// The chosen action (q, r).
    pub action: Coord,
    /// Improved policy over all legal moves (same order as `coords`).
    pub improved_policy: Vec<f64>,
    /// Legal move coordinates (same order as `improved_policy`).
    pub coords: Vec<Coord>,
    /// Root visit counts per legal move (same order as `coords`).
    /// Unexpanded moves contribute 0. Sum equals total simulations spent at
    /// the root minus any that failed to reach a child (rare). Provides a
    /// soft alternative to `improved_policy` for inspection: visit counts
    /// reflect cumulative search effort and are smooth in a way the
    /// σ-amplified `improved_policy` is not.
    pub visit_counts: Vec<u32>,
    /// Per-child Q value at the root (same order as `coords`), from the root
    /// player's perspective. Visited children use their empirical Q; unvisited
    /// children use the `v_mix` fallback from `QContext` — i.e. the same value
    /// σ(Q) sees during sequential halving — so consumers can interpret the
    /// vector uniformly without branching on visit count. For the shortcut /
    /// early-exit paths the root has zero visited children, so every entry
    /// equals the root's network value estimate.
    pub per_child_q: Vec<f64>,
    /// Per-child network prior π(a) at the root (same order as `coords`).
    /// This is the softmax over the root policy logits restricted to legal
    /// moves — the same distribution the root expand step stores — i.e.
    /// *after* any root Dirichlet mixing but *before* Gumbel noise and
    /// σ(Q) sharpening.
    pub per_child_prior: Vec<f64>,
    /// Indices into `coords` of the m=`m_actions` Gumbel-Top-K survivors at
    /// the root (sorted by descending gumbel+logit score, as returned by
    /// `gumbel_top_k`). Length is `min(m_actions, coords.len())`. For the
    /// shortcut / early-exit paths this is the candidate set the shortcut
    /// scanned; the chosen action is guaranteed to lie inside it.
    pub candidate_indices: Vec<usize>,
    /// At mr=2 roots with `forced_candidate_capture_k > 0`, the top-K p₂
    /// coords (by conditional prior) for the chosen p₁'s post-state.
    /// Empty otherwise (mr=1 roots, capture disabled, or chosen p₁ led to
    /// terminal post-state).
    pub chosen_action_forced_candidates: Vec<Coord>,
}

/// Run Gumbel MCTS from a game state.
///
/// The `eval_fn` callback evaluates a batch of game states:
///   `eval_fn(states) -> (logits_per_state, values)`
/// where logits_per_state[i] maps action → logit for state i,
/// and values[i] is the scalar value estimate.
///
/// `forced_candidates` (E.4-inject, p₂ side): when `Some(coords)` and the
/// root is mr=1 (mid-turn), the supplied coordinates are guaranteed to appear
/// in the root's Gumbel-Top-K candidate set. They get fresh independent
/// Gumbel-noise draws so SH ranking treats them fairly, and they displace
/// the lowest-Gumbel-scored non-forced candidates when not already present.
/// Ignored at mr=2 roots — the mechanism is only meaningful for the p₂
/// search (the second placement of a HeXO turn).
///
/// # Errors
/// Returns `Err` if the game state is terminal.
pub fn gumbel_mcts<R, F>(
    game: &GameState,
    config: &MCTSConfig,
    rng: &mut R,
    forced_candidates: Option<&[Coord]>,
    eval_fn: &mut F,
) -> Result<MCTSResult, &'static str>
where
    R: Rng,
    F: FnMut(&[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>),
{
    if game.is_terminal() {
        return Err("Cannot run MCTS on terminal state");
    }

    let legal_moves = game.legal_moves();
    if legal_moves.is_empty() {
        return Err("No legal moves in non-terminal state");
    }

    // Step 1: Evaluate the root state — eval_fn returns raw logits, not priors.
    //
    // Under `dedup_skip`, the cache is consulted *only if* the runtime
    // skip flag is on. This lets the wall-clock prototype A/B both modes
    // from a single compiled binary.
    #[cfg(feature = "dedup_skip")]
    let (root_logits_owned, root_value): (HashMap<Coord, f64>, f64) = {
        if super::dedup_count::skip_enabled_tls() {
            let fp = super::dedup_count::position_fingerprint(game, None);
            if let Some(entry) = super::dedup_count::eval_cache_get_tls(fp) {
                (entry.policy_logits, entry.value)
            } else {
                let (logits_list, values) = eval_fn(&[game.clone()]);
                let lm = logits_list.into_iter().next().unwrap();
                let v = values[0];
                super::dedup_count::eval_cache_insert_tls(
                    fp,
                    super::dedup_count::CachedEval {
                        policy_logits: lm.clone(),
                        value: v,
                    },
                );
                (lm, v)
            }
        } else {
            let (logits_list, values) = eval_fn(&[game.clone()]);
            (logits_list.into_iter().next().unwrap(), values[0])
        }
    };
    #[cfg(feature = "dedup_skip")]
    let root_logits_map = &root_logits_owned;

    #[cfg(not(feature = "dedup_skip"))]
    let (logits_list, values) = eval_fn(&[game.clone()]);
    #[cfg(not(feature = "dedup_skip"))]
    let root_logits_map = &logits_list[0];
    #[cfg(not(feature = "dedup_skip"))]
    let root_value = values[0];

    // Build coords and logits arrays (sorted order matching legal_moves).
    // The depth-N forcing shortcut (Step 3.5, Phase B) never appends to
    // `coords`: the solver is radius-aware, so its winning move is always a
    // member of `legal_moves()` (defensively guarded there too).
    let coords: Vec<Coord> = legal_moves.clone();
    let mut logits: Vec<f64> = coords
        .iter()
        .map(|c| root_logits_map.get(c).copied().unwrap_or(0.0))
        .collect();

    // Root Dirichlet noise (AlphaZero-style exploration applied before
    // Gumbel-Top-k candidate sampling). Mixes the noise into prior
    // probability space and converts back to logits so that both
    // Gumbel-Top-k (which sees `logits`) and the root priors used by the
    // search (computed by softmax(logits) below) are consistent.
    // Disabled when `root_dirichlet_alpha == 0.0` — the legacy code path
    // is bit-equivalent in that case.
    if config.root_dirichlet_alpha > 0.0
        && config.root_dirichlet_fraction > 0.0
        && !logits.is_empty()
    {
        use rand_distr::{Distribution, Gamma};
        let n = logits.len();
        let gamma = Gamma::new(config.root_dirichlet_alpha, 1.0)
            .expect("root_dirichlet_alpha must be > 0 here");
        let mut gamma_samples: Vec<f64> = (0..n).map(|_| gamma.sample(rng)).collect();
        let gamma_sum: f64 = gamma_samples.iter().sum();
        if gamma_sum > 0.0 {
            for g in gamma_samples.iter_mut() {
                *g /= gamma_sum;
            }
            let priors = super::scoring::softmax(&logits);
            let frac = config.root_dirichlet_fraction.clamp(0.0, 1.0);
            for i in 0..n {
                let mixed = (1.0 - frac) * priors[i] + frac * gamma_samples[i];
                // Convert back to logit space; eps guards against log(0).
                logits[i] = (mixed + 1e-30).ln();
            }
        }
    }

    // Compute priors via softmax of raw logits
    let priors_vec = super::scoring::softmax(&logits);
    let root_priors: HashMap<Coord, f64> = coords
        .iter()
        .zip(priors_vec.iter())
        .map(|(&c, &p)| (c, p))
        .collect();

    // Step 2: Build root node
    let current_player = game
        .current_player()
        .expect("Non-terminal game should have current player");

    let mut root = MCTSNode::new(1.0, None, current_player);
    root.game_state = Some(game.clone());
    root.expand(root_priors, root_value);

    #[cfg(feature = "dedup_count")]
    {
        super::dedup_count::record_expansion_tls(None, game);
    }

    // Step 3: Gumbel top-k sampling
    let (mut candidate_indices, mut gumbel_samples) =
        gumbel_top_k(&logits, config.m_actions, rng, !config.disable_gumbel_noise);

    // Step 3.1 (E.4-inject consumer): forced candidate injection.
    //
    // When `forced_candidates` is supplied at a mr=1 root, guarantee that
    // each of those coords (translated to its index in `coords`/legal_moves)
    // appears in the final candidate set. If a forced coord is already
    // among the Gumbel-Top-K winners, no change. Otherwise we replace the
    // lowest-Gumbel-scored *non-forced* survivor with the forced index,
    // keeping the total candidate count at `m_actions`. Each newly-injected
    // index gets a fresh independent Gumbel(0,1) draw assigned in
    // `gumbel_samples` so SH treats it on equal footing with the
    // naturally-sampled candidates.
    //
    // Ignored at mr=2 roots: the mechanism's purpose is to plumb the p₂
    // candidate set learned during the p₁ search into the p₂ search, which
    // is the mr=1 root. Passing forced coords at a mr=2 root is silently
    // a no-op (documented in the function comment).
    if let Some(forced) = forced_candidates {
        if game.moves_remaining_this_turn() == 1 && !forced.is_empty() {
            use rand_distr::{Distribution, Uniform};
            let uniform = Uniform::new(1e-20, 1.0 - 1e-20).unwrap();
            let noise = !config.disable_gumbel_noise;
            let sample_gumbel = |rng: &mut R| -> f64 {
                if !noise {
                    return 0.0;
                }
                let u: f64 = uniform.sample(rng);
                -(-u.ln()).ln()
            };
            // Translate forced coords to indices in `coords`; silently
            // drop coords not present in `legal_moves` (they can't be
            // candidates).
            let mut forced_indices: Vec<usize> = Vec::new();
            for &c in forced {
                if let Some(idx) = coords.iter().position(|&x| x == c) {
                    if !forced_indices.contains(&idx) {
                        forced_indices.push(idx);
                    }
                }
            }
            // For each forced index NOT already in candidate_indices,
            // draw a fresh Gumbel and inject by displacing the lowest-
            // Gumbel-scored non-forced candidate.
            for &fi in &forced_indices {
                if candidate_indices.contains(&fi) {
                    continue;
                }
                // Find the lowest-scored non-forced candidate.
                // Score = gumbel_samples[idx] + logits[idx] (matches
                // gumbel_top_k's ordering).
                let mut worst_pos: Option<usize> = None;
                let mut worst_score = f64::INFINITY;
                for (pos, &idx) in candidate_indices.iter().enumerate() {
                    if forced_indices.contains(&idx) {
                        continue; // don't displace another forced index
                    }
                    let score = gumbel_samples[idx] + logits[idx];
                    if score < worst_score {
                        worst_score = score;
                        worst_pos = Some(pos);
                    }
                }
                let Some(pos) = worst_pos else { continue };
                // Assign a fresh Gumbel draw to the injected index.
                gumbel_samples[fi] = sample_gumbel(rng);
                candidate_indices[pos] = fi;
            }
        }
    }

    // Step 3.5: Forced-win shortcut.
    //
    // Phase A (depth-1): if any candidate makes the game terminal as a win
    // for the current player, return a one-hot improved policy on it and
    // skip sequential halving. Scoped to the Gumbel-top-K candidate set
    // (not all legal moves) to bound cost; uninformed priors will miss some
    // forced wins but the network's own learning concentrates priors on gap
    // squares over time.
    //
    // Phase B (depth-N, generalized VCF): HeXO turns are two placements. If
    // we're at the start of a turn (moves_remaining_this_turn() == 2), run
    // the bounded fully-forcing solver (`forcing::solve`) from the current
    // position. Unlike the old exhaustive depth-2 scan (which could only
    // ever see a forced win completed within THIS single turn, with no
    // opponent move interposed), the solver searches multiple turns deep
    // with the opponent's forced blocks interposed -- e.g. a "double-four"
    // fork that isn't decisive until 2-3 turns later. On a proven win,
    // return one-hot on the attacker's first placement. The solver is not
    // candidate-scoped (it reasons about the whole board's threat structure
    // directly), so the winning move may not already be a Gumbel-Top-K
    // candidate; `candidate_indices` is extended below if needed. The solver
    // is radius-aware, so the winning move IS always a `game.legal_moves()`
    // member (i.e. already a `coords` entry) -- defensively re-checked below
    // rather than assumed. Depth/node-budget bounded by `SELF_PLAY_DEPTH_CAP`
    // / `SELF_PLAY_NODE_BUDGET` to cap worst-case latency when no forcing win
    // exists.
    let me = game.current_player();
    let me_player = me.expect("non-terminal game guaranteed Some(current_player)");
    let stones_ref = game.stones();
    let max_dist = (game.config().win_length - 1) as i32;

    // Phase A pre-check: for m1 to be a depth-1 win, the (win_length)-window
    // containing m1 must already hold (win_length - 1) own stones; equivalently,
    // some axis through m1 has ≥ (win_length - 1) own stones within max_dist
    // cells. Skip the clone otherwise.
    let min_own_d1 = (game.config().win_length as usize).saturating_sub(1);
    for &idx in &candidate_indices {
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
        let mut test_game = game.clone();
        if test_game.apply_move(m1).is_ok() && test_game.is_terminal() {
            if test_game.winner() == me {
                let mut improved_policy = vec![0.0; coords.len()];
                improved_policy[idx] = 1.0;
                let mut visit_counts = vec![0u32; coords.len()];
                visit_counts[idx] = 1;
                let qctx = QContext::new(&root, &coords);
                let per_child_q: Vec<f64> =
                    coords.iter().map(|c| qctx.completed_q[c]).collect();
                let per_child_prior = priors_vec.clone();
                return Ok(MCTSResult {
                    action: m1,
                    improved_policy,
                    coords,
                    visit_counts,
                    per_child_q,
                    per_child_prior,
                    candidate_indices: candidate_indices.clone(),
                    // Depth-1 shortcut: chosen p₁ is terminal. The post-state
                    // has no legal p₂s to capture priors from.
                    chosen_action_forced_candidates: Vec::new(),
                });
            }
        }
    }

    // Change (A): fire at mr>=1, not only at start-of-turn (mr==2). At an
    // mr==1 root the solver models "attacker places its 1 remaining stone ->
    // defender full turn -> attacker full turns"; its `pv[0]` is then the
    // single completing placement, so the one-hot construction below is
    // already correct for mr==1.
    if !config.disable_forcing_solver && game.moves_remaining_this_turn() >= 1 {
        // 0 = sentinel -> compile-time default; any non-zero value overrides.
        let cap = if config.forcing_depth_cap == 0 {
            SELF_PLAY_DEPTH_CAP
        } else {
            config.forcing_depth_cap
        };
        let budget = if config.forcing_node_budget == 0 {
            SELF_PLAY_NODE_BUDGET
        } else {
            config.forcing_node_budget
        };
        // Wide generator: the tight hot×(hot∪block) generator provably misses
        // wins whose turn includes a quiet threat-BUILDING placement (the
        // quiet-builder miss, 2026-07-10). `solve_wide` is a strict superset —
        // every tight Win is still found and `No` stays sound — at ~1.5-1.7×
        // mean per-call cost (corpus A/B), negligible at one call per
        // placement. On a Win, `pv` may pair a hot cell with a quiet builder;
        // the pair is order-invariant within the turn, so the two-hot / Q=+1
        // construction below is unchanged.
        if let forcing::Outcome::Win(w) = forcing::solve_wide(game, cap, budget) {
            let first_move = w.first_move;
            // Defensive guard: `forcing::solve` is radius-aware, so
            // `first_move` should always be a member of `game.legal_moves()`.
            // Belt-and-suspenders against a future solver regression: never
            // return an action that isn't legal -- fall through to normal
            // search instead.
            if game.legal_moves_set().contains(&first_move) {
                // `first_move` is guaranteed to already be in `coords` (==
                // `legal_moves()`) by the guard above, so this lookup always
                // succeeds.
                let idx = coords
                    .iter()
                    .position(|&c| c == first_move)
                    .expect("first_move verified legal, so it is in coords");

                // Change (B): at an mr==2 root the winning turn is the
                // attacker's ORDER-INVARIANT placement pair {pv[0], pv[1]}, so
                // the correct policy target is TWO-HOT (0.5 on each). Fall back
                // to one-hot on pv[0] when either guard fails:
                //   1. pv.len() < 2  -- a single-stone completion (e.g. a win
                //      not seen by the candidate-scoped depth-1 precheck) has
                //      pv.len() == 1.
                //   2. pv[1] absent from the root legal moves -- in the tight-
                //      radius regime pv[1] may only become legal after pv[0].
                // At mr==1 the winning turn is a single placement, so the pair
                // logic is intentionally not applied (one-hot on pv[0]).
                let pv1_idx = if game.moves_remaining_this_turn() == 2 && w.pv.len() >= 2 {
                    coords.iter().position(|&c| c == w.pv[1])
                } else {
                    None
                };

                let mut improved_policy = vec![0.0; coords.len()];
                let mut visit_counts = vec![0u32; coords.len()];
                let mut candidate_indices = candidate_indices.clone();
                // Preserve the `MCTSResult::candidate_indices` invariant ("the
                // chosen action is guaranteed to lie inside it") relied on by
                // callers such as `acting::exploration_weights`'s Q-margin gate.
                if !candidate_indices.contains(&idx) {
                    candidate_indices.push(idx);
                }
                match pv1_idx {
                    Some(idx2) => {
                        // Two-hot on the order-invariant attacker pair.
                        improved_policy[idx] = 0.5;
                        improved_policy[idx2] = 0.5;
                        visit_counts[idx] = 1;
                        visit_counts[idx2] = 1;
                        if !candidate_indices.contains(&idx2) {
                            candidate_indices.push(idx2);
                        }
                    }
                    None => {
                        improved_policy[idx] = 1.0;
                        visit_counts[idx] = 1;
                    }
                }

                let qctx = QContext::new(&root, &coords);
                let per_child_q: Vec<f64> =
                    coords.iter().map(|c| qctx.completed_q[c]).collect();
                let per_child_prior = priors_vec.clone();
                return Ok(MCTSResult {
                    // `action` stays pv[0] in both cases (greedy determinism).
                    action: first_move,
                    improved_policy,
                    coords,
                    visit_counts,
                    per_child_q,
                    per_child_prior,
                    candidate_indices,
                    // The forced win may span multiple turns; unlike the old
                    // single-turn shortcut, there's no well-defined "post-p₁ p₂
                    // set" to seed the next search with. Leaving empty.
                    chosen_action_forced_candidates: Vec::new(),
                });
            }
        }
    }

    // Step 3.6 (E.4-inject capture): conditional p₂ priors per Gumbel-Top-K
    // candidate at mr=2 roots.
    //
    // When `forced_candidate_capture_k > 0` and we are at start-of-turn
    // (mr=2), we re-embed the post-p₁ state for every candidate p₁ to read
    // off the network's conditional prior over the legal p₂s. The chosen
    // p₁'s top-K p₂s are returned in MCTSResult so the caller can pass them
    // as `forced_candidates` to the next gumbel_mcts call (the p₂ search).
    //
    // The SH elimination and final argmax are NOT biased by this — only
    // the *next* search benefits from the captured set. When capture is
    // disabled (k=0) or we're at mr=1, no extra evaluation runs and
    // `captured_p2_priors` stays `None`, leaving the existing search path
    // bit-identical.
    //
    // `captured_p2_priors[pos]`: for the candidate at position `pos`
    // within `candidate_indices`, either `Some(Vec<(Coord, f64)>)` of
    // (p₂ coord, conditional prior) pairs over its legal p₂s, or `None`
    // when the candidate's post-p₁ state was terminal (no p₂s exist).
    let captured_p2_priors: Option<Vec<Option<Vec<(Coord, f64)>>>> =
        if config.forced_candidate_capture_k > 0 && game.moves_remaining_this_turn() == 2 {
            let mut per_candidate: Vec<Option<Vec<(Coord, f64)>>> =
                vec![None; candidate_indices.len()];
            // Collect non-terminal post-p₁ states for batched evaluation;
            // remember each one's slot in `candidate_indices` so we can
            // re-align after the eval returns.
            let mut pending_pos: Vec<usize> = Vec::new();
            let mut pending_states: Vec<GameState> = Vec::new();
            for (pos, &idx) in candidate_indices.iter().enumerate() {
                let m1 = coords[idx];
                let mut g = game.clone();
                if g.apply_move(m1).is_err() {
                    // Should not happen — candidates come from legal_moves.
                    continue;
                }
                if g.is_terminal() {
                    // Terminal post-p₁: no p₂s to capture; leave as None.
                    continue;
                }
                pending_pos.push(pos);
                pending_states.push(g);
            }
            if !pending_states.is_empty() {
                let (post_logits_per_state, _values) = eval_fn(&pending_states);
                for (k, &pos) in pending_pos.iter().enumerate() {
                    // Filter the candidate's logits to its legal p₂s and
                    // convert to priors via softmax. We want the *full*
                    // softmax over legal p₂s (not just the top-K logits)
                    // because the same coord can appear with low logit but
                    // be the top-K's K-th member after normalization.
                    let p2_state = &pending_states[k];
                    let p2_moves = p2_state.legal_moves();
                    let map = &post_logits_per_state[k];
                    let p2_logits: Vec<f64> =
                        p2_moves.iter().map(|c| map.get(c).copied().unwrap_or(0.0)).collect();
                    let p2_priors = super::scoring::softmax(&p2_logits);
                    let pairs: Vec<(Coord, f64)> = p2_moves
                        .into_iter()
                        .zip(p2_priors.into_iter())
                        .collect();
                    per_candidate[pos] = Some(pairs);
                }
            }
            Some(per_candidate)
        } else {
            None
        };

    // Step 4: Sequential halving with batched evaluation
    let vl_magnitude = config.virtual_loss;
    let surviving_indices = sequential_halving(
        &mut root,
        &candidate_indices,
        &gumbel_samples,
        &coords,
        &logits,
        config,
        &mut |root_node, batch_actions, c_visit, c_scale| {
            simulate_batch_with_eval(
                root_node,
                batch_actions,
                c_visit,
                c_scale,
                vl_magnitude,
                eval_fn,
            );
        },
    );

    // Step 5: Compute improved policy over all legal moves
    let improved_policy =
        compute_improved_policy(&logits, &coords, &root, config.c_visit, config.c_scale);

    // Step 6: Select action from survivors using Gumbel + logits + σ(Q).
    let qctx = QContext::new(&root, &coords);

    let mut best_score = f64::NEG_INFINITY;
    let mut selected_coord = coords[surviving_indices[0]];

    for &idx in &surviving_indices {
        let coord = coords[idx];
        let norm_q = qctx.norm_q[&coord];

        let score = gumbel_samples[idx]
            + logits[idx]
            + sigma(norm_q, qctx.max_child_visits, config.c_visit, config.c_scale);

        if score > best_score {
            best_score = score;
            selected_coord = coord;
        }
    }

    let visit_counts: Vec<u32> = coords
        .iter()
        .map(|c| root.children.get(c).map(|n| n.visit_count).unwrap_or(0))
        .collect();

    // Diagnostic fields: reuse the QContext already built for action
    // selection above. `per_child_q` mirrors what σ(Q) sees during halving
    // (visited → empirical Q from root POV; unvisited → v_mix fallback).
    let per_child_q: Vec<f64> = coords.iter().map(|c| qctx.completed_q[c]).collect();
    // `per_child_prior` is the root expand-time prior (post-Dirichlet,
    // pre-Gumbel, pre-σ(Q)). `priors_vec` is aligned with `coords` by
    // construction (built from the same `logits` indexing).
    let per_child_prior = priors_vec.clone();

    // E.4-inject: if capture is enabled and the chosen action's post-p₁
    // priors were computed (non-terminal post-state), take the top-K p₂s
    // by prior magnitude. Deterministic top-K — no Gumbel sampling.
    // Empty otherwise (capture disabled, mr=1 root, or terminal-post-p₁).
    let chosen_action_forced_candidates: Vec<Coord> = match captured_p2_priors.as_ref() {
        Some(per_candidate) => {
            // Locate the chosen action's position within candidate_indices.
            // SH always returns a survivor that was in candidate_indices,
            // so this lookup must succeed.
            let chosen_pos = candidate_indices
                .iter()
                .position(|&idx| coords[idx] == selected_coord);
            match chosen_pos.and_then(|p| per_candidate[p].as_ref()) {
                Some(pairs) => {
                    let k = config.forced_candidate_capture_k.min(pairs.len());
                    let mut sorted: Vec<(Coord, f64)> = pairs.clone();
                    sorted.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    sorted.into_iter().take(k).map(|(c, _)| c).collect()
                }
                None => Vec::new(),
            }
        }
        None => Vec::new(),
    };

    Ok(MCTSResult {
        action: selected_coord,
        improved_policy,
        coords,
        visit_counts,
        per_child_q,
        per_child_prior,
        candidate_indices,
        chosen_action_forced_candidates,
    })
}

/// Run a batch of forced-action simulations, calling eval_fn for non-terminal leaves.
///
/// `batch_actions` may contain the same action repeated `sims_per_action` times
/// when SH inner-loop fusion is active. `vl_magnitude > 0.0` enables virtual
/// loss: each selected non-terminal leaf gets a virtual-loss bump applied to
/// its path *before* the next selection runs, so descents below the forced root
/// child diversify within the fused batch. After the batched `eval_fn` returns,
/// each path's VL is reverted before the real backup runs.
fn simulate_batch_with_eval<F>(
    root: &mut MCTSNode,
    batch_actions: &[Coord],
    c_visit: u32,
    c_scale: f64,
    vl_magnitude: f64,
    eval_fn: &mut F,
) where
    F: FnMut(&[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>),
{
    // Phase 1: Select leaves for each forced action, applying virtual loss
    // between selections so subsequent descents see prior selections' paths
    // as in-flight (visited + pessimised) rather than fresh.
    let mut pending_selections = Vec::new();
    let mut pending_states = Vec::new();

    for &action in batch_actions {
        match simulate_select(root, Some(action), c_visit, c_scale) {
            None => {
                // Terminal leaf — backup already applied; no VL needed since
                // no batched eval will be performed for this slot.
            }
            Some(selection) => {
                // Resolve the leaf game state (shared borrow scope), then drop
                // it so we can take a mutable borrow for VL bookkeeping.
                let leaf_state: Option<GameState> = {
                    let mut node: &MCTSNode = root;
                    for &a in &selection.path_actions {
                        node = node.children.get(&a).expect("path broken");
                    }
                    node.game_state.clone()
                };
                if let Some(game_state) = leaf_state {
                    pending_states.push(game_state);
                    apply_virtual_loss(root, &selection, vl_magnitude);
                    pending_selections.push(selection);
                }
            }
        }
    }

    if pending_states.is_empty() {
        return;
    }

    // Phase 2: Batch evaluate all non-terminal leaves — eval_fn returns raw logits.
    //
    // Under `dedup_skip`, first consult the per-search eval cache. States whose
    // fingerprint already has a cached (logits, value) are NOT sent to the
    // evaluator; their results come straight from the cache. The remaining
    // (miss) states are batched into a single eval_fn call, and their results
    // are inserted into the cache for future hits within the same search.
    #[cfg(feature = "dedup_skip")]
    let (logits_list, values): (Vec<HashMap<Coord, f64>>, Vec<f64>) = if !super::dedup_count::skip_enabled_tls() {
        eval_fn(&pending_states)
    } else {
        let mut fingerprints: Vec<u128> = Vec::with_capacity(pending_states.len());
        for (i, selection) in pending_selections.iter().enumerate() {
            let path = &selection.path_actions;
            let mut node: &MCTSNode = root;
            let mut parent_state: Option<&GameState> = root.game_state.as_ref();
            for (depth, &a) in path.iter().enumerate() {
                let child = node.children.get(&a).expect("dedup_skip path broken");
                if depth + 1 == path.len() {
                    break;
                }
                parent_state = child.game_state.as_ref();
                node = child;
            }
            let fp = super::dedup_count::position_fingerprint(
                &pending_states[i],
                parent_state,
            );
            fingerprints.push(fp);
        }

        let n = pending_states.len();
        let mut out_logits: Vec<Option<HashMap<Coord, f64>>> = (0..n).map(|_| None).collect();
        let mut out_values: Vec<f64> = vec![0.0; n];
        let mut miss_idxs: Vec<usize> = Vec::new();
        let mut miss_states: Vec<GameState> = Vec::new();
        for (i, fp) in fingerprints.iter().enumerate() {
            if let Some(entry) = super::dedup_count::eval_cache_get_tls(*fp) {
                out_logits[i] = Some(entry.policy_logits);
                out_values[i] = entry.value;
            } else {
                miss_idxs.push(i);
                miss_states.push(pending_states[i].clone());
            }
        }

        if !miss_states.is_empty() {
            let (miss_logits, miss_vals) = eval_fn(&miss_states);
            for (k, &i) in miss_idxs.iter().enumerate() {
                let lm = miss_logits[k].clone();
                let v = miss_vals[k];
                super::dedup_count::eval_cache_insert_tls(
                    fingerprints[i],
                    super::dedup_count::CachedEval {
                        policy_logits: lm.clone(),
                        value: v,
                    },
                );
                out_logits[i] = Some(lm);
                out_values[i] = v;
            }
        }

        (
            out_logits.into_iter().map(|o| o.unwrap()).collect(),
            out_values,
        )
    };

    #[cfg(not(feature = "dedup_skip"))]
    let (logits_list, values) = eval_fn(&pending_states);

    // Phase 3: Revert virtual loss along each path, then convert logits to
    // priors and apply the real backup. Revert MUST happen before
    // `complete_simulation`'s `apply_backup` so the path's `visit_count` /
    // `value_sum` end up reflecting only the real sim, not the in-flight VL
    // adjustment.
    for (i, selection) in pending_selections.iter().enumerate() {
        revert_virtual_loss(root, selection, vl_magnitude);

        let logits_map = &logits_list[i];
        // Convert logits to priors via softmax
        let coords: Vec<Coord> = logits_map.keys().copied().collect();
        let logit_vals: Vec<f64> = coords.iter().map(|c| logits_map[c]).collect();
        let prior_vals = super::scoring::softmax(&logit_vals);
        let leaf_priors: HashMap<Coord, f64> = coords
            .into_iter()
            .zip(prior_vals)
            .collect();
        let leaf_value = values[i];
        complete_simulation(root, selection, leaf_priors, leaf_value);

        #[cfg(feature = "dedup_count")]
        {
            // Walk the path to locate (parent_state, child_state) for the
            // just-expanded leaf and record against the thread-local
            // instrumentation singleton.
            let mut node: &MCTSNode = root;
            let mut parent_state: Option<&GameState> = root.game_state.as_ref();
            let path = &selection.path_actions;
            for (depth, &a) in path.iter().enumerate() {
                let child = node.children.get(&a).expect("instr path broken");
                if depth + 1 == path.len() {
                    if let Some(child_state) = child.game_state.as_ref() {
                        super::dedup_count::record_expansion_tls(parent_state, child_state);
                    }
                    break;
                }
                parent_state = child.game_state.as_ref();
                node = child;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{GameConfig, Player};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn small_config() -> GameConfig {
        GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        }
    }

    /// Dummy eval_fn that returns uniform logits (0.0) and value 0.0.
    fn dummy_eval(states: &[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let mut all_logits = Vec::new();
        let mut all_values = Vec::new();
        for state in states {
            let moves = state.legal_moves();
            let logits: HashMap<Coord, f64> = moves.iter().map(|&m| (m, 0.0)).collect();
            all_logits.push(logits);
            all_values.push(0.0);
        }
        (all_logits, all_values)
    }

    #[test]
    fn gumbel_mcts_returns_legal_move() {
        let game = GameState::with_config(small_config());
        let legal = game.legal_moves();
        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();
        assert!(legal.contains(&result.action));
    }

    #[test]
    fn gumbel_mcts_policy_sums_to_one() {
        let game = GameState::with_config(small_config());
        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();
        let sum: f64 = result.improved_policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gumbel_mcts_policy_length_matches_legal_moves() {
        let game = GameState::with_config(small_config());
        let n_legal = game.legal_moves().len();
        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();
        assert_eq!(result.improved_policy.len(), n_legal);
        assert_eq!(result.coords.len(), n_legal);
    }

    #[test]
    fn gumbel_mcts_terminal_returns_error() {
        let config_game = GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 80,
        };
        let mut game = GameState::with_config(config_game);
        game.apply_move((1, 0)).unwrap(); // P2
        game.apply_move((2, 0)).unwrap(); // P2 wins with win_length=2
        assert!(game.is_terminal());

        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval);
        assert!(result.is_err());
    }

    #[test]
    fn gumbel_mcts_deterministic_with_same_seed() {
        let game = GameState::with_config(small_config());
        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);

        let r1 = gumbel_mcts(&game, &config, &mut rng1, None, &mut dummy_eval).unwrap();
        let r2 = gumbel_mcts(&game, &config, &mut rng2, None, &mut dummy_eval).unwrap();
        assert_eq!(r1.action, r2.action);
    }

    #[test]
    fn gumbel_mcts_zero_simulations() {
        let game = GameState::with_config(small_config());
        let config = MCTSConfig {
            n_simulations: 0,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Should still return a valid action (based on Gumbel + logits alone)
        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();
        assert!(game.legal_moves().contains(&result.action));
    }

    #[test]
    fn biased_eval_prefers_high_value_action() {
        // eval_fn that returns value=0.9 when it's your turn at states where
        // target_move was played (good position), and value=-0.9 otherwise.
        // After search, the action should favor the high-value move.
        let game = GameState::with_config(small_config());
        let legal = game.legal_moves();
        let target_move = legal[0]; // first legal move gets the boost

        // This eval gives massive logit advantage to target_move at the root
        // and value=0 for all states, so Q values are zero and the improved
        // policy reduces to softmax(logits), which should heavily favor target_move.
        let mut biased_eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
            let mut all_logits = Vec::new();
            let mut all_values = Vec::new();
            for state in states {
                let moves = state.legal_moves();
                let logits: HashMap<Coord, f64> = moves
                    .iter()
                    .map(|&m| {
                        if m == target_move {
                            (m, 10.0)
                        } else {
                            (m, 0.0)
                        }
                    })
                    .collect();
                all_logits.push(logits);
                all_values.push(0.0);
            }
            (all_logits, all_values)
        };

        let config = MCTSConfig {
            n_simulations: 32,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut biased_eval).unwrap();

        // The improved policy should strongly favor the target_move due to the
        // large logit advantage at the root.
        let target_idx = result.coords.iter().position(|&c| c == target_move).unwrap();
        let target_prob = result.improved_policy[target_idx];
        assert!(
            target_prob > 0.3,
            "Target move should get high policy weight, got {}",
            target_prob,
        );
    }

    #[test]
    fn gumbel_mcts_finds_winning_move() {
        // Set up a game state where one move wins immediately (win_length=2,
        // one stone away). With enough simulations, MCTS should find it.
        // Use a simple eval that returns uniform logits but the terminal
        // detection gives +1/-1.
        let config_game = GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 80,
        };
        let mut game = GameState::with_config(config_game);
        // P1 is at (0,0). P2 starts with 2 moves.
        // P2 plays non-adjacent moves within radius 2 of existing stones.
        game.apply_move((2, 0)).unwrap();  // P2 move 1 (hex-dist 2 from origin)
        game.apply_move((-2, 0)).unwrap(); // P2 move 2 (hex-dist 2 from origin)
        // Now it's P1's turn with 2 moves. P1 already has (0,0).
        // P1 plays at (1,0) → 2 adjacent P1 stones → P1 wins with win_length=2!
        assert_eq!(game.current_player(), Some(Player::P1));
        let winning_move = (1, 0); // adjacent to P1's stone at (0,0)
        assert!(game.legal_moves().contains(&winning_move));

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Uniform eval: all logits 0, value 0. MCTS must rely on terminal
        // detection to discover the win.
        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();

        // Any move adjacent to (0,0) wins with win_length=2.
        // Early termination should find one of them.
        let winning_moves: Vec<Coord> = hexo_engine::types::HEX_DIRS.iter()
            .map(|&(dq, dr)| (dq, dr))
            .filter(|c| game.legal_moves().contains(c))
            .collect();

        let is_winning_action = winning_moves.contains(&result.action);
        assert!(
            is_winning_action,
            "MCTS should find a winning move: action={:?}, winning_moves={:?}",
            result.action,
            winning_moves,
        );
    }

    /// Regression: early-termination on a winning move must return a one-hot
    /// improved policy, not the raw softmax (which was the old buggy behavior
    /// when compute_improved_policy was called before any simulations ran).
    #[test]
    fn gumbel_mcts_winning_move_returns_one_hot_policy() {
        let config_game = GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 80,
        };
        let mut game = GameState::with_config(config_game);
        game.apply_move((2, 0)).unwrap();
        game.apply_move((-2, 0)).unwrap();

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        // Run multiple seeds to cover both early-exit and full-search paths
        // (early exit triggers only when a winning move is among the Gumbel
        // candidates, which depends on the random sample).
        for seed in 0..20 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();

            // Find the winning action's index in the policy
            let action_idx = result.coords.iter()
                .position(|&c| c == result.action)
                .expect("action must be in coords");

            // The action must be a winning move
            let mut test_game = game.clone();
            test_game.apply_move(result.action).unwrap();
            assert!(test_game.is_terminal(), "seed {seed}: action should win");

            // Policy must be one-hot on the winning move
            assert_eq!(
                result.improved_policy[action_idx], 1.0,
                "seed {seed}: winning move should have probability 1.0, got {}",
                result.improved_policy[action_idx],
            );
            for (i, &p) in result.improved_policy.iter().enumerate() {
                if i != action_idx {
                    assert_eq!(p, 0.0,
                        "seed {seed}: non-winning move {i} should have probability 0.0, got {p}");
                }
            }
        }
    }

    /// Depth-2 forced-win shortcut: when the current player has 2 placements
    /// remaining this turn and a candidate's first placement leaves the same
    /// player able to win on their second placement, MCTS must short-circuit.
    /// Change (B): the winning turn is the order-invariant pair {(1,0),(2,0)},
    /// so the shortcut emits a TWO-HOT policy (0.5 on each gap), not a one-hot.
    #[test]
    fn gumbel_mcts_two_placement_forced_win_returns_two_hot_policy() {
        // win_length = 4: P1 has stones at (0,0) and (3,0). Filling the gaps
        // at (1,0) and (2,0) (in either order) makes a 4-in-a-row along the
        // q-axis. Stones are spaced so that ONLY this specific gap pair
        // produces a 2-placement forced win (no other 4-window is reachable
        // in two placements). Single-placement does NOT win (depth-1
        // shortcut must not fire).
        let config_game = GameConfig {
            win_length: 4,
            placement_radius: 1,
            max_moves: 80,
        };
        // Minimal position: two P1 stones along the q-axis 3 cells apart.
        // radius=1 keeps the legal-move set small (~10 cells) so m_actions
        // covers it entirely and the depth-2 shortcut deterministically
        // sees both gap squares.
        let stones = vec![
            ((0, 0), Player::P1),
            ((3, 0), Player::P1),
        ];
        let game = GameState::from_state(&stones, Player::P1, 2, config_game);

        // Sanity: both gap squares are legal, neither alone wins.
        for &gap in &[(1, 0), (2, 0)] {
            assert!(
                game.legal_moves().contains(&gap),
                "gap {gap:?} must be legal in the constructed position"
            );
            let mut g = game.clone();
            g.apply_move(gap).unwrap();
            assert!(
                !g.is_terminal(),
                "single placement at {gap:?} should not win (depth-1 must not fire)"
            );
        }

        // m_actions large enough that all legal moves are in the candidate
        // set, ensuring the depth-2 shortcut sees both gap squares.
        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 64,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        for seed in 0..20 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();

            assert!(
                result.action == (1, 0) || result.action == (2, 0),
                "seed {seed}: action {:?} must be one of the forced-win gaps",
                result.action
            );

            let gap0 = result
                .coords
                .iter()
                .position(|&c| c == (1, 0))
                .expect("(1,0) in coords");
            let gap1 = result
                .coords
                .iter()
                .position(|&c| c == (2, 0))
                .expect("(2,0) in coords");

            // Two-hot: both gaps of the winning pair carry 0.5, all else 0.0.
            assert!(
                (result.improved_policy[gap0] - 0.5).abs() < 1e-9,
                "seed {seed}: gap (1,0) should have probability 0.5, got {}",
                result.improved_policy[gap0]
            );
            assert!(
                (result.improved_policy[gap1] - 0.5).abs() < 1e-9,
                "seed {seed}: gap (2,0) should have probability 0.5, got {}",
                result.improved_policy[gap1]
            );
            for (i, &p) in result.improved_policy.iter().enumerate() {
                if i != gap0 && i != gap1 {
                    assert_eq!(
                        p, 0.0,
                        "seed {seed}: off-pair move {i} should have probability 0.0"
                    );
                }
            }
        }
    }

    /// Guard: on this own_4 pattern at mr==1 the shortcut must not produce a
    /// forced-win one-hot. Under Change (A) the solver now DOES run at mr==1,
    /// but for this fixture the single remaining placement can't force a win
    /// (the two gaps `(1,0)`/`(2,0)` can't both be filled before the opponent's
    /// full turn), so `forcing::solve` returns `No` and the search falls
    /// through to a non-one-hot policy -- exactly as when the solver was gated
    /// off at mr==1 entirely.
    #[test]
    fn gumbel_mcts_depth2_shortcut_skipped_mid_turn() {
        let config_game = GameConfig {
            win_length: 4,
            placement_radius: 1,
            max_moves: 80,
        };
        let stones = vec![
            ((0, 0), Player::P1),
            ((3, 0), Player::P1),
        ];
        // moves_remaining = 1: P1 will place once, then P2 moves.
        let game = GameState::from_state(&stones, Player::P1, 1, config_game);

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 64,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();

        // No single-placement win exists here, so depth-1 doesn't fire either.
        // Improved policy must therefore come from sequential halving — i.e.
        // it must NOT be one-hot on either gap. (We assert no entry hits 1.0
        // exactly; a real search distributes weight.)
        assert!(
            result.improved_policy.iter().all(|&p| p < 0.999),
            "depth-2 shortcut must not fire mid-turn; got policy {:?}",
            result.improved_policy
        );
    }

    /// REQ-0e-bis smoke test: with `dedup_skip`, running the same search
    /// twice in a row (so the second run sees a fully populated eval cache)
    /// must yield the same chosen action and improved policy as the first.
    /// Verifies the cache-hit path produces tree nodes equivalent to a real
    /// expansion.
    #[cfg(feature = "dedup_skip")]
    #[test]
    fn dedup_skip_cache_hit_matches_real_eval() {
        let game = GameState::with_config(small_config());
        let config = MCTSConfig {
            n_simulations: 32,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        crate::mcts::dedup_count::set_skip_enabled_tls(true);
        // Run 1: cache empty -> all real evals.
        crate::mcts::dedup_count::eval_cache_reset_tls();
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let r1 = gumbel_mcts(&game, &config, &mut rng1, None, &mut dummy_eval).unwrap();
        let (h1, m1, _) = crate::mcts::dedup_count::eval_cache_snapshot_tls();

        // Run 2: cache populated -> many cache hits, no resets.
        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let r2 = gumbel_mcts(&game, &config, &mut rng2, None, &mut dummy_eval).unwrap();
        let (h2, m2, _) = crate::mcts::dedup_count::eval_cache_snapshot_tls();

        assert_eq!(r1.action, r2.action);
        assert_eq!(r1.coords, r2.coords);
        for (a, b) in r1.improved_policy.iter().zip(r2.improved_policy.iter()) {
            assert!((a - b).abs() < 1e-9, "policy diverged: {} vs {}", a, b);
        }
        // Run 2 should have hit the cache far more than run 1.
        assert!(h2 > h1, "second run should have more cache hits ({} vs {})", h2, h1);
        // And it should have done strictly fewer fresh evals than run 1.
        assert!(m2 - m1 < m1, "second run should miss less ({} new vs {} initial)", m2 - m1, m1);

        crate::mcts::dedup_count::eval_cache_reset_tls();
        crate::mcts::dedup_count::set_skip_enabled_tls(false);
    }

    #[test]
    fn gumbel_mcts_early_terminates_on_win() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let eval_count = AtomicU32::new(0);

        let mut eval_fn = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
            eval_count.fetch_add(states.len() as u32, Ordering::Relaxed);
            let mut logits = Vec::new();
            let mut values = Vec::new();
            for s in states {
                let moves = s.legal_moves();
                let n = moves.len();
                let priors: HashMap<Coord, f64> =
                    moves.iter().map(|&c| (c, 1.0 / n as f64)).collect();
                logits.push(priors);
                values.push(0.0);
            }
            (logits, values)
        };

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let game = GameState::with_config(small_config());
        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut eval_fn);
        assert!(result.is_ok());
        // Just verify it completes — early termination is an optimization,
        // correctness is preserved regardless.
    }

    #[test]
    fn gumbel_mcts_with_virtual_loss_completes() {
        // SH inner-loop fusion is active (virtual_loss > 0). The search must
        // still return a valid action + improved policy with all probability
        // mass on legal moves. This is the smoke test for Phase 2.
        let game = GameState::with_config(small_config());
        let legal = game.legal_moves();

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.5,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();
        assert!(legal.contains(&result.action));
        let sum: f64 = result.improved_policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(result.coords.len(), legal.len());
    }

    #[test]
    fn gumbel_mcts_fused_sh_calls_eval_at_most_once_per_phase() {
        // Fusion is gated on `virtual_loss > 0`. With VL > 0, each SH
        // while-iteration emits exactly one `eval_fn` call: 1 root +
        // num_phases halving iterations + budget top-up iterations on the
        // final pair (mctx parity). For n=32/m=8 the halving phases spend
        // 8+8+10=26 sims; the leftover 6 fit in one top-up iteration
        // (spa=5 → up to 10 sims), so the bound is 1 + num_phases + 1.
        // With VL = 0, the original serial loop runs (sims_per_action calls
        // per iteration), so call count is strictly higher.
        let game = GameState::with_config(small_config());
        let n_legal = game.legal_moves().len();
        let num_candidates = n_legal.min(8);
        let num_phases = (num_candidates as f64).log2().ceil() as u32;
        let fused_expected_max = 1 + num_phases + 1;

        // VL > 0 → fused, bounded by 1 + num_phases + 1 top-up.
        let config_fused = MCTSConfig {
            n_simulations: 32, m_actions: 8, c_visit: 50, c_scale: 1.0,
            virtual_loss: 0.5,
            ..Default::default()
        };
        let calls_fused = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let calls_clone = calls_fused.clone();
        let mut eval_fused =
            move |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
                calls_clone.set(calls_clone.get() + 1);
                dummy_eval(states)
            };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let _ = gumbel_mcts(&game, &config_fused, &mut rng, None, &mut eval_fused).unwrap();
        let fused = calls_fused.get();
        assert!(
            fused <= fused_expected_max,
            "fused: expected ≤ {fused_expected_max} eval calls, got {fused} (n_candidates={num_candidates}, num_phases={num_phases})",
        );
        assert!(fused >= 1, "fused: no eval calls at all");

        // VL = 0 → serial, strictly more calls than the fused path (or equal
        // in pathological tiny-budget cases). Use the SAME seed so the only
        // difference is the inner-loop structure.
        let config_serial = MCTSConfig { virtual_loss: 0.0, ..config_fused.clone() };
        let calls_serial = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let calls_clone = calls_serial.clone();
        let mut eval_serial =
            move |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
                calls_clone.set(calls_clone.get() + 1);
                dummy_eval(states)
            };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let _ = gumbel_mcts(&game, &config_serial, &mut rng, None, &mut eval_serial).unwrap();
        let serial = calls_serial.get();
        assert!(
            serial >= fused,
            "serial path should emit at least as many eval calls as fused (serial={serial}, fused={fused})",
        );
    }

    // -----------------------------------------------------------------
    // E.4-inject: joint p₂ candidate injection
    // -----------------------------------------------------------------

    /// Default `forced_candidate_capture_k = 0` must leave the legacy
    /// search bit-identical: same action, same improved policy, same
    /// candidate set, and `chosen_action_forced_candidates` empty.
    /// Bonus guarantee: zero extra eval_fn calls. Anchors the disabled
    /// path so the new field never silently grows side effects.
    #[test]
    fn e4_inject_capture_zero_bit_equivalent_to_legacy() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let game = GameState::with_config(small_config());
        let cfg_off = MCTSConfig {
            n_simulations: 16,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            forced_candidate_capture_k: 0,
            ..Default::default()
        };

        let counter1 = Rc::new(RefCell::new(0u32));
        let cc1 = counter1.clone();
        let mut counting_eval_1 = move |states: &[GameState]| {
            *cc1.borrow_mut() += 1;
            dummy_eval(states)
        };

        let mut rng1 = ChaCha8Rng::seed_from_u64(99);
        let r_off = gumbel_mcts(&game, &cfg_off, &mut rng1, None, &mut counting_eval_1).unwrap();
        let calls_off = *counter1.borrow();

        // Run again with the legacy default (capture not set) using the
        // same seed and confirm bit-equivalence.
        let cfg_legacy = MCTSConfig {
            n_simulations: 16,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        let counter2 = Rc::new(RefCell::new(0u32));
        let cc2 = counter2.clone();
        let mut counting_eval_2 = move |states: &[GameState]| {
            *cc2.borrow_mut() += 1;
            dummy_eval(states)
        };
        let mut rng2 = ChaCha8Rng::seed_from_u64(99);
        let r_legacy =
            gumbel_mcts(&game, &cfg_legacy, &mut rng2, None, &mut counting_eval_2).unwrap();
        let calls_legacy = *counter2.borrow();

        assert_eq!(r_off.action, r_legacy.action);
        assert_eq!(r_off.coords, r_legacy.coords);
        assert_eq!(r_off.visit_counts, r_legacy.visit_counts);
        for (a, b) in r_off
            .improved_policy
            .iter()
            .zip(r_legacy.improved_policy.iter())
        {
            assert_eq!(a.to_bits(), b.to_bits(),
                "improved_policy must be bit-identical");
        }
        assert_eq!(r_off.candidate_indices, r_legacy.candidate_indices);
        assert_eq!(
            r_off.chosen_action_forced_candidates,
            Vec::<Coord>::new(),
            "capture=0 must leave chosen_action_forced_candidates empty",
        );
        assert_eq!(
            r_legacy.chosen_action_forced_candidates,
            Vec::<Coord>::new(),
            "legacy default must leave chosen_action_forced_candidates empty",
        );
        assert_eq!(calls_off, calls_legacy,
            "capture=0 must do no extra eval_fn calls");
    }

    /// At a mr=1 root, capture is a no-op regardless of `capture_k`
    /// because the next placement starts a new turn (opponent's). The
    /// captured `chosen_action_forced_candidates` must be empty and the
    /// extra eval batch must not run.
    #[test]
    fn e4_inject_no_capture_at_mr1() {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Build a mr=1 state: P1's opening at (0,0), then play one P2
        // stone so mr drops from 2 to 1 for P2.
        let mut game = GameState::with_config(small_config());
        let legal = game.legal_moves();
        game.apply_move(legal[0]).unwrap();
        assert_eq!(game.moves_remaining_this_turn(), 1);

        let counter = Rc::new(RefCell::new(0u32));
        let cc = counter.clone();
        let mut counting_eval = move |states: &[GameState]| {
            *cc.borrow_mut() += 1;
            dummy_eval(states)
        };

        let cfg_on = MCTSConfig {
            n_simulations: 32,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            forced_candidate_capture_k: 4,
            ..Default::default()
        };

        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let result = gumbel_mcts(&game, &cfg_on, &mut rng, None, &mut counting_eval).unwrap();
        let calls_on = *counter.borrow();

        assert!(
            result.chosen_action_forced_candidates.is_empty(),
            "mr=1 root must produce empty chosen_action_forced_candidates"
        );

        // Repeat with capture off and confirm the eval count matches.
        let cfg_off = MCTSConfig {
            forced_candidate_capture_k: 0,
            ..cfg_on.clone()
        };
        *counter.borrow_mut() = 0;
        let mut rng2 = ChaCha8Rng::seed_from_u64(13);
        let _ = gumbel_mcts(&game, &cfg_off, &mut rng2, None, &mut counting_eval).unwrap();
        let calls_off = *counter.borrow();
        assert_eq!(
            calls_on, calls_off,
            "mr=1 capture must not do any extra eval_fn calls"
        );
    }

    /// At a mr=2 root with `capture_k > 0` and `forced_candidates=None`,
    /// the action and improved policy must be bit-identical to the
    /// capture-off path (the extra eval happens but the result is
    /// otherwise unchanged). `chosen_action_forced_candidates` is
    /// non-empty and contains coords from the chosen action's post-state.
    #[test]
    fn e4_inject_capture_on_action_bit_identical_to_off() {
        // Start-of-turn (mr=2) position with diverse legal moves so
        // capture has multiple candidates to evaluate.
        let cfg_game = GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        };
        let stones = vec![((0, 0), Player::P1)];
        let game = GameState::from_state(&stones, Player::P2, 2, cfg_game);
        assert_eq!(game.moves_remaining_this_turn(), 2);

        let cfg_off = MCTSConfig {
            n_simulations: 32,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            forced_candidate_capture_k: 0,
            ..Default::default()
        };
        let cfg_on = MCTSConfig {
            forced_candidate_capture_k: 4,
            ..cfg_off.clone()
        };

        let mut rng1 = ChaCha8Rng::seed_from_u64(77);
        let r_off = gumbel_mcts(&game, &cfg_off, &mut rng1, None, &mut dummy_eval).unwrap();
        let mut rng2 = ChaCha8Rng::seed_from_u64(77);
        let r_on = gumbel_mcts(&game, &cfg_on, &mut rng2, None, &mut dummy_eval).unwrap();

        assert_eq!(r_off.action, r_on.action,
            "capture-on action must equal capture-off action when bonuses are not applied");
        assert_eq!(r_off.coords, r_on.coords);
        assert_eq!(r_off.visit_counts, r_on.visit_counts);
        for (a, b) in r_off
            .improved_policy
            .iter()
            .zip(r_on.improved_policy.iter())
        {
            assert_eq!(a.to_bits(), b.to_bits(),
                "improved_policy must be bit-identical between capture-on and capture-off");
        }
        assert_eq!(r_off.candidate_indices, r_on.candidate_indices);

        // Off path leaves the new field empty.
        assert!(r_off.chosen_action_forced_candidates.is_empty());
        // On path captures the chosen p₁'s top-K p₂s.
        assert!(
            !r_on.chosen_action_forced_candidates.is_empty(),
            "capture-on must populate chosen_action_forced_candidates at mr=2"
        );
        assert!(
            r_on.chosen_action_forced_candidates.len() <= 4,
            "chosen_action_forced_candidates length must not exceed capture_k=4 (got {})",
            r_on.chosen_action_forced_candidates.len(),
        );
    }

    /// With a hand-crafted eval_fn that returns specific conditional priors
    /// at each post-p₁ state, the captured top-K p₂s must equal the K
    /// coords with the largest conditional priors at the chosen p₁'s
    /// post-state. Tests the deterministic top-K-by-prior selection.
    #[test]
    fn e4_inject_capture_returns_correct_top_k_p2s() {
        // Build a mr=2 position with a small legal-move set (so we know
        // what coords appear). win_length=4, placement_radius=1 keeps
        // legal moves down to a single ring around (0,0).
        let cfg_game = GameConfig {
            win_length: 4,
            placement_radius: 1,
            max_moves: 80,
        };
        let stones = vec![((0, 0), Player::P1)];
        let game = GameState::from_state(&stones, Player::P2, 2, cfg_game);
        assert_eq!(game.moves_remaining_this_turn(), 2);

        // Pick a target post-p₁ coord we expect SH to select.
        let legal = game.legal_moves();
        assert!(legal.len() >= 4);
        let target_p1 = legal[0];

        // Designate which p₂ coords should have the highest priors when
        // the post-p₁ state for `target_p1` is evaluated. We'll arrange
        // logits so that {favored_a, favored_b, favored_c} have logit=10
        // while all other p₂s have logit=0.
        // The actual coord identities are determined by the post-state's
        // legal_moves(); we'll query that and pick the first 3.
        let mut target_game = game.clone();
        target_game.apply_move(target_p1).unwrap();
        let target_p2_legal = target_game.legal_moves();
        assert!(target_p2_legal.len() >= 3,
            "test setup: target_p1 post-state must have ≥3 legal p₂s");
        let favored_a = target_p2_legal[0];
        let favored_b = target_p2_legal[1];
        let favored_c = target_p2_legal[2];

        // eval_fn that heavily boosts the favored p₂s at any state whose
        // stones contain target_p1 (and the original P1 opening). At the
        // root and elsewhere, returns uniform logits.
        let mut eval = move |states: &[GameState]| {
            let mut all_logits = Vec::new();
            let mut all_values = Vec::new();
            for s in states {
                let stones = s.stones();
                let is_target_post = stones.contains_key(&target_p1);
                let moves = s.legal_moves();
                let logits: HashMap<Coord, f64> = moves
                    .into_iter()
                    .map(|m| {
                        if is_target_post
                            && (m == favored_a || m == favored_b || m == favored_c)
                        {
                            (m, 10.0)
                        } else {
                            (m, 0.0)
                        }
                    })
                    .collect();
                all_logits.push(logits);
                all_values.push(0.0);
            }
            (all_logits, all_values)
        };

        // Force m_actions large enough that target_p1 (legal[0]) is in
        // the Gumbel-Top-K. Also force enough sims that SH actually picks
        // it. With uniform root logits and gumbel chooses what it chooses,
        // we can't pin which p₁ SH picks — but we CAN verify that *if*
        // the chosen action is `target_p1`, then the captured top-K p₂s
        // are exactly {favored_a, favored_b, favored_c} (in some order;
        // ranked by prior magnitude). We iterate seeds until we find one
        // where SH picks target_p1, then assert.
        let cfg = MCTSConfig {
            n_simulations: 16,
            m_actions: legal.len(), // all candidates Gumbel-Top-K survivors
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            forced_candidate_capture_k: 3,
            ..Default::default()
        };

        let mut found = false;
        for seed in 0..200u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let result = gumbel_mcts(&game, &cfg, &mut rng, None, &mut eval).unwrap();
            if result.action != target_p1 {
                continue;
            }
            // Capture should produce exactly the 3 favored p₂s.
            let captured: std::collections::HashSet<Coord> =
                result.chosen_action_forced_candidates.iter().copied().collect();
            let expected: std::collections::HashSet<Coord> =
                [favored_a, favored_b, favored_c].iter().copied().collect();
            assert_eq!(
                captured, expected,
                "seed {seed}: captured top-3 must equal favored p₂ set (favored={favored_a:?},{favored_b:?},{favored_c:?}); got {:?}",
                result.chosen_action_forced_candidates,
            );
            assert_eq!(
                result.chosen_action_forced_candidates.len(),
                3,
                "seed {seed}: captured set length must equal capture_k=3"
            );
            found = true;
            break;
        }
        assert!(found,
            "test seed sweep failed to land on target_p1 — adjust seed range or m_actions");
    }

    /// When the chosen p₁ leads to a terminal post-state (own win via the
    /// depth-1 shortcut), `chosen_action_forced_candidates` must be empty
    /// — no priors are readable from a terminal state.
    #[test]
    fn e4_inject_no_capture_on_terminal_post_p1() {
        // P2 already has 3 in a row; mr=2 means a single placement at
        // (4,0) wins. The depth-1 shortcut catches this and returns
        // immediately; the resulting MCTSResult must report an empty
        // captured set regardless of capture_k.
        let cfg_game = GameConfig {
            win_length: 4,
            placement_radius: 1,
            max_moves: 80,
        };
        let stones = vec![
            ((0, 0), Player::P1),
            ((1, 0), Player::P2),
            ((2, 0), Player::P2),
            ((3, 0), Player::P2),
        ];
        let game = GameState::from_state(&stones, Player::P2, 2, cfg_game);
        let win = (4, 0);
        assert!(game.legal_moves().contains(&win));
        let mut probe = game.clone();
        probe.apply_move(win).unwrap();
        assert!(probe.is_terminal());

        let cfg = MCTSConfig {
            n_simulations: 16,
            m_actions: 64,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            forced_candidate_capture_k: 4,
            ..Default::default()
        };

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let result = gumbel_mcts(&game, &cfg, &mut rng, None, &mut dummy_eval).unwrap();
        // The chosen action must be the win (depth-1 shortcut).
        assert_eq!(result.action, win);
        assert!(
            result.chosen_action_forced_candidates.is_empty(),
            "terminal-post-p₁ chosen action must leave captured set empty"
        );
    }

    /// At a mr=1 root with `forced_candidates=Some(coords)` containing
    /// coords that are NOT in the natural Gumbel-Top-K, the resulting
    /// `candidate_indices` must include both forced coords' indices and
    /// the lowest-Gumbel-scored non-forced candidates must be displaced.
    /// Total candidate count remains `m_actions`.
    #[test]
    fn e4_inject_forced_candidates_displace_lowest_gumbel() {
        // Build a mr=1 root. Use win_length=6, placement_radius=2 so the
        // legal set is ~30 cells (large enough that m_actions < N and
        // the lowest-Gumbel candidates are clearly distinguishable).
        let cfg_game = GameConfig {
            win_length: 6,
            placement_radius: 2,
            max_moves: 200,
        };
        // P1's opening at (0,0), one P2 stone, mr=1 means P2 has placed
        // once and is on their second placement of the turn.
        let stones = vec![
            ((0, 0), Player::P1),
            ((1, 0), Player::P2),
        ];
        let game = GameState::from_state(&stones, Player::P2, 1, cfg_game);
        assert_eq!(game.moves_remaining_this_turn(), 1);

        let legal = game.legal_moves();
        let m = 8usize;
        assert!(legal.len() >= m + 4,
            "legal moves ({}) must exceed m+4 to leave room for displacement", legal.len());

        // First run capture-off / no-forced to learn the natural
        // Gumbel-Top-K. We use the same seed to inspect the natural
        // candidate set, then construct forced coords that include both
        // a member and a non-member to test merge behaviour.
        let cfg = MCTSConfig {
            n_simulations: 8,
            m_actions: m,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let r_natural =
            gumbel_mcts(&game, &cfg, &mut rng, None, &mut dummy_eval).unwrap();
        let natural_candidates: std::collections::HashSet<Coord> =
            r_natural.candidate_indices.iter().map(|&i| r_natural.coords[i]).collect();

        // Pick 2 forced coords from outside the natural top-K, picking
        // deterministically the first two non-members from `legal`.
        let forced: Vec<Coord> = legal
            .iter()
            .copied()
            .filter(|c| !natural_candidates.contains(c))
            .take(2)
            .collect();
        assert_eq!(forced.len(), 2, "test setup: need 2 non-natural-candidate coords");

        // Re-run with the same seed but forced_candidates=Some(forced).
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let r_forced =
            gumbel_mcts(&game, &cfg, &mut rng2, Some(&forced), &mut dummy_eval).unwrap();

        // Final candidate set must include both forced coords.
        let final_candidates: std::collections::HashSet<Coord> =
            r_forced.candidate_indices.iter().map(|&i| r_forced.coords[i]).collect();
        for c in &forced {
            assert!(
                final_candidates.contains(c),
                "forced coord {:?} must appear in candidate_indices; got {:?}",
                c, final_candidates,
            );
        }

        // Total candidate count must equal m (no growth).
        assert_eq!(
            r_forced.candidate_indices.len(),
            m,
            "forced injection must preserve candidate count = m_actions"
        );
        // No duplicates in candidate_indices.
        let unique: std::collections::HashSet<usize> =
            r_forced.candidate_indices.iter().copied().collect();
        assert_eq!(unique.len(), r_forced.candidate_indices.len(),
            "candidate_indices must have no duplicates");
    }

    /// When all forced coords are already in the natural Gumbel-Top-K,
    /// passing them as `forced_candidates` must be a no-op: the final
    /// candidate set (as a SET) is unchanged. No duplicates appear.
    #[test]
    fn e4_inject_forced_candidates_already_in_top_k_no_op() {
        let cfg_game = GameConfig {
            win_length: 6,
            placement_radius: 2,
            max_moves: 200,
        };
        let stones = vec![
            ((0, 0), Player::P1),
            ((1, 0), Player::P2),
        ];
        let game = GameState::from_state(&stones, Player::P2, 1, cfg_game);
        assert_eq!(game.moves_remaining_this_turn(), 1);

        let cfg = MCTSConfig {
            n_simulations: 8,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let r_natural =
            gumbel_mcts(&game, &cfg, &mut rng, None, &mut dummy_eval).unwrap();
        let natural_candidates: std::collections::HashSet<Coord> =
            r_natural.candidate_indices.iter().map(|&i| r_natural.coords[i]).collect();

        // Pick 2 coords that ARE in the natural top-K.
        let forced: Vec<Coord> = natural_candidates.iter().copied().take(2).collect();
        assert_eq!(forced.len(), 2);

        // Re-run with the same seed but forced=Some(forced).
        let mut rng2 = ChaCha8Rng::seed_from_u64(99);
        let r_forced =
            gumbel_mcts(&game, &cfg, &mut rng2, Some(&forced), &mut dummy_eval).unwrap();

        // The set of candidates must be unchanged (the candidate_indices
        // list order may shift if implementation chooses, but coord-set
        // identity must hold).
        let final_candidates: std::collections::HashSet<Coord> =
            r_forced.candidate_indices.iter().map(|&i| r_forced.coords[i]).collect();
        assert_eq!(
            natural_candidates, final_candidates,
            "forced coords already in top-K must produce unchanged candidate SET"
        );
        // No duplicates.
        let unique: std::collections::HashSet<usize> =
            r_forced.candidate_indices.iter().copied().collect();
        assert_eq!(unique.len(), r_forced.candidate_indices.len(),
            "candidate_indices must have no duplicates");
    }

    /// Passing `forced_candidates=Some(coords)` to a mr=2 root must be a
    /// no-op (no injection, no Gumbel-draw differences). The mechanism is
    /// only meaningful for the p₂ search (the mr=1 root). Same-seed runs
    /// with and without forced_candidates must be bit-identical.
    #[test]
    fn e4_inject_forced_candidates_ignored_at_mr2() {
        let cfg_game = GameConfig {
            win_length: 6,
            placement_radius: 2,
            max_moves: 200,
        };
        let stones = vec![((0, 0), Player::P1)];
        let game = GameState::from_state(&stones, Player::P2, 2, cfg_game);
        assert_eq!(game.moves_remaining_this_turn(), 2);

        let cfg = MCTSConfig {
            n_simulations: 8,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        // Pick 2 forced coords from legal moves (arbitrary).
        let legal = game.legal_moves();
        assert!(legal.len() >= 2);
        let forced = vec![legal[0], legal[1]];

        let mut rng1 = ChaCha8Rng::seed_from_u64(55);
        let r_none = gumbel_mcts(&game, &cfg, &mut rng1, None, &mut dummy_eval).unwrap();
        let mut rng2 = ChaCha8Rng::seed_from_u64(55);
        let r_forced =
            gumbel_mcts(&game, &cfg, &mut rng2, Some(&forced), &mut dummy_eval).unwrap();

        assert_eq!(r_none.action, r_forced.action,
            "forced_candidates at mr=2 must be a no-op");
        assert_eq!(r_none.coords, r_forced.coords);
        assert_eq!(r_none.visit_counts, r_forced.visit_counts);
        assert_eq!(r_none.candidate_indices, r_forced.candidate_indices);
        for (a, b) in r_none
            .improved_policy
            .iter()
            .zip(r_forced.improved_policy.iter())
        {
            assert_eq!(a.to_bits(), b.to_bits(),
                "mr=2 improved_policy must be bit-identical regardless of forced_candidates");
        }
    }

    /// Depth-3 forced win: a win the OLD single-turn depth-2 scan could
    /// never see (it only ever checked whether THIS turn's own two
    /// placements complete the game outright), but the generalized
    /// `forcing::solve`-backed shortcut finds.
    ///
    /// Construction: a q-axis 3-run at the origin -- (0,0),(1,0),(2,0) --
    /// ready to become a forcing "open four" with one more placement, plus,
    /// far away, an r-axis 2-run and a third-hex-axis 2-run positioned so
    /// they share a single common "next" cell ((100,102)). Turn 1 spends
    /// one placement forcing the q-axis open four (which the opponent must
    /// fully block) and its other, otherwise-idle placement on that shared
    /// cell -- promoting BOTH the r-axis and third-axis runs to 3-runs at
    /// once. That recreates the classic two-3-runs-sharing-a-corner shape
    /// (`forcing::tests::mate_in_two_fork`) one turn later, at (100,102)
    /// instead of the origin, so turn 2 forks it into an unstoppable
    /// double-open-four and turn 3 completes it: depth 3 overall.
    #[test]
    fn gumbel_mcts_depth3_forced_win_returns_two_hot_policy() {
        let stones: Vec<(Coord, Player)> = [
            (0, 0), (1, 0), (2, 0),   // q-axis 3-run
            (100, 100), (100, 101),  // r-axis 2-run
            (98, 104), (99, 103),    // third-axis 2-run
        ]
        .into_iter()
        .map(|c| (c, Player::P1))
        .collect();
        let game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);

        // Verify (independently of gumbel_mcts) that this is a genuine
        // depth-3 forced win at the exact self-play budgets the shortcut
        // uses -- not depth <=2, which even a smarter single-turn check
        // might stumble onto.
        let solved = match forcing::solve(&game, SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET) {
            forcing::Outcome::Win(w) => w,
            forcing::Outcome::No => panic!("expected a forced win, got No"),
            forcing::Outcome::BudgetExceeded => panic!("expected a forced win, got BudgetExceeded"),
        };
        assert_eq!(solved.depth, 3, "test position must be exactly depth-3");

        // Change (B): mr==2 root -> the winning turn is the attacker's
        // order-invariant pair {pv[0], pv[1]}, both legal at the root here, so
        // the shortcut emits a TWO-HOT policy (0.5 each), not a one-hot.
        assert!(solved.pv.len() >= 2, "depth-3 win must have a >=2-move pv");
        let coords = game.legal_moves();
        let idx0 = coords.iter().position(|&c| c == solved.pv[0]).expect("pv[0] legal");
        let idx1 = coords.iter().position(|&c| c == solved.pv[1]).expect("pv[1] legal");
        assert_ne!(idx0, idx1);

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        for seed in 0..10 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();

            assert_eq!(
                result.action, solved.first_move,
                "seed {seed}: action should be the solver's winning first move (pv[0])"
            );
            assert!(
                (result.improved_policy[idx0] - 0.5).abs() < 1e-9,
                "seed {seed}: pv[0] should carry 0.5, got {}",
                result.improved_policy[idx0]
            );
            assert!(
                (result.improved_policy[idx1] - 0.5).abs() < 1e-9,
                "seed {seed}: pv[1] should carry 0.5, got {}",
                result.improved_policy[idx1]
            );
            for (i, &p) in result.improved_policy.iter().enumerate() {
                if i != idx0 && i != idx1 {
                    assert_eq!(
                        p, 0.0,
                        "seed {seed}: off-pair move {i} should have probability 0.0"
                    );
                }
            }
        }
    }

    /// Runtime toggle for the VCF forcing-solver shortcut. On the SAME
    /// depth-3 forced-win fixture as
    /// `gumbel_mcts_depth3_forced_win_returns_two_hot_policy` (a win ONLY
    /// `forcing::solve` can see -- the depth-1 terminal shortcut cannot
    /// fire), the default config (`disable_forcing_solver == false`) must
    /// take the solver shortcut (Change (B): a two-hot on the solver's
    /// order-invariant winning pair {pv[0], pv[1]}), while
    /// `disable_forcing_solver == true` must skip it and fall through to
    /// normal sequential-halving MCTS (a non-one-hot policy). This locks in
    /// the no-op-by-default guarantee and the toggle's effect.
    #[test]
    fn gumbel_mcts_disable_forcing_solver_skips_shortcut() {
        let stones: Vec<(Coord, Player)> = [
            (0, 0), (1, 0), (2, 0),   // q-axis 3-run
            (100, 100), (100, 101),  // r-axis 2-run
            (98, 104), (99, 103),    // third-axis 2-run
        ]
        .into_iter()
        .map(|c| (c, Player::P1))
        .collect();
        let game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);

        // Independently confirm the fixture is a genuine forced win the
        // solver catches (so disabling the solver is what changes behaviour).
        let solved = match forcing::solve(&game, SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET) {
            forcing::Outcome::Win(w) => w,
            forcing::Outcome::No => panic!("expected a forced win, got No"),
            forcing::Outcome::BudgetExceeded => panic!("expected a forced win, got BudgetExceeded"),
        };

        let base = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let on_config = MCTSConfig { disable_forcing_solver: false, ..base.clone() };
        let off_config = MCTSConfig { disable_forcing_solver: true, ..base };

        // Change (B): mr==2 root -> two-hot on the order-invariant pair.
        assert!(solved.pv.len() >= 2, "depth-3 win must have a >=2-move pv");
        let coords = game.legal_moves();
        let idx0 = coords.iter().position(|&c| c == solved.pv[0]).expect("pv[0] legal");
        let idx1 = coords.iter().position(|&c| c == solved.pv[1]).expect("pv[1] legal");

        for seed in 0..10 {
            // Solver ON (default): two-hot on the solver's winning pair.
            let mut rng_on = ChaCha8Rng::seed_from_u64(seed);
            let on = gumbel_mcts(&game, &on_config, &mut rng_on, None, &mut dummy_eval).unwrap();
            assert_eq!(
                on.action, solved.first_move,
                "seed {seed}: solver-on action should be the solver's winning first move"
            );
            assert!(
                (on.improved_policy[idx0] - 0.5).abs() < 1e-9
                    && (on.improved_policy[idx1] - 0.5).abs() < 1e-9,
                "seed {seed}: solver-on policy should be two-hot (0.5/0.5) on the winning pair"
            );

            // Solver OFF: shortcut skipped, normal MCTS runs. With uniform
            // priors and n=64/m=16 the visits spread, so the policy must NOT
            // be one-hot on any move.
            let mut rng_off = ChaCha8Rng::seed_from_u64(seed);
            let off = gumbel_mcts(&game, &off_config, &mut rng_off, None, &mut dummy_eval).unwrap();
            assert!(
                off.improved_policy.iter().all(|&p| p < 0.999),
                "seed {seed}: solver-off policy must not be one-hot (shortcut must be skipped), got {:?}",
                off.improved_policy
            );
        }
    }

    /// Runtime `forcing_depth_cap` knob + the `0 = SELF_PLAY_DEPTH_CAP`
    /// sentinel. On the SAME depth-3 forced-win fixture (a win the solver only
    /// proves at depth 3), `forcing_depth_cap: 0` resolves to the compile-time
    /// default (6) so the shortcut fires (two-hot on the winning pair), while
    /// `forcing_depth_cap: 2` caps iterative deepening below the win depth so
    /// `forcing::solve` cannot prove it and the shortcut is skipped (policy
    /// falls through to normal MCTS, not two-hot). Proves both the sentinel and
    /// that `config.forcing_depth_cap` is actually threaded into `forcing::solve`.
    #[test]
    fn gumbel_mcts_forcing_depth_cap_gates_shortcut() {
        let stones: Vec<(Coord, Player)> = [
            (0, 0), (1, 0), (2, 0),   // q-axis 3-run
            (100, 100), (100, 101),  // r-axis 2-run
            (98, 104), (99, 103),    // third-axis 2-run
        ]
        .into_iter()
        .map(|c| (c, Player::P1))
        .collect();
        let game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);

        // Independently: it is a depth-3 forced win at the default budgets, and
        // a depth cap of 2 (< 3) cannot prove it (solve returns non-Win).
        let solved = match forcing::solve(&game, SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET) {
            forcing::Outcome::Win(w) => w,
            _ => panic!("fixture must be a forced win at the default budgets"),
        };
        assert_eq!(solved.depth, 3, "fixture must be exactly depth-3");
        assert!(
            !matches!(forcing::solve(&game, 2, SELF_PLAY_NODE_BUDGET), forcing::Outcome::Win(_)),
            "a depth cap of 2 must not prove a depth-3 win"
        );

        let coords = game.legal_moves();
        let idx0 = coords.iter().position(|&c| c == solved.pv[0]).expect("pv[0] legal");
        let idx1 = coords.iter().position(|&c| c == solved.pv[1]).expect("pv[1] legal");

        let base = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        // 0 == sentinel -> SELF_PLAY_DEPTH_CAP (6): shortcut fires.
        let default_cap = MCTSConfig { forcing_depth_cap: 0, ..base.clone() };
        // 2 < 3: shortcut cannot prove the win -> skipped.
        let low_cap = MCTSConfig { forcing_depth_cap: 2, ..base };

        let is_two_hot = |p: &[f64]| {
            (p[idx0] - 0.5).abs() < 1e-9 && (p[idx1] - 0.5).abs() < 1e-9
        };

        for seed in 0..5 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let fired = gumbel_mcts(&game, &default_cap, &mut rng, None, &mut dummy_eval).unwrap();
            assert!(
                is_two_hot(&fired.improved_policy),
                "seed {seed}: forcing_depth_cap=0 (=default 6) must fire the shortcut (two-hot)"
            );

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let skipped = gumbel_mcts(&game, &low_cap, &mut rng, None, &mut dummy_eval).unwrap();
            assert!(
                !is_two_hot(&skipped.improved_policy),
                "seed {seed}: forcing_depth_cap=2 must skip the shortcut (not two-hot), got {:?}",
                skipped.improved_policy
            );
        }
    }

    /// Runtime `forcing_node_budget` knob + the `0 = SELF_PLAY_NODE_BUDGET`
    /// sentinel. On the SAME depth-3 forced-win fixture, `forcing_node_budget: 0`
    /// resolves to the compile-time default (2000) so the solve completes and
    /// the shortcut fires (two-hot), while an absurdly low `forcing_node_budget: 1`
    /// exhausts the budget before the win is proven so the shortcut is skipped.
    /// Proves the node-budget sentinel + wiring, independent of the depth knob.
    #[test]
    fn gumbel_mcts_forcing_node_budget_gates_shortcut() {
        let stones: Vec<(Coord, Player)> = [
            (0, 0), (1, 0), (2, 0),   // q-axis 3-run
            (100, 100), (100, 101),  // r-axis 2-run
            (98, 104), (99, 103),    // third-axis 2-run
        ]
        .into_iter()
        .map(|c| (c, Player::P1))
        .collect();
        let game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);

        // Independently: full budget proves the win; a 1-node budget cannot.
        let solved = match forcing::solve(&game, SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET) {
            forcing::Outcome::Win(w) => w,
            _ => panic!("fixture must be a forced win at the default budgets"),
        };
        assert!(
            !matches!(forcing::solve(&game, SELF_PLAY_DEPTH_CAP, 1), forcing::Outcome::Win(_)),
            "a 1-node budget must not prove this win"
        );

        let coords = game.legal_moves();
        let idx0 = coords.iter().position(|&c| c == solved.pv[0]).expect("pv[0] legal");
        let idx1 = coords.iter().position(|&c| c == solved.pv[1]).expect("pv[1] legal");

        let base = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        // 0 == sentinel -> SELF_PLAY_NODE_BUDGET (2000): shortcut fires.
        let default_budget = MCTSConfig { forcing_node_budget: 0, ..base.clone() };
        // 1 node: solve exhausts budget before proving -> shortcut skipped.
        let tiny_budget = MCTSConfig { forcing_node_budget: 1, ..base };

        let is_two_hot = |p: &[f64]| {
            (p[idx0] - 0.5).abs() < 1e-9 && (p[idx1] - 0.5).abs() < 1e-9
        };

        for seed in 0..5 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let fired = gumbel_mcts(&game, &default_budget, &mut rng, None, &mut dummy_eval).unwrap();
            assert!(
                is_two_hot(&fired.improved_policy),
                "seed {seed}: forcing_node_budget=0 (=default 2000) must fire the shortcut (two-hot)"
            );

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let skipped = gumbel_mcts(&game, &tiny_budget, &mut rng, None, &mut dummy_eval).unwrap();
            assert!(
                !is_two_hot(&skipped.improved_policy),
                "seed {seed}: forcing_node_budget=1 must skip the shortcut (not two-hot), got {:?}",
                skipped.improved_policy
            );
        }
    }

    /// Change (A): the forcing-solver shortcut must also fire at an mr==1 root
    /// (one placement left this turn), not only at start-of-turn (mr==2).
    ///
    /// Fixture: a "double-open-five" fork. P1 holds a q-axis 4-run
    /// (10..=13,0) and an r-axis 4-run (9,1..=4). The empty cell (9,0) is the
    /// shared extension: placing it makes TWO open fives at once (four
    /// distinct completions: (8,0),(14,0),(9,-1),(9,5)). With only one
    /// placement left this turn, the attacker plays (9,0); the defender then
    /// gets a full 2-placement turn but can block at most 2 of the 4
    /// completions, so the attacker completes next turn -- a genuine mr==1
    /// forced win (solver depth-2). Crucially (9,0) is NOT an immediate
    /// single-stone terminal win (only 5-in-a-row, win_length is 6), so the
    /// depth-1 terminal precheck does NOT fire; only `forcing::solve` sees it.
    /// Before Change (A) the gate was `== 2`, so at mr==1 the shortcut was
    /// skipped and the policy came from sequential halving (never exactly
    /// one-hot); after Change (A) it is one-hot on the single completing
    /// placement (9,0).
    #[test]
    fn gumbel_mcts_forcing_solver_fires_at_mr1_one_hot() {
        let stones: Vec<(Coord, Player)> = [
            (10, 0), (11, 0), (12, 0), (13, 0), // q-axis 4-run
            (9, 1), (9, 2), (9, 3), (9, 4),     // r-axis 4-run
        ]
        .into_iter()
        .map(|c| (c, Player::P1))
        .collect();
        // mr == 1: one placement left this turn.
        let game = GameState::from_state(&stones, Player::P1, 1, GameConfig::FULL_HEXO);
        assert_eq!(game.moves_remaining_this_turn(), 1);

        // Independently confirm: (a) it is a solver-only forced win whose
        // single completing placement is (9,0), and (b) no legal move is an
        // immediate terminal win (so the depth-1 precheck cannot fire).
        let solved = match forcing::solve(&game, SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET) {
            forcing::Outcome::Win(w) => w,
            other => panic!("fixture must be a forced win, got No={}", matches!(other, forcing::Outcome::No)),
        };
        assert_eq!(solved.first_move, (9, 0), "solver's single completing placement");
        let coords = game.legal_moves();
        assert!(coords.contains(&(9, 0)), "(9,0) must be legal");
        let any_immediate = coords.iter().any(|&c| {
            let mut g = game.clone();
            g.apply_move(c).is_ok() && g.is_terminal()
        });
        assert!(!any_immediate, "no single-stone terminal win exists (depth-1 must not fire)");

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        for seed in 0..10 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();

            assert_eq!(
                result.action, (9, 0),
                "seed {seed}: mr==1 shortcut must return the solver's completing placement"
            );
            let idx = result
                .coords
                .iter()
                .position(|&c| c == (9, 0))
                .expect("(9,0) in coords");
            assert_eq!(
                result.improved_policy[idx], 1.0,
                "seed {seed}: mr==1 completion must be one-hot (prob 1.0)"
            );
            for (i, &p) in result.improved_policy.iter().enumerate() {
                if i != idx {
                    assert_eq!(p, 0.0, "seed {seed}: non-completion move {i} must be 0.0");
                }
            }
            // Invariant: the chosen action lies inside candidate_indices.
            assert!(
                result.candidate_indices.contains(&idx),
                "seed {seed}: chosen action must be in candidate_indices"
            );
        }
    }

    /// Change (B): at an mr==2 root the winning turn is the attacker's
    /// ORDER-INVARIANT placement pair {pv[0], pv[1]}, so the forcing-solver
    /// shortcut must emit a TWO-HOT policy target (0.5 on each of pv[0]/pv[1])
    /// rather than a one-hot on pv[0]. `action` stays pv[0] (greedy
    /// determinism), and both indices seed visit_counts and candidate_indices.
    ///
    /// Fixture: the deterministic double-four fork from
    /// `forcing::tests::futile_pv_pair_is_greedy_not_canonical_on_real_board`
    /// (P1 3-runs on q and r sharing (0,0), plus a P2 stone at (-1,0) to break
    /// the symmetry). Its winning first turn plays two distinct fork cells,
    /// both legal at the root under the radius-8 config.
    #[test]
    fn gumbel_mcts_forcing_solver_two_hot_at_mr2() {
        let mut stones: Vec<(Coord, Player)> = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]
            .into_iter()
            .map(|c| (c, Player::P1))
            .collect();
        stones.push(((-1, 0), Player::P2));
        let game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);
        assert_eq!(game.moves_remaining_this_turn(), 2);

        let solved = match forcing::solve(&game, SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET) {
            forcing::Outcome::Win(w) => w,
            _ => panic!("fixture must be a forced win"),
        };
        assert!(solved.pv.len() >= 2, "two-hot requires a >=2-move pv");
        let coords = game.legal_moves();
        let idx0 = coords
            .iter()
            .position(|&c| c == solved.pv[0])
            .expect("pv[0] must be legal at root");
        let idx1 = coords
            .iter()
            .position(|&c| c == solved.pv[1])
            .expect("pv[1] must be legal at root for this fixture");
        assert_ne!(idx0, idx1, "pv[0] and pv[1] are distinct placements");

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };

        for seed in 0..10 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let result = gumbel_mcts(&game, &config, &mut rng, None, &mut dummy_eval).unwrap();

            assert_eq!(
                result.action, solved.first_move,
                "seed {seed}: action must stay pv[0] (greedy determinism)"
            );
            assert!(
                (result.improved_policy[idx0] - 0.5).abs() < 1e-9,
                "seed {seed}: pv[0] must carry 0.5, got {}",
                result.improved_policy[idx0]
            );
            assert!(
                (result.improved_policy[idx1] - 0.5).abs() < 1e-9,
                "seed {seed}: pv[1] must carry 0.5, got {}",
                result.improved_policy[idx1]
            );
            for (i, &p) in result.improved_policy.iter().enumerate() {
                if i != idx0 && i != idx1 {
                    assert_eq!(p, 0.0, "seed {seed}: off-pair move {i} must be 0.0");
                }
            }
            // Two-hot visit_counts and candidate_indices membership.
            assert_eq!(result.visit_counts[idx0], 1, "seed {seed}: pv[0] visit");
            assert_eq!(result.visit_counts[idx1], 1, "seed {seed}: pv[1] visit");
            assert!(
                result.candidate_indices.contains(&idx0)
                    && result.candidate_indices.contains(&idx1),
                "seed {seed}: both pair indices must be in candidate_indices"
            );
        }
    }

}
