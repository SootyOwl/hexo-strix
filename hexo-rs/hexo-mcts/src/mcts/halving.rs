use hexo_engine::Coord;
use rand::Rng;
use rand::distr::{Distribution, Uniform};

use super::node::MCTSNode;
use super::scoring::{QContext, sigma};

/// Sample Gumbel(0,1) noise and return top-min(m, n) action indices.
///
/// Returns `(candidate_indices, gumbel_samples)` where:
/// - `candidate_indices` are indices into the actions array, sorted by descending
///   score (gumbel + logit).
/// - `gumbel_samples` are the full Gumbel noise values for all actions.
///
/// When `noise` is `false`, the random draw is skipped entirely: every Gumbel
/// is `0.0`, so candidates collapse to the top-k actions by logit and the RNG
/// is not advanced. Used at evaluation/play to get the paper's deterministic,
/// zero-noise action selection.
pub fn gumbel_top_k<R: Rng>(
    logits: &[f64],
    m: usize,
    rng: &mut R,
    noise: bool,
) -> (Vec<usize>, Vec<f64>) {
    let n = logits.len();
    let k = m.min(n);

    let gumbels: Vec<f64> = if noise {
        let uniform = Uniform::new(1e-20, 1.0 - 1e-20).unwrap();
        (0..n)
            .map(|_| {
                let u: f64 = uniform.sample(rng);
                -(-u.ln()).ln()
            })
            .collect()
    } else {
        vec![0.0; n]
    };

    let mut scores: Vec<(usize, f64)> = gumbels
        .iter()
        .zip(logits.iter())
        .enumerate()
        .map(|(i, (&g, &l))| (i, g + l))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let candidates: Vec<usize> = scores.iter().take(k).map(|&(i, _)| i).collect();

    (candidates, gumbels)
}

/// Compute the improved policy distribution from a completed search.
///
/// π_improved(a) = softmax(logits(a) + σ(normalized_completedQ(a)))
///
/// Q-values are read from the root's perspective. Unvisited actions get v_mix.
pub fn compute_improved_policy(
    logits: &[f64],
    coords: &[Coord],
    root: &MCTSNode,
    c_visit: u32,
    c_scale: f64,
) -> Vec<f64> {
    let qctx = QContext::new(root, coords);

    let log_scores: Vec<f64> = coords
        .iter()
        .zip(logits.iter())
        .map(|(coord, logit)| logit + sigma(qctx.norm_q[coord], qctx.max_child_visits, c_visit, c_scale))
        .collect();

    // Step 4: softmax
    super::scoring::softmax(&log_scores)
}

use super::MCTSConfig;

/// Allocate simulations across candidates using Sequential Halving.
///
/// Each phase distributes simulations to surviving candidates. After each phase,
/// the bottom half (by Gumbel score) is eliminated, down to a final pair. Any
/// budget left over from floor rounding is spent in extra rounds over that pair
/// (mctx parity), so exactly `n_simulations` are used whenever possible.
///
/// The evaluation callback `eval_fn` receives a list of game states that need
/// network evaluation and returns `(priors_per_state, values)`.
///
/// Returns the surviving candidate indices (the final pair, best-first); the
/// caller picks the winner by argmax of gumbel + logit + σ(Q).
pub fn sequential_halving<F>(
    root: &mut MCTSNode,
    candidate_indices: &[usize],
    candidate_gumbels: &[f64],
    coords: &[Coord],
    logits: &[f64],
    config: &MCTSConfig,
    eval_fn: &mut F,
) -> Vec<usize>
where
    F: FnMut(&mut MCTSNode, &[Coord], u32, f64) -> (),
{
    let n_simulations = config.n_simulations;
    let c_visit = config.c_visit;
    let c_scale = config.c_scale;

    if n_simulations == 0 {
        return candidate_indices.to_vec();
    }
    if candidate_indices.len() <= 1 {
        return candidate_indices.to_vec();
    }

    let num_phases = (candidate_indices.len() as f64).log2().ceil() as u32;
    let mut remaining: Vec<usize> = candidate_indices.to_vec();
    let mut total_sims_used: u32 = 0;

    // mctx parity (seq_halving.get_sequence_of_considered_visits): loop until
    // the budget is exactly spent, never shrinking the candidate set below the
    // final pair. Leftover sims from the floor in `sims_per_action` therefore
    // top up the last head-to-head instead of being dropped. The final winner
    // is picked by the caller's argmax over the returned survivors.
    while total_sims_used < n_simulations && remaining.len() > 1 {
        let sims_per_action =
            (n_simulations / (num_phases * remaining.len() as u32)).max(1);

        // Two paths:
        //
        //   `virtual_loss > 0.0` — fused inner loop. Collect
        //   `sims_per_action × remaining.len()` leaves into a single batch
        //   per phase, with VL applied between leaf selections so descents
        //   diversify. Reduces `eval_fn` dispatches by `sims_per_action`x.
        //
        //   `virtual_loss == 0.0` — original serial loop. Each iteration
        //   submits one batch of `remaining.len()` graphs to `eval_fn` and
        //   relies on real Q-backups from prior iterations to drive
        //   descent diversification.
        //
        // Clean 2026-05-13 same-binary A/B at production batcher settings
        // (mb=128, t=5ms, 31 workers, sims=128, full HeXO, n=50 games):
        // fusion alone +6% games/s (within noise), fusion + vl=0.5 +19%
        // games/s. Moves/game trended longer under fusion but n=50 is too
        // small to call it. Path gated pending SPRT vs serial.
        //
        // NOTE: `virtual_loss` is EFFECTIVELY BOOLEAN — only its sign matters,
        // not its magnitude (verified 2026-06-05, docs/research/2026-06-05-vl-
        // fidelity). `apply_virtual_loss` diversifies via a magnitude-blind
        // `visit_count += 1` feeding the Gumbel Eq.7 count penalty
        // `N(a)/(1+sum_N)` (see select.rs); the `value_sum -= magnitude` channel
        // is washed out by min-max Q normalisation + tiny sigma at low visits.
        // So fused fidelity-vs-serial is a step function, flat across vl 0.001..25.
        // (Unlike PUCT, where VL magnitude tunes parallel-descent repulsion.)
        if config.virtual_loss > 0.0 {
            let mut batch_actions: Vec<Coord> =
                Vec::with_capacity(sims_per_action as usize * remaining.len());
            'budget: for _ in 0..sims_per_action {
                for &idx in &remaining {
                    if total_sims_used >= n_simulations {
                        break 'budget;
                    }
                    batch_actions.push(coords[idx]);
                    total_sims_used += 1;
                }
            }
            if !batch_actions.is_empty() {
                eval_fn(root, &batch_actions, c_visit, c_scale);
            }
        } else {
            for _ in 0..sims_per_action {
                if total_sims_used >= n_simulations {
                    break;
                }
                let mut batch_actions: Vec<Coord> = Vec::with_capacity(remaining.len());
                for &idx in &remaining {
                    if total_sims_used >= n_simulations {
                        break;
                    }
                    batch_actions.push(coords[idx]);
                    total_sims_used += 1;
                }
                if !batch_actions.is_empty() {
                    eval_fn(root, &batch_actions, c_visit, c_scale);
                }
            }
        }

        // Eliminate bottom half by Gumbel score.
        let qctx = QContext::new(root, coords);

        remaining.sort_by(|&a, &b| {
            let score_a = gumbel_score(
                a, coords, candidate_gumbels, logits, &qctx, c_visit, c_scale,
            );
            let score_b = gumbel_score(
                b, coords, candidate_gumbels, logits, &qctx, c_visit, c_scale,
            );
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let keep = ((remaining.len() + 1) / 2).max(2); // ceil(len/2), floor at the final pair
        remaining.truncate(keep);
    }

    remaining
}

fn gumbel_score(
    idx: usize,
    coords: &[Coord],
    gumbels: &[f64],
    logits: &[f64],
    qctx: &QContext,
    c_visit: u32,
    c_scale: f64,
) -> f64 {
    let action = coords[idx];
    let norm_q = qctx.norm_q[&action];
    gumbels[idx] + logits[idx] + sigma(norm_q, qctx.max_child_visits, c_visit, c_scale)
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap as HashMap;

    use super::*;
    use hexo_engine::Player;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn gumbel_top_k_returns_min_m_n() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let logits = vec![1.0, 2.0, 3.0];
        let (candidates, gumbels) = gumbel_top_k(&logits, 5, &mut rng, true);
        assert_eq!(candidates.len(), 3); // min(5, 3)
        assert_eq!(gumbels.len(), 3);
    }

    #[test]
    fn gumbel_top_k_deterministic() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        let (c1, g1) = gumbel_top_k(&logits, 2, &mut rng1, true);
        let (c2, g2) = gumbel_top_k(&logits, 2, &mut rng2, true);
        assert_eq!(c1, c2);
        assert_eq!(g1, g2);
    }

    #[test]
    fn gumbel_top_k_sorted_descending() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (candidates, gumbels) = gumbel_top_k(&logits, 3, &mut rng, true);

        // Verify candidates are sorted by descending gumbel+logit score
        for i in 0..candidates.len() - 1 {
            let score_a = gumbels[candidates[i]] + logits[candidates[i]];
            let score_b = gumbels[candidates[i + 1]] + logits[candidates[i + 1]];
            assert!(score_a >= score_b);
        }
    }

    #[test]
    fn gumbel_top_k_noise_off_is_deterministic_top_logit() {
        // noise = false: all Gumbels zero, candidates = top-k by logit, and the
        // RNG is never advanced (so two different seeds agree).
        let logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let mut rng_a = ChaCha8Rng::seed_from_u64(1);
        let mut rng_b = ChaCha8Rng::seed_from_u64(999);
        let (cand_a, gumbels_a) = gumbel_top_k(&logits, 3, &mut rng_a, false);
        let (cand_b, _) = gumbel_top_k(&logits, 3, &mut rng_b, false);

        assert!(gumbels_a.iter().all(|&g| g == 0.0));
        assert_eq!(cand_a, vec![1, 3, 4]); // indices of logits 5,4,3
        assert_eq!(cand_a, cand_b); // seed-independent: RNG not consumed
    }

    #[test]
    fn compute_improved_policy_sums_to_one() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1), (-1, 0)];
        let logits = vec![1.0, 2.0, 0.5];
        let priors: HashMap<Coord, f64> =
            coords.iter().map(|&c| (c, 1.0 / 3.0)).collect();
        root.expand(priors, 0.5);

        let policy = compute_improved_policy(&logits, &coords, &root, 50, 1.0);
        let sum: f64 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compute_improved_policy_length_matches_coords() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1)];
        let logits = vec![1.0, 2.0];
        let priors: HashMap<Coord, f64> = coords.iter().map(|&c| (c, 0.5)).collect();
        root.expand(priors, 0.0);

        let policy = compute_improved_policy(&logits, &coords, &root, 50, 1.0);
        assert_eq!(policy.len(), coords.len());
    }

    #[test]
    fn sequential_halving_zero_budget() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let config = MCTSConfig {
            n_simulations: 0,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let candidates = vec![0, 1, 2, 3];
        let gumbels = vec![1.0, 2.0, 3.0, 4.0];
        let coords = vec![(1, 0), (0, 1), (-1, 0), (0, -1)];
        let logits = vec![0.0; 4];

        let result = sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, _, _, _| {},
        );
        assert_eq!(result, candidates);
    }

    #[test]
    fn sequential_halving_single_candidate() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let config = MCTSConfig {
            n_simulations: 16,
            m_actions: 1,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let candidates = vec![0];
        let gumbels = vec![1.0];
        let coords = vec![(1, 0)];
        let logits = vec![0.0];

        let result = sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, _, _, _| {},
        );
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn sequential_halving_eliminates() {
        // With 4 candidates and non-zero budget, should end with fewer
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1), (-1, 0), (0, -1)];
        let priors: HashMap<Coord, f64> = coords.iter().map(|&c| (c, 0.25)).collect();
        root.expand(priors, 0.0);

        let config = MCTSConfig {
            n_simulations: 16,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let candidates = vec![0, 1, 2, 3];
        // Different gumbels so elimination has a preference
        let gumbels = vec![4.0, 3.0, 2.0, 1.0];
        let logits = vec![0.0; 4];

        // Dummy eval_fn that doesn't actually evaluate (just for testing structure)
        let result = sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, _, _, _| {},
        );
        assert!(result.len() < candidates.len());
        assert!(!result.is_empty());
    }

    #[test]
    fn sequential_halving_respects_budget() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1), (-1, 0), (0, -1)];
        let priors: HashMap<Coord, f64> = coords.iter().map(|&c| (c, 0.25)).collect();
        root.expand(priors, 0.0);

        let config = MCTSConfig {
            n_simulations: 4, // very small budget
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let candidates = vec![0, 1, 2, 3];
        let gumbels = vec![1.0; 4];
        let logits = vec![0.0; 4];

        let mut total_eval_calls = 0u32;
        sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, actions, _, _| {
                total_eval_calls += actions.len() as u32;
            },
        );
        assert!(total_eval_calls <= 4);
    }

    #[test]
    fn sequential_halving_spends_full_budget_and_tops_up_final_pair() {
        // mctx parity: when n is not divisible by the phase arithmetic, the
        // leftover budget goes to extra rounds over the final pair instead of
        // being dropped. n=200/m=16: phases spend 3/6/12/25 per survivor
        // (194 sims); the remaining 6 add 3 more visits to each finalist.
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords: Vec<Coord> = (0..16).map(|i| (i, 0)).collect();
        let priors: HashMap<Coord, f64> =
            coords.iter().map(|&c| (c, 1.0 / 16.0)).collect();
        root.expand(priors, 0.0);

        let config = MCTSConfig {
            n_simulations: 200,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let candidates: Vec<usize> = (0..16).collect();
        // Descending gumbels: with a no-op eval_fn there are no Q updates,
        // so elimination preserves this order and candidates 0/1 are the pair.
        let gumbels: Vec<f64> = (0..16).map(|i| (16 - i) as f64).collect();
        let logits = vec![0.0; 16];

        let mut total = 0u32;
        let mut per_action: HashMap<Coord, u32> = HashMap::default();
        sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, actions, _, _| {
                total += actions.len() as u32;
                for &a in actions {
                    *per_action.entry(a).or_insert(0) += 1;
                }
            },
        );

        assert_eq!(total, 200, "full simulation budget must be spent");
        assert_eq!(per_action[&coords[0]], 49, "finalist gets 46 + 3 top-up");
        assert_eq!(per_action[&coords[1]], 49, "finalist gets 46 + 3 top-up");
    }

    #[test]
    fn sequential_halving_top_up_fused_path_matches_serial_budget() {
        // The virtual-loss fused path must spend the identical budget.
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords: Vec<Coord> = (0..16).map(|i| (i, 0)).collect();
        let priors: HashMap<Coord, f64> =
            coords.iter().map(|&c| (c, 1.0 / 16.0)).collect();
        root.expand(priors, 0.0);

        let config = MCTSConfig {
            n_simulations: 200,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.5,
            ..Default::default()
        };
        let candidates: Vec<usize> = (0..16).collect();
        let gumbels: Vec<f64> = (0..16).map(|i| (16 - i) as f64).collect();
        let logits = vec![0.0; 16];

        let mut total = 0u32;
        sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, actions, _, _| {
                total += actions.len() as u32;
            },
        );
        assert_eq!(total, 200, "fused path must spend the full budget too");
    }

    #[test]
    fn compute_improved_policy_favors_high_q() {
        // Root with two children: one visited with Q=1.0, one with Q=-1.0.
        // The improved policy should put more weight on the high-Q action.
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1)];
        let logits = vec![0.0, 0.0]; // equal logits so only Q matters
        let priors: HashMap<Coord, f64> = coords.iter().map(|&c| (c, 0.5)).collect();
        root.expand(priors, 0.0);

        // Child (1,0): same player, Q=1.0
        let mut c_good = MCTSNode::new(0.5, Some((1, 0)), Player::P1);
        c_good.visit_count = 10;
        c_good.value_sum = 10.0; // Q = 1.0

        // Child (0,1): same player, Q=-1.0
        let mut c_bad = MCTSNode::new(0.5, Some((0, 1)), Player::P1);
        c_bad.visit_count = 10;
        c_bad.value_sum = -10.0; // Q = -1.0

        root.children.insert((1, 0), c_good);
        root.children.insert((0, 1), c_bad);

        let policy = compute_improved_policy(&logits, &coords, &root, 50, 1.0);
        // coords[0] = (1,0) has Q=1.0, coords[1] = (0,1) has Q=-1.0
        // With equal logits, the high-Q action should dominate the policy.
        assert!(
            policy[0] > policy[1],
            "Policy for Q=1.0 action ({}) should exceed Q=-1.0 action ({})",
            policy[0],
            policy[1],
        );
        // The difference should be substantial, not just a rounding artifact
        assert!(policy[0] > 0.9, "High-Q action should dominate: {}", policy[0]);
    }
}
