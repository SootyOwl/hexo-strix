use rustc_hash::FxHashMap as HashMap;

use hexo_engine::{Coord, Player};

use super::node::MCTSNode;

/// Return child's Q-value from the parent's perspective.
///
/// Child nodes store Q from their own perspective. When parent and child have
/// different current players, the Q must be negated. When same player
/// (mid-turn in HeXO's 2-placement turns), Q is used as-is.
pub fn q_from_parent(child: &MCTSNode, parent_player: Player) -> f64 {
    let q = child.q_value();
    if child.current_player != parent_player {
        -q
    } else {
        q
    }
}

/// Score transformation σ(q) = (c_visit + max_child_visits) × c_scale × q
///
/// Gumbel AlphaZero (Danihelka et al., 2022), used in both root action
/// selection (Sequential Halving) and non-root selection (Section 5).
pub fn sigma(q: f64, max_child_visits: u32, c_visit: u32, c_scale: f64) -> f64 {
    (c_visit as f64 + max_child_visits as f64) * c_scale * q
}

/// Min-max normalize Q-values to [0, 1].
///
/// If `q_min == q_max` (all equal or single value), every entry maps to 0.0.
/// Values outside [q_min, q_max] are clamped.
pub fn normalize_q(
    q_values: &HashMap<Coord, f64>,
    q_min: f64,
    q_max: f64,
) -> HashMap<Coord, f64> {
    if q_values.is_empty() {
        return HashMap::default();
    }
    if !(q_max > q_min) {
        return q_values.keys().map(|&a| (a, 0.0)).collect();
    }
    let span = q_max - q_min;
    q_values
        .iter()
        .map(|(&a, &q)| (a, ((q - q_min) / span).clamp(0.0, 1.0)))
        .collect()
}

/// Mixed value combining network estimate with empirical Q (Eq. 33).
///
/// v_mix = (v_hat + total_N / sum_visited_prior × Σ_visited(π(a)·Q(a))) / (1 + total_N)
///
/// Q-values are read from the parent's perspective.
/// When no children have been visited, returns `value_estimate`.
pub fn v_mix(
    value_estimate: f64,
    children: &HashMap<Coord, MCTSNode>,
    parent_player: Player,
) -> f64 {
    let total_n: u32 = children.values().map(|c| c.visit_count).sum();
    if total_n == 0 {
        return value_estimate;
    }

    let mut sum_visited_prior = 0.0;
    let mut sum_pi_q = 0.0;
    for child in children.values() {
        if child.visit_count > 0 {
            sum_visited_prior += child.prior;
            sum_pi_q += child.prior * q_from_parent(child, parent_player);
        }
    }

    if sum_visited_prior < 1e-8 {
        return value_estimate;
    }

    let numerator = value_estimate + (total_n as f64 / sum_visited_prior) * sum_pi_q;
    numerator / (1.0 + total_n as f64)
}

/// Pre-computed Q-normalization context for a node's children.
///
/// Encapsulates the common pattern of:
/// 1. Computing v_mix for unvisited children
/// 2. Building completed_q (visited → Q from parent, unvisited → v_mix)
/// 3. Computing q_min/q_max from visited children only
/// 4. Normalizing via min-max
/// 5. Finding max child visits
pub struct QContext {
    pub completed_q: HashMap<Coord, f64>,
    pub norm_q: HashMap<Coord, f64>,
    pub max_child_visits: u32,
    pub vmix_val: f64,
}

impl QContext {
    /// Build Q context for a node's children from the parent's perspective.
    /// `all_actions` is the full set of actions to consider.
    pub fn new(node: &MCTSNode, all_actions: &[Coord]) -> Self {
        let parent_player = node.current_player;

        // Step 1: v_mix
        let vmix_val = v_mix(node.value_estimate, &node.children, parent_player);

        // Step 2: completed_q (pre-sized to avoid HashMap rehash during the loop)
        let mut completed_q: HashMap<Coord, f64> =
            HashMap::with_capacity_and_hasher(all_actions.len(), Default::default());
        for &a in all_actions {
            if let Some(child) = node.children.get(&a) {
                if child.visit_count > 0 {
                    completed_q.insert(a, q_from_parent(child, parent_player));
                    continue;
                }
            }
            completed_q.insert(a, vmix_val);
        }

        // Step 3: q_min/q_max from visited children only
        let visited_qs: Vec<f64> = all_actions
            .iter()
            .filter_map(|a| {
                node.children
                    .get(a)
                    .filter(|c| c.visit_count > 0)
                    .map(|c| q_from_parent(c, parent_player))
            })
            .collect();

        let (q_min, q_max) = if visited_qs.is_empty() {
            (0.0, 0.0)
        } else {
            (
                visited_qs.iter().cloned().fold(f64::INFINITY, f64::min),
                visited_qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            )
        };

        // Step 4: normalize
        let norm_q = normalize_q(&completed_q, q_min, q_max);

        // Step 5: max child visits
        let max_child_visits = all_actions
            .iter()
            .filter_map(|a| node.children.get(a).map(|c| c.visit_count))
            .max()
            .unwrap_or(0);

        Self {
            completed_q,
            norm_q,
            max_child_visits,
            vmix_val,
        }
    }
}

/// Numerically stable softmax over a slice of logits.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_basic() {
        // (50 + 10) * 1.0 * 0.5 = 30.0
        assert!((sigma(0.5, 10, 50, 1.0) - 30.0).abs() < 1e-10);
    }

    #[test]
    fn sigma_zero_q() {
        assert_eq!(sigma(0.0, 100, 50, 1.0), 0.0);
    }

    #[test]
    fn sigma_custom_scale() {
        // (50 + 0) * 2.0 * 1.0 = 100.0
        assert!((sigma(1.0, 0, 50, 2.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_q_basic() {
        let q = [((0, 0), 0.0), ((1, 0), 1.0)].into_iter().collect::<HashMap<_,_>>();
        let norm = normalize_q(&q, 0.0, 1.0);
        assert!((norm[&(0, 0)] - 0.0).abs() < 1e-10);
        assert!((norm[&(1, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_q_equal_min_max() {
        let q = [((0, 0), 0.5), ((1, 0), 0.5)].into_iter().collect::<HashMap<_,_>>();
        let norm = normalize_q(&q, 0.5, 0.5);
        assert_eq!(norm[&(0, 0)], 0.0);
        assert_eq!(norm[&(1, 0)], 0.0);
    }

    #[test]
    fn normalize_q_empty() {
        let q: HashMap<Coord, f64> = HashMap::default();
        let norm = normalize_q(&q, 0.0, 1.0);
        assert!(norm.is_empty());
    }

    #[test]
    fn normalize_q_clamps() {
        let q = [((0, 0), -1.0), ((1, 0), 2.0)].into_iter().collect::<HashMap<_,_>>();
        let norm = normalize_q(&q, 0.0, 1.0);
        assert_eq!(norm[&(0, 0)], 0.0);
        assert_eq!(norm[&(1, 0)], 1.0);
    }

    #[test]
    fn v_mix_no_visits() {
        let children: HashMap<Coord, MCTSNode> = HashMap::default();
        assert_eq!(v_mix(0.5, &children, Player::P1), 0.5);
    }

    #[test]
    fn v_mix_with_visits_same_player() {
        // priors=[0.8, 0.2], N=[3, 1], Q=[1.0, 0.0], v_hat=0.5
        // Same player as parent → no negation
        // total_N = 4, sum_visited_prior = 1.0
        // sum_pi_q = 0.8*1.0 + 0.2*0.0 = 0.8
        // v_mix = (0.5 + 4/1.0 * 0.8) / (1 + 4) = (0.5 + 3.2) / 5 = 0.74
        let mut c1 = MCTSNode::new(0.8, Some((1, 0)), Player::P1);
        c1.visit_count = 3;
        c1.value_sum = 3.0; // Q = 1.0
        let mut c2 = MCTSNode::new(0.2, Some((0, 1)), Player::P1);
        c2.visit_count = 1;
        c2.value_sum = 0.0; // Q = 0.0

        let children: HashMap<Coord, MCTSNode> = [            ((1, 0), c1),
            ((0, 1), c2),].into_iter().collect::<HashMap<_,_>>();
        let result = v_mix(0.5, &children, Player::P1);
        assert!((result - 0.74).abs() < 1e-10);
    }

    #[test]
    fn v_mix_different_player() {
        // Child has different player → Q negated
        // prior=1.0, N=2, Q=0.5 from child's POV → -0.5 from parent's POV
        // v_mix = (0.0 + 2/1.0 * (1.0 * -0.5)) / (1 + 2) = (0.0 - 1.0) / 3 = -1/3
        let mut c = MCTSNode::new(1.0, Some((1, 0)), Player::P2);
        c.visit_count = 2;
        c.value_sum = 1.0; // Q = 0.5

        let children: HashMap<Coord, MCTSNode> =
            [((1, 0), c)].into_iter().collect::<HashMap<_,_>>();
        let result = v_mix(0.0, &children, Player::P1);
        assert!((result - (-1.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn v_mix_partial_visits() {
        // Two children, only one visited. sum_visited_prior uses only visited child.
        let mut c1 = MCTSNode::new(0.6, Some((1, 0)), Player::P1);
        c1.visit_count = 4;
        c1.value_sum = 2.0; // Q = 0.5
        let c2 = MCTSNode::new(0.4, Some((0, 1)), Player::P1);
        // c2 unvisited

        let children: HashMap<Coord, MCTSNode> = [            ((1, 0), c1),
            ((0, 1), c2),].into_iter().collect::<HashMap<_,_>>();
        // total_N = 4, sum_visited_prior = 0.6 (only c1)
        // sum_pi_q = 0.6 * 0.5 = 0.3
        // v_mix = (0.1 + 4/0.6 * 0.3) / (1 + 4) = (0.1 + 2.0) / 5 = 0.42
        let result = v_mix(0.1, &children, Player::P1);
        assert!((result - 0.42).abs() < 1e-10);
    }

    #[test]
    fn softmax_uniform() {
        let result = softmax(&[0.0, 0.0, 0.0]);
        for &v in &result {
            assert!((v - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let result = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn softmax_numerical_stability() {
        let result = softmax(&[1000.0, 1000.0, 1000.0]);
        for &v in &result {
            assert!((v - 1.0 / 3.0).abs() < 1e-10);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn softmax_empty() {
        let result = softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn q_from_parent_same_player() {
        let mut child = MCTSNode::new(0.5, Some((1, 0)), Player::P1);
        child.visit_count = 2;
        child.value_sum = 1.0; // Q = 0.5
        assert!((q_from_parent(&child, Player::P1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn q_from_parent_different_player() {
        let mut child = MCTSNode::new(0.5, Some((1, 0)), Player::P2);
        child.visit_count = 2;
        child.value_sum = 1.0; // Q = 0.5, negated → -0.5
        assert!((q_from_parent(&child, Player::P1) - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn v_mix_zero_prior_guard() {
        // Visited children with near-zero priors (sum < 1e-8) should return value_estimate.
        // This guards against division by ~0 in the v_mix formula.
        let mut c1 = MCTSNode::new(1e-12, Some((1, 0)), Player::P1);
        c1.visit_count = 3;
        c1.value_sum = 1.5; // Q = 0.5
        let mut c2 = MCTSNode::new(1e-12, Some((0, 1)), Player::P1);
        c2.visit_count = 2;
        c2.value_sum = -1.0; // Q = -0.5

        let children: HashMap<Coord, MCTSNode> = [            ((1, 0), c1),
            ((0, 1), c2),].into_iter().collect::<HashMap<_,_>>();
        let val_est = 0.42;
        let result = v_mix(val_est, &children, Player::P1);
        // sum_visited_prior = 2e-12 < 1e-8, so should return value_estimate
        assert_eq!(result, val_est);
    }

    #[test]
    fn q_from_parent_unvisited() {
        // Child with visit_count=0 should return 0.0 regardless of player,
        // because q_value() returns 0.0 for unvisited nodes.
        let child_same = MCTSNode::new(0.5, Some((1, 0)), Player::P1);
        assert_eq!(child_same.visit_count, 0);
        assert_eq!(q_from_parent(&child_same, Player::P1), 0.0);

        let child_diff = MCTSNode::new(0.5, Some((0, 1)), Player::P2);
        assert_eq!(child_diff.visit_count, 0);
        // Even with negation, -0.0 == 0.0
        assert_eq!(q_from_parent(&child_diff, Player::P1), 0.0);
    }
}
