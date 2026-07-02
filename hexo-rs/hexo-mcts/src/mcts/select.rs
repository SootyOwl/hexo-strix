use hexo_engine::Coord;

use super::node::MCTSNode;
use super::scoring::{QContext, sigma, softmax};

/// Select the best child action using Gumbel AlphaZero Eq. 7 (Section 5).
///
/// Algorithm (non-root selection):
/// 1. Compute completedQ for each child (visited → Q, unvisited → v_mix).
/// 2. Normalize using min-max over visited Q only.
/// 3. Compute π'(a) = softmax(log(prior) + σ(normalized_Q)).
/// 4. Select argmax_a [ π'(a) - N(a) / (1 + Σ_b N(b)) ].
///
/// # Panics
/// Panics if the node is not expanded or has no legal actions.
pub fn select_child(node: &MCTSNode, c_visit: u32, c_scale: f64) -> Coord {
    assert!(node.is_expanded(), "Cannot select child from unexpanded node");
    assert!(
        !node.child_priors.is_empty(),
        "Node has no legal actions"
    );

    let all_actions: Vec<Coord> = node.child_priors.keys().copied().collect();

    // Steps 1-2: Q normalization context
    let qctx = QContext::new(node, &all_actions);

    // Step 3: σ scores and π'

    let logits: Vec<f64> = all_actions
        .iter()
        .map(|a| {
            let prior = node.child_priors[a];
            (prior + 1e-8).ln() + sigma(qctx.norm_q[a], qctx.max_child_visits, c_visit, c_scale)
        })
        .collect();
    let pi_prime = softmax(&logits);

    // Step 4: selection criterion
    let total_n: u32 = all_actions
        .iter()
        .filter_map(|a| node.children.get(a).map(|c| c.visit_count))
        .sum();
    let total_n_f = total_n as f64;

    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, &a) in all_actions.iter().enumerate() {
        let n_a = node
            .children
            .get(&a)
            .map_or(0, |c| c.visit_count) as f64;
        let score = pi_prime[i] - n_a / (1.0 + total_n_f);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    all_actions[best_idx]
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap as HashMap;

    use super::*;
    use hexo_engine::Player;

    fn make_expanded_node(priors: &[(Coord, f64)]) -> MCTSNode {
        let mut node = MCTSNode::new(1.0, None, Player::P1);
        let child_priors: HashMap<Coord, f64> = priors.iter().cloned().collect();
        node.expand(child_priors, 0.0);
        node
    }

    #[test]
    fn equal_q_highest_prior_wins() {
        // No visits, so completedQ = v_mix for all. Highest prior should win.
        let node = make_expanded_node(&[((1, 0), 0.9), ((0, 1), 0.1)]);
        let action = select_child(&node, 50, 1.0);
        assert_eq!(action, (1, 0));
    }

    #[test]
    fn all_unvisited_degrades_to_prior_ranking() {
        let node = make_expanded_node(&[((1, 0), 0.1), ((0, 1), 0.9)]);
        let action = select_child(&node, 50, 1.0);
        assert_eq!(action, (0, 1));
    }

    #[test]
    fn unvisited_get_v_mix() {
        // One child visited with Q=1.0, one unvisited. Unvisited should get v_mix.
        let mut node = make_expanded_node(&[((1, 0), 0.5), ((0, 1), 0.5)]);
        // Create and visit child (1,0) manually
        let mut child = MCTSNode::new(0.5, Some((1, 0)), Player::P1);
        child.visit_count = 5;
        child.value_sum = 5.0; // Q = 1.0
        node.children.insert((1, 0), child);

        // select_child should still run without panicking and prefer visited good child
        let _action = select_child(&node, 50, 1.0);
        // Just verify it doesn't crash; the exact choice depends on v_mix interplay
    }

    #[test]
    #[should_panic(expected = "Cannot select child from unexpanded node")]
    fn panics_on_unexpanded() {
        let node = MCTSNode::new(1.0, None, Player::P1);
        select_child(&node, 50, 1.0);
    }

    #[test]
    fn prefers_less_visited_child() {
        // Two equal-prior children, one visited many times. Should prefer less-visited.
        let mut node = make_expanded_node(&[((1, 0), 0.5), ((0, 1), 0.5)]);

        let mut c1 = MCTSNode::new(0.5, Some((1, 0)), Player::P1);
        c1.visit_count = 100;
        c1.value_sum = 50.0; // Q = 0.5

        let mut c2 = MCTSNode::new(0.5, Some((0, 1)), Player::P1);
        c2.visit_count = 1;
        c2.value_sum = 0.5; // Q = 0.5

        node.children.insert((1, 0), c1);
        node.children.insert((0, 1), c2);

        let action = select_child(&node, 50, 1.0);
        assert_eq!(action, (0, 1));
    }

    #[test]
    fn cross_player_q_affects_selection() {
        // Parent is P1 with two children:
        //   - (1,0): P2 child with high Q (good for P2 = bad for P1)
        //   - (0,1): P1 child with low Q (modest for P1)
        // The P1-child should be preferred because the P2-child's Q gets negated
        // from P1's perspective.
        let mut node = make_expanded_node(&[((1, 0), 0.5), ((0, 1), 0.5)]);

        // P2 child: Q = 0.8 from P2's POV → -0.8 from P1's POV
        let mut c_p2 = MCTSNode::new(0.5, Some((1, 0)), Player::P2);
        c_p2.visit_count = 10;
        c_p2.value_sum = 8.0; // Q = 0.8

        // P1 child: Q = 0.3 from P1's POV → 0.3 from P1's POV (same player)
        let mut c_p1 = MCTSNode::new(0.5, Some((0, 1)), Player::P1);
        c_p1.visit_count = 10;
        c_p1.value_sum = 3.0; // Q = 0.3

        node.children.insert((1, 0), c_p2);
        node.children.insert((0, 1), c_p1);

        let action = select_child(&node, 50, 1.0);
        // (0,1) has Q=0.3 from parent POV, (1,0) has Q=-0.8 from parent POV.
        // With equal priors and equal visits, the higher Q action should win.
        assert_eq!(action, (0, 1));
    }

    #[test]
    fn three_actions_selection() {
        // 3 candidates with different priors and visit counts.
        // Verify the selection formula picks the right action.
        let mut node = make_expanded_node(&[
            ((1, 0), 0.5),
            ((0, 1), 0.3),
            ((-1, 0), 0.2),
        ]);

        // (1,0): high prior, many visits, mediocre Q
        let mut c1 = MCTSNode::new(0.5, Some((1, 0)), Player::P1);
        c1.visit_count = 20;
        c1.value_sum = 10.0; // Q = 0.5

        // (0,1): medium prior, few visits, good Q
        let mut c2 = MCTSNode::new(0.3, Some((0, 1)), Player::P1);
        c2.visit_count = 2;
        c2.value_sum = 1.6; // Q = 0.8

        // (-1,0): low prior, few visits, bad Q
        let mut c3 = MCTSNode::new(0.2, Some((-1, 0)), Player::P1);
        c3.visit_count = 2;
        c3.value_sum = -1.0; // Q = -0.5

        node.children.insert((1, 0), c1);
        node.children.insert((0, 1), c2);
        node.children.insert((-1, 0), c3);

        let action = select_child(&node, 50, 1.0);
        // (0,1) should win: decent prior, few visits (exploration bonus), high Q.
        // (1,0) is heavily visited so its pi'(a) - N(a)/(1+total_N) term is penalized.
        assert_eq!(action, (0, 1));
    }
}
