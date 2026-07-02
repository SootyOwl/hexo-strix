//! Batched Gumbel MCTS: run N searches in lockstep, batching eval calls.
//!
//! Instead of running N independent MCTS searches (each calling eval_fn
//! separately), this module synchronises them so leaf evaluations from
//! all N searches are collected into a single eval_fn call per halving round.

use rustc_hash::FxHashMap as HashMap;

use hexo_engine::{Coord, GameState};
use rand::Rng;

use super::MCTSConfig;
use super::halving::{compute_improved_policy, gumbel_top_k};
use super::node::MCTSNode;
use super::scoring::{QContext, sigma};
use super::simulate::{complete_simulation, simulate_select};

/// Result for one game's MCTS search.
pub struct BatchedMCTSResult {
    pub action: Coord,
    pub improved_policy: Vec<f64>,
    pub coords: Vec<Coord>,
}

/// Per-search state held between phases.
struct SearchState {
    root: MCTSNode,
    coords: Vec<Coord>,
    logits: Vec<f64>,
    candidate_indices: Vec<usize>,
    gumbel_samples: Vec<f64>,
    remaining: Vec<usize>,
}

/// Run Gumbel MCTS for N game states simultaneously, batching all eval calls.
///
/// The `eval_fn` callback is called with states from ALL N searches combined,
/// so the GPU processes one large batch instead of N small ones.
pub fn batched_gumbel_mcts<R, F>(
    games: &[GameState],
    config: &MCTSConfig,
    rng: &mut R,
    eval_fn: &mut F,
) -> Result<Vec<BatchedMCTSResult>, &'static str>
where
    R: Rng,
    F: FnMut(&[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>),
{
    let n = games.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // --- Phase 1: Batch root evaluation ---
    // Collect all N root states into one eval call.
    let non_terminal: Vec<usize> = (0..n)
        .filter(|&i| !games[i].is_terminal())
        .collect();

    if non_terminal.is_empty() {
        return Err("All games are terminal");
    }

    let root_states: Vec<GameState> = non_terminal.iter().map(|&i| games[i].clone()).collect();
    let (all_logits, all_values) = eval_fn(&root_states);

    // --- Phase 2: Initialize search state for each game ---
    let mut searches: Vec<SearchState> = Vec::with_capacity(non_terminal.len());

    for (eval_idx, &game_idx) in non_terminal.iter().enumerate() {
        let game = &games[game_idx];
        let legal_moves = game.legal_moves();
        let root_logits_map = &all_logits[eval_idx];
        let root_value = all_values[eval_idx];

        let coords: Vec<Coord> = legal_moves.clone();
        let logits: Vec<f64> = coords
            .iter()
            .map(|c| root_logits_map.get(c).copied().unwrap_or(0.0))
            .collect();

        let priors_vec = super::scoring::softmax(&logits);
        let root_priors: HashMap<Coord, f64> = coords
            .iter()
            .zip(priors_vec.iter())
            .map(|(&c, &p)| (c, p))
            .collect();

        let current_player = game
            .current_player()
            .expect("Non-terminal game should have current player");

        let mut root = MCTSNode::new(1.0, None, current_player);
        root.game_state = Some(game.clone());
        root.expand(root_priors, root_value);

        let (candidate_indices, gumbel_samples) =
            gumbel_top_k(&logits, config.m_actions, rng, !config.disable_gumbel_noise);
        let remaining = candidate_indices.clone();

        searches.push(SearchState {
            root,
            coords,
            logits,
            candidate_indices,
            gumbel_samples,
            remaining,
        });
    }

    // --- Phase 3: Sequential halving in lockstep ---
    let n_simulations = config.n_simulations;
    let c_visit = config.c_visit;
    let c_scale = config.c_scale;

    if n_simulations > 0 {
        let mut sims_used: Vec<u32> = vec![0; searches.len()];

        // mctx parity: iterate until every search has spent its full budget,
        // never shrinking a candidate set below the final pair. Leftover sims
        // from the floor in `sims_per_action` top up the last head-to-head
        // instead of being dropped (mirrors halving.rs::sequential_halving).
        while searches
            .iter()
            .enumerate()
            .any(|(i, s)| sims_used[i] < n_simulations && s.remaining.len() > 1)
        {
            // Compute sims_per_action for each search
            let num_phases_per_search: Vec<u32> = searches
                .iter()
                .map(|s| {
                    if s.remaining.len() <= 1 {
                        1
                    } else {
                        (s.candidate_indices.len() as f64).log2().ceil() as u32
                    }
                })
                .collect();

            let sims_per_action: Vec<u32> = searches
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    if s.remaining.len() <= 1 || sims_used[i] >= n_simulations {
                        0
                    } else {
                        (n_simulations / (num_phases_per_search[i] * s.remaining.len() as u32)).max(1)
                    }
                })
                .collect();

            let max_sims = sims_per_action.iter().copied().max().unwrap_or(0);

            for _sim_round in 0..max_sims {
                // Collect all pending leaf states across all searches
                struct PendingLeaf {
                    search_idx: usize,
                    selection: super::simulate::LeafSelection,
                }

                let mut pending_states: Vec<GameState> = Vec::new();
                let mut pending_leaves: Vec<PendingLeaf> = Vec::new();

                for (si, search) in searches.iter_mut().enumerate() {
                    if sims_used[si] >= n_simulations || search.remaining.len() <= 1 {
                        continue;
                    }
                    if _sim_round >= sims_per_action[si] {
                        continue;
                    }

                    // For each remaining candidate, run simulate_select
                    for &idx in &search.remaining.clone() {
                        if sims_used[si] >= n_simulations {
                            break;
                        }
                        let action = search.coords[idx];
                        sims_used[si] += 1;

                        match simulate_select(&mut search.root, Some(action), c_visit, c_scale) {
                            None => {
                                // Terminal leaf — backup already applied
                            }
                            Some(selection) => {
                                // Get leaf game state
                                let mut node: &MCTSNode = &search.root;
                                for &a in &selection.path_actions {
                                    node = node.children.get(&a).expect("path broken");
                                }
                                if let Some(game_state) = &node.game_state {
                                    pending_states.push(game_state.clone());
                                    pending_leaves.push(PendingLeaf {
                                        search_idx: si,
                                        selection,
                                    });
                                }
                            }
                        }
                    }
                }

                // Batch evaluate ALL pending leaves from ALL searches
                if !pending_states.is_empty() {
                    let (logits_list, values) = eval_fn(&pending_states);

                    // Distribute results back to each search
                    for (i, leaf) in pending_leaves.iter().enumerate() {
                        let logits_map = &logits_list[i];
                        let coords: Vec<Coord> = logits_map.keys().copied().collect();
                        let logit_vals: Vec<f64> = coords.iter().map(|c| logits_map[c]).collect();
                        let prior_vals = super::scoring::softmax(&logit_vals);
                        let leaf_priors: HashMap<Coord, f64> =
                            coords.into_iter().zip(prior_vals).collect();
                        let leaf_value = values[i];

                        complete_simulation(
                            &mut searches[leaf.search_idx].root,
                            &leaf.selection,
                            leaf_priors,
                            leaf_value,
                        );
                    }
                }
            }

            // Eliminate bottom half for each search
            for (si, search) in searches.iter_mut().enumerate() {
                if search.remaining.len() <= 1 || sims_used[si] == 0 {
                    continue;
                }

                let qctx = QContext::new(&search.root, &search.coords);
                search.remaining.sort_by(|&a, &b| {
                    let score_a = search.gumbel_samples[a]
                        + search.logits[a]
                        + sigma(qctx.norm_q[&search.coords[a]], qctx.max_child_visits, c_visit, c_scale);
                    let score_b = search.gumbel_samples[b]
                        + search.logits[b]
                        + sigma(qctx.norm_q[&search.coords[b]], qctx.max_child_visits, c_visit, c_scale);
                    score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
                });
                let keep = ((search.remaining.len() + 1) / 2).max(2); // floor at the final pair
                search.remaining.truncate(keep);
            }
        }
    }

    // --- Phase 4: Finalize each search ---
    let mut results = Vec::with_capacity(searches.len());

    for search in &searches {
        let improved_policy = compute_improved_policy(
            &search.logits,
            &search.coords,
            &search.root,
            config.c_visit,
            config.c_scale,
        );

        let qctx = QContext::new(&search.root, &search.coords);
        let mut best_score = f64::NEG_INFINITY;
        let mut selected_coord = search.coords[search.remaining[0]];

        for &idx in &search.remaining {
            let coord = search.coords[idx];
            let norm_q = qctx.norm_q[&coord];
            let score = search.gumbel_samples[idx]
                + search.logits[idx]
                + sigma(norm_q, qctx.max_child_visits, config.c_visit, config.c_scale);
            if score > best_score {
                best_score = score;
                selected_coord = coord;
            }
        }

        results.push(BatchedMCTSResult {
            action: selected_coord,
            improved_policy,
            coords: search.coords.clone(),
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::GameConfig;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn uniform_eval(states: &[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let logits = states
            .iter()
            .map(|s| s.legal_moves().iter().map(|&c| (c, 0.0)).collect())
            .collect();
        (logits, vec![0.0; states.len()])
    }

    #[test]
    fn batched_lockstep_spends_full_budget_via_top_up() {
        // mctx parity for the lockstep copy of SH: n=200/m=16 must spend all
        // 200 sims (the nominal phases only cover 194). win_length=7 keeps
        // every leaf non-terminal at the depths this search reaches, so the
        // eval_fn sees exactly 1 root state + 1 state per simulation.
        let game = GameState::with_config(GameConfig {
            win_length: 7,
            placement_radius: 2,
            max_moves: 200,
        });
        assert!(game.legal_moves().len() >= 16);

        let config = MCTSConfig {
            n_simulations: 200,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut states_seen = 0usize;
        let mut eval = |states: &[GameState]| {
            states_seen += states.len();
            uniform_eval(states)
        };
        batched_gumbel_mcts(&[game], &config, &mut rng, &mut eval).unwrap();

        assert_eq!(states_seen, 1 + 200, "1 root eval + full sim budget");
    }
}
