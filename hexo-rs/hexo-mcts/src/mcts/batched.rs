//! Batched Gumbel MCTS: run N searches in lockstep, batching eval calls.
//!
//! Instead of running N independent MCTS searches (each calling eval_fn
//! separately), this module synchronises them so leaf evaluations from
//! all N searches are collected into a single eval_fn call per halving round.

use rustc_hash::FxHashMap as HashMap;

use hexo_engine::{Coord, GameState};
use rand::Rng;

use super::MCTSConfig;
use super::forcing;
use super::gumbel_mcts::{SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET};
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

    // --- Phase 0: filter out fully-terminal games ---
    let non_terminal: Vec<usize> = (0..n)
        .filter(|&i| !games[i].is_terminal())
        .collect();

    if non_terminal.is_empty() {
        return Err("All games are terminal");
    }

    // `results[pos]` corresponds to `non_terminal[pos]` -- filled either by
    // the forced-win shortcut below or by the halving search in Phase 4.
    let mut results: Vec<Option<BatchedMCTSResult>> =
        (0..non_terminal.len()).map(|_| None).collect();

    // --- Phase 0.5: forced-win shortcut pre-pass.
    //
    // Mirrors `gumbel_mcts`'s Step 3.5 (Phase B): for each non-terminal game
    // with at least one placement left this turn (`moves_remaining_this_turn()
    // >= 1`, Change (A)), run the bounded fully-forcing solver BEFORE this game
    // enters the halving/eval rounds. On a proven win, the game gets its
    // shortcut result immediately (Change (B): two-hot on the order-invariant
    // winning pair at an mr==2 root, else one-hot on pv[0]) and is excluded
    // from both the root-eval batch and the halving loop below -- its NN eval
    // is skipped entirely, same as the single-game shortcut. Games with
    // `No`/`BudgetExceeded` fall through to normal batched search unchanged.
    let mut search_pos: Vec<usize> = Vec::new();
    for (pos, &game_idx) in non_terminal.iter().enumerate() {
        let game = &games[game_idx];
        // Change (A): fire at mr>=1, not only start-of-turn (mr==2). At an
        // mr==1 root the solver's `pv[0]` is the single completing placement,
        // so the one-hot fallback below is correct there.
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
            // Wide generator, mirroring `gumbel_mcts` Step 3.5: strict
            // superset of tight `solve` (quiet threat-building partners
            // included), soundness unchanged, ~1.5-1.7× mean per-call cost.
            if let forcing::Outcome::Win(w) = forcing::solve_wide(game, cap, budget) {
                // Defensive guard: `forcing::solve` is radius-aware, so
                // `first_move` should always be a member of
                // `game.legal_moves()`. Belt-and-suspenders against a future
                // solver regression: never return an action that isn't
                // legal -- fall through to normal search instead.
                if game.legal_moves_set().contains(&w.first_move) {
                    let coords = game.legal_moves();
                    let idx = coords
                        .iter()
                        .position(|&c| c == w.first_move)
                        .expect("first_move verified legal, so it is in coords");

                    // Change (B): at an mr==2 root the winning turn is the
                    // attacker's order-invariant pair {pv[0], pv[1]}, so the
                    // policy target is TWO-HOT (0.5 each). Fall back to one-hot
                    // on pv[0] when pv.len() < 2 or pv[1] is not a root legal
                    // move (tight-radius regime). At mr==1 the winning turn is
                    // a single placement -> one-hot on pv[0].
                    // `improved_policy` is exactly what the batched training
                    // export records as this game's policy target.
                    let pv1_idx = if game.moves_remaining_this_turn() == 2 && w.pv.len() >= 2 {
                        coords.iter().position(|&c| c == w.pv[1])
                    } else {
                        None
                    };
                    let mut improved_policy = vec![0.0; coords.len()];
                    match pv1_idx {
                        Some(idx2) => {
                            improved_policy[idx] = 0.5;
                            improved_policy[idx2] = 0.5;
                        }
                        None => improved_policy[idx] = 1.0,
                    }
                    results[pos] = Some(BatchedMCTSResult {
                        // `action` stays pv[0] (greedy determinism).
                        action: w.first_move,
                        improved_policy,
                        coords,
                    });
                    continue;
                }
            }
        }
        search_pos.push(pos);
    }

    if search_pos.is_empty() {
        return Ok(results
            .into_iter()
            .map(|r| r.expect("every slot filled by forced-win shortcut"))
            .collect());
    }

    // --- Phase 1: Batch root evaluation ---
    // Collect root states for every game still needing normal search into
    // one eval call.
    let root_states: Vec<GameState> = search_pos
        .iter()
        .map(|&pos| games[non_terminal[pos]].clone())
        .collect();
    let (all_logits, all_values) = eval_fn(&root_states);

    // --- Phase 2: Initialize search state for each game ---
    let mut searches: Vec<SearchState> = Vec::with_capacity(search_pos.len());

    for (eval_idx, &pos) in search_pos.iter().enumerate() {
        let game_idx = non_terminal[pos];
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
    for (search_i, search) in searches.iter().enumerate() {
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

        let pos = search_pos[search_i];
        results[pos] = Some(BatchedMCTSResult {
            action: selected_coord,
            improved_policy,
            coords: search.coords.clone(),
        });
    }

    Ok(results
        .into_iter()
        .map(|r| r.expect("every slot filled by forced-win shortcut or halving search"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{GameConfig, Player};
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

    /// The batched path must apply the same forced-win shortcut as
    /// `gumbel_mcts` (Step 3.5, Phase B) to EACH game in the batch
    /// independently: a forced-win game gets its shortcut result (Change (B):
    /// a two-hot policy on the solver's order-invariant winning pair at an
    /// mr==2 root) and never touches `eval_fn`, while a quiet game in the same
    /// batch still runs normal sequential halving. The `improved_policy` this
    /// records IS the game's training-export policy target.
    #[test]
    fn batched_forced_win_shortcut_is_per_game_two_hot_others_search_normally() {
        // Depth-3 forced win fixture, identical to
        // `gumbel_mcts_depth3_forced_win_returns_two_hot_policy` in
        // gumbel_mcts.rs: a q-axis 3-run plus two 2-runs positioned to
        // recreate `forcing::tests::mate_in_two_fork` one turn later.
        let stones: Vec<(Coord, Player)> = [
            (0, 0), (1, 0), (2, 0),
            (100, 100), (100, 101),
            (98, 104), (99, 103),
        ]
        .into_iter()
        .map(|c| (c, Player::P1))
        .collect();
        let forced_win_game = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);

        let solved = match forcing::solve(&forced_win_game, SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET) {
            forcing::Outcome::Win(w) => w,
            _ => panic!("fixture must be an independently-verified forced win"),
        };
        assert_eq!(solved.depth, 3, "fixture must be exactly depth-3");

        // Quiet position: fresh start-of-game state, no forcing structure
        // at all, so the solver must return `No` and the game must fall
        // through to normal batched search.
        let quiet_game = GameState::with_config(GameConfig::FULL_HEXO);
        assert_eq!(quiet_game.moves_remaining_this_turn(), 2);
        assert!(
            matches!(
                forcing::solve(&quiet_game, SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET),
                forcing::Outcome::No
            ),
            "quiet fixture must have no forced win"
        );

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        // Track which states reach eval_fn -- the forced-win game's states
        // (in particular its post-win-move continuations) must never appear,
        // since its NN eval is skipped entirely by the shortcut.
        let mut eval_call_count = 0usize;
        let mut eval = |states: &[GameState]| {
            eval_call_count += 1;
            uniform_eval(states)
        };

        let results = batched_gumbel_mcts(
            &[forced_win_game.clone(), quiet_game.clone()],
            &config,
            &mut rng,
            &mut eval,
        )
        .unwrap();
        assert_eq!(results.len(), 2);

        // The quiet game alone still needs eval calls (root + sims), so the
        // shortcut short-circuiting the forced-win game must not have
        // starved the batch of eval_fn calls entirely.
        assert!(eval_call_count > 0, "quiet game must still reach eval_fn");

        // Forced-win game: two-hot on the solver's order-invariant winning
        // pair {pv[0], pv[1]} (both legal at the root here); action stays pv[0].
        let forced_result = &results[0];
        assert_eq!(
            forced_result.action, solved.first_move,
            "batched action should be the solver's winning first move (pv[0])"
        );
        assert!(solved.pv.len() >= 2, "depth-3 win must have a >=2-move pv");
        let idx0 = forced_result
            .coords
            .iter()
            .position(|&c| c == solved.pv[0])
            .expect("pv[0] must be in coords");
        let idx1 = forced_result
            .coords
            .iter()
            .position(|&c| c == solved.pv[1])
            .expect("pv[1] must be in coords");
        assert_ne!(idx0, idx1);
        assert!(
            (forced_result.improved_policy[idx0] - 0.5).abs() < 1e-9,
            "pv[0] should carry 0.5, got {}",
            forced_result.improved_policy[idx0]
        );
        assert!(
            (forced_result.improved_policy[idx1] - 0.5).abs() < 1e-9,
            "pv[1] should carry 0.5, got {}",
            forced_result.improved_policy[idx1]
        );
        for (i, &p) in forced_result.improved_policy.iter().enumerate() {
            if i != idx0 && i != idx1 {
                assert_eq!(p, 0.0, "off-pair move {i} should have probability 0.0");
            }
        }

        // Quiet game: normal search result -- with uniform priors and
        // n=64/m=16 sims, visits spread across more than one candidate, so
        // the improved_policy must NOT collapse to a one-hot vector.
        let quiet_result = &results[1];
        let quiet_nonzero = quiet_result
            .improved_policy
            .iter()
            .filter(|&&p| p > 0.0)
            .count();
        assert!(
            quiet_nonzero > 1,
            "quiet position should have multiple non-zero policy entries, got {quiet_nonzero}"
        );
    }

    /// Change (A): the batched forced-win shortcut must also fire at an mr==1
    /// root, mirroring `gumbel_mcts_forcing_solver_fires_at_mr1_one_hot`. Same
    /// double-open-five fork: placing (9,0) is a solver-only forced win (not an
    /// immediate single-stone terminal win), so the shortcut must emit a
    /// one-hot on (9,0) and skip the NN eval for that game.
    #[test]
    fn batched_forced_win_shortcut_fires_at_mr1_one_hot() {
        let stones: Vec<(Coord, Player)> = [
            (10, 0), (11, 0), (12, 0), (13, 0),
            (9, 1), (9, 2), (9, 3), (9, 4),
        ]
        .into_iter()
        .map(|c| (c, Player::P1))
        .collect();
        let game = GameState::from_state(&stones, Player::P1, 1, GameConfig::FULL_HEXO);
        assert_eq!(game.moves_remaining_this_turn(), 1);

        let solved = match forcing::solve(&game, SELF_PLAY_DEPTH_CAP, SELF_PLAY_NODE_BUDGET) {
            forcing::Outcome::Win(w) => w,
            _ => panic!("fixture must be a forced win"),
        };
        assert_eq!(solved.first_move, (9, 0));

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
            virtual_loss: 0.0,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut eval = |states: &[GameState]| uniform_eval(states);

        let results = batched_gumbel_mcts(&[game.clone()], &config, &mut rng, &mut eval).unwrap();
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.action, (9, 0), "mr==1 shortcut must return the completing placement");
        let idx = r.coords.iter().position(|&c| c == (9, 0)).expect("(9,0) in coords");
        assert_eq!(r.improved_policy[idx], 1.0, "mr==1 completion must be one-hot");
        for (i, &p) in r.improved_policy.iter().enumerate() {
            if i != idx {
                assert_eq!(p, 0.0, "non-completion move {i} must be 0.0");
            }
        }
    }
}
