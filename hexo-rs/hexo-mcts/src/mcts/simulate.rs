use hexo_engine::{Coord, Player};
use rustc_hash::FxHashMap as HashMap;

use super::backup::compute_backup_values;
use super::node::MCTSNode;
use super::select::select_child;

/// Result of selecting a path to a leaf node during simulation.
pub struct LeafSelection {
    /// Players at each depth along the path (root to leaf).
    pub path_players: Vec<Player>,
    /// Actions taken at each step (length = path_players.len() - 1).
    pub path_actions: Vec<Coord>,
    /// Whether the leaf is terminal.
    pub is_terminal: bool,
    /// If terminal: the game outcome from the leaf's perspective.
    /// +1.0 = leaf's player won, -1.0 = lost, 0.0 = draw.
    pub terminal_value: Option<f64>,
}

/// Select a path from root to an unexpanded leaf, optionally forcing the first action.
///
/// Returns a `LeafSelection` describing the path taken. The caller is responsible
/// for evaluating non-terminal leaves (via network) and calling `apply_backup`.
pub fn select_leaf(
    root: &mut MCTSNode,
    forced_action: Option<Coord>,
    c_visit: u32,
    c_scale: f64,
) -> LeafSelection {
    let mut path_players = vec![root.current_player];
    let mut path_actions: Vec<Coord> = Vec::new();

    // Navigate to the first child
    let first_action = if let Some(action) = forced_action {
        action
    } else if root.is_expanded() {
        select_child(root, c_visit, c_scale)
    } else {
        // Root not expanded and no forced action — return root as leaf
        return LeafSelection {
            path_players,
            path_actions,
            is_terminal: root.game_state.as_ref().is_some_and(|g| g.is_terminal()),
            terminal_value: None,
        };
    };

    // Walk down the tree using the standard mutable linked-list traversal pattern.
    // select_child takes &MCTSNode (shared reborrow), returns a Copy Coord, then
    // get_or_create_child takes &mut self. Reassigning `node` releases the old borrow.
    let mut node = root;
    let mut action = first_action;

    loop {
        let child = node.get_or_create_child(action);
        path_actions.push(action);
        path_players.push(child.current_player);

        let is_terminal = child
            .game_state
            .as_ref()
            .is_some_and(|g| g.is_terminal());

        if !child.is_expanded() || is_terminal {
            let terminal_value = if is_terminal {
                let game = child.game_state.as_ref().unwrap();
                Some(match game.winner() {
                    None => 0.0, // draw
                    Some(winner) => {
                        // Value from the leaf node's perspective
                        if winner == child.current_player {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                })
            } else {
                None
            };

            return LeafSelection {
                path_players,
                path_actions,
                is_terminal,
                terminal_value,
            };
        }

        action = select_child(child, c_visit, c_scale);
        node = child;
    }
}

/// Apply backup values along a path through the tree.
///
/// `path_actions` contains the actions taken at each step. The function navigates
/// from root, applying visit_count and value_sum updates at each node.
pub fn apply_backup(
    root: &mut MCTSNode,
    path_actions: &[Coord],
    backup_values: &[f64],
) {
    debug_assert_eq!(
        backup_values.len(),
        path_actions.len() + 1,
        "backup_values length must be path_actions + 1 (root)"
    );
    // First value is for the root
    root.visit_count += 1;
    root.value_sum += backup_values[0];

    let mut node: &mut MCTSNode = root;
    for (i, &action) in path_actions.iter().enumerate() {
        let child = node.children.get_mut(&action).expect("backup path broken");
        child.visit_count += 1;
        child.value_sum += backup_values[i + 1];
        node = child;
    }
}

/// Apply virtual-loss bookkeeping along a path through the tree.
///
/// Used by Sequential Halving inner-loop fusion (Phase 2): when several leaves
/// are selected and held pending a single batched network evaluation, each
/// selected path gets a virtual "loss" applied so subsequent selections within
/// the same fused batch see the in-flight path as visited+pessimised, encouraging
/// diversification.
///
/// Mirrors `apply_backup` but with `leaf_value = -magnitude` (the leaf is treated
/// as if it had been evaluated to a loss from the leaf player's perspective), and
/// also increments `virtual_loss_count` on every touched node so a paired
/// `revert_virtual_loss` can roll the bookkeeping back.
///
/// No-op when `magnitude <= 0.0` (the disabled-fusion path).
pub fn apply_virtual_loss(
    root: &mut MCTSNode,
    selection: &LeafSelection,
    magnitude: f64,
) {
    if magnitude <= 0.0 {
        return;
    }
    let backup_vals =
        super::backup::compute_backup_values(&selection.path_players, -magnitude);
    debug_assert_eq!(backup_vals.len(), selection.path_actions.len() + 1);

    root.visit_count += 1;
    root.value_sum += backup_vals[0];
    root.virtual_loss_count += 1;

    let mut node: &mut MCTSNode = root;
    for (i, &action) in selection.path_actions.iter().enumerate() {
        let child = node
            .children
            .get_mut(&action)
            .expect("apply_virtual_loss: path broken");
        child.visit_count += 1;
        child.value_sum += backup_vals[i + 1];
        child.virtual_loss_count += 1;
        node = child;
    }
}

/// Revert a previously applied virtual-loss path. The `selection` and `magnitude`
/// must match the matching `apply_virtual_loss` call; otherwise the tree state
/// will not be restored correctly.
///
/// No-op when `magnitude <= 0.0`.
pub fn revert_virtual_loss(
    root: &mut MCTSNode,
    selection: &LeafSelection,
    magnitude: f64,
) {
    if magnitude <= 0.0 {
        return;
    }
    let backup_vals =
        super::backup::compute_backup_values(&selection.path_players, -magnitude);
    debug_assert_eq!(backup_vals.len(), selection.path_actions.len() + 1);

    root.visit_count = root
        .visit_count
        .checked_sub(1)
        .expect("revert_virtual_loss: root visit_count underflow");
    root.value_sum -= backup_vals[0];
    root.virtual_loss_count = root
        .virtual_loss_count
        .checked_sub(1)
        .expect("revert_virtual_loss: root virtual_loss_count underflow");

    let mut node: &mut MCTSNode = root;
    for (i, &action) in selection.path_actions.iter().enumerate() {
        let child = node
            .children
            .get_mut(&action)
            .expect("revert_virtual_loss: path broken");
        child.visit_count = child
            .visit_count
            .checked_sub(1)
            .expect("revert_virtual_loss: child visit_count underflow");
        child.value_sum -= backup_vals[i + 1];
        child.virtual_loss_count = child
            .virtual_loss_count
            .checked_sub(1)
            .expect("revert_virtual_loss: child virtual_loss_count underflow");
        node = child;
    }
}

/// Run a full simulation: select leaf, compute backup values, apply them.
///
/// For non-terminal leaves, the caller must provide the network evaluation
/// via `leaf_value` and `leaf_priors`. For terminal leaves, the value is
/// computed from the game outcome.
///
/// Returns `Some((path_actions, leaf_index))` if the leaf needs network evaluation,
/// or `None` if the leaf was terminal (backup already applied).
pub fn simulate_select(
    root: &mut MCTSNode,
    forced_action: Option<Coord>,
    c_visit: u32,
    c_scale: f64,
) -> Option<LeafSelection> {
    let selection = select_leaf(root, forced_action, c_visit, c_scale);

    if selection.is_terminal {
        let value = selection.terminal_value.unwrap_or(0.0);
        let backup_vals = compute_backup_values(&selection.path_players, value);
        apply_backup(root, &selection.path_actions, &backup_vals);
        return None;
    }

    Some(selection)
}

/// After network evaluation, expand the leaf and apply backup.
///
/// Navigate to the leaf using `path_actions`, expand it with the provided priors
/// and value, then backup.
pub fn complete_simulation(
    root: &mut MCTSNode,
    selection: &LeafSelection,
    leaf_priors: HashMap<Coord, f64>,
    leaf_value: f64,
) {
    // Navigate to the leaf
    let mut node: &mut MCTSNode = root;
    for &action in &selection.path_actions {
        node = node.children.get_mut(&action).expect("path broken");
    }

    // Expand the leaf
    node.expand(leaf_priors, leaf_value);

    // Backup
    let backup_vals = compute_backup_values(&selection.path_players, leaf_value);
    apply_backup(root, &selection.path_actions, &backup_vals);
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{GameConfig, GameState};
    use rustc_hash::FxHashMap as HashMap;

    fn small_config() -> GameConfig {
        GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        }
    }

    fn make_root() -> MCTSNode {
        let game = GameState::with_config(small_config());
        let moves = game.legal_moves();
        let mut priors = HashMap::default();
        for &m in &moves {
            priors.insert(m, 1.0 / moves.len() as f64);
        }
        let mut root = MCTSNode::new(1.0, None, Player::P2);
        root.game_state = Some(game);
        root.expand(priors, 0.0);
        root
    }

    #[test]
    fn select_leaf_unexpanded_child() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        let action = moves[0];

        let selection = select_leaf(&mut root, Some(action), 50, 1.0);
        assert_eq!(selection.path_players.len(), 2); // root + child
        assert_eq!(selection.path_actions.len(), 1);
        assert_eq!(selection.path_actions[0], action);
        assert!(!selection.is_terminal);
    }

    #[test]
    fn select_leaf_terminal() {
        // Create a game that's almost won, then force the winning move
        let config = GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 80,
        };
        let mut game = GameState::with_config(config);
        // P2 plays (1,0) — this gives P2 two stones adjacent, winning with win_length=2?
        // Actually P1 is at (0,0), P2 goes first. P2 plays (1,0) — that's only 1 P2 stone.
        // We need win_length=2 and two P2 stones in a row.
        // P2 plays at (1,0), P2 plays at (2,0) — but wait, P2 has 2 moves per turn.
        game.apply_move((1, 0)).unwrap(); // P2 move 1

        // After this, (1,0) has P2. P1 is at (0,0). P2 needs one more adjacent.
        // If P2 plays at (2,0), that's 2 P2 stones in a row → win with win_length=2
        let moves = game.legal_moves();
        let winning_move = (2, 0);
        assert!(moves.contains(&winning_move));

        let mut priors = HashMap::default();
        for &m in &moves {
            priors.insert(m, 1.0 / moves.len() as f64);
        }

        let mut root = MCTSNode::new(1.0, None, Player::P2);
        root.game_state = Some(game);
        root.expand(priors, 0.0);

        let selection = select_leaf(&mut root, Some(winning_move), 50, 1.0);
        assert!(selection.is_terminal);
        assert!(selection.terminal_value.is_some());
    }

    #[test]
    fn apply_backup_updates_counts() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        let action = moves[0];

        // Create a child
        root.get_or_create_child(action);

        let backup_vals = vec![0.5, -0.5]; // root, child
        apply_backup(&mut root, &[action], &backup_vals);

        assert_eq!(root.visit_count, 1);
        assert!((root.value_sum - 0.5).abs() < 1e-10);
        let child = root.children.get(&action).unwrap();
        assert_eq!(child.visit_count, 1);
        assert!((child.value_sum - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn simulate_select_returns_none_for_terminal() {
        let config = GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 80,
        };
        let mut game = GameState::with_config(config);
        game.apply_move((1, 0)).unwrap(); // P2 move 1

        let moves = game.legal_moves();
        let mut priors = HashMap::default();
        for &m in &moves {
            priors.insert(m, 1.0 / moves.len() as f64);
        }

        let mut root = MCTSNode::new(1.0, None, Player::P2);
        root.game_state = Some(game);
        root.expand(priors, 0.0);

        let result = simulate_select(&mut root, Some((2, 0)), 50, 1.0);
        assert!(result.is_none()); // terminal → backup already done
        assert_eq!(root.visit_count, 1); // root was backed up
    }

    #[test]
    fn simulate_select_returns_some_for_non_terminal() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();

        let result = simulate_select(&mut root, Some(moves[0]), 50, 1.0);
        assert!(result.is_some());
        assert_eq!(root.visit_count, 0); // not yet backed up
    }

    #[test]
    fn complete_simulation_expands_and_backs_up() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        let action = moves[0];

        let selection = simulate_select(&mut root, Some(action), 50, 1.0).unwrap();

        // Simulate network eval: provide priors and value for the leaf
        let leaf_priors = [((0, 0), 0.5), ((1, 1), 0.5)].into_iter().collect::<HashMap<_,_>>();
        complete_simulation(&mut root, &selection, leaf_priors, 0.3);

        assert_eq!(root.visit_count, 1);
        let child = root.children.get(&action).unwrap();
        assert_eq!(child.visit_count, 1);
        assert!(child.is_expanded());
    }

    #[test]
    fn apply_virtual_loss_zero_magnitude_is_noop() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        let action = moves[0];
        let selection = simulate_select(&mut root, Some(action), 50, 1.0).unwrap();

        let pre_root_visits = root.visit_count;
        let pre_root_sum = root.value_sum;
        let pre_child_visits = root.children[&action].visit_count;

        apply_virtual_loss(&mut root, &selection, 0.0);

        assert_eq!(root.visit_count, pre_root_visits);
        assert_eq!(root.value_sum, pre_root_sum);
        assert_eq!(root.virtual_loss_count, 0);
        assert_eq!(root.children[&action].visit_count, pre_child_visits);
        assert_eq!(root.children[&action].virtual_loss_count, 0);
    }

    #[test]
    fn apply_virtual_loss_increments_counts_along_path() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        let action = moves[0];
        let selection = simulate_select(&mut root, Some(action), 50, 1.0).unwrap();

        apply_virtual_loss(&mut root, &selection, 1.0);

        assert_eq!(root.visit_count, 1);
        assert_eq!(root.virtual_loss_count, 1);
        let child = root.children.get(&action).unwrap();
        assert_eq!(child.visit_count, 1);
        assert_eq!(child.virtual_loss_count, 1);
    }

    #[test]
    fn apply_then_revert_virtual_loss_restores_state() {
        // Apply VL on a path, then revert; counts and sums must match pre-apply state.
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        let action = moves[0];
        let selection = simulate_select(&mut root, Some(action), 50, 1.0).unwrap();

        // Snapshot every node touched by the would-be VL path.
        let pre_root_visits = root.visit_count;
        let pre_root_sum = root.value_sum;
        let pre_child_visits = root.children[&action].visit_count;
        let pre_child_sum = root.children[&action].value_sum;

        apply_virtual_loss(&mut root, &selection, 0.75);
        // Sanity: state must have changed.
        assert_ne!(root.visit_count, pre_root_visits);

        revert_virtual_loss(&mut root, &selection, 0.75);

        assert_eq!(root.visit_count, pre_root_visits);
        assert!((root.value_sum - pre_root_sum).abs() < 1e-12);
        assert_eq!(root.virtual_loss_count, 0);
        let child = root.children.get(&action).unwrap();
        assert_eq!(child.visit_count, pre_child_visits);
        assert!((child.value_sum - pre_child_sum).abs() < 1e-12);
        assert_eq!(child.virtual_loss_count, 0);
    }

    #[test]
    fn virtual_loss_changes_select_child_decision() {
        // With equal priors and no real visits, select_child picks deterministically.
        // After applying VL to that pick, select_child must pick something else
        // (the previous winner is now visited+pessimised).
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        assert!(moves.len() >= 2, "test needs >=2 legal moves");

        // Baseline: select_child without VL.
        let baseline = super::super::select::select_child(&root, 50, 1.0);

        // Force a descent to `baseline` so we get a real LeafSelection and the
        // child node exists on the tree (required for apply_virtual_loss).
        let selection =
            simulate_select(&mut root, Some(baseline), 50, 1.0).unwrap();
        apply_virtual_loss(&mut root, &selection, 1.0);

        let after_vl = super::super::select::select_child(&root, 50, 1.0);
        assert_ne!(
            after_vl, baseline,
            "VL applied to {baseline:?} should diversify the next selection",
        );
    }

    #[test]
    fn virtual_loss_single_legal_move_still_selects_it() {
        // Single-legal-move node: even with VL applied, select_child must return it.
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        root.game_state = Some(GameState::with_config(small_config()));
        let only_action = (0, 0);
        root.expand([(only_action, 1.0)].into_iter().collect::<HashMap<_,_>>(), 0.0);
        // Manually create the child without descending the game tree (the
        // game_state at root for small_config has multiple legal moves; we
        // only care about the select_child invariant here).
        let mut child = MCTSNode::new(1.0, Some(only_action), Player::P2);
        child.visit_count = 5;
        child.value_sum = -5.0; // simulate severe VL pessimism
        child.virtual_loss_count = 5;
        root.children.insert(only_action, child);

        let action = super::super::select::select_child(&root, 50, 1.0);
        assert_eq!(action, only_action);
    }
}
