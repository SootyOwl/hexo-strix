use rustc_hash::FxHashMap as HashMap;

use hexo_engine::{Coord, GameState, Player};

/// A node in the MCTS search tree.
///
/// Each node represents a game state reached by playing `action` from the parent.
/// Nodes use lazy expansion: after network evaluation, `child_priors` stores the
/// prior probabilities for legal actions, but child `MCTSNode`s are only created
/// when first selected via `get_or_create_child`.
pub struct MCTSNode {
    pub prior: f64,
    pub action: Option<Coord>,
    pub current_player: Player,
    pub value_estimate: f64,
    pub visit_count: u32,
    pub value_sum: f64,
    /// Number of in-flight virtual-loss applications at this node. Used by
    /// Sequential Halving inner-loop fusion (Phase 2 of the VL prototype).
    /// When > 0, this node has unresolved VL bumps included in its
    /// `visit_count` and `value_sum`; reverting must zero this back out
    /// before the real backup is applied.
    pub virtual_loss_count: u32,
    pub children: HashMap<Coord, MCTSNode>,
    pub game_state: Option<GameState>,
    pub child_priors: HashMap<Coord, f64>,
    expanded: bool,
}

impl MCTSNode {
    pub fn new(prior: f64, action: Option<Coord>, current_player: Player) -> Self {
        Self {
            prior,
            action,
            current_player,
            value_estimate: 0.0,
            visit_count: 0,
            value_sum: 0.0,
            virtual_loss_count: 0,
            children: HashMap::default(),
            game_state: None,
            child_priors: HashMap::default(),
            expanded: false,
        }
    }

    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            return 0.0;
        }
        self.value_sum / self.visit_count as f64
    }

    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    /// Mark this node as expanded after network evaluation.
    /// Stores the prior probabilities and value estimate.
    pub fn expand(&mut self, child_priors: HashMap<Coord, f64>, value_estimate: f64) {
        // Pre-size children to the eventual maximum to avoid HashMap rehash
        // as get_or_create_child inserts new children incrementally during
        // tree descent.
        self.children.reserve(child_priors.len());
        self.child_priors = child_priors;
        self.value_estimate = value_estimate;
        self.expanded = true;
    }

    /// Get an existing child or lazily create one from the game state.
    ///
    /// # Panics
    /// Panics if `action` is not in `child_priors` or if `game_state` is `None`.
    pub fn get_or_create_child(&mut self, action: Coord) -> &mut MCTSNode {
        if self.children.contains_key(&action) {
            return self.children.get_mut(&action).unwrap();
        }

        let prior = *self
            .child_priors
            .get(&action)
            .unwrap_or_else(|| panic!("Action {action:?} not in node's child_priors"));

        let parent_state = self
            .game_state
            .as_ref()
            .expect("Cannot create child: node has no game_state");

        let mut child_state = parent_state.clone();
        child_state
            .apply_move(action)
            .expect("Action from child_priors should be legal");

        let child_player = child_state
            .current_player()
            .unwrap_or_else(|| self.current_player.opponent());

        let mut child = MCTSNode::new(prior, Some(action), child_player);
        child.game_state = Some(child_state);

        self.children.insert(action, child);
        self.children.get_mut(&action).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::GameConfig;

    fn small_config() -> GameConfig {
        GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        }
    }

    #[test]
    fn fresh_node_defaults() {
        let node = MCTSNode::new(0.5, None, Player::P1);
        assert_eq!(node.visit_count, 0);
        assert_eq!(node.value_sum, 0.0);
        assert!(!node.is_expanded());
        assert!(node.children.is_empty());
        assert!(node.child_priors.is_empty());
        assert_eq!(node.prior, 0.5);
    }

    #[test]
    fn q_value_unvisited_is_zero() {
        let node = MCTSNode::new(0.5, None, Player::P1);
        assert_eq!(node.q_value(), 0.0);
    }

    #[test]
    fn q_value_after_visits() {
        let mut node = MCTSNode::new(0.5, None, Player::P1);
        node.visit_count = 3;
        node.value_sum = 1.5;
        assert!((node.q_value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn expand_sets_expanded() {
        let mut node = MCTSNode::new(0.5, None, Player::P1);
        let priors = [((1, 0), 0.6), ((0, 1), 0.4)].into_iter().collect::<HashMap<_,_>>();
        node.expand(priors, 0.3);
        assert!(node.is_expanded());
        assert_eq!(node.value_estimate, 0.3);
    }

    #[test]
    fn expand_stores_priors() {
        let mut node = MCTSNode::new(0.5, None, Player::P1);
        let priors = [((1, 0), 0.6), ((0, 1), 0.4)].into_iter().collect::<HashMap<_,_>>();
        node.expand(priors, 0.3);
        assert_eq!(node.child_priors.len(), 2);
        assert!((node.child_priors[&(1, 0)] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn get_or_create_child_creates_lazily() {
        let mut node = MCTSNode::new(1.0, None, Player::P2);
        let game = GameState::with_config(small_config());
        // Game starts with P2 to move
        let moves = game.legal_moves();
        let action = moves[0];
        let mut priors = HashMap::default();
        for &m in &moves {
            priors.insert(m, 1.0 / moves.len() as f64);
        }
        node.game_state = Some(game);
        node.expand(priors, 0.0);

        assert!(node.children.is_empty());
        let child = node.get_or_create_child(action);
        assert_eq!(child.action, Some(action));
        assert!(child.game_state.is_some());
        assert_eq!(node.children.len(), 1);
    }

    #[test]
    fn get_or_create_child_returns_existing() {
        let mut node = MCTSNode::new(1.0, None, Player::P2);
        let game = GameState::with_config(small_config());
        let moves = game.legal_moves();
        let action = moves[0];
        let mut priors = HashMap::default();
        for &m in &moves {
            priors.insert(m, 1.0 / moves.len() as f64);
        }
        node.game_state = Some(game);
        node.expand(priors, 0.0);

        // Create child first time
        node.get_or_create_child(action);
        // Second call should return same child (no new children created)
        node.get_or_create_child(action);
        assert_eq!(node.children.len(), 1);
    }

    #[test]
    #[should_panic(expected = "not in node's child_priors")]
    fn get_or_create_child_panics_unknown_action() {
        let mut node = MCTSNode::new(1.0, None, Player::P2);
        let game = GameState::with_config(small_config());
        node.game_state = Some(game);
        node.expand(HashMap::default(), 0.0);

        // (99, 99) is not in child_priors
        node.get_or_create_child((99, 99));
    }

    #[test]
    fn child_inherits_correct_player() {
        let mut node = MCTSNode::new(1.0, None, Player::P2);
        let game = GameState::with_config(small_config());
        let moves = game.legal_moves();
        let action = moves[0];
        let priors = [(action, 1.0)].into_iter().collect::<HashMap<_,_>>();
        node.game_state = Some(game);
        node.expand(priors, 0.0);

        let child = node.get_or_create_child(action);
        // P2 has 2 moves per turn; after one move, still P2's turn
        assert_eq!(child.current_player, Player::P2);
    }
}
