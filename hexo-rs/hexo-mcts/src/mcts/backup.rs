use hexo_engine::Player;

/// Compute the backup value to add at each depth along a path.
///
/// The path is represented as a slice of `Player` values from root (index 0)
/// to leaf (last index). The `leaf_value` is the value from the leaf's own
/// perspective (the current player at the leaf).
///
/// Walking backwards from the leaf toward the root:
/// - The leaf gets `leaf_value` directly.
/// - When crossing from child to parent, if the players differ, the value is
///   negated (opponent's gain is your loss). If the players are the same
///   (mid-turn in HeXO's 2-placement turns), the value carries unchanged.
///
/// Returns a `Vec<f64>` of the same length as `path_players`, where each
/// element is the value to add to that node's `value_sum`.
pub fn compute_backup_values(path_players: &[Player], leaf_value: f64) -> Vec<f64> {
    let n = path_players.len();
    if n == 0 {
        return Vec::new();
    }

    let mut values = vec![0.0; n];
    let mut current = leaf_value;

    for i in (0..n).rev() {
        values[i] = current;
        if i > 0 && path_players[i] != path_players[i - 1] {
            current = -current;
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_player_path() {
        // [P1, P1, P1], value=1.0 → all get 1.0 (no player change)
        let path = [Player::P1, Player::P1, Player::P1];
        let values = compute_backup_values(&path, 1.0);
        assert_eq!(values, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn alternating_players() {
        // [P1, P2, P1], value=1.0
        // leaf(P1)=1.0, P1→P2 flip: -1.0, P2→P1 flip: 1.0
        let path = [Player::P1, Player::P2, Player::P1];
        let values = compute_backup_values(&path, 1.0);
        assert_eq!(values, vec![1.0, -1.0, 1.0]);
    }

    #[test]
    fn hexo_path_p1_p2_p2() {
        // [P1, P2, P2], value=1.0
        // leaf(P2)=1.0, P2→P2 same: 1.0, P2→P1 flip: -1.0
        let path = [Player::P1, Player::P2, Player::P2];
        let values = compute_backup_values(&path, 1.0);
        assert_eq!(values, vec![-1.0, 1.0, 1.0]);
    }

    #[test]
    fn draw_value() {
        let path = [Player::P1, Player::P2, Player::P1];
        let values = compute_backup_values(&path, 0.0);
        assert_eq!(values, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn single_node() {
        let path = [Player::P1];
        let values = compute_backup_values(&path, 0.7);
        assert_eq!(values, vec![0.7]);
    }

    #[test]
    fn empty_path() {
        let values = compute_backup_values(&[], 1.0);
        assert!(values.is_empty());
    }

    #[test]
    fn hexo_full_turn_cycle() {
        // Realistic HeXO: P2(2 moves) → P1(2 moves) → P2(2 moves)
        // [P2, P2, P1, P1, P2, P2], leaf value=1.0
        // Working backwards:
        //   P2=1.0, P2→P2 same, P2=1.0, P1→P2 flip, P1=-1.0,
        //   P1→P1 same, P1=-1.0, P2→P1 flip, P2=1.0, P2→P2 same, P2=1.0
        let path = [Player::P2, Player::P2, Player::P1, Player::P1, Player::P2, Player::P2];
        let values = compute_backup_values(&path, 1.0);
        assert_eq!(values, vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0]);
    }

    #[test]
    fn fractional_value_with_mixed_players() {
        // [P1, P1, P2, P1] with leaf_value=0.3
        // Working backwards from leaf (index 3, P1):
        //   depth 3 (P1): current = 0.3
        //   depth 2 (P2): P1 != P2 → negate → current = -0.3
        //   depth 1 (P1): P2 != P1 → negate → current = 0.3
        //   depth 0 (P1): P1 == P1 → no flip → current = 0.3
        let path = [Player::P1, Player::P1, Player::P2, Player::P1];
        let values = compute_backup_values(&path, 0.3);
        assert!((values[3] - 0.3).abs() < 1e-10);   // leaf
        assert!((values[2] - (-0.3)).abs() < 1e-10); // P2 node
        assert!((values[1] - 0.3).abs() < 1e-10);    // P1 node (after double flip)
        assert!((values[0] - 0.3).abs() < 1e-10);    // root P1 (same as depth 1)
    }

    #[test]
    fn negative_leaf_value() {
        // [P1, P2] with leaf_value=-0.7
        // depth 1 (P2): current = -0.7
        // depth 0 (P1): P2 != P1 → negate → current = 0.7
        let path = [Player::P1, Player::P2];
        let values = compute_backup_values(&path, -0.7);
        assert!((values[1] - (-0.7)).abs() < 1e-10); // leaf P2
        assert!((values[0] - 0.7).abs() < 1e-10);    // root P1 (negated)
    }
}
