use crate::types::Player;

/// Tracks whose turn it is and how many moves remain in the current turn.
///
/// P1's first placement is hardcoded at (0,0), so the game effectively
/// starts with P2 to move with 2 moves remaining.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    P1Turn { moves_left: u8 },
    P2Turn { moves_left: u8 },
    GameOver,
}

impl TurnState {
    /// Returns the player whose turn it currently is, or None if the game is over.
    pub fn current_player(self) -> Option<Player> {
        match self {
            TurnState::P1Turn { .. } => Some(Player::P1),
            TurnState::P2Turn { .. } => Some(Player::P2),
            TurnState::GameOver => None,
        }
    }

    /// Returns how many moves are left in this turn, or None if the game is over.
    pub fn moves_remaining(self) -> Option<u8> {
        match self {
            TurnState::P1Turn { moves_left } => Some(moves_left),
            TurnState::P2Turn { moves_left } => Some(moves_left),
            TurnState::GameOver => None,
        }
    }

    /// Advance the state after a stone placement.
    ///
    /// If `winner` is true, transitions to `GameOver`.
    /// Otherwise, decrements `moves_left`; when it reaches 0, switches to the
    /// other player with `moves_left = 2`.
    pub fn advance(self, winner: bool) -> TurnState {
        if winner {
            return TurnState::GameOver;
        }
        match self {
            TurnState::P1Turn { moves_left: 2 } => TurnState::P1Turn { moves_left: 1 },
            TurnState::P1Turn { moves_left: 1 } => TurnState::P2Turn { moves_left: 2 },
            TurnState::P1Turn { moves_left: _ } => unreachable!("invalid moves_left for P1Turn"),
            TurnState::P2Turn { moves_left: 2 } => TurnState::P2Turn { moves_left: 1 },
            TurnState::P2Turn { moves_left: 1 } => TurnState::P1Turn { moves_left: 2 },
            TurnState::P2Turn { moves_left: _ } => unreachable!("invalid moves_left for P2Turn"),
            TurnState::GameOver => TurnState::GameOver,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- current_player() ---

    #[test]
    fn current_player_p1_turn() {
        let state = TurnState::P1Turn { moves_left: 2 };
        assert_eq!(state.current_player(), Some(Player::P1));
    }

    #[test]
    fn current_player_p2_turn() {
        let state = TurnState::P2Turn { moves_left: 2 };
        assert_eq!(state.current_player(), Some(Player::P2));
    }

    #[test]
    fn current_player_game_over() {
        assert_eq!(TurnState::GameOver.current_player(), None);
    }

    // --- moves_remaining() ---

    #[test]
    fn moves_remaining_start_of_turn() {
        let state = TurnState::P2Turn { moves_left: 2 };
        assert_eq!(state.moves_remaining(), Some(2));
    }

    #[test]
    fn moves_remaining_after_first_placement() {
        let state = TurnState::P2Turn { moves_left: 1 };
        assert_eq!(state.moves_remaining(), Some(1));
    }

    #[test]
    fn moves_remaining_game_over() {
        assert_eq!(TurnState::GameOver.moves_remaining(), None);
    }

    // --- advance(winner) normal flow ---

    #[test]
    fn advance_decrements_moves_left_from_2_to_1() {
        let state = TurnState::P2Turn { moves_left: 2 };
        let next = state.advance(false);
        assert_eq!(next, TurnState::P2Turn { moves_left: 1 });
    }

    #[test]
    fn advance_switches_player_when_moves_left_reaches_0() {
        let state = TurnState::P2Turn { moves_left: 1 };
        let next = state.advance(false);
        assert_eq!(next, TurnState::P1Turn { moves_left: 2 });
    }

    #[test]
    fn advance_winner_mid_turn_transitions_to_game_over() {
        let state = TurnState::P2Turn { moves_left: 2 };
        let next = state.advance(true);
        assert_eq!(next, TurnState::GameOver);
    }

    #[test]
    fn advance_winner_on_last_move_transitions_to_game_over() {
        let state = TurnState::P1Turn { moves_left: 1 };
        let next = state.advance(true);
        assert_eq!(next, TurnState::GameOver);
    }

    #[test]
    fn advance_game_over_returns_game_over() {
        let next = TurnState::GameOver.advance(false);
        assert_eq!(next, TurnState::GameOver);
    }

    // --- Full cycle: P2(2) -> P2(1) -> P1(2) -> P1(1) -> P2(2) ---

    #[test]
    fn full_turn_cycle_p2_to_p1_to_p2() {
        let s0 = TurnState::P2Turn { moves_left: 2 };
        assert_eq!(s0.current_player(), Some(Player::P2));
        assert_eq!(s0.moves_remaining(), Some(2));

        let s1 = s0.advance(false);
        assert_eq!(s1, TurnState::P2Turn { moves_left: 1 });
        assert_eq!(s1.current_player(), Some(Player::P2));
        assert_eq!(s1.moves_remaining(), Some(1));

        let s2 = s1.advance(false);
        assert_eq!(s2, TurnState::P1Turn { moves_left: 2 });
        assert_eq!(s2.current_player(), Some(Player::P1));
        assert_eq!(s2.moves_remaining(), Some(2));

        let s3 = s2.advance(false);
        assert_eq!(s3, TurnState::P1Turn { moves_left: 1 });
        assert_eq!(s3.current_player(), Some(Player::P1));
        assert_eq!(s3.moves_remaining(), Some(1));

        let s4 = s3.advance(false);
        assert_eq!(s4, TurnState::P2Turn { moves_left: 2 });
        assert_eq!(s4.current_player(), Some(Player::P2));
        assert_eq!(s4.moves_remaining(), Some(2));
    }

    #[test]
    fn advance_p1_decrements_from_2_to_1() {
        let state = TurnState::P1Turn { moves_left: 2 };
        let next = state.advance(false);
        assert_eq!(next, TurnState::P1Turn { moves_left: 1 });
    }
}
