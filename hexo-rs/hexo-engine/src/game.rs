use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::board::Board;
use crate::hex::hex_offsets;
use crate::legal_moves::legal_moves;
use crate::turn::TurnState;
use crate::types::{Coord, Player};
use crate::win::check_win;

/// Configuration parameters for a HeXO game.
#[derive(Debug, Clone, Copy)]
pub struct GameConfig {
    /// Number of consecutive stones needed to win.
    pub win_length: u8,
    /// Hex-distance radius within which moves are legal.
    pub placement_radius: i32,
    /// Maximum total stone placements before the game is a draw.
    pub max_moves: u32,
}

impl GameConfig {
    /// Standard HeXO configuration: 6-in-a-row, radius 8, 200 total moves.
    pub const FULL_HEXO: GameConfig = GameConfig {
        win_length: 6,
        placement_radius: 8,
        max_moves: 200,
    };
}

/// Errors returned by `GameState::apply_move`.
#[derive(Debug, PartialEq, Eq)]
pub enum MoveError {
    /// The game has already ended (win or draw).
    GameOver,
    /// The target cell is already occupied.
    CellOccupied,
    /// The target cell is outside the placement radius.
    OutOfRange,
}

/// The full game state: board, whose turn it is, move counter, and outcome.
#[derive(Clone)]
pub struct GameState {
    board: Board,
    turn: TurnState,
    config: GameConfig,
    move_count: u32,
    winner: Option<Player>,
    /// Precomputed hex-circle offsets for the configured radius (shared across clones).
    offsets: Arc<[Coord]>,
    /// Cached set of legal moves, updated incrementally on each apply_move.
    cached_legal: HashSet<Coord>,
}

impl GameState {
    /// Creates a new game with the standard FULL_HEXO configuration.
    pub fn new() -> Self {
        Self::with_config(GameConfig::FULL_HEXO)
    }

    /// Creates a new game with a custom configuration.
    ///
    /// P1's first stone is placed at (0,0) by `Board::new()`.
    /// The game starts with P2 to move with 2 moves remaining.
    ///
    /// # Panics
    /// Panics if `win_length < 2`, `placement_radius < 1`, or `max_moves < 1`.
    pub fn with_config(config: GameConfig) -> Self {
        assert!(config.win_length >= 2, "win_length must be >= 2");
        assert!(config.placement_radius >= 1, "placement_radius must be >= 1");
        assert!(config.max_moves >= 1, "max_moves must be >= 1");

        let board = Board::new();
        let offsets: Arc<[Coord]> = hex_offsets(config.placement_radius).into();
        let initial_legal: HashSet<Coord> =
            legal_moves(&board, config.placement_radius).into_iter().collect();
        GameState {
            board,
            turn: TurnState::P2Turn { moves_left: 2 },
            config,
            move_count: 0,
            winner: None,
            offsets,
            cached_legal: initial_legal,
        }
    }

    /// Attempts to place the current player's stone at `coord`.
    pub fn apply_move(&mut self, coord: Coord) -> Result<(), MoveError> {
        if self.is_terminal() {
            return Err(MoveError::GameOver);
        }

        if self.board.get(coord).is_some() {
            return Err(MoveError::CellOccupied);
        }

        if !self.cached_legal.contains(&coord) {
            return Err(MoveError::OutOfRange);
        }

        let player = self
            .turn
            .current_player()
            .expect("turn has current player when not terminal");
        self.board
            .place(coord, player)
            .expect("cell was verified empty");

        // Update legal moves cache incrementally.
        self.cached_legal.remove(&coord);
        let stones = self.board.stones();
        for &(dq, dr) in self.offsets.iter() {
            let cell = (coord.0 + dq, coord.1 + dr);
            if !stones.contains_key(&cell) {
                self.cached_legal.insert(cell);
            }
        }

        self.move_count += 1;

        let won = check_win(&self.board, coord, player, self.config.win_length);
        if won {
            self.winner = Some(player);
        }

        let draw = !won && self.move_count >= self.config.max_moves;
        self.turn = self.turn.advance(won || draw);

        Ok(())
    }

    /// Returns all legal moves for the current position, sorted lexicographically.
    pub fn legal_moves(&self) -> Vec<Coord> {
        if self.is_terminal() {
            return Vec::new();
        }
        let mut moves: Vec<Coord> = self.cached_legal.iter().copied().collect();
        moves.sort_unstable();
        moves
    }

    /// Returns the number of legal moves without allocating.
    pub fn legal_move_count(&self) -> usize {
        if self.is_terminal() { 0 } else { self.cached_legal.len() }
    }

    /// Returns a reference to the internal legal moves set. Iteration order is
    /// arbitrary (not sorted). Use this on hot paths where sorting is unnecessary.
    pub fn legal_moves_set(&self) -> &HashSet<Coord> {
        &self.cached_legal
    }

    /// Returns `true` when the game has ended (win or draw).
    pub fn is_terminal(&self) -> bool {
        self.turn == TurnState::GameOver
    }

    /// Returns the winner, or `None` if there is no winner (game ongoing or draw).
    pub fn winner(&self) -> Option<Player> {
        self.winner
    }

    /// Returns the player who should move next, or `None` if the game is over.
    pub fn current_player(&self) -> Option<Player> {
        self.turn.current_player()
    }

    /// Returns how many moves the current player has left this turn.
    /// Returns 0 when the game is over.
    pub fn moves_remaining_this_turn(&self) -> u8 {
        self.turn.moves_remaining().unwrap_or(0)
    }

    /// Returns all placed stones as `(coord, player)` pairs.
    pub fn placed_stones(&self) -> Vec<(Coord, Player)> {
        self.board
            .stones()
            .iter()
            .map(|(&coord, &player)| (coord, player))
            .collect()
    }

    /// Returns a reference to the underlying stone map (no allocation).
    pub fn stones(&self) -> &HashMap<Coord, Player> {
        self.board.stones()
    }

    /// Returns the total number of moves made so far (not counting P1's opening stone).
    pub fn move_count(&self) -> u32 {
        self.move_count
    }

    /// Returns a reference to the game configuration.
    pub fn config(&self) -> &GameConfig {
        &self.config
    }

    /// Reconstructs a non-terminal `GameState` directly from a stone set and
    /// turn info, bypassing the move-replay path. Intended for tests and
    /// benchmarks that need to materialise positions extracted from external
    /// sources (e.g. self-play binaries) without reconstructing move order.
    ///
    /// `stones` must contain the (0,0) opening for P1. No win check is run —
    /// the caller is responsible for only passing non-terminal positions.
    pub fn from_state(
        stones: &[(Coord, Player)],
        current_player: Player,
        moves_remaining: u8,
        config: GameConfig,
    ) -> Self {
        assert!(moves_remaining == 1 || moves_remaining == 2);
        let mut board = Board::new();
        // Board::new() seeds (0,0)=P1; replay the rest from the input.
        for &(coord, player) in stones {
            if coord == (0, 0) && player == Player::P1 {
                continue;
            }
            board
                .place(coord, player)
                .expect("from_state: stones must not collide");
        }
        let offsets: Arc<[Coord]> = hex_offsets(config.placement_radius).into();
        let cached_legal: HashSet<Coord> =
            legal_moves(&board, config.placement_radius).into_iter().collect();
        let turn = match current_player {
            Player::P1 => TurnState::P1Turn { moves_left: moves_remaining },
            Player::P2 => TurnState::P2Turn { moves_left: moves_remaining },
        };
        let move_count = stones.len().saturating_sub(1) as u32;
        GameState {
            board,
            turn,
            config,
            move_count,
            winner: None,
            offsets,
            cached_legal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // 1. Initial state
    // ------------------------------------------------------------------

    #[test]
    fn initial_state_p2_to_move() {
        let gs = GameState::new();
        assert_eq!(gs.current_player(), Some(Player::P2));
    }

    #[test]
    fn initial_state_two_moves_remaining() {
        let gs = GameState::new();
        assert_eq!(gs.moves_remaining_this_turn(), 2);
    }

    #[test]
    fn initial_state_not_terminal() {
        let gs = GameState::new();
        assert!(!gs.is_terminal());
    }

    #[test]
    fn initial_state_no_winner() {
        let gs = GameState::new();
        assert_eq!(gs.winner(), None);
    }

    #[test]
    fn initial_state_p1_at_origin() {
        let gs = GameState::new();
        let stones = gs.placed_stones();
        assert!(
            stones.contains(&((0, 0), Player::P1)),
            "expected P1 stone at origin, got {:?}",
            stones
        );
    }

    // ------------------------------------------------------------------
    // 2. apply_move basic
    // ------------------------------------------------------------------

    #[test]
    fn apply_move_valid_decrements_moves_remaining() {
        let mut gs = GameState::new();
        gs.apply_move((1, 0)).unwrap();
        assert_eq!(gs.moves_remaining_this_turn(), 1);
    }

    #[test]
    fn apply_move_two_moves_switches_player_to_p1() {
        let mut gs = GameState::new();
        gs.apply_move((1, 0)).unwrap();
        gs.apply_move((2, 0)).unwrap();
        assert_eq!(gs.current_player(), Some(Player::P1));
    }

    #[test]
    fn apply_move_occupied_returns_error() {
        let mut gs = GameState::new();
        let result = gs.apply_move((0, 0));
        assert_eq!(result, Err(MoveError::CellOccupied));
    }

    #[test]
    fn apply_move_out_of_range_returns_error() {
        let mut gs = GameState::new();
        let result = gs.apply_move((100, 100));
        assert_eq!(result, Err(MoveError::OutOfRange));
    }

    #[test]
    fn apply_move_after_game_over_returns_error() {
        let config = GameConfig {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
        };
        let mut gs = GameState::with_config(config);
        // P2 builds 4-in-a-row on q-axis: (1,0),(2,0),(3,0),(4,0)
        // P1 plays scattered to avoid accidental wins
        gs.apply_move((1, 0)).unwrap();  // P2
        gs.apply_move((2, 0)).unwrap();  // P2
        gs.apply_move((0, 3)).unwrap();  // P1
        gs.apply_move((0, -3)).unwrap(); // P1
        gs.apply_move((3, 0)).unwrap();  // P2
        gs.apply_move((4, 0)).unwrap();  // P2 → 4-in-a-row → win
        assert!(gs.is_terminal());
        assert_eq!(gs.winner(), Some(Player::P2));
        assert_eq!(gs.apply_move((5, 0)), Err(MoveError::GameOver));
    }

    // ------------------------------------------------------------------
    // 3. Win detection
    // ------------------------------------------------------------------

    #[test]
    fn win_on_first_of_two_moves_ends_immediately() {
        let config = GameConfig {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
        };
        let mut gs = GameState::with_config(config);
        // P2 builds along r-axis
        gs.apply_move((0, 1)).unwrap();  // P2
        gs.apply_move((0, 2)).unwrap();  // P2
        gs.apply_move((1, 1)).unwrap();  // P1 scattered
        gs.apply_move((-1, -1)).unwrap(); // P1 scattered
        gs.apply_move((0, 3)).unwrap();  // P2 — 3 in a row
        assert!(!gs.is_terminal());
        gs.apply_move((0, 4)).unwrap();  // P2 — 4 in a row → win
        assert!(gs.is_terminal());
        assert_eq!(gs.winner(), Some(Player::P2));
    }

    // ------------------------------------------------------------------
    // 4. Draw
    // ------------------------------------------------------------------

    #[test]
    fn draw_by_move_limit() {
        let config = GameConfig {
            win_length: 6,
            placement_radius: 8,
            max_moves: 4,
        };
        let mut gs = GameState::with_config(config);
        gs.apply_move((1, 0)).unwrap();
        gs.apply_move((0, 1)).unwrap();
        gs.apply_move((-1, 0)).unwrap();
        assert!(!gs.is_terminal());
        gs.apply_move((0, -1)).unwrap(); // 4th move → draw
        assert!(gs.is_terminal());
        assert_eq!(gs.winner(), None);
        assert_eq!(gs.apply_move((1, 1)), Err(MoveError::GameOver));
    }

    // ------------------------------------------------------------------
    // 5. Legal moves, placed_stones, clone, config
    // ------------------------------------------------------------------

    #[test]
    fn placed_cell_removed_from_legal_moves() {
        let mut gs = GameState::new();
        gs.apply_move((1, 0)).unwrap();
        let moves = gs.legal_moves();
        assert!(!moves.contains(&(1, 0)));
    }

    #[test]
    fn placed_stones_grows() {
        let mut gs = GameState::new();
        assert_eq!(gs.placed_stones().len(), 1);
        gs.apply_move((1, 0)).unwrap();
        assert_eq!(gs.placed_stones().len(), 2);
        gs.apply_move((0, 1)).unwrap();
        assert_eq!(gs.placed_stones().len(), 3);
    }

    #[test]
    fn clone_is_independent() {
        let mut gs = GameState::new();
        let gs2 = gs.clone();
        gs.apply_move((1, 0)).unwrap();
        assert_eq!(gs.placed_stones().len(), 2);
        assert_eq!(gs2.placed_stones().len(), 1);
    }

    #[test]
    fn custom_config_changes_legal_move_count() {
        let config = GameConfig {
            win_length: 4,
            placement_radius: 4,
            max_moves: 200,
        };
        let gs = GameState::with_config(config);
        assert_eq!(gs.legal_moves().len(), 60);
    }

    // ------------------------------------------------------------------
    // 6. Config validation
    // ------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "win_length must be >= 2")]
    fn invalid_config_win_length_zero() {
        GameState::with_config(GameConfig { win_length: 0, placement_radius: 8, max_moves: 200 });
    }

    #[test]
    #[should_panic(expected = "placement_radius must be >= 1")]
    fn invalid_config_negative_radius() {
        GameState::with_config(GameConfig { win_length: 6, placement_radius: -1, max_moves: 200 });
    }

    #[test]
    #[should_panic(expected = "max_moves must be >= 1")]
    fn invalid_config_zero_max_moves() {
        GameState::with_config(GameConfig { win_length: 6, placement_radius: 8, max_moves: 0 });
    }

}
