use std::collections::HashMap;
use std::fmt;

use crate::types::{Coord, Player};

/// Sparse board storage: maps placed coordinates to their owning player.
#[derive(Clone)]
pub struct Board {
    stones: HashMap<Coord, Player>,
}

/// Error returned when attempting to place a stone on an occupied cell.
#[derive(Debug, PartialEq, Eq)]
pub enum PlaceError {
    CellOccupied,
}

impl fmt::Display for PlaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlaceError::CellOccupied => write!(f, "cell is already occupied"),
        }
    }
}

impl std::error::Error for PlaceError {}

impl Board {
    /// Creates a new board with P1's opening stone at the origin (0, 0).
    pub fn new() -> Self {
        let mut stones = HashMap::new();
        stones.insert((0, 0), Player::P1);
        Board { stones }
    }

    /// Places `player`'s stone at `coord`.
    ///
    /// Returns `Err(PlaceError::CellOccupied)` if the cell already has a stone.
    pub fn place(&mut self, coord: Coord, player: Player) -> Result<(), PlaceError> {
        if self.stones.contains_key(&coord) {
            return Err(PlaceError::CellOccupied);
        }
        self.stones.insert(coord, player);
        Ok(())
    }

    /// Returns the player whose stone occupies `coord`, or `None` if empty.
    pub fn get(&self, coord: Coord) -> Option<Player> {
        self.stones.get(&coord).copied()
    }

    /// Returns a reference to the underlying stone map.
    pub fn stones(&self) -> &HashMap<Coord, Player> {
        &self.stones
    }

    /// Returns the total number of stones on the board.
    pub fn stone_count(&self) -> usize {
        self.stones.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_p1_at_origin() {
        let board = Board::new();
        assert_eq!(board.get((0, 0)), Some(Player::P1));
    }

    #[test]
    fn place_on_empty_cell_succeeds() {
        let mut board = Board::new();
        let result = board.place((1, 0), Player::P2);
        assert!(result.is_ok());
        assert_eq!(board.get((1, 0)), Some(Player::P2));
    }

    #[test]
    fn place_on_occupied_cell_fails() {
        let mut board = Board::new();
        // (0,0) is already occupied by P1 from new()
        let result = board.place((0, 0), Player::P2);
        assert_eq!(result, Err(PlaceError::CellOccupied));
    }

    #[test]
    fn get_on_empty_cell_returns_none() {
        let board = Board::new();
        assert_eq!(board.get((99, 99)), None);
    }

    #[test]
    fn stones_returns_all_placed_stones() {
        let mut board = Board::new();
        board.place((1, 0), Player::P2).unwrap();
        let stones = board.stones();
        assert_eq!(stones.len(), 2);
        assert_eq!(stones.get(&(0, 0)), Some(&Player::P1));
        assert_eq!(stones.get(&(1, 0)), Some(&Player::P2));
    }

    #[test]
    fn clone_is_independent() {
        let mut original = Board::new();
        let mut cloned = original.clone();

        // Mutate the clone
        cloned.place((5, 5), Player::P2).unwrap();

        // Original should be unaffected
        assert_eq!(original.get((5, 5)), None);

        // Mutate the original
        original.place((3, 3), Player::P1).unwrap();

        // Clone should be unaffected
        assert_eq!(cloned.get((3, 3)), None);
    }
}
