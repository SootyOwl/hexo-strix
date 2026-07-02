use std::fmt;

/// Axial coordinate (q, r) on the hex grid.
pub type Coord = (i32, i32);

/// The 6 neighbor directions in axial coordinates.
pub const HEX_DIRS: [Coord; 6] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
];

/// The 3 win-detection axes (one direction per axis).
pub const WIN_AXES: [Coord; 3] = [
    (1, 0),
    (0, 1),
    (1, -1),
];

/// A player in the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    P1,
    P2,
}

impl Player {
    /// Returns the opposing player.
    pub fn opponent(self) -> Self {
        match self {
            Player::P1 => Player::P2,
            Player::P2 => Player::P1,
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Player::P1 => write!(f, "P1"),
            Player::P2 => write!(f, "P2"),
        }
    }
}
