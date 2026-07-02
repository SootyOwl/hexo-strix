use std::collections::HashSet;

use crate::board::Board;
use crate::hex::hex_offsets;
use crate::types::Coord;

/// Returns all legal move positions for a board state.
///
/// A legal move is any empty cell within hex-distance ≤ `radius` of any
/// existing stone. The result is sorted lexicographically by (q, r) ascending.
pub fn legal_moves(board: &Board, radius: i32) -> Vec<Coord> {
    let offsets = hex_offsets(radius);
    let stones = board.stones();
    let mut candidates: HashSet<Coord> = HashSet::with_capacity(offsets.len());

    for &stone in stones.keys() {
        for &(dq, dr) in &offsets {
            let cell = (stone.0 + dq, stone.1 + dr);
            if !stones.contains_key(&cell) {
                candidates.insert(cell);
            }
        }
    }

    let mut moves: Vec<Coord> = candidates.into_iter().collect();
    moves.sort_unstable();
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::types::Player;

    #[test]
    fn initial_board_radius_8_has_216_legal_moves() {
        let board = Board::new();
        let moves = legal_moves(&board, 8);
        assert_eq!(
            moves.len(),
            216,
            "expected 216 legal moves (217 cells - 1 occupied), got {}",
            moves.len()
        );
    }

    #[test]
    fn all_moves_within_radius() {
        let board = Board::new();
        let radius = 8;
        let moves = legal_moves(&board, radius);
        // The only stone is at (0,0), so all returned coords must be within
        // hex-distance ≤ radius of (0,0).
        for &(q, r) in &moves {
            let dist = q.abs().max(r.abs()).max((q + r).abs());
            assert!(
                dist <= radius,
                "coord ({q}, {r}) has hex-distance {dist} > radius {radius}"
            );
        }
    }

    #[test]
    fn no_occupied_cells_in_moves() {
        let board = Board::new();
        let moves = legal_moves(&board, 8);
        for coord in &moves {
            assert!(
                board.get(*coord).is_none(),
                "occupied cell {coord:?} should not appear in legal moves"
            );
        }
    }

    #[test]
    fn no_duplicates_in_moves() {
        let board = Board::new();
        let moves = legal_moves(&board, 8);
        let unique: HashSet<Coord> = moves.iter().copied().collect();
        assert_eq!(
            moves.len(),
            unique.len(),
            "legal moves contains duplicates"
        );
    }

    #[test]
    fn second_stone_expands_legal_move_set() {
        let mut board = Board::new();
        let moves_before = legal_moves(&board, 8);

        // Place a second stone far enough to add new cells outside the original
        // radius-8 circle around origin. (9, 0) is 9 steps away, so it exposes
        // cells that the origin stone alone cannot reach.
        board.place((9, 0), Player::P2).unwrap();
        let moves_after = legal_moves(&board, 8);

        assert!(
            moves_after.len() > moves_before.len(),
            "placing a stone at (9,0) should expand the legal move set"
        );
    }

    #[test]
    fn smaller_radius_4_yields_60_moves() {
        let board = Board::new();
        let moves = legal_moves(&board, 4);
        assert_eq!(
            moves.len(),
            60,
            "expected 60 legal moves for radius 4 (61 cells - 1 occupied), got {}",
            moves.len()
        );
    }
}
