use crate::board::Board;
use crate::types::{Coord, Player, WIN_AXES};

/// Counts consecutive stones belonging to `player` starting from `coord`
/// (exclusive), stepping in direction `dir`.
/// Stops at the first empty cell or opponent stone.
fn count_consecutive(board: &Board, coord: Coord, dir: Coord, player: Player, max: u8) -> u8 {
    let mut count = 0u8;
    let mut cur = (coord.0 + dir.0, coord.1 + dir.1);
    while count < max {
        match board.get(cur) {
            Some(p) if p == player => {
                count += 1;
                cur = (cur.0 + dir.0, cur.1 + dir.1);
            }
            _ => break,
        }
    }
    count
}

/// Returns true if placing at `coord` for `player` creates a run of
/// at least `win_length` stones along any of the three hex axes.
pub fn check_win(board: &Board, coord: Coord, player: Player, win_length: u8) -> bool {
    let max = win_length - 1; // max consecutive needed in any one direction
    for &(dq, dr) in &WIN_AXES {
        let fwd = count_consecutive(board, coord, (dq, dr), player, max);
        let bwd = count_consecutive(board, coord, (-dq, -dr), player, max.saturating_sub(fwd));
        if 1 + fwd + bwd >= win_length {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a board from a list of (coord, player) pairs.
    /// Note: Board::new() always places P1 at (0,0), so we start from a
    /// scratch-like board by working with coordinates that don't collide,
    /// or we accept the implicit P1 stone at (0,0) and work around it.
    /// For isolation we build a fresh board and place exactly the stones we want.
    fn build_board(stones: &[(Coord, Player)]) -> Board {
        // Board::new() places P1 at (0,0). We create a clean slate by using a
        // helper that bypasses that — but Board::new() is the only constructor,
        // so we use it and then explicitly include (0,0) in our test layouts
        // (placed by new()) or use coordinates that don't include (0,0).
        // Simplest: just call new() and place the extra stones, being careful
        // not to double-place (0,0).
        let mut board = Board::new();
        for &(coord, player) in stones {
            if coord == (0, 0) {
                // (0,0) is already P1; only skip if it's the same player.
                // If it differs this is a test-setup error.
                debug_assert_eq!(player, Player::P1, "Board::new puts P1 at origin");
                continue;
            }
            board.place(coord, player).expect("test: duplicate stone in build_board");
        }
        board
    }

    // ---- count_consecutive tests ----

    #[test]
    fn count_consecutive_empty_returns_zero() {
        let board = build_board(&[]);
        // Walking right from (0,0) — the cell to the right (1,0) is empty.
        let n = count_consecutive(&board, (0, 0), (1, 0), Player::P1, u8::MAX);
        assert_eq!(n, 0);
    }

    #[test]
    fn count_consecutive_three_in_a_row() {
        // P1 stones at (1,0), (2,0), (3,0); start from (0,0) going right.
        let board = build_board(&[
            ((0, 0), Player::P1),
            ((1, 0), Player::P1),
            ((2, 0), Player::P1),
            ((3, 0), Player::P1),
        ]);
        let n = count_consecutive(&board, (0, 0), (1, 0), Player::P1, u8::MAX);
        assert_eq!(n, 3);
    }

    #[test]
    fn count_consecutive_stops_at_opponent() {
        // P1 at (1,0), P2 at (2,0).
        let board = build_board(&[
            ((0, 0), Player::P1),
            ((1, 0), Player::P1),
            ((2, 0), Player::P2),
            ((3, 0), Player::P1),
        ]);
        let n = count_consecutive(&board, (0, 0), (1, 0), Player::P1, u8::MAX);
        assert_eq!(n, 1); // only (1,0) before opponent
    }

    #[test]
    fn count_consecutive_stops_at_empty_gap() {
        // P1 at (1,0), gap at (2,0), P1 at (3,0).
        let board = build_board(&[
            ((0, 0), Player::P1),
            ((1, 0), Player::P1),
            ((3, 0), Player::P1),
        ]);
        let n = count_consecutive(&board, (0, 0), (1, 0), Player::P1, u8::MAX);
        assert_eq!(n, 1); // stops at gap (2,0)
    }

    // ---- check_win tests ----

    #[test]
    fn check_win_single_stone_no_win() {
        // Board::new() only has P1 at (0,0); win_length=6 → no win.
        let board = build_board(&[((0, 0), Player::P1)]);
        assert!(!check_win(&board, (0, 0), Player::P1, 6));
    }

    #[test]
    fn check_win_six_in_a_row_axis_q() {
        // 6 P1 stones along (1,0) axis: (0,0)..(5,0)
        let board = build_board(&[
            ((0, 0), Player::P1),
            ((1, 0), Player::P1),
            ((2, 0), Player::P1),
            ((3, 0), Player::P1),
            ((4, 0), Player::P1),
            ((5, 0), Player::P1),
        ]);
        assert!(check_win(&board, (3, 0), Player::P1, 6));
    }

    #[test]
    fn check_win_six_in_a_row_axis_r() {
        // 6 P1 stones along (0,1) axis: (10,0)..(10,5)
        let board = build_board(&[
            ((10, 0), Player::P1),
            ((10, 1), Player::P1),
            ((10, 2), Player::P1),
            ((10, 3), Player::P1),
            ((10, 4), Player::P1),
            ((10, 5), Player::P1),
        ]);
        assert!(check_win(&board, (10, 2), Player::P1, 6));
    }

    #[test]
    fn check_win_six_in_a_row_axis_diagonal() {
        // 6 P1 stones along (1,-1) axis: (10,0),(11,-1),(12,-2),(13,-3),(14,-4),(15,-5)
        let board = build_board(&[
            ((10, 0), Player::P1),
            ((11, -1), Player::P1),
            ((12, -2), Player::P1),
            ((13, -3), Player::P1),
            ((14, -4), Player::P1),
            ((15, -5), Player::P1),
        ]);
        assert!(check_win(&board, (12, -2), Player::P1, 6));
    }

    #[test]
    fn check_win_five_is_not_enough_for_six() {
        // Only 5 P1 stones along q axis.
        let board = build_board(&[
            ((0, 0), Player::P1),
            ((1, 0), Player::P1),
            ((2, 0), Player::P1),
            ((3, 0), Player::P1),
            ((4, 0), Player::P1),
        ]);
        assert!(!check_win(&board, (2, 0), Player::P1, 6));
    }

    #[test]
    fn check_win_seven_also_wins() {
        // 7 P1 stones along q axis; win_length=6 → still wins (AC-10c).
        let board = build_board(&[
            ((0, 0), Player::P1),
            ((1, 0), Player::P1),
            ((2, 0), Player::P1),
            ((3, 0), Player::P1),
            ((4, 0), Player::P1),
            ((5, 0), Player::P1),
            ((6, 0), Player::P1),
        ]);
        assert!(check_win(&board, (3, 0), Player::P1, 6));
    }

    #[test]
    fn check_win_wrong_player_no_win() {
        // 6 P1 stones; asking for P2 win → false (REQ-13 / AC-13).
        let board = build_board(&[
            ((0, 0), Player::P1),
            ((1, 0), Player::P1),
            ((2, 0), Player::P1),
            ((3, 0), Player::P1),
            ((4, 0), Player::P1),
            ((5, 0), Player::P1),
        ]);
        assert!(!check_win(&board, (5, 0), Player::P2, 6));
    }

    #[test]
    fn check_win_four_in_a_row_curriculum_variant() {
        // 4-in-a-row variant (AC-19b): 4 P1 stones along q axis.
        let board = build_board(&[
            ((0, 0), Player::P1),
            ((1, 0), Player::P1),
            ((2, 0), Player::P1),
            ((3, 0), Player::P1),
        ]);
        assert!(check_win(&board, (2, 0), Player::P1, 4));
    }

    #[test]
    fn check_win_three_not_enough_for_four() {
        // 3 P1 stones; win_length=4 → no win (AC-10b for 4-variant).
        let board = build_board(&[
            ((0, 0), Player::P1),
            ((1, 0), Player::P1),
            ((2, 0), Player::P1),
        ]);
        assert!(!check_win(&board, (1, 0), Player::P1, 4));
    }
}
