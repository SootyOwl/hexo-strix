//! Per-node threat features for graph construction.
//!
//! For a node and each of the 3 win axes, slide a `win_length` window over
//! the 2*win_length-1 cells centred on the node (every window containing the
//! node) and find, per player, the best stone count among windows free of the
//! other player's stones. A clean window with >= win_length-2 stones is a
//! "win this turn" line (turns are two placements).

use std::collections::HashMap;

use crate::types::{Coord, Player, WIN_AXES};

/// Returns [own_max_line, opp_max_line, own_threat_axes, opp_threat_axes],
/// normalised by win_length (lines) and 3 (axes). "Own" = `to_move`.
pub fn node_threat_features(
    stones: &HashMap<Coord, Player>,
    coord: Coord,
    to_move: Player,
    win_length: u8,
) -> [f32; 4] {
    debug_assert!(
        (3..=7).contains(&win_length),
        "win_length {win_length} outside supported 3..=7"
    );
    let wl = win_length as i32;
    let n_cells = (2 * wl - 1) as usize; // at most 13 for wl <= 7
    let opp = to_move.opponent();
    let mut own_max = 0i32;
    let mut opp_max = 0i32;
    let mut own_axes = 0i32;
    let mut opp_axes = 0i32;
    for &(dq, dr) in &WIN_AXES {
        let mut cells = [None::<Player>; 13];
        for (i, k) in (-(wl - 1)..wl).enumerate() {
            cells[i] = stones.get(&(coord.0 + k * dq, coord.1 + k * dr)).copied();
        }
        let cells = &cells[..n_cells];
        let mut axis_own = 0i32;
        let mut axis_opp = 0i32;
        for start in 0..wl as usize {
            let window = &cells[start..start + wl as usize];
            let own_n = window.iter().filter(|c| **c == Some(to_move)).count() as i32;
            let opp_n = window.iter().filter(|c| **c == Some(opp)).count() as i32;
            if opp_n == 0 {
                axis_own = axis_own.max(own_n);
            }
            if own_n == 0 {
                axis_opp = axis_opp.max(opp_n);
            }
        }
        own_max = own_max.max(axis_own);
        opp_max = opp_max.max(axis_opp);
        if axis_own >= wl - 2 {
            own_axes += 1;
        }
        if axis_opp >= wl - 2 {
            opp_axes += 1;
        }
    }
    [
        own_max as f32 / wl as f32,
        opp_max as f32 / wl as f32,
        own_axes as f32 / 3.0,
        opp_axes as f32 / 3.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Player;
    use std::collections::HashMap;

    fn stones(items: &[((i32, i32), Player)]) -> HashMap<Coord, Player> {
        items.iter().copied().collect()
    }

    #[test]
    fn empty_board_all_zero() {
        let s = stones(&[]);
        assert_eq!(node_threat_features(&s, (0, 0), Player::P1, 4), [0.0; 4]);
    }

    #[test]
    fn three_own_in_row_wl4() {
        // P1 to move; P1 stones at (1,0),(2,0),(3,0). Node (0,0): the window
        // [(0,0)..(3,0)] along axis (1,0) holds 3 own stones, no opponent.
        let s = stones(&[((1, 0), Player::P1), ((2, 0), Player::P1), ((3, 0), Player::P1)]);
        let f = node_threat_features(&s, (0, 0), Player::P1, 4);
        assert_eq!(f[0], 3.0 / 4.0);          // own_max_line
        assert_eq!(f[1], 0.0);                 // opp_max_line (P2 has no stones)
        assert_eq!(f[2], 1.0 / 3.0);          // own_threat_axes: 3 >= 4-2 on one axis
        assert_eq!(f[3], 0.0);
    }

    #[test]
    fn same_position_from_p2_perspective_swaps_own_opp() {
        let s = stones(&[((1, 0), Player::P1), ((2, 0), Player::P1), ((3, 0), Player::P1)]);
        let f = node_threat_features(&s, (0, 0), Player::P2, 4);
        assert_eq!(f[0], 0.0);
        assert_eq!(f[1], 3.0 / 4.0);
        assert_eq!(f[2], 0.0);
        assert_eq!(f[3], 1.0 / 3.0);
    }

    #[test]
    fn opponent_stone_blocks_window() {
        // P1 at (1,0),(3,0); P2 at (2,0). Best clean own window through (0,0)
        // holds only the P1 stone at (1,0) (window [(-2,0)..(1,0)]). Every
        // window that contains P2's stone at (2,0) also contains P1's stone
        // at (1,0), so no P2-clean window has any P2 stones.
        let s = stones(&[((1, 0), Player::P1), ((3, 0), Player::P1), ((2, 0), Player::P2)]);
        let f = node_threat_features(&s, (0, 0), Player::P1, 4);
        assert_eq!(f[0], 1.0 / 4.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.0);  // 1 < 2
        assert_eq!(f[3], 0.0);
    }

    #[test]
    fn stone_node_counts_itself() {
        // The node itself is a P1 stone with one neighbour: counts of 2.
        let s = stones(&[((0, 0), Player::P1), ((1, 0), Player::P1)]);
        let f = node_threat_features(&s, (0, 0), Player::P1, 4);
        assert_eq!(f[0], 2.0 / 4.0);
        assert_eq!(f[2], 1.0 / 3.0);  // 2 >= 2
    }

    #[test]
    fn mixed_window_counts_for_neither_player() {
        // P1 at (0,0),(1,0); P2 at (2,0); query (0,0), to_move=P1, wl=4.
        // Axis (1,0) windows through (0,0):
        //   [(-3,0)..(0,0)] own=1 opp=0 -> clean, 1
        //   [(-2,0)..(1,0)] own=2 opp=0 -> clean, 2
        //   [(-1,0)..(2,0)] own=2 opp=1 -> mixed, counts for NEITHER
        //   [( 0,0)..(3,0)] own=2 opp=1 -> mixed, counts for NEITHER
        // axis_own=2. axis_opp=0: every window contains (0,0)=P1, so no
        // P1-free window exists for P2 anywhere on this axis.
        let s = stones(&[((0, 0), Player::P1), ((1, 0), Player::P1), ((2, 0), Player::P2)]);
        let f = node_threat_features(&s, (0, 0), Player::P1, 4);
        assert_eq!(f[0], 2.0 / 4.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 1.0 / 3.0); // 2 >= 2 on axis (1,0) only
        assert_eq!(f[3], 0.0);
    }

    #[test]
    fn multi_axis_max_and_axis_count() {
        // Axis (1,0): P1 at (1,0),(2,0),(3,0) -> clean window count 3.
        // Axis (0,1): P1 at (0,1),(0,2)       -> clean window count 2.
        // f[0] takes the larger (3); both axes are >= wl-2 = 2, so f[2]=2/3.
        let s = stones(&[
            ((1, 0), Player::P1),
            ((2, 0), Player::P1),
            ((3, 0), Player::P1),
            ((0, 1), Player::P1),
            ((0, 2), Player::P1),
        ]);
        let f = node_threat_features(&s, (0, 0), Player::P1, 4);
        assert_eq!(f[0], 3.0 / 4.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 2.0 / 3.0);
        assert_eq!(f[3], 0.0);
    }

    #[test]
    fn near_saturation_wl6() {
        // wl=6: 5 own stones in the clean window [(0,0)..(5,0)] -> f[0]=5/6;
        // 5 >= wl-2 = 4 on one axis -> f[2]=1/3.
        let s = stones(&[
            ((1, 0), Player::P1),
            ((2, 0), Player::P1),
            ((3, 0), Player::P1),
            ((4, 0), Player::P1),
            ((5, 0), Player::P1),
        ]);
        let f = node_threat_features(&s, (0, 0), Player::P1, 6);
        assert_eq!(f[0], 5.0 / 6.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 1.0 / 3.0);
        assert_eq!(f[3], 0.0);
    }
}
