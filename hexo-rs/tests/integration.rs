use hexo_rs::game::{GameConfig, GameState, MoveError};
use hexo_rs::hex::hex_distance;
use hexo_rs::{Coord, Player};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Helper: create a 4-in-a-row config
// ---------------------------------------------------------------------------

fn config_4() -> GameConfig {
    GameConfig {
        win_length: 4,
        placement_radius: 4,
        max_moves: 200,
    }
}

// ---------------------------------------------------------------------------
// 1. full_game_p2_wins_4_in_a_row — P2 wins along r-axis
//
// Turn structure (2 moves per player per turn):
//   P2 turn1: (0,1), (0,2)
//   P1 turn1: (2,1), (-2,-1)   [scattered]
//   P2 turn2: (0,3), (0,4)     → 4-in-a-row on r-axis → P2 wins
// ---------------------------------------------------------------------------

#[test]
fn full_game_p2_wins_4_in_a_row() {
    let mut gs = GameState::with_config(config_4());

    // P2 turn 1
    gs.apply_move((0, 1)).unwrap();
    gs.apply_move((0, 2)).unwrap();
    // P1 turn 1 (scattered — cannot accidentally form 4-in-a-row)
    gs.apply_move((2, 1)).unwrap();
    gs.apply_move((-2, -1)).unwrap();
    // P2 turn 2
    gs.apply_move((0, 3)).unwrap();
    gs.apply_move((0, 4)).unwrap(); // completes (0,1),(0,2),(0,3),(0,4)

    assert!(gs.is_terminal(), "game must be terminal after P2 wins");
    assert_eq!(gs.winner(), Some(Player::P2));
    assert_eq!(gs.current_player(), None, "no current player after game over");
}

// ---------------------------------------------------------------------------
// 2. win_mid_turn_stops_game
//
// P2 wins on the FIRST of two moves in a turn → second move must be rejected.
// P1 plays scattered (1,1), (-1,-1), (2,1), (-2,-1) to avoid any 4-in-a-row
// on the q-axis or elsewhere.
//
// Setup:
//   P2 turn1: (0,1), (0,2)
//   P1 turn1: (1,1), (-1,-1)    [scattered]
//   P2 turn2: (0,3), (2,-1)     [only 3 in r-row so far]
//   P1 turn2: (2,1), (-2,-1)    [scattered]
//   P2 turn3: (0,4) → WIN (first move of turn) → second move rejected
// ---------------------------------------------------------------------------

#[test]
fn win_mid_turn_stops_game() {
    let mut gs = GameState::with_config(config_4());

    // P2 turn 1 — lay 2 on r-axis
    gs.apply_move((0, 1)).unwrap();
    gs.apply_move((0, 2)).unwrap();
    // P1 turn 1 — scattered
    gs.apply_move((1, 1)).unwrap();
    gs.apply_move((-1, -1)).unwrap();
    // P2 turn 2 — one more on r-axis + a scattered stone
    gs.apply_move((0, 3)).unwrap();
    gs.apply_move((2, -1)).unwrap(); // scattered, not on r-axis
    // P1 turn 2 — scattered
    gs.apply_move((2, 1)).unwrap();
    gs.apply_move((-2, -1)).unwrap();
    // P2 turn 3 — first move should win
    assert_eq!(
        gs.moves_remaining_this_turn(),
        2,
        "P2 should have 2 moves left at start of turn 3"
    );
    gs.apply_move((0, 4)).unwrap(); // (0,1)+(0,2)+(0,3)+(0,4) → WIN

    assert!(gs.is_terminal(), "game must be terminal immediately after win");
    assert_eq!(gs.winner(), Some(Player::P2));

    // The second move of that turn must be rejected.
    let result = gs.apply_move((0, -1));
    assert_eq!(
        result,
        Err(MoveError::GameOver),
        "second move of winning turn must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 3. p1_can_win — P1 completes 4-in-a-row on diagonal axis (1,-1)
//
// P1 starts at (0,0). Diagonal axis: (1,-1),(2,-2),(3,-3).
// P2 plays scattered to avoid interfering or accidentally winning.
//
// Turn structure:
//   P2 turn1: (0,2), (0,-2)     [scattered]
//   P1 turn1: (1,-1), (2,-2)
//   P2 turn2: (2,2), (-2,2)     [scattered]
//   P1 turn2: (3,-3) → P1 has (0,0),(1,-1),(2,-2),(3,-3) = 4-in-a-row → wins
// ---------------------------------------------------------------------------

#[test]
fn p1_can_win() {
    let mut gs = GameState::with_config(config_4());

    // P2 turn 1 — scattered (r-axis safe distances)
    gs.apply_move((0, 2)).unwrap();
    gs.apply_move((0, -2)).unwrap();
    // P1 turn 1 — start diagonal
    gs.apply_move((1, -1)).unwrap();
    gs.apply_move((2, -2)).unwrap();
    // P2 turn 2 — scattered
    gs.apply_move((2, 2)).unwrap();
    gs.apply_move((-2, 2)).unwrap();
    // P1 turn 2 — complete the diagonal; only needs 1 move to win
    gs.apply_move((3, -3)).unwrap(); // (0,0),(1,-1),(2,-2),(3,-3) → WIN

    assert!(gs.is_terminal(), "game must be terminal after P1 wins on diagonal");
    assert_eq!(gs.winner(), Some(Player::P1));
}

// ---------------------------------------------------------------------------
// 4. no_false_positive_win_l_shape
//
// Place 5 P2 stones in an L-shape so no 4 are collinear.
// Verify that no win is declared after any of those placements.
//
// L-shape on q-axis + r-axis bend:
//   (1,0),(2,0),(3,0) — 3 on q-axis
//   (1,1),(1,2)       — 2 on r-stem from (1,0)
//
// Turn structure (P2 gets 2 per turn, P1 scattered between):
//   P2 turn1: (1,0), (2,0)
//   P1 turn1: (0,3), (0,-3)
//   P2 turn2: (3,0), (1,1)
//   P1 turn2: (-3,0) -- safe distance
//             need a second P1 move here, use (-3, 1)
//             Actually P1 has 2 moves per turn.
//   P1 turn2: (-3,0), (-3,1)  -- both scattered
//   P2 turn3: (1,2), ...      -- only need one more P2 stone
// ---------------------------------------------------------------------------

#[test]
fn no_false_positive_win_l_shape() {
    let mut gs = GameState::with_config(config_4());

    // P2 turn 1
    gs.apply_move((1, 0)).unwrap();
    gs.apply_move((2, 0)).unwrap();
    assert!(!gs.is_terminal());
    assert_eq!(gs.winner(), None);

    // P1 turn 1 — scattered
    gs.apply_move((0, 3)).unwrap();
    gs.apply_move((0, -3)).unwrap();

    // P2 turn 2
    gs.apply_move((3, 0)).unwrap(); // 3rd on q-axis (not yet 4)
    assert!(!gs.is_terminal(), "3 in a row should not trigger win");
    gs.apply_move((1, 1)).unwrap(); // branch off
    assert!(!gs.is_terminal(), "L-bend, still no win");

    // P1 turn 2 — scattered
    gs.apply_move((-3, 0)).unwrap();
    gs.apply_move((-3, 1)).unwrap();

    // P2 turn 3 — add 5th stone to L
    gs.apply_move((1, 2)).unwrap(); // extends r-stem: (1,0),(1,1),(1,2) = 3 on r, no win
    assert!(
        !gs.is_terminal(),
        "5 P2 stones in L-shape must not trigger a win"
    );
    assert_eq!(
        gs.winner(),
        None,
        "no winner expected for L-shape arrangement"
    );
}

// ---------------------------------------------------------------------------
// 5. legal_moves_all_in_range
//
// After a few moves, verify that every coord in legal_moves() is within
// placement_radius of at least one placed stone.
// ---------------------------------------------------------------------------

#[test]
fn legal_moves_all_in_range() {
    let mut gs = GameState::with_config(config_4());

    // Make a handful of moves to diversify the board
    gs.apply_move((1, 0)).unwrap();
    gs.apply_move((-1, 0)).unwrap();
    gs.apply_move((0, 1)).unwrap();
    gs.apply_move((0, -1)).unwrap();

    let stones: Vec<Coord> = gs.placed_stones().into_iter().map(|(c, _)| c).collect();
    let radius = gs.config().placement_radius;

    for mv in gs.legal_moves() {
        let in_range = stones
            .iter()
            .any(|&s| hex_distance(s, mv) <= radius);
        assert!(
            in_range,
            "legal move {:?} is not within radius {} of any stone",
            mv,
            radius
        );
    }
}

// ---------------------------------------------------------------------------
// 6. win_on_final_move_is_win_not_draw
//
// A winning move at exactly move_count == max_moves should be a win, not draw.
// ---------------------------------------------------------------------------

#[test]
fn win_on_final_move_is_win_not_draw() {
    // max_moves = 6, win_length = 4. P2 wins on the 6th move.
    let config = GameConfig {
        win_length: 4,
        placement_radius: 8,
        max_moves: 6,
    };
    let mut gs = GameState::with_config(config);
    // P2 builds along r-axis
    gs.apply_move((0, 1)).unwrap();  // P2 move 1
    gs.apply_move((0, 2)).unwrap();  // P2 move 2
    gs.apply_move((1, 1)).unwrap();  // P1 move 3 (scattered)
    gs.apply_move((-1, -1)).unwrap(); // P1 move 4 (scattered)
    gs.apply_move((0, 3)).unwrap();  // P2 move 5 — 3 in a row
    // Move 6 = max_moves. P2 wins with 4-in-a-row. Should be win, not draw.
    gs.apply_move((0, 4)).unwrap();  // P2 move 6 — 4 in a row!
    assert!(gs.is_terminal());
    assert_eq!(gs.winner(), Some(Player::P2), "win should take precedence over draw");
}

// ---------------------------------------------------------------------------
// 7. fuzz_random_games_no_panics
//
// 500 random 4-in-a-row games (radius 4), deterministic seeding.
// Assert no panics.
// ---------------------------------------------------------------------------

fn pick_move(moves: &[Coord], seed: u64) -> Coord {
    let mut h = DefaultHasher::new();
    seed.hash(&mut h);
    let idx = (h.finish() as usize) % moves.len();
    moves[idx]
}

#[test]
fn fuzz_random_games_no_panics() {
    let fuzz_config = GameConfig {
        win_length: 4,
        placement_radius: 4,
        max_moves: 200,
    };

    for seed in 0u64..500 {
        let mut gs = GameState::with_config(GameConfig {
            win_length: fuzz_config.win_length,
            placement_radius: fuzz_config.placement_radius,
            max_moves: fuzz_config.max_moves,
        });
        let mut step = 0u64;
        while !gs.is_terminal() {
            let moves = gs.legal_moves();
            if moves.is_empty() {
                break;
            }
            let mv = pick_move(&moves, seed ^ (step << 32));
            gs.apply_move(mv).expect("legal move must not fail");
            step += 1;
        }
        // Invariant: terminal games must have a winner or be a draw
        if gs.is_terminal() {
            let stones = gs.placed_stones().len();
            assert!(
                stones >= 2,
                "seed {seed}: terminal game has only {stones} stones"
            );
            assert_eq!(
                gs.move_count() as usize,
                stones - 1, // minus P1's opening stone
                "seed {seed}: move_count doesn't match placed stones"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 7. fuzz_random_games_full_hexo
//
// 100 random full HeXO games, assert no panics.
// ---------------------------------------------------------------------------

#[test]
fn fuzz_random_games_full_hexo() {
    for seed in 0u64..100 {
        let mut gs = GameState::new();
        let mut step = 0u64;
        while !gs.is_terminal() {
            let moves = gs.legal_moves();
            if moves.is_empty() {
                break;
            }
            let mv = pick_move(&moves, seed ^ (step << 32));
            gs.apply_move(mv).expect("legal move must not fail");
            step += 1;
        }
    }
}
