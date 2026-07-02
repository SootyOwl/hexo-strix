use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use hexo_engine::game::GameConfig;
use hexo_engine::GameState;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Build a deterministic game state with approximately `target_stones` placed stones.
///
/// Uses a simple seeded pseudo-random selection based on `DefaultHasher`. The
/// `seed` value is mixed with a counter to produce a deterministic sequence of
/// index choices. If the game terminates before the target is reached we return
/// whatever state we have.
fn build_game_state(target_stones: usize, config: GameConfig) -> GameState {
    let mut game = GameState::with_config(config);
    // P1's opening stone at (0,0) already counts as 1 stone.
    let mut counter: u64 = 42;

    while game.placed_stones().len() < target_stones && !game.is_terminal() {
        let moves = game.legal_moves();
        if moves.is_empty() {
            break;
        }
        // Derive a deterministic index from counter using DefaultHasher with seed 42.
        let mut hasher = DefaultHasher::new();
        42u64.hash(&mut hasher);
        counter.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % moves.len();
        counter += 1;

        let coord = moves[idx];
        // Ignore errors (should not happen with legal moves)
        let _ = game.apply_move(coord);
    }

    game
}

fn bench_legal_moves_50(c: &mut Criterion) {
    let game = build_game_state(50, GameConfig::FULL_HEXO);
    c.bench_function("legal_moves_50_stones", |b| {
        b.iter(|| {
            criterion::black_box(game.legal_moves());
        });
    });
}

fn bench_legal_moves_100(c: &mut Criterion) {
    let game = build_game_state(100, GameConfig::FULL_HEXO);
    c.bench_function("legal_moves_100_stones", |b| {
        b.iter(|| {
            criterion::black_box(game.legal_moves());
        });
    });
}

fn bench_apply_move_50(c: &mut Criterion) {
    let game = build_game_state(50, GameConfig::FULL_HEXO);
    c.bench_function("apply_move_50_stones", |b| {
        b.iter_batched(
            || {
                // Setup: clone the state and pick a legal move.
                let state = game.clone();
                let moves = state.legal_moves();
                let coord = moves[0];
                (state, coord)
            },
            |(mut state, coord)| {
                criterion::black_box(state.apply_move(coord))
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_clone_50(c: &mut Criterion) {
    let game = build_game_state(50, GameConfig::FULL_HEXO);
    c.bench_function("clone_50_stones", |b| {
        b.iter(|| {
            criterion::black_box(game.clone());
        });
    });
}

fn bench_clone_100(c: &mut Criterion) {
    let game = build_game_state(100, GameConfig::FULL_HEXO);
    c.bench_function("clone_100_stones", |b| {
        b.iter(|| {
            criterion::black_box(game.clone());
        });
    });
}

fn bench_mcts_cycle_clone_50(c: &mut Criterion) {
    let game = build_game_state(50, GameConfig::FULL_HEXO);
    c.bench_function("mcts_clone_cycle_50_stones", |b| {
        b.iter_batched(
            || {
                let state = game.clone();
                let moves = state.legal_moves();
                let coord = moves[0];
                (state, coord)
            },
            |(mut state, coord)| {
                criterion::black_box(state.apply_move(coord))
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_legal_moves_50,
    bench_legal_moves_100,
    bench_apply_move_50,
    bench_clone_50,
    bench_clone_100,
    bench_mcts_cycle_clone_50,
);
criterion_main!(benches);
