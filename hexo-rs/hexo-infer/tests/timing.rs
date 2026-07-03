//! Informational release-mode timing (native). Run manually:
//!   HEXO_REAL_WEIGHTS=... cargo test -p hexo-infer --release --test timing -- --nocapture --ignored
use hexo_infer::InferModel;
use hexo_rs::hexo_engine::{GameConfig, GameState};
use hexo_rs::mcts::gumbel_mcts::gumbel_mcts;
use hexo_rs::mcts::MCTSConfig;
use rand::SeedableRng;

#[test]
#[ignore = "informational timing; run with --ignored"]
fn real_model_best_move_timing() {
    let Ok(path) = std::env::var("HEXO_REAL_WEIGHTS") else { return };
    let model = InferModel::from_safetensors(&std::fs::read(path).unwrap()).unwrap();
    let game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 6, max_moves: 300 });
    let config = MCTSConfig {
        n_simulations: 64, m_actions: 16, c_visit: 50, c_scale: 1.0,
        disable_gumbel_noise: true, ..Default::default()
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let mut eval = |states: &[GameState]| model.eval_states(states);
    let t0 = std::time::Instant::now();
    let result = gumbel_mcts(&game, &config, &mut rng, None, &mut eval).unwrap();
    let dt = t0.elapsed();
    eprintln!("real model best_move @ sims=64 (radius 6, native release): {dt:?} -> {:?}", result.action);
}
