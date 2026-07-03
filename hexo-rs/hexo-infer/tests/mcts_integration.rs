//! hexo-infer as the `gumbel_mcts` evaluator: end-to-end native search test.

use hexo_infer::InferModel;
use hexo_rs::hexo_engine::{GameConfig, GameState};
use hexo_rs::mcts::gumbel_mcts::gumbel_mcts; // NOT re-exported at mcts::
use hexo_rs::mcts::MCTSConfig;
use rand::SeedableRng;

fn tiny_model() -> InferModel {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.safetensors");
    InferModel::from_safetensors(&std::fs::read(p).unwrap()).unwrap()
}

#[test]
fn tiny_model_plays_a_legal_move() {
    let model = tiny_model();
    let game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
    // c_visit/c_scale MUST be set: Default zeroes them and sigma(Q) collapses to 0.
    let config = MCTSConfig {
        n_simulations: 32,
        m_actions: 8,
        c_visit: 50,
        c_scale: 1.0,
        disable_gumbel_noise: true,
        ..Default::default()
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let mut eval = |states: &[GameState]| model.eval_states(states);
    let result = gumbel_mcts(&game, &config, &mut rng, None, &mut eval).unwrap();
    assert!(game.legal_moves().contains(&result.action));
    assert_eq!(result.improved_policy.len(), game.legal_moves().len());
    // Improved policy is a normalized distribution.
    let sum: f64 = result.improved_policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "improved policy sums to {sum}");
}

/// With disable_gumbel_noise + no dirichlet, the search consumes no randomness:
/// identical seeds AND different seeds must give the same move (seed is inert).
#[test]
fn search_is_deterministic() {
    let model = tiny_model();
    let game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
    let config = MCTSConfig {
        n_simulations: 32,
        m_actions: 8,
        c_visit: 50,
        c_scale: 1.0,
        disable_gumbel_noise: true,
        ..Default::default()
    };
    let mut run = |seed: u64| {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut eval = |states: &[GameState]| model.eval_states(states);
        gumbel_mcts(&game, &config, &mut rng, None, &mut eval).unwrap().action
    };
    let a42 = run(42);
    assert_eq!(a42, run(42), "same seed must repeat");
    assert_eq!(a42, run(99), "different seed must not matter (noise disabled)");
}

/// Search must actually influence selection: the improved policy differs from the
/// raw prior softmax (sigma(Q) sharpening) — guards against a future refactor
/// zeroing c_scale via `..Default::default()`.
#[test]
fn search_sharpens_the_prior() {
    let model = tiny_model();
    let game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
    let config = MCTSConfig {
        n_simulations: 64,
        m_actions: 16,
        c_visit: 50,
        c_scale: 1.0,
        disable_gumbel_noise: true,
        ..Default::default()
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let mut eval = |states: &[GameState]| model.eval_states(states);
    let result = gumbel_mcts(&game, &config, &mut rng, None, &mut eval).unwrap();

    // Raw prior: softmax over the root logits (same order as legal_moves()).
    let (maps, _) = model.eval_states(std::slice::from_ref(&game));
    let legal = game.legal_moves();
    let logits: Vec<f64> = legal.iter().map(|c| maps[0][c]).collect();
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    let prior: Vec<f64> = exps.iter().map(|e| e / z).collect();

    let diff: f64 = result
        .improved_policy
        .iter()
        .zip(prior.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-6, "improved policy identical to prior — search had no effect (diff {diff:e})");
}