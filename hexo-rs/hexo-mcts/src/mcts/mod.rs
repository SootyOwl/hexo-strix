pub mod acting;
// batcher.rs uses std::time::Instant (panics at runtime on wasm32-unknown-unknown) and
// is only consumed by the python-feature path + the self_play bin, never the serial
// gumbel_mcts search. Gate it off wasm so a future wiring can't reintroduce a runtime
// panic on the target — a compile-time guarantee stronger than the grep invariant.
#[cfg(not(target_arch = "wasm32"))]
pub mod batcher;
pub mod backup;
pub mod batched;
pub mod forcing;
pub mod gumbel_mcts;
pub mod halving;
pub mod node;
pub mod scoring;
pub mod select;
pub mod simulate;

#[cfg(feature = "dedup_count")]
pub mod dedup_count;

/// MCTS configuration parameters.
///
/// `Default` is derived for ergonomic test construction via
/// `MCTSConfig { n_simulations: 64, m_actions: 16, ..Default::default() }`,
/// but the default values for `n_simulations` and `m_actions` are 0 and not
/// usable at runtime — always set these explicitly in real configurations.
#[derive(Clone, Default)]
pub struct MCTSConfig {
    pub n_simulations: u32,
    pub m_actions: usize,
    pub c_visit: u32,
    pub c_scale: f64,
    /// Virtual-loss magnitude used during SH inner-loop fusion. 0.0 disables
    /// fusion (serial sims, current behaviour); >0.0 enables leaf-parallel
    /// MCTS within each SH phase. Typical range 0.3–1.0.
    pub virtual_loss: f64,
    /// Dirichlet noise concentration α applied to root priors before
    /// Gumbel-Top-k candidate sampling. 0.0 disables (bit-equivalent to
    /// pre-noise behaviour). For HeXO with ~200 legal moves at full board,
    /// α ≈ 0.05 is a reasonable starting point (AlphaZero heuristic
    /// α ≈ 10 / branching_factor). Lower α → more concentrated noise
    /// (sparser exploration boost). Higher α → flatter noise (uniform-ish).
    pub root_dirichlet_alpha: f64,
    /// Mix fraction ε for Dirichlet noise at the root.
    /// `mixed_prior = (1 - ε) * network_prior + ε * Dirichlet(α)`.
    /// 0.0 disables (no noise applied regardless of α). Standard AlphaZero
    /// uses 0.25. Ignored when `root_dirichlet_alpha == 0.0`.
    pub root_dirichlet_fraction: f64,
    /// If > 0, at mr=2 roots, additionally batch-evaluates the post-p₁ state
    /// for each Gumbel-Top-K candidate p₁ and exposes the *chosen* p₁'s
    /// top-K p₂s by conditional prior in
    /// `MCTSResult.chosen_action_forced_candidates`. The caller is expected
    /// to pass these as `forced_candidates` to the next `gumbel_mcts` call
    /// (the p₂ search). 0 (default) disables capture entirely (bit-equivalent
    /// to current behaviour).
    pub forced_candidate_capture_k: usize,
    /// If `true`, skip the Gumbel(0,1) draws in root candidate sampling so the
    /// search is fully deterministic: candidates become the top-`m_actions`
    /// actions by logit and the final pick is `argmax(logit + σ(Q))`. This is
    /// the paper's zero-noise evaluation behaviour. `false` (the derived
    /// default) keeps Gumbel noise on — bit-equivalent to prior behaviour, so
    /// self-play is unaffected. Note the improved-policy *target* never used
    /// the Gumbel noise, so this only changes which action is played.
    pub disable_gumbel_noise: bool,
    /// If `true`, skip the VCF forcing-solver (`forcing::solve`) shortcut run
    /// at mr=2 roots, so search falls through to normal Gumbel MCTS instead of
    /// short-circuiting on a proven forced win. `false` (the derived default)
    /// keeps the solver shortcut ON — bit-equivalent to prior behaviour, so
    /// self-play is unaffected. Only gates the multi-turn `forcing::solve`
    /// call; the depth-1 terminal-win shortcut ("if I can win this move, win")
    /// always stays on.
    pub disable_forcing_solver: bool,
    /// Runtime override for the VCF forcing-solver iterative-deepening depth
    /// cap (in attacker *turns*) used by the mr>=1 shortcut. `0` (the derived
    /// default) is a sentinel meaning "use the compile-time
    /// `gumbel_mcts::SELF_PLAY_DEPTH_CAP` constant" — so an unset config is
    /// bit-identical to prior behaviour. Any non-zero value caps the search at
    /// that depth instead. Like `n_simulations`/`m_actions`, the raw `0` is not
    /// itself usable as a real cap at runtime; it only selects the default.
    pub forcing_depth_cap: u8,
    /// Runtime override for the VCF forcing-solver per-position node budget
    /// (the hard ceiling on solve effort) used by the mr>=1 shortcut. `0` (the
    /// derived default) is a sentinel meaning "use the compile-time
    /// `gumbel_mcts::SELF_PLAY_NODE_BUDGET` constant" — so an unset config is
    /// bit-identical to prior behaviour. Any non-zero value uses that budget
    /// instead. As with `forcing_depth_cap`, raw `0` only selects the default.
    pub forcing_node_budget: u64,
}
