use rustc_hash::FxHashMap as HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use hexo_engine::game::{GameConfig, GameState, MoveError};
use hexo_engine::types::{Coord, Player};

use crate::mcts::gumbel_mcts;
use crate::mcts::MCTSConfig;

fn player_str(p: Player) -> &'static str {
    match p {
        Player::P1 => "P1",
        Player::P2 => "P2",
    }
}

#[pyclass(name = "GameConfig", module = "hexo_rs", skip_from_py_object)]
#[derive(Clone)]
struct PyGameConfig {
    inner: GameConfig,
}

#[pymethods]
impl PyGameConfig {
    #[new]
    fn new(win_length: u8, placement_radius: i32, max_moves: u32) -> PyResult<Self> {
        if win_length < 2 {
            return Err(PyValueError::new_err("win_length must be >= 2"));
        }
        if placement_radius < 1 {
            return Err(PyValueError::new_err("placement_radius must be >= 1"));
        }
        if max_moves < 1 {
            return Err(PyValueError::new_err("max_moves must be >= 1"));
        }
        Ok(PyGameConfig {
            inner: GameConfig { win_length, placement_radius, max_moves },
        })
    }

    #[staticmethod]
    fn full_hexo() -> Self {
        PyGameConfig { inner: GameConfig::FULL_HEXO }
    }

    #[getter]
    fn win_length(&self) -> u8 {
        self.inner.win_length
    }

    #[getter]
    fn placement_radius(&self) -> i32 {
        self.inner.placement_radius
    }

    #[getter]
    fn max_moves(&self) -> u32 {
        self.inner.max_moves
    }

    fn __repr__(&self) -> String {
        format!(
            "GameConfig(win_length={}, placement_radius={}, max_moves={})",
            self.inner.win_length, self.inner.placement_radius, self.inner.max_moves,
        )
    }

    /// Pickle support: reconstruct via the regular constructor.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cls = py.get_type::<PyGameConfig>();
        let args = (
            self.inner.win_length,
            self.inner.placement_radius,
            self.inner.max_moves,
        );
        Ok((cls, args).into_pyobject(py)?.unbind().into())
    }
}

#[pyclass(name = "GameState", module = "hexo_rs")]
struct PyGameState {
    inner: GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyGameConfig>) -> Self {
        let inner = match config {
            Some(cfg) => GameState::with_config(cfg.inner),
            None => GameState::new(),
        };
        PyGameState { inner }
    }

    fn apply_move(&mut self, q: i32, r: i32) -> PyResult<()> {
        self.inner.apply_move((q, r)).map_err(|e| match e {
            MoveError::GameOver => PyValueError::new_err("game is over"),
            MoveError::CellOccupied => PyValueError::new_err("cell is occupied"),
            MoveError::OutOfRange => PyValueError::new_err("move is out of range"),
        })
    }

    /// Construct a GameState directly from a list of placed stones.
    ///
    /// Bypasses turn-order and placement-radius enforcement, so the caller can
    /// build arbitrary positions for puzzle / setup workflows. The board is
    /// always seeded with P1 at (0,0); a redundant (0,0)=P1 entry in `stones`
    /// is silently ignored.
    ///
    /// Args:
    ///     stones: list of ((q, r), "P1"|"P2") pairs.
    ///     current_player: "P1" or "P2" — whose turn it is in the resulting state.
    ///     moves_remaining: 1 or 2 — placements left this turn.
    ///     config: GameConfig (optional, defaults to FULL_HEXO).
    #[staticmethod]
    #[pyo3(signature = (stones, current_player, moves_remaining=2, config=None))]
    fn from_state(
        stones: Vec<((i32, i32), String)>,
        current_player: String,
        moves_remaining: u8,
        config: Option<&PyGameConfig>,
    ) -> PyResult<Self> {
        if moves_remaining != 1 && moves_remaining != 2 {
            return Err(PyValueError::new_err("moves_remaining must be 1 or 2"));
        }
        let parse_player = |s: &str| -> PyResult<Player> {
            match s {
                "P1" => Ok(Player::P1),
                "P2" => Ok(Player::P2),
                other => Err(PyValueError::new_err(format!(
                    "player must be 'P1' or 'P2', got {other:?}"
                ))),
            }
        };
        let cur = parse_player(&current_player)?;
        let mut typed_stones: Vec<(Coord, Player)> = Vec::with_capacity(stones.len());
        for (coord, p) in stones {
            typed_stones.push((coord, parse_player(&p)?));
        }
        let cfg = config.map(|c| c.inner).unwrap_or(GameConfig::FULL_HEXO);
        let inner = GameState::from_state(&typed_stones, cur, moves_remaining, cfg);
        Ok(PyGameState { inner })
    }

    fn legal_moves(&self) -> Vec<(i32, i32)> {
        self.inner.legal_moves()
    }

    fn legal_move_count(&self) -> usize {
        self.inner.legal_move_count()
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    fn winner(&self) -> Option<&'static str> {
        self.inner.winner().map(player_str)
    }

    fn current_player(&self) -> Option<&'static str> {
        self.inner.current_player().map(player_str)
    }

    fn moves_remaining_this_turn(&self) -> u8 {
        self.inner.moves_remaining_this_turn()
    }

    fn placed_stones(&self) -> Vec<((i32, i32), &'static str)> {
        self.inner
            .placed_stones()
            .into_iter()
            .map(|(coord, player)| (coord, player_str(player)))
            .collect()
    }

    fn move_count(&self) -> u32 {
        self.inner.move_count()
    }

    fn config(&self) -> PyGameConfig {
        PyGameConfig { inner: *self.inner.config() }
    }

    /// Pickle support. Reconstructs the state via `from_state`, which only
    /// supports non-terminal positions (the engine's `from_state` doesn't run a
    /// win check). Reanalyze never targets terminal positions anyway — the
    /// trainer catches the resulting MCTS failure — so refusing to pickle them
    /// is a safe limitation.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.inner.is_terminal() {
            return Err(PyValueError::new_err(
                "Cannot pickle a terminal GameState (from_state cannot reconstruct \
                 terminal positions; reanalyze never targets terminals).",
            ));
        }
        let current_player = self
            .inner
            .current_player()
            .map(player_str)
            .expect("non-terminal state has a current player")
            .to_string();
        let moves_remaining = self.inner.moves_remaining_this_turn();
        let stones: Vec<((i32, i32), String)> = self
            .inner
            .placed_stones()
            .into_iter()
            .map(|(coord, player)| (coord, player_str(player).to_string()))
            .collect();
        let config = PyGameConfig { inner: *self.inner.config() };

        // Resolve the bound constructor at pickle time so unpickling does not
        // depend on lookup details of the module attribute.
        let cls = py.get_type::<PyGameState>();
        let constructor = cls.getattr("from_state")?;
        let args = (stones, current_player, moves_remaining, config);
        Ok((constructor, args).into_pyobject(py)?.unbind().into())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn clone(&self) -> Self {
        PyGameState { inner: self.inner.clone() }
    }

    fn __copy__(&self) -> Self {
        PyGameState { inner: self.inner.clone() }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        PyGameState { inner: self.inner.clone() }
    }

    fn __repr__(&self) -> String {
        let status = if self.inner.is_terminal() {
            match self.inner.winner() {
                Some(p) => format!("{} wins", player_str(p)),
                None => "draw".into(),
            }
        } else {
            let p = player_str(self.inner.current_player().unwrap());
            let rem = self.inner.moves_remaining_this_turn();
            format!("{p} to move, {rem} left")
        };
        format!(
            "GameState({} stones, {})",
            self.inner.placed_stones().len(),
            status,
        )
    }
}

#[pyclass(name = "MCTSConfig", skip_from_py_object)]
#[derive(Clone)]
struct PyMCTSConfig {
    inner: MCTSConfig,
}

#[pymethods]
impl PyMCTSConfig {
    #[new]
    #[pyo3(signature = (
        n_simulations,
        m_actions,
        c_visit,
        c_scale,
        virtual_loss=0.0,
        root_dirichlet_alpha=0.0,
        root_dirichlet_fraction=0.0,
        forced_candidate_capture_k=0,
        disable_gumbel_noise=false,
    ))]
    fn new(
        n_simulations: u32,
        m_actions: usize,
        c_visit: u32,
        c_scale: f64,
        virtual_loss: f64,
        root_dirichlet_alpha: f64,
        root_dirichlet_fraction: f64,
        forced_candidate_capture_k: usize,
        disable_gumbel_noise: bool,
    ) -> PyResult<Self> {
        if m_actions < 1 {
            return Err(PyValueError::new_err("m_actions must be >= 1"));
        }
        if !c_scale.is_finite() || c_scale <= 0.0 {
            return Err(PyValueError::new_err("c_scale must be a positive finite number"));
        }
        if !virtual_loss.is_finite() || virtual_loss < 0.0 {
            return Err(PyValueError::new_err(
                "virtual_loss must be a non-negative finite number",
            ));
        }
        if !root_dirichlet_alpha.is_finite() || root_dirichlet_alpha < 0.0 {
            return Err(PyValueError::new_err(
                "root_dirichlet_alpha must be a non-negative finite number",
            ));
        }
        if !root_dirichlet_fraction.is_finite()
            || !(0.0..=1.0).contains(&root_dirichlet_fraction)
        {
            return Err(PyValueError::new_err(
                "root_dirichlet_fraction must be in [0, 1]",
            ));
        }
        Ok(PyMCTSConfig {
            inner: MCTSConfig {
                n_simulations,
                m_actions,
                c_visit,
                c_scale,
                virtual_loss,
                root_dirichlet_alpha,
                root_dirichlet_fraction,
                forced_candidate_capture_k,
                disable_gumbel_noise,
            },
        })
    }
}

/// Run Gumbel MCTS from a game state.
///
/// Args:
///     game: hexo_rs.GameState (non-terminal)
///     eval_fn: callable(list[GameState]) -> tuple[list[dict[(int,int), float]], list[float]]
///     config: hexo_rs.MCTSConfig
///     seed: optional int for deterministic RNG
///
/// Returns:
///     tuple[tuple[int,int], list[float]] — (action, improved_policy)
///
/// E.4-inject forced candidates: this wrapper does NOT expose
/// `forced_candidates`. Callers that need the inject mechanism must use
/// `gumbel_mcts_with_diagnostics`.
#[pyfunction(name = "gumbel_mcts")]
#[pyo3(signature = (game, eval_fn, config, seed=None))]
fn py_gumbel_mcts(
    py: Python<'_>,
    game: &PyGameState,
    eval_fn: Py<PyAny>,
    config: &PyMCTSConfig,
    seed: Option<u64>,
) -> PyResult<((i32, i32), Vec<f64>)> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_os_rng(),
    };

    let mut eval_error: Option<PyErr> = None;

    let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        // Build list of PyGameState objects for the callback
        let py_states: Vec<PyGameState> = states
            .iter()
            .map(|s| PyGameState { inner: s.clone() })
            .collect();

        // Call the Python eval_fn
        let result = match eval_fn.call1(py, (py_states,)) {
            Ok(r) => r,
            Err(e) => {
                eval_error = Some(e);
                let dummy = states.iter().map(|_| HashMap::default()).collect();
                return (dummy, vec![0.0; states.len()]);
            }
        };

        // Parse result: (list[list[float]], list[float]) — flat logit arrays
        let parsed: Result<(Vec<Vec<f64>>, Vec<f64>), _> = result.extract(py);
        let (logits_per_state, values_raw) = match parsed {
            Ok(v) => v,
            Err(e) => {
                eval_error = Some(e.into());
                let dummy = states.iter().map(|_| HashMap::default()).collect();
                return (dummy, vec![0.0; states.len()]);
            }
        };

        // Map flat logit arrays to coord→logit HashMaps using legal_moves()
        let logits_maps: Vec<HashMap<Coord, f64>> = states
            .iter()
            .zip(logits_per_state)
            .map(|(state, logits)| {
                let moves = state.legal_moves();
                moves.into_iter().zip(logits).collect()
            })
            .collect();

        (logits_maps, values_raw)
    };

    let result = gumbel_mcts::gumbel_mcts(&game.inner, &config.inner, &mut rng, None, &mut eval)
        .map_err(PyValueError::new_err)?;

    // Check if the eval callback encountered a Python error
    if let Some(e) = eval_error {
        return Err(e);
    }

    Ok((result.action, result.improved_policy))
}

/// Run Gumbel MCTS on many positions in lockstep with fused leaf evaluation.
///
/// All searches advance together: each simulation phase gathers the pending
/// leaves of every game into one eval_fn call, so a single GPU forward pass
/// serves the whole batch.
///
/// Args:
///     games: list[hexo_rs.GameState] — all must be non-terminal
///     eval_fn: callable(list[GameState]) -> tuple[list[list[float]], list[float]]
///     config: hexo_rs.MCTSConfig
///     seed: optional int for deterministic RNG
///
/// Returns:
///     list[tuple[tuple[int,int], list[float]]] — one (action, improved_policy)
///     per input state; improved_policy is ordered like state.legal_moves()
///     (sorted by coord), matching gumbel_mcts and TrainingExample.policy_target.
#[pyfunction(name = "batched_gumbel_mcts")]
#[pyo3(signature = (games, eval_fn, config, seed=None))]
fn py_batched_gumbel_mcts(
    py: Python<'_>,
    games: Vec<PyRef<PyGameState>>,
    eval_fn: Py<PyAny>,
    config: &PyMCTSConfig,
    seed: Option<u64>,
) -> PyResult<Vec<((i32, i32), Vec<f64>)>> {
    use crate::mcts::batched::batched_gumbel_mcts;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let states: Vec<GameState> = games.iter().map(|g| g.inner.clone()).collect();
    if let Some(t) = states.iter().position(|s| s.is_terminal()) {
        return Err(PyValueError::new_err(format!(
            "batched_gumbel_mcts: state {t} is terminal"
        )));
    }

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_os_rng(),
    };

    let mut eval_error: Option<String> = None;

    let eval_fn_ref = &eval_fn;
    let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        match call_python_eval(py, eval_fn_ref, states) {
            Ok(result) => result,
            Err(e) => {
                eval_error = Some(e);
                let dummy = states.iter().map(|_| HashMap::default()).collect();
                (dummy, vec![0.0; states.len()])
            }
        }
    };

    let results = batched_gumbel_mcts(&states, &config.inner, &mut rng, &mut eval)
        .map_err(PyValueError::new_err)?;

    // Check if the eval callback encountered a Python error
    if let Some(e) = eval_error {
        return Err(if e.contains("KeyboardInterrupt") {
            pyo3::exceptions::PyKeyboardInterrupt::new_err(e)
        } else {
            PyValueError::new_err(e)
        });
    }

    Ok(results
        .into_iter()
        .map(|r| (r.action, r.improved_policy))
        .collect())
}

/// Run Gumbel MCTS and return root visit counts alongside the standard outputs.
///
/// Equivalent to `gumbel_mcts` except the returned tuple also contains a
/// per-legal-move visit-count vector aligned with `legal_moves()`. Useful for
/// tooling that wants a smoother readout of the search than the σ-amplified
/// improved policy (which collapses to near-one-hot under default c_visit=50).
///
/// Args:
///     game: hexo_rs.GameState (non-terminal)
///     eval_fn: callable(list[GameState]) -> tuple[list[list[float]], list[float]]
///     config: hexo_rs.MCTSConfig
///     seed: optional int for deterministic RNG
///
/// Returns:
///     tuple[tuple[int,int], list[float], list[int]]
///       — (action, improved_policy, visit_counts)
#[pyfunction(name = "gumbel_mcts_with_stats")]
#[pyo3(signature = (game, eval_fn, config, seed=None))]
fn py_gumbel_mcts_with_stats(
    py: Python<'_>,
    game: &PyGameState,
    eval_fn: Py<PyAny>,
    config: &PyMCTSConfig,
    seed: Option<u64>,
) -> PyResult<((i32, i32), Vec<f64>, Vec<u32>)> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_os_rng(),
    };

    let mut eval_error: Option<PyErr> = None;

    let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let py_states: Vec<PyGameState> = states
            .iter()
            .map(|s| PyGameState { inner: s.clone() })
            .collect();

        let result = match eval_fn.call1(py, (py_states,)) {
            Ok(r) => r,
            Err(e) => {
                eval_error = Some(e);
                let dummy = states.iter().map(|_| HashMap::default()).collect();
                return (dummy, vec![0.0; states.len()]);
            }
        };

        let parsed: Result<(Vec<Vec<f64>>, Vec<f64>), _> = result.extract(py);
        let (logits_per_state, values_raw) = match parsed {
            Ok(v) => v,
            Err(e) => {
                eval_error = Some(e.into());
                let dummy = states.iter().map(|_| HashMap::default()).collect();
                return (dummy, vec![0.0; states.len()]);
            }
        };

        let logits_maps: Vec<HashMap<Coord, f64>> = states
            .iter()
            .zip(logits_per_state)
            .map(|(state, logits)| {
                let moves = state.legal_moves();
                moves.into_iter().zip(logits).collect()
            })
            .collect();

        (logits_maps, values_raw)
    };

    let result = gumbel_mcts::gumbel_mcts(&game.inner, &config.inner, &mut rng, None, &mut eval)
        .map_err(PyValueError::new_err)?;

    if let Some(e) = eval_error {
        return Err(e);
    }

    Ok((result.action, result.improved_policy, result.visit_counts))
}

/// Run Gumbel MCTS and return per-child diagnostics alongside the standard outputs.
///
/// Extends `gumbel_mcts_with_stats` with four vectors aligned with the
/// legal-move ordering returned by `GameState.legal_moves()`, plus the
/// E.4-inject capture output:
///
/// - `per_child_q`: Q(root, action) from the root player's perspective.
///   Visited children use empirical Q; unvisited children use the `v_mix`
///   fallback that σ(Q) sees during sequential halving, so the vector can be
///   consumed uniformly without branching on visit count.
/// - `per_child_prior`: the network prior π(action) at the root — softmax of
///   the policy logits restricted to legal moves, after any root Dirichlet
///   mixing, before Gumbel noise and σ(Q) sharpening.
/// - `candidate_indices`: indices into `legal_moves()` of the Gumbel-Top-K
///   survivors at the root (length `min(m_actions, len(legal_moves))`).
/// - `chosen_action_forced_candidates`: at mr=2 roots with
///   `forced_candidate_capture_k > 0`, the top-K p₂ coords (by conditional
///   prior) for the chosen p₁'s post-state. Empty otherwise.
///
/// Args:
///     game: hexo_rs.GameState (non-terminal)
///     eval_fn: callable(list[GameState]) -> tuple[list[list[float]], list[float]]
///     config: hexo_rs.MCTSConfig
///     seed: optional int for deterministic RNG
///     forced_candidates: optional list of (q, r) coords (E.4-inject). When
///         supplied at a mr=1 root, these coords are guaranteed to appear in
///         the Gumbel-Top-K candidate set. Ignored at mr=2 roots.
///
/// Returns:
///     tuple[tuple[int,int], list[float], list[int], list[float], list[float], list[int], list[tuple[int,int]]]
///       — (action, improved_policy, visit_counts, per_child_q, per_child_prior,
///          candidate_indices, chosen_action_forced_candidates)
#[pyfunction(name = "gumbel_mcts_with_diagnostics")]
#[pyo3(signature = (game, eval_fn, config, seed=None, forced_candidates=None))]
fn py_gumbel_mcts_with_diagnostics(
    py: Python<'_>,
    game: &PyGameState,
    eval_fn: Py<PyAny>,
    config: &PyMCTSConfig,
    seed: Option<u64>,
    forced_candidates: Option<Vec<(i32, i32)>>,
) -> PyResult<(
    (i32, i32),
    Vec<f64>,
    Vec<u32>,
    Vec<f64>,
    Vec<f64>,
    Vec<usize>,
    Vec<(i32, i32)>,
)> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_os_rng(),
    };

    let mut eval_error: Option<PyErr> = None;

    let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let py_states: Vec<PyGameState> = states
            .iter()
            .map(|s| PyGameState { inner: s.clone() })
            .collect();

        let result = match eval_fn.call1(py, (py_states,)) {
            Ok(r) => r,
            Err(e) => {
                eval_error = Some(e);
                let dummy = states.iter().map(|_| HashMap::default()).collect();
                return (dummy, vec![0.0; states.len()]);
            }
        };

        let parsed: Result<(Vec<Vec<f64>>, Vec<f64>), _> = result.extract(py);
        let (logits_per_state, values_raw) = match parsed {
            Ok(v) => v,
            Err(e) => {
                eval_error = Some(e.into());
                let dummy = states.iter().map(|_| HashMap::default()).collect();
                return (dummy, vec![0.0; states.len()]);
            }
        };

        let logits_maps: Vec<HashMap<Coord, f64>> = states
            .iter()
            .zip(logits_per_state)
            .map(|(state, logits)| {
                let moves = state.legal_moves();
                moves.into_iter().zip(logits).collect()
            })
            .collect();

        (logits_maps, values_raw)
    };

    let forced_ref: Option<&[Coord]> = forced_candidates.as_deref();
    let result = gumbel_mcts::gumbel_mcts(
        &game.inner,
        &config.inner,
        &mut rng,
        forced_ref,
        &mut eval,
    )
    .map_err(PyValueError::new_err)?;

    if let Some(e) = eval_error {
        return Err(e);
    }

    Ok((
        result.action,
        result.improved_policy,
        result.visit_counts,
        result.per_child_q,
        result.per_child_prior,
        result.candidate_indices,
        result.chosen_action_forced_candidates,
    ))
}

/// Build graph arrays from a game state (Rust-accelerated game_to_graph).
///
/// Returns a dict with keys: features, edge_src, edge_dst, legal_mask, stone_mask,
/// coords, num_nodes — all as flat Python lists ready for torch.tensor().
#[pyfunction(name = "game_to_graph_raw", signature = (game, threat_features=false, relative_stones=false))]
fn py_game_to_graph_raw(
    py: Python<'_>,
    game: &PyGameState,
    threat_features: bool,
    relative_stones: bool,
) -> PyResult<Py<PyAny>> {
    if game.inner.is_terminal() {
        return Err(PyValueError::new_err(
            "Cannot construct graph for a terminal game state.",
        ));
    }

    let g = crate::graph::game_to_graph_raw_opts(&game.inner, threat_features, relative_stones);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("features", g.features)?;
    dict.set_item("edge_src", g.edge_src)?;
    dict.set_item("edge_dst", g.edge_dst)?;
    dict.set_item("legal_mask", g.legal_mask)?;
    dict.set_item("stone_mask", g.stone_mask)?;
    dict.set_item("neighbor_index", g.neighbor_index)?;
    dict.set_item("coords", g.coords)?;
    dict.set_item("num_nodes", g.num_nodes)?;
    Ok(dict.into())
}

/// Build graph arrays for a batch of game states in parallel.
///
/// Returns a list of dicts, one per state. Uses rayon for parallel construction.
#[pyfunction(name = "game_to_graph_batch", signature = (games, threat_features=false, relative_stones=false))]
fn py_game_to_graph_batch(
    py: Python<'_>,
    games: Vec<Py<PyGameState>>,
    threat_features: bool,
    relative_stones: bool,
) -> PyResult<Vec<Py<PyAny>>> {
    // Extract inner GameStates while we hold the GIL
    let states: Vec<GameState> = games
        .iter()
        .map(|g| g.borrow(py).inner.clone())
        .collect();

    // Parallel graph construction (no GIL needed — pure Rust)
    let graphs = crate::graph::game_to_graph_batch_opts(&states, threat_features, relative_stones);

    graphs
        .into_iter()
        .map(|g| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("features", g.features)?;
            dict.set_item("edge_src", g.edge_src)?;
            dict.set_item("edge_dst", g.edge_dst)?;
            dict.set_item("legal_mask", g.legal_mask)?;
            dict.set_item("stone_mask", g.stone_mask)?;
            dict.set_item("neighbor_index", g.neighbor_index)?;
            dict.set_item("coords", g.coords)?;
            dict.set_item("num_nodes", g.num_nodes)?;
            Ok(dict.into())
        })
        .collect()
}

/// Resolve the lean-schema flags into a `LeanOpts`, validating `moves_scope`.
///
/// `moves_scope="graph"` is not yet implemented in the native builder; it is
/// rejected with a clear error rather than silently mis-encoding.
fn resolve_lean_opts(
    axis_relational: bool,
    compact_stone_onehot: bool,
    node_coords: bool,
    moves_scope: &str,
) -> PyResult<crate::axis_graph::LeanOpts> {
    use crate::axis_graph::MovesScope;
    let scope = match moves_scope {
        "node" => MovesScope::Node,
        "graph" => {
            return Err(PyValueError::new_err(
                "moves_scope='graph' is not yet implemented in the native axis \
                 graph builder; only 'node' is supported.",
            ));
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "moves_scope must be 'node' or 'graph', got {other:?}"
            )));
        }
    };
    Ok(crate::axis_graph::LeanOpts {
        axis_relational,
        compact_stone_onehot,
        node_coords,
        moves_scope: scope,
    })
}

/// Populate a graph dict with the edge representation matching `lean`: the
/// relational schema emits `edge_type`/`edge_dist` + `global_edge_src`/
/// `global_edge_dst`; the legacy schema emits `edge_attr`.
fn set_axis_edge_items(
    dict: &Bound<'_, PyDict>,
    g: crate::axis_graph::AxisGraphData,
    axis_relational: bool,
) -> PyResult<()> {
    dict.set_item("features", g.features)?;
    dict.set_item("edge_src", g.edge_src)?;
    dict.set_item("edge_dst", g.edge_dst)?;
    if axis_relational {
        dict.set_item("edge_type", g.edge_type)?;
        dict.set_item("edge_dist", g.edge_dist)?;
        dict.set_item("global_edge_src", g.global_edge_src)?;
        dict.set_item("global_edge_dst", g.global_edge_dst)?;
    } else {
        dict.set_item("edge_attr", g.edge_attr)?;
    }
    dict.set_item("legal_mask", g.legal_mask)?;
    dict.set_item("stone_mask", g.stone_mask)?;
    dict.set_item("coords", g.coords)?;
    dict.set_item("num_nodes", g.num_nodes)?;
    Ok(())
}

/// Build axis-window graph arrays from a game state (Rust-accelerated).
///
/// Legacy schema returns a dict with keys: features, edge_src, edge_dst,
/// edge_attr, legal_mask, stone_mask, coords, num_nodes. With
/// `axis_relational=true` the edge_attr key is replaced by edge_type, edge_dist,
/// global_edge_src, global_edge_dst (see the D6-invariant lean schema). All
/// values are flat Python lists ready for torch.tensor().
#[pyfunction(name = "game_to_axis_graph_raw", signature = (game, prune_empty_edges=false, threat_features=false, relative_stones=false, axis_relational=false, compact_stone_onehot=false, node_coords=true, moves_scope="node"))]
#[allow(clippy::too_many_arguments)]
fn py_game_to_axis_graph_raw(
    py: Python<'_>,
    game: &PyGameState,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
    axis_relational: bool,
    compact_stone_onehot: bool,
    node_coords: bool,
    moves_scope: &str,
) -> PyResult<Py<PyAny>> {
    if game.inner.is_terminal() {
        return Err(PyValueError::new_err(
            "Cannot construct graph for a terminal game state.",
        ));
    }

    let lean = resolve_lean_opts(axis_relational, compact_stone_onehot, node_coords, moves_scope)?;
    let g = crate::axis_graph::game_to_axis_graph_raw_lean(
        &game.inner,
        prune_empty_edges,
        threat_features,
        relative_stones,
        &lean,
    );

    let dict = pyo3::types::PyDict::new(py);
    set_axis_edge_items(&dict, g, axis_relational)?;
    Ok(dict.into())
}

/// Build axis-window graph arrays for a batch of game states in parallel.
///
/// Returns a list of dicts, one per state. Uses rayon for parallel construction.
#[pyfunction(name = "game_to_axis_graph_batch", signature = (games, prune_empty_edges=false, threat_features=false, relative_stones=false, axis_relational=false, compact_stone_onehot=false, node_coords=true, moves_scope="node"))]
#[allow(clippy::too_many_arguments)]
fn py_game_to_axis_graph_batch(
    py: Python<'_>,
    games: Vec<Py<PyGameState>>,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
    axis_relational: bool,
    compact_stone_onehot: bool,
    node_coords: bool,
    moves_scope: &str,
) -> PyResult<Vec<Py<PyAny>>> {
    let lean = resolve_lean_opts(axis_relational, compact_stone_onehot, node_coords, moves_scope)?;

    // Extract inner GameStates while we hold the GIL
    let states: Vec<GameState> = games
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let gs = g.borrow(py);
            if gs.inner.is_terminal() {
                return Err(PyValueError::new_err(format!(
                    "Cannot construct graph for a terminal game state (game index {i}).",
                )));
            }
            Ok(gs.inner.clone())
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Parallel graph construction (no GIL needed — pure Rust)
    let graphs = crate::axis_graph::game_to_axis_graph_batch_lean(
        &states,
        prune_empty_edges,
        threat_features,
        relative_stones,
        &lean,
    );

    graphs
        .into_iter()
        .map(|g| {
            let dict = pyo3::types::PyDict::new(py);
            set_axis_edge_items(&dict, g, axis_relational)?;
            Ok(dict.into())
        })
        .collect()
}

/// Produce 11 D6-augmented graph variants from a game state.
///
/// Returns list of (graph_dict, permutation) tuples.
/// permutation[i] = original legal move index that maps to augmented index i.
#[pyfunction(name = "augment_graph")]
fn py_augment_graph(py: Python<'_>, game: &PyGameState) -> PyResult<Vec<(Py<PyAny>, Vec<usize>)>> {
    if game.inner.is_terminal() {
        return Err(PyValueError::new_err(
            "Cannot augment graph for a terminal game state.",
        ));
    }

    let mut stones = game.inner.placed_stones();
    stones.sort_by_key(|&(c, _)| c);
    let legal = game.inner.legal_moves();

    let player_feat: f32 = match game.inner.current_player() {
        Some(Player::P1) => 1.0,
        _ => -1.0,
    };
    let moves_feat: f32 = game.inner.moves_remaining_this_turn() as f32 / 2.0;

    let augmented = crate::graph::augment_graph(&stones, &legal, player_feat, moves_feat);

    augmented
        .into_iter()
        .map(|(g, perm)| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("features", g.features)?;
            dict.set_item("edge_src", g.edge_src)?;
            dict.set_item("edge_dst", g.edge_dst)?;
            dict.set_item("legal_mask", g.legal_mask)?;
            dict.set_item("stone_mask", g.stone_mask)?;
            dict.set_item("neighbor_index", g.neighbor_index)?;
            dict.set_item("coords", g.coords)?;
            dict.set_item("num_nodes", g.num_nodes)?;
            Ok((dict.into(), perm))
        })
        .collect()
}

/// Produce 11 D6-augmented axis-window graph variants from a game state.
///
/// Returns list of (graph_dict, permutation) tuples.
/// permutation[i] = original legal move index that maps to augmented index i.
#[pyfunction(name = "augment_axis_graph")]
fn py_augment_axis_graph(py: Python<'_>, game: &PyGameState) -> PyResult<Vec<(Py<PyAny>, Vec<usize>)>> {
    if game.inner.is_terminal() {
        return Err(PyValueError::new_err(
            "Cannot augment graph for a terminal game state.",
        ));
    }

    let augmented = crate::axis_graph::augment_axis_graph(&game.inner);

    augmented
        .into_iter()
        .map(|(g, perm)| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("features", g.features)?;
            dict.set_item("edge_src", g.edge_src)?;
            dict.set_item("edge_dst", g.edge_dst)?;
            dict.set_item("edge_attr", g.edge_attr)?;
            dict.set_item("legal_mask", g.legal_mask)?;
            dict.set_item("stone_mask", g.stone_mask)?;
            dict.set_item("coords", g.coords)?;
            dict.set_item("num_nodes", g.num_nodes)?;
            Ok((dict.into(), perm))
        })
        .collect()
}

/// Call the Python eval_fn with a batch of GameStates.
/// Must be called while holding the GIL.
/// Returns Err on Python exceptions (including KeyboardInterrupt).
fn call_python_eval(
    py: Python<'_>,
    eval_fn: &Py<PyAny>,
    states: &[GameState],
) -> Result<(Vec<HashMap<Coord, f64>>, Vec<f64>), String> {
    let py_states: Vec<PyGameState> = states
        .iter()
        .map(|s| PyGameState { inner: s.clone() })
        .collect();

    let result = eval_fn.call1(py, (py_states,))
        .map_err(|e| format!("{e}"))?;

    let (logits_per_state, values_raw): (Vec<Vec<f64>>, Vec<f64>) = result.extract(py)
        .map_err(|e| format!("eval parse error: {e}"))?;

    let logits_maps: Vec<HashMap<Coord, f64>> = states
        .iter()
        .zip(logits_per_state)
        .map(|(state, logits)| {
            let moves = state.legal_moves();
            moves.into_iter().zip(logits).collect()
        })
        .collect();

    Ok((logits_maps, values_raw))
}

/// Play N complete self-play games, returning trajectories.
///
/// Each game is played move-by-move using Gumbel MCTS. All games share
/// the same eval_fn, so GPU batch utilisation comes from MCTS's internal
/// batching (m_actions leaves evaluated per halving round).
///
/// Returns a list of games, each game is a list of (action, improved_policy)
/// tuples — one per move played. The Python side handles trajectory →
/// TrainingExample conversion.
///
/// Args:
///     game_config: hexo_rs.GameConfig
///     eval_fn: callable(list[GameState]) -> tuple[list[list[float]], list[float]]
///     mcts_config: hexo_rs.MCTSConfig
///     n_games: number of games to play
///     exploration_moves: first N moves use stochastic action selection
///     playout_cap_fraction: fraction of moves using reduced sim budget (0.0 = disabled)
///     playout_cap_divisor: reduced budget = n_simulations / divisor (1 = disabled)
///     seed: optional RNG seed
///
/// Returns:
///     list of (trajectory, winner) tuples, where:
///       trajectory: list of (state_snapshot, action, improved_policy, current_player)
///       winner: "P1", "P2", or None (draw)
#[pyfunction(name = "batched_self_play")]
#[pyo3(signature = (game_config, eval_fn, mcts_config, n_games, exploration_moves=0, playout_cap_fraction=0.0, playout_cap_divisor=1, seed=None))]
fn py_batched_self_play(
    py: Python<'_>,
    game_config: &PyGameConfig,
    eval_fn: Py<PyAny>,
    mcts_config: &PyMCTSConfig,
    n_games: usize,
    exploration_moves: usize,
    playout_cap_fraction: f64,
    playout_cap_divisor: u32,
    seed: Option<u64>,
) -> PyResult<Vec<(Vec<(PyGameState, (i32, i32), Vec<f64>, &'static str)>, Option<&'static str>)>> {
    use std::sync::mpsc;
    use std::time::Duration;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use crate::mcts::batcher::{BatcherConfig, EvalRequest, EvalResponse, batcher_loop};

    let mcts_cfg = mcts_config.inner.clone();
    let game_cfg = game_config.inner;

    // Per-game deterministic seeds
    let mut master_rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_os_rng(),
    };
    let game_seeds: Vec<u64> = (0..n_games).map(|_| master_rng.random()).collect();

    // Release the GIL for the parallel section.
    // Game threads do pure Rust MCTS; the batcher re-acquires the GIL per batch.
    // An abort flag lets the batcher signal game threads to stop on error
    // (e.g. KeyboardInterrupt).
    use std::sync::atomic::{AtomicBool, Ordering};

    let result: Result<Vec<(Vec<(GameState, Coord, Vec<f64>, Player)>, Option<Player>)>, String> =
        py.detach(|| {
            // Channel created inside detach so Receiver doesn't cross the Ungil boundary
            let (request_tx, request_rx) = mpsc::sync_channel::<EvalRequest>(n_games);
            let abort = AtomicBool::new(false);
            std::thread::scope(|scope| {
                // Spawn one thread per game
                let mut handles = Vec::with_capacity(n_games);
                let abort_ref = &abort;
                for i in 0..n_games {
                    let tx = request_tx.clone();
                    let cfg = mcts_cfg.clone();
                    let gcfg = game_cfg;
                    let game_seed = game_seeds[i];

                    let handle = scope.spawn(move || {
                        let mut rng = ChaCha8Rng::seed_from_u64(game_seed);
                        let mut game = GameState::with_config(gcfg);
                        let mut trajectory: Vec<(GameState, Coord, Vec<f64>, Player)> = Vec::new();
                        let mut move_count = 0usize;

                        // Per-game eval closure: sends to batcher, blocks for response.
                        // Returns dummy data if the batcher has shut down (abort).
                        let mut eval_fn = |states: &[GameState]|
                            -> (Vec<HashMap<Coord, f64>>, Vec<f64>)
                        {
                            if abort_ref.load(Ordering::Relaxed) {
                                let dummy = states.iter().map(|_| HashMap::default()).collect();
                                return (dummy, vec![0.0; states.len()]);
                            }
                            let (resp_tx, resp_rx) = mpsc::sync_channel::<EvalResponse>(1);
                            let req = EvalRequest {
                                states: states.to_vec(),
                                response_tx: resp_tx,
                            };
                            match tx.send(req) {
                                Ok(()) => {}
                                Err(_) => {
                                    // Batcher has exited (abort or finished)
                                    let dummy = states.iter().map(|_| HashMap::default()).collect();
                                    return (dummy, vec![0.0; states.len()]);
                                }
                            }
                            match resp_rx.recv() {
                                Ok(resp) => (resp.logits, resp.values),
                                Err(_) => {
                                    let dummy = states.iter().map(|_| HashMap::default()).collect();
                                    (dummy, vec![0.0; states.len()])
                                }
                            }
                        };

                        while !game.is_terminal() && !abort_ref.load(Ordering::Relaxed) {
                            let snapshot = game.clone();
                            let current_player = game.current_player()
                                .expect("non-terminal has current player");

                            // Per-move playout cap: use reduced budget with probability playout_cap_fraction
                            let effective_cfg = if playout_cap_fraction > 0.0
                                && playout_cap_divisor > 1
                                && rng.random::<f64>() < playout_cap_fraction
                            {
                                MCTSConfig {
                                    n_simulations: (cfg.n_simulations / playout_cap_divisor).max(1),
                                    ..cfg.clone()
                                }
                            } else {
                                cfg.clone()
                            };

                            let mcts_result = match gumbel_mcts::gumbel_mcts(
                                &game, &effective_cfg, &mut rng, None, &mut eval_fn,
                            ) {
                                Ok(r) => r,
                                Err(_) => break, // terminal or no legal moves
                            };

                            // If aborted during MCTS, don't push garbage to trajectory
                            if abort_ref.load(Ordering::Relaxed) {
                                break;
                            }

                            let action = if move_count < exploration_moves {
                                // Sample from improved policy
                                let policy = &mcts_result.improved_policy;
                                let total: f64 = policy.iter().sum();
                                if total > 0.0 {
                                    let r: f64 = rng.random::<f64>() * total;
                                    let mut cumsum = 0.0;
                                    let mut chosen_idx = 0;
                                    for (k, &p) in policy.iter().enumerate() {
                                        cumsum += p;
                                        if cumsum > r {
                                            chosen_idx = k;
                                            break;
                                        }
                                    }
                                    mcts_result.coords[chosen_idx]
                                } else {
                                    mcts_result.action
                                }
                            } else {
                                mcts_result.action
                            };

                            trajectory.push((
                                snapshot,
                                action,
                                mcts_result.improved_policy,
                                current_player,
                            ));

                            game.apply_move(action).expect("valid action from MCTS");
                            move_count += 1;
                        }

                        let winner = game.winner();
                        (trajectory, winner)
                    });
                    handles.push(handle);
                }

                // Drop our sender so the batcher can detect when all games finish
                drop(request_tx);

                // Run the batcher on this thread (the main thread within allow_threads).
                // Re-acquires the GIL for each batch eval via Python::with_gil.
                let batcher_config = BatcherConfig {
                    max_batch_size: n_games * mcts_cfg.m_actions,
                    timeout: Duration::from_millis(10),
                };

                let eval_fn_ref = &eval_fn;
                let batcher_result = batcher_loop(&request_rx, &batcher_config, &mut |states: &[GameState]| {
                    Python::attach(|py| {
                        call_python_eval(py, eval_fn_ref, states)
                    })
                });

                // If batcher failed (e.g. KeyboardInterrupt), signal game threads to stop
                // and unblock any threads currently waiting on eval responses.
                if batcher_result.is_err() {
                    abort.store(true, Ordering::Relaxed);

                    loop {
                        match request_rx.try_recv() {
                            Ok(req) => {
                                let n = req.states.len();
                                let _ = req.response_tx.send(EvalResponse {
                                    logits: (0..n).map(|_| HashMap::default()).collect(),
                                    values: vec![0.0; n],
                                });
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => break,
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                        }
                    }
                }

                // Collect results from all game threads (they will exit quickly if aborted)
                let mut all_trajectories = Vec::with_capacity(n_games);
                for handle in handles {
                    match handle.join() {
                        Ok(traj) => all_trajectories.push(traj),
                        Err(_) => {} // thread panicked, skip it
                    }
                }

                // Propagate batcher error (e.g. KeyboardInterrupt)
                if let Err(e) = batcher_result {
                    return Err(e);
                }

                Ok(all_trajectories)
            })
        });

    let raw_trajectories = result.map_err(|e| {
        if e.contains("KeyboardInterrupt") {
            pyo3::exceptions::PyKeyboardInterrupt::new_err(e)
        } else {
            PyValueError::new_err(e)
        }
    })?;

    // Convert Rust types to Python types (on the main thread with GIL held)
    let py_trajectories: Vec<(Vec<(PyGameState, (i32, i32), Vec<f64>, &'static str)>, Option<&'static str>)> =
        raw_trajectories
            .into_iter()
            .map(|(traj, winner)| {
                let py_traj = traj.into_iter()
                    .map(|(state, action, policy, player)| {
                        (PyGameState { inner: state }, action, policy, player_str(player))
                    })
                    .collect();
                let py_winner = winner.map(|w| player_str(w));
                (py_traj, py_winner)
            })
            .collect();

    Ok(py_trajectories)
}

/// Play N self-play games using Rust-native TorchScript inference.
///
/// No Python callback needed — the model runs entirely in Rust via libtorch.
/// Returns trajectories in the same format as batched_self_play.
///
/// Args:
///     game_config: hexo_rs.GameConfig
///     model_path: path to TorchScript .pt file
///     mcts_config: hexo_rs.MCTSConfig
///     n_games: number of games to play
///     exploration_moves: first N moves use stochastic action selection
///     device: "cuda" or "cpu"
///     seed: optional RNG seed
#[cfg(feature = "torch")]
#[pyfunction(name = "native_self_play")]
#[pyo3(signature = (game_config, model_path, mcts_config, n_games, exploration_moves=0, device="cuda", seed=None, graph_type="hex"))]
fn py_native_self_play(
    _py: Python<'_>,
    game_config: &PyGameConfig,
    model_path: &str,
    mcts_config: &PyMCTSConfig,
    n_games: usize,
    exploration_moves: usize,
    device: &str,
    seed: Option<u64>,
    graph_type: &str,
) -> PyResult<Vec<Vec<(PyGameState, (i32, i32), Vec<f64>, &'static str)>>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use crate::inference::{GraphType, TorchModel};
    use crate::mcts::batched::batched_gumbel_mcts;

    let tch_device = match device {
        "cuda" => tch::Device::Cuda(0),
        "cpu" => tch::Device::Cpu,
        _ => return Err(PyValueError::new_err(format!("Unknown device: {device}"))),
    };

    let gt = match graph_type {
        "axis" => GraphType::Axis,
        "hex" => GraphType::Hex,
        _ => return Err(PyValueError::new_err(format!("Unknown graph_type: {graph_type}"))),
    };

    let model = TorchModel::load_with_graph_type(model_path, tch_device, gt)
        .map_err(|e| PyValueError::new_err(format!("Failed to load model: {e}")))?;

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_os_rng(),
    };

    let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        model.evaluate(states)
    };

    let mut games: Vec<GameState> = (0..n_games)
        .map(|_| GameState::with_config(game_config.inner))
        .collect();
    let mut trajectories: Vec<Vec<(PyGameState, (i32, i32), Vec<f64>, &'static str)>> =
        (0..n_games).map(|_| Vec::new()).collect();
    let mut move_counts: Vec<usize> = vec![0; n_games];
    let mut active: Vec<bool> = vec![true; n_games];

    loop {
        let active_indices: Vec<usize> = (0..n_games)
            .filter(|&i| active[i] && !games[i].is_terminal())
            .collect();

        if active_indices.is_empty() {
            break;
        }

        let active_states: Vec<GameState> = active_indices.iter().map(|&i| games[i].clone()).collect();
        let mcts_results = batched_gumbel_mcts(&active_states, &mcts_config.inner, &mut rng, &mut eval)
            .map_err(PyValueError::new_err)?;

        for (result_idx, &game_idx) in active_indices.iter().enumerate() {
            let mcts_result = &mcts_results[result_idx];
            let snapshot = PyGameState { inner: games[game_idx].clone() };
            let current_player = games[game_idx].current_player().map(player_str).unwrap_or("?");

            let action = if move_counts[game_idx] < exploration_moves {
                let policy = &mcts_result.improved_policy;
                let total: f64 = policy.iter().sum();
                if total > 0.0 {
                    let r: f64 = rng.random::<f64>() * total;
                    let mut cumsum = 0.0;
                    let mut chosen_idx = 0;
                    for (i, &p) in policy.iter().enumerate() {
                        cumsum += p;
                        if cumsum > r {
                            chosen_idx = i;
                            break;
                        }
                    }
                    mcts_result.coords[chosen_idx]
                } else {
                    mcts_result.action
                }
            } else {
                mcts_result.action
            };

            trajectories[game_idx].push((
                snapshot,
                action,
                mcts_result.improved_policy.clone(),
                current_player,
            ));

            games[game_idx].apply_move(action).expect("valid action from MCTS");
            move_counts[game_idx] += 1;
        }

        for i in 0..n_games {
            if active[i] && games[i].is_terminal() {
                active[i] = false;
            }
        }
    }

    Ok(trajectories)
}

/// Collate multiple game states into flat batch tensors for direct model consumption.
///
/// Bypasses PyG Batch.from_data_list() by returning pre-collated flat arrays
/// ready for torch.tensor() conversion.
///
/// Returns a tuple: (features, edge_src, edge_dst, legal_mask, stone_mask, batch, num_graphs, legal_counts, legal_idx, stone_idx, stone_batch)
#[pyfunction(name = "game_states_to_batch", signature = (states, threat_features=false, relative_stones=false))]
fn py_game_states_to_batch(
    py: Python<'_>,
    states: Vec<Py<PyGameState>>,
    threat_features: bool,
    relative_stones: bool,
) -> PyResult<(
    Vec<f32>,   // features (flat)
    Vec<i64>,   // edge_index_src
    Vec<i64>,   // edge_index_dst
    Vec<bool>,  // legal_mask
    Vec<bool>,  // stone_mask
    Vec<i64>,   // batch vector
    usize,      // num_graphs
    Vec<usize>, // legal_counts
    Vec<i64>,   // legal_idx
    Vec<i64>,   // stone_idx
    Vec<i64>,   // stone_batch
)> {
    use crate::batch_tensors::collate_graphs;
    use crate::graph::game_to_graph_raw_opts;

    let inner_states: Vec<_> = states.iter().map(|s| s.borrow(py).inner.clone()).collect();
    let graphs: Vec<_> = inner_states
        .iter()
        .map(|s| game_to_graph_raw_opts(s, threat_features, relative_stones))
        .collect();
    let bt = collate_graphs(&graphs);
    Ok((
        bt.features,
        bt.edge_index_src,
        bt.edge_index_dst,
        bt.legal_mask,
        bt.stone_mask,
        bt.batch,
        bt.num_graphs,
        bt.legal_counts,
        bt.legal_idx,
        bt.stone_idx,
        bt.stone_batch,
    ))
}

/// Reinterpret a slice as native-endian raw bytes (single memcpy on the Python side).
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: T is a plain-old-data numeric/bool type with no padding between
    // elements of a slice; lifetime is tied to the borrow.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Clone the inner `GameState`s out of a batch of `PyGameState`s, rejecting any
/// terminal state (graphs cannot be built for terminal positions). The error
/// message carries the offending game index.
fn collect_inner_states(py: Python<'_>, states: &[Py<PyGameState>]) -> PyResult<Vec<GameState>> {
    states
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let gs = g.borrow(py);
            if gs.inner.is_terminal() {
                return Err(PyValueError::new_err(format!(
                    "Cannot construct graph for a terminal game state (game index {i})."
                )));
            }
            Ok(gs.inner.clone())
        })
        .collect::<PyResult<Vec<_>>>()
}

/// Build the shared zero-copy byte-buffer dict for a collated AXIS batch. Fills
/// the keys common to both batch builders (x, edge_src/dst, edge_attr,
/// legal_mask, batch, coords, legal_counts, legal_idx, stone_idx, stone_batch,
/// num_graphs, n_feat); callers may add extra keys (e.g. `permutations`).
fn axis_batch_to_pydict<'py>(
    py: Python<'py>,
    bt: &crate::batch_tensors::AxisBatchTensors,
) -> PyResult<Bound<'py, PyDict>> {
    let total_nodes = bt.batch.len();
    let n_feat = if total_nodes > 0 { bt.features.len() / total_nodes } else { 0 };

    let dict = PyDict::new(py);
    dict.set_item("x", pyo3::types::PyBytes::new(py, as_bytes(&bt.features)))?;
    dict.set_item("edge_src", pyo3::types::PyBytes::new(py, as_bytes(&bt.edge_index_src)))?;
    dict.set_item("edge_dst", pyo3::types::PyBytes::new(py, as_bytes(&bt.edge_index_dst)))?;
    dict.set_item("edge_attr", pyo3::types::PyBytes::new(py, as_bytes(&bt.edge_attr)))?;
    dict.set_item("legal_mask", pyo3::types::PyBytes::new(py, as_bytes(&bt.legal_mask)))?;
    dict.set_item("batch", pyo3::types::PyBytes::new(py, as_bytes(&bt.batch)))?;
    dict.set_item("coords", pyo3::types::PyBytes::new(py, as_bytes(&bt.coords)))?;
    dict.set_item("legal_counts", pyo3::types::PyBytes::new(py, as_bytes(&bt.legal_counts)))?;
    dict.set_item("legal_idx", pyo3::types::PyBytes::new(py, as_bytes(&bt.legal_idx)))?;
    dict.set_item("stone_idx", pyo3::types::PyBytes::new(py, as_bytes(&bt.stone_idx)))?;
    dict.set_item("stone_batch", pyo3::types::PyBytes::new(py, as_bytes(&bt.stone_batch)))?;
    dict.set_item("num_graphs", bt.num_graphs)?;
    dict.set_item("n_feat", n_feat)?;
    Ok(dict)
}

/// Collate AXIS graphs for a batch of states into flat native-endian byte
/// buffers, ready for zero-copy `torch.frombuffer` — avoids the per-element
/// Python list -> tensor conversion cost of the dict-based builders.
///
/// Returns a dict: x (f32), edge_src/edge_dst (i64), edge_attr (f32, E*5),
/// legal_mask (bool), batch (i64), coords (i32, N*2), legal_counts (i64),
/// legal_idx/stone_idx/stone_batch (i64) as bytes; num_graphs and n_feat as ints.
#[pyfunction(
    name = "game_states_to_axis_batch_bytes",
    signature = (states, prune_empty_edges=false, threat_features=false, relative_stones=false)
)]
fn py_game_states_to_axis_batch_bytes(
    py: Python<'_>,
    states: Vec<Py<PyGameState>>,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> PyResult<Py<PyDict>> {
    use crate::batch_tensors::collate_axis_graphs;

    let inner_states = collect_inner_states(py, &states)?;

    let bt = py.detach(|| {
        let graphs = crate::axis_graph::game_to_axis_graph_batch_opts(
            &inner_states,
            prune_empty_edges,
            threat_features,
            relative_stones,
        );
        collate_axis_graphs(&graphs)
    });

    let dict = axis_batch_to_pydict(py, &bt)?;
    Ok(dict.into())
}

/// Collate D6-augmented AXIS graphs for a batch of states into flat
/// native-endian byte buffers (same layout as `game_states_to_axis_batch_bytes`),
/// plus per-state policy permutations for training-time reconstruction.
///
/// `transform_indices[i]` selects `D6_TRANSFORMS[transform_indices[i]]` for
/// state `i` (0 = identity, 1..=11 = non-identity D6 transforms). The returned
/// dict mirrors `game_states_to_axis_batch_bytes` exactly, with one extra key
/// `permutations`: a list of per-state lists where `permutation[new_legal_idx]
/// = old_legal_idx`, so the caller can reorder policy targets to the augmented
/// legal-move order.
#[pyfunction(
    name = "augment_axis_states_to_batch_bytes",
    signature = (states, transform_indices, prune_empty_edges=false, threat_features=false, relative_stones=false)
)]
fn py_augment_axis_states_to_batch_bytes(
    py: Python<'_>,
    states: Vec<Py<PyGameState>>,
    transform_indices: Vec<usize>,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> PyResult<Py<PyDict>> {
    use crate::batch_tensors::collate_axis_graphs;
    use hexo_engine::symmetry::D6_TRANSFORMS;

    if transform_indices.len() != states.len() {
        return Err(PyValueError::new_err(format!(
            "transform_indices length ({}) must equal states length ({})",
            transform_indices.len(),
            states.len()
        )));
    }
    for (i, &idx) in transform_indices.iter().enumerate() {
        if idx >= D6_TRANSFORMS.len() {
            return Err(PyValueError::new_err(format!(
                "transform_indices[{i}] = {idx} out of range (must be < {})",
                D6_TRANSFORMS.len()
            )));
        }
    }

    let inner_states = collect_inner_states(py, &states)?;

    let (bt, permutations) = py.detach(|| {
        use rayon::prelude::*;
        // Build per-state (graph, permutation) pairs in parallel. rayon's
        // par_iter().map().collect() preserves input order, which is required
        // so that `permutations[i]` and the collated graphs line up with state
        // `i` (and the caller's policy targets).
        let pairs: Vec<(crate::axis_graph::AxisGraphData, Vec<usize>)> = inner_states
            .par_iter()
            .zip(transform_indices.par_iter())
            .map(|(state, &idx)| {
                crate::axis_graph::augment_axis_graph_single(
                    state,
                    D6_TRANSFORMS[idx],
                    prune_empty_edges,
                    threat_features,
                    relative_stones,
                )
            })
            .collect();
        let mut graphs = Vec::with_capacity(pairs.len());
        let mut permutations = Vec::with_capacity(pairs.len());
        for (graph, perm) in pairs {
            graphs.push(graph);
            permutations.push(perm);
        }
        (collate_axis_graphs(&graphs), permutations)
    });

    let dict = axis_batch_to_pydict(py, &bt)?;
    let perms = pyo3::types::PyList::empty(py);
    for perm in &permutations {
        perms.append(pyo3::types::PyList::new(py, perm)?)?;
    }
    dict.set_item("permutations", perms)?;
    Ok(dict.into())
}

// ---------------------------------------------------------------------------
// Phase 0 DAG-MCTS spike: thread-local instrumentation bindings.
// Feature-gated; only present when built with `--features dedup_count`.
// ---------------------------------------------------------------------------

#[cfg(feature = "dedup_count")]
#[pyfunction(name = "reset_dedup_counters")]
fn py_reset_dedup_counters() {
    crate::mcts::dedup_count::reset_tls();
}

#[cfg(feature = "dedup_count")]
#[pyfunction(name = "snapshot_dedup_counters")]
fn py_snapshot_dedup_counters(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let (total, within, cross, unique) = crate::mcts::dedup_count::snapshot_tls();
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("total_expansions", total)?;
    dict.set_item("within_turn_hits", within)?;
    dict.set_item("cross_turn_hits", cross)?;
    dict.set_item("unique_fingerprints", unique)?;
    Ok(dict.into())
}

#[cfg(feature = "dedup_skip")]
#[pyfunction(name = "set_skip_enabled")]
fn py_set_skip_enabled(enabled: bool) {
    crate::mcts::dedup_count::set_skip_enabled_tls(enabled);
}

#[cfg(feature = "dedup_skip")]
#[pyfunction(name = "reset_eval_cache")]
fn py_reset_eval_cache() {
    crate::mcts::dedup_count::eval_cache_reset_tls();
}

#[cfg(feature = "dedup_skip")]
#[pyfunction(name = "snapshot_eval_cache")]
fn py_snapshot_eval_cache(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let (hits, misses, size) = crate::mcts::dedup_count::eval_cache_snapshot_tls();
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("hits", hits)?;
    dict.set_item("misses", misses)?;
    dict.set_item("size", size)?;
    Ok(dict.into())
}

#[pymodule]
fn hexo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameConfig>()?;
    m.add_class::<PyGameState>()?;
    m.add_class::<PyMCTSConfig>()?;
    m.add_function(wrap_pyfunction!(py_gumbel_mcts, m)?)?;
    m.add_function(wrap_pyfunction!(py_batched_gumbel_mcts, m)?)?;
    m.add_function(wrap_pyfunction!(py_gumbel_mcts_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py_gumbel_mcts_with_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(py_game_to_graph_raw, m)?)?;
    m.add_function(wrap_pyfunction!(py_game_to_graph_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_game_states_to_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_game_states_to_axis_batch_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(py_augment_axis_states_to_batch_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(py_game_to_axis_graph_raw, m)?)?;
    m.add_function(wrap_pyfunction!(py_game_to_axis_graph_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_augment_graph, m)?)?;
    m.add_function(wrap_pyfunction!(py_augment_axis_graph, m)?)?;
    m.add_function(wrap_pyfunction!(py_batched_self_play, m)?)?;
    #[cfg(feature = "torch")]
    m.add_function(wrap_pyfunction!(py_native_self_play, m)?)?;
    #[cfg(feature = "dedup_count")]
    {
        m.add_function(wrap_pyfunction!(py_reset_dedup_counters, m)?)?;
        m.add_function(wrap_pyfunction!(py_snapshot_dedup_counters, m)?)?;
    }
    #[cfg(feature = "dedup_skip")]
    {
        m.add_function(wrap_pyfunction!(py_set_skip_enabled, m)?)?;
        m.add_function(wrap_pyfunction!(py_reset_eval_cache, m)?)?;
        m.add_function(wrap_pyfunction!(py_snapshot_eval_cache, m)?)?;
    }
    Ok(())
}
