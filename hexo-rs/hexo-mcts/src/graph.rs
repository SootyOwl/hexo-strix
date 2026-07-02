//! Graph construction: convert a GameState to raw arrays for PyG Data objects.
//!
//! Mirrors the Python `game_to_graph` but returns flat Vecs instead of tensors.
//! Python wraps these into `torch.tensor` + `torch_geometric.data.Data`.

use std::collections::HashMap;

use hexo_engine::hex::hex_distance;
use hexo_engine::symmetry::D6_TRANSFORMS;
use hexo_engine::types::{Coord, HEX_DIRS, Player};
use hexo_engine::GameState;

/// Raw graph data ready to be wrapped into PyG Data on the Python side.
#[derive(serde::Serialize)]
pub struct GraphData {
    /// Node features, flattened row-major. Per-node width is 8 (absolute) or
    /// 7 (`relative_stones`: to_move dropped), +4 with threat features —
    /// i.e. N×{7,8,11,12}.
    pub features: Vec<f32>,
    /// Edge source indices.
    pub edge_src: Vec<i64>,
    /// Edge destination indices.
    pub edge_dst: Vec<i64>,
    /// Legal move mask (true for legal-move nodes).
    pub legal_mask: Vec<bool>,
    /// Stone mask (true for placed-stone nodes).
    pub stone_mask: Vec<bool>,
    /// Neighbor indices per node: N×6, flattened row-major.
    /// For each node, 6 entries giving the index of each hex neighbor.
    /// -1 (as i64) for missing neighbors (boundary nodes).
    /// Neighbor order matches HEX_DIRS: (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)
    pub neighbor_index: Vec<i64>,
    /// Coordinates, flattened: N×2 row-major (q, r).
    pub coords: Vec<i32>,
    /// Number of nodes.
    pub num_nodes: usize,
}

/// Build graph arrays from pre-sorted stones and legal moves plus global features.
///
/// `stones` and `legal` must already be sorted by coord tuple (q, r).
///
/// Node ordering:
///   - Placed stones first, sorted by (q, r)
///   - Legal moves next, sorted by (q, r)
///
/// Node features, absolute layout (8-dim base, or 12-dim when
/// `threat_features` is set):
///   [0:3] Stone-type one-hot: [1,0,0]=P1, [0,1,0]=P2, [0,0,1]=empty
///   [3]   Current player: +1.0 if P1, -1.0 if P2
///   [4]   Moves remaining this turn / 2.0
///   [5]   Normalised q (relative to stone centroid)
///   [6]   Normalised r (relative to stone centroid)
///   [7]   Inverse distance to nearest stone (0 for stones themselves)
///   [8:12] (optional) threat encoding, filled by `fill_threat_features`:
///          [own_max_line, opp_max_line, own_threat_axes, opp_threat_axes]
///          where "own" = current player to move. Zero-initialised here.
///
/// With `relative_stones` the stone one-hot becomes side-to-move relative
/// ([1,0,0]=own, [0,1,0]=opp where own = `player_feat > 0` ? P1 : P2), the
/// to_move channel [3] is dropped (redundant in a relative frame), and the
/// remaining base features shift down one slot (7-dim base, 11 with threat
/// features):
///   [0:3] own/opp/empty, [3] moves remaining / 2.0, [4] norm q, [5] norm r,
///   [6] inv stone dist, [7:11] (optional) threat encoding.
fn build_graph(
    stones: &[(Coord, Player)],
    legal: &[Coord],
    player_feat: f32,
    moves_feat: f32,
    threat_features: bool,
    relative_stones: bool,
) -> GraphData {
    let base_dim: usize = if relative_stones { 7 } else { 8 };
    let fdim: usize = base_dim + if threat_features { 4 } else { 0 };
    // Offset of the post-one-hot base features relative to the relative
    // layout: absolute keeps to_move at [3], pushing the rest up by one.
    let off: usize = usize::from(!relative_stones);
    // "Own" side for the relative one-hot, derived from player_feat so the
    // terminal-state fallback matches the absolute to_move fill
    // (current_player() == None → player_feat = -1.0 → own = P2).
    let own_is_p1 = player_feat > 0.0;
    let n_stones = stones.len();
    let n_legal = legal.len();
    let n = n_stones + n_legal;

    // Coords (flattened N×2)
    let mut coords = Vec::with_capacity(n * 2);
    for &(coord, _) in stones {
        coords.push(coord.0);
        coords.push(coord.1);
    }
    for &(q, r) in legal {
        coords.push(q);
        coords.push(r);
    }

    // Stone centroid and spread
    let (centroid_q, centroid_r, spread) = if n_stones > 0 {
        let sum_q: f64 = stones.iter().map(|&((q, _), _)| q as f64).sum();
        let sum_r: f64 = stones.iter().map(|&((_, r), _)| r as f64).sum();
        let cq = sum_q / n_stones as f64;
        let cr = sum_r / n_stones as f64;
        let max_dev = stones
            .iter()
            .map(|&((q, r), _)| (q as f64 - cq).abs().max((r as f64 - cr).abs()))
            .fold(0.0f64, f64::max);
        (cq, cr, max_dev.max(1.0))
    } else {
        (0.0, 0.0, 1.0)
    };

    // Stone coords as a vec for fast nearest-stone lookup
    let stone_coords: Vec<Coord> = stones.iter().map(|&(c, _)| c).collect();

    // Features (flattened N×fdim)
    let mut features = vec![0.0f32; n * fdim];

    // Stone features
    for (i, &(_, player)) in stones.iter().enumerate() {
        let base = i * fdim;
        let col = if relative_stones {
            // 0 = own (side to move), 1 = opp
            usize::from((player == Player::P1) != own_is_p1)
        } else {
            match player {
                Player::P1 => 0,
                Player::P2 => 1,
            }
        };
        features[base + col] = 1.0;
        if !relative_stones {
            features[base + 3] = player_feat;
        }
        features[base + 3 + off] = moves_feat;
        let q = coords[i * 2] as f64;
        let r = coords[i * 2 + 1] as f64;
        features[base + 4 + off] = ((q - centroid_q) / spread) as f32;
        features[base + 5 + off] = ((r - centroid_r) / spread) as f32;
        // features[base + 6 + off] = 0.0 (stones get 0 for inverse distance)
    }

    // Legal move features
    for j in 0..n_legal {
        let idx = n_stones + j;
        let base = idx * fdim;
        features[base + 2] = 1.0; // empty one-hot
        if !relative_stones {
            features[base + 3] = player_feat;
        }
        features[base + 3 + off] = moves_feat;
        let q = coords[idx * 2] as f64;
        let r = coords[idx * 2 + 1] as f64;
        features[base + 4 + off] = ((q - centroid_q) / spread) as f32;
        features[base + 5 + off] = ((r - centroid_r) / spread) as f32;

        // Inverse distance to nearest stone
        let coord = legal[j];
        let min_d = stone_coords
            .iter()
            .map(|&sc| hex_distance(coord, sc))
            .min()
            .unwrap_or(1);
        features[base + 6 + off] = 1.0 / min_d.max(1) as f32;
    }

    // Edges (hex adjacency)
    let mut coord_to_idx: HashMap<Coord, usize> = HashMap::with_capacity(n);
    for i in 0..n_stones {
        coord_to_idx.insert((coords[i * 2], coords[i * 2 + 1]), i);
    }
    for j in 0..n_legal {
        let idx = n_stones + j;
        coord_to_idx.insert((coords[idx * 2], coords[idx * 2 + 1]), idx);
    }

    let mut edge_src = Vec::new();
    let mut edge_dst = Vec::new();
    for i in 0..n {
        let q = coords[i * 2];
        let r = coords[i * 2 + 1];
        for &(dq, dr) in &HEX_DIRS {
            if let Some(&j) = coord_to_idx.get(&(q + dq, r + dr)) {
                edge_src.push(i as i64);
                edge_dst.push(j as i64);
            }
        }
    }

    // Self-loops (one per node) — pre-baked so GNN layers don't allocate them per-forward-call
    for i in 0..n {
        edge_src.push(i as i64);
        edge_dst.push(i as i64);
    }

    // Neighbor index (N×6, -1 for missing)
    let mut neighbor_index = Vec::with_capacity(n * 6);
    for i in 0..n {
        let q = coords[i * 2];
        let r = coords[i * 2 + 1];
        for &(dq, dr) in &HEX_DIRS {
            match coord_to_idx.get(&(q + dq, r + dr)) {
                Some(&j) => neighbor_index.push(j as i64),
                None => neighbor_index.push(-1),
            }
        }
    }

    // Masks
    let mut legal_mask = vec![false; n];
    let mut stone_mask = vec![false; n];
    for i in 0..n_stones {
        stone_mask[i] = true;
    }
    for j in 0..n_legal {
        legal_mask[n_stones + j] = true;
    }

    GraphData {
        features,
        edge_src,
        edge_dst,
        legal_mask,
        stone_mask,
        neighbor_index,
        coords,
        num_nodes: n,
    }
}

/// Fill the 4 threat-encoding dims (`base_dim..base_dim + 4`) for the first
/// `n_real` nodes of a `(base_dim + 4)`-stride feature array. The threat dims
/// are [own_max_line, opp_max_line, own_threat_axes, opp_threat_axes],
/// own = current player to move.
///
/// `base_dim` is the width of the base node features (8 absolute, 7 with
/// relative stones); `features` must have stride `base_dim + 4` and `coords`
/// must be the matching flat (q, r) array for the graph's node ordering.
pub(crate) fn fill_threat_features(
    features: &mut [f32],
    coords: &[i32],
    n_real: usize,
    game: &GameState,
    base_dim: usize,
) {
    let to_move = game
        .current_player()
        .expect("fill_threat_features called on terminal GameState; current_player is None");
    let wl = game.config().win_length;
    let stone_map = game.stones();
    let stride = base_dim + 4;
    for idx in 0..n_real {
        let c = (coords[idx * 2], coords[idx * 2 + 1]);
        let tf = hexo_engine::threat::node_threat_features(stone_map, c, to_move, wl);
        let base = idx * stride;
        features[base + base_dim..base + base_dim + 4].copy_from_slice(&tf);
    }
}

/// Build graph arrays from a GameState.
///
/// Extracts stones, legal moves, and global features from the game state,
/// then delegates to `build_graph`.
pub fn game_to_graph_raw(game: &GameState) -> GraphData {
    game_to_graph_raw_opts(game, false, false)
}

/// Build graph arrays with optional threat-encoding node features and
/// optional side-to-move-relative (own/opp) stone one-hots. Node dim:
/// 8 (absolute), 7 (relative), 12 (absolute + threat), 11 (relative + threat).
pub fn game_to_graph_raw_opts(
    game: &GameState,
    threat_features: bool,
    relative_stones: bool,
) -> GraphData {
    let mut stones = game.placed_stones();
    stones.sort_by_key(|&(coord, _)| coord);
    let legal = game.legal_moves(); // already sorted by engine

    let player_feat: f32 = match game.current_player() {
        Some(Player::P1) => 1.0,
        _ => -1.0,
    };
    let moves_feat: f32 = game.moves_remaining_this_turn() as f32 / 2.0;

    let mut g = build_graph(
        &stones,
        &legal,
        player_feat,
        moves_feat,
        threat_features,
        relative_stones,
    );
    if threat_features {
        // hex graph has no dummy node; all num_nodes are real
        let base_dim = if relative_stones { 7 } else { 8 };
        fill_threat_features(&mut g.features, &g.coords, g.num_nodes, game, base_dim);
    }
    g
}

/// Apply all 11 non-identity D6 transforms to produce augmented graphs.
///
/// Returns Vec of (GraphData, permutation) where permutation[i] gives the
/// index in the ORIGINAL legal moves array that maps to position i in the
/// augmented (re-sorted) legal moves.
pub fn augment_graph(
    stones: &[(Coord, Player)],
    legal: &[Coord],
    player_feat: f32,
    moves_feat: f32,
) -> Vec<(GraphData, Vec<usize>)> {
    // Skip transform 0 (identity)
    D6_TRANSFORMS[1..]
        .iter()
        .map(|transform| {
            // Transform and sort stones
            let mut t_stones: Vec<(Coord, Player)> = stones
                .iter()
                .map(|&(c, p)| (transform(c), p))
                .collect();
            t_stones.sort_by_key(|&(c, _)| c);

            // Transform legal moves, keeping track of original indices
            let mut indexed_legal: Vec<(Coord, usize)> = legal
                .iter()
                .enumerate()
                .map(|(i, &c)| (transform(c), i))
                .collect();
            indexed_legal.sort_by_key(|&(c, _)| c);

            let t_legal: Vec<Coord> = indexed_legal.iter().map(|&(c, _)| c).collect();
            let permutation: Vec<usize> = indexed_legal.iter().map(|&(_, i)| i).collect();

            // augmentation path does not support threat_features or
            // relative_stones; always 8-dim absolute
            let graph = build_graph(&t_stones, &t_legal, player_feat, moves_feat, false, false);
            (graph, permutation)
        })
        .collect()
}

/// Build graph arrays for a batch of game states in parallel using rayon.
pub fn game_to_graph_batch_opts(
    games: &[GameState],
    threat_features: bool,
    relative_stones: bool,
) -> Vec<GraphData> {
    use rayon::prelude::*;
    games
        .par_iter()
        .map(|g| game_to_graph_raw_opts(g, threat_features, relative_stones))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::GameConfig;

    fn small_game() -> GameState {
        GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        })
    }

    #[test]
    fn graph_has_correct_node_count() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // 1 stone (P1 at origin) + legal_moves
        let expected = 1 + game.legal_moves().len();
        assert_eq!(g.num_nodes, expected);
        assert_eq!(g.features.len(), expected * 8);
        assert_eq!(g.coords.len(), expected * 2);
    }

    #[test]
    fn threat_features_widen_to_12_hex() {
        let game = small_game();
        let g = game_to_graph_raw_opts(&game, true, false);
        assert_eq!(g.features.len(), g.num_nodes * 12);
        let g8 = game_to_graph_raw_opts(&game, false, false);
        assert_eq!(g8.features.len(), g8.num_nodes * 8);
        for node in 0..g8.num_nodes {
            for d in 0..8 {
                assert_eq!(g.features[node * 12 + d], g8.features[node * 8 + d]);
            }
        }
        // Threat dims must actually be written: node 0 is the lone P1 stone at
        // origin and P2 is to move ("own" = P2). The stone counts itself in a
        // P2-clean window, so opp_max_line (dim 9) = 1/win_length = 1/4.
        assert_eq!(
            g.features[0 * 12 + 9],
            0.25,
            "threat dims must be populated for real nodes"
        );
    }

    /// small_game after one full P2 turn: P1 stone at origin, P2 stones at
    /// (1,0) and (-1,0), P1 to move.
    fn p1_to_move_game() -> GameState {
        let mut game = small_game();
        game.apply_move((1, 0)).unwrap();
        game.apply_move((-1, 0)).unwrap();
        game
    }

    #[test]
    fn relative_stones_layout_p2_to_move() {
        // Initial position: P2 to move, so the lone P1 stone is "opp".
        let game = small_game();
        let g = game_to_graph_raw_opts(&game, false, true);
        assert_eq!(g.features.len(), g.num_nodes * 7);
        assert_eq!(g.features[0], 0.0, "node 0 (P1 stone) is not own for P2");
        assert_eq!(g.features[1], 1.0, "node 0 (P1 stone) is opp for P2");
        assert_eq!(g.features[2], 0.0, "node 0 is not empty");

        // Base features at shifted indices match the absolute builder:
        // rel [2] == abs [2] (empty); rel [3..7] == abs [4..8]; no to_move.
        let g8 = game_to_graph_raw_opts(&game, false, false);
        for node in 0..g.num_nodes {
            assert_eq!(g.features[node * 7 + 2], g8.features[node * 8 + 2]);
            for d in 0..4 {
                assert_eq!(
                    g.features[node * 7 + 3 + d],
                    g8.features[node * 8 + 4 + d],
                    "node {node}: shifted base feature {d} mismatch"
                );
            }
        }
    }

    #[test]
    fn relative_stones_own_opp_flip_with_side_to_move() {
        // P1 to move: stones sorted by coord are (-1,0)=P2, (0,0)=P1, (1,0)=P2.
        let game = p1_to_move_game();
        let g = game_to_graph_raw_opts(&game, false, true);
        // (-1,0): P2 stone, P1 to move → opp
        assert_eq!(g.features[0 * 7], 0.0);
        assert_eq!(g.features[0 * 7 + 1], 1.0);
        // (0,0): P1 stone, P1 to move → own
        assert_eq!(g.features[1 * 7], 1.0);
        assert_eq!(g.features[1 * 7 + 1], 0.0);
        // (1,0): P2 stone → opp
        assert_eq!(g.features[2 * 7], 0.0);
        assert_eq!(g.features[2 * 7 + 1], 1.0);

        // The absolute builder for the same position has its to_move channel;
        // relative graphs have no such column — verify the legal nodes carry
        // moves_feat at [3] instead (abs keeps it at [4]).
        let g8 = game_to_graph_raw_opts(&game, false, false);
        assert_eq!(g8.features[3], 1.0, "abs: P1 to move");
        for node in 0..g.num_nodes {
            assert_eq!(g.features[node * 7 + 3], g8.features[node * 8 + 4]);
        }
    }

    #[test]
    fn relative_threat_features_at_7_to_11() {
        let game = p1_to_move_game();
        let g11 = game_to_graph_raw_opts(&game, true, true);
        assert_eq!(g11.features.len(), g11.num_nodes * 11);
        let g12 = game_to_graph_raw_opts(&game, true, false);
        // Threat dims are already side-to-move relative; identical values at
        // 7..11 (relative) vs 8..12 (absolute).
        for node in 0..g11.num_nodes {
            for d in 0..4 {
                assert_eq!(
                    g11.features[node * 11 + 7 + d],
                    g12.features[node * 12 + 8 + d],
                    "node {node}: threat dim {d} mismatch between layouts"
                );
            }
        }
        // Threat dims must actually be populated (not all-zero).
        assert!(
            g11.features
                .chunks(11)
                .any(|f| f[7..11].iter().any(|&v| v != 0.0)),
            "threat dims must be populated for real nodes"
        );
    }

    #[test]
    fn features_are_8_dim() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        assert_eq!(g.features.len() % 8, 0);
    }

    #[test]
    fn stone_one_hot_is_correct() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // First node is P1 stone at (0,0)
        assert_eq!(g.features[0], 1.0); // P1
        assert_eq!(g.features[1], 0.0); // not P2
        assert_eq!(g.features[2], 0.0); // not empty
    }

    #[test]
    fn legal_node_has_empty_one_hot() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // Second node (first legal move) should be empty
        let base = 1 * 8; // node index 1
        assert_eq!(g.features[base], 0.0);     // not P1
        assert_eq!(g.features[base + 1], 0.0); // not P2
        assert_eq!(g.features[base + 2], 1.0); // empty
    }

    #[test]
    fn masks_are_correct() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        assert!(g.stone_mask[0]);   // first node is a stone
        assert!(!g.legal_mask[0]);  // first node is not legal
        assert!(!g.stone_mask[1]);  // second node is not a stone
        assert!(g.legal_mask[1]);   // second node is legal
    }

    #[test]
    fn edges_are_symmetric() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // Every (src, dst) should have a corresponding (dst, src)
        let edges: std::collections::HashSet<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(&s, &d)| (s, d))
            .collect();
        for (&s, &d) in g.edge_src.iter().zip(&g.edge_dst) {
            assert!(edges.contains(&(d, s)), "missing reverse edge ({d}, {s})");
        }
    }

    #[test]
    fn current_player_feature() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // Game starts with P2 to move → player_feat = -1.0
        assert_eq!(g.features[3], -1.0);
    }

    #[test]
    fn inverse_distance_is_positive_for_legal() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        let n_stones = g.stone_mask.iter().filter(|&&b| b).count();
        for j in 0..g.legal_mask.iter().filter(|&&b| b).count() {
            let base = (n_stones + j) * 8;
            assert!(g.features[base + 7] > 0.0, "legal node {j} should have positive inv distance");
        }
    }

    #[test]
    fn stone_inverse_distance_is_zero() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        assert_eq!(g.features[7], 0.0); // stone node's inv distance
    }

    #[test]
    fn neighbor_index_correct_for_origin() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // Node 0 is the stone at (0,0). Check its 6 neighbors are present
        // (all should be legal moves adjacent to origin)
        let ni = &g.neighbor_index[0..6];
        // At least some neighbors should be valid (>= 0)
        assert!(ni.iter().any(|&idx| idx >= 0));
        // None should be out of bounds
        for &idx in ni {
            assert!(idx == -1 || (idx >= 0 && (idx as usize) < g.num_nodes));
        }
    }

    #[test]
    fn neighbor_index_has_correct_length() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        assert_eq!(g.neighbor_index.len(), g.num_nodes * 6);
    }

    #[test]
    fn augment_produces_11_graphs() {
        let game = small_game();
        let mut stones = game.placed_stones();
        stones.sort_by_key(|&(c, _)| c);
        let legal = game.legal_moves();
        let results = augment_graph(&stones, &legal, -1.0, 1.0);
        assert_eq!(results.len(), 11);
    }

    #[test]
    fn augmented_graphs_have_same_node_count() {
        let game = small_game();
        let mut stones = game.placed_stones();
        stones.sort_by_key(|&(c, _)| c);
        let legal = game.legal_moves();
        let original = game_to_graph_raw(&game);
        let augmented = augment_graph(&stones, &legal, -1.0, 1.0);
        for (g, _perm) in &augmented {
            assert_eq!(g.num_nodes, original.num_nodes);
            assert_eq!(g.features.len(), original.features.len());
        }
    }

    #[test]
    fn permutation_has_correct_length() {
        let game = small_game();
        let mut stones = game.placed_stones();
        stones.sort_by_key(|&(c, _)| c);
        let legal = game.legal_moves();
        let augmented = augment_graph(&stones, &legal, -1.0, 1.0);
        for (_g, perm) in &augmented {
            assert_eq!(perm.len(), legal.len());
        }
    }

    #[test]
    fn self_loops_present() {
        let stones = vec![((0, 0), Player::P1)];
        let legal = vec![(1, 0), (0, 1)];
        let graph = build_graph(&stones, &legal, 1.0, 0.5, false, false);

        let n = graph.num_nodes;
        assert_eq!(n, 3);

        let mut self_loop_count = 0;
        for k in 0..graph.edge_src.len() {
            if graph.edge_src[k] == graph.edge_dst[k] {
                self_loop_count += 1;
            }
        }
        assert_eq!(self_loop_count, n, "expected one self-loop per node");
    }

    #[test]
    fn permutation_is_a_valid_bijection() {
        let game = small_game();
        let mut stones = game.placed_stones();
        stones.sort_by_key(|&(c, _)| c);
        let legal = game.legal_moves();
        let augmented = augment_graph(&stones, &legal, -1.0, 1.0);
        for (_g, perm) in &augmented {
            let mut sorted = perm.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), legal.len());
        }
    }
}
