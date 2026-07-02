//! Axis-window graph construction: connect nodes along the 3 hex axes within
//! a distance window, with edge features.
//!
//! Unlike `graph.rs` (hex-adjacency, distance 1 only), this builder creates
//! edges along each WIN_AXIS direction up to `win_length - 1` steps, plus a
//! global dummy node connected to all other nodes.

use std::collections::HashMap;

use hexo_engine::hex::hex_distance;
use hexo_engine::types::{Coord, Player, WIN_AXES};
use hexo_engine::GameState;

/// Raw axis-window graph data ready to be wrapped into PyG Data on the Python side.
#[derive(serde::Serialize)]
pub struct AxisGraphData {
    /// Node features, flattened row-major, including the dummy node. Per-node
    /// width is 8 (absolute) or 7 (`relative_stones`: to_move dropped), +4
    /// with threat features — i.e. (N+1)×{7,8,11,12}. The dummy node keeps
    /// zeros in the threat dims.
    pub features: Vec<f32>,
    /// Edge source indices.
    pub edge_src: Vec<i64>,
    /// Edge destination indices.
    pub edge_dst: Vec<i64>,
    /// Edge attributes, flattened: E×5 row-major.
    pub edge_attr: Vec<f32>,
    /// Legal move mask (true for legal-move nodes).
    pub legal_mask: Vec<bool>,
    /// Stone mask (true for placed-stone nodes).
    pub stone_mask: Vec<bool>,
    /// Coordinates, flattened: (N+1)×2 row-major (q, r).
    pub coords: Vec<i32>,
    /// Number of nodes (N+1, includes dummy).
    pub num_nodes: usize,
}

/// Node classification for walk-stopping logic.
#[derive(Clone, Copy)]
enum NodeKind {
    Stone(Player),
    Empty,
}

/// Extract axis index (0, 1, or 2) from edge_attr at position e.
fn axis_idx_of(edge_attr: &[f32], e: usize) -> u8 {
    let base = e * 5;
    if edge_attr[base] > 0.5 { 0 }
    else if edge_attr[base + 1] > 0.5 { 1 }
    else { 2 }
}

/// Build axis-window graph arrays from pre-sorted stones and legal moves.
///
/// `stones` and `legal` must already be sorted by coord tuple (q, r).
fn build_axis_graph(
    stones: &[(Coord, Player)],
    legal: &[Coord],
    player_feat: f32,
    moves_feat: f32,
    win_length: u8,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> AxisGraphData {
    // Node-feature layout mirrors graph.rs::build_graph: 8-dim absolute base
    // or 7-dim side-to-move-relative base (own/opp one-hot, to_move dropped),
    // +4 optional threat dims.
    let base_dim: usize = if relative_stones { 7 } else { 8 };
    let fdim: usize = base_dim + if threat_features { 4 } else { 0 };
    let off: usize = usize::from(!relative_stones);
    // Terminal fallback matches the absolute to_move fill: player_feat = -1.0
    // when current_player() is None → own = P2.
    let own_is_p1 = player_feat > 0.0;
    debug_assert!(win_length >= 2, "win_length must be >= 2, got {win_length}");
    let n_stones = stones.len();
    let n_legal = legal.len();
    let n_real = n_stones + n_legal; // nodes without dummy
    let n = n_real + 1; // total nodes including dummy
    let dummy_idx = n_real;
    let window = win_length as i32 - 1;

    // --- Coordinate arrays and lookup maps ---

    let mut coords = Vec::with_capacity(n * 2);
    let mut coord_to_idx: HashMap<Coord, usize> = HashMap::with_capacity(n_real);
    let mut node_kind: Vec<NodeKind> = Vec::with_capacity(n);

    for (i, &(coord, player)) in stones.iter().enumerate() {
        coords.push(coord.0);
        coords.push(coord.1);
        coord_to_idx.insert(coord, i);
        node_kind.push(NodeKind::Stone(player));
    }
    for (j, &coord) in legal.iter().enumerate() {
        let idx = n_stones + j;
        coords.push(coord.0);
        coords.push(coord.1);
        coord_to_idx.insert(coord, idx);
        node_kind.push(NodeKind::Empty);
    }
    // Dummy node coords
    coords.push(0);
    coords.push(0);
    // node_kind for dummy is not used directly; we handle dummy separately.

    // --- Stone centroid and spread ---

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

    let stone_coords: Vec<Coord> = stones.iter().map(|&(c, _)| c).collect();

    // --- Node features (N+1)×fdim ---

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
        // features[base + 6 + off] = 0.0 (stones)
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

        let coord = legal[j];
        let min_d = stone_coords
            .iter()
            .map(|&sc| hex_distance(coord, sc))
            .min()
            .unwrap_or(1);
        features[base + 6 + off] = 1.0 / min_d.max(1) as f32;
    }

    // Dummy node features carry the global scalars, no one-hot. Absolute
    // layout: [0,0,0, player_feat, moves_feat, 0, 0, 0]; relative layout:
    // [0,0,0, moves_feat, 0, 0, 0] (to_move doesn't exist in the relative
    // frame). Threat dims, if present, stay zero in both modes.
    let dummy_base = dummy_idx * fdim;
    if !relative_stones {
        features[dummy_base + 3] = player_feat;
    }
    features[dummy_base + 3 + off] = moves_feat;

    // --- Masks ---

    let mut legal_mask = vec![false; n];
    let mut stone_mask = vec![false; n];
    for i in 0..n_stones {
        stone_mask[i] = true;
    }
    for j in 0..n_legal {
        legal_mask[n_stones + j] = true;
    }
    // Dummy: both false (default)

    // --- Axis-window edges ---

    let mut edge_src: Vec<i64> = Vec::new();
    let mut edge_dst: Vec<i64> = Vec::new();
    let mut edge_attr: Vec<f32> = Vec::new();

    // For each node i, for each of 3 axes, walk BOTH directions.
    // Each walk creates only the forward edge (i→j); the opposite
    // walk from j creates j→i, so stopping is applied symmetrically
    // from both perspectives.
    for i in 0..n_real {
        let iq = coords[i * 2];
        let ir = coords[i * 2 + 1];
        let i_kind = node_kind[i];

        for (axis_idx, &(dq, dr)) in WIN_AXES.iter().enumerate() {
            for &sign in &[1i32, -1] {
                let sdq = dq * sign;
                let sdr = dr * sign;
                for d in 1..=window {
                    let tq = iq + sdq * d;
                    let tr = ir + sdr * d;

                    let j = match coord_to_idx.get(&(tq, tr)) {
                        Some(&j) => j,
                        None => break, // off-graph: stop walking this ray
                    };

                    let j_kind = node_kind[j];

                    // Skip empty→empty edges when pruning is enabled,
                    // but continue the walk so empties can still reach
                    // stones on the far side of other empties.
                    let both_empty = matches!(i_kind, NodeKind::Empty)
                        && matches!(j_kind, NodeKind::Empty);

                    if !(prune_empty_edges && both_empty) {
                        // Source player identity: +1 P1, -1 P2, 0 empty.
                        // The receiving node (dst) uses this to know what's
                        // sending it a message — critical for empty nodes to
                        // detect opponent threats via incoming edges.
                        let src_player_i = match i_kind {
                            NodeKind::Stone(Player::P1) => 1.0f32,
                            NodeKind::Stone(Player::P2) => -1.0f32,
                            NodeKind::Empty => 0.0,
                        };
                        let src_player_j = match j_kind {
                            NodeKind::Stone(Player::P1) => 1.0f32,
                            NodeKind::Stone(Player::P2) => -1.0f32,
                            NodeKind::Empty => 0.0,
                        };

                        // Add bidirectional edges (i→j and j→i).
                        let signed_dist = (d * sign) as f32;
                        edge_src.push(i as i64);
                        edge_dst.push(j as i64);
                        let mut attr_fwd = [0.0f32; 5];
                        attr_fwd[axis_idx] = 1.0;
                        attr_fwd[3] = signed_dist;
                        attr_fwd[4] = src_player_i;  // i is the source
                        edge_attr.extend_from_slice(&attr_fwd);

                        edge_src.push(j as i64);
                        edge_dst.push(i as i64);
                        let mut attr_rev = [0.0f32; 5];
                        attr_rev[axis_idx] = 1.0;
                        attr_rev[3] = -signed_dist;
                        attr_rev[4] = src_player_j;  // j is the source
                        edge_attr.extend_from_slice(&attr_rev);
                    }

                    // Walk stopping:
                    // - Stones stop at opponent stones (line is blocked).
                    // - Empties stop at ANY stone (they only need edges
                    //   to their nearest stone neighbors on each axis;
                    //   the stones carry long-range line information).
                    let should_stop = match i_kind {
                        NodeKind::Stone(pi) => {
                            matches!(j_kind, NodeKind::Stone(pj) if pj != pi)
                        }
                        NodeKind::Empty => {
                            matches!(j_kind, NodeKind::Stone(_))
                        }
                    };
                    if should_stop {
                        break;
                    }
                }
            }
        }
    }

    // --- Dedup axis edges ---
    // Walking both directions creates duplicates when both i→j and j→i
    // walks discover the same pair. Keep the first occurrence.
    {
        let n_edges = edge_src.len();
        let mut seen = std::collections::HashSet::with_capacity(n_edges);
        let mut keep = Vec::with_capacity(n_edges);
        for e in 0..n_edges {
            let key = (edge_src[e], edge_dst[e], axis_idx_of(&edge_attr, e));
            if seen.insert(key) {
                keep.push(e);
            }
        }
        if keep.len() < n_edges {
            let mut new_src = Vec::with_capacity(keep.len());
            let mut new_dst = Vec::with_capacity(keep.len());
            let mut new_attr = Vec::with_capacity(keep.len() * 5);
            for &e in &keep {
                new_src.push(edge_src[e]);
                new_dst.push(edge_dst[e]);
                new_attr.extend_from_slice(&edge_attr[e * 5..(e + 1) * 5]);
            }
            edge_src = new_src;
            edge_dst = new_dst;
            edge_attr = new_attr;
        }
    }

    // --- Dummy edges: bidirectional to all real nodes ---

    for i in 0..n_real {
        // dummy -> i
        edge_src.push(dummy_idx as i64);
        edge_dst.push(i as i64);
        edge_attr.extend_from_slice(&[0.0; 5]);

        // i -> dummy
        edge_src.push(i as i64);
        edge_dst.push(dummy_idx as i64);
        edge_attr.extend_from_slice(&[0.0; 5]);
    }

    AxisGraphData {
        features,
        edge_src,
        edge_dst,
        edge_attr,
        legal_mask,
        stone_mask,
        coords,
        num_nodes: n,
    }
}

/// Fill the threat-encoding node dims of `graph` from `game`'s board, when
/// `threat_features` is set. Real nodes only — the dummy node (last) keeps its
/// zeros, so `n_real = num_nodes - 1`. `base_dim` is the stone-feature width
/// (7 relative, 8 absolute) that the threat dims are appended after. The caller
/// must pass the game whose board the threats should be read from (the original
/// game for `game_to_axis_graph_raw_opts`, the transformed game for augment).
fn fill_axis_threat(
    graph: &mut AxisGraphData,
    game: &GameState,
    threat_features: bool,
    relative_stones: bool,
) {
    if threat_features {
        let n_real = graph.num_nodes - 1;
        let base_dim = if relative_stones { 7 } else { 8 };
        crate::graph::fill_threat_features(&mut graph.features, &graph.coords, n_real, game, base_dim);
    }
}

/// Build axis-window graph arrays from a GameState.
pub fn game_to_axis_graph_raw(game: &GameState) -> AxisGraphData {
    game_to_axis_graph_raw_opts(game, false, false, false)
}

/// Build axis-window graph arrays with optional empty-edge pruning, optional
/// threat-encoding node features, and optional side-to-move-relative
/// (own/opp) stone one-hots. Node dim: 8 (absolute), 7 (relative),
/// 12 (absolute + threat), 11 (relative + threat).
pub fn game_to_axis_graph_raw_opts(
    game: &GameState,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> AxisGraphData {
    let mut stones = game.placed_stones();
    stones.sort_by_key(|&(coord, _)| coord);
    let legal = game.legal_moves(); // already sorted by engine

    let player_feat: f32 = match game.current_player() {
        Some(Player::P1) => 1.0,
        _ => -1.0,
    };
    let moves_feat: f32 = game.moves_remaining_this_turn() as f32 / 2.0;
    let win_length = game.config().win_length;

    let mut g = build_axis_graph(
        &stones,
        &legal,
        player_feat,
        moves_feat,
        win_length,
        prune_empty_edges,
        threat_features,
        relative_stones,
    );
    // Real nodes only — threats read from the original game's board.
    fill_axis_threat(&mut g, game, threat_features, relative_stones);
    g
}

/// Build axis-window graph arrays for a batch of game states in parallel.
pub fn game_to_axis_graph_batch(games: &[GameState]) -> Vec<AxisGraphData> {
    game_to_axis_graph_batch_opts(games, false, false, false)
}

/// Build axis-window graph arrays for a batch with optional empty-edge
/// pruning, optional threat-encoding node features, and optional
/// side-to-move-relative stone one-hots.
pub fn game_to_axis_graph_batch_opts(
    games: &[GameState],
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> Vec<AxisGraphData> {
    use rayon::prelude::*;
    games
        .par_iter()
        .map(|g| game_to_axis_graph_raw_opts(g, prune_empty_edges, threat_features, relative_stones))
        .collect()
}

/// Produce 11 augmented axis-window graphs by applying the non-identity D6
/// transforms to the input game state. Returns `(AxisGraphData, permutation)`
/// pairs, where `permutation[new_legal_idx] = old_legal_idx`.
pub fn augment_axis_graph(game: &GameState) -> Vec<(AxisGraphData, Vec<usize>)> {
    augment_axis_graph_opts(game, false)
}

/// Augment with optional empty-edge pruning (8-dim absolute features).
pub fn augment_axis_graph_opts(game: &GameState, prune_empty_edges: bool) -> Vec<(AxisGraphData, Vec<usize>)> {
    augment_axis_graph_all_opts(game, prune_empty_edges, false, false)
}

/// Augment with optional empty-edge pruning and optional threat / relative
/// node-feature encodings. Loops the 11 non-identity D6 transforms, calling
/// [`augment_axis_graph_single`] for each.
pub fn augment_axis_graph_all_opts(
    game: &GameState,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> Vec<(AxisGraphData, Vec<usize>)> {
    use hexo_engine::symmetry::D6_TRANSFORMS;

    D6_TRANSFORMS[1..]
        .iter()
        .map(|&transform| {
            augment_axis_graph_single(
                game,
                transform,
                prune_empty_edges,
                threat_features,
                relative_stones,
            )
        })
        .collect()
}

/// Apply a single D6 `transform` to `game`'s board and build the axis-window
/// graph for the transformed position, honouring the threat / relative
/// node-feature flags. Returns `(graph, permutation)` where
/// `permutation[new_legal_idx] = old_legal_idx`.
///
/// The transform is applied directly to the stone and legal-move coordinates
/// (then re-sorted), which is equivalent to — and avoids the cost of — replaying
/// moves on a freshly reconstructed `GameState`. When `threat_features` is set
/// the threat dims must be computed against the transformed board, so a
/// transformed `GameState` is materialised via [`GameState::from_state`] purely
/// to drive `fill_threat_features` (mirroring `game_to_axis_graph_raw_opts`).
pub fn augment_axis_graph_single(
    game: &GameState,
    transform: fn(Coord) -> Coord,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> (AxisGraphData, Vec<usize>) {
    let mut stones = game.placed_stones();
    stones.sort_by_key(|&(c, _)| c);
    let legal = game.legal_moves(); // already sorted

    let player_feat: f32 = match game.current_player() {
        Some(Player::P1) => 1.0,
        _ => -1.0,
    };
    let moves_feat: f32 = game.moves_remaining_this_turn() as f32 / 2.0;
    let win_length = game.config().win_length;

    // Transform and sort stones.
    let mut t_stones: Vec<(Coord, Player)> = stones
        .iter()
        .map(|&(c, p)| (transform(c), p))
        .collect();
    t_stones.sort_by_key(|&(c, _)| c);

    // Transform legal moves, tracking original indices for the permutation.
    let mut indexed_legal: Vec<(Coord, usize)> = legal
        .iter()
        .enumerate()
        .map(|(i, &c)| (transform(c), i))
        .collect();
    indexed_legal.sort_by_key(|&(c, _)| c);

    let t_legal: Vec<Coord> = indexed_legal.iter().map(|&(c, _)| c).collect();
    let permutation: Vec<usize> = indexed_legal.iter().map(|&(_, i)| i).collect();

    let mut graph = build_axis_graph(
        &t_stones,
        &t_legal,
        player_feat,
        moves_feat,
        win_length,
        prune_empty_edges,
        threat_features,
        relative_stones,
    );

    if threat_features {
        // `build_axis_graph` zeroes the threat dims; fill them from the
        // transformed board, exactly as `game_to_axis_graph_raw_opts` does.
        let t_game = GameState::from_state(
            &t_stones,
            game.current_player()
                .expect("augment_axis_graph_single called on terminal GameState"),
            game.moves_remaining_this_turn(),
            *game.config(),
        );
        fill_axis_threat(&mut graph, &t_game, threat_features, relative_stones);
    }

    (graph, permutation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{GameConfig, GameState};
    use std::collections::{HashMap, HashSet};

    fn small_game() -> GameState {
        GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        })
    }

    /// Apply moves to create a mid-game state with stones from both players.
    /// P1 starts at (0,0). P2 gets 2 moves, then alternating 2 each.
    fn mid_game() -> GameState {
        let mut game = small_game();
        // P2 turn (2 moves)
        game.apply_move((1, 0)).unwrap();
        game.apply_move((-1, 0)).unwrap();
        // P1 turn (2 moves)
        game.apply_move((0, 1)).unwrap();
        game.apply_move((0, -1)).unwrap();
        // P2 turn (2 moves)
        game.apply_move((1, -1)).unwrap();
        game.apply_move((-1, 1)).unwrap();
        // P1 turn (2 moves)
        game.apply_move((2, 0)).unwrap();
        game.apply_move((-2, 0)).unwrap();
        // P2 turn (2 moves)
        game.apply_move((0, 2)).unwrap();
        game.apply_move((0, -2)).unwrap();
        game
    }

    // AC-1a: num_nodes == n_stones + n_legal + 1
    #[test]
    fn ac1a_initial_position_node_count() {
        let game = small_game();
        let g = game_to_axis_graph_raw(&game);
        let n_stones = game.placed_stones().len();
        let n_legal = game.legal_moves().len();
        assert_eq!(g.num_nodes, n_stones + n_legal + 1);
        assert_eq!(g.features.len(), g.num_nodes * 8);
        assert_eq!(g.coords.len(), g.num_nodes * 2);
    }

    #[test]
    fn threat_features_widen_to_12() {
        let game = small_game();
        let g = game_to_axis_graph_raw_opts(&game, false, true, false);
        assert_eq!(g.features.len(), g.num_nodes * 12);
        let g8 = game_to_axis_graph_raw_opts(&game, false, false, false);
        assert_eq!(g8.features.len(), g8.num_nodes * 8);
        for node in 0..g8.num_nodes {
            for d in 0..8 {
                assert_eq!(g.features[node * 12 + d], g8.features[node * 8 + d]);
            }
        }
        let dummy = g.num_nodes - 1;
        for d in 8..12 {
            assert_eq!(g.features[dummy * 12 + d], 0.0);
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

    #[test]
    fn relative_stones_layout_axis() {
        // mid_game: P1 to move next? Trace: P2,P1,P2,P1,P2 turns → P1 to move.
        // Use both the initial position (P2 to move) and mid_game (P1 to move)
        // to verify the own/opp flip.
        let game = small_game(); // P2 to move; lone P1 stone at origin
        let g = game_to_axis_graph_raw_opts(&game, false, false, true);
        assert_eq!(g.features.len(), g.num_nodes * 7);
        // Node 0 = P1 stone at origin → opp for P2
        assert_eq!(g.features[0], 0.0);
        assert_eq!(g.features[1], 1.0);
        // Shifted base features match the absolute builder for real nodes:
        // rel [2] == abs [2]; rel [3..7] == abs [4..8].
        let g8 = game_to_axis_graph_raw_opts(&game, false, false, false);
        let n_real = g.num_nodes - 1;
        for node in 0..n_real {
            assert_eq!(g.features[node * 7 + 2], g8.features[node * 8 + 2]);
            for d in 0..4 {
                assert_eq!(
                    g.features[node * 7 + 3 + d],
                    g8.features[node * 8 + 4 + d],
                    "node {node}: shifted base feature {d} mismatch"
                );
            }
        }
        // Dummy node in the relative layout: moves_feat at [3] (the relative
        // moves_remaining slot), everything else zero (no to_move channel).
        let dummy = g.num_nodes - 1;
        assert_eq!(
            g.features[dummy * 7 + 3],
            g8.features[dummy * 8 + 4],
            "dummy carries moves_feat at the shifted slot"
        );
        assert_eq!(g.features[dummy * 7 + 3], 1.0, "moves_remaining 2/2.0 = 1.0");
        for d in [0, 1, 2, 4, 5, 6] {
            assert_eq!(g.features[dummy * 7 + d], 0.0, "dummy dim {d} must be zero");
        }

        // Flip check: with P1 to move (mid_game), a P1 stone is "own".
        let game2 = mid_game();
        let g2 = game_to_axis_graph_raw_opts(&game2, false, false, true);
        let stones: HashMap<Coord, Player> = game2.placed_stones().into_iter().collect();
        assert_eq!(
            game2.current_player(),
            Some(Player::P1),
            "mid_game fixture should have P1 to move"
        );
        let n_real2 = g2.num_nodes - 1;
        let mut checked = 0;
        for node in 0..n_real2 {
            if !g2.stone_mask[node] {
                continue;
            }
            let c = (g2.coords[node * 2], g2.coords[node * 2 + 1]);
            let expected_own = stones[&c] == Player::P1;
            assert_eq!(g2.features[node * 7] == 1.0, expected_own, "node {node} own bit");
            assert_eq!(g2.features[node * 7 + 1] == 1.0, !expected_own, "node {node} opp bit");
            checked += 1;
        }
        assert!(checked >= 5, "should have checked several stones");
    }

    #[test]
    fn relative_threat_features_axis_11_dim() {
        let game = mid_game();
        let g11 = game_to_axis_graph_raw_opts(&game, false, true, true);
        assert_eq!(g11.features.len(), g11.num_nodes * 11);
        let g12 = game_to_axis_graph_raw_opts(&game, false, true, false);
        let n_real = g11.num_nodes - 1;
        // Threat values at 7..11 (relative) match 8..12 (absolute).
        for node in 0..n_real {
            for d in 0..4 {
                assert_eq!(
                    g11.features[node * 11 + 7 + d],
                    g12.features[node * 12 + 8 + d],
                    "node {node}: threat dim {d} mismatch between layouts"
                );
            }
        }
        // Threat dims must actually be populated for some real node.
        assert!(
            g11.features[..n_real * 11]
                .chunks(11)
                .any(|f| f[7..11].iter().any(|&v| v != 0.0)),
            "threat dims must be populated"
        );
        // Dummy node: moves_feat at [3], everything else zero — including the
        // threat dims 7..11.
        let dummy = g11.num_nodes - 1;
        assert_eq!(
            g11.features[dummy * 11 + 3],
            g12.features[dummy * 12 + 4],
            "dummy carries moves_feat at the shifted slot"
        );
        for d in (0..11).filter(|&d| d != 3) {
            assert_eq!(g11.features[dummy * 11 + d], 0.0, "dummy dim {d} must be zero");
        }
    }

    // AC-1b: All axis edges have |signed_distance| <= win_length
    #[test]
    fn ac1b_signed_distance_within_window() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let wl = game.config().win_length as f32;
        let n_edges = g.edge_src.len();
        for e in 0..n_edges {
            let dist = g.edge_attr[e * 5 + 3];
            assert!(
                dist.abs() <= wl,
                "edge {e}: |signed_distance| = {} > win_length = {}",
                dist.abs(),
                wl
            );
        }
    }

    // AC-1c: edge_attr is 5-dim per edge
    #[test]
    fn ac1c_edge_attr_length() {
        let game = small_game();
        let g = game_to_axis_graph_raw(&game);
        assert_eq!(g.edge_attr.len(), g.edge_src.len() * 5);
    }

    // AC-1d: Axis one-hot sums to 0.0 (dummy) or 1.0 (axis edges)
    #[test]
    fn ac1d_axis_one_hot_valid() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let dummy = (g.num_nodes - 1) as i64;
        let n_edges = g.edge_src.len();
        for e in 0..n_edges {
            let oh_sum: f32 =
                g.edge_attr[e * 5] + g.edge_attr[e * 5 + 1] + g.edge_attr[e * 5 + 2];
            let is_dummy_edge =
                g.edge_src[e] == dummy || g.edge_dst[e] == dummy;
            if is_dummy_edge {
                assert_eq!(
                    oh_sum, 0.0,
                    "dummy edge {e} should have axis one-hot sum 0.0, got {oh_sum}"
                );
            } else {
                assert_eq!(
                    oh_sum, 1.0,
                    "axis edge {e} should have axis one-hot sum 1.0, got {oh_sum}"
                );
            }
        }
    }

    // AC-1e: Edges are symmetric — every (s,d) has a corresponding (d,s)
    #[test]
    fn ac1e_edges_symmetric() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let edges: HashSet<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(s, d)| (*s, *d))
            .collect();
        for (s, d) in g.edge_src.iter().zip(&g.edge_dst) {
            assert!(
                edges.contains(&(*d, *s)),
                "missing reverse edge ({d}, {s}) for ({s}, {d})"
            );
        }
    }

    // AC-1f: Dummy node connects to every non-dummy node (bidirectional)
    #[test]
    fn ac1f_dummy_connects_all() {
        let game = small_game();
        let g = game_to_axis_graph_raw(&game);
        let dummy = (g.num_nodes - 1) as i64;
        let n_real = g.num_nodes - 1;

        // Collect nodes connected to dummy as source
        let from_dummy: HashSet<i64> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .filter(|(s, _)| **s == dummy)
            .map(|(_, d)| *d)
            .collect();
        // Collect nodes connected to dummy as destination
        let to_dummy: HashSet<i64> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .filter(|(_, d)| **d == dummy)
            .map(|(s, _)| *s)
            .collect();

        for i in 0..n_real {
            assert!(
                from_dummy.contains(&(i as i64)),
                "dummy should connect to node {i}"
            );
            assert!(
                to_dummy.contains(&(i as i64)),
                "node {i} should connect to dummy"
            );
        }
    }

    // AC-1g: All hex-neighbor edges (distance 1) present — axis edges subsume hex adjacency
    #[test]
    fn ac1g_subsumes_hex_neighbors() {
        use crate::graph::game_to_graph_raw;

        let game = mid_game();
        let hex_g = game_to_graph_raw(&game);
        let axis_g = game_to_axis_graph_raw(&game);

        // Build coord->index maps for both graphs
        // Hex graph has no dummy, axis graph does. Use coords to match.
        let hex_edges: HashSet<(i32, i32, i32, i32)> = hex_g
            .edge_src
            .iter()
            .zip(&hex_g.edge_dst)
            .filter(|(s, d)| s != d) // exclude self-loops (pre-baked in hex graph)
            .map(|(s, d)| {
                let si = *s as usize;
                let di = *d as usize;
                (hex_g.coords[si * 2], hex_g.coords[si * 2 + 1],
                 hex_g.coords[di * 2], hex_g.coords[di * 2 + 1])
            })
            .collect();

        let axis_edges: HashSet<(i32, i32, i32, i32)> = axis_g
            .edge_src
            .iter()
            .zip(&axis_g.edge_dst)
            .filter(|(s, d)| {
                **s != (axis_g.num_nodes - 1) as i64
                    && **d != (axis_g.num_nodes - 1) as i64
            })
            .map(|(s, d)| {
                let si = *s as usize;
                let di = *d as usize;
                (axis_g.coords[si * 2], axis_g.coords[si * 2 + 1],
                 axis_g.coords[di * 2], axis_g.coords[di * 2 + 1])
            })
            .collect();

        for edge in &hex_edges {
            assert!(
                axis_edges.contains(edge),
                "hex neighbor edge ({},{}) -> ({},{}) missing from axis graph",
                edge.0,
                edge.1,
                edge.2,
                edge.3,
            );
        }
    }

    // AC-1h: src_player edge feature encodes source node's player identity
    #[test]
    fn ac1h_src_player_features() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let dummy = (g.num_nodes - 1) as i64;

        // Build coord -> player map from game state
        let stones = game.placed_stones();
        let stone_player: HashMap<Coord, Player> = stones.iter().cloned().collect();

        let mut found_p1_src = false;
        let mut found_p2_src = false;
        let mut found_empty_src = false;

        let n_edges = g.edge_src.len();
        for e in 0..n_edges {
            let s = g.edge_src[e] as usize;
            let d = g.edge_dst[e] as usize;
            if s as i64 == dummy || d as i64 == dummy {
                continue;
            }
            let src_player = g.edge_attr[e * 5 + 4];
            let sq = g.coords[s * 2];
            let sr = g.coords[s * 2 + 1];

            if g.stone_mask[s] {
                // Source is a stone: src_player should be +1 (P1) or -1 (P2)
                let expected = match stone_player.get(&(sq, sr)) {
                    Some(&Player::P1) => 1.0f32,
                    Some(&Player::P2) => -1.0f32,
                    _ => panic!("stone_mask set but no player found at ({sq},{sr})"),
                };
                assert_eq!(
                    src_player, expected,
                    "edge {e} ({s}->{d}): src stone at ({sq},{sr}) should be {expected}, got {src_player}"
                );
                if expected > 0.0 { found_p1_src = true; }
                else { found_p2_src = true; }
            } else if g.legal_mask[s] {
                // Source is empty: src_player should be 0
                assert_eq!(
                    src_player, 0.0,
                    "edge {e} ({s}->{d}): src empty at ({sq},{sr}) should be 0.0, got {src_player}"
                );
                found_empty_src = true;
            }
        }
        assert!(found_p1_src, "should find at least one edge with P1 source");
        assert!(found_p2_src, "should find at least one edge with P2 source");
        assert!(found_empty_src, "should find at least one edge with empty source");
    }

    // AC-1i: Known 3-stone collinear configuration produces exact expected edge count
    #[test]
    fn ac1i_three_collinear_stones() {
        // Create a game and place 3 stones in a line along axis (1,0)
        // P1 at (0,0) by default. P2 places at (1,0) and (2,0).
        let mut game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 3,
            max_moves: 80,
        });
        // P2 turn: place at (1,0) and (2,0)
        game.apply_move((1, 0)).unwrap();
        game.apply_move((2, 0)).unwrap();

        let g = game_to_axis_graph_raw(&game);
        let dummy = (g.num_nodes - 1) as i64;

        // Count non-dummy edges
        let axis_edge_count = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .filter(|(s, d)| **s != dummy && **d != dummy)
            .count();

        // Pin the exact non-dummy axis edge count as a snapshot test.
        // This count covers all edges (stone-stone, stone-legal, legal-stone,
        // legal-legal) discovered by the walk logic, excluding dummy edges.
        eprintln!("AC-1i: non-dummy axis edge count = {axis_edge_count}");
        assert!(axis_edge_count > 0, "should have non-dummy edges");
        // Snapshot: pin exact count so any walk-logic change is caught.
        assert_eq!(
            axis_edge_count, 574,
            "pinned non-dummy axis edge count"
        );

        // Count edges involving only the 3 stones (indices 0, 1, 2)
        let stone_stone_edges: Vec<_> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .enumerate()
            .filter(|(_, (s, d))| {
                (**s as usize) < 3 && (**d as usize) < 3
            })
            .collect();

        // With bidirectional walks, all 3 pairs are connected:
        // (0,0)↔(1,0), (1,0)↔(2,0), (0,0)↔(2,0) = 3×2 = 6 directed edges.
        assert_eq!(
            stone_stone_edges.len(),
            6,
            "3 collinear stones should produce 6 directed stone-stone edges, got {}",
            stone_stone_edges.len()
        );
    }

    // AC-1k: No duplicate edges
    #[test]
    fn ac1k_no_duplicate_edges() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let edges: Vec<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(s, d)| (*s, *d))
            .collect();
        let unique: HashSet<(i64, i64)> = edges.iter().copied().collect();
        assert_eq!(
            edges.len(),
            unique.len(),
            "found {} duplicate edges",
            edges.len() - unique.len()
        );
    }

    // AC-1l: For every edge pair (i,j) and (j,i), signed distances sum to zero
    #[test]
    fn ac1l_signed_distance_antisymmetric() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let edge_map: HashMap<(i64, i64), f32> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .enumerate()
            .map(|(e, (s, d))| ((*s, *d), g.edge_attr[e * 5 + 3]))
            .collect();

        for (&(s, d), &dist) in &edge_map {
            let rev_dist = edge_map
                .get(&(d, s))
                .expect(&format!("missing reverse edge ({d},{s})"));
            assert!(
                (dist + rev_dist).abs() < 1e-6,
                "signed distances for ({s},{d}) and ({d},{s}) should sum to 0: {dist} + {rev_dist}"
            );
        }
    }

    // AC-1m: Edge count for 50-stone game is in a reasonable range; hex/axis ratio < 15.0
    #[test]
    fn ac1m_edge_count_50_stones() {
        use crate::graph::game_to_graph_raw;

        let mut game = GameState::with_config(GameConfig {
            win_length: 6,
            placement_radius: 6,
            max_moves: 200,
        });
        // Place 49 more stones (P1 already at origin)
        let mut placed = 0;
        while placed < 49 && !game.is_terminal() {
            let moves = game.legal_moves();
            if moves.is_empty() {
                break;
            }
            game.apply_move(moves[0]).unwrap();
            placed += 1;
        }

        let g = game_to_axis_graph_raw(&game);
        let hex_g = game_to_graph_raw(&game);
        let n_edges = g.edge_src.len();
        let n_real = g.num_nodes - 1;
        let dummy_edges = 2 * n_real;
        let axis_edges = n_edges - dummy_edges;
        let hex_edges = hex_g.edge_src.len();

        assert!(
            n_edges >= dummy_edges,
            "should have at least {dummy_edges} edges (dummy), got {n_edges}"
        );
        assert!(
            axis_edges > 0,
            "should have axis edges beyond just dummy edges"
        );

        let ratio = axis_edges as f64 / hex_edges.max(1) as f64;
        eprintln!(
            "AC-1m: 50-stone game: {} nodes, {} axis edges, {} hex edges, ratio = {:.2}",
            g.num_nodes, axis_edges, hex_edges, ratio
        );
        assert!(
            ratio < 15.0,
            "axis/hex edge ratio {ratio:.2} exceeds design threshold 15.0"
        );
    }

    // AC-1n: Two stones at distance win_length-1 connected; at distance win_length not connected
    #[test]
    fn ac1n_boundary_distance() {
        // win_length=4, so window=3
        // P1 at (0,0). Place P2 at (3,0) => distance 3 = win_length-1 => should connect
        // Place P2 at (0,4) => distance 4 = win_length => should NOT connect
        let mut game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 5,
            max_moves: 80,
        });
        // P2 places at (3,0) and (0,4)
        game.apply_move((3, 0)).unwrap();
        game.apply_move((0, 4)).unwrap();

        let g = game_to_axis_graph_raw(&game);

        let mut coord_idx: HashMap<Coord, usize> = HashMap::new();
        for i in 0..g.num_nodes - 1 {
            let q = g.coords[i * 2];
            let r = g.coords[i * 2 + 1];
            coord_idx.insert((q, r), i);
        }

        let idx_origin = *coord_idx.get(&(0, 0)).unwrap();
        let idx_3_0 = *coord_idx.get(&(3, 0)).unwrap();

        let edges: HashSet<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(s, d)| (*s, *d))
            .collect();

        // Distance 3 = win_length-1: should be connected
        assert!(
            edges.contains(&(idx_origin as i64, idx_3_0 as i64)),
            "stones at distance win_length-1 should be connected"
        );

        // Distance 4 = win_length: should NOT be connected
        let idx_0_4 = *coord_idx.get(&(0, 4)).unwrap();
        assert!(
            !edges.contains(&(idx_origin as i64, idx_0_4 as i64)),
            "stones at distance win_length should NOT be connected"
        );
    }

    // --- Walk symmetry tests ---
    // Verify that the bidirectional walk produces symmetric edges:
    // for every edge (A→B, axis, dist, sc), the reverse (B→A, axis, -dist, sc) exists.

    /// Helper: build (src, dst, axis_idx) -> (signed_dist, dst_player) map,
    /// excluding dummy edges.
    fn edge_attr_map(g: &AxisGraphData) -> HashMap<(i64, i64, u8), (f32, f32)> {
        let dummy = (g.num_nodes - 1) as i64;
        let mut map = HashMap::new();
        for e in 0..g.edge_src.len() {
            let s = g.edge_src[e];
            let d = g.edge_dst[e];
            if s == dummy || d == dummy { continue; }
            let ax = axis_idx_of(&g.edge_attr, e);
            let dist = g.edge_attr[e * 5 + 3];
            let dp = g.edge_attr[e * 5 + 4];
            map.insert((s, d, ax), (dist, dp));
        }
        map
    }

    /// Helper: assert every edge has a reverse with negated distance.
    /// dst_player values are NOT expected to match (they encode opposite
    /// endpoints), so we only check existence and distance antisymmetry.
    fn assert_edges_symmetric(g: &AxisGraphData, label: &str) {
        let map = edge_attr_map(g);
        for (&(s, d, ax), &(dist, _dp)) in &map {
            let rev = map.get(&(d, s, ax));
            assert!(
                rev.is_some(),
                "{label}: edge ({s},{d}) axis={ax} dist={dist} has no reverse"
            );
            let (rev_dist, _rev_dp) = rev.unwrap();
            assert!(
                (rev_dist + dist).abs() < 0.01,
                "{label}: edge ({s},{d}) axis={ax} dist={dist}, reverse dist={rev_dist} (expected {})",
                -dist
            );
        }
    }

    // AC-S1: Symmetry on all existing test positions
    #[test]
    fn ac_symmetry_small_game() {
        assert_edges_symmetric(&game_to_axis_graph_raw(&small_game()), "small_game");
    }

    #[test]
    fn ac_symmetry_mid_game() {
        assert_edges_symmetric(&game_to_axis_graph_raw(&mid_game()), "mid_game");
    }

    // AC-S2: Symmetry with stones on all 3 axes, both colors
    #[test]
    fn ac_symmetry_multi_axis() {
        // Place stones along all 3 axes from origin, alternating colors.
        // P1 at (0,0). Then P2,P1,P2,P1... on axes (1,0), (0,1), (1,-1).
        let mut game = GameState::with_config(GameConfig {
            win_length: 5,
            placement_radius: 4,
            max_moves: 80,
        });
        let moves = [
            (2, 0),  // P2 on axis (1,0)
            (0, 2),  // P1 on axis (0,1)
            (-2, 2), // P2 on axis (1,-1)
            (1, 0),  // P1 on axis (1,0)
            (0, -2), // P2 on axis (0,1)
            (2, -2), // P1 on axis (1,-1)
        ];
        for &m in &moves {
            game.apply_move(m).unwrap();
        }
        assert_edges_symmetric(&game_to_axis_graph_raw(&game), "multi_axis");
    }

    // AC-S3: Symmetry for a line of same-color stones with empties on both ends
    #[test]
    fn ac_symmetry_line_with_extensions() {
        // P1 at (0,0), then P2 far away, then P1 extends the line.
        // Result: P1 line at (0,0),(1,0),(2,0) with empties at (-1,0) and (3,0).
        let mut game = GameState::with_config(GameConfig {
            win_length: 5,
            placement_radius: 4,
            max_moves: 80,
        });
        game.apply_move((0, 3)).unwrap();  // P2 far away
        game.apply_move((0, -3)).unwrap(); // P2 far away
        game.apply_move((1, 0)).unwrap();  // P1
        game.apply_move((2, 0)).unwrap();  // P2 — now P2 is adjacent to P1 line

        let g = game_to_axis_graph_raw(&game);
        assert_edges_symmetric(&g, "line_with_extensions");
    }

    // AC-S4: Symmetry for empty nodes flanking opponent stones
    #[test]
    fn ac_symmetry_empty_flanking_opponent() {
        // Position: P1 at (0,0), P2 at (1,0) and (2,0).
        // Empty at (-1,0) and (3,0) should have symmetric access to the stones.
        let mut game = GameState::with_config(GameConfig {
            win_length: 5,
            placement_radius: 4,
            max_moves: 80,
        });
        game.apply_move((1, 0)).unwrap(); // P2
        game.apply_move((2, 0)).unwrap(); // P2

        let g = game_to_axis_graph_raw(&game);
        assert_edges_symmetric(&g, "empty_flanking_opponent");

        // Also verify both empty flanks see at least one stone
        let mut coord_idx: HashMap<Coord, usize> = HashMap::new();
        for i in 0..g.num_nodes - 1 {
            coord_idx.insert((g.coords[i * 2], g.coords[i * 2 + 1]), i);
        }
        let map = edge_attr_map(&g);

        // (-1,0) should have an edge to (0,0) on axis (1,0)
        if let Some(&idx_neg1) = coord_idx.get(&(-1, 0)) {
            if let Some(&idx_0) = coord_idx.get(&(0, 0)) {
                assert!(
                    map.contains_key(&(idx_neg1 as i64, idx_0 as i64, 0)),
                    "(-1,0) should see (0,0) on axis (1,0)"
                );
            }
        }
        // (3,0) should have an edge to (2,0) on axis (1,0)
        if let Some(&idx_3) = coord_idx.get(&(3, 0)) {
            if let Some(&idx_2) = coord_idx.get(&(2, 0)) {
                assert!(
                    map.contains_key(&(idx_3 as i64, idx_2 as i64, 0)),
                    "(3,0) should see (2,0) on axis (1,0)"
                );
            }
        }
    }

    // AC-S5: Symmetry preserved under D6 augmentation
    #[test]
    fn ac_symmetry_augmented() {
        let game = mid_game();
        for (i, (g, _perm)) in augment_axis_graph(&game).into_iter().enumerate() {
            assert_edges_symmetric(&g, &format!("augment[{i}]"));
        }
    }

    // AC-1j: game_to_axis_graph_batch matches individual game_to_axis_graph_raw
    #[test]
    fn ac1j_batch_matches_individual() {
        let games = vec![small_game(), mid_game()];
        let batch = game_to_axis_graph_batch(&games);
        assert_eq!(batch.len(), games.len());

        for (i, game) in games.iter().enumerate() {
            let individual = game_to_axis_graph_raw(game);
            assert_eq!(batch[i].num_nodes, individual.num_nodes);
            assert_eq!(batch[i].features, individual.features);
            assert_eq!(batch[i].edge_src.len(), individual.edge_src.len());

            // Build (src, dst) -> [f32; 5] maps for both and compare
            let batch_attr_map: HashMap<(i64, i64), Vec<f32>> = batch[i]
                .edge_src
                .iter()
                .zip(&batch[i].edge_dst)
                .enumerate()
                .map(|(e, (s, d))| {
                    let attr = batch[i].edge_attr[e * 5..(e + 1) * 5].to_vec();
                    ((*s, *d), attr)
                })
                .collect();
            let ind_attr_map: HashMap<(i64, i64), Vec<f32>> = individual
                .edge_src
                .iter()
                .zip(&individual.edge_dst)
                .enumerate()
                .map(|(e, (s, d))| {
                    let attr = individual.edge_attr[e * 5..(e + 1) * 5].to_vec();
                    ((*s, *d), attr)
                })
                .collect();

            assert_eq!(
                batch_attr_map.len(),
                ind_attr_map.len(),
                "game {i}: edge count mismatch"
            );
            for (key, batch_attr) in &batch_attr_map {
                let ind_attr = ind_attr_map
                    .get(key)
                    .unwrap_or_else(|| panic!("game {i}: missing edge {:?} in individual", key));
                assert_eq!(
                    batch_attr, ind_attr,
                    "game {i}: edge_attr mismatch for edge {:?}",
                    key
                );
            }
        }
    }

    // AC: Verify actual node feature values for specific nodes.
    #[test]
    fn ac_node_features() {
        let game = small_game(); // P1 at origin, P2 to move
        let g = game_to_axis_graph_raw(&game);

        // Node 0 is P1 stone (stones sorted by coord, (0,0) is first/only stone)
        let base0 = 0 * 8;
        // P1 one-hot: [1, 0, 0]
        assert_eq!(g.features[base0], 1.0, "node 0: P1 one-hot[0]");
        assert_eq!(g.features[base0 + 1], 0.0, "node 0: P1 one-hot[1]");
        assert_eq!(g.features[base0 + 2], 0.0, "node 0: P1 one-hot[2]");
        // player_feat: P2 to move => -1.0
        assert_eq!(g.features[base0 + 3], -1.0, "node 0: player_feat");
        // moves_feat: moves_remaining / 2.0 = 2/2.0 = 1.0
        assert_eq!(g.features[base0 + 4], 1.0, "node 0: moves_feat");

        // Node 1 is first legal move (empty)
        let n_stones = game.placed_stones().len();
        assert_eq!(n_stones, 1);
        let base1 = 1 * 8; // first legal move node
        // Empty one-hot: [0, 0, 1]
        assert_eq!(g.features[base1 + 2], 1.0, "node 1: empty one-hot");
        // inv_distance > 0 for legal moves near a stone
        assert!(g.features[base1 + 7] > 0.0, "node 1: inv_distance > 0");

        // Dummy node (last)
        let dummy_idx = g.num_nodes - 1;
        let dummy_base = dummy_idx * 8;
        // Dummy has no type one-hot
        assert_eq!(g.features[dummy_base], 0.0, "dummy: one-hot[0]");
        assert_eq!(g.features[dummy_base + 1], 0.0, "dummy: one-hot[1]");
        assert_eq!(g.features[dummy_base + 2], 0.0, "dummy: one-hot[2]");
        // Dummy still has player_feat and moves_feat
        assert_eq!(g.features[dummy_base + 3], -1.0, "dummy: player_feat");
        assert_eq!(g.features[dummy_base + 4], 1.0, "dummy: moves_feat");
    }

    // AC: Verify full-window connectivity (no walk-stopping).
    #[test]
    fn ac_full_window_edges() {
        // P1 at (0,0), P2 at (1,0) and (2,0). win_length=4 => window=3.
        let mut game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 3,
            max_moves: 80,
        });
        // P2 places at (1,0) and (2,0)
        game.apply_move((1, 0)).unwrap();
        game.apply_move((2, 0)).unwrap();

        let g = game_to_axis_graph_raw(&game);

        // Build coord -> index map
        let mut coord_idx: HashMap<Coord, usize> = HashMap::new();
        for i in 0..g.num_nodes - 1 {
            coord_idx.insert((g.coords[i * 2], g.coords[i * 2 + 1]), i);
        }

        let idx_00 = *coord_idx.get(&(0, 0)).expect("(0,0) should exist");
        let idx_10 = *coord_idx.get(&(1, 0)).expect("(1,0) should exist");
        let idx_20 = *coord_idx.get(&(2, 0)).expect("(2,0) should exist");

        let edges: HashSet<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(s, d)| (*s, *d))
            .collect();

        // With bidirectional walks: P1 at (0,0) walks positive to P2 at (1,0)
        // and stops. P2 at (2,0) walks NEGATIVE through P2 at (1,0) (same color)
        // and reaches P1 at (0,0) (opponent, stops but edge created first).
        // So all pairs are connected.
        assert!(
            edges.contains(&(idx_00 as i64, idx_10 as i64)),
            "(0,0)->(1,0) should exist"
        );
        assert!(
            edges.contains(&(idx_10 as i64, idx_20 as i64)),
            "(1,0)->(2,0) should exist"
        );
        assert!(
            edges.contains(&(idx_00 as i64, idx_20 as i64)),
            "(0,0)->(2,0) should exist (created by (2,0) walking negative)"
        );
    }

    // --- augment_axis_graph tests ---

    #[test]
    fn augment_axis_produces_11_graphs() {
        let game = small_game();
        let results = augment_axis_graph(&game);
        assert_eq!(results.len(), 11);
    }

    #[test]
    fn augmented_axis_same_node_count() {
        let game = mid_game();
        let original = game_to_axis_graph_raw(&game);
        for (g, _perm) in augment_axis_graph(&game) {
            assert_eq!(g.num_nodes, original.num_nodes);
            assert_eq!(g.features.len(), original.features.len());
            assert_eq!(g.legal_mask.len(), original.legal_mask.len());
            assert_eq!(g.stone_mask.len(), original.stone_mask.len());
        }
    }

    #[test]
    fn augmented_axis_same_edge_count() {
        let game = mid_game();
        let original = game_to_axis_graph_raw(&game);
        for (g, _perm) in augment_axis_graph(&game) {
            assert_eq!(
                g.edge_src.len(),
                original.edge_src.len(),
                "augmented graph should have same number of edges"
            );
            assert_eq!(g.edge_attr.len(), original.edge_attr.len());
        }
    }

    #[test]
    fn augmented_axis_permutation_valid() {
        let game = mid_game();
        let legal = game.legal_moves();
        for (_g, perm) in augment_axis_graph(&game) {
            assert_eq!(perm.len(), legal.len());
            let mut sorted = perm.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), legal.len(), "permutation should be a bijection");
        }
    }

    #[test]
    fn augmented_axis_edge_attr_valid() {
        // Every augmented graph should have valid edge_attr:
        // axis one-hots sum to 0 (dummy) or 1 (axis), distance within window
        let game = mid_game();
        let wl = game.config().win_length as f32;
        for (g, _perm) in augment_axis_graph(&game) {
            let dummy = (g.num_nodes - 1) as i64;
            for e in 0..g.edge_src.len() {
                let oh_sum = g.edge_attr[e * 5] + g.edge_attr[e * 5 + 1] + g.edge_attr[e * 5 + 2];
                let is_dummy = g.edge_src[e] == dummy || g.edge_dst[e] == dummy;
                if is_dummy {
                    assert_eq!(oh_sum, 0.0, "augmented dummy edge {e}: axis oh should be 0");
                } else {
                    assert_eq!(oh_sum, 1.0, "augmented axis edge {e}: axis oh should be 1");
                    let dist = g.edge_attr[e * 5 + 3];
                    assert!(
                        dist.abs() <= wl,
                        "augmented edge {e}: |dist| = {} > {}",
                        dist.abs(),
                        wl
                    );
                }
            }
        }
    }

    #[test]
    fn augmented_axis_all_distinct() {
        // All 11 augmented graphs should have distinct feature vectors
        // (coords may collide for symmetric positions, but features differ
        // because stone identities are transformed too)
        let game = mid_game();
        let mut feature_sets: Vec<Vec<f32>> = Vec::new();
        for (g, _perm) in augment_axis_graph(&game) {
            feature_sets.push(g.features.clone());
        }
        // At least some graphs should differ — not all 11 can be identical
        // for a game with 2 distinct players. Use pairwise comparison.
        let mut all_same = true;
        for i in 1..feature_sets.len() {
            if feature_sets[i] != feature_sets[0] {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "all augmented graphs are identical — expected variety");
    }

    #[test]
    fn augmented_axis_preserves_stone_legal_counts() {
        let game = mid_game();
        let original = game_to_axis_graph_raw(&game);
        let orig_stones: usize = original.stone_mask.iter().filter(|&&b| b).count();
        let orig_legal: usize = original.legal_mask.iter().filter(|&&b| b).count();

        for (g, _perm) in augment_axis_graph(&game) {
            let n_stones: usize = g.stone_mask.iter().filter(|&&b| b).count();
            let n_legal: usize = g.legal_mask.iter().filter(|&&b| b).count();
            assert_eq!(n_stones, orig_stones);
            assert_eq!(n_legal, orig_legal);
        }
    }

    /// D6-equivariance of the augment helper across ALL 4 node-feature
    /// encodings: building the axis graph from transformed stones/legal (the
    /// augment path) must match building it from a `GameState` reconstructed on
    /// the transformed board (the honest path), for the same flags.
    #[test]
    fn augment_axis_graph_single_equivariant_all_encodings() {
        use hexo_engine::symmetry::D6_TRANSFORMS;

        let game = mid_game();
        let cfg = *game.config();
        let stones = game.placed_stones();
        let current_player = game.current_player().unwrap();
        let moves_remaining = game.moves_remaining_this_turn();

        let prune = false;
        for (threat, relative) in [(false, false), (false, true), (true, false), (true, true)] {
            // Every non-identity D6 transform must be equivariant.
            for k in 1..D6_TRANSFORMS.len() {
                let transform = D6_TRANSFORMS[k];
                let (aug_graph, perm) =
                    augment_axis_graph_single(&game, transform, prune, threat, relative);

                // Honest path: reconstruct the transformed board and build from it.
                let t_stones: Vec<(Coord, Player)> =
                    stones.iter().map(|&(c, p)| (transform(c), p)).collect();
                let t_game =
                    GameState::from_state(&t_stones, current_player, moves_remaining, cfg);
                let honest = game_to_axis_graph_raw_opts(&t_game, prune, threat, relative);

                let ctx = format!("threat={threat} relative={relative} transform={k}");
                assert_eq!(aug_graph.features, honest.features, "features mismatch ({ctx})");
                assert_eq!(aug_graph.edge_src, honest.edge_src, "edge_src mismatch ({ctx})");
                assert_eq!(aug_graph.edge_dst, honest.edge_dst, "edge_dst mismatch ({ctx})");
                assert_eq!(aug_graph.edge_attr, honest.edge_attr, "edge_attr mismatch ({ctx})");
                assert_eq!(aug_graph.legal_mask, honest.legal_mask, "legal_mask mismatch ({ctx})");
                assert_eq!(aug_graph.stone_mask, honest.stone_mask, "stone_mask mismatch ({ctx})");
                assert_eq!(aug_graph.coords, honest.coords, "coords mismatch ({ctx})");
                assert_eq!(aug_graph.num_nodes, honest.num_nodes, "num_nodes mismatch ({ctx})");

                // Expected feature dim per encoding.
                let base = if relative { 7 } else { 8 };
                let fdim = base + if threat { 4 } else { 0 };
                assert_eq!(
                    aug_graph.features.len(),
                    aug_graph.num_nodes * fdim,
                    "feature dim mismatch ({ctx})"
                );

                // Permutation maps augmented legal index -> original legal index,
                // and must be a permutation of 0..n_legal.
                let n_legal = perm.len();
                let mut seen = vec![false; n_legal];
                for &orig in &perm {
                    assert!(orig < n_legal, "perm out of range ({ctx})");
                    assert!(!seen[orig], "perm not injective ({ctx})");
                    seen[orig] = true;
                }
            }
        }
    }

    /// The batched-binding path collates `augment_axis_graph_single` outputs via
    /// `collate_axis_graphs`. For a single state, the collated `x`/`legal_mask`
    /// must equal the augment helper's graph bytes (no reordering / loss). This
    /// mirrors what `py_augment_axis_states_to_batch_bytes` emits for one state.
    #[test]
    fn augment_axis_single_state_collate_matches_helper() {
        use crate::batch_tensors::collate_axis_graphs;
        use hexo_engine::symmetry::D6_TRANSFORMS;

        let game = mid_game();
        let k = 5;
        for (threat, relative) in [(false, false), (false, true), (true, false), (true, true)] {
            let (graph, _perm) =
                augment_axis_graph_single(&game, D6_TRANSFORMS[k], false, threat, relative);
            let bt = collate_axis_graphs(std::slice::from_ref(&graph));
            assert_eq!(bt.features, graph.features, "x mismatch (threat={threat} relative={relative})");
            assert_eq!(bt.legal_mask, graph.legal_mask, "legal_mask mismatch (threat={threat} relative={relative})");
            assert_eq!(bt.num_graphs, 1);
        }
    }
}
