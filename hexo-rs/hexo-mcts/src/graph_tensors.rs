//! `GraphTensors` — raw graph data for one game state, without any `tch` dependency.
//!
//! This module is intentionally outside the `torch` feature gate so that
//! future consumers (e.g. `SubprocessModel`) can use it without linking libtorch.

use hexo_engine::types::{Coord};
use hexo_engine::GameState;

use crate::axis_graph::game_to_axis_graph_raw_opts;
use crate::graph::game_to_graph_raw;

/// Graph type for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphType {
    Hex,
    Axis,
}

/// Raw graph tensors for one game state.
pub struct GraphTensors {
    pub features: Vec<f32>,    // N*fdim flat (7/8 base, +4 with threat features)
    pub edge_src: Vec<i64>,
    pub edge_dst: Vec<i64>,
    pub edge_attr: Option<Vec<f32>>,  // E*5 flat, only for axis graphs
    pub legal_mask: Vec<bool>,
    pub stone_mask: Vec<bool>,
    pub legal_coords: Vec<Coord>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl GraphTensors {
    /// Create from a hex graph reference (clones data).
    pub fn from_hex(g: &crate::graph::GraphData) -> Self {
        let legal_coords: Vec<Coord> = g.legal_mask.iter()
            .enumerate()
            .filter(|&(_, &is_legal)| is_legal)
            .map(|(i, _)| (g.coords[i * 2], g.coords[i * 2 + 1]))
            .collect();
        let num_edges = g.edge_src.len();
        GraphTensors {
            features: g.features.clone(),
            edge_src: g.edge_src.clone(),
            edge_dst: g.edge_dst.clone(),
            edge_attr: None,
            legal_mask: g.legal_mask.clone(),
            stone_mask: g.stone_mask.clone(),
            legal_coords,
            num_nodes: g.num_nodes,
            num_edges,
        }
    }

    /// Create from an axis graph reference (clones data).
    pub fn from_axis(g: &crate::axis_graph::AxisGraphData) -> Self {
        let legal_coords: Vec<Coord> = g.legal_mask.iter()
            .enumerate()
            .filter(|&(_, &is_legal)| is_legal)
            .map(|(i, _)| (g.coords[i * 2], g.coords[i * 2 + 1]))
            .collect();
        let num_edges = g.edge_src.len();
        GraphTensors {
            features: g.features.clone(),
            edge_src: g.edge_src.clone(),
            edge_dst: g.edge_dst.clone(),
            edge_attr: Some(g.edge_attr.clone()),
            legal_mask: g.legal_mask.clone(),
            stone_mask: g.stone_mask.clone(),
            legal_coords,
            num_nodes: g.num_nodes,
            num_edges,
        }
    }
}

impl From<crate::graph::GraphData> for GraphTensors {
    fn from(g: crate::graph::GraphData) -> Self {
        Self::from_hex(&g)
    }
}

impl From<crate::axis_graph::AxisGraphData> for GraphTensors {
    fn from(g: crate::axis_graph::AxisGraphData) -> Self {
        Self::from_axis(&g)
    }
}

/// Build graph tensors from a game state (mirrors graph.rs::build_graph).
pub fn build_graph_tensors(game: &GameState) -> GraphTensors {
    // Call game_to_graph_raw to get raw graph data, then convert to GraphTensors.
    let graph_data = game_to_graph_raw(game);
    GraphTensors::from_hex(&graph_data)
}

/// Build graph tensors from a game state using axis-window graph construction.
pub fn build_axis_graph_tensors(game: &GameState) -> GraphTensors {
    build_axis_graph_tensors_opts(game, false)
}

/// Build axis graph tensors with optional empty-edge pruning.
///
/// Only reachable via `TorchModel::build_graphs`/`evaluate` (the legacy
/// `native_self_play` PyO3 path); the self_play binary builds its graphs on
/// game threads with the threaded `--threat-features` / `--relative-stones`
/// flags instead, so this path stays 8-dim absolute (threat_features = false,
/// relative_stones = false).
pub fn build_axis_graph_tensors_opts(game: &GameState, prune_empty_edges: bool) -> GraphTensors {
    let axis_data = game_to_axis_graph_raw_opts(game, prune_empty_edges, false, false);

    // Extract legal_coords from coords + legal_mask
    let legal_coords: Vec<Coord> = axis_data.legal_mask
        .iter()
        .enumerate()
        .filter(|&(_, &is_legal)| is_legal)
        .map(|(i, _)| (axis_data.coords[i * 2], axis_data.coords[i * 2 + 1]))
        .collect();

    let num_edges = axis_data.edge_src.len();

    GraphTensors {
        features: axis_data.features,
        edge_src: axis_data.edge_src,
        edge_dst: axis_data.edge_dst,
        edge_attr: Some(axis_data.edge_attr),
        legal_mask: axis_data.legal_mask,
        stone_mask: axis_data.stone_mask,
        legal_coords,
        num_nodes: axis_data.num_nodes,
        num_edges,
    }
}
