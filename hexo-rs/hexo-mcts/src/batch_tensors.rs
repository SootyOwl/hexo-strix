//! Batch tensor construction: collate multiple GraphData into flat tensors
//! ready for direct model consumption, bypassing PyG Batch.from_data_list().

use crate::axis_graph::AxisGraphData;
use crate::graph::GraphData;

pub struct BatchTensors {
    pub features: Vec<f32>,       // total_nodes * 8, row-major
    pub edge_index_src: Vec<i64>, // total_edges (with node offsets applied)
    pub edge_index_dst: Vec<i64>,
    pub legal_mask: Vec<bool>,    // total_nodes (keep for backward compat)
    pub stone_mask: Vec<bool>,    // total_nodes (keep for backward compat)
    pub batch: Vec<i64>,          // total_nodes, each = graph index
    pub num_graphs: usize,
    pub legal_counts: Vec<usize>, // per-graph count of legal nodes
    // precomputed index tensors (eliminates GPU nonzero calls)
    pub legal_idx: Vec<i64>,      // global indices where legal_mask is true
    pub stone_idx: Vec<i64>,      // global indices where stone_mask is true
    pub stone_batch: Vec<i64>,    // graph index per stone node
}

pub fn collate_graphs(graphs: &[GraphData]) -> BatchTensors {
    let num_graphs = graphs.len();
    let total_nodes: usize = graphs.iter().map(|g| g.num_nodes).sum();
    let total_edges: usize = graphs.iter().map(|g| g.edge_src.len()).sum();

    let mut features = Vec::with_capacity(total_nodes * 8);
    let mut edge_index_src = Vec::with_capacity(total_edges);
    let mut edge_index_dst = Vec::with_capacity(total_edges);
    let mut legal_mask = Vec::with_capacity(total_nodes);
    let mut stone_mask = Vec::with_capacity(total_nodes);
    let mut batch = Vec::with_capacity(total_nodes);
    let mut legal_counts = Vec::with_capacity(num_graphs);

    let mut legal_idx = Vec::new();
    let mut stone_idx = Vec::new();
    let mut stone_batch = Vec::new();
    let mut node_offset: i64 = 0;

    for (graph_idx, g) in graphs.iter().enumerate() {
        features.extend_from_slice(&g.features);
        legal_mask.extend_from_slice(&g.legal_mask);
        stone_mask.extend_from_slice(&g.stone_mask);

        for &src in &g.edge_src {
            edge_index_src.push(src + node_offset);
        }
        for &dst in &g.edge_dst {
            edge_index_dst.push(dst + node_offset);
        }

        for _ in 0..g.num_nodes {
            batch.push(graph_idx as i64);
        }

        legal_counts.push(g.legal_mask.iter().filter(|&&m| m).count());

        for (local_i, (&is_legal, &is_stone)) in g.legal_mask.iter().zip(&g.stone_mask).enumerate() {
            let global_i = node_offset + local_i as i64;
            if is_legal {
                legal_idx.push(global_i);
            }
            if is_stone {
                stone_idx.push(global_i);
                stone_batch.push(graph_idx as i64);
            }
        }

        node_offset += g.num_nodes as i64;
    }

    BatchTensors {
        features,
        edge_index_src,
        edge_index_dst,
        legal_mask,
        stone_mask,
        batch,
        num_graphs,
        legal_counts,
        legal_idx,
        stone_idx,
        stone_batch,
    }
}

pub struct AxisBatchTensors {
    pub features: Vec<f32>,       // total_nodes * n_feat, row-major
    pub edge_index_src: Vec<i64>, // total_edges (with node offsets applied)
    pub edge_index_dst: Vec<i64>,
    pub edge_attr: Vec<f32>,      // total_edges * 5, row-major
    pub legal_mask: Vec<bool>,    // total_nodes
    pub batch: Vec<i64>,          // total_nodes, each = graph index
    pub coords: Vec<i32>,         // total_nodes * 2, row-major (q, r)
    pub num_graphs: usize,
    pub legal_counts: Vec<i64>,   // per-graph count of legal nodes
    // precomputed index tensors (eliminates GPU nonzero calls)
    pub legal_idx: Vec<i64>,
    pub stone_idx: Vec<i64>,
    pub stone_batch: Vec<i64>,
}

pub fn collate_axis_graphs(graphs: &[AxisGraphData]) -> AxisBatchTensors {
    let num_graphs = graphs.len();
    let total_nodes: usize = graphs.iter().map(|g| g.num_nodes).sum();
    let total_edges: usize = graphs.iter().map(|g| g.edge_src.len()).sum();
    let n_feat = if total_nodes > 0 {
        graphs[0].features.len() / graphs[0].num_nodes
    } else {
        0
    };

    let mut bt = AxisBatchTensors {
        features: Vec::with_capacity(total_nodes * n_feat),
        edge_index_src: Vec::with_capacity(total_edges),
        edge_index_dst: Vec::with_capacity(total_edges),
        edge_attr: Vec::with_capacity(total_edges * 5),
        legal_mask: Vec::with_capacity(total_nodes),
        batch: Vec::with_capacity(total_nodes),
        coords: Vec::with_capacity(total_nodes * 2),
        num_graphs,
        legal_counts: Vec::with_capacity(num_graphs),
        legal_idx: Vec::new(),
        stone_idx: Vec::new(),
        stone_batch: Vec::new(),
    };

    let mut node_offset: i64 = 0;
    for (graph_idx, g) in graphs.iter().enumerate() {
        bt.features.extend_from_slice(&g.features);
        bt.legal_mask.extend_from_slice(&g.legal_mask);
        bt.coords.extend_from_slice(&g.coords);
        bt.edge_attr.extend_from_slice(&g.edge_attr);
        bt.edge_index_src.extend(g.edge_src.iter().map(|&s| s + node_offset));
        bt.edge_index_dst.extend(g.edge_dst.iter().map(|&d| d + node_offset));
        bt.batch.extend(std::iter::repeat(graph_idx as i64).take(g.num_nodes));
        bt.legal_counts.push(g.legal_mask.iter().filter(|&&m| m).count() as i64);

        for (local_i, (&is_legal, &is_stone)) in g.legal_mask.iter().zip(&g.stone_mask).enumerate() {
            let global_i = node_offset + local_i as i64;
            if is_legal {
                bt.legal_idx.push(global_i);
            }
            if is_stone {
                bt.stone_idx.push(global_i);
                bt.stone_batch.push(graph_idx as i64);
            }
        }

        node_offset += g.num_nodes as i64;
    }

    bt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::game_to_graph_raw;
    use hexo_engine::GameState;

    #[test]
    fn collate_two_graphs() {
        let g1 = game_to_graph_raw(&GameState::new());
        let g2 = game_to_graph_raw(&GameState::new());
        let n1 = g1.num_nodes;
        let n2 = g2.num_nodes;

        let bt = collate_graphs(&[g1, g2]);

        assert_eq!(bt.num_graphs, 2);
        assert_eq!(bt.batch.len(), n1 + n2);
        assert_eq!(bt.features.len(), (n1 + n2) * 8);
        assert_eq!(bt.legal_counts.len(), 2);
        assert!(bt.batch[..n1].iter().all(|&b| b == 0));
        assert!(bt.batch[n1..].iter().all(|&b| b == 1));
        assert!(bt.edge_index_src.iter().any(|&s| s >= n1 as i64));

        // legal_idx should index into legal positions
        assert_eq!(bt.legal_idx.len(), bt.legal_mask.iter().filter(|&&b| b).count());
        // stone_idx should index into stone positions
        assert_eq!(bt.stone_idx.len(), bt.stone_mask.iter().filter(|&&b| b).count());
        // stone_batch length matches stone_idx
        assert_eq!(bt.stone_batch.len(), bt.stone_idx.len());
        // all legal_idx values should be in range
        assert!(bt.legal_idx.iter().all(|&i| (i as usize) < bt.batch.len()));
        // all stone_idx values should be in range
        assert!(bt.stone_idx.iter().all(|&i| (i as usize) < bt.batch.len()));
    }

    #[test]
    fn collate_two_axis_graphs() {
        use crate::axis_graph::game_to_axis_graph_raw;

        let g1 = game_to_axis_graph_raw(&GameState::new());
        let g2 = game_to_axis_graph_raw(&GameState::new());
        let n1 = g1.num_nodes;
        let n2 = g2.num_nodes;
        let e1 = g1.edge_src.len();
        let n_feat = g1.features.len() / n1;

        let bt = collate_axis_graphs(&[g1, g2]);

        assert_eq!(bt.num_graphs, 2);
        assert_eq!(bt.batch.len(), n1 + n2);
        assert_eq!(bt.features.len(), (n1 + n2) * n_feat);
        assert_eq!(bt.coords.len(), (n1 + n2) * 2);
        assert_eq!(bt.edge_attr.len(), bt.edge_index_src.len() * 5);
        assert!(bt.batch[..n1].iter().all(|&b| b == 0));
        assert!(bt.batch[n1..].iter().all(|&b| b == 1));
        // second graph's edges are offset by n1
        assert!(bt.edge_index_src[e1..].iter().all(|&s| s >= n1 as i64));
        assert_eq!(
            bt.legal_idx.len(),
            bt.legal_mask.iter().filter(|&&b| b).count()
        );
        assert_eq!(bt.stone_idx.len(), bt.stone_batch.len());
        assert_eq!(
            bt.legal_counts.iter().sum::<i64>() as usize,
            bt.legal_idx.len()
        );
    }
}
