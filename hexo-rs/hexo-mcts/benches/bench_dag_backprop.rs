//! Phase 0 DAG-MCTS backprop calibration microbenchmark.
//!
//! Measures the per-sim CPU cost delta between:
//!   1. Tree MCTS backprop: O(1) running-average update at each node on the path.
//!   2. DAG MCGS backprop: idempotent recursive Q recompute, where each touched
//!      node re-derives its Q from its children via FxHashMap lookups.
//!
//! Caveats:
//!   - Mock tree-shaped DAG (no actual transposition sharing). Real DAGs will
//!     have hotter cache lines for shared children, so this is an upper-ish
//!     bound on cache cost / lower-ish bound on dedup benefit.
//!   - Cache-warm: the same path is replayed many times by criterion. Real
//!     production traversals are more scattered.
//!   - Single-threaded, no contention, default system allocator.
//!   - Sign handling is omitted (constant +1) — we are isolating the
//!     hashmap-lookup + arithmetic cost, not player-flip logic.
//!
//! See docs/research/2026-04-07-dag-dedup-spike.md ("Backprop calibration").

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rustc_hash::FxHashMap;
use std::hint::black_box;

#[derive(Clone)]
struct MockDagEdge {
    child_hash: u128,
    edge_visits: u32,
}

#[derive(Clone)]
struct MockDagNode {
    u: f64,
    q: f64,
    children: Vec<MockDagEdge>,
}

type DagMap = FxHashMap<u128, Box<MockDagNode>>;

/// Build a balanced tree-shaped mock DAG with the given branching factor and
/// depth. Returns the populated map plus the root hash and a single random
/// root-to-leaf path of node hashes (root first, leaf last).
fn build_mock_dag(branching: usize, depth: usize, seed: u64) -> (DagMap, u128, Vec<u128>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut map: DagMap = FxHashMap::default();
    let mut next_hash: u128 = 1;

    // Recursively build, returning the hash assigned to the new node.
    fn build(
        map: &mut DagMap,
        rng: &mut StdRng,
        next_hash: &mut u128,
        branching: usize,
        depth_remaining: usize,
    ) -> u128 {
        let my_hash = *next_hash;
        *next_hash += 1;

        let u: f64 = rng.random_range(-1.0..1.0);

        let children = if depth_remaining == 0 {
            Vec::new()
        } else {
            let mut v = Vec::with_capacity(branching);
            for _ in 0..branching {
                let child_hash = build(map, rng, next_hash, branching, depth_remaining - 1);
                v.push(MockDagEdge {
                    child_hash,
                    edge_visits: 1,
                });
            }
            v
        };

        let node = MockDagNode { u, q: u, children };
        map.insert(my_hash, Box::new(node));
        my_hash
    }

    let root_hash = build(&mut map, &mut rng, &mut next_hash, branching, depth);

    // Pick a deterministic root-to-leaf path (random child at each level).
    let mut path = Vec::with_capacity(depth + 1);
    let mut cur = root_hash;
    path.push(cur);
    for _ in 0..depth {
        let node = map.get(&cur).unwrap();
        if node.children.is_empty() {
            break;
        }
        let idx = rng.random_range(0..node.children.len());
        cur = node.children[idx].child_hash;
        path.push(cur);
    }

    (map, root_hash, path)
}

/// Tree-MCTS-style backprop: walk the path leaf-to-root, doing an O(1)
/// running-average update at each node. Mirrors the per-edge increment +
/// value_sum bookkeeping in `mcts/backup.rs` and `mcts/node.rs`. Sign handling
/// is omitted to keep the comparison apples-to-apples with the DAG variant.
fn tree_backprop(map: &mut DagMap, path: &[u128], leaf_value: f64) {
    let v = leaf_value;
    for &h in path.iter().rev() {
        let node = map.get_mut(&h).unwrap();
        // Treat node.q as the running mean and use a synthetic visit count
        // derived from the sum of edge_visits (matches the "node visit = 1 +
        // sum(edge visits)" convention used in the DAG variant below, so the
        // arithmetic cost is comparable).
        let n: u32 = 1 + node.children.iter().map(|e| e.edge_visits).sum::<u32>();
        let n_new = n + 1;
        node.q += (v - node.q) / n_new as f64;
        // Bump one outgoing edge to simulate the edge-visit increment that
        // tree MCTS performs on the selected child edge.
        if let Some(edge) = node.children.first_mut() {
            edge.edge_visits += 1;
        }
        black_box(node.q);
    }
}

/// DAG-MCGS-style idempotent recursive Q recompute. Walk the path leaf-to-root.
/// At each node V, recompute V.q from V.children via FxHashMap lookups (the
/// load-bearing cost — pointer chasing into other Box<MockDagNode>s). Then
/// bump the parent's selected edge.
fn dag_backprop(map: &mut DagMap, path: &[u128]) {
    // Walk leaf-to-root.
    for i in (0..path.len()).rev() {
        let h = path[i];

        // Step 1: recompute Q from children. Borrow children list immutably
        // by cloning the Vec briefly (the production code will be able to do
        // this without a clone via split-borrow tricks; the clone here is a
        // small constant overhead and is conservative — i.e., it makes the
        // DAG benchmark slightly *slower* than production, not faster).
        let (u_v, children) = {
            let node = map.get(&h).unwrap();
            (node.u, node.children.clone())
        };

        let n_v: u32 = 1 + children.iter().map(|e| e.edge_visits).sum::<u32>();
        let mut sum: f64 = 0.0;
        for e in &children {
            // The hashmap lookup is the load-bearing cost.
            let child = map.get(&e.child_hash).unwrap();
            sum += child.q * e.edge_visits as f64;
        }
        let new_q = (u_v + sum) / n_v as f64;
        map.get_mut(&h).unwrap().q = new_q;
        black_box(new_q);

        // Step 2: bump the parent's selected edge (the one that points to h).
        if i > 0 {
            let parent_hash = path[i - 1];
            let parent = map.get_mut(&parent_hash).unwrap();
            for e in parent.children.iter_mut() {
                if e.child_hash == h {
                    e.edge_visits += 1;
                    break;
                }
            }
        }
    }
}

fn bench_grid(c: &mut Criterion) {
    let branchings = [15usize, 30, 60];
    let depths = [10usize, 20, 40];

    let mut group = c.benchmark_group("dag_backprop");
    // Keep runtime sane: smaller sample size for the big cells.
    group.sample_size(50);

    for &b in &branchings {
        for &d in &depths {
            // Skip cells that would allocate ridiculous amounts of memory
            // (b=60, d=40 would be 60^40 nodes — obviously not feasible).
            // We cap total node count at ~2M to keep the bench under a few
            // hundred MB. For deep+wide cells we fall back to a "spine" tree:
            // a single full subtree at depth min(d, max_full) and a linear
            // tail. This still measures the per-node cost correctly because
            // backprop only walks one path.
            let full_depth_for_branch = match b {
                15 => 5, // 15^5 ~= 760k
                30 => 4, // 30^4 ~= 810k
                60 => 3, // 60^3 ~= 216k
                _ => 4,
            };
            let full_depth = d.min(full_depth_for_branch);

            // Build the full tree to `full_depth`, then we'll measure cost on a
            // path of length `d`. To get a path of length d > full_depth, we
            // graft a linear chain onto the chosen leaf. Easier: build a full
            // subtree to full_depth, then keep extending the chosen path with
            // synthetic single-child nodes.
            let (mut map, _root, mut path) = build_mock_dag(b, full_depth, (b * 1000 + d) as u64);

            // Extend path with linear single-child nodes if needed.
            if d > full_depth {
                let mut next_hash: u128 = (map.len() as u128) + 100;
                let extra = d - full_depth;
                let mut prev = *path.last().unwrap();
                let mut rng = StdRng::seed_from_u64((b * 7919 + d) as u64);
                for _ in 0..extra {
                    let h = next_hash;
                    next_hash += 1;
                    let u: f64 = rng.random_range(-1.0..1.0);
                    map.insert(
                        h,
                        Box::new(MockDagNode {
                            u,
                            q: u,
                            children: Vec::new(),
                        }),
                    );
                    // Attach as a child of `prev`.
                    let parent = map.get_mut(&prev).unwrap();
                    parent.children.push(MockDagEdge {
                        child_hash: h,
                        edge_visits: 1,
                    });
                    path.push(h);
                    prev = h;
                }
            }

            let id = format!("b{}_d{}", b, d);

            // Tree variant
            {
                let mut map_clone = map.clone();
                let path_clone = path.clone();
                group.bench_with_input(BenchmarkId::new("tree", &id), &id, |bencher, _| {
                    bencher.iter(|| {
                        tree_backprop(&mut map_clone, &path_clone, black_box(0.5));
                    });
                });
            }

            // DAG variant
            {
                let mut map_clone = map.clone();
                let path_clone = path.clone();
                group.bench_with_input(BenchmarkId::new("dag", &id), &id, |bencher, _| {
                    bencher.iter(|| {
                        dag_backprop(&mut map_clone, &path_clone);
                    });
                });
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_grid);
criterion_main!(benches);
