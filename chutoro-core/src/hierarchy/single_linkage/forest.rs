//! Single-linkage forest construction from mutual-reachability MST edges.
//!
//! The MST edges encode the same hierarchy as the full mutual-reachability
//! graph. We recover the hierarchy by sorting edges by weight and merging
//! components with a union-find structure, creating a dendrogram node for each
//! merge.

use crate::mst::MstEdge;

use super::super::union_find::DisjointSet;
use super::{LinkageNode, SingleLinkageForest};

impl SingleLinkageForest {
    fn merge_edges(
        dsu: &mut DisjointSet,
        nodes: &mut Vec<LinkageNode>,
        edges_sorted: Vec<MstEdge>,
    ) {
        for edge in edges_sorted {
            let left_root = dsu.find(edge.source());
            let right_root = dsu.find(edge.target());
            if left_root == right_root {
                continue;
            }
            let left_node = dsu.component_node[left_root];
            let right_node = dsu.component_node[right_root];
            let new_id = nodes.len();
            let size = nodes[left_node].size + nodes[right_node].size;
            nodes.push(LinkageNode {
                left: Some(left_node),
                right: Some(right_node),
                weight: edge.weight(),
                size,
                point: None,
            });
            let merged = dsu.union(left_root, right_root);
            dsu.component_node[merged] = new_id;
        }
    }

    fn collect_roots(dsu: &mut DisjointSet, node_count: usize) -> Vec<usize> {
        let mut roots: Vec<usize> = (0..node_count)
            .filter_map(|node| {
                let root = dsu.find(node);
                (root == node).then_some(dsu.component_node[root])
            })
            .collect();

        roots.sort_unstable();
        roots.dedup();
        roots
    }

    pub(super) fn from_mst(node_count: usize, edges: &[MstEdge]) -> Self {
        let mut nodes = Vec::with_capacity(node_count.saturating_mul(2).saturating_sub(1));
        for point in 0..node_count {
            nodes.push(LinkageNode {
                left: None,
                right: None,
                weight: 0.0,
                size: 1,
                point: Some(point),
            });
        }

        let mut dsu = DisjointSet::new(node_count);
        let mut edges_sorted = edges.to_vec();
        edges_sorted.sort_unstable();

        Self::merge_edges(&mut dsu, &mut nodes, edges_sorted);
        let roots = Self::collect_roots(&mut dsu, node_count);
        Self { nodes, roots }
    }
}
