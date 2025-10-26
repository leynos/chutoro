//! Node storage for the CPU HNSW graph.
//!
//! Maintains per-level neighbour lists and provides accessors used during
//! search, insertion, and trimming.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    neighbours: Vec<Vec<usize>>,
}

impl Node {
    pub(crate) fn new(level: usize) -> Self {
        let mut neighbours = Vec::with_capacity(level + 1);
        neighbours.resize_with(level + 1, Vec::new);
        Self { neighbours }
    }

    pub(crate) fn neighbours(&self, level: usize) -> &[usize] {
        debug_assert!(
            level < self.neighbours.len(),
            "levels are initialised during construction"
        );
        self.neighbours
            .get(level)
            .expect("levels are initialised during construction")
            .as_slice()
    }

    pub(crate) fn neighbours_mut(&mut self, level: usize) -> &mut Vec<usize> {
        self.neighbours
            .get_mut(level)
            .expect("levels are initialised during construction")
    }
}
