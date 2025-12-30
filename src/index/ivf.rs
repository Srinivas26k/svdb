//! # Inverted File (IVF) Clustering
//!
//! Partitions the vector space into Voronoi cells to accelerate search
//! by restricting the search scope to relevant partitions.

use anyhow::Result;
use parking_lot::RwLock;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::hnsw::{HNSWConfig, HNSWIndex};
use crate::Vector;

/// IVF Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFConfig {
    /// Number of partitions (centroids)
    pub nlist: usize,
    /// Number of partitions to probe during search
    pub nprobe: usize,
    /// Maximum iterations for K-Means training
    pub max_iterations: usize,
    /// Convergence tolerance for K-Means
    pub tolerance: f32,
    /// Configuration for the internal HNSW graphs
    pub hnsw_config: HNSWConfig,
}

impl Default for IVFConfig {
    fn default() -> Self {
        Self {
            nlist: 100, // Good for ~1M vectors
            nprobe: 10,
            max_iterations: 20,
            tolerance: 0.001,
            hnsw_config: HNSWConfig::default(),
        }
    }
}

/// The IVF Index structure
/// Manages centroids and the partitioned HNSW graphs
pub struct IVFIndex {
    pub config: IVFConfig,
    /// Centroids (cluster centers) - size of `nlist`
    pub centroids: Vec<Vector>,
    /// The partitions themselves. Each is an independent HNSW graph.
    /// We use RwLock to allow concurrent searching/updating of different partitions.
    pub partitions: Vec<Arc<RwLock<HNSWIndex>>>,
}

impl IVFIndex {
    /// Create a new empty IVF index
    pub fn new(config: IVFConfig) -> Self {
        let mut partitions = Vec::with_capacity(config.nlist);
        for _ in 0..config.nlist {
            partitions.push(Arc::new(RwLock::new(HNSWIndex::new(
                config.hnsw_config.clone(),
            ))));
        }

        Self {
            config,
            centroids: Vec::new(),
            partitions,
        }
    }

    /// Train the centroids using K-Means clustering
    ///
    /// This is a "Course Quantization" step.
    /// Returns the centroids and assigns the training vectors to partitions (though assignment isn't stored here).
    pub fn train(&mut self, training_data: &[Vector]) -> Result<()> {
        if training_data.is_empty() {
            return Ok(());
        }

        let k = self.config.nlist;
        let dim = training_data[0].dim();

        println!(
            "IVF: Training {} centroids on {} vectors...",
            k,
            training_data.len()
        );

        // 1. Initialize Centroids (Random Point Pick)
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        for _ in 0..k {
            let idx = rng.gen_range(0..training_data.len());
            centroids.push(training_data[idx].data.clone());
        }

        // 2. K-Means Iterations
        for iter in 0..self.config.max_iterations {
            // Assignment Step: Assign each vector to nearest centroid
            // We sum vectors per cluster to recompute mean later
            let assignments: Vec<(usize, Vec<f32>)> = training_data
                .par_iter()
                .map(|vec| {
                    let mut best_dist = f32::MAX;
                    let mut best_k = 0;

                    // Simple linear scan for centroid finding (Simd optimized dot product would be better)
                    // Since K is small (e.g. 100-1000), this is acceptable.
                    for (i, center) in centroids.iter().enumerate() {
                        let dist = crate::search::euclidean_distance(&vec.data, center);
                        if dist < best_dist {
                            best_dist = dist;
                            best_k = i;
                        }
                    }
                    (best_k, vec.data.clone())
                })
                .collect();

            // Update Step: Recompute centroids
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];

            for (cluster_idx, vec_data) in assignments {
                counts[cluster_idx] += 1;
                for i in 0..dim {
                    new_centroids[cluster_idx][i] += vec_data[i];
                }
            }

            let mut diff = 0.0;
            for i in 0..k {
                if counts[i] > 0 {
                    let factor = 1.0 / counts[i] as f32;
                    for val in new_centroids[i].iter_mut().take(dim) {
                        *val *= factor;
                    }
                } else {
                    // Re-init empty cluster with random point to avoid dead clusters
                    let idx = rng.gen_range(0..training_data.len());
                    new_centroids[i] = training_data[idx].data.clone();
                }

                diff += crate::search::euclidean_distance(&centroids[i], &new_centroids[i]);
            }

            centroids = new_centroids;

            println!("  -> Iteration {}: diff = {:.4}", iter + 1, diff);
            if diff < self.config.tolerance {
                println!("  -> Converged.");
                break;
            }
        }

        self.centroids = centroids.into_iter().map(Vector::new).collect();
        Ok(())
    }

    /// Find the closest partition ID for a vector
    pub fn find_partition(&self, vector: &Vector) -> usize {
        let mut best_dist = f32::MAX;
        let mut best_k = 0;

        for (i, center) in self.centroids.iter().enumerate() {
            // We use the same distance metric as HNSW usually, assume Cosine/Euclidean
            // Ideally should match config.
            let dist = crate::search::cosine_similarity(&vector.data, &center.data);
            // Note: cosine_similarity is "higher is better", we need distance "lower is better"
            // So convert sim to dist: 1.0 - sim
            let dist = 1.0 - dist;

            if dist < best_dist {
                best_dist = dist;
                best_k = i;
            }
        }
        best_k
    }

    /// Add a vector to the index.
    ///
    /// Note: This does NOT add it to global storage (vectors.bin),
    /// it only adds the ID to the appropriate partition graph.
    pub fn add(
        &self,
        id: u64,
        vector: &Vector,
        distance_fn: &impl Fn(u64, u64) -> f32,
    ) -> Result<()> {
        if self.centroids.is_empty() {
            anyhow::bail!("IVF Index not trained");
        }

        let partition_id = self.find_partition(vector);

        let partition = &self.partitions[partition_id];
        partition.read().insert(id, distance_fn)?; // READ lock for HNSW, HNSW handles internal Write locks?
                                                   // Wait, HNSW insert needs to mutate the graph structure.
                                                   // HNSWIndex uses RwLocks internally for its nodes map.
                                                   // HNSWIndex::insert takes &self (immutable reference).
                                                   // Check hnsw.rs line 181: pub fn insert<F>(&self, ...
                                                   // Yes, HNSWIndex is thread-safe with internal locking.

        Ok(())
    }
}
