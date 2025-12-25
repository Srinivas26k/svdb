//! K-NN search module using Hamming distance
//!
//! Parallelized search across memory-mapped vectors.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::storage::VectorStorage;
use crate::types::QuantizedVector;
use crate::quantize::{hamming_distance, hamming_to_similarity};

/// Search result with distance (used internally)
#[derive(Debug, Clone, Copy)]
struct SearchCandidate {
    id: u64,
    distance: u32,
}

impl Eq for SearchCandidate {}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max heap: larger distances at top (we want to keep smaller distances)
        other.distance.cmp(&self.distance)
    }
}

/// Perform k-NN search using Hamming distance with parallel processing.
///
/// # Arguments
/// * `storage` - Vector storage to search
/// * `query` - Quantized query vector
/// * `k` - Number of nearest neighbors to return
///
/// # Returns
/// * `Result<Vec<(u64, f32)>>` - List of (id, similarity_score) tuples
pub fn search_hamming(
    storage: &VectorStorage,
    query: &QuantizedVector,
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    let count = storage.count() as usize;

    if count == 0 {
        return Ok(Vec::new());
    }

    let actual_k = k.min(count);

    // Collect all vectors for parallel processing
    let vectors: Vec<_> = storage.iter().collect();

    // Parallel distance computation
    let distances: Vec<_> = vectors
        .par_iter()
        .map(|(id, vector)| {
            let distance = hamming_distance(query, vector);
            SearchCandidate {
                id: *id,
                distance,
            }
        })
        .collect();

    // Find top-k using min-heap
    let mut heap = BinaryHeap::with_capacity(actual_k + 1);

    for candidate in distances {
        heap.push(candidate);
        if heap.len() > actual_k {
            heap.pop(); // Remove the worst (largest distance)
        }
    }

    // Convert to results and sort by score (descending)
    let mut results: Vec<_> = heap
        .into_iter()
        .map(|candidate| {
            let score = hamming_to_similarity(candidate.distance);
            (candidate.id, score)
        })
        .collect();

    // Sort by score descending (best matches first)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::quantize_vector;
    use tempfile::TempDir;

    #[test]
    fn test_search_basic() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = VectorStorage::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Add some test vectors
        let vec1 = quantize_vector(&vec![1.0; 1536]);
        let vec2 = quantize_vector(&vec![-1.0; 1536]);
        
        storage.append(&vec1).unwrap();
        storage.append(&vec2).unwrap();

        // Search for vec1
        let results = search_hamming(&storage, &vec1, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // First result should be vec1
        assert!(results[0].1 > 0.99); // Very high similarity
    }

    #[test]
    fn test_search_empty_db() {
        let temp_dir = TempDir::new().unwrap();
        let storage = VectorStorage::new(temp_dir.path().to_str().unwrap()).unwrap();

        let query = quantize_vector(&vec![1.0; 1536]);
        let results = search_hamming(&storage, &query, 10).unwrap();
        
        assert_eq!(results.len(), 0);
    }
}
