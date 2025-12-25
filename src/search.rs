//! K-NN search module using cosine similarity
//!
//! Parallelized search across memory-mapped f32 vectors.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::storage::VectorStorage;
use crate::types::EmbeddedVector;

/// Compute dot product of two f32 vectors
#[inline]
fn dot_product(a: &EmbeddedVector, b: &EmbeddedVector) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Compute L2 norm (magnitude) of a vector
#[inline]
fn l2_norm(v: &EmbeddedVector) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute cosine similarity between two vectors
/// Returns value in [-1, 1] where 1 is identical, 0 is orthogonal, -1 is opposite
#[inline]
pub fn cosine_similarity(a: &EmbeddedVector, b: &EmbeddedVector) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

/// Search result with similarity score (used internally)
#[derive(Debug, Clone, Copy)]
struct SearchCandidate {
    id: u64,
    score: f32,
}

impl Eq for SearchCandidate {}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse comparison for max-heap (higher scores first)
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: keep highest scores, remove lowest
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Perform k-NN search using cosine similarity with parallel processing.
///
/// # Arguments
/// * `storage` - Vector storage to search
/// * `query` - Query vector (full precision f32)
/// * `k` - Number of nearest neighbors to return
///
/// # Returns
/// * `Result<Vec<(u64, f32)>>` - List of (id, similarity_score) tuples, sorted by score descending
pub fn search_cosine(
    storage: &VectorStorage,
    query: &EmbeddedVector,
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    let count = storage.count() as usize;

    if count == 0 {
        return Ok(Vec::new());
    }

    let actual_k = k.min(count);

    // Collect all vectors for parallel processing
    let vectors: Vec<_> = storage.iter().collect();

    // Parallel similarity computation using rayon
    let similarities: Vec<_> = vectors
        .par_iter()
        .map(|(id, vector)| {
            let score = cosine_similarity(query, vector);
            SearchCandidate {
                id: *id,
                score,
            }
        })
        .collect();

    // Find top-k using max-heap (keep highest scores)
    let mut heap = BinaryHeap::with_capacity(actual_k + 1);

    for candidate in similarities {
        heap.push(candidate);
        if heap.len() > actual_k {
            heap.pop(); // Remove the worst (lowest score)
        }
    }

    // Convert to results and sort by score (descending)
    let mut results: Vec<_> = heap
        .into_iter()
        .map(|candidate| (candidate.id, candidate.score))
        .collect();

    // Sort by score descending (best matches first)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cosine_similarity_identical() {
        let vec = [1.0f32; 1536];
        let sim = cosine_similarity(&vec, &vec);
        assert!((sim - 1.0).abs() < 0.001); // Should be 1.0
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut vec_a = [0.0f32; 1536];
        vec_a[0] = 1.0;
        
        let mut vec_b = [0.0f32; 1536];
        vec_b[1] = 1.0;
        
        let sim = cosine_similarity(&vec_a, &vec_b);
        assert!(sim.abs() < 0.001); // Should be ~0.0
    }

    #[test]
    fn test_search_basic() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = VectorStorage::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Add some test vectors
        let vec1 = [0.5f32; 1536];
        let mut vec2 = [0.0f32; 1536];
        vec2[0] = 1.0; // Different vector
        
        storage.append(&vec1).unwrap();
        storage.append(&vec2).unwrap();

        // Search for vec1
        let results = search_cosine(&storage, &vec1, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // First result should be vec1
        assert!(results[0].1 > 0.99); // Very high similarity
    }

    #[test]
    fn test_search_empty_db() {
        let temp_dir = TempDir::new().unwrap();
        let storage = VectorStorage::new(temp_dir.path().to_str().unwrap()).unwrap();

        let query = [1.0f32; 1536];
        let results = search_cosine(&storage, &query, 10).unwrap();
        
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_search_ranking() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = VectorStorage::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Create query vector
        let mut query = [0.0f32; 1536];
        query[0] = 1.0;
        query[1] = 1.0;

        // Vector 1: Exact match
        storage.append(&query).unwrap();
        
        // Vector 2: Partial match
        let mut vec2 = [0.0f32; 1536];
        vec2[0] = 1.0;
        storage.append(&vec2).unwrap();
        
        // Vector 3: No match
        let vec3 = [0.5f32; 1536];
        storage.append(&vec3).unwrap();

        let results = search_cosine(&storage, &query, 3).unwrap();
        
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0); // Exact match first
        assert!(results[0].1 > results[1].1); // Scores descending
        assert!(results[1].1 > results[2].1);
    }
}
