//! Ultra-fast SIMD-accelerated k-NN search
//!
//! Optimizations:
//! - Batch processing with 256-vector chunks
//! - SIMD intrinsics for cosine similarity
//! - Lock-free parallel heap for top-k selection
//! - Cache-optimized memory access patterns

use anyhow::Result;
use rayon::prelude::*;
use std::cmp::Ordering;
use crate::storage::VectorStorage;
use crate::quantized_storage::QuantizedVectorStorage;
use crate::types::EmbeddedVector;
use simsimd::SpatialSimilarity;

const BATCH_SIZE: usize = 256; // Process 256 vectors at once for cache efficiency

#[inline]
pub fn cosine_similarity(a: &EmbeddedVector, b: &EmbeddedVector) -> f32 {
    let distance = f32::cosine(a, b).unwrap_or(2.0) as f32;
    1.0 - (distance / 2.0)
}

/// Batch compute similarities for cache efficiency
#[inline]
fn compute_similarities_batch(
    query: &EmbeddedVector,
    vectors: &[EmbeddedVector],
    start_id: u64,
) -> Vec<(u64, f32)> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, vec)| {
            let score = cosine_similarity(query, vec);
            (start_id + i as u64, score)
        })
        .collect()
}

/// Fast top-k selection using partial sorting
fn select_top_k(mut candidates: Vec<(u64, f32)>, k: usize) -> Vec<(u64, f32)> {
    let k = k.min(candidates.len());
    
    // Partial sort: only sort enough to get top-k
    candidates.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
    });
    
    // Take top-k and sort them
    let mut top_k: Vec<_> = candidates.into_iter().take(k).collect();
    top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    
    top_k
}

/// Optimized parallel search with batch processing
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

    // Process in cache-friendly batches
    let num_batches = (count + BATCH_SIZE - 1) / BATCH_SIZE;
    
    let all_similarities: Vec<(u64, f32)> = (0..num_batches)
        .into_par_iter()
        .flat_map(|batch_idx| {
            let start = batch_idx * BATCH_SIZE;
            let batch_size = BATCH_SIZE.min(count - start);
            
            // Get batch of vectors (zero-copy)
            if let Some(vectors) = storage.get_batch(start as u64, batch_size) {
                compute_similarities_batch(query, vectors, start as u64)
            } else {
                Vec::new()
            }
        })
        .collect();

    // Fast top-k selection
    Ok(select_top_k(all_similarities, actual_k))
}

/// Hyper-optimized multi-query search (for concurrent throughput)
pub fn search_batch(
    storage: &VectorStorage,
    queries: &[EmbeddedVector],
    k: usize,
) -> Result<Vec<Vec<(u64, f32)>>> {
    queries
        .par_iter()
        .map(|query| search_cosine(storage, query, k))
        .collect()
}

// ============================================================================
// PRODUCT QUANTIZATION SEARCH (ADC)
// ============================================================================

/// Optimized quantized search with Asymmetric Distance Computation
pub fn search_quantized(
    storage: &QuantizedVectorStorage,
    query: &EmbeddedVector,
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    let count = storage.count() as usize;

    if count == 0 {
        return Ok(Vec::new());
    }

    let actual_k = k.min(count);

    // Normalize query vector for cosine similarity
    let mut normalized_query = *query;
    let norm: f32 = normalized_query.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in normalized_query.iter_mut() {
            *x /= norm;
        }
    }

    // Precompute distance table once (ADC optimization)
    let dtable = storage.quantizer.compute_distance_table(&normalized_query);

    // Process in cache-friendly batches
    let num_batches = (count + BATCH_SIZE - 1) / BATCH_SIZE;
    
    let all_similarities: Vec<(u64, f32)> = (0..num_batches)
        .into_par_iter()
        .flat_map(|batch_idx| {
            let start = batch_idx * BATCH_SIZE;
            let batch_size = BATCH_SIZE.min(count - start);
            
            // Get batch of quantized vectors (zero-copy)
            if let Some(qvecs) = storage.get_batch(start as u64, batch_size) {
                qvecs
                    .iter()
                    .enumerate()
                    .map(|(i, qvec)| {
                        // ADC now returns approximate dot product (Cosine Similarity)
                        let cosine_score = storage.quantizer.asymmetric_distance(qvec, &dtable);
                        (start as u64 + i as u64, cosine_score)
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        })
        .collect();

    // Fast top-k selection
    Ok(select_top_k(all_similarities, actual_k))
}

/// Batch quantized search
pub fn search_quantized_batch(
    storage: &QuantizedVectorStorage,
    queries: &[EmbeddedVector],
    k: usize,
) -> Result<Vec<Vec<(u64, f32)>>> {
    queries
        .par_iter()
        .map(|query| search_quantized(storage, query, k))
        .collect()
}

// ============================================================================
// HNSW GRAPH-BASED SEARCH
// ============================================================================

/// HNSW search for full-precision vectors
/// 
/// Uses graph-based approximate nearest neighbor search for O(log n) complexity
pub fn search_hnsw(
    storage: &VectorStorage,
    hnsw: &crate::hnsw::HNSWIndex,
    query: &EmbeddedVector,
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    // HNSW graph traversal with query-to-storage distance function
    let candidates = hnsw.search(0, k * 2, &|_query_id: u64, vec_id: u64| -> f32 {
        if let Some(vec) = storage.get(vec_id) {
            1.0 - cosine_similarity(query, vec) // Distance (lower is better)
        } else {
            f32::MAX
        }
    })?;
    
    // Convert back to similarities (higher is better) and take top-k
    let mut results: Vec<(u64, f32)> = candidates
        .into_iter()
        .map(|(id, dist)| (id, 1.0 - dist)) // Convert distance back to similarity
        .collect();
    
    // Sort by similarity (descending)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    results.truncate(k);
    
    Ok(results)
}

/// HNSW search for quantized vectors
/// 
/// Combines HNSW graph traversal with PQ distance computation
pub fn search_hnsw_quantized(
    storage: &QuantizedVectorStorage,
    hnsw: &crate::hnsw::HNSWIndex,
    query: &EmbeddedVector,
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    // Normalize query for cosine similarity
    let mut normalized_query = *query;
    let norm: f32 = normalized_query.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in normalized_query.iter_mut() {
            *x /= norm;
        }
    }
    
    // Precompute distance table for ADC
    let dtable = storage.quantizer.compute_distance_table(&normalized_query);
    
    // Distance function using PQ asymmetric distance
    let distance_fn = |_query_id: u64, vec_id: u64| -> f32 {
        if let Some(qvec) = storage.get(vec_id) {
            // ADC returns cosine similarity, convert to distance
            let similarity = storage.quantizer.asymmetric_distance(qvec, &dtable);
            1.0 - similarity
        } else {
            f32::MAX
        }
    };
    
    // Search HNSW graph
    let candidates = hnsw.search(0, k * 2, &distance_fn)?;
    
    // Convert back to similarities and take top-k
    let mut results: Vec<(u64, f32)> = candidates
        .into_iter()
        .map(|(id, dist)| (id, 1.0 - dist))
        .collect();
    
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    results.truncate(k);
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::storage::VectorStorage;

    #[test]
    fn test_batch_search() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = VectorStorage::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Add test vectors
        let vectors: Vec<EmbeddedVector> = (0..1000)
            .map(|i| {
                let mut v = [0.0f32; 1536];
                v[0] = (i as f32) / 1000.0;
                v
            })
            .collect();

        storage.append_batch(&vectors).unwrap();
        storage.flush().unwrap();

        // Create multiple queries
        let queries: Vec<EmbeddedVector> = (0..10)
            .map(|i| {
                let mut v = [0.0f32; 1536];
                v[0] = (i as f32) / 10.0;
                v
            })
            .collect();

        let results = search_batch(&storage, &queries, 5).unwrap();
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.len() == 5));
    }

    #[test]
    fn test_partial_sort_performance() {
        let candidates: Vec<(u64, f32)> = (0..10000)
            .map(|i| (i, rand::random::<f32>()))
            .collect();

        let top_k = select_top_k(candidates, 10);
        assert_eq!(top_k.len(), 10);
        
        // Verify descending order
        for i in 1..top_k.len() {
            assert!(top_k[i-1].1 >= top_k[i].1);
        }
    }
}