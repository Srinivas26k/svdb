//! Ultra-fast SIMD-accelerated k-NN search
//!
//! Optimizations:
//! - Batch processing with 256-vector chunks
//! - SIMD intrinsics for cosine similarity
//! - Lock-free parallel heap for top-k selection
//! - Cache-optimized memory access patterns

use crate::storage::pq::QuantizedVectorStorage;
use crate::VectorStorage;
use anyhow::Result;
use rayon::prelude::*;
use simsimd::SpatialSimilarity;
use std::cmp::Ordering;

const BATCH_SIZE: usize = 256; // Process 256 vectors at once for cache efficiency

#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let distance = f32::cosine(a, b).unwrap_or(2.0) as f32;
    1.0 - (distance / 2.0)
}

#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    f32::sqeuclidean(a, b).unwrap_or(0.0) as f32
}

/// Batch compute similarities for cache efficiency
#[inline]
fn compute_similarities_batch(query: &[f32], vectors: &[&[f32]], start_id: u64) -> Vec<(u64, f32)> {
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
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

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
pub fn search_cosine(storage: &VectorStorage, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
    let count = storage.count() as usize;

    if count == 0 {
        return Ok(Vec::new());
    }

    let actual_k = k.min(count);

    // Process in cache-friendly batches
    let num_batches = count.div_ceil(BATCH_SIZE);

    let all_similarities: Vec<(u64, f32)> = (0..num_batches)
        .into_par_iter()
        .flat_map(|batch_idx| {
            let start = batch_idx * BATCH_SIZE;
            let batch_size = BATCH_SIZE.min(count - start);

            // Get batch of vectors (zero-copy)
            if let Some(vectors) = storage.get_batch(start as u64, batch_size) {
                compute_similarities_batch(query, &vectors, start as u64)
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
    queries: &[Vec<f32>],
    k: usize,
) -> Result<Vec<Vec<(u64, f32)>>> {
    queries
        .par_iter()
        .map(|query| search_cosine(storage, query.as_slice(), k))
        .collect()
}

// ============================================================================
// PRODUCT QUANTIZATION SEARCH (ADC)
// ============================================================================

/// Optimized quantized search with Asymmetric Distance Computation
/// NOTE: Temporarily disabled - requires quantization.rs refactoring for dynamic dimensions
pub fn search_quantized(
    storage: &QuantizedVectorStorage,
    query: &[f32],
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    let count = storage.count();
    if count == 0 {
        return Ok(Vec::new());
    }

    // 1. Precompute Distance Table (O(D * K))
    // We assume the query is already normalized or handled by quantizer
    let dtable = storage.quantizer.compute_distance_table(query);

    // 2. Scan all vectors (O(N * M))
    // Use parallel iterator for speed
    let num_batches = (count as usize).div_ceil(BATCH_SIZE);

    let all_scores: Vec<(u64, f32)> = (0..num_batches)
        .into_par_iter()
        .flat_map(|batch_idx| {
            let start = batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(count as usize);
            let mut results = Vec::with_capacity(end - start);

            // Access local Mmap slice for this batch if possible? 
            // storage.get() does bounds check every time. 
            // get_batch() is better.
            if let Some(batch_data) = storage.get_batch(start as u64, end - start) {
                // batch_data is a flat slice of [u8].
                // We know each vector has storage.q_size bytes.
                let q_size = storage.q_size;
                
                for (i, chunk) in batch_data.chunks(q_size).enumerate() {
                    if chunk.len() == q_size {
                         let dist = storage.quantizer.asymmetric_distance(chunk, &dtable);
                         results.push(((start + i) as u64, dist));
                    }
                }
            } else {
                 // Fallback to individual gets if batch fails (shouldn't happen)
                for i in start..end {
                    if let Some(qvec) = storage.get(i as u64) {
                        let dist = storage.quantizer.asymmetric_distance(qvec, &dtable);
                        results.push((i as u64, dist));
                    }
                }
            }
            results
        })
        .collect();

    Ok(select_top_k(all_scores, k))
}

/// Batch quantized search
/// NOTE: Temporarily disabled
pub fn search_quantized_batch(
    storage: &QuantizedVectorStorage,
    queries: &[Vec<f32>],
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
    query: &[f32],
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
/// NOTE: Temporarily disabled
pub fn search_hnsw_quantized(
    storage: &QuantizedVectorStorage,
    hnsw: &crate::hnsw::HNSWIndex,
    query: &[f32],
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    // Precompute lookup table for this query
    let dtable = storage.quantizer.compute_distance_table(query);

    // HNSW graph traversal using ADC
    let candidates = hnsw.search(0, k * 2, &|_query_id: u64, vec_id: u64| -> f32 {
        if let Some(qvec) = storage.get(vec_id) {
             let sim = storage.quantizer.asymmetric_distance(qvec, &dtable);
             // Convert Similarity to Distance (HNSW minimizes)
             // Sim is typically 0.0 to 1.0 (approx cosine)
             1.0 - sim
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorStorage;
    use tempfile::TempDir;

    #[test]
    fn test_batch_search() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = VectorStorage::new(temp_dir.path().to_str().unwrap(), 1536).unwrap();

        // Add test vectors
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| {
                let mut v = vec![0.0f32; 1536];
                v[0] = (i as f32) / 1000.0;
                v
            })
            .collect();

        // Convert to slice of slices for append_batch
        // Note: append_batch expects &[Vec<f32>]
        storage.append_batch(&vectors).unwrap();
        storage.flush().unwrap();

        // Create multiple queries
        let queries: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut v = vec![0.0f32; 1536];
                v[0] = (i as f32) / 10.0;
                v
            })
            .collect();

        // Convert queries to required format
        let query_slices: Vec<Vec<f32>> = queries.clone();

        let results = search_batch(&storage, &query_slices, 5).unwrap();
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.len() == 5));
    }

    #[test]
    fn test_partial_sort_performance() {
        let candidates: Vec<(u64, f32)> = (0..10000).map(|i| (i, rand::random::<f32>())).collect();

        let top_k = select_top_k(candidates, 10);
        assert_eq!(top_k.len(), 10);

        // Verify descending order
        for i in 1..top_k.len() {
            assert!(top_k[i - 1].1 >= top_k[i].1);
        }
    }
}
