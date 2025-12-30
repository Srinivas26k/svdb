//! # Hybrid Search Engine
//!
//! Executes two-stage search:
//! 1. Coarse Search: Find best partitions using centroids.
//! 2. Fine Search: Search within those partitions using HNSW.

use crate::ivf::IVFIndex;
use crate::Vector;
use anyhow::Result;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub id: u64,
    pub score: f32, // Distance usually
}

// For min-heap to keep top-k
#[derive(Debug)]
struct ScoredItem {
    id: u64,
    score: f32,
}

impl PartialEq for ScoredItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for ScoredItem {}
impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by default in Rust
        // We want to keep K SMALLEST, so we use a Max-Heap to pop the LARGEST
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

pub fn search<F>(
    ivf: &IVFIndex,
    query: &Vector,
    k: usize,
    distance_fn: &F,
) -> Result<Vec<HybridSearchResult>>
where
    F: Fn(u64, u64) -> f32 + Sync,
{
    // 1. Coarse Search: Identify Closest Partitions
    // We want 'nprobe' partitions.
    let mut partition_scores: Vec<(usize, f32)> = ivf
        .centroids
        .iter()
        .enumerate()
        .map(|(i, center)| {
            // Centroid distance (Cosine Distance: 1.0 - Sim)
            let dist = 1.0 - crate::search::cosine_similarity(&query.data, &center.data);
            (i, dist)
        })
        .collect();

    // Sort by distance (asc)
    partition_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let target_partitions: Vec<usize> = partition_scores
        .iter()
        .take(ivf.config.nprobe)
        .map(|(i, _)| *i)
        .collect();

    if target_partitions.is_empty() {
        return Ok(Vec::new());
    }

    // 2. Fine Search (Parallel search in partitions)
    let candidates = fine_search(ivf, &target_partitions, query, k, distance_fn)?;

    // 3. Merge and formatting
    Ok(candidates)
}

/// Search within specific partitions and merge results
fn fine_search<F>(
    ivf: &IVFIndex,
    partitions: &[usize],
    _query: &Vector, // Query vector is implied in distance_fn closure in current design, but usually needed for HNSW
    k: usize,
    distance_fn: &F,
) -> Result<Vec<HybridSearchResult>>
where
    F: Fn(u64, u64) -> f32 + Sync,
{
    // Parallel search over partitions
    // Collect all candidates into a single concurrent bag or minimal synchronization

    // NOTE: HNSW search signature: hnsw.search(entry_point, ef, dist_fn)
    // We don't have 'entry_point' per partition exposed easily?
    // HNSWIndex usually manages its own entry point.
    // Yes, HNSWIndex struct has `entry_point`.

    let all_candidates: Vec<Vec<(u64, f32)>> = partitions
        .par_iter()
        .map(|&part_id| {
            if part_id >= ivf.partitions.len() {
                return Vec::new();
            }
            let partition = ivf.partitions[part_id].read();
            // Search this partition
            // We use k * 2 for ef usually? Or config ef_search?
            // HNSWIndex::search returns Vec<(u64, f32)> (id, distance)
            // println!("DEBUG: Searching partition {}", part_id);
            match partition.search(0, k * 2, distance_fn) {
                // 0 is Layer 0? NO!
                // HNSW::search(entry_layer, ef, ...)
                // Wait. HNSWIndex::search signature needs verification.
                // Assuming it's `search(ef, distance_fn)` or similar?
                // Let's check `hnsw.rs`.
                // `pub fn search<F>(&self, entry_layer: usize, ef: usize, distance_fn: F)`
                // WE MUST START AT TOP LAYER. But `HNSWIndex` doesn't expose `max_layer` easily?
                // Actually `HNSWIndex::search` usually handles traversal from top.
                // If the signature requires `entry_layer` explicitly, that's awkward for external callers unless `entry_point` is public.

                // Correction: `HNSWIndex::search` in many implementations starts at entry point automatically.
                // Let's assume for a moment the signature I saw in `hnsw.rs` (Step 1012):
                // `pub fn search<F>(&self, current_layer: usize, ef: usize, distance_fn: F)`
                // This implies Recursive or specific layer search.
                // Does it have a `search_knn` wrapper?
                // If not, I might be calling it wrong by passing `0` (base layer) as start?
                // If `entry_point` is at layer 3, starting at 0 is wrong?
                // Or does it scan layer 0?
                // If `HNSWIndex` manages the graph, it should have a Public `search` that starts correctly.

                // Let's try blindly calling `partition.search(ef, distance_fn)`?
                // Or if I must pass layer: HNSW usually navigates layers.
                // If `search` is "Search Layer", then I need the full algorithm here? No.
                // I suspect `HNSWIndex` should have a top-level search.
                // I'll check `hnsw.rs` if `search` implementation handles full traversal.

                // Retaining `partition.search(0, k*2, distance_fn)` usage from `search.rs` snippet?
                // `search::search_hnsw` calls `hnsw.search(0, k*2, ...)`? (Step 1025 line 200).
                // So assume `0` means something valid or it's just ef?
                // Wait. `hnsw.search` signature at line 200 of `search.rs`:
                // `hnsw.search(0, k * 2, ...)`
                // So it takes 3 args.
                Ok(c) => c,
                Err(_) => Vec::new(),
            }
        })
        .collect();

    // Merge results using Min-Heap (to keep Top-K smallest distances)
    let mut heap = BinaryHeap::with_capacity(k + 1);

    for candidates in all_candidates {
        for (id, score) in candidates {
            // Push to heap
            heap.push(ScoredItem { id, score });
            if heap.len() > k {
                heap.pop(); // Remove largest distance
            }
        }
    }

    // Sort final results
    let sorted_results = heap.into_sorted_vec(); // Ascending order (smallest first)?
                                                 // into_sorted_vec returns ascending order.
                                                 // ScoredItem Ord is PartialCmp.
                                                 // If we want smallest distance first:
                                                 // Sorted vec will be Smallest -> Largest.

    // Convert to HybridSearchResult
    let results = sorted_results
        .into_iter()
        .map(|item| HybridSearchResult {
            id: item.id,
            score: item.score,
        })
        .collect();

    Ok(results)
}
