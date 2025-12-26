/// Example: Using HNSW for fast approximate nearest neighbor search
/// 
/// This example demonstrates:
/// 1. Creating a database with HNSW indexing
/// 2. Adding vectors (automatically builds the graph)
/// 3. Searching with O(log n) complexity
/// 4. Comparing HNSW vs flat search performance

use srvdb::{SvDB, Vector, VectorEngine, HNSWConfig};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SrvDB HNSW Example ===\n");
    
    // Generate some random test vectors
    let n_vectors = 1000;
    let dim = 1536;
    
    println!("Generating {} random {}-dimensional vectors...", n_vectors, dim);
    let vectors: Vec<Vector> = (0..n_vectors)
        .map(|i| {
            let mut data: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) as f32).sin())
                .collect();
            
            // Normalize
            let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut data {
                *x /= norm;
            }
            
            Vector::new(data)
        })
        .collect();
    
    let metadata: Vec<String> = (0..n_vectors)
        .map(|i| format!(r#"{{"id": {}}}"#, i))
        .collect();
    
    // ========================================================================
    // Part 1: Flat Search (baseline)
    // ========================================================================
    
    println!("\n--- Part 1: Flat Search (baseline) ---");
    let mut db_flat = SvDB::new("./example_flat_db")?;
    
    println!("Adding {} vectors...", n_vectors);
    let start = Instant::now();
    db_flat.add_batch(&vectors, &metadata)?;
    let add_time = start.elapsed();
    println!("✓ Added in {:?}", add_time);
    
    // Search
    let query = &vectors[0];
    let k = 10;
    
    println!("\nSearching for top-{} similar vectors...", k);
    let start = Instant::now();
    let results = db_flat.search(query, k)?;
    let search_time = start.elapsed();
    
    println!("✓ Flat search completed in {:?}", search_time);
    println!("  Top result: ID={}, score={:.4}", results[0].id, results[0].score);
    
    // ========================================================================
    // Part 2: HNSW Search (optimized)
    // ========================================================================
    
    println!("\n--- Part 2: HNSW Search (optimized) ---");
    
    // Configure HNSW
    let hnsw_config = HNSWConfig {
        m: 16,                 // 16 connections per node
        m_max: 16,
        m_max0: 32,            // 32 connections at layer 0
        ef_construction: 200,  // Candidate list size during construction
        ef_search: 50,         // Candidate list size during search
        ml: 1.0 / 16f32.ln(),  // Level multiplier
        use_quantization: false,
    };
    
    println!("HNSW Config: M={}, ef_construction={}, ef_search={}",
             hnsw_config.m, hnsw_config.ef_construction, hnsw_config.ef_search);
    
    let mut db_hnsw = SvDB::new_with_hnsw("./example_hnsw_db", hnsw_config)?;
    
    println!("Adding {} vectors (building HNSW graph)...", n_vectors);
    let start = Instant::now();
    db_hnsw.add_batch(&vectors, &metadata)?;
    let add_time_hnsw = start.elapsed();
    println!("✓ Added in {:?}", add_time_hnsw);
    
    // Search with HNSW
    println!("\nSearching for top-{} similar vectors...", k);
    let start = Instant::now();
    let results_hnsw = db_hnsw.search(query, k)?;
    let search_time_hnsw = start.elapsed();
    
    println!("✓ HNSW search completed in {:?}", search_time_hnsw);
    println!("  Top result: ID={}, score={:.4}", results_hnsw[0].id, results_hnsw[0].score);
    
    // ========================================================================
    // Part 3: Performance Comparison
    // ========================================================================
    
    println!("\n--- Performance Comparison ---");
    println!("Search Time:");
    println!("  Flat: {:?}", search_time);
    println!("  HNSW: {:?}", search_time_hnsw);
    
    let speedup = search_time.as_secs_f64() / search_time_hnsw.as_secs_f64();
    println!("  Speedup: {:.1}x faster", speedup);
    
    // Compare results (check recall)
    let flat_top_ids: std::collections::HashSet<_> = 
        results.iter().map(|r| r.id).collect();
    let hnsw_top_ids: std::collections::HashSet<_> = 
        results_hnsw.iter().map(|r| r.id).collect();
    
    let overlap: usize = flat_top_ids.intersection(&hnsw_top_ids).count();
    let recall = overlap as f32 / k as f32;
    
    println!("\nRecall@{}: {:.1}%", k, recall * 100.0);
    println!("(HNSW is approximate, so some results may differ)");
    
    // ========================================================================
    // Part 4: Tuning ef_search
    // ========================================================================
    
    println!("\n--- Part 4: Tuning ef_search for Recall/Speed Tradeoff ---");
    
    for ef in [10, 50, 100, 200] {
        db_hnsw.set_ef_search(ef);
        
        let start = Instant::now();
        let results_tuned = db_hnsw.search(query, k)?;
        let time_tuned = start.elapsed();
        
        let tuned_ids: std::collections::HashSet<_> = 
            results_tuned.iter().map(|r| r.id).collect();
        let overlap = flat_top_ids.intersection(&tuned_ids).count();
        let recall = overlap as f32 / k as f32;
        
        println!("ef_search={:3} → {:?}, recall={:.1}%", 
                 ef, time_tuned, recall * 100.0);
    }
    
    println!("\n✓ Example complete!");
    println!("Higher ef_search → better recall but slower search");
    println!("Lower ef_search → faster search but lower recall");
    
    // Cleanup
    std::fs::remove_dir_all("./example_flat_db")?;
    std::fs::remove_dir_all("./example_hnsw_db")?;
    
    Ok(())
}
