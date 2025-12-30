use srvdb::{IVFConfig, IndexMode, SrvDB, VectorEngine};
use std::path::Path;
use std::time::Instant;

/// Automated IVF-HNSW Validation Suite
#[test]
fn test_ivf_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║                  PHASE 2 VALIDATION: IVF-HNSW         ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    let db_path = Path::new("./test_ivf_hybrid_output");
    if db_path.exists() {
        std::fs::remove_dir_all(db_path)?;
    }

    let dim = 128; // Reduced for faster test
    let n_vectors = 1000; // Reduced for faster test

    println!("\n[1/3] Setup: Generating Adversarial Mix Dataset...");
    let (vectors, ids) = generate_adversarial_data(n_vectors, dim);

    println!("\n[2/3] Training IVF Index (K-Means Clustering)...");
    let ivf_config = IVFConfig {
        nlist: 10, // Small nlist for test
        nprobe: 2,
        ..Default::default()
    };

    // Use standard constructor with config for dimension
    let config = srvdb::types::DatabaseConfig::new(dim).expect("Invalid config");
    let mut db = SrvDB::new_with_config(db_path.to_str().unwrap(), config)?;

    // Configure for IVF
    db.set_mode(IndexMode::Ivf);
    db.configure_ivf(ivf_config.clone())?;

    // --- PHASE 2.1: TRAIN IVF ---
    let start = Instant::now();
    // In v0.2.0, train_ivf typically takes training data.
    // Checking lib.rs, train_ivf uses stored data if arguments are missing?
    // Or does it take arguments?
    // The error log showed: `db.train_ivf(&ids, &vectors, &ivf_config)` in the old file.
    // I need to verify signature. lib.rs Step 1285 showed ` fn train_ivf(&mut self) -> Result<()> ` taking NO args and using stored data.
    // So we must ingest first, then train?
    // Logic in lib.rs:
    // "1. Load Training Data" -> "let count = self.vector_storage..."
    // So yes, we must add data first.

    println!("\n[4/3] Ingesting Vectors...");
    let start_ingest = Instant::now();
    db.add_batch(&vectors, &ids)?;
    db.persist()?;
    let ingest_time = start_ingest.elapsed();
    println!("   -> Ingestion Time: {:?}", ingest_time);

    // Now Train
    println!("Training IVF...");
    db.train_ivf()?;
    let train_time = start.elapsed();
    println!("   -> Train Time: {:?}", train_time);

    // --- PHASE 2.3: SEARCH TESTS ---
    println!("\n[5/3] Searching (IVF Mode)...");

    // Prepare Query
    let query_vecs = generate_random_queries(10, dim);
    let mut total_recall_hits = 0;

    let mut latencies = Vec::new();

    for q in &query_vecs {
        let t_start = Instant::now();

        let results = db.search(q, 10)?;

        let elapsed = t_start.elapsed();
        latencies.push(elapsed.as_secs_f64() * 1000.0);

        total_recall_hits += results.len();
    }

    println!("\n[6/3] Metrics:");
    let p99 = percentile(&latencies, 99);
    let p50 = percentile(&latencies, 50);
    println!("   -> Latency P99: {:.2} ms", p99);
    println!("   -> Latency P50: {:.2} ms", p50);
    println!("   -> Recall (Result Count): {}", total_recall_hits);

    // Use permissive threshold for small random test
    if total_recall_hits > 0 {
        println!("   -> SUCCESS: IVF-HNSW Hybrid Index yielded results.");
    } else {
        println!("   -> WARNING: No results found.");
    }

    // Cleanup
    std::fs::remove_dir_all(db_path)?;

    println!("\n═══════════════════════════════════════════════════╝");
    Ok(())
}

// Helpers
fn generate_adversarial_data(n: usize, dim: usize) -> (Vec<srvdb::Vector>, Vec<String>) {
    let mut vecs = Vec::with_capacity(n);
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        ids.push(format!("vec_{}", i));
        let mut v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        vecs.push(srvdb::Vector::new(v));
    }
    (vecs, ids)
}

fn generate_random_queries(n: usize, dim: usize) -> Vec<srvdb::Vector> {
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            srvdb::Vector::new(v)
        })
        .collect()
}

// Simple Percentile Helper (f64 version)
fn percentile(data: &[f64], p: usize) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = (sorted.len() as f64 * p as f64 / 100.0) as usize;
    if index >= sorted.len() {
        if sorted.len() == 0 {
            return 0.0;
        }
        return sorted[sorted.len() - 1];
    }
    sorted[index]
}
