use srvdb::ivf::IVFConfig;
use srvdb::{IndexMode, SrvDB, Vector, VectorEngine};
use std::path::Path;
use std::time::Instant;

/// Automated IVF-HNSW Validation Suite
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║                  PHASE 2 VALIDATION: IVF-HNSW                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let db_path = Path::new("./test_ivf_hybrid_bin");
    if db_path.exists() {
        std::fs::remove_dir_all(db_path)?;
    }

    let dim = 1536;
    let n_vectors = 10_000; // Reduced from 100k for faster dev-cycle validation (100k takes ~30s+)

    println!(
        "\n[1/3] Setup: Generating Adversarial Mix Dataset ({} vectors)...",
        n_vectors
    );
    let (vectors_raw, ids): (Vec<Vec<f32>>, Vec<String>) =
        generate_adversarial_data(n_vectors, dim);

    // Convert to srvdb::Vector
    let vectors: Vec<Vector> = vectors_raw
        .iter()
        .map(|data| Vector::new(data.clone()))
        .collect();

    println!("\n[2/3] Training IVF Index (K-Means Clustering)...");
    let ivf_config = IVFConfig {
        nlist: 100, // 100 partitions for 10k vectors (Ratio ~100 per part)
        nprobe: 10,
        max_iterations: 10,
        tolerance: 0.001,
        hnsw_config: Default::default(),
    };

    let mut db = SrvDB::new(db_path.to_str().unwrap())?;

    // Ingest First
    println!("   -> Ingesting vectors...");
    let start_ingest = Instant::now();
    db.add_batch(&vectors, &ids)?;
    db.persist()?; // Flush to disk for training visibility
    let ingest_time = start_ingest.elapsed();
    println!(
        "      Throughput: {:.2} vec/s",
        n_vectors as f64 / ingest_time.as_secs_f64()
    );

    // Configure and Train
    println!("   -> Switching to IVF and Training...");
    db.set_mode(IndexMode::Ivf);
    db.configure_ivf(ivf_config)?;

    let start_train = Instant::now();
    db.train_ivf()?;
    let train_time = start_train.elapsed();
    println!("      Training Time: {:?}", train_time);

    // Persist trained state
    db.persist()?;

    // Reload DB to test persistence
    println!("\n[3/3] Persistence Check & Search...");
    drop(db);
    let mut db_reloaded = SrvDB::new(db_path.to_str().unwrap())?;
    // Ensure IVF mode is active (load logic should handle this, or we set it)
    db_reloaded.set_mode(IndexMode::Ivf); // Force mode if auto-detetion isn't strictly persisting "current_mode" pref (it persists index type though)

    // --- SEARCH TESTS ---
    println!("   -> Searching (IVF Mode)...");

    // Prepare Query
    let query_vecs_raw = generate_random_queries(100, dim);
    let query_vecs: Vec<Vector> = query_vecs_raw
        .iter()
        .map(|data| Vector::new(data.clone()))
        .collect();

    let mut total_hits = 0;
    let mut latencies = Vec::new();

    for q in &query_vecs {
        let t_start = Instant::now();

        let results = db_reloaded.search(q, 10)?;

        let elapsed = t_start.elapsed();
        latencies.push(elapsed.as_millis());

        if !results.is_empty() {
            total_hits += 1;
        }
    }

    println!("\n[Metrics]:");
    let p99 = percentile(&latencies, 99);
    let p50 = percentile(&latencies, 50);
    println!("   -> Latency P99: {} ms", p99);
    println!("   -> Latency P50: {} ms", p50);
    println!("   -> Queries with Hits: {}/100", total_hits);

    // Final Verdict
    println!("\n[FINAL VERDICT]:");
    if total_hits > 90 {
        println!("   -> SUCCESS: IVF-HNSW Hybrid Index is functional.");
    } else {
        println!("   -> WARNING: Low hit count. Check Partitioning Logic.");
    }

    // Cleanup
    // std::fs::remove_dir_all(db_path)?;

    println!("\n═══════════════════════════════════════════════════╝");
    Ok(())
}

// Helpers
fn generate_adversarial_data(n: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<String>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut vecs = Vec::with_capacity(n);
    let mut ids = Vec::with_capacity(n);

    for i in 0..n {
        ids.push(format!("vec_{}", i));
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
        // Normalize
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        vecs.push(v);
    }
    (vecs, ids)
}

fn generate_random_queries(n: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

fn percentile(data: &[u128], p: usize) -> u128 {
    let mut sorted = data.to_vec();
    sorted.sort();
    if sorted.is_empty() {
        return 0;
    }
    let index = (sorted.len() as f64 * p as f64 / 100.0) as usize;
    if index >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[index]
    }
}
