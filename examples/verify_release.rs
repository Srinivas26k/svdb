use srvdb::types::DatabaseConfig;
use srvdb::{IndexMode, SrvDB, Vector, VectorEngine};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║       SrvDB v0.2.0 Release Verification Script        ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    let dim = 128; // Use 128 for speed
    let n_vecs = 10_000;

    // -----------------------------------------------------------------------
    // TEST 1: Flat Index (Baseline)
    // -----------------------------------------------------------------------
    println!("\n[1/2] Verifying Flat Index (Baseline)...");
    let path_flat = "./verify_flat";
    if Path::new(path_flat).exists() {
        std::fs::remove_dir_all(path_flat)?;
    }

    let config = DatabaseConfig::new(dim)?;
    let mut db = SrvDB::new_with_config(path_flat, config)?;
    db.set_mode(IndexMode::Flat);

    let (vecs, ids) = generate_data(n_vecs, dim);

    let start = Instant::now();
    db.add_batch(&vecs, &ids)?;
    db.persist()?; // Flush to disk/mmap
    println!(
        "   -> Ingested {} vectors in {:.2}ms",
        n_vecs,
        start.elapsed().as_millis()
    );

    let query = &vecs[0];
    let results = db.search(query, 5)?;
    println!(
        "   -> Search results for vector 0: {:?}",
        results.iter().map(|r| r.id).collect::<Vec<_>>()
    );
    assert_eq!(results[0].id, 0, "Self-match should be ID 0");
    println!("   -> SUCCESS: Flat Index workings.");

    // -----------------------------------------------------------------------
    // TEST 2: SQ8 Index (Scalar Quantization)
    // -----------------------------------------------------------------------
    println!("\n[2/2] Verifying SQ8 (Scalar Quantization)...");
    let path_sq = "./verify_sq8";
    if Path::new(path_sq).exists() {
        std::fs::remove_dir_all(path_sq)?;
    }

    let mut config_sq = DatabaseConfig::new(dim)?;
    // In actual usage, user might set this via set_mode(Sq8) or training
    // But let's try new_scalar_quantized constructor which is the intended public API for SQ8
    // But new_scalar_quantized takes training vectors immediately.

    // Extract raw data for training
    let raw_vecs: Vec<Vec<f32>> = vecs.iter().map(|v| v.data.clone()).collect();

    let start_train = Instant::now();
    // Re-create DB for SQ8
    let mut db_sq = SrvDB::new_scalar_quantized(path_sq, dim, &raw_vecs)?;
    // Note: new_scalar_quantized trains AND initializes storage
    // But it does NOT ingest the vectors (it just uses them for training).
    // So we must add them.
    println!(
        "   -> Trained SQ8 in {:.2}ms",
        start_train.elapsed().as_millis()
    );

    let start_add = Instant::now();
    db_sq.add_batch(&vecs, &ids)?;
    db_sq.persist()?; // Flush
    println!(
        "   -> Ingested (Quantized) in {:.2}ms",
        start_add.elapsed().as_millis()
    );

    let results_sq = db_sq.search(query, 5)?;
    println!(
        "   -> Search results (SQ8): {:?}",
        results_sq.iter().map(|r| r.id).collect::<Vec<_>>()
    );
    assert_eq!(results_sq[0].id, 0, "Self-match should be ID 0 even in SQ8");
    println!("   -> SUCCESS: SQ8 Index workings.");

    // Cleanup
    std::fs::remove_dir_all(path_flat)?;
    std::fs::remove_dir_all(path_sq)?;

    println!("\n═══════════════════════════════════════════════════╝");
    println!("All verification tests PASSED.");
    Ok(())
}

fn generate_data(n: usize, dim: usize) -> (Vec<Vector>, Vec<String>) {
    let mut vecs = Vec::with_capacity(n);
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        ids.push(format!("id_{}", i));
        let mut v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        vecs.push(Vector::new(v));
    }
    (vecs, ids)
}
