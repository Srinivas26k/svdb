//! # SrvDB Auto-Tuning Orchestrator
//!
//! Executes the adaptive strategy by inspecting the runtime environment
//! and updating the database configuration dynamically.

use crate::strategy::{self, AutoTunerConfig, IndexMode};
use crate::{SrvDB, VectorEngine};

/// Apply the auto-tuning strategy to the database instance.
///
/// This function:
/// 1. Detects available RAM.
/// 2. Checks current dataset size.
/// 3. Asks the Strategy Engine for the best mode.
/// 4. Updates the database configuration.
pub fn apply_auto_strategy(db: &mut SrvDB) {
    // 1. Detect Hardware
    let ram_gb = detect_available_ram_gb();
    let config = AutoTunerConfig::new(ram_gb);

    // 2. Inspect Data
    let count = db.count() as usize;

    // 3. Decide Mode
    let best_mode = strategy::decide_mode(&config, count);

    // 4. Execute Strategy
    // We only switch if the mode is different.
    // Note: We need a way to check current high-level mode.
    // For now, we just enforce the decision.
    println!(
        "SrvDB Auto-Tuner: System RAM {} GB | Vectors {} -> Selected Mode: {:?}",
        ram_gb, count, best_mode
    );

    match best_mode {
        IndexMode::Flat => {
            db.index_type = crate::types::IndexType::Flat;
            db.config.enabled = false;
        }
        IndexMode::Hnsw => {
            db.index_type = crate::types::IndexType::HNSW;
            db.config.enabled = false;
            // TODO: Auto-tune HNSW params based on latency budget
        }
        IndexMode::Sq8 => {
            db.index_type = crate::types::IndexType::ScalarQuantized;
            db.config.enabled = true;
            db.config.mode = crate::types::QuantizationMode::Scalar;
        }
        IndexMode::Auto => {
            // Should not happen as return from decide_mode
        }
        IndexMode::Ivf => {
            // Not auto-selected yet
        }
    }
}

/// Detect total system RAM in GB.
///
/// Uses /proc/meminfo on Linux for zero-dependency detection.
/// Fallback to generic safe default (4GB) on other OS.
fn detect_available_ram_gb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    // Line format: "MemTotal:        16316252 kB"
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb as f64 / 1024.0 / 1024.0;
                        }
                    }
                }
            }
        }
    }

    // Fallback for non-Linux or read failure
    4.0
}

/// Check if we need to migrate strategy based on new data.
///
/// Called during `persist()`. If the Auto-Tuner decides a new mode is better
/// than the current one (e.g. Flat -> HNSW), this function performs the migration.
pub fn check_and_migrate(db: &mut SrvDB) -> anyhow::Result<()> {
    // Only migrate if we are in Auto mode
    if db.current_mode != IndexMode::Auto {
        return Ok(());
    }

    let ram_gb = detect_available_ram_gb();
    let config = AutoTunerConfig::new(ram_gb);
    let count = db.count() as usize;
    let new_mode = strategy::decide_mode(&config, count);

    // If decision matches current internal state, do nothing
    let current_internal_mode = match db.index_type {
        crate::types::IndexType::Flat => IndexMode::Flat,
        crate::types::IndexType::HNSW => IndexMode::Hnsw,
        crate::types::IndexType::ScalarQuantized => IndexMode::Sq8,
        crate::types::IndexType::IVF => IndexMode::Ivf,
        _ => IndexMode::Flat, // Ignore PQ for now
    };

    if new_mode == current_internal_mode {
        return Ok(());
    }

    println!(
        "SrvDB Adaptive Core: Migrating from {:?} to {:?}",
        current_internal_mode, new_mode
    );

    match new_mode {
        IndexMode::Hnsw => migrate_to_hnsw(db)?,
        IndexMode::Sq8 => migrate_to_sq8(db)?,
        IndexMode::Flat => {} // Downgrade rarely happens/supported
        IndexMode::Auto => {}
        IndexMode::Ivf => {
            // IVF migration complicated (needs training).
            // For now we don't auto-migrate TO IVF, only manual set.
        }
    }

    Ok(())
}

fn migrate_to_hnsw(db: &mut SrvDB) -> anyhow::Result<()> {
    use crate::hnsw::{HNSWConfig, HNSWIndex};

    // 1. Initialize HNSW
    let hnsw_config = HNSWConfig::default();
    let index = HNSWIndex::new(hnsw_config.clone());

    // 2. Build Graph (Time consuming!)
    println!("  -> Building HNSW Graph for {} vectors...", db.count());

    if let Some(ref mut vstorage) = db.vector_storage {
        vstorage.flush()?; // Ensure data is on disk for reading
    }

    if let Some(ref vstorage) = db.vector_storage {
        // We need a distance function that references vstorage.
        // Since we can't borrow db immutably inside the loop while mutating,
        // we might ideally use the `add` logic, but `add` is incremental.
        // HNSW persistence needs to be robust.
        // For now, we unfortunately have to clone vectors or rely on `vstorage` read access
        // distinct from `index` write access.
        // HNSW insert takes a closure.

        let _count = vstorage.count();
        // Since HNSW insert needs random access to vectors, and vstorage is on disk/mmap,
        // we can share a reference if we are careful.
        // But `migrate_to_hnsw` takes `&mut SrvDB`.

        // Workaround: We define distance fn using a read-only view if possible,
        // or effectively we just assume we can read from vstorage.
        // Actually, `vstorage.get(id)` is fast.

        // We need to pass the closure to `index.insert`.
        // The closure needs access to `vstorage`.
        // Since `index` is separate from `vstorage` in `SrvDB`, we can split borrows?
        // No, `db` is borrowed mutably.
        // We can temporarily take `vector_storage` out of `db`?
    } else {
        anyhow::bail!("Cannot migrate to HNSW: No vector storage found.");
    }

    // Splitting borrows for migration
    let vstorage = db.vector_storage.as_ref().unwrap();

    // Define distance function
    let dist_fn = |a: u64, b: u64| -> f32 {
        if let (Some(va), Some(vb)) = (vstorage.get(a), vstorage.get(b)) {
            // naive cosine distance
            let dot: f32 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = va.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = vb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a * norm_b == 0.0 {
                1.0
            } else {
                1.0 - (dot / (norm_a * norm_b))
            }
        } else {
            1.0 // Max distance
        }
    };

    for i in 0..db.count() {
        index.insert(i, &dist_fn)?;
    }

    // 3. Update DB State
    db.hnsw_index = Some(index);
    db.hnsw_config = Some(hnsw_config);
    db.index_type = crate::types::IndexType::HNSW;

    println!("  -> HNSW Migration Complete.");
    Ok(())
}

fn migrate_to_sq8(db: &mut SrvDB) -> anyhow::Result<()> {
    use crate::types::{IndexType, QuantizationMode};
    use crate::ScalarQuantizedStorage;

    // 1. Read all vectors (or sample) for training
    // 2. Train Quantizer
    // 3. Create ScalarStorage
    // 4. Ingest all
    // 5. Drop VectorStorage

    println!("  -> Compressing to SQ8 (Scalar Quantization)...");

    if db.vector_storage.is_none() {
        return Ok(());
    }

    // Ensure vectors are flushed to disk so mmap can read them
    db.vector_storage.as_mut().unwrap().flush()?;

    // We need to clone vectors because we'll be replacing the storage
    // And ScalarQuantizedStorage::new_with_training consumes training vectors

    let vstorage = db.vector_storage.as_ref().unwrap();
    let count = vstorage.count();
    let dim = if count > 0 {
        vstorage.get(0).unwrap().len()
    } else {
        1536
    };

    // Collect training data (Up to 5000 vectors)
    let sample_size = std::cmp::min(count as usize, 5000);
    let mut training_data = Vec::with_capacity(sample_size);
    for i in 0..sample_size {
        if let Some(vec) = vstorage.get(i as u64) {
            training_data.push(vec.to_vec());
        }
    }

    // Create Storage
    // We use `db.path` which is now available
    // Note: This creates "scalar_quantized.bin" and "scalar_quantizer.json" in db.path
    let path_str = db.path.to_str().ok_or(anyhow::anyhow!("Invalid DB path"))?;

    let mut scalar_storage =
        ScalarQuantizedStorage::new_with_training(path_str, dim, &training_data)?;

    // 4. Ingest All (Convert remaining if any, or just all if we trained on subset?)
    // ScalarQuantizedStorage logic usually trains on provided vectors but we need to Add *all* vectors to it.
    // The `new_with_training` creates the quantizer but might not insert the vectors?
    // Let's check `ScalarQuantizedStorage::new_with_training` implementation.
    // It usually returns a storage ready to accept vectors.
    // Wait, check `new_scalar_quantized` in lib.rs? It just returns the storage.
    // We need to APPEND all vectors from `vector_storage` to `scalar_storage`.

    println!("  -> Quantizing {} vectors...", count);
    for i in 0..count {
        if let Some(vec) = vstorage.get(i) {
            scalar_storage.append(vec)?;
        }
    }

    // 5. Update DB State
    scalar_storage.flush()?;
    db.scalar_storage = Some(scalar_storage);
    db.vector_storage = None; // Drop full precision storage to save RAM/Disk
                              // Ideally we should delete `vectors.bin` but that's destructive.
                              // For now we just detach it.

    db.config.enabled = true;
    db.config.mode = QuantizationMode::Scalar;
    db.index_type = IndexType::ScalarQuantized;

    println!("  -> SQ8 Migration Complete.");
    Ok(())
}
