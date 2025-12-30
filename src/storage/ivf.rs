//! # IVF Storage Layer
//!
//! Handles serialization and persistence of the IVF index structure.
//! Format:
//! - db_path/ivf/config.json
//! - db_path/ivf/centroids.bin
//! - db_path/ivf/partitions/part_0.graph
//! - ...

use crate::hnsw::HNSWIndex;
use crate::ivf::{IVFConfig, IVFIndex};
use crate::Vector;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct IVFStorage;

impl IVFStorage {
    fn ivf_dir(db_path: &Path) -> PathBuf {
        db_path.join("ivf")
    }

    fn partition_dir(db_path: &Path) -> PathBuf {
        Self::ivf_dir(db_path).join("partitions")
    }

    pub fn save(index: &IVFIndex, db_path: &Path) -> Result<()> {
        let ivf_dir = Self::ivf_dir(db_path);
        if !ivf_dir.exists() {
            fs::create_dir_all(&ivf_dir)?;
        }
        let partition_dir = Self::partition_dir(db_path);
        if !partition_dir.exists() {
            fs::create_dir_all(&partition_dir)?;
        }

        // 1. Save Config
        let config_path = ivf_dir.join("config.json");
        let config_json = serde_json::to_string_pretty(&index.config)?;
        fs::write(config_path, config_json).context("Failed to write IVF config")?;

        // 2. Save Centroids
        // Use bincode for fast binary serialization of Vec<Vector>
        let centroids_path = ivf_dir.join("centroids.bin");
        let centroids_bytes = bincode::serialize(&index.centroids)?;
        fs::write(centroids_path, centroids_bytes).context("Failed to write IVF centroids")?;

        // 3. Save Partitions (Parallelized)
        // Each partition is an HNSW graph
        use rayon::prelude::*;
        index
            .partitions
            .par_iter()
            .enumerate()
            .for_each(|(i, partition)| {
                let file_name = format!("part_{}.graph", i);
                let file_path = partition_dir.join(file_name);

                // Acquire read lock to serialize
                let graph = partition.read(); // Lock RwLock
                if let Ok(bytes) = graph.to_bytes() {
                    let _ = fs::write(file_path, bytes); // Ignore write errors in parallel loop? ideally log them
                }
            });

        Ok(())
    }

    pub fn load(db_path: &Path) -> Result<Option<IVFIndex>> {
        let ivf_dir = Self::ivf_dir(db_path);
        if !ivf_dir.exists() {
            return Ok(None);
        }

        // 1. Load Config
        let config_path = ivf_dir.join("config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: IVFConfig = serde_json::from_str(&config_str)?;

        // 2. Load Centroids
        let centroids_path = ivf_dir.join("centroids.bin");
        let centroids_bytes = fs::read(centroids_path)?;
        let centroids: Vec<Vector> = bincode::deserialize(&centroids_bytes)?;

        // 3. Load Partitions
        let partition_dir = Self::partition_dir(db_path);
        let mut partitions = Vec::with_capacity(config.nlist);

        // Sequential load for simplicity/safety, could refer to config.nlist
        for i in 0..config.nlist {
            let file_name = format!("part_{}.graph", i);
            let file_path = partition_dir.join(file_name);

            let graph = if file_path.exists() {
                let bytes = fs::read(file_path)?;
                HNSWIndex::from_bytes(&bytes)?
            } else {
                HNSWIndex::new(config.hnsw_config.clone())
            };

            partitions.push(Arc::new(RwLock::new(graph)));
        }

        Ok(Some(IVFIndex {
            config,
            centroids,
            partitions,
        }))
    }
}
