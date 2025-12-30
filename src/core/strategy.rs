//! # SrvDB Adaptive Strategy Engine
//!
//! The "Brain" of SrvDB. This module defines the decision logic for selecting the optimal
//! indexing strategy based on hardware constraints and data characteristics.
//!
//! "Choice is the enemy of speed." - We decide so the developer doesn't have to.

use serde::{Deserialize, Serialize};

/// High-level indexing mode selected by the user or the Auto-Tuner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IndexMode {
    /// Exact search (Brute Force). O(n).
    /// Best for small datasets (<50k) where 100% recall is critical.
    #[default]
    Flat,

    /// Approximate search (HNSW). O(log n).
    /// Best for real-time applications requiring sub-10ms latency.
    Hnsw,

    /// Scalar Quantization (SQ8). 4x Compression.
    /// Best for memory-constrained environments (e.g., Laptops, Edge).
    Sq8,

    /// Inverted File with HNSW Refinement (IVF-HNSW).
    /// Best for massive datasets (>1M vectors).
    Ivf,

    /// The "Magic" mode.
    /// SrvDB analyzes resources and data size to select the best strategy at runtime.
    Auto,
}

/// Configuration for the Auto-Tuning Engine.
/// Defines thresholds for switching strategies.
#[derive(Debug, Clone)]
pub struct AutoTunerConfig {
    /// Total available system RAM in GB.
    pub available_ram_gb: f64,

    /// Dataset size threshold to switch from Flat to HNSW/SQ8.
    /// Default: 50,000 vectors.
    pub dataset_size_threshold: usize,

    /// Latency budget in milliseconds (P99).
    /// If projected latency > this, we prioritize HNSW.
    pub latency_budget_ms: f64,
}

impl Default for AutoTunerConfig {
    fn default() -> Self {
        Self {
            available_ram_gb: 4.0, // Conservative default
            dataset_size_threshold: 50_000,
            latency_budget_ms: 50.0,
        }
    }
}

impl AutoTunerConfig {
    /// Detect system hardware and return a config.
    /// Note: Actual detection happens in `auto_tune.rs` to keep this pure logic.
    pub fn new(ram_gb: f64) -> Self {
        Self {
            available_ram_gb: ram_gb,
            ..Default::default()
        }
    }
}

/// The Decision Engine.
///
/// Analyzes constraints and returns the optimal IndexMode.
///
/// # Logic Flow
/// 1. **Safety First**: If RAM is critically low (<2GB) and dataset is large, force SQ8.
/// 2. **Accuracy First**: If dataset is small (<50k), use Flat for 100% recall.
/// 3. **Speed/Scale**: If dataset is large (>50k):
///    - If plenty of RAM (>8GB), use HNSW for speed.
///    - If constrained RAM (<8GB), use SQ8 for efficiency.
pub fn decide_mode(config: &AutoTunerConfig, dataset_size: usize) -> IndexMode {
    // 1. Accuracy Mode (Small Data)
    // For small datasets, Flat is fast enough and provides perfect recall.
    if dataset_size < config.dataset_size_threshold {
        return IndexMode::Flat;
    }

    // 2. Safe Mode (Low RAM)
    // If we have very limited memory, we prioritized compression (SQ8) to avoid OOM.
    // SQ8 uses ~25-30% of the RAM of Flat/HNSW.
    if config.available_ram_gb < 4.0 {
        return IndexMode::Sq8;
    }

    // 3. Efficiency Mode (Medium RAM)
    // Between 4GB and 16GB, it's a trade-off.
    // If the dataset is massive (e.g. >1M), HNSW memory overhead is too high.
    // For now, let's say if we have < 8GB, we still prefer SQ8 to be safe for other apps.
    if config.available_ram_gb < 8.0 {
        return IndexMode::Sq8;
    }

    // 4. Speed Mode (High RAM)
    // We have resources. Prioritize sub-millisecond latency.
    IndexMode::Hnsw
}

// TODO: Add IVF support for v0.3.0 logic.
