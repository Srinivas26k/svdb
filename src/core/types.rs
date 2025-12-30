//! Core type definitions for SrvDB v0.2.0
//! Major upgrade: Dynamic embedding dimensions (128-4096)

use serde::{Deserialize, Serialize};

/// Vector with dynamic dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub data: Vec<f32>,
}

impl Vector {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Normalize to unit length (L2 norm = 1.0)
    pub fn normalize(&mut self) {
        let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in self.data.iter_mut() {
                *x /= norm;
            }
        }
    }
}

/// Search result with ID, similarity score, and metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub metadata: Option<String>,
}

impl SearchResult {
    pub fn new(id: u64, score: f32, metadata: Option<String>) -> Self {
        Self {
            id,
            score,
            metadata,
        }
    }
}

/// Database configuration with dimension support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub dimension: usize,
    pub quantization: QuantizationConfig,
    pub index_type: IndexType,
}

impl DatabaseConfig {
    pub fn new(dimension: usize) -> anyhow::Result<Self> {
        if !(128..=4096).contains(&dimension) {
            anyhow::bail!("Dimension must be between 128 and 4096, got {}", dimension);
        }

        Ok(Self {
            dimension,
            quantization: QuantizationConfig::default(),
            index_type: IndexType::Flat,
        })
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.quantization.enabled {
            // Ensure dimension is divisible by M for PQ
            if !self.dimension.is_multiple_of(self.quantization.m) {
                anyhow::bail!(
                    "Dimension {} must be divisible by M={} for Product Quantization",
                    self.dimension,
                    self.quantization.m
                );
            }
        }
        Ok(())
    }

    /// Auto-tune PQ parameters based on dimension
    pub fn auto_tune_pq(&mut self) {
        // Heuristic: M = dim / 8 (8 dimensions per subquantizer)
        let optimal_m = (self.dimension / 8).clamp(16, 256);
        self.quantization.m = optimal_m;
        self.quantization.d_sub = self.dimension / optimal_m;
    }
}

/// Index type selection
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IndexType {
    Flat,             // Brute force O(n)
    HNSW,             // Graph-based O(log n)
    ScalarQuantized,  // SQ8 compression (4x)
    ProductQuantized, // PQ compression (32x)
    HNSWQuantized,    // HNSW + PQ hybrid
    IVF,              // Inverted File with HNSW Refinement
}

/// Quantization configuration with multiple modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub enabled: bool,
    pub mode: QuantizationMode,
    pub m: usize,     // Number of sub-quantizers (PQ only)
    pub k: usize,     // Centroids per sub-quantizer (PQ only)
    pub d_sub: usize, // Dimensions per sub-space (PQ only)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantizationMode {
    None,
    Scalar,  // SQ8: 4x compression, fast, 95%+ recall
    Product, // PQ: 32x compression, slower, 85-95% recall
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: QuantizationMode::None,
            m: 192, // Will be auto-tuned
            k: 256,
            d_sub: 8, // Will be auto-tuned
        }
    }
}

/// Internal vector file header (now with dimension info)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VectorHeader {
    pub magic: u32,
    pub version: u16,
    pub dimension: u16, // NEW: Support 128-4096
    pub count: u64,
    pub quantization_mode: u8, // 0=None, 1=Scalar, 2=Product
    pub index_type: u8,        // 0=Flat, 1=HNSW, etc.
    pub reserved: [u8; 6],
}

impl VectorHeader {
    pub const MAGIC: u32 = 0x53764442; // "SrvDB"
    pub const VERSION: u16 = 4; // Version 4: Dynamic dimensions
    pub const SIZE: usize = std::mem::size_of::<VectorHeader>();

    pub fn new(dimension: usize) -> anyhow::Result<Self> {
        if dimension > u16::MAX as usize {
            anyhow::bail!("Dimension {} exceeds maximum {}", dimension, u16::MAX);
        }

        Ok(Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            dimension: dimension as u16,
            count: 0,
            quantization_mode: 0,
            index_type: 0,
            reserved: [0; 6],
        })
    }

    pub fn new_quantized(dimension: usize, mode: u8) -> anyhow::Result<Self> {
        if dimension > u16::MAX as usize {
            anyhow::bail!("Dimension {} exceeds maximum {}", dimension, u16::MAX);
        }

        Ok(Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            dimension: dimension as u16,
            count: 0,
            quantization_mode: mode,
            index_type: 0,
            reserved: [0; 6],
        })
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.magic != Self::MAGIC {
            anyhow::bail!(
                "Invalid magic number: expected 0x{:08X}, got 0x{:08X}",
                Self::MAGIC,
                self.magic
            );
        }

        if self.version > Self::VERSION {
            anyhow::bail!("Unsupported version: {}", self.version);
        }

        if !(128..=4096).contains(&(self.dimension as usize)) {
            anyhow::bail!("Invalid dimension: {}", self.dimension);
        }

        Ok(())
    }
}

/// Scalar Quantization (SQ8) - Simple 4x compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    pub min_vals: Vec<f32>,
    pub max_vals: Vec<f32>,
    pub dimension: usize,
}

impl ScalarQuantizer {
    /// Train scalar quantizer by finding min/max per dimension
    pub fn train(training_data: &[Vec<f32>]) -> anyhow::Result<Self> {
        if training_data.is_empty() {
            anyhow::bail!("Training data cannot be empty");
        }

        let dimension = training_data[0].len();
        let mut min_vals = vec![f32::MAX; dimension];
        let mut max_vals = vec![f32::MIN; dimension];

        for vec in training_data {
            if vec.len() != dimension {
                anyhow::bail!("Inconsistent dimensions in training data");
            }

            for (i, &val) in vec.iter().enumerate() {
                min_vals[i] = min_vals[i].min(val);
                max_vals[i] = max_vals[i].max(val);
            }
        }

        Ok(Self {
            min_vals,
            max_vals,
            dimension,
        })
    }

    /// Encode float32 vector to uint8 (4x compression)
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        vector
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let min = self.min_vals[i];
                let max = self.max_vals[i];
                let range = max - min;

                if range < 1e-10 {
                    return 0u8;
                }

                let normalized = ((val - min) / range * 255.0).clamp(0.0, 255.0);
                normalized as u8
            })
            .collect()
    }

    /// Decode uint8 vector back to approximate float32
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        codes
            .iter()
            .enumerate()
            .map(|(i, &code)| {
                let min = self.min_vals[i];
                let max = self.max_vals[i];
                let range = max - min;

                min + (code as f32 / 255.0) * range
            })
            .collect()
    }

    /// Asymmetric distance: compare quantized vector to full-precision query
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let decoded = self.decode(codes);

        // Compute cosine similarity
        let dot: f32 = query.iter().zip(decoded.iter()).map(|(a, b)| a * b).sum();

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let decoded_norm: f32 = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();

        if query_norm < 1e-10 || decoded_norm < 1e-10 {
            return 0.0;
        }

        dot / (query_norm * decoded_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_dimensions() {
        let config = DatabaseConfig::new(384).unwrap();
        assert_eq!(config.dimension, 384);

        // Test invalid dimensions
        assert!(DatabaseConfig::new(64).is_err());
        assert!(DatabaseConfig::new(5000).is_err());
    }

    #[test]
    fn test_auto_tune_pq() {
        let mut config = DatabaseConfig::new(768).unwrap();
        config.auto_tune_pq();

        // Should be divisible
        assert_eq!(768 % config.quantization.m, 0);
        assert_eq!(config.quantization.d_sub, 768 / config.quantization.m);
    }

    #[test]
    fn test_scalar_quantizer() {
        let training: Vec<Vec<f32>> = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.1, 0.6, 0.9],
            vec![0.2, 0.4, 1.0],
        ];

        let sq = ScalarQuantizer::train(&training).unwrap();

        let test_vec = vec![0.15, 0.55, 0.95];
        let encoded = sq.encode(&test_vec);
        assert_eq!(encoded.len(), 3);

        let decoded = sq.decode(&encoded);

        // Check reconstruction error is small
        for (orig, dec) in test_vec.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.1);
        }
    }

    #[test]
    fn test_header_validation() {
        let header = VectorHeader::new(1536).unwrap();
        assert!(header.validate().is_ok());

        // Test invalid dimension
        let invalid_header = VectorHeader::new(50).unwrap();
        assert!(invalid_header.validate().is_err());
    }
}
