//! Product Quantization (PQ) for vector compression
//!
//! Implements FAISS-style Product Quantization:
//! - ~32x memory compression (float32 -> uint8)
//! - Dynamic dimension support
//! - Asymmetric Distance Computation (ADC) for fast search
//! - K-means clustering for codebook training

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32;

/// Constants
pub const K: usize = 256; // Centroids per sub-quantizer (fits in u8)

/// Quantized vector representation (M bytes)
pub type QuantizedVector = Vec<u8>;

/// Distance lookup table for Asymmetric Distance Computation
#[derive(Debug, Clone)]
pub struct DistanceTable {
    pub tables: Vec<Vec<f32>>, // M tables, each with K distances
}

/// Codebook for a single sub-quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    pub centroids: Vec<Vec<f32>>, // K centroids, each d_sub dim
}

/// Normalize a vector to unit length (L2 norm = 1.0)
#[inline]
fn normalize_vector(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

impl Codebook {
    /// Create new codebook with k-means clustering
    pub fn train(vectors: &[Vec<f32>], k: usize, max_iter: usize) -> Result<Self> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot train on empty dataset");
        }
        
        let d_sub = vectors[0].len();

        // Use vectors as-is (don't normalize sub-vectors)
        // Cosine similarity will be computed via magnitude normalization in distance calculation
        let mut centroids = Self::kmeans_plusplus_init(vectors, k, d_sub);

        for _ in 0..max_iter {
            // Assignment step
            let assignments: Vec<usize> = vectors
                .par_iter()
                .map(|v| Self::nearest_centroid(v, &centroids))
                .collect();

            // Update step
            let mut new_centroids = vec![vec![0.0f32; d_sub]; k];
            let mut counts = vec![0usize; k];

            for (vec, &cluster_id) in vectors.iter().zip(assignments.iter()) {
                for i in 0..d_sub {
                    new_centroids[cluster_id][i] += vec[i];
                }
                counts[cluster_id] += 1;
            }

            // Average to get new centroids
            let mut converged = true;
            for i in 0..k {
                if counts[i] > 0 {
                    for val in new_centroids[i].iter_mut() {
                        *val /= counts[i] as f32;
                    }
                    // Normalize centroids to unit length for proper cosine similarity
                    normalize_vector(&mut new_centroids[i]);

                    // Check convergence
                    if Self::euclidean_distance(&centroids[i], &new_centroids[i]) > 1e-6 {
                        converged = false;
                    }
                } else {
                    // Reinitialize empty clusters
                    let random_idx = rand::random::<usize>() % vectors.len();
                    new_centroids[i] = vectors[random_idx].clone();
                    converged = false;
                }
            }

            centroids = new_centroids;

            if converged {
                break;
            }
        }

        Ok(Self { centroids })
    }

    /// K-means++ initialization for better clustering
    fn kmeans_plusplus_init(vectors: &[Vec<f32>], k: usize, d_sub: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // Pick first centroid randomly
        let first_idx = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first_idx].clone());

        // Pick remaining centroids with probability proportional to distance squared
        for _ in 1..k {
            let distances: Vec<f32> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| Self::euclidean_distance(v, c))
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                })
                .map(|d| d * d)
                .collect();

            let total: f32 = distances.iter().sum();
            
            if total <= 1e-10 {
                // Determine random remaining
                 let random_idx = rng.gen_range(0..vectors.len());
                centroids.push(vectors[random_idx].clone());
                continue;
            }

            let mut target = rng.gen::<f32>() * total;

            for (i, &dist) in distances.iter().enumerate() {
                target -= dist;
                if target <= 0.0 {
                    centroids.push(vectors[i].clone());
                    break;
                }
            }
        }

        centroids
    }

    /// Find nearest centroid index using cosine similarity
    #[inline]
    fn nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> usize {
        // Calculate magnitude of input vector
        let vec_magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        if vec_magnitude < 1e-10 {
            return 0; // Default to first centroid if zero vector
        }

        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                // Compute centroid magnitude
                let c_magnitude: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();

                if c_magnitude < 1e-10 {
                    return (i, -1.0); // Very low similarity for zero centroid
                }

                // Compute cosine similarity
                let dot: f32 = vec.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                let cosine = dot / (vec_magnitude * c_magnitude);
                (i, cosine)
            })
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Euclidean distance between two vectors
    #[inline]
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Encode a sub-vector to its nearest centroid index
    #[inline]
    pub fn encode(&self, sub_vec: &[f32]) -> u8 {
        // Use sub-vector as-is, distance computation handles normalization
        Self::nearest_centroid(sub_vec, &self.centroids) as u8
    }

    /// Compute distances from query sub-vector to all centroids
    /// Centroids are normalized, query sub-vector is not
    /// Returns NEGATIVE cosine similarity (for minimization-based search)
    #[inline]
    pub fn compute_distances(&self, query_sub: &[f32]) -> Vec<f32> {
        // Calculate magnitude of query sub-vector
        let query_magnitude: f32 = query_sub.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut distances = vec![0.0f32; K];

        // If query magnitude is near zero, return zeros
        if query_magnitude < 1e-10 {
            return distances;
        }

        for (i, centroid) in self.centroids.iter().enumerate() {
            // Compute dot product (centroid is normalized to unit length)
            let dot_product: f32 = query_sub
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| a * b)
                .sum();

            // Divide by query magnitude to get cosine (centroid magnitude is 1.0)
            let cosine = dot_product / query_magnitude;

            // Negate because we minimize distance but want to maximize similarity
            distances[i] = -cosine;
        }
        distances
    }
}

/// Product Quantizer - splits vectors into M sub-vectors and quantizes each
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    pub m: usize,
    pub k: usize,
    pub d_sub: usize,
    pub codebooks: Vec<Codebook>,
}

impl ProductQuantizer {
    /// Train a new product quantizer on training data
    pub fn train(training_data: &[Vec<f32>], m: usize) -> Result<Self> {
        if training_data.is_empty() {
            anyhow::bail!("Training data cannot be empty");
        }
        
        let dim = training_data[0].len();
        if dim % m != 0 {
             anyhow::bail!("Dimension {} must be divisible by m={}", dim, m);
        }
        let d_sub = dim / m;

        println!(
            "Training Product Quantizer with M={}, K={}, D_sub={}",
            m, K, d_sub
        );

        // Train codebook for each sub-quantizer
        let codebooks: Result<Vec<Codebook>> = (0..m)
            .into_par_iter()
            .map(|m_idx| {
                println!("Training sub-quantizer {}/{}", m_idx + 1, m);

                // Extract sub-vectors for this sub-quantizer
                let sub_vectors: Vec<Vec<f32>> = training_data
                    .iter()
                    .map(|vec| {
                        let start = m_idx * d_sub;
                        let end = start + d_sub;
                        vec[start..end].to_vec()
                    })
                    .collect();

                // Train codebook for this sub-space (normalization happens inside)
                Codebook::train(&sub_vectors, K, 20)
                    .context(format!("Failed to train codebook {}", m_idx))
            })
            .collect();

        Ok(Self {
            m,
            k: K,
            d_sub,
            codebooks: codebooks?,
        })
    }

    /// Quantize a full vector into quantized representation
    #[inline]
    pub fn quantize(&self, vector: &[f32]) -> QuantizedVector {
        let mut quantized = vec![0u8; self.m];

        for (m_idx, codebook) in self.codebooks.iter().enumerate() {
            let start = m_idx * self.d_sub;
            let end = start + self.d_sub;
            let sub_vec = &vector[start..end];

            quantized[m_idx] = codebook.encode(sub_vec);
        }

        quantized
    }
    
    // Legacy alias
    pub fn encode(&self, vector: &[f32]) -> QuantizedVector {
        self.quantize(vector)
    }

    /// Compute distance table for asymmetric distance computation
    pub fn compute_distance_table(&self, query: &[f32]) -> DistanceTable {
        let tables: Vec<Vec<f32>> = self
            .codebooks
            .iter()
            .enumerate()
            .map(|(m_idx, codebook)| {
                let start = m_idx * self.d_sub;
                let end = start + self.d_sub;
                let query_sub = &query[start..end];

                codebook.compute_distances(query_sub)
            })
            .collect();

        DistanceTable { tables }
    }

    /// Asymmetric distance computation using precomputed distance table
    /// Returns approximate Cosine Similarity (averaged across all sub-spaces)
    #[inline]
    pub fn asymmetric_distance(&self, qvec: &[u8], dtable: &DistanceTable) -> f32 {
        let mut sum_neg_cosine = 0.0f32;
        for (m_idx, &code) in qvec.iter().enumerate() {
            sum_neg_cosine += dtable.tables[m_idx][code as usize];
        }
        // Negate to get positive, then divide by M to get average cosine similarity
        -sum_neg_cosine / (self.m as f32)
    }

    /// Save codebooks to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).context("Failed to serialize quantizer")
    }

    /// Load codebooks from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).context("Failed to deserialize quantizer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_training() {
        let d_sub = 8;
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| {
                let mut v = vec![0.0f32; d_sub];
                for j in 0..d_sub {
                    v[j] = (i as f32 / 1000.0) + (j as f32 / 10.0);
                }
                v
            })
            .collect();

        let codebook = Codebook::train(&vectors, 16, 20).unwrap();
        assert_eq!(codebook.centroids.len(), 16);
    }

    #[test]
    fn test_pq_encode_decode() {
        let dim = 128;
        let m = 16;
        // Create simple training data
        let training: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                for j in 0..dim {
                    v[j] = (i as f32 / 100.0) + (j as f32 / 1536.0);
                }
                v
            })
            .collect();

        let pq = ProductQuantizer::train(&training, m).unwrap();

        let test_vec = vec![0.5f32; dim];
        let quantized = pq.quantize(&test_vec);

        assert_eq!(quantized.len(), m);
        // All codes should be within valid range
        assert!(quantized.iter().all(|&c| (c as usize) < K));
    }
}
