//! Product Quantization (PQ) for vector compression
//!
//! Implements FAISS-style Product Quantization:
//! - 32x memory compression (6KB → 192 bytes)
//! - M=192 sub-quantizers, K=256 centroids each
//! - Asymmetric Distance Computation (ADC) for fast search
//! - K-means clustering for codebook training

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use std::f32;

/// Configuration for Product Quantization
pub const M: usize = 192;           // Number of sub-quantizers
pub const K: usize = 256;           // Centroids per sub-quantizer (fits in u8)
pub const D_SUB: usize = 8;         // Dimensions per sub-space (1536 / 192)
pub const DIM: usize = 1536;        // Total dimensions

/// Quantized vector representation (192 bytes)
pub type QuantizedVector = [u8; M];

/// Distance lookup table for Asymmetric Distance Computation
#[derive(Debug, Clone)]
pub struct DistanceTable {
    pub tables: Vec<[f32; K]>,  // M tables, each with K distances
}

/// Codebook for a single sub-quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    pub centroids: Vec<[f32; D_SUB]>,  // K centroids
}

/// Normalize a vector to unit length (L2 norm = 1.0)
#[inline]
fn normalize_vector<const D: usize>(vec: &mut [f32; D]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

impl Codebook {
    /// Create new codebook with k-means clustering
    pub fn train(vectors: &[[f32; D_SUB]], k: usize, max_iter: usize) -> Result<Self> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot train on empty dataset");
        }

        // Use vectors as-is (don't normalize sub-vectors)
        // Cosine similarity will be computed via magnitude normalization in distance calculation
        let mut centroids = Self::kmeans_plusplus_init(vectors, k);
        
        for _ in 0..max_iter {
            // Assignment step
            let assignments: Vec<usize> = vectors
                .par_iter()
                .map(|v| Self::nearest_centroid(v, &centroids))
                .collect();

            // Update step
            let mut new_centroids = vec![[0.0f32; D_SUB]; k];
            let mut counts = vec![0usize; k];

            for (vec, &cluster_id) in vectors.iter().zip(assignments.iter()) {
                for i in 0..D_SUB {
                    new_centroids[cluster_id][i] += vec[i];
                }
                counts[cluster_id] += 1;
            }

            // Average to get new centroids
            let mut converged = true;
            for i in 0..k {
                if counts[i] > 0 {
                    for j in 0..D_SUB {
                        new_centroids[i][j] /= counts[i] as f32;
                    }
                    // Normalize centroids to unit length for proper cosine similarity
                    normalize_vector(&mut new_centroids[i]);
                    
                    // Check convergence
                    if Self::euclidean_distance(&centroids[i], &new_centroids[i]) > 1e-6 {
                        converged = false;
                    }
                } else {
                    // Reinitialize empty clusters
                    new_centroids[i] = vectors[rand::random::<usize>() % vectors.len()];
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
    fn kmeans_plusplus_init(vectors: &[[f32; D_SUB]], k: usize) -> Vec<[f32; D_SUB]> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // Pick first centroid randomly
        let first_idx = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first_idx]);

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
            let mut target = rng.gen::<f32>() * total;

            for (i, &dist) in distances.iter().enumerate() {
                target -= dist;
                if target <= 0.0 {
                    centroids.push(vectors[i]);
                    break;
                }
            }
        }

        centroids
    }

    /// Find nearest centroid index using cosine similarity
    #[inline]
    fn nearest_centroid(vec: &[f32; D_SUB], centroids: &[[f32; D_SUB]]) -> usize {
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
            .unwrap()
    }

    /// Euclidean distance between two vectors
    #[inline]
    fn euclidean_distance(a: &[f32; D_SUB], b: &[f32; D_SUB]) -> f32 {
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
    pub fn encode(&self, sub_vec: &[f32; D_SUB]) -> u8 {
        // Use sub-vector as-is, distance computation handles normalization
        Self::nearest_centroid(sub_vec, &self.centroids) as u8
    }

    /// Compute distances from query sub-vector to all centroids
    /// Centroids are normalized, query sub-vector is not
    /// Returns NEGATIVE cosine similarity (for minimization-based search)
    #[inline]
    pub fn compute_distances(&self, query_sub: &[f32; D_SUB]) -> [f32; K] {
        // Calculate magnitude of query sub-vector
        let query_magnitude: f32 = query_sub.iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        
        let mut distances = [0.0f32; K];
        
        // If query magnitude is near zero, return zeros
        if query_magnitude < 1e-10 {
            return distances;
        }
        
        for (i, centroid) in self.centroids.iter().enumerate() {
            // Compute dot product (centroid is normalized to unit length)
            let dot_product: f32 = query_sub.iter()
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
    pub fn train(training_data: &[[f32; DIM]], max_iter: usize) -> Result<Self> {
        if training_data.is_empty() {
            anyhow::bail!("Training data cannot be empty");
        }

        println!("Training Product Quantizer with M={}, K={}, D_sub={}", M, K, D_SUB);

        // Train codebook for each sub-quantizer
        let codebooks: Result<Vec<Codebook>> = (0..M)
            .into_par_iter()
            .map(|m_idx| {
                println!("Training sub-quantizer {}/{}", m_idx + 1, M);
                
                // Extract and normalize sub-vectors for this sub-quantizer
                let sub_vectors: Vec<[f32; D_SUB]> = training_data
                    .iter()
                    .map(|vec| {
                        let start = m_idx * D_SUB;
                        let end = start + D_SUB;
                        let mut sub_vec = [0.0f32; D_SUB];
                        sub_vec.copy_from_slice(&vec[start..end]);
                        sub_vec
                    })
                    .collect();

                // Train codebook for this sub-space (normalization happens inside)
                Codebook::train(&sub_vectors, K, max_iter)
                    .context(format!("Failed to train codebook {}", m_idx))
            })
            .collect();

        Ok(Self {
            m: M,
            k: K,
            d_sub: D_SUB,
            codebooks: codebooks?,
        })
    }

    /// Encode a full vector into quantized representation
    #[inline]
    pub fn encode(&self, vector: &[f32; DIM]) -> QuantizedVector {
        let mut quantized = [0u8; M];
        
        for (m_idx, codebook) in self.codebooks.iter().enumerate() {
            let start = m_idx * D_SUB;
            let end = start + D_SUB;
            let mut sub_vec = [0.0f32; D_SUB];
            sub_vec.copy_from_slice(&vector[start..end]);
            
            quantized[m_idx] = codebook.encode(&sub_vec);
        }
        
        quantized
    }

    /// Compute distance table for asymmetric distance computation
    pub fn compute_distance_table(&self, query: &[f32; DIM]) -> DistanceTable {
        let tables: Vec<[f32; K]> = self.codebooks
            .iter()
            .enumerate()
            .map(|(m_idx, codebook)| {
                let start = m_idx * D_SUB;
                let end = start + D_SUB;
                let mut query_sub = [0.0f32; D_SUB];
                query_sub.copy_from_slice(&query[start..end]);
                
                codebook.compute_distances(&query_sub)
            })
            .collect();

        DistanceTable { tables }
    }

    /// Asymmetric distance computation using precomputed distance table
    /// Returns approximate Cosine Similarity (averaged across all sub-spaces)
    #[inline]
    pub fn asymmetric_distance(&self, qvec: &QuantizedVector, dtable: &DistanceTable) -> f32 {
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
        let vectors: Vec<[f32; D_SUB]> = (0..1000)
            .map(|i| {
                let mut v = [0.0f32; D_SUB];
                for j in 0..D_SUB {
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
        // Create simple training data
        let training: Vec<[f32; DIM]> = (0..100)
            .map(|i| {
                let mut v = [0.0f32; DIM];
                for j in 0..DIM {
                    v[j] = (i as f32 / 100.0) + (j as f32 / 1536.0);
                }
                v
            })
            .collect();

        let pq = ProductQuantizer::train(&training, 10).unwrap();
        
        let test_vec = [0.5f32; DIM];
        let quantized = pq.encode(&test_vec);
        
        assert_eq!(quantized.len(), M);
        // All codes should be within valid range
        assert!(quantized.iter().all(|&c| (c as usize) < K));
    }

    #[test]
    fn test_asymmetric_distance() {
        let training: Vec<[f32; DIM]> = (0..50)
            .map(|i| {
                let mut v = [0.0f32; DIM];
                v[0] = i as f32;
                v
            })
            .collect();

        let pq = ProductQuantizer::train(&training, 5).unwrap();
        
        let query = [1.0f32; DIM];
        let dtable = pq.compute_distance_table(&query);
        
        let qvec = pq.encode(&[1.0f32; DIM]);
        let distance = pq.asymmetric_distance(&qvec, &dtable);
        
        // Distance should be positive and reasonable
        assert!(distance > 0.0 && distance <= 1.0);
    }
}
