//! Binary quantization module
//!
//! Converts f32 vectors to binary representation (1 bit per dimension)
//! for efficient Hamming distance computation.

use crate::types::QuantizedVector;

/// Quantize a 1536-dimensional float vector to binary representation.
///
/// FIX APPLIED: Uses the vector's Mean as the threshold instead of 0.0.
/// This ensures balanced bit distribution (high entropy) even for non-centered vectors.
///
/// # Arguments
/// * `vector` - Slice of 1536 f32 values
///
/// # Returns
/// * `QuantizedVector` - 192-byte binary representation
pub fn quantize_vector(vector: &[f32]) -> QuantizedVector {
    assert_eq!(vector.len(), 1536, "Vector must be 1536 dimensions");

    // STEP 1: Calculate the Mean (Adaptive Threshold)
    let sum: f32 = vector.iter().sum();
    let mean = sum / vector.len() as f32;

    let mut quantized = [0u8; 192];

    // STEP 2: Quantize based on Mean
    for (byte_idx, chunk) in vector.chunks(8).enumerate() {
        let mut byte = 0u8;
        for (bit_idx, &value) in chunk.iter().enumerate() {
            // If value is above average -> 1, else -> 0
            if value > mean {
                byte |= 1 << bit_idx;
            }
        }
        quantized[byte_idx] = byte;
    }

    quantized
}

/// Compute Hamming distance between two quantized vectors.
///
/// Uses XOR + popcount for efficient bit counting.
#[inline]
pub fn hamming_distance(a: &QuantizedVector, b: &QuantizedVector) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Convert Hamming distance to normalized similarity score (0.0 to 1.0).
#[inline]
pub fn hamming_to_similarity(distance: u32) -> f32 {
    let max_distance = 1536.0;
    1.0 - (distance as f32 / max_distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_adaptive() {
        // A vector with all positive values [0.1, 0.2, ... 0.8]
        // Old Logic: All 1s (useless)
        // New Logic: Mean is ~0.45, so 0.1->0 and 0.8->1 (useful!)
        let mut vec = vec![0.0; 1536];
        for i in 0..1536 {
            vec[i] = i as f32; // 0.0 to 1535.0
        }
        
        let quantized = quantize_vector(&vec);
        
        // Count bits set (should be roughly 50%)
        let total_bits: u32 = quantized.iter().map(|b| b.count_ones()).sum();
        let expected = 1536 / 2;
        let diff = (total_bits as i32 - expected as i32).abs();
        
        // Allow small margin of error due to chunk alignment
        assert!(diff < 10, "Adaptive quantization failed to balance bits. Ones: {}", total_bits);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = [0xFF; 192];
        let b = [0xFF; 192];
        assert_eq!(hamming_distance(&a, &b), 0);
    }
}