//! Binary quantization module
//!
//! Converts f32 vectors to binary representation (1 bit per dimension)
//! for efficient Hamming distance computation.

use crate::types::QuantizedVector;

/// Quantize a 1536-dimensional float vector to binary representation.
///
/// Conversion: f32 > 0.0 → 1, otherwise → 0
/// Packs 8 bits into each byte, resulting in 192 bytes (1536/8).
///
/// # Arguments
/// * `vector` - Slice of 1536 f32 values
///
/// # Returns
/// * `QuantizedVector` - 192-byte binary representation
pub fn quantize_vector(vector: &[f32]) -> QuantizedVector {
    assert_eq!(vector.len(), 1536, "Vector must be 1536 dimensions");

    let mut quantized = [0u8; 192];

    for (byte_idx, chunk) in vector.chunks(8).enumerate() {
        let mut byte = 0u8;
        for (bit_idx, &value) in chunk.iter().enumerate() {
            if value > 0.0 {
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
///
/// # Arguments
/// * `a` - First quantized vector
/// * `b` - Second quantized vector
///
/// # Returns
/// * `u32` - Hamming distance (number of differing bits)
#[inline]
pub fn hamming_distance(a: &QuantizedVector, b: &QuantizedVector) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Convert Hamming distance to normalized similarity score (0.0 to 1.0).
///
/// Lower Hamming distance = higher similarity.
///
/// # Arguments
/// * `distance` - Hamming distance
///
/// # Returns
/// * `f32` - Similarity score where 1.0 is identical, 0.0 is completely different
#[inline]
pub fn hamming_to_similarity(distance: u32) -> f32 {
    let max_distance = 1536.0; // Maximum possible distance
    1.0 - (distance as f32 / max_distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_positive() {
        let vec = vec![1.0; 1536];
        let quantized = quantize_vector(&vec);
        // All bits should be set
        assert!(quantized.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_quantize_negative() {
        let vec = vec![-1.0; 1536];
        let quantized = quantize_vector(&vec);
        // All bits should be unset
        assert!(quantized.iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = [0xFF; 192];
        let b = [0xFF; 192];
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_opposite() {
        let a = [0xFF; 192];
        let b = [0x00; 192];
        assert_eq!(hamming_distance(&a, &b), 1536);
    }

    #[test]
    fn test_similarity_score() {
        assert_eq!(hamming_to_similarity(0), 1.0);
        assert_eq!(hamming_to_similarity(1536), 0.0);
        assert!((hamming_to_similarity(768) - 0.5).abs() < 0.01);
    }
}
