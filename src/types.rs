//! Core type definitions for SrvDB

/// Vector with floating-point data
#[derive(Debug, Clone)]
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
        Self { id, score, metadata }
    }
}

/// Full precision embedded vector (1536 dimensions = 6144 bytes)
pub type EmbeddedVector = [f32; 1536];

/// Convert Vec<f32> to fixed-size array
pub fn to_embedded_vector(data: &[f32]) -> anyhow::Result<EmbeddedVector> {
    if data.len() != 1536 {
        anyhow::bail!("Expected 1536 dimensions, got {}", data.len());
    }
    let mut array = [0.0f32; 1536];
    array.copy_from_slice(data);
    Ok(array)
}

/// Internal vector file header
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VectorHeader {
    pub magic: u32,
    pub count: u64,
    pub version: u16,
    pub reserved: [u8; 6],
}

impl VectorHeader {
    pub const MAGIC: u32 = 0x53764442; // "SvDB"
    pub const VERSION: u16 = 2; // Version 2: Full precision
    pub const SIZE: usize = std::mem::size_of::<VectorHeader>();

    pub fn new() -> Self {
        Self {
            magic: Self::MAGIC,
            count: 0,
            version: Self::VERSION,
            reserved: [0; 6],
        }
    }
}

impl Default for VectorHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let data = vec![0.5; 1536];
        let vec = Vector::new(data.clone());
        assert_eq!(vec.dim(), 1536);
        assert_eq!(vec.data, data);
    }

    #[test]
    fn test_to_embedded_vector() {
        let data = vec![0.1; 1536];
        let embedded = to_embedded_vector(&data).unwrap();
        assert_eq!(embedded.len(), 1536);
        assert_eq!(embedded[0], 0.1);
    }

    #[test]
    fn test_invalid_dimensions() {
        let data = vec![0.1; 100];
        let result = to_embedded_vector(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_size() {
        // Header should be a reasonable size
        assert!(VectorHeader::SIZE > 0 && VectorHeader::SIZE < 256);
        assert_eq!(VectorHeader::SIZE, std::mem::size_of::<VectorHeader>());
    }
}