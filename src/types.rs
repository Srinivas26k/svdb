//! Core type definitions for SvDB

/// Represents a vector with floating-point data
#[derive(Debug, Clone)]
pub struct Vector {
    /// Vector data (must be 1536 dimensions for MVP)
    pub data: Vec<f32>,
}

impl Vector {
    /// Create a new vector from raw f32 data
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Get the dimensionality of the vector
    pub fn dim(&self) -> usize {
        self.data.len()
    }
}

/// Represents a search result with ID, similarity score, and metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Internal vector ID
    pub id: u64,
    /// Similarity score (0.0 to 1.0, higher is more similar)
    pub score: f32,
    /// Optional metadata JSON string
    pub metadata: Option<String>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(id: u64, score: f32, metadata: Option<String>) -> Self {
        Self { id, score, metadata }
    }
}

/// Binary quantized vector (192 bytes for 1536 dimensions)
pub type QuantizedVector = [u8; 192];

/// Internal vector header stored in vectors.bin
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VectorHeader {
    /// Magic number for validation
    pub magic: u32,
    /// Total number of vectors
    pub count: u64,
    /// Version number
    pub version: u16,
    /// Reserved for future use
    pub reserved: [u8; 6],
}

impl VectorHeader {
    pub const MAGIC: u32 = 0x53764442; // "SvDB" in hex
    pub const VERSION: u16 = 1;
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
