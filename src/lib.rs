//! # SvDB - Zero-Gravity Embedded Vector Database
//!
//! **"All the intelligence, none of the weight."**
//!
//! SvDB is an embedded vector database that runs locally on devices with minimal RAM usage
//! via memory mapping and achieves sub-millisecond search speeds using binary quantization.
//!
//! ## Key Features
//! - **Offline-First**: No network calls, no latency, no cloud bills
//! - **RAM-Efficient**: Uses `mmap` to let the OS manage memory
//! - **Blazing Fast**: Bitwise operations (XOR/Popcount) for coarse search
//!
//! ## Example Usage
//! ```rust,no_run
//! use srvdb::{VectorEngine, SvDB, Vector};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Initialize database
//! let mut db = SvDB::new("./my_vectors")?;
//!
//! // Add vectors
//! let vec = Vector { data: vec![0.1; 1536] };
//! let id = db.add(&vec, r#"{"title": "example"}"#)?;
//!
//! // Search for similar vectors
//! let results = db.search(&vec, 10)?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use std::path::Path;

// Public API types
pub mod types;
pub use types::{Vector, SearchResult};

// Core modules
mod storage;
mod search;
mod metadata;

// Python bindings module
pub mod python_bindings;

pub use storage::VectorStorage;
pub use metadata::MetadataStore;

/// The main Vector Engine trait that defines the core operations
/// for the embedded vector database.
pub trait VectorEngine {
    /// Initialize database at the specified path.
    ///
    /// Creates necessary storage files (`vectors.bin` and `metadata.db`)
    /// if they don't exist, or opens existing database.
    ///
    /// # Arguments
    /// * `path` - Directory path where database files will be stored
    ///
    /// # Returns
    /// * `Result<Self>` - Initialized database instance
    fn new(path: &str) -> Result<Self>
    where
        Self: Sized;

    /// Add a vector with associated metadata to the database.
    ///
    /// The vector is quantized to binary representation and appended
    /// to the vector storage. Metadata is stored separately.
    ///
    /// # Arguments
    /// * `vec` - The input vector (1536 dimensions for OpenAI standard)
    /// * `meta` - JSON string containing metadata
    ///
    /// # Returns
    /// * `Result<u64>` - Auto-incremented internal ID for the vector
    fn add(&mut self, vec: &Vector, meta: &str) -> Result<u64>;

    /// Search for k nearest neighbors using binary quantized search.
    ///
    /// Uses Hamming distance for coarse search across all vectors,
    /// parallelized using rayon for optimal performance.
    ///
    /// # Arguments
    /// * `query` - Query vector to search for
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// * `Result<Vec<SearchResult>>` - List of search results with IDs, scores, and metadata
    fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>>;

    /// Retrieve metadata for a specific vector by ID.
    ///
    /// # Arguments
    /// * `id` - Internal vector ID
    ///
    /// # Returns
    /// * `Result<Option<String>>` - Metadata JSON string if found
    fn get_metadata(&self, id: u64) -> Result<Option<String>>;

    /// Force flush all pending writes to disk.
    ///
    /// Ensures durability by syncing both vector data and metadata.
    ///
    /// # Returns
    /// * `Result<()>`
    fn persist(&mut self) -> Result<()>;
}

/// Main implementation of the Vector Engine
pub struct SvDB {
    /// Memory-mapped vector storage
    vector_storage: VectorStorage,
    /// Key-value metadata store
    metadata_store: MetadataStore,
    /// Current vector count
    count: u64,
}

impl VectorEngine for SvDB {
    fn new(path: &str) -> Result<Self> {
        // Create directory if it doesn't exist
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        // Initialize vector storage (mmap-backed)
        let vector_storage = VectorStorage::new(path)?;
        
        // Initialize metadata store (redb-backed)
        let metadata_store = MetadataStore::new(path)?;
        
        // Get current count from storage
        let count = vector_storage.count();

        Ok(Self {
            vector_storage,
            metadata_store,
            count,
        })
    }

    fn add(&mut self, vec: &Vector, meta: &str) -> Result<u64> {
        // Validate vector dimensions
        if vec.data.len() != 1536 {
            anyhow::bail!("Vector must have exactly 1536 dimensions");
        }

        // Convert to embedded vector array
        let embedded = types::to_embedded_vector(&vec.data)
            .map_err(|e| anyhow::anyhow!(e))?;

        // Append to vector storage
        let id = self.vector_storage.append(&embedded)?;

        // Store metadata
        self.metadata_store.set(id, meta)?;

        self.count += 1;

        Ok(id)
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        // Validate query dimensions
        if query.data.len() != 1536 {
            anyhow::bail!("Query vector must have exactly 1536 dimensions");
        }

        // Convert to embedded vector array
        let embedded_query = types::to_embedded_vector(&query.data)
            .map_err(|e| anyhow::anyhow!(e))?;

        // Search using cosine similarity
        let results = search::search_cosine(
            &self.vector_storage,
            &embedded_query,
            k,
        )?;

        // Enrich with metadata
        let mut enriched_results = Vec::with_capacity(results.len());
        for (id, score) in results {
            let metadata = self.metadata_store.get(id)?;
            enriched_results.push(SearchResult {
                id,
                score,
                metadata,
            });
        }

        Ok(enriched_results)
    }

    fn get_metadata(&self, id: u64) -> Result<Option<String>> {
        self.metadata_store.get(id)
    }

    fn persist(&mut self) -> Result<()> {
        self.vector_storage.flush()?;
        self.metadata_store.flush()?;
        Ok(())
    }
}

impl SvDB {
    /// Get the number of vectors stored
    pub fn count(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let db = SvDB::new(temp_dir.path().to_str().unwrap());
        assert!(db.is_ok());
    }

    #[test]
    fn test_add_and_search() {
        let temp_dir = TempDir::new().unwrap();
        let mut db = SvDB::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Create a test vector
        let vec = Vector {
            data: vec![0.5; 1536],
        };

        // Add vector
        let id = db.add(&vec, r#"{"test": "data"}"#).unwrap();
        assert_eq!(id, 0);
        
        // Persist to flush buffered data before search
        db.persist().unwrap();

        // Search for the same vector
        let results = db.search(&vec, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn test_invalid_dimensions() {
        let temp_dir = TempDir::new().unwrap();
        let mut db = SvDB::new(temp_dir.path().to_str().unwrap()).unwrap();

        let vec = Vector {
            data: vec![0.5; 100], // Wrong dimension
        };

        let result = db.add(&vec, "{}");
        assert!(result.is_err());
    }
}
