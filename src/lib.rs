//! # SvDB - Ultra-Fast Embedded Vector Database
//!
//! Production-grade vector database optimized for:
//! - 100k+ vectors/sec ingestion
//! - Sub-5ms search latency
//! - <100MB memory for 10k vectors
//! - 200+ concurrent QPS
//!
//! ## Architecture
//! - 8MB buffered writes with atomic counters
//! - SIMD-accelerated cosine similarity
//! - Lock-free parallel search with batch processing
//! - Memory-mapped zero-copy reads

use anyhow::Result;
use std::path::Path;

pub mod types;
pub use types::{Vector, SearchResult};

mod storage;
mod search;
mod metadata;

#[cfg(feature = "pyo3")]
pub mod python_bindings;

pub use storage::VectorStorage;
pub use metadata::MetadataStore;

/// High-performance vector database trait
pub trait VectorEngine {
    fn new(path: &str) -> Result<Self> where Self: Sized;
    fn add(&mut self, vec: &Vector, meta: &str) -> Result<u64>;
    fn add_batch(&mut self, vecs: &[Vector], metas: &[String]) -> Result<Vec<u64>>;
    fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>>;
    fn search_batch(&self, queries: &[Vector], k: usize) -> Result<Vec<Vec<SearchResult>>>;
    fn get_metadata(&self, id: u64) -> Result<Option<String>>;
    fn persist(&mut self) -> Result<()>;
    fn count(&self) -> u64;
}

/// Main database implementation
pub struct SvDB {
    pub(crate) vector_storage: VectorStorage,
    pub(crate) metadata_store: MetadataStore,
}

impl VectorEngine for SvDB {
    fn new(path: &str) -> Result<Self> {
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        let vector_storage = VectorStorage::new(path)?;
        let metadata_store = MetadataStore::new(path)?;

        Ok(Self {
            vector_storage,
            metadata_store,
        })
    }

    fn add(&mut self, vec: &Vector, meta: &str) -> Result<u64> {
        if vec.data.len() != 1536 {
            anyhow::bail!("Vector must be 1536-dimensional");
        }

        let embedded = types::to_embedded_vector(&vec.data)?;
        let id = self.vector_storage.append(&embedded)?;
        self.metadata_store.set(id, meta)?;

        Ok(id)
    }

    fn add_batch(&mut self, vecs: &[Vector], metas: &[String]) -> Result<Vec<u64>> {
        if vecs.len() != metas.len() {
            anyhow::bail!("Vectors and metadata counts must match");
        }

        // Convert all vectors
        let embedded: Result<Vec<_>> = vecs
            .iter()
            .map(|v| {
                if v.data.len() != 1536 {
                    anyhow::bail!("All vectors must be 1536-dimensional");
                }
                types::to_embedded_vector(&v.data)
            })
            .collect();

        let embedded = embedded?;

        // Batch append vectors
        let ids = self.vector_storage.append_batch(&embedded)?;

        // Store metadata
        for (id, meta) in ids.iter().zip(metas.iter()) {
            self.metadata_store.set(*id, meta)?;
        }

        Ok(ids)
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        if query.data.len() != 1536 {
            anyhow::bail!("Query must be 1536-dimensional");
        }

        let embedded_query = types::to_embedded_vector(&query.data)?;
        let results = search::search_cosine(&self.vector_storage, &embedded_query, k)?;

        let mut enriched = Vec::with_capacity(results.len());
        for (id, score) in results {
            let metadata = self.metadata_store.get(id)?;
            enriched.push(SearchResult { id, score, metadata });
        }

        Ok(enriched)
    }

    fn search_batch(&self, queries: &[Vector], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        let embedded_queries: Result<Vec<_>> = queries
            .iter()
            .map(|q| {
                if q.data.len() != 1536 {
                    anyhow::bail!("All queries must be 1536-dimensional");
                }
                types::to_embedded_vector(&q.data)
            })
            .collect();

        let embedded_queries = embedded_queries?;
        let batch_results = search::search_batch(&self.vector_storage, &embedded_queries, k)?;

        // Enrich with metadata
        batch_results
            .into_iter()
            .map(|results| {
                results
                    .into_iter()
                    .map(|(id, score)| {
                        let metadata = self.metadata_store.get(id)?;
                        Ok(SearchResult { id, score, metadata })
                    })
                    .collect()
            })
            .collect()
    }

    fn get_metadata(&self, id: u64) -> Result<Option<String>> {
        self.metadata_store.get(id)
    }

    fn persist(&mut self) -> Result<()> {
        self.vector_storage.flush()?;
        self.metadata_store.flush()?;
        Ok(())
    }

    fn count(&self) -> u64 {
        self.vector_storage.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_batch_operations() {
        let temp_dir = TempDir::new().unwrap();
        let mut db = SvDB::new(temp_dir.path().to_str().unwrap()).unwrap();

        let vectors: Vec<Vector> = (0..100)
            .map(|i| Vector::new(vec![i as f32 / 100.0; 1536]))
            .collect();

        let metas: Vec<String> = (0..100)
            .map(|i| format!(r#"{{"id": {}}}"#, i))
            .collect();

        let ids = db.add_batch(&vectors, &metas).unwrap();
        assert_eq!(ids.len(), 100);

        db.persist().unwrap();

        let results = db.search(&vectors[0], 5).unwrap();
        assert_eq!(results.len(), 5);
        assert!(results[0].score > 0.99);
    }

    #[test]
    fn test_concurrent_search() {
        use std::sync::Arc;
        use std::thread;

        let temp_dir = TempDir::new().unwrap();
        let mut db = SvDB::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Add vectors
        let vectors: Vec<Vector> = (0..1000)
            .map(|_| Vector::new(vec![rand::random::<f32>(); 1536]))
            .collect();
        let metas: Vec<String> = (0..1000).map(|i| format!(r#"{{"id": {}}}"#, i)).collect();
        db.add_batch(&vectors, &metas).unwrap();
        db.persist().unwrap();

        let db = Arc::new(db);
        let query = Vector::new(vec![0.5; 1536]);

        // Spawn multiple search threads
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let db = Arc::clone(&db);
                let q = query.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = db.search(&q, 10);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }
}