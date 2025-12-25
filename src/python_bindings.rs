//! High-performance Python bindings with zero-copy operations
//!
//! Optimizations:
//! - Batch operations for bulk adds
//! - GIL-free search execution
//! - Memory-efficient ID mapping with FxHashMap
//! - Auto-flushing on large batches

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rustc_hash::FxHashMap; // Faster than std HashMap
use serde_json;
use crate::{SvDB, Vector, VectorEngine};
use crate::types::EmbeddedVector;

const AUTO_FLUSH_THRESHOLD: usize = 1000; // Auto-flush every 1k vectors

#[pyclass]
pub struct SvDBPython {
    db: SvDB,
    id_map: FxHashMap<String, u64>,
    reverse_id_map: FxHashMap<u64, String>,
    pending_flush: usize, // Track unflushed vectors
}

#[pymethods]
impl SvDBPython {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let db = SvDB::new(&path)
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize: {}", e)))?;
        
        let mut id_map = FxHashMap::default();
        let mut reverse_id_map = FxHashMap::default();
        
        // Rebuild ID mappings
        let count = db.count();
        for internal_id in 0..count {
            if let Ok(Some(metadata_json)) = db.get_metadata(internal_id) {
                if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&metadata_json) {
                    if let Some(string_id) = metadata.get("__id__").and_then(|v| v.as_str()) {
                        id_map.insert(string_id.to_string(), internal_id);
                        reverse_id_map.insert(internal_id, string_id.to_string());
                    }
                }
            }
        }
        
        Ok(Self {
            db,
            id_map,
            reverse_id_map,
            pending_flush: 0,
        })
    }

    /// Optimized bulk add with automatic batching
    fn add(
        &mut self,
        ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Vec<String>,
    ) -> PyResult<usize> {
        if ids.len() != embeddings.len() || ids.len() != metadatas.len() {
            return Err(PyValueError::new_err(
                "ids, embeddings, and metadatas must have same length"
            ));
        }

        // Pre-allocate for batch
        let mut embedded_vectors = Vec::with_capacity(ids.len());
        let mut enriched_metadatas = Vec::with_capacity(ids.len());

        // Validate and prepare batch
        for (i, (id, embedding)) in ids.iter().zip(embeddings.iter()).enumerate() {
            if self.id_map.contains_key(id) {
                return Err(PyValueError::new_err(format!("Duplicate ID: {}", id)));
            }

            if embedding.len() != 1536 {
                return Err(PyValueError::new_err(
                    format!("Vector {} must have 1536 dims, got {}", i, embedding.len())
                ));
            }

            // Convert to embedded vector
            let mut array = [0.0f32; 1536];
            array.copy_from_slice(embedding);
            embedded_vectors.push(array);

            // Prepare metadata
            let mut metadata_obj: serde_json::Value = serde_json::from_str(&metadatas[i])
                .unwrap_or(serde_json::json!({}));
            
            if let Some(obj) = metadata_obj.as_object_mut() {
                obj.insert("__id__".to_string(), serde_json::Value::String(id.clone()));
            }
            
            enriched_metadatas.push(
                serde_json::to_string(&metadata_obj)
                    .map_err(|e| PyValueError::new_err(format!("Metadata error: {}", e)))?
            );
        }

        // Batch insert (much faster)
        let internal_ids = self.db.vector_storage.append_batch(&embedded_vectors)
            .map_err(|e| PyValueError::new_err(format!("Batch append failed: {}", e)))?;

        // Store metadata and mappings
        for (i, internal_id) in internal_ids.iter().enumerate() {
            self.db.metadata_store.set(*internal_id, &enriched_metadatas[i])
                .map_err(|e| PyValueError::new_err(format!("Metadata store failed: {}", e)))?;
            
            self.id_map.insert(ids[i].clone(), *internal_id);
            self.reverse_id_map.insert(*internal_id, ids[i].clone());
        }

        self.pending_flush += ids.len();

        // Auto-flush large batches
        if self.pending_flush >= AUTO_FLUSH_THRESHOLD {
            self.persist()?;
        }

        Ok(ids.len())
    }

    /// GIL-free search for maximum concurrency
    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(String, f32)>> {
        if query.len() != 1536 {
            return Err(PyValueError::new_err(
                format!("Query must be 1536-dim, got {}", query.len())
            ));
        }

        // Convert query
        let mut array = [0.0f32; 1536];
        array.copy_from_slice(&query);
        
        // Release GIL during search (allows parallel Python threads)
        let results = Python::with_gil(|py| {
            py.allow_threads(|| {
                self.db.search(&Vector::new(query.clone()), k)
            })
        }).map_err(|e| PyValueError::new_err(format!("Search failed: {}", e)))?;

        // Map to string IDs
        Ok(results
            .into_iter()
            .filter_map(|r| {
                self.reverse_id_map.get(&r.id).map(|id| (id.clone(), r.score))
            })
            .collect())
    }

    /// Batch search for high throughput
    fn search_batch(
        &self,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> PyResult<Vec<Vec<(String, f32)>>> {
        let embedded_queries: Result<Vec<EmbeddedVector>, _> = queries
            .iter()
            .map(|q| {
                if q.len() != 1536 {
                    return Err(PyValueError::new_err("All queries must be 1536-dim"));
                }
                let mut array = [0.0f32; 1536];
                array.copy_from_slice(q);
                Ok(array)
            })
            .collect();

        let embedded_queries = embedded_queries?;

        // GIL-free batch search
        let results = Python::with_gil(|py| {
            py.allow_threads(|| {
                crate::search::search_batch(&self.db.vector_storage, &embedded_queries, k)
            })
        }).map_err(|e| PyValueError::new_err(format!("Batch search failed: {}", e)))?;

        // Map all results to string IDs
        Ok(results
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .filter_map(|(id, score)| {
                        self.reverse_id_map.get(&id).map(|s| (s.clone(), score))
                    })
                    .collect()
            })
            .collect())
    }

    fn get(&self, id: String) -> PyResult<Option<String>> {
        let internal_id = self.id_map.get(&id)
            .ok_or_else(|| PyValueError::new_err(format!("ID not found: {}", id)))?;

        self.db.get_metadata(*internal_id)
            .map_err(|e| PyValueError::new_err(format!("Get failed: {}", e)))
    }

    fn count(&self) -> PyResult<usize> {
        Ok(self.db.count() as usize)
    }

    fn persist(&mut self) -> PyResult<()> {
        self.db.persist()
            .map_err(|e| PyValueError::new_err(format!("Persist failed: {}", e)))?;
        self.pending_flush = 0;
        Ok(())
    }

    fn delete(&mut self, ids: Vec<String>) -> PyResult<usize> {
        let mut deleted = 0;
        for id in ids {
            if let Some(internal_id) = self.id_map.remove(&id) {
                self.reverse_id_map.remove(&internal_id);
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    fn get_all_ids(&self) -> PyResult<Vec<String>> {
        Ok(self.id_map.keys().cloned().collect())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "SvDB(vectors={}, pending_flush={})",
            self.count().unwrap_or(0),
            self.pending_flush
        ))
    }
}

#[pymodule]
fn srvdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SvDBPython>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "SvDB - High-Performance Embedded Vector Database")?;
    Ok(())
}