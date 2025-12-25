//! Python bindings for SvDB using PyO3
//!
//! Provides a ChromaDB-style API for Python integration

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use serde_json;
use crate::{SvDB, Vector, VectorEngine};

/// Python wrapper for SvDB with ChromaDB-style API
#[pyclass]
pub struct SvDBPython {
    db: SvDB,
    /// Map string IDs to internal u64 IDs
    id_map: HashMap<String, u64>,
    /// Reverse map for lookups
    reverse_id_map: HashMap<u64, String>,
    next_internal_id: u64,
}

#[pymethods]
impl SvDBPython {
    /// Create a new SvDB instance
    ///
    /// Args:
    ///     path: Directory path where database files will be stored
    ///
    /// Returns:
    ///     SvDBPython: Initialized database instance
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let db = SvDB::new(&path)
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize database: {}", e)))?;
        
        let mut id_map = HashMap::new();
        let mut reverse_id_map = HashMap::new();
        let mut next_internal_id = 0;
        
        // Rebuild ID mappings from existing metadata (if database exists)
        let count = db.count();
        for internal_id in 0..count {
            if let Ok(Some(metadata_json)) = db.get_metadata(internal_id) {
                // Parse metadata to extract string ID
                if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&metadata_json) {
                    if let Some(string_id) = metadata.get("__id__").and_then(|v| v.as_str()) {
                        id_map.insert(string_id.to_string(), internal_id);
                        reverse_id_map.insert(internal_id, string_id.to_string());
                        next_internal_id = next_internal_id.max(internal_id + 1);
                    }
                }
            }
        }
        
        Ok(Self {
            db,
            id_map,
            reverse_id_map,
            next_internal_id,
        })
    }

    /// Add vectors in bulk (ChromaDB-style)
    ///
    /// Args:
    ///     ids: List of string IDs for each vector
    ///     embeddings: List of vectors (each is a list of floats)
    ///     metadatas: List of JSON strings for metadata
    ///
    /// Returns:
    ///     int: Number of vectors added
    ///
    /// Example:
    ///     >>> db.add(
    ///     ...     ids=["doc1", "doc2"],
    ///     ...     embeddings=[[0.1] * 1536, [0.2] * 1536],
    ///     ...     metadatas=['{"title": "Doc 1"}', '{"title": "Doc 2"}']
    /// ... )
    ///     2
    fn add(
        &mut self,
        ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Vec<String>,
    ) -> PyResult<usize> {
        // Validate inputs
        if ids.len() != embeddings.len() || ids.len() != metadatas.len() {
            return Err(PyValueError::new_err(
                "ids, embeddings, and metadatas must have the same length"
            ));
        }

        let mut added_count = 0;

        for (i, (id, embedding)) in ids.iter().zip(embeddings.iter()).enumerate() {
            // Check if ID already exists
            if self.id_map.contains_key(id) {
                return Err(PyValueError::new_err(
                    format!("Duplicate ID: {}", id)
                ));
            }

            // Validate embedding dimensions
            if embedding.len() != 1536 {
                return Err(PyValueError::new_err(
                    format!("Embedding at index {} must have 1536 dimensions, got {}", i, embedding.len())
                ));
            }

            // Create vector and add to database
            let vec = Vector::new(embedding.clone());
            let user_metadata = &metadatas[i];
            
            // Inject string ID into metadata for persistence
            let mut metadata_obj: serde_json::Value = serde_json::from_str(user_metadata)
                .unwrap_or(serde_json::json!({}));
            
            // Add __id__ field (reserved for internal use)
            if let Some(obj) = metadata_obj.as_object_mut() {
                obj.insert("__id__".to_string(), serde_json::Value::String(id.clone()));
            }
            
            let metadata_with_id = serde_json::to_string(&metadata_obj)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize metadata: {}", e)))?;

            let internal_id = self.db.add(&vec, &metadata_with_id)
                .map_err(|e| PyValueError::new_err(format!("Failed to add vector: {}", e)))?;

            // Store ID mappings
            self.id_map.insert(id.clone(), internal_id);
            self.reverse_id_map.insert(internal_id, id.clone());
            self.next_internal_id = internal_id + 1;

            added_count += 1;
        }

        Ok(added_count)
    }

    /// Search for similar vectors
    ///
    /// Args:
    ///     query: Query vector (list of 1536 floats)
    ///     k: Number of results to return
    ///
    /// Returns:
    ///     List of tuples (id, score) sorted by similarity (highest first)
    ///
    /// Example:
    ///     >>> results = db.search(query=[0.1] * 1536, k=10)
    ///     >>> for id, score in results:
    ///     ...     print(f"{id}: {score}")
    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(String, f32)>> {
        // Validate query dimensions
        if query.len() != 1536 {
            return Err(PyValueError::new_err(
                format!("Query must have 1536 dimensions, got {}", query.len())
            ));
        }

        let vec = Vector::new(query);
        
        let results = self.db.search(&vec, k)
            .map_err(|e| PyValueError::new_err(format!("Search failed: {}", e)))?;

        // Convert internal IDs to string IDs
        let py_results: Vec<(String, f32)> = results
            .into_iter()
            .filter_map(|result| {
                self.reverse_id_map.get(&result.id).map(|id| (id.clone(), result.score))
            })
            .collect();

        Ok(py_results)
    }

    /// Get metadata for a specific ID
    ///
    /// Args:
    ///     id: String ID of the vector
    ///
    /// Returns:
    ///     Optional JSON string with metadata, or None if not found
    fn get(&self, id: String) -> PyResult<Option<String>> {
        let internal_id = self.id_map.get(&id)
            .ok_or_else(|| PyValueError::new_err(format!("ID not found: {}", id)))?;

        let metadata = self.db.get_metadata(*internal_id)
            .map_err(|e| PyValueError::new_err(format!("Failed to get metadata: {}", e)))?;

        Ok(metadata)
    }

    /// Get the number of vectors in the database
    ///
    /// Returns:
    ///     int: Number of vectors stored
    fn count(&self) -> PyResult<usize> {
        // Use actual DB count, not ID map (which is empty on reopen)
        Ok(self.db.count() as usize)
    }

    /// Persist all changes to disk
    ///
    /// Returns:
    ///     None
    fn persist(&mut self) -> PyResult<()> {
        self.db.persist()
            .map_err(|e| PyValueError::new_err(format!("Failed to persist: {}", e)))?;
        Ok(())
    }

    /// Delete vectors by ID
    ///
    /// Args:
    ///     ids: List of string IDs to delete
    ///
    /// Returns:
    ///     int: Number of vectors deleted
    fn delete(&mut self, ids: Vec<String>) -> PyResult<usize> {
        let mut deleted_count = 0;

        for id in ids {
            if let Some(internal_id) = self.id_map.remove(&id) {
                self.reverse_id_map.remove(&internal_id);
                deleted_count += 1;
            }
        }

        Ok(deleted_count)
    }

    /// Get all IDs in the database
    ///
    /// Returns:
    ///     List of string IDs
    fn get_all_ids(&self) -> PyResult<Vec<String>> {
        Ok(self.id_map.keys().cloned().collect())
    }

    /// Python representation
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("SvDB(vectors={}, path=<internal>)", self.id_map.len()))
    }
}

/// Python module initialization
#[pymodule]
fn srvdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SvDBPython>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "SvDB - Zero-Gravity Embedded Vector Database")?;
    Ok(())
}
