//! Enhanced Python bindings for SrvDB v0.2.0
//! Features: Dynamic dimensions, SQ8, better error messages

use crate::VectorEngine;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use serde_json;

#[pyclass]
pub struct SrvDBPython {
    db: crate::SrvDB,
    id_map: FxHashMap<String, u64>,
    reverse_id_map: FxHashMap<u64, String>,
    dimension: usize,
    index_type: String,
}

#[pymethods]
impl SrvDBPython {
    /// Create new database with specified dimension (128-4096)
    ///
    /// Args:
    ///     path: Database directory
    ///     dimension: Vector dimension (default: 1536)
    ///     mode: Index mode - 'flat', 'hnsw', 'sq8', 'pq' (default: 'flat')
    ///
    /// Examples:
    ///     >>> db = srvdb.SrvDBPython("./db", dimension=384)  # MiniLM
    ///     >>> db = srvdb.SrvDBPython("./db", dimension=768, mode='sq8')  # Cohere
    #[new]
    #[pyo3(signature = (path, dimension=1536, mode="flat"))]
    fn new(path: String, dimension: usize, mode: &str) -> PyResult<Self> {
        // Validate dimension
        if !(128..=4096).contains(&dimension) {
            return Err(PyValueError::new_err(format!(
                "Dimension must be between 128 and 4096, got {}. \
                Common values: 384 (MiniLM), 768 (MPNet/Cohere), 1024 (Cohere v3), 1536 (OpenAI)",
                dimension
            )));
        }

        let mut config = crate::types::DatabaseConfig::new(dimension)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Set mode
        match mode.to_lowercase().as_str() {
            "flat" => {
                config.index_type = crate::types::IndexType::Flat;
            }
            "hnsw" => {
                config.index_type = crate::types::IndexType::HNSW;
            }
            "sq8" | "scalar" => {
                config.index_type = crate::types::IndexType::ScalarQuantized;
                config.quantization.enabled = true;
                config.quantization.mode = crate::types::QuantizationMode::Scalar;
            }
            "auto" => {
                // Auto mode logic will be triggered after init
                // For init config, we start safer (Flat) or let Apply Strategy decide later.
                // We'll mark it as Flat initially but return the object with IndexMode::Auto
                config.index_type = crate::types::IndexType::Flat;
            }
            "pq" | "product" => {
                config.index_type = crate::types::IndexType::ProductQuantized;
                config.quantization.enabled = true;
                config.quantization.mode = crate::types::QuantizationMode::Product;
                config.auto_tune_pq(); // Auto-calculate M and d_sub
            }
            "ivf" => {
                config.index_type = crate::types::IndexType::IVF;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid mode '{}'. Choose: 'flat', 'hnsw', 'sq8', 'pq', or 'auto'",
                    mode
                )));
            }
        }

        let mut db = crate::SrvDB::new_with_config(&path, config)
            .map_err(|e| PyRuntimeError::new_err(format!("Database init failed: {}", e)))?;

        // Apply Auto Strategy if requested
        if mode.to_lowercase() == "auto" {
            db.set_mode(crate::IndexMode::Auto);
        } else if mode.to_lowercase() == "sq8" || mode.to_lowercase() == "scalar" {
            db.current_mode = crate::IndexMode::Sq8;
        } else if mode.to_lowercase() == "hnsw" {
            db.current_mode = crate::IndexMode::Hnsw;
        } else if mode.to_lowercase() == "ivf" {
            // We need to set mode explicitly to trigger IVF init
            db.set_mode(crate::IndexMode::Ivf);
        }

        Ok(Self {
            db,
            id_map: FxHashMap::default(),
            reverse_id_map: FxHashMap::default(),
            dimension,
            index_type: mode.to_string(),
        })
    }

    /// Create database with HNSW indexing
    ///
    /// Args:
    ///     path: Database directory
    ///     dimension: Vector dimension
    ///     m: Graph connectivity (default: 16)
    ///     ef_construction: Build quality (default: 200)
    ///     ef_search: Search quality (default: 50)
    #[staticmethod]
    #[pyo3(signature = (path, dimension=1536, m=16, ef_construction=200, ef_search=50))]
    fn new_hnsw(
        path: String,
        dimension: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> PyResult<Self> {
        let _config = crate::types::DatabaseConfig::new(dimension)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // config.index_type = crate::types::IndexType::HNSW;

        let hnsw_config = crate::hnsw::HNSWConfig::new(m, ef_construction, ef_search);
        let db = crate::SrvDB::new_with_hnsw(&path, dimension, hnsw_config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            db,
            id_map: FxHashMap::default(),
            reverse_id_map: FxHashMap::default(),
            dimension,
            index_type: "hnsw".to_string(),
        })
    }

    /// Create database with Scalar Quantization (4x compression)
    ///
    /// Args:
    ///     path: Database directory
    ///     dimension: Vector dimension
    ///     training_vectors: Sample vectors for training (1000+ recommended)
    ///
    /// Notes:
    ///     - 4x compression (float32 -> uint8)
    ///     - 95%+ recall typical
    ///     - Much faster than PQ
    #[staticmethod]
    fn new_scalar_quantized(
        path: String,
        dimension: usize,
        training_vectors: Vec<Vec<f32>>,
    ) -> PyResult<Self> {
        if training_vectors.is_empty() {
            return Err(PyValueError::new_err(
                "Training vectors required. Provide 1000+ sample vectors for best results.",
            ));
        }

        // Validate dimensions
        for (i, vec) in training_vectors.iter().enumerate() {
            if vec.len() != dimension {
                return Err(PyValueError::new_err(format!(
                    "Training vector {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    dimension
                )));
            }
        }

        let db = crate::SrvDB::new_scalar_quantized(&path, dimension, &training_vectors)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            db,
            id_map: FxHashMap::default(),
            reverse_id_map: FxHashMap::default(),
            dimension,
            index_type: "sq8".to_string(),
        })
    }

    /// Create database with Product Quantization (32x compression)
    ///
    /// Args:
    ///     path: Database directory
    ///     dimension: Vector dimension
    ///     training_vectors: Sample vectors for training (1000+ recommended)
    ///
    /// Notes:
    ///     - 32x compression (6KB -> 192 bytes)
    ///     - Good recall with re-ranking
    ///     - Supports any dimension divisible by M
    #[staticmethod]
    fn new_product_quantized(
        path: String,
        dimension: usize,
        training_vectors: Vec<Vec<f32>>,
    ) -> PyResult<Self> {
        if training_vectors.is_empty() {
             return Err(PyValueError::new_err(
                "Training vectors required. Provide 1000+ sample vectors for best results.",
            ));
        }

        // Validate dimensions
        for (i, vec) in training_vectors.iter().enumerate() {
            if vec.len() != dimension {
                 return Err(PyValueError::new_err(format!(
                    "Training vector {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    dimension
                )));
            }
        }

        // Convert to internal Vector type
        let internal_vectors: Vec<crate::Vector> = training_vectors
            .into_iter()
            .map(crate::Vector::new)
            .collect();

        let db = crate::SrvDB::new_quantized(&path, &internal_vectors)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            db,
            id_map: FxHashMap::default(),
            reverse_id_map: FxHashMap::default(),
            dimension,
            index_type: "pq".to_string(),
        })
    }

    /// Add vectors to database
    ///
    /// Args:
    ///     ids: List of string IDs
    ///     embeddings: List of vectors (each must match database dimension)
    ///     metadatas: List of JSON metadata strings
    ///
    /// Returns:
    ///     Number of vectors added
    fn add(
        &mut self,
        ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Vec<String>,
    ) -> PyResult<usize> {
        if ids.len() != embeddings.len() || ids.len() != metadatas.len() {
            return Err(PyValueError::new_err(
                "ids, embeddings, and metadatas must have same length",
            ));
        }

        // Validate dimensions
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != self.dimension {
                return Err(PyValueError::new_err(format!(
                    "Vector {} has dimension {}, expected {}. \
                    Database was created with dimension {}. \
                    Common models: 384 (MiniLM), 768 (MPNet), 1536 (OpenAI)",
                    i,
                    emb.len(),
                    self.dimension,
                    self.dimension
                )));
            }
        }

        // Check for duplicates
        for id in &ids {
            if self.id_map.contains_key(id) {
                return Err(PyValueError::new_err(format!(
                    "Duplicate ID: '{}'. Use unique IDs or call delete() first.",
                    id
                )));
            }
        }

        // Prepare vectors
        let vectors: Vec<crate::Vector> = embeddings.into_iter().map(crate::Vector::new).collect();

        // Enrich metadata with IDs
        let enriched_metas: Vec<String> = ids
            .iter()
            .zip(metadatas.iter())
            .map(|(id, meta)| {
                let mut obj: serde_json::Value =
                    serde_json::from_str(meta).unwrap_or(serde_json::json!({}));

                if let Some(obj) = obj.as_object_mut() {
                    obj.insert("__id__".to_string(), serde_json::Value::String(id.clone()));
                }

                serde_json::to_string(&obj).unwrap()
            })
            .collect();

        // Batch insert
        let internal_ids = VectorEngine::add_batch(&mut self.db, &vectors, &enriched_metas)
            .map_err(|e| PyRuntimeError::new_err(format!("Add failed: {}", e)))?;

        // Update mappings
        for (i, internal_id) in internal_ids.iter().enumerate() {
            self.id_map.insert(ids[i].clone(), *internal_id);
            self.reverse_id_map.insert(*internal_id, ids[i].clone());
        }

        Ok(ids.len())
    }

    /// Search for k nearest neighbors
    ///
    /// Args:
    ///     query: Query vector (must match database dimension)
    ///     k: Number of results
    ///
    /// Returns:
    ///     List of (id, score) tuples sorted by score descending
    fn search(&mut self, query: Vec<f32>, k: usize) -> PyResult<Vec<(String, f32)>> {
        if query.len() != self.dimension {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} doesn't match database dimension {}",
                query.len(),
                self.dimension
            )));
        }

        // Auto-persist before search to ensure all vectors are searchable
        VectorEngine::persist(&mut self.db)
            .map_err(|e| PyRuntimeError::new_err(format!("Auto-persist failed: {}", e)))?;

        let results = Python::with_gil(|py| {
            py.allow_threads(|| VectorEngine::search(&self.db, &crate::Vector::new(query), k))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {}", e)))?;

        Ok(results
            .into_iter()
            .filter_map(|r| {
                self.reverse_id_map
                    .get(&r.id)
                    .map(|id| (id.clone(), r.score))
            })
            .collect())
    }

    /// Batch search
    fn search_batch(
        &mut self,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> PyResult<Vec<Vec<(String, f32)>>> {
        // Validate dimensions
        for (i, q) in queries.iter().enumerate() {
            if q.len() != self.dimension {
                return Err(PyValueError::new_err(format!(
                    "Query {} dimension {} doesn't match database dimension {}",
                    i,
                    q.len(),
                    self.dimension
                )));
            }
        }

        // Auto-persist before search
        VectorEngine::persist(&mut self.db)
            .map_err(|e| PyRuntimeError::new_err(format!("Auto-persist failed: {}", e)))?;

        let query_vecs: Vec<crate::Vector> = queries.into_iter().map(crate::Vector::new).collect();

        let results = Python::with_gil(|py| {
            py.allow_threads(|| VectorEngine::search_batch(&self.db, &query_vecs, k))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Batch search failed: {}", e)))?;

        Ok(results
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .filter_map(|r| {
                        self.reverse_id_map
                            .get(&r.id)
                            .map(|id| (id.clone(), r.score))
                    })
                    .collect()
            })
            .collect())
    }

    fn get(&self, id: String) -> PyResult<Option<String>> {
        let internal_id = self
            .id_map
            .get(&id)
            .ok_or_else(|| PyValueError::new_err(format!("ID not found: {}", id)))?;

        VectorEngine::get_metadata(&self.db, *internal_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn count(&self) -> usize {
        VectorEngine::count(&self.db) as usize
    }

    fn persist(&mut self) -> PyResult<()> {
        VectorEngine::persist(&mut self.db)
            .map_err(|e| PyRuntimeError::new_err(format!("Persist failed: {}", e)))
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

    /// Get all IDs in the database
    fn get_all_ids(&self) -> PyResult<Vec<String>> {
        Ok(self.id_map.keys().cloned().collect())
    }

    /// Set index mode
    fn set_mode(&mut self, mode: String) -> PyResult<()> {
        match mode.to_lowercase().as_str() {
            "flat" => self.db.set_mode(crate::IndexMode::Flat),
            "hnsw" => self.db.set_mode(crate::IndexMode::Hnsw),
            "sq8" | "scalar" => self.db.set_mode(crate::IndexMode::Sq8),
            "ivf" => self.db.set_mode(crate::IndexMode::Ivf),
            "auto" => self.db.set_mode(crate::IndexMode::Auto),
            _ => return Err(PyValueError::new_err(format!("Invalid mode: {}", mode))),
        }
        self.index_type = mode;
        Ok(())
    }

    /// Configure IVF parameters
    ///
    /// Args:
    ///     nlist: Number of partitions (centroids)
    ///     nprobe: Number of partitions to search (search quality vs speed)
    #[pyo3(signature = (nlist=100, nprobe=10))]
    fn configure_ivf(&mut self, nlist: usize, nprobe: usize) -> PyResult<()> {
        let config = crate::ivf::IVFConfig {
            nlist,
            nprobe,
            max_iterations: 10,
            tolerance: 0.001,
            hnsw_config: Default::default(),
        };
        self.db
            .configure_ivf(config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Train IVF index
    fn train_ivf(&mut self) -> PyResult<()> {
        // Python holds the GIL, but training is heavy. Release GIL?
        // Yes, allow threads.
        Python::with_gil(|py| py.allow_threads(|| self.db.train_ivf()))
            .map_err(|e| PyRuntimeError::new_err(format!("IVF Training failed: {}", e)))
    }

    fn info(&self) -> PyResult<String> {
        let stats = if let Some(stats) = self.db.get_stats() {
            format!(
                "vectors={}, memory={}MB, compression={:.1}x",
                stats.vector_count,
                stats.memory_bytes / 1024 / 1024,
                stats.compression_ratio
            )
        } else {
            format!("vectors={}", self.count())
        };

        Ok(format!(
            "SrvDB(dimension={}, mode='{}', {})",
            self.dimension, self.index_type, stats
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        self.info()
    }
}

#[pymodule]
fn srvdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SrvDBPython>()?;
    m.add("__version__", "0.2.0")?;
    m.add(
        "__doc__",
        "SrvDB v0.2.0 - Production Vector Database\n\n\
        Features:\n\
        - Dynamic dimensions (128-4096)\n\
        - Multiple index types (Flat, HNSW, SQ8, PQ)\n\
        - 4-32x compression with SQ8/PQ\n\
        - Sub-5ms latency\n\
        - <100MB memory for 10k vectors",
    )?;

    Ok(())
}
