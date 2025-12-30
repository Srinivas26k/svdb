# SrvDB v0.2.0 Release Notes

**Release Date:** December 30, 2024

## 🚀 Major Features

### 1. Inverted File System (IVF-HNSW)
This release introduces a fully scalable **IVF-HNSW Hybrid Index**, designed to handle massive datasets by partitioning the vector space into Voronoi cells.
- **Scalability**: Capable of indexing 100k+ vectors with efficient partitioning.
- **Two-Stage Search**: Uses K-Means centroids for coarse search and HNSW graphs for fine refinement.
- **Performance**: Sub-10ms latency (P50: ~6ms at 100k vectors) with **100% recall** on test datasets.
- **Python API**: `db.set_mode("ivf")`, `db.configure_ivf()`.

### 2. Scalar Quantization (SQ8)
- **4x Compression**: Vectors are stored as 8-bit integers (uint8) instead of 32-bit floats.
- **Disk Efficiency**: Reduces storage footprint significantly (e.g., 60MB → 16MB for 10k vectors).
- **Latency**: Higher than HNSW (10-45ms) but ideal for read-heavy, storage-constrained environments.

### 3. Dynamic Dimension Support
- **Unrestricted Dimensions**: SrvDB now supports any vector dimension from **128 to 4096** at runtime.
- **Models**: Verified support for OpenAI (1536), Cohere (1024), Nomic (768), and MiniLM (384).
- **Zero-Config**: Dimension is inferred from the first added batch or explicitly set.

## ✨ What's New

### Auto-Tuning Strategy ("Adaptive Core")
SrvDB can now automatically select the best indexing strategy based on your hardware and dataset size:
```python
db = srvdb.SrvDBPython("./db", mode="auto")
# Automatically selects:
# - Flat: < 50k vectors
# - HNSW: > 50k vectors (High RAM)
# - SQ8: > 50k vectors (Low RAM)
```

### Python API Extensions
- **IVF Training**:
  ```python
  db.set_mode("ivf")
  db.configure_ivf(nlist=1024, nprobe=16)
  db.train_ivf()
  ```
- **Explicit Configuration**: `SrvDBPython(path, dimension=1536, mode="flat")`.

## 📊 Benchmarks

**Environment**: Consumer Linux Laptop (x86_64, 5.6GB RAM)

| Mode | Ingestion (vec/s) | Latency P99 (ms) | Recall@10 | Disk Usage (100k) |
|------|-------------------|------------------|-----------|------------------|
| **Flat** | 8,211 | 4.67 | 99.9% | 600 MB |
| **HNSW** | 8,772 | 6.73 | 99.9% | 620 MB |
| **IVF-HNSW** | 10,900 | 19.0 (P99) | 100% | ~630 MB |
| **SQ8** | 9,560 | 45.79 | 92.4% | **160 MB** |

## 🔧 Technical Improvements

- **Zero-Copy Persistence**: Leverages `mmap` for instant database startup.
- **Concurrent Access**: `Rayon` parallelism used for IVF partition searching.
- **Storage Layer Refactor**: Unified `VectorStorage` trait supporting both `f32` and `u8` backends.

## 🔄 Migration Guide

### Breaking Changes
1. **Constructor Signature**: `SrvDB::new` now requires a `dimension` parameter (or infers it).
   ```python
   # Old
   db = srvdb.SrvDB("./path")
   # New
   db = srvdb.SrvDBPython("./path", dimension=1536)
   ```
2. **File Format**: The storage format has changed to support dynamic headers. v0.1.x databases are **not compatible** and must be re-indexed.

### 📦 Installation
```bash
pip install srvdb==0.2.0
```

## 👥 Contributors
- Srinivas Nampalli (@Srinivas26k)
