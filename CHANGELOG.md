# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-30

### 🚀 Major Features
- **Dynamic Dimensions Support**: SrvDB now supports vector dimensions from 128 to 4096 (determined at runtime) instead of compile-time fixed dimensions.
    - Updated `Vector`, `VectorStorage`, and `QuantizedVectorStorage` to use `Vec<f32>`/`&[f32]`.
    - Removed `const N: usize` generic parameters throughout the codebase.
- **True SQ8 Disk Compression**:
    - Refactored `ScalarQuantizedStorage` to store vectors as `u8` bytes on disk.
    - Achieves **4x raw compression** (32-bit float → 8-bit integer) compared to uncompressed storage.
    - Observed ~2.6x to ~4x effective reduction in total index size depending on metadata overhead.
    - Maintains **100% recall** on tested datasets.
- **HNSW Indexing Integration**:
    - Added initial support for HNSW graph-based indexing for faster search (O(log n)).
- **IVF-HNSW Hybrid Indexing**:
    - Implemented **Inverted File System (IVF)** with HNSW refinement (`IVFIndex`).
    - Uses **K-Means Clustering** to partition vector space into Voronoi cells.
    - Two-stage search: **Coarse Search** (Centroids) + **Fine Search** (HNSW within partitions).
    - Scalable to **100k+ vectors** with sub-20ms latency and minimal memory overhead.
    - Python API: `db.set_mode("ivf")`, `db.configure_ivf()`, `db.train_ivf()`.

### 🐛 Bug Fixes
- Fixed critical bug where SQ8 storage was wrapping explicit `f32` floats, negating compression benefits.
- Fixed `panic` in `select_top_k` when candidate set was empty.
- Fixed routing logic in `SrvDB::add`, `SrvDB::search`, and `SrvDB::persist` to correctly handle `ScalarQuantizedStorage`.
- Fixed multiple compilation errors related to type mismatches and missing imports during the v0.2.0 upgrade.
- Fixed Python binding initialization for SQ8 mode (`new_scalar_quantized`).
- Fixed **HNSW Panic** where inserting a node into a new top layer attempted to link to non-existent nodes.
- Fixed persistence race condition where training data was not visible due to missing flush.

### ⚡ Performance
- **Ingestion**: Optimized `add_batch` implementation for SQ8, achieving ~25k vectors/sec on standard hardware.
- **Zero-Copy**: Leveraged memory mapping (`mmap`) for instant startup and low memory footprint in SQ8 mode.

### 🛠 breaking Changes
- `SrvDB::new` and related constructors now require a `dimension` argument (or infer it from config).
- Python API updated to accept `dimension` in constructor.
- Data format change: v0.2.0 storage format is not backward compatible with v0.1.x due to dynamic dimension header changes.

## [0.1.9] - Previous Versions
- Initial release with Flat Indexing.
- Basic Python bindings.
- Fixed-dimension support.
