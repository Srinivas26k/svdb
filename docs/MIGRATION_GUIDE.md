# SrvDB v0.2.0 Migration Guide

This guide helps you upgrade from SrvDB v0.1.x to v0.2.0. This release introduces significant architecture changes for better performance and flexibility.

## Major Changes

- **Renaming**: `SvDB` is now `SrvDB`. Python class is `SrvDBPython`, Rust struct is `SrvDB`.
- **Dynamic Dimension Support**: No longer hardcoded to 1536. You must specify dimensions explicitly.
- **Module Restructuring**: Code has been moved to cleaner domain-specific modules (`core`, `index`, `storage`, `api`).
- **New Index Types**: Added IVF (Hybrid) and SQ8 quantization support.

## Rust API Changes

### 1. Initialization
Old:
```rust
// v0.1.x (Hardcoded 1536 dimensions)
let mut db = SrvDB::new("/path/to/db").unwrap();
```

New:
```rust
// v0.2.0
use srvdb::core::types::DatabaseConfig;

let config = DatabaseConfig::new(1536)?; // Specify dimension
let mut db = srvdb::SrvDB::new_with_config("/path/to/db", config)?;
```

### 2. Module Imports
The crate exports have changed. Update your use statements:

| Old Path | New Path |
| :--- | :--- |
| `srvdb::Vector` | `srvdb::core::types::Vector` or `Vec<f32>` (raw) |
| `srvdb::IndexMode` | `srvdb::core::strategy::IndexMode` |
| `srvdb::quantization` | `srvdb::utils::quantization` |

### 3. Vector Type
We moved away from a custom `EmbeddedVector` to standard `Vec<f32>` for better interoperability.
```rust
// Adding data
let vec: Vec<f32> = vec![0.0; 1536]; // Standard vector
db.add(vec, meta)?;
```

## Python API Changes

### 1. Initialization
You must now provide the `dimension` argument if you are not using the default (1536 is default but explicit is better).

Old:
```python
# v0.1.x
db = srvdb.SrvDBPython("./db")
```

New:
```python
# v0.2.0
# Support for 384, 768, 1536, etc.
db = srvdb.SrvDBPython("./db", dimension=768)
```

### 2. New Modes
You can explicitly set modes like SQ8 or HNSW during init or runtime:

```python
# Create SQ8 compressed database (4x smaller)
db = srvdb.SrvDBPython.new_scalar_quantized("./db", 768, training_vectors)

# Create HNSW index
db = srvdb.SrvDBPython("./db", 768, mode="hnsw")
```

## Troubleshooting

- **"Dimension mismatch"**: Ensure the `dimension` passed to constructor matches your data.
- **"Vector storage not initialized"**: Some modes (like purely in-memory PQ) might behave differently, but `Flat`, `HNSW`, `SQ8` use disk persistence. Ensure you call `db.persist()` before terminating if you want to ensure data safety, though search usually auto-persists.

## Support
Open an issue on GitHub if you encounter problems upgrading.
