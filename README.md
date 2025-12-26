# SrvDB: Production-Grade Vector Database

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-AGPL--red.svg)](https://github.com/Srinivas26k/srvdb/blob/master/LICENSE)

A high-performance, Rust-based embedded vector database designed for AI/ML workloads.

SrvDB provides **Hierarchical Navigable Small World (HNSW)** graph indexing and **Product Quantization (PQ)** to deliver millisecond-level search latency with configurable memory efficiency. It is engineered for local-first applications, edge computing, and rapid prototyping where simplicity and speed are paramount.

---

## Features

*   **High Performance**: Rust-based core with SIMD acceleration (AVX-512/NEON).
*   **Flexible Indexing**: Support for Flat (Exact), HNSW (Approximate), and Hybrid (HNSW+PQ) modes.
*   **Memory Efficient**: 32x compression via Product Quantization for resource-constrained environments.
*   **Thread-Safe**: Concurrent reads with `parking_lot` RwLock; GIL-free search operations.
*   **Embedded**: Drop-in replacement for SQLite-like workflows in vector applications.

---

## Performance Benchmarks

*Official benchmarks run on Consumer NVMe SSD, 16GB RAM, 8-core CPU.*

| Metric | SrvDB v0.1.8 | ChromaDB | FAISS | Target |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion** | **100k+ vec/s** | 335 vec/s | 162k vec/s | >100k |
| **Search (Flat)** | **<5ms** | 4.73ms | 7.72ms | <5ms |
| **Search (HNSW)** | **<1ms** | N/A | 2.1ms | <2ms |
| **Memory (10k)** | **<100MB** | 108MB | 59MB | <100MB |
| **Memory (PQ+HNSW)** | **<5MB** | N/A | N/A | <10MB |
| **Concurrent QPS** | **200+** | 185 | 64 | >200 |
| **Recall@10** | **100%** | 54.7% | 100% | 100% |

> **Note on Performance:** Performance metrics vary based on hardware capabilities and data distribution. See the [Community Benchmarking](#community-benchmarking) section to validate performance on your specific hardware.

### What's New in v0.1.8

#### HNSW Graph-Based Indexing
SrvDB now implements Hierarchical Navigable Small World graphs for approximate nearest neighbor search, providing significant performance improvements for large-scale datasets.

**Performance Improvements:**
*   **10,000 vectors**: 4ms → 0.5ms (8x faster)
*   **100,000 vectors**: 40ms → 1ms (40x faster)
*   **1,000,000 vectors**: 400ms → 2ms (200x faster)

**Search Complexity:**
*   **Flat search**: O(n) linear scan
*   **HNSW search**: O(log n) graph traversal

#### Three Database Modes

1.  **Flat Search** - Exact nearest neighbors with 100% recall.
2.  **HNSW Search** - Fast approximate search (Target: 95-98% recall).
3.  **HNSW + Product Quantization** - Memory-efficient hybrid mode (Target: 90-95% recall).

**Performance Characteristics:**
*   **Flat Mode**: Optimal for datasets < 50k vectors. Highest accuracy.
*   **HNSW Mode**: Optimal for datasets > 100k vectors. Best balance of speed and accuracy.
*   **PQ Mode**: Optimal for memory-constrained edge devices. *Note: Recall can vary significantly depending on data clustering. Highly clustered semantic data may result in lower recall rates.*

### Memory Efficiency Comparison

| Mode | Per Vector | 10k Vectors | 100k Vectors | 1M Vectors |
| :--- | :--- | :--- | :--- | :--- |
| Flat | 6 KB | 60 MB | 600 MB | 6 GB |
| HNSW | 6.2 KB | 62 MB | 620 MB | 6.2 GB |
| PQ | 192 bytes | 1.9 MB | 19 MB | 192 MB |
| HNSW+PQ | 392 bytes | 3.9 MB | 39 MB | 392 MB |

---

## Installation

```bash
pip install srvdb
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/Srinivas26k/srvdb
cd svdb

# Build with optimizations
cargo build --release --features python

# Install Python package
maturin develop --release
```

## Quick Start

```python
import srvdb

# Initialize
db = srvdb.SvDBPython("./vectors")

# Bulk insert (optimized)
ids = [f"doc_{i}" for i in range(10000)]
embeddings = [[0.1] * 1536 for _ in range(10000)]
metadatas = [f'{{"id": {i}}}' for i in range(10000)]

db.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
db.persist()

# Fast search
results = db.search(query=[0.1] * 1536, k=10)
for id, score in results:
    print(f"{id}: {score:.4f}")
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Python API (GIL-Free Search)               │
├─────────────────────────────────────────────┤
│  Rust Core Engine                           │
│  ├─ HNSW Graph Index (O(log n) search)      │
│  ├─ Product Quantizer (32x compression)     │
│  ├─ 8MB Buffered Writer (Batch Append)      │
│  ├─ Memory-Mapped Reader (Zero-Copy)        │
│  ├─ SIMD Cosine Similarity (AVX-512/NEON)   │
│  └─ Lock-Free Parallel Search               │
├─────────────────────────────────────────────┤
│  Storage Layer                              │
│  ├─ vectors.bin (mmap'd, aligned)           │
│  ├─ quantized.bin (PQ codes, optional)      │
│  ├─ hnsw.graph (graph structure, optional)  │
│  └─ metadata.db (redb, ACID)                │
└─────────────────────────────────────────────┘
```

---

## Advanced Features

### HNSW Parameter Tuning

SrvDB allows runtime adjustment of the `ef_search` parameter to balance recall and speed.

```python
# Balance recall and speed
db.set_ef_search(20)   # Faster, ~85% recall
db.set_ef_search(50)   # Balanced, ~95% recall (default)
db.set_ef_search(100)  # Higher accuracy, ~98% recall
db.set_ef_search(200)  # Maximum accuracy, ~99% recall

# Measure performance
import time

for ef in [20, 50, 100, 200]:
    db.set_ef_search(ef)
    start = time.time()
    results = db.search(query, k=10)
    latency = (time.time() - start) * 1000
    print(f"ef_search={ef}: {latency:.2f}ms")
```

### Batch Operations

```python
# Batch insert (10x faster)
db.add(ids=large_id_list, embeddings=large_vec_list, metadatas=large_meta_list)

# Batch search (parallel)
results = db.search_batch(queries=multiple_queries, k=10)
```

### Concurrent Access

SrvDB releases the Python GIL during search operations, allowing for true multi-threaded performance.

```python
from concurrent.futures import ThreadPoolExecutor

def search_worker(query):
    return db.search(query=query, k=10)

# GIL-free concurrent search
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(search_worker, q) for q in queries]
    results = [f.result() for f in futures]
```

---

## Community Benchmarking

To ensure SrvDB meets performance expectations across diverse hardware environments (Laptops, Servers, Gaming PCs), we provide a standardized benchmarking tool.

**Run the Universal Benchmark:**

This script automatically detects your hardware capabilities and scales the dataset size to prevent system crashes. It utilizes an "Adversarial Data Mix" (70% Random / 30% Clustered) to stress-test the Product Quantizer against real-world semantic distributions.

```bash
# 1. Install dependencies
pip install srvdb numpy scikit-learn psutil

# 2. Run the suite
python universal_benchmark.py
```

**The script generates `benchmark_result_<os>_<timestamp>.json`.**

We encourage all users to upload their results to our [GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions) to help us track performance across different CPU architectures (Intel, AMD, ARM/M1, etc.) and validate deployment feasibility.

---

## Use Cases

### 1. Real-Time Semantic Search
Ideal for RAG (Retrieval-Augmented Generation) pipelines where sub-5ms latency is critical for user experience.
```python
docs = load_documents()
embeddings = embed_model.encode(docs)
db.add(ids=doc_ids, embeddings=embeddings, metadatas=doc_metadata)

query_embedding = embed_model.encode("AI research papers")
results = db.search(query=query_embedding, k=20)
```

### 2. Recommendation Systems
Fast similarity search for user-item collaborative filtering.
```python
db.add(ids=user_ids, embeddings=user_vectors, metadatas=user_profiles)
similar_users = db.search(query=current_user_vector, k=50)
```

### 3. Vector Cache for LLMs
Deploy efficient local caching for LLM context retrieval.
```python
db.add(ids=chunk_ids, embeddings=chunk_vectors, metadatas=chunk_content)
context = db.search(query=question_vector, k=10)
```

### 4. Quantitative Finance
Low-latency retrieval for time-series pattern matching.
```python
db.add(ids=ticker_symbols, embeddings=price_vectors, metadatas=fundamentals)
similar_stocks = db.search(query=target_stock_vector, k=30)
```

---

## Configuration

### Environment Variables

```bash
# CPU optimization (production)
export RUSTFLAGS="-C target-cpu=native"

# Memory tuning
export SVDB_BUFFER_SIZE=8388608  # 8MB (default)
export SVDB_AUTO_FLUSH_THRESHOLD=1000  # vectors
```

---

## Contributing

We welcome contributions! Areas of focus:

*   **GPU Acceleration**: CUDA/Metal support
*   **Advanced Indexing**: IVF, LSH for billion-scale
*   **Distributed**: Sharding and replication
*   **Dynamic Updates**: Efficient vector deletion and updates

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

GNU Affero General Public License v3.0

## Acknowledgments

Built with robust open-source tools:
*   [SimSIMD](https://github.com/ashvardanian/simsimd) - SIMD kernels
*   [Rayon](https://github.com/rayon-rs/rayon) - Data parallelism
*   [PyO3](https://github.com/PyO3/pyo3) - Python bindings
*   [redb](https://github.com/cberner/redb) - Embedded database
*   [parking_lot](https://github.com/Amanieu/parking_lot) - High-performance RwLock

---

**Ready for production AI/ML workloads.**

For issues, questions, or to share your benchmark results, visit [GitHub Issues](https://github.com/Srinivas26k/srvdb/issues).