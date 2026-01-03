# SrvDB: Embedded Vector Database for Offline AI Applications

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-AGPL-red.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/index.md)

**[📚 Read the Full Documentation](docs/index.md)**


SrvDB is a Rust-based embedded vector database designed for offline and edge AI deployments. It provides exact nearest neighbor search with a focus on simplicity, deterministic behavior, and zero external dependencies.

---

## Design Philosophy

**Target Use Cases:**
- **RAG (Retrieval-Augmented Generation)**: Local knowledge bases where accuracy is critical.
- **Edge Deployment**: Raspberry Pi, mobile devices, and air-gapped systems.
- **Desktop Analytics**: High-throughput batch processing on consumer hardware.
- **Offline Applications**: Environments where network connectivity cannot be assumed.

**Core Principles:**
- **Simplicity**: Drop-in deployment (`pip install srvdb`) with zero configuration required.
- **Accuracy First**: Exact search modes guarantee 100% recall, ensuring reliable AI responses.
- **Resource Efficiency**: Optimized for low-memory CPUs and limited storage bandwidth.

---

## Architecture

```
Storage Layer          Index Layer             API Layer
---------------        -----------------       --------------
| vectors.bin |        | Flat Index    |       | Python API |
| (mmap)      |        | (Exact)       |       | (PyO3)     |
|             |        |               |       | (Native)   |
| metadata.db |        | HNSW Graph    |       | Rust API   |
| (redb)      |        | (Approximate) |       |            |
|             |        |               |       |            |
----------------       ----------------        --------------
     ^                      ^                       ^
  SIMD Accel          Thread Safety         GIL-Free Search
(AVX-512/NEON)        (parking_lot)          (Concurrent)
```

---

## Performance Characteristics

### Benchmarks on M1 MacBook (16GB RAM, Apple Silicon)

| Mode | Ingestion | Search Latency (P99) | Memory (RAM) | Disk Usage | Recall@10 |
|------|-----------|---------------------|--------------|------------|-----------|
| Flat | 23,978 vec/s | 11.2ms | 78MB | 594MB | 99.9% |
| HNSW | 23,562 vec/s | 10.6ms | 21MB | 594MB | 99.9% |
| HNSW+PQ | 4,613 vec/s | 3.5ms | -79MB* | 28MB | 13.4%** |

*Negative value indicates memory reclamation during PQ training  
**PQ recall degrades significantly on clustered semantic data**

### Benchmarks on Consumer Linux Laptop (5.6GB RAM, x86_64)

| Mode | Ingestion | Search Latency (P99) | Memory (RAM) | Disk Usage | Recall@10 |
|------|-----------|---------------------|--------------|------------|-----------|
| Flat | 8,211 vec/s | 4.67ms | 42.16MB | 59.52MB | 99.9% |
| HNSW | 8,772 vec/s | 6.73ms | -1.01MB | 59.52MB | 99.9% |
| SQ8 (Scalar Quantization) | 9,560 vec/s | 45.79ms | 61.49MB | 16.03MB | 92.4% |
| SQ8 (IVF-HNSW) | 2,142 vec/s | 5.98ms | 43.38MB | 16.03MB | 92.4% |

**Observation**: The SQ8 mode offers 4x disk compression (60MB -> 16MB) but introduces significant latency (45ms). The IVF-HNSW variant attempts to mitigate this by combining compression with graph indexing but remains latency-bound compared to pure graph approaches.

---

## Installation

```bash
pip install srvdb
```

### Build from Source

```bash
git clone https://github.com/Srinivas26k/srvdb
cd srvdb
cargo build --release --features python
maturin develop --release
```

---

## Quick Start

```python
import srvdb
import numpy as np

# Initialize database (Default: 1536-dim, Flat mode)
db = srvdb.SrvDBPython("./vectors", dimension=1536, mode="flat")

# Bulk insert
ids = [f"doc_{i}" for i in range(10000)]
embeddings = np.random.randn(10000, 1536).astype(np.float32)
metadatas = [f'{{"id": {i}}}' for i in range(10000)]

db.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)
db.persist()

# Fast search
results = db.search(query=[0.1] * 1536, k=10)

for id, score in results:
    print(f"{id}: {score:.4f}")
```

---

## Supported Embedding Models

**Supported Dimensions: 128 - 4096 (Runtime)**

**Compatible Models:**
- OpenAI `text-embedding-ada-002` (1536-dim)
- OpenAI `text-embedding-3-small` (1536-dim)
- Nomic `nomic-embed-text-v1` (768-dim, 1536-dim variant)
- Cohere `embed-english-v3.0` (1024-dim, unsupported) - *Use workaround below*

**Workaround for Non-1536 Embeddings:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
texts = ["sample text 1", "sample text 2"]
embeddings = model.encode(texts)  # Shape: (2, 384)

# Pad to 1536 dimensions
padded = np.pad(embeddings, ((0, 0), (0, 1536 - 384)), mode='constant')
padded = padded.astype(np.float32)

# Now compatible with SrvDB
db.add(
    ids=["id1", "id2"],
    embeddings=padded.tolist(),
    metadatas=['{"source": "text1"}', '{"source": "text2"}']
)
```

---

## Indexing Modes

### 1. Flat Index (Exact Search)
**Description:** Brute-force linear scan with SIMD-accelerated cosine similarity.

**When to Use:**
- Datasets < 50,000 vectors.
- 100% recall required.
- Predictable latency needed.

**Characteristics:**
- Time Complexity: O(n)
- Space Complexity: 6KB per vector (1536-dim x 4 bytes)
- Recall: 100% (exact)

### 2. HNSW Graph Index
**Description:** Hierarchical Navigable Small World graph for O(log n) search.

**When to Use:**
- Datasets > 100,000 vectors.
- Sub-millisecond latency required.
- 95-99% recall acceptable.

**Characteristics:**
- Time Complexity: O(log n)
- Space Complexity: ~6.2KB per vector (graph overhead: ~200 bytes)
- Recall: 95-99.9% (configurable via `ef_search`)

### 3. HNSW + Product Quantization (Memory-Efficient Hybrid)
**Description:** Combines HNSW with 32x vector compression.

**When to Use:**
- Memory-constrained environments (edge devices).
- Dataset > 500,000 vectors.
- 85-95% recall acceptable.

**Characteristics:**
- Time Complexity: O(log n)
- Space Complexity: ~392 bytes per vector (32x compression)
- Recall: 90-95% (dependent on data distribution)

**Critical Limitation:** PQ mode exhibits recall degradation (13-20%) on clustered semantic data (e.g., document embeddings from the same topic). It is safe for uniformly distributed data but **not recommended for RAG applications** unless high recall loss is acceptable.

### 4. SQ8 (Scalar Quantization)
**Description:** Stores vectors as 8-bit integers on disk with 32x compression. Optimized for read-heavy workloads.

**When to Use:**
- Read-mostly workloads (archival, backups).
- Disk storage is at a premium.

**Characteristics:**
- Time Complexity: O(n) (Linear scan)
- Space Complexity: ~1.5KB per vector (32-bit integer)
- Recall: 92.4% (Adversarial Mix: 70% Random / 30% Clustered)
- Latency: Significantly higher than HNSW (10-45ms P99 at 50k vectors).

### 5. IVF-HNSW (Scalable Inverted File)
**Description:** Partitions vector space into Voronoi cells (Inverted File) and refines candidates using HNSW. Optimized for datasets exceeding 1M vectors.

**When to Use:**
- Massive scale datasets (1M+ vectors).
- Sub-10ms latency requirement (P99).
- High recall (90-99%) maintained at scale.

**Architecture:**
- **Coarse Search:** Identifies closest Voronoi cell centers (partitions) for the query.
- **Fine Search:** Performs HNSW graph traversal only within the selected partitions.
- **Parallelism:** Uses Rayon for concurrent partition access.

**Performance:**
- Latency: ~6ms P50 at 100k vectors.
- Scalability: Linear throughput up to 1M vectors.
- Memory: Low overhead per partition compared to full graph.

---

## API Reference

### Initialization

### Initialization

```python
# Standard Initialization
# modes: 'flat', 'hnsw', 'sq8', 'pq', 'ivf', 'auto'
db = srvdb.SrvDBPython(path="db_path", dimension=1536, mode="flat")

# HNSW Mode with Custom Parameters
db = srvdb.SrvDBPython.new_hnsw(
    path="db_hnsw",
    dimension=1536,
    m=16,
    ef_construction=200,
    ef_search=50
)

# Scalar Quantization (SQ8) - Requires training data
# 4x compression, good for large datasets on disk
db = srvdb.SrvDBPython.new_scalar_quantized(
    path="db_sq8",
    dimension=1536,
    training_vectors=vectors_list # List[List[float]]
)

# Product Quantization (PQ) - Requires training data
# 32x compression, optimized for edge/low-memory
db = srvdb.SrvDBPython.new_product_quantized(
    path="db_pq",
    dimension=1536,
    training_vectors=vectors_list
)
```

### Operations

```python
# Insert vectors
db.add(
    ids: List[str],
    embeddings: List[List[float]],  # 1536-dim default
    metadatas: List[str]             # JSON strings
) -> int

# Search
db.search(
    query: List[float],  # 1536-dim
    k: int
) -> List[Tuple[str, float]]

# Batch search (parallel)
db.search_batch(
    queries: List[List[float]],
    k: int
) -> List[List[Tuple[str, float]]]

# HNSW runtime tuning
db.set_ef_search(ef: int) -> None  # HNSW and IVF modes only

# Get metadata
db.get(id: str) -> Optional[str]

# Count vectors
db.count() -> int

# Persist to disk
db.persist() -> None

# IVF specific training
# Switch to IVF mode first: db.set_mode("ivf")
db.configure_ivf(nlist=100, nprobe=10)
db.train_ivf() # Uses existing data in DB to train
```

---

## Performance Tuning

### HNSW Parameters

```python
# High accuracy (slower, more memory)
db = srvdb.SrvDBPython.new_with_hnsw(
    path,
    m=32,              # More connections
    ef_construction=500,
    ef_search=200
)

# Balanced (recommended)
db = srvdb.SrvDBPython.new_with_hnsw(
    path,
    m=16,
    ef_construction=200,
    ef_search=50
)

# Fast (lower accuracy)
db = srvdb.SrvDBPython.new_with_hnsw(
    path,
    m=8,
    ef_construction=100,
    ef_search=20
)
```

---

## Benchmarking Your Hardware

We provide a standardized benchmark script to validate performance on your specific hardware.

```bash
pip install srvdb numpy scikit-learn psutil
python universal_benchmark.py
```

The script automatically detects available RAM, adjusts dataset size (10k-1M vectors), and uses an adversarial data mix (70% random / 30% clustered) to stress-test quantization accuracy.

**Community Contribution:**  
Share your results in [GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions) to help validate performance across different CPU architectures (Intel, AMD, ARM/M1).

---

## Known Limitations

### Critical Limitations

1.  **Dimensionality:** v0.2.0 fully supports dynamic dimensions (128-4096).
2.  **Product Quantization (PQ) Recall:** PQ mode exhibits severe recall degradation (13-20%) on clustered semantic data (e.g., RAG documents). It is recommended to use Flat or HNSW modes for semantic tasks.
3.  **Concurrent Write Contention:** Single-writer design; multiple processes cannot write simultaneously.
4.  **No Dynamic Updates:** Vector deletion/update requires index rebuild.

### Minor Limitations

5.  **SQ8 Latency:** Scalar Quantization (SQ8) is optimized for disk compression and read-heavy workloads but incurs significant latency (10-45ms) for search.
6.  **Memory Measurement:** Benchmark reports may show inconsistent memory deltas due to OS caching behavior.

---

## Future Work

### v0.2.1 Roadmap (Q1 2026)

**High Priority:**
- [ ] Incremental vector updates/deletion.
- [ ] Async I/O for ingestion.
- [ ] Memory optimization (target: <100MB for 10k vectors).

**Medium Priority:**
- [ ] IVF optimization (OPQ - Optimized Product Quantization) to solve PQ recall loss on clustered data.
- [ ] Filtered search (metadata-based pre-filtering).
- [ ] GPU acceleration (CUDA/Metal) for SIMD operations.

**Low Priority:**
- [ ] Distributed sharding (multi-node deployment).
- [ ] Approximate quantization (scalar/binary).

---

## Contributing

We invite you to help shape the future of offline vector search.

**Areas of Focus:**
1.  **Algorithm Improvements:** Better clustering for PQ/OPQ, alternative indexing structures (IVF, LSH).
2.  **Engineering:** Dimension flexibility, async I/O refactoring, memory profiling.
3.  **Testing:** Benchmark validation on diverse hardware and integration with popular embedding models.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

---

## License

SrvDB is open-source software licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

### 🟢 Open Source Use
You are free to use SrvDB for personal projects, academic research, and open-source applications under the terms of the AGPLv3.

### 🔴 Commercial & Cloud Use
If you are building a proprietary application, a cloud service (SaaS), or embedding SrvDB in a commercial product where you cannot or do not wish to open-source your code, you must purchase a **Commercial License**.

**Commercial Licensing Benefits:**
* Exemption from AGPLv3 open-source requirements.
* Priority support & direct access to the maintainer.
* Legal assurance for enterprise deployment.

For commercial licensing inquiries, please contact: **srinivasvarma764@gmail.com**

---

## Acknowledgments

SrvDB relies on the following open-source projects:

- [SimSIMD](https://github.com/ashvardanian/simsimd) - SIMD distance kernels
- [Rayon](https://github.com/rayon-rs/rayon) - Data parallelism
- [PyO3](https://github.com/PyO3/pyo3) - Python-Rust bindings
- [redb](https://github.com/cberner/redb) - Embedded key-value store
- [parking_lot](https://github.com/Amanieu/parking_lot) - Fast synchronization primitives

---

## Support

- **Issues:** [GitHub Issues](https://github.com/Srinivas26k/srvdb/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions)
- **Email:** srinivasvarma764@gmail.com

For bug reports or feature requests, please include system information and minimal reproduction code.