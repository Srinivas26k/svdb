# Indexing Modes

SrvDB is a multi-modal vector database. Unlike most databases that force a single indexing strategy, SrvDB allows you to choose the best trade-off between **recall** (accuracy), **latency** (speed), and **memory usage**.

## Summary Table

| Mode | Search Algo | Recall | Latency | Memory | Best For |
|------|-------------|--------|---------|--------|----------|
| **Flat** | Exact (Linear) | 100% | High (O(n)) | High | < 50k vectors, Perfect accuracy |
| **HNSW** | Graph (Approx) | 99%+ | Ultra Low | High | > 50k vectors, Real-time Search |
| **SQ8** | Exact (Linear) | ~92% | High | Low (4x) | Archival, Disk-heavy apps |
| **PQ** | Graph + Compr. | ~90% | Low | Ultra Low (32x) | Edge devices, 1M+ vectors |
| **IVF** | Partition + Graph | ~95% | Low | Moderate | Distributed/v.Large datasets |

---

## 1. Flat Index (Default)

**Ideally for: Small datasets (< 50k vectors) or when 100% accuracy is required.**

The Flat index stores full-precision (`f32`) vectors and performs a brute-force linear scan using SIMD-accelerated Cosine Similarity.

- **Pros**: 100% Recall. No training required. Simple.
- **Cons**: Slowest search speed for large datasets. High memory usage (RAM).

```python
# Initialize Flat Mode
db = srvdb.SrvDBPython("path/to/db", dimension=1536, mode="flat")
```

## 2. HNSW Graph Index

**Ideally for: General production use, RAG, and large datasets (50k - 10M vectors).**

[Hierarchical Navigable Small World (HNSW)](https://arxiv.org/abs/1603.09320) represents vectors as nodes in a graph. Search works by traversing the graph from top layers (long jumps) to bottom layers (local neighborhood).

- **Pros**: Extremely fast search (sub-millisecond). Very high recall (>99%).
- **Cons**: High memory usage (vectors + graph connections). Slower ingestion (indexing takes time).

```python
# Initialize HNSW Mode with custom parameters
# m = max connections per node
# ef_construction = depth of search during build
db = srvdb.SrvDBPython.new_hnsw(
    "path/to/hnsw_db", 
    dimension=1536, 
    m=16, 
    ef_construction=200, 
    ef_search=50
)
```

## 3. Product Quantization (HNSW + PQ)

**Ideally for: Memory-constrained environments (Edge, Raspberry Pi) or massive datasets.**

Product Quantization (PQ) compresses vectors by breaking them into sub-vectors and clustering them. SrvDB combines this with an HNSW graph for navigation.

- **Pros**: Massive memory reduction (32x compression). Fast search.
- **Cons**: Lower recall (approx 85-95%). Requires "training" phase to learn clusters.

```python
# PQ requires a training set of vectors
db = srvdb.SrvDBPython.new_product_quantized(
    "path/to/pq_db",
    dimension=1536,
    training_vectors=my_sample_vectors
)
```

> **Warning**: PQ can exhibit recall loss on highly clustered semantic data (e.g., many documents about the same topic). Test your recall for RAG apps.

## 4. Scalar Quantization (SQ8)

**Ideally for: Disk-heavy workloads where RAM is scarce but exact-like search is needed.**

Converts 32-bit floats to 8-bit integers.

- **Pros**: 4x memory reduction.
- **Cons**: Slower search than HNSW (still linear scan).

## 5. IVF (Inverted File Index)

**Ideally for: Extremely large datasets (1M+).**

Partitions the vector space into "Voronoi cells". Search only checks cells that are close to the query.

- **Pros**: Good balance of speed and recall for massive scales.
- **Cons**: More complex setup (requires training partitions).
