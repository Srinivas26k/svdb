# Optimization & Tuning

SrvDB provides powerful configuration options to tune the performance of your vector index. The most critical tuning happens when using the **HNSW** (Graph) mode.

## HNSW Parameters

When creating an HNSW index, there are three key parameters that control the trade-off between **Recall (Accuracy)**, **Speed**, and **Memory Size**.

```python
db = srvdb.SrvDBPython.new_hnsw(
    path="...",
    dimension=1536,
    m=16,                # Max connections per node
    ef_construction=200, # Build-time accuracy
    ef_search=50         # Runtime accuracy (can be changed later)
)
```

### 1. `m` (Max Connections)
- **What it is**: The maximum number of edges (neighbors) each node in the graph can have.
- **Typical Values**: 16, 32, 64.
- **Effect**:
    - **Higher**: Better recall, but higher memory usage and slightly slower construction.
    - **Lower**: Lower memory, faster build.
- **Recommendation**: `16` is a good default. Use `32` for high dimensionality (1000+) or hard datasets.

### 2. `ef_construction` (Build Depth)
- **What it is**: The size of the dynamic candidate list during graph construction. "How hard we try to find the best neighbors when inserting."
- **Typical Values**: 100, 200, 500.
- **Effect**:
    - **Higher**: Better graph quality (higher recall later), but significantly slower indexing (ingestion).
    - **Lower**: Fast indexing, but search might miss neighbors.
- **Recommendation**: `200` is a safe default. Use `400+` if index build time is not a concern but search accuracy is critical.

### 3. `ef_search` (Search Depth)
- **What it is**: The size of the candidate list during *search*.
- **Typical Values**: 50, 100, 200.
- **Effect**:
    - **Higher**: Higher recall, slower search latency.
    - **Lower**: Ultra-fast search, potential for lower accuracy.
- **Note**: This parameter can be changed at **runtime** without rebuilding the index!

```python
# Tune at runtime based on your needs
db.set_ef_search(100) # High accuracy
db.search(...)

db.set_ef_search(20)  # High speed
db.search(...)
```

## Performance Tips

### Batch Ingestion
Always use `add_batch` instead of `add` in a loop.
- **Why**: `add_batch` minimizes Python <-> Rust context switching overhead and allows internal optimizations.

```python
# Bad
for vec in existing:
    db.add(...)

# Good
db.add_batch(ids, vectors, metadata)
```

### Memory Management
If you are running out of RAM:
1. **Switch to PQ**: `new_product_quantized` reduces memory by 32x.
2. **Reduce `m`**: Lowering `m` from 32 to 16 cuts the graph memory overhead in half.
3. **Use `persist()`**: SrvDB uses mmap. Ensure you call `persist()` to flush dirty pages to disk, letting the OS manage RAM more effectively.
