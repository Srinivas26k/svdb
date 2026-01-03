# Benchmarking

Performance varies significantly depending on CPU (Intel vs ARM, AVX-512 support), RAM speed, and disk I/O. We include a universal benchmark script to help you validate SrvDB's performance on your specific hardware.

## Running the Benchmark

SrvDB comes with a benchmark script that:
1. Generates synthetic data (Random + Clustered mix).
2. Indexes it using different modes (Flat, HNSW).
3. Measures Ingestion Speed, Search Latency (P50, P99), and Recall.

### Step 1: Install Dependencies
```bash
pip install srvdb numpy scikit-learn psutil
```

### Step 2: Download/Run the Script
(If you cloned the repo, this script is in `universal_benchmark.py`).

```python
# Save this as benchmark.py and run it
import srvdb
# ... (or use the one provided in the repository)
```

**Recommended**: Run the official script from the repository:
```bash
curl -O https://raw.githubusercontent.com/Srinivas26k/srvdb/main/universal_benchmark.py
python universal_benchmark.py
```

## Interpreting Results

You will see output similar to this:

```
=== System Info ===
OS: Linux
CPU Cores: 16
RAM: 32.0 GB
SIMD: AVX-512 Detected

=== Benchmark Results (100,000 vectors) ===

| Mode | Ingestion (vec/s) | Search P99 (ms) | Recall@10 | Memory (MB) |
|------|-------------------|-----------------|-----------|-------------|
| Flat | 23,000            | 11.2            | 100.0%    | 78          |
| HNSW | 21,000            | 0.8             | 99.9%     | 120         |
```

- **Ingestion**: Higher is better. >10k is good for Python.
- **Search P99**: Lower is better. <5ms is ideal for real-time apps.
- **Recall**: Top-10 recall. >95% is usually required for RAG. 
