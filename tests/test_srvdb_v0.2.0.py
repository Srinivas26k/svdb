#!/usr/bin/env python3
"""
===========================================================
SrvDB v0.2.0 Comprehensive Benchmark Suite
===========================================================

Tests Dynamic Dimensions, SQ8 Compression, and HNSW Indexing.
Includes optional FAISS baseline comparison.

Usage:
    pip install srvdb numpy scikit-learn psutil
    python test_srvdb_v0.2.0.py [--dim 768] [--with-faiss]
"""

import srvdb
import numpy as np
import time
import os
import shutil
import json
import platform
import psutil
import gc
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURATION & SYSTEM DETECTION
# =========================================================

class Config:
    def __init__(self, dim=1536, compare_faiss=False):
        self.dim = dim
        self.compare_faiss = compare_faiss
        self.n_vectors = 50_000  # Safe default for laptops
        self.n_queries = 100
        self.top_k = 10
        self.seed = 42

    def adjust_for_ram(self):
        """Auto-scale dataset size to prevent crashes"""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative limits
        if ram_gb < 8:
            self.n_vectors = 10_000
        elif ram_gb < 16:
            self.n_vectors = 30_000
        else:
            self.n_vectors = 50_000
            
        print(f"[System] RAM: {ram_gb:.1f}GB | Dataset Size: {self.n_vectors:,} vectors")

def get_rss_mb():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except:
        return 0.0

def get_disk_mb(path):
    if not os.path.exists(path): return 0.0
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / 1024 / 1024

# =========================================================
# 2. DATA GENERATION (ADVERSARIAL MIX)
# =========================================================

def generate_adversarial_mix(n_total, dim, seed):
    """
    Generates 70% Random + 30% Clustered data.
    Critical for testing Quantization (SQ8/PQ) robustness.
    """
    print(f"  [Data] Generating {n_total} vectors (Dim: {dim}, Mix: 70% Random / 30% Clustered)...")
    np.random.seed(seed)
    
    n_random = int(n_total * 0.7)
    n_blobs = n_total - n_random
    
    # Random Noise
    X_random = np.random.randn(n_random, dim).astype(np.float32)
    
    # Clustered Data (Stress Test)
    X_blobs, _ = make_blobs(
        n_samples=n_blobs, 
        n_features=dim, 
        centers=5,             # Tight clusters to stress quantizers
        cluster_std=0.4, 
        random_state=seed
    )
    X_blobs = X_blobs.astype(np.float32)
    
    # Combine, Shuffle, Normalize
    X = np.vstack([X_random, X_blobs])
    np.random.shuffle(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    # Split Train/Query
    n_queries = 100
    train_vecs = X[:n_total - n_queries]
    query_vecs = X[n_total - n_queries:]
    train_ids = [f"doc_{i}" for i in range(len(train_vecs))]
    
    return train_vecs, query_vecs, train_ids

def compute_ground_truth(train_vecs, query_vecs, k):
    print("  [Data] Computing Ground Truth (Brute Force)...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine', n_jobs=-1).fit(train_vecs)
    _, indices = nbrs.kneighbors(query_vecs)
    return indices

# =========================================================
# 3. BENCHMARK ENGINE
# =========================================================

def benchmark_mode(mode_name, db_path, train_vecs, train_ids, query_vecs, gt_indices, config, ef_values=None):
    """
    Modes: 'flat', 'hnsw', 'sq8', 'pq'
    """
    print(f"\n{'='*70}")
    print(f"  MODE: {mode_name.upper()}")
    print(f"{'='*70}")
    
    # 1. Init
    clean_dir(db_path)
    ram_start = get_rss_mb()
    t0 = time.time()
    
    try:
        db = None
        
        # --- MODE SELECTOR ---
        if mode_name == 'flat':
            db = srvdb.SrvDBPython(db_path)
            
        elif mode_name == 'hnsw':
            # Assuming default HNSW params for simplicity, or use .new_with_hnsw
            db = srvdb.SrvDBPython(db_path) 
            # Tuning EF later in search phase
            
        elif mode_name == 'sq8':
            # v0.2.0 API - Requires training vectors
            if hasattr(srvdb.SrvDBPython, 'new_scalar_quantized'):
                # Pass a subset of training vectors for initialization
                train_subset = train_vecs[:1000].tolist()
                db = srvdb.SrvDBPython.new_scalar_quantized(db_path, config.dim, train_subset)
            else:
                print("    [SKIP] SQ8 API not found in this version.")
                return None

        elif mode_name == 'pq':
            # v0.2.0 API - PQ might not be exposed or is WIP
            if hasattr(srvdb.SrvDBPython, 'new_quantized'):
                train_subset = train_vecs[:2000].tolist()
                db = srvdb.SrvDBPython.new_quantized(db_path, train_subset)
            else:
                print("    [SKIP] PQ API not found (new_quantized missing).")
                return None
            
        init_time = time.time() - t0
        
        if db is None: return None
        
        # 2. Ingest
        t0 = time.time()
        batch_size = 1000
        for i in range(0, len(train_vecs), batch_size):
            end = min(i + batch_size, len(train_vecs))
            db.add(train_ids[i:end], train_vecs[i:end].tolist(), [f'{{"id":{x}}}' for x in range(i, end)])
        
        db.persist()
        ingest_time = time.time() - t0
        
        ram_after_ingest = get_rss_mb()
        disk_usage = get_disk_mb(db_path)
        
        # 3. Search & Recall
        search_stats = {}
        
        # Determine EF values to test
        ef_list = [50] if mode_name in ['flat', 'sq8'] else (ef_values or [20, 50, 100])
        
        for ef in ef_list:
            if hasattr(db, "set_ef_search"):
                db.set_ef_search(ef)
            
            latencies = []
            recall_hits = 0
            total_hits = len(query_vecs) * config.top_k
            
            # Warmup
            _ = db.search(query_vecs[0].tolist(), config.top_k)
            
            for q_idx, q in enumerate(query_vecs):
                t_start = time.perf_counter()
                res = db.search(q.tolist(), config.top_k)
                latencies.append((time.perf_counter() - t_start) * 1000)
                
                found = set(r[0] for r in res)
                expected = set(train_ids[i] for i in gt_indices[q_idx])
                recall_hits += len(found & expected)
            
            search_stats[f"ef_{ef}"] = {
                "latency_p99_ms": round(np.percentile(latencies, 99), 3),
                "latency_p50_ms": round(np.median(latencies), 3),
                "recall_pct": round((recall_hits / total_hits) * 100, 2)
            }
            if mode_name in ['flat', 'sq8']: break # No need to test multiple EFs

        # Cleanup
        del db
        gc.collect()
        ram_end = get_rss_mb()
        
        return {
            "mode": mode_name,
            "init_time_s": round(init_time, 3),
            "ingestion_time_s": round(ingest_time, 3),
            "ingestion_rate": round(len(train_vecs) / ingest_time, 2),
            "disk_usage_mb": round(disk_usage, 2),
            "ram_peak_mb": round(ram_after_ingest - ram_start, 2),
            "ram_final_mb": round(ram_end, 2),
            "search_performance": search_stats
        }
        
    except Exception as e:
        print(f"    [ERROR] {e}")
        return {"mode": mode_name, "error": str(e)}

def benchmark_faiss_baseline(train_vecs, query_vecs, gt_indices, config):
    """
    Optional competitor baseline using FAISS (Flat Index).
    Requires `pip install faiss-cpu`.
    """
    print(f"\n{'='*70}")
    print(f"  MODE: FAISS (Baseline Comparison)")
    print(f"{'='*70}")
    
    try:
        import faiss
    except ImportError:
        print("    [SKIP] FAISS not installed. Run `pip install faiss-cpu` to enable comparison.")
        return None

    t0 = time.time()
    ram_start = get_rss_mb()
    
    # FAISS Flat Index (Inner Product)
    # Normalize vectors for FAISS (it expects non-normalized for IP usually, but we normalize for cosine)
    # We use IndexFlatIP since our vectors are normalized (Cosine = Dot Product)
    index = faiss.IndexFlatIP(config.dim)
    index.add(train_vecs.astype(np.float32))
    
    ingest_time = time.time() - t0
    
    # Search
    latencies = []
    recall_hits = 0
    total_hits = len(query_vecs) * config.top_k
    
    for q in query_vecs:
        t_start = time.perf_counter()
        D, I = index.search(q.reshape(1, -1).astype(np.float32), config.top_k)
        latencies.append((time.perf_counter() - t_start) * 1000)
        
        # FAISS returns indices
        found = set(train_ids[i] for i in I[0])
        expected = set(train_ids[i] for i in gt_indices[len(latencies)-1])
        recall_hits += len(found & expected)

    del index
    gc.collect()
    ram_end = get_rss_mb()
    
    return {
        "mode": "faiss_flat",
        "ingestion_time_s": round(ingest_time, 3),
        "ingestion_rate": round(len(train_vecs) / ingest_time, 2),
        "search_performance": {
            "default": {
                "latency_p99_ms": round(np.percentile(latencies, 99), 3),
                "latency_p50_ms": round(np.median(latencies), 3),
                "recall_pct": round((recall_hits / total_hits) * 100, 2)
            }
        },
        "ram_peak_mb": round(ram_end - ram_start, 2)
    }

def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# =========================================================
# 4. MAIN ORCHESTRATOR
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="SrvDB v0.2.0 Benchmark")
    parser.add_argument("--dim", type=int, default=1536, choices=[768, 1536, 3072], help="Embedding dimension (Default: 1536)")
    parser.add_argument("--faiss", action="store_true", help="Include FAISS baseline comparison")
    args = parser.parse_args()
    
    config = Config(dim=args.dim, compare_faiss=args.faiss)
    config.adjust_for_ram()
    
    print("=" * 70)
    print("  SRVDB v0.2.0 BENCHMARK")
    print("=" * 70)
    print(f"  Target Dimension: {config.dim}")
    print(f"  Dataset Strategy: Adversarial Mix (Stress Test for Quantizers)")
    
    # 1. Data Prep
    train_vecs, query_vecs, train_ids = generate_adversarial_mix(config.n_vectors, config.dim, config.seed)
    gt_indices = compute_ground_truth(train_vecs, query_vecs, config.top_k)
    
    # 2. Run Benchmarks
    results = {
        "benchmark_info": {
            "version": "2.0.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dimensions": config.dim,
            "dataset_sizes_tested": [], 
            "system": {
                "os": platform.system(),
                "machine": platform.machine(),
                "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "cpu_cores": psutil.cpu_count()
            }
        },
        "results": {}
    }
    
    DB_ROOT = "./v0.2.0_bench_db"
    
    # Scalability Sizes
    dataset_sizes = [10_000, 50_000] # Adjust based on RAM
    if config.n_vectors > 50_000:
        dataset_sizes.append(config.n_vectors)
    
    results["benchmark_info"]["dataset_sizes_tested"] = dataset_sizes

    for size in dataset_sizes:
        print(f"\n" + "=" * 70)
        print(f"  SCALABILITY TEST: {size:,} Vectors")
        print("=" * 70)
        
        # Regen data for this size
        train_vecs_s, query_vecs_s, train_ids_s = generate_adversarial_mix(size, config.dim, config.seed)
        gt_indices_s = compute_ground_truth(train_vecs_s, query_vecs_s, config.top_k)
        
        results["results"][size] = {}

        modes_to_run = ['flat', 'hnsw', 'sq8']
        if config.dim == 1536:
             # Skip PQ for now as API is missing
             pass
             
        for mode in modes_to_run:
            db_path = os.path.join(DB_ROOT, f"{mode}_{size}")
            res = benchmark_mode(mode, db_path, train_vecs_s, train_ids_s, query_vecs_s, gt_indices_s, config)
            if res:
                results["results"][size][mode] = res

        # Optional: FAISS
        if config.compare_faiss:
             res_faiss = benchmark_faiss_baseline(train_vecs_s, query_vecs_s, gt_indices_s, config)
             if res_faiss:
                  results["results"][size]["faiss"] = res_faiss

    # 3. Save Report
    filename = f"srvdb_v2.0.0_benchmark_{config.dim}dim.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
        
    # 4. Print Summary
    print("\n" + "=" * 70)
    print("  EXECUTIVE SUMMARY")
    print("=" * 70)
    
    for size in dataset_sizes:
        print(f"\n--- Dataset Size: {size:,} ---")
        for mode, data in results["results"][size].items():
            if "error" in data:
                print(f"  [{mode.upper()}] FAILED: {data['error']}")
                continue
                
            # Get best stats (usually EF=50 or default)
            best_perf_key = "ef_50" if "ef_50" in data['search_performance'] else "default"
            if best_perf_key not in data['search_performance']:
                 best_perf_key = list(data['search_performance'].keys())[0]
            
            perf = data['search_performance'][best_perf_key]
            
            qps = 1000.0 / perf['latency_p50_ms'] if perf['latency_p50_ms'] > 0 else 0
            
            print(f"\n  [{mode.upper()}]")
            print(f"    Ingestion:   {data['ingestion_rate']:,.0f} vec/s")
            print(f"    Disk Size:   {data['disk_usage_mb']:.2f} MB")
            print(f"    Latency P99: {perf['latency_p99_ms']:.2f} ms")
            print(f"    Latency P50: {perf['latency_p50_ms']:.2f} ms")
            print(f"    QPS (est):   {qps:,.0f} queries/s")
            print(f"    Recall:      {perf['recall_pct']:.1f}%")
            
            # Analysis
            if mode == 'sq8':
                print(f"    STATUS: Compression mode.")
                
    print(f"\n  Report saved to: {filename}")

if __name__ == "__main__":
    main()