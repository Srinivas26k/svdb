#!/usr/bin/env python3
"""
===========================================================
SrvDB Universal Benchmark Suite
===========================================================

A hardware-agnostic benchmark script designed for community contribution.
It automatically adjusts dataset size based on available RAM and uses
an "Adversarial Data Mix" (70% Random / 30% Clustered)
to stress-test Product Quantization (PQ) accuracy.

Usage:
    pip install srvdb numpy scikit-learn psutil
    python universal_benchmark.py

Output:
    benchmark_result_<os>_<timestamp>.json
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
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. SYSTEM DETECTION & CONFIGURATION
# =========================================================

class SystemProfile:
    """Detects hardware and selects safe dataset size."""
    def __init__(self):
        self.ram_total = psutil.virtual_memory().total
        self.cpu_cores = os.cpu_count()
        self.cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 0
        self.platform = platform.system()
        
        # Safety Logic: Select dataset size based on RAM
        if self.ram_total < 8_000_000_000:  # < 8GB
            self.n_vectors = 10_000
            self.tier = "Laptop (Safe Mode)"
        elif self.ram_total < 32_000_000_000: # < 32GB
            self.n_vectors = 100_000
            self.tier = "Desktop/Server"
        else:
            self.n_vectors = 1_000_000
            self.tier = "High-Performance Cluster"

    def __str__(self):
        ram_gb = self.ram_total / (1024**3)
        return (f"{self.platform} | RAM: {ram_gb:.1f}GB | "
                f"CPU: {self.cpu_cores} Cores | "
                f"Test Size: {self.n_vectors:,} ({self.tier})")

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

def generate_adversarial_data(n_total, dim=1536, seed=42):
    """
    Generates 70% Random + 30% Clustered data.
    This mix simulates real-world semantic distributions where 
    Product Quantization (PQ) often struggles.
    """
    print(f"  -> Generating {n_total} vectors (Adversarial Mix)...")
    np.random.seed(seed)
    
    n_random = int(n_total * 0.7)
    n_blobs = n_total - n_random
    
    # 1. Generate Random Noise (70%)
    X_random = np.random.randn(n_random, dim).astype(np.float32)
    
    # 2. Generate Tight Clusters (30%) - This is the "Adversarial" part
    X_blobs, _ = make_blobs(
        n_samples=n_blobs, 
        n_features=dim, 
        centers=10, 
        cluster_std=0.4, 
        random_state=seed
    )
    X_blobs = X_blobs.astype(np.float32)
    
    # 3. Combine and Normalize
    X = np.vstack([X_random, X_blobs])
    np.random.shuffle(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    # 4. Split Train/Query
    n_queries = 100
    train_vecs = X[:n_total - n_queries]
    query_vecs = X[n_total - n_queries:]
    train_ids = [f"doc_{i}" for i in range(len(train_vecs))]
    
    return train_vecs, query_vecs, train_ids

def compute_ground_truth(train_vecs, query_vecs, k=10):
    print("  -> Computing Ground Truth (Brute Force)...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine', n_jobs=-1).fit(train_vecs)
    _, indices = nbrs.kneighbors(query_vecs)
    return indices

# =========================================================
# 3. BENCHMARK ENGINE
# =========================================================

def run_benchmark():
    print("=" * 70)
    print("  SRVDB UNIVERSAL BENCHMARK")
    print("=" * 70)
    
    # Detect System
    system = SystemProfile()
    print(f"\nHardware Detected: {system}")
    
    # Prepare Data
    train_vecs, query_vecs, train_ids = generate_adversarial_data(system.n_vectors)
    gt_indices = compute_ground_truth(train_vecs, query_vecs)
    
    # Init Report
    report = {
        "benchmark_version": "1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "os": system.platform,
            "machine": platform.machine(),
            "ram_gb": round(system.ram_total / (1024**3), 2),
            "cpu_cores": system.cpu_cores,
            "cpu_freq_mhz": system.cpu_freq
        },
        "config": {
            "dataset_size": system.n_vectors,
            "dimensions": 1536,
            "n_queries": 100,
            "data_type": "Adversarial Mix (70% Random / 30% Clustered)"
        },
        "results": {}
    }
    
    DB_ROOT = "./benchmark_universal_db"
    if os.path.exists(DB_ROOT):
        shutil.rmtree(DB_ROOT)
    
    modes = ['flat', 'hnsw', 'pq']
    
    for mode in modes:
        print(f"\n--- Benchmarking Mode: {mode.upper()} ---")
        db_path = os.path.join(DB_ROOT, mode)
        mode_stats = {}
        
        try:
            # 1. Init
            t0 = time.time()
            ram_start = get_rss_mb()
            
            if mode == 'pq':
                # Train PQ on subset (1k or 10%)
                train_subset = max(1000, len(train_vecs) // 10)
                db = srvdb.SvDBPython.new_quantized(db_path, train_vecs[:train_subset].tolist())
            else:
                db = srvdb.SvDBPython(db_path)
            
            init_time = time.time() - t0
            
            # 2. Ingest
            t0 = time.time()
            batch_size = 1000
            for i in range(0, len(train_vecs), batch_size):
                end = min(i + batch_size, len(train_vecs))
                db.add(
                    train_ids[i:end],
                    train_vecs[i:end].tolist(),
                    [f'{{"id":{x}}}' for x in range(i, end)]
                )
            db.persist()
            ingest_time = time.time() - t0
            
            # 3. Metrics
            ram_peak = get_rss_mb()
            disk_usage = get_disk_mb(db_path)
            
            # 4. Search & Recall
            # For Flat, test default. For HNSW/PQ, sweep EF
            ef_list = [50] if mode == 'flat' else [20, 50, 100]
            search_perf = {}
            
            for ef in ef_list:
                if hasattr(db, "set_ef_search"):
                    db.set_ef_search(ef)
                
                latencies = []
                recall_hits = 0
                total_hits = 100 * 10
                
                # Warmup
                _ = db.search(query_vecs[0].tolist(), 10)
                
                for q_idx, q in enumerate(query_vecs):
                    t_start = time.perf_counter()
                    res = db.search(q.tolist(), 10)
                    latencies.append((time.perf_counter() - t_start) * 1000)
                    
                    # Recall Check
                    found = set(r[0] for r in res)
                    expected = set(train_ids[i] for i in gt_indices[q_idx])
                    recall_hits += len(found & expected)
                
                search_perf[f"ef_{ef}"] = {
                    "latency_p99_ms": round(np.percentile(latencies, 99), 3),
                    "latency_p50_ms": round(np.median(latencies), 3),
                    "recall_pct": round((recall_hits / total_hits) * 100, 2)
                }

            del db
            gc.collect()
            
            mode_stats = {
                "init_time_s": round(init_time, 3),
                "ingestion_time_s": round(ingest_time, 3),
                "ingestion_rate": round(len(train_vecs) / ingest_time, 2),
                "disk_usage_mb": round(disk_usage, 2),
                "ram_peak_delta_mb": round(ram_peak - ram_start, 2),
                "search_performance": search_perf
            }
            
            report["results"][mode] = mode_stats
            
        except AttributeError as e:
            print(f"  -> SKIPPED (Method not found): {e}")
            report["results"][mode] = {"error": "method_missing"}
        except Exception as e:
            print(f"  -> ERROR: {e}")
            report["results"][mode] = {"error": str(e)}

    # Save
    filename = f"benchmark_result_{platform.system()}_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"  BENCHMARK COMPLETE")
    print(f"  Report Saved: {filename}")
    print("=" * 70)
    print("\nTo contribute to SrvDB:")
    print(f"1. Upload '{filename}' to GitHub Issues.")
    print("2. Mention your CPU/RAM specs in the issue description.")

if __name__ == "__main__":
    run_benchmark()