#!/usr/bin/env python3
"""
Product Quantization Benchmark for SrvDB
Compare full precision vs 32x compressed PQ mode
"""

import srvdb
import numpy as np
import time
import tempfile
import shutil
import os
import psutil

def get_memory_mb():
    """Get current process memory in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def generate_vectors(n, dim=1536):
    """Generate random normalized vectors"""
    vecs = np.random.randn(n, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).tolist()

def benchmark_mode(mode, n_vectors=10000, n_queries=100):
    """Benchmark either 'full' or 'pq' mode"""
    print(f"\n{'='*60}")
    print(f"  {mode.upper()} PRECISION MODE")
    print(f"{'='*60}\n")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate data
        print(f"Generating {n_vectors} vectors...")
        vectors = generate_vectors(n_vectors)
        ids = [f"doc_{i}" for i in range(n_vectors)]
        metadatas = [f'{{"id": {i}}}' for i in range(n_vectors)]
        
        # Initialize database
        print(f"Initializing database...")
        mem_before = get_memory_mb()
        start = time.time()
        
        if mode == "pq":
            # Use first 1000 vectors for training
            training_data = vectors[:1000]
            db = srvdb.SvDBPython.new_quantized(temp_dir, training_data)
        else:
            db = srvdb.SvDBPython(temp_dir)
        
        init_time = time.time() - start
        print(f"  Init time: {init_time:.2f}s")
        
        # Ingestion benchmark
        print(f"\nIngesting {n_vectors} vectors...")
        start = time.time()
        db.add(ids, vectors, metadatas)
        db.persist()
        ingestion_time = time.time() - start
        ingestion_rate = n_vectors / ingestion_time
        
        mem_after = get_memory_mb()
        mem_used = mem_after - mem_before
        
        print(f"  Ingestion time: {ingestion_time:.2f}s")
        print(f"  Ingestion rate: {ingestion_rate:,.0f} vec/s")
        print(f"  Memory used: {mem_used:.1f} MB")
        
        # Get stats for PQ mode
        if mode == "pq":
            stats = db.get_stats()
            if stats:
                vec_count, memory_bytes, compression = stats
                print(f"\n  PQ Statistics:")
                print(f"    Vectors: {vec_count:,}")
                print(f"    Memory: {memory_bytes / 1024 / 1024:.1f} MB")
                print(f"    Compression: {compression:.1f}x")
        
        # Search benchmark
        print(f"\nSearching {n_queries} queries...")
        query_vectors = generate_vectors(n_queries)
        
        latencies = []
        for query in query_vectors:
            start = time.time()
            results = db.search(query, k=10)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
        
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        print(f"  Latency P50: {p50:.2f}ms")
        print(f"  Latency P95: {p95:.2f}ms")
        print(f"  Latency P99: {p99:.2f}ms")
        
        # Batch search benchmark
        print(f"\nBatch searching {n_queries} queries...")
        start = time.time()
        batch_results = db.search_batch(query_vectors, k=10)
        batch_time = time.time() - start
        qps = n_queries / batch_time
        
        print(f"  Batch time: {batch_time:.2f}s")
        print(f"  QPS: {qps:.0f}")
        
        # Accuracy check (for PQ, check if results make sense)
        print(f"\nAccuracy check:")
        test_vec = vectors[0]
        results = db.search(test_vec, k=5)
        print(f"  Top-5 results for vector 0:")
        for doc_id, score in results[:5]:
            print(f"    {doc_id}: {score:.4f}")
        
        return {
            "mode": mode,
            "n_vectors": n_vectors,
            "init_time": init_time,
            "ingestion_time": ingestion_time,
            "ingestion_rate": ingestion_rate,
            "memory_mb": mem_used,
            "latency_p50": p50,
            "latency_p95": p95,
            "qps": qps,
        }
        
    finally:
        shutil.rmtree(temp_dir)

def compare_modes():
    """Compare full precision vs PQ"""
    print("\n" + "="*60)
    print("  SRVDB PRODUCT QUANTIZATION BENCHMARK")
    print("="*60)
    
    # Run benchmarks
    full_stats = benchmark_mode("full", n_vectors=10000, n_queries=100)
    pq_stats = benchmark_mode("pq", n_vectors=10000, n_queries=100)
    
    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}\n")
    
    print(f"{'Metric':<25} {'Full Precision':<20} {'PQ (32x)':<20}")
    print("-" * 65)
    
    print(f"{'Ingestion Rate':<25} {full_stats['ingestion_rate']:>15,.0f} vec/s {pq_stats['ingestion_rate']:>15,.0f} vec/s")
    print(f"{'Memory Usage':<25} {full_stats['memory_mb']:>18.1f} MB {pq_stats['memory_mb']:>18.1f} MB")
    print(f"{'Search Latency (P50)':<25} {full_stats['latency_p50']:>18.2f} ms {pq_stats['latency_p50']:>18.2f} ms")
    print(f"{'Search Latency (P95)':<25} {full_stats['latency_p95']:>18.2f} ms {pq_stats['latency_p95']:>18.2f} ms")
    print(f"{'Batch QPS':<25} {full_stats['qps']:>20.0f} {pq_stats['qps']:>20.0f}")
    
    # Calculate improvements
    memory_reduction = full_stats['memory_mb'] / pq_stats['memory_mb']
    latency_change = ((pq_stats['latency_p50'] - full_stats['latency_p50']) / full_stats['latency_p50']) * 100
    
    print(f"\n{'Benefits:':<25}")
    print(f"  Memory reduction: {memory_reduction:.1f}x")
    print(f"  Latency change: {latency_change:+.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    compare_modes()
