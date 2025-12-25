#!/usr/bin/env python3
"""
SrvDB v0.1.4 Release Benchmark
=================================
Comprehensive validation suite for production release.

Targets:
  - Ingestion: > 500 vecs/sec (expect ~800+)
  - Latency:   < 15ms (10k vectors, k=10)
  - Recall:    100% (exact match)
  - Integrity: PASS (data recovery after flush)
"""

import srvdb
import time
import random
import shutil
import os
from pathlib import Path

# Configuration
DB_PATH = "./bench_release_db"
NUM_VECTORS_INGEST = 1000
NUM_VECTORS_SEARCH = 10000
VECTOR_DIM = 1536
K = 10

# Targets
TARGET_INGESTION_RATE = 500  # vecs/sec
TARGET_LATENCY_MS = 15  # milliseconds
TARGET_RECALL = 100  # percentage


def print_header():
    """Print benchmark header."""
    print("╔══════════════════════════════════════╗")
    print("║   SrvDB v0.1.4 Release Benchmark    ║")
    print("╚══════════════════════════════════════╝")
    print()


def cleanup_db():
    """Clean up test database."""
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)


def generate_random_vector():
    """Generate a random 1536-dimensional vector."""
    return [random.random() - 0.5 for _ in range(VECTOR_DIM)]


def test_ingestion_speed():
    """Test 1: Ingestion Speed - The Killer Feature!"""
    print("=" * 50)
    print("TEST 1: Ingestion Speed (Priority #1)")
    print("=" * 50)
    
    cleanup_db()
    db = srvdb.SvDBPython(DB_PATH)
    
    # Generate test data
    print(f"Generating {NUM_VECTORS_INGEST} test vectors...")
    vectors = [generate_random_vector() for _ in range(NUM_VECTORS_INGEST)]
    ids = [f"vec_{i}" for i in range(NUM_VECTORS_INGEST)]
    metadatas = [f'{{"id": {i}}}' for i in range(NUM_VECTORS_INGEST)]
    
    # Measure ingestion time
    print(f"Ingesting {NUM_VECTORS_INGEST} vectors...")
    start_time = time.time()
    
    db.add(ids=ids, embeddings=vectors, metadatas=metadatas)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate throughput
    vecs_per_sec = NUM_VECTORS_INGEST / elapsed_time
    
    # Flush to ensure data is persisted
    db.persist()
    
    # Results
    status = "✓ PASS" if vecs_per_sec >= TARGET_INGESTION_RATE else "✗ FAIL"
    print(f"\nResults:")
    print(f"  Time:       {elapsed_time:.2f}s")
    print(f"  Throughput: {vecs_per_sec:.0f} vecs/sec")
    print(f"  Target:     ≥{TARGET_INGESTION_RATE} vecs/sec")
    print(f"  Status:     {status}")
    
    return vecs_per_sec >= TARGET_INGESTION_RATE, vecs_per_sec


def test_search_latency():
    """Test 2: Search Latency."""
    print("\n" + "=" * 50)
    print("TEST 2: Search Latency")
    print("=" * 50)
    
    cleanup_db()
    db = srvdb.SvDBPython(DB_PATH)
    
    # Create a larger database for realistic latency testing
    print(f"Creating database with {NUM_VECTORS_SEARCH} vectors...")
    vectors = [generate_random_vector() for _ in range(NUM_VECTORS_SEARCH)]
    ids = [f"vec_{i}" for i in range(NUM_VECTORS_SEARCH)]
    metadatas = [f'{{"id": {i}}}' for i in range(NUM_VECTORS_SEARCH)]
    
    db.add(ids=ids, embeddings=vectors, metadatas=metadatas)
    db.persist()
    
    # Measure search latency
    print(f"Measuring search latency (k={K})...")
    query = generate_random_vector()
    
    # Warm-up search
    _ = db.search(query=query, k=K)
    
    # Actual timing (average of 10 searches)
    num_searches = 10
    start_time = time.time()
    for _ in range(num_searches):
        _ = db.search(query=query, k=K)
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / num_searches) * 1000
    
    # Results
    status = "✓ PASS" if avg_latency_ms <= TARGET_LATENCY_MS else "✗ FAIL"
    print(f"\nResults:")
    print(f"  Avg Latency: {avg_latency_ms:.1f}ms")
    print(f"  Target:      ≤{TARGET_LATENCY_MS}ms")
    print(f"  Database:    {NUM_VECTORS_SEARCH:,} vectors")
    print(f"  Status:      {status}")
    
    return avg_latency_ms <= TARGET_LATENCY_MS, avg_latency_ms


def test_recall_accuracy():
    """Test 3: Recall Accuracy - Exact Match."""
    print("\n" + "=" * 50)
    print("TEST 3: Recall Accuracy")
    print("=" * 50)
    
    cleanup_db()
    db = srvdb.SvDBPython(DB_PATH)
    
    # Add test vectors
    num_test = 100
    print(f"Adding {num_test} test vectors...")
    test_vectors = [generate_random_vector() for _ in range(num_test)]
    ids = [f"test_{i}" for i in range(num_test)]
    metadatas = [f'{{"idx": {i}}}' for i in range(num_test)]
    
    db.add(ids=ids, embeddings=test_vectors, metadatas=metadatas)
    db.persist()
    
    # Test exact match recall
    print(f"Testing exact match recall ({num_test} queries)...")
    correct_matches = 0
    
    for i, query_vec in enumerate(test_vectors):
        results = db.search(query=query_vec, k=1)
        if results and results[0][0] == f"test_{i}":
            correct_matches += 1
    
    recall_percentage = (correct_matches / num_test) * 100
    
    # Results
    status = "✓ PASS" if recall_percentage == 100 else "✗ FAIL"
    print(f"\nResults:")
    print(f"  Correct:    {correct_matches}/{num_test}")
    print(f"  Recall:     {recall_percentage:.0f}%")
    print(f"  Target:     100%")
    print(f"  Status:     {status}")
    
    return recall_percentage == 100, recall_percentage


def test_data_integrity():
    """Test 4: Data Integrity - Persist & Reload."""
    print("\n" + "=" * 50)
    print("TEST 4: Data Integrity")
    print("=" * 50)
    
    cleanup_db()
    
    # Phase 1: Write data
    print("Phase 1: Writing data...")
    db = srvdb.SvDBPython(DB_PATH)
    
    num_vectors = 1000
    vectors = [generate_random_vector() for _ in range(num_vectors)]
    ids = [f"persist_{i}" for i in range(num_vectors)]
    metadatas = [f'{{"value": {i}}}' for i in range(num_vectors)]
    
    db.add(ids=ids, embeddings=vectors, metadatas=metadatas)
    db.persist()
    
    print(f"  Wrote:    {num_vectors} vectors")
    
    # IMPORTANT: Delete db object to close the database before reopening
    del db
    
    # Phase 2: Reload and verify
    print("Phase 2: Reloading database...")
    db2 = srvdb.SvDBPython(DB_PATH)
    
    recovered_count = db2.count()
    print(f"  Recovered: {recovered_count} vectors")
    
    # Verify all vectors are retrievable
    print("Phase 3: Verifying all vectors...")
    verified = 0
    for i in range(num_vectors):
        metadata = db2.get(f"persist_{i}")
        if metadata:
            verified += 1
    
    # Results
    integrity_pass = (recovered_count == num_vectors) and (verified == num_vectors)
    status = "✓ PASS" if integrity_pass else "✗ FAIL"
    
    print(f"\nResults:")
    print(f"  Written:    {num_vectors}")
    print(f"  Recovered:  {recovered_count}")
    print(f"  Verified:   {verified}")
    print(f"  Status:     {status}")
    
    return integrity_pass, verified


def main():
    """Run all benchmark tests."""
    print_header()
    
    results = {}
    
    # Run all tests
    try:
        pass_1, ingest_rate = test_ingestion_speed()
        results['ingestion'] = (pass_1, ingest_rate, f"{ingest_rate:.0f} vecs/sec")
        
        pass_2, latency = test_search_latency()
        results['latency'] = (pass_2, latency, f"{latency:.1f}ms")
        
        pass_3, recall = test_recall_accuracy()
        results['recall'] = (pass_3, recall, f"{recall:.0f}%")
        
        pass_4, verified = test_data_integrity()
        results['integrity'] = (pass_4, verified, f"{verified}/1000")
        
    finally:
        cleanup_db()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_pass = all(result[0] for result in results.values())
    
    print(f"\n{'✓' if results['ingestion'][0] else '✗'} Ingestion:    {results['ingestion'][2]:>15}  (target: >{TARGET_INGESTION_RATE})")
    print(f"{'✓' if results['latency'][0] else '✗'} Latency:      {results['latency'][2]:>15}  (target: <{TARGET_LATENCY_MS}ms)")
    print(f"{'✓' if results['recall'][0] else '✗'} Recall:       {results['recall'][2]:>15}  (target: 100%)")
    print(f"{'✓' if results['integrity'][0] else '✗'} Integrity:    {results['integrity'][2]:>15}  (target: 1000/1000)")
    
    print("\n" + "=" * 50)
    if all_pass:
        print("Status: READY FOR RELEASE 🚀")
    else:
        print("Status: NEEDS ATTENTION ⚠️")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
