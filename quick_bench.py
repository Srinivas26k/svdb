#!/usr/bin/env python3
"""Quick benchmark test for SrvDB v0.1.7 optimizations"""

import srvdb
import time
import random
import shutil
import os

DB_PATH = "./quick_bench_db"

def cleanup():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

def gen_vector():
    return [random.random() - 0.5 for _ in range(1536)]

print("╔══════════════════════════════════════════════════════════════════╗")
print("║     SrvDB v0.1.7 Quick Benchmark (Testing Optimizations)       ║")
print("╚══════════════════════════════════════════════════════════════════╝\n")

# Test 1: Ingestion throughput
print("Test 1: Ingestion Throughput")
print("=" * 70)
cleanup()
db = srvdb.SvDBPython(DB_PATH)

num_vectors = 1000
vectors = [gen_vector() for _ in range(num_vectors)]
ids = [f"vec_{i}" for i in range(num_vectors)]
metas = [f'{{"idx": {i}}}' for i in range(num_vectors)]

print(f"Ingesting {num_vectors:,} vectors...")
start = time.time()
db.add(ids=ids, embeddings=vectors, metadatas=metas)
db.persist()
elapsed = time.time() - start

throughput = num_vectors / elapsed
print(f"  Time:       {elapsed:.2f}s")
print(f"  Throughput: {throughput:.0f} vec/s")
print(f"  Status:     {'✅ PASS' if throughput >= 500 else '⚠️  CHECK'}")

# Test 2: Search latency
print(f"\nTest 2: Search Latency")
print("=" * 70)
query = gen_vector()

# Warmup
db.search(query=query, k=10)

# Measure
num_searches = 100
start = time.time()
for _ in range(num_searches):
    results = db.search(query=query, k=10)
elapsed = time.time() - start

avg_latency_ms = (elapsed / num_searches) * 1000
print(f"  Avg Latency: {avg_latency_ms:.2f}ms ({num_searches} queries)")
print(f"  Status:      {'✅ PASS' if avg_latency_ms < 15 else '⚠️  CHECK'}")

# Test 3: Recall
print(f"\nTest 3: Recall Accuracy")
print("=" * 70)
correct = 0
for i, vec in enumerate(vectors[:100]):
    results = db.search(query=vec, k=1)
    if results and results[0][0] == f"vec_{i}":
        correct += 1

recall = (correct / 100) * 100
print(f"  Correct: {correct}/100")
print(f"  Recall:  {recall:.0f}%")
print(f"  Status:  {'✅ PASS' if recall == 100 else '❌ FAIL'}")

# Test 4: Data integrity (reload)
print(f"\nTest 4: Data Integrity")
print("=" * 70)
del db
db2 = srvdb.SvDBPython(DB_PATH)
recovered = db2.count()
print(f"  Written:   {num_vectors}")
print(f"  Recovered: {recovered}")
print(f"  Status:    {'✅ PASS' if recovered == num_vectors else '❌ FAIL'}")

cleanup()

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
