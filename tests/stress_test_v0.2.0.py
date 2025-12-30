import srvdb
import numpy as np
import os
import shutil
import psutil
import time
from pathlib import Path

# =================CONFIGURATION=================
DIMENSION = 768  # Standard HuggingFace dimension
N_VECTORS = 50_000  # Enough to overcome metadata overhead
BATCH_SIZE = 5_000

DB_FLAT = "./stress_test_flat"
DB_SQ8 = "./stress_test_sq8"
# ===============================================

def get_dir_size_mb(path):
    if not os.path.exists(path): return 0.0
    return sum(f.stat().st_size for f in Path(path).glob('**/*') if f.is_file()) / (1024 * 1024)

def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def generate_batch(size, dim):
    # Generate random normalized vectors
    vecs = np.random.randn(size, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.tolist()

def run_stress_test():
    print(f"🔥 STARTING v0.2.0 STRESS TEST")
    print(f"   Vectors: {N_VECTORS:,} | Dim: {DIMENSION}")
    
    # Clean previous runs
    if os.path.exists(DB_FLAT): shutil.rmtree(DB_FLAT)
    if os.path.exists(DB_SQ8): shutil.rmtree(DB_SQ8)

    # ---------------------------------------------------------
    # 1. BASELINE: Flat Index (Float32)
    # ---------------------------------------------------------
    print(f"\n[1/2] Building FLAT Index (Baseline)...")
    start_ram = get_ram_mb()
    db_flat = srvdb.SrvDBPython(DB_FLAT, dimension=DIMENSION)
    
    t0 = time.time()
    for i in range(0, N_VECTORS, BATCH_SIZE):
        batch = generate_batch(BATCH_SIZE, DIMENSION)
        ids = [f"f_{j}" for j in range(i, i+BATCH_SIZE)]
        metas = [""] * BATCH_SIZE
        db_flat.add(ids, batch, metas)
        print(f"   -> Added {i+BATCH_SIZE}...", end="\r")
    
    db_flat.persist()
    time_flat = time.time() - t0
    peak_ram_flat = get_ram_mb() - start_ram
    size_flat = get_dir_size_mb(DB_FLAT)
    print(f"\n   ✅ Done in {time_flat:.2f}s")

    # ---------------------------------------------------------
    # 2. CHALLENGER: SQ8 Index (Int8)
    # ---------------------------------------------------------
    print(f"\n[2/2] Building SQ8 Index (v0.2.0 Feature)...")
    start_ram = get_ram_mb()
    
    # Training set for SQ8 (just needs to be representative)
    training_data = generate_batch(1000, DIMENSION)
    
    # Initialize SQ8
    db_sq8 = srvdb.SrvDBPython.new_scalar_quantized(
        DB_SQ8, 
        dimension=DIMENSION, 
        training_vectors=training_data
    )
    
    t0 = time.time()
    for i in range(0, N_VECTORS, BATCH_SIZE):
        batch = generate_batch(BATCH_SIZE, DIMENSION)
        ids = [f"s_{j}" for j in range(i, i+BATCH_SIZE)]
        metas = [""] * BATCH_SIZE
        db_sq8.add(ids, batch, metas)
        print(f"   -> Added {i+BATCH_SIZE}...", end="\r")
    
    db_sq8.persist()
    time_sq8 = time.time() - t0
    peak_ram_sq8 = get_ram_mb() - start_ram
    size_sq8 = get_dir_size_mb(DB_SQ8)
    print(f"\n   ✅ Done in {time_sq8:.2f}s")

    # ---------------------------------------------------------
    # 3. ANALYSIS & RESULTS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📊 v0.2.0 PERFORMANCE REPORT")
    print("="*50)
    
    print(f"{'Metric':<20} | {'Flat (Base)':<12} | {'SQ8 (New)':<12} | {'Improvement':<12}")
    print("-" * 65)
    
    # Disk Usage
    ratio = size_flat / size_sq8 if size_sq8 > 0 else 0
    print(f"{'Disk Size':<20} | {size_flat:.1f} MB      | {size_sq8:.1f} MB      | {ratio:.1f}x smaller")
    
    # RAM Usage (Did we fix the spike?)
    print(f"{'Peak RAM Delta':<20} | {peak_ram_flat:.1f} MB      | {peak_ram_sq8:.1f} MB      | {(peak_ram_flat - peak_ram_sq8):.1f} MB diff")
    
    # Speed
    print(f"{'Ingest Speed':<20} | {N_VECTORS/time_flat:.0f} vec/s   | {N_VECTORS/time_sq8:.0f} vec/s   |")
    
    print("-" * 65)
    
    # Validation Check
    if ratio > 3.0:
        print("✅ SUCCESS: Scalar Quantization is working (~4x compression expected).")
    elif ratio > 1.5:
        print("⚠️ WARNING: Compression is lower than expected. Check overhead.")
    else:
        print("❌ FAILURE: SQ8 is not compressing data significantly.")

    if peak_ram_sq8 < 500:
        print("✅ SUCCESS: RAM usage is stable (Streaming works).")
    else:
        print("❌ FAILURE: Memory spike detected!")

if __name__ == "__main__":
    run_stress_test()