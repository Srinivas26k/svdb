#!/usr/bin/env python3
"""
PQ Accuracy Verification Script
===============================
Tests if Product Quantization (PQ) mode correctly returns Cosine Similarity scores.

CRITICAL REQUIREMENT: 
Input vectors MUST be normalized. This script handles normalization in Python.

Expected Results:
1. Full Precision Self-Match Score: Exactly 1.0000
2. PQ Self-Match Score: ~0.95 - 1.0000 (Slight loss due to quantization is normal)
3. The top result ID must match the query ID.

If scores are < -1.0, internal normalization in the Rust backend is broken.
"""

import srvdb
import numpy as np
import tempfile
import shutil
import os

def generate_normalized_data(n_vectors, dim):
    print(f"Generating {n_vectors} normalized vectors (dim={dim})...")
    # Generate random gaussian data
    vectors_raw = np.random.randn(n_vectors, dim).astype(np.float32)
    # Calculate L2 norms
    norms = np.linalg.norm(vectors_raw, axis=1, keepdims=True)
    # Divide by norms to get unit length vectors
    vectors = (vectors_raw / norms).tolist()
    
    ids = [f"doc_{i}" for i in range(n_vectors)]
    metadatas = [f'{{"id": {i}}}' for i in range(n_vectors)]
    return ids, vectors, metadatas

def run_test(db, mode_name, vectors, test_indices):
    print(f"\n--- {mode_name.upper()} MODE ---")
    print(f"Testing self-match accuracy on {len(test_indices)} vectors...")
    
    for idx in test_indices:
        query_vec = vectors[idx]
        expected_id = f"doc_{idx}"
        
        # Search for itself
        results = db.search(query_vec, k=5)
        
        if not results:
            print(f"  ❌ doc_{idx}: No results found")
            continue
            
        top_id, top_score = results[0]
        match_status = "✓" if top_id == expected_id else "❌ ID MISMATCH"
        
        print(f"  {match_status} {expected_id}: Score = {top_score:.4f}")
        
        # Check score thresholds based on mode
        if mode_name == "PQ":
            if top_score < 0.9:
                 print(f"      ⚠️ WARNING: Score too low for self-match (Expected >0.9)")
            if top_score < -1.0:
                 print(f"      ⛔ CRITICAL FAILURE: Score < -1.0 is mathematically impossible for Cosine Similarity.")
        elif mode_name == "Full Precision":
             if not np.isclose(top_score, 1.0, atol=1e-5):
                  print(f"      ⛔ CRITICAL FAILURE: Full precision self-match must be 1.0")

def main():
    print("=" * 60)
    print("  SRVDB PQ ACCURACY DIAGNOSTIC")
    print("=" * 60)
    
    # Use a fixed temp path for easier debugging if needed, clean it first
    temp_dir = "./test_pq_accuracy_db"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # 1. Generate Data
        n_vectors = 1000
        dim = 1536
        ids, vectors, metadatas = generate_normalized_data(n_vectors, dim)
        # Test indices spread across the dataset
        test_indices = [0, 10, 50, 100, 250, 500, 750, 999]
        
        # ---------------------------------------------------------
        # 2. Test PQ Mode
        # ---------------------------------------------------------
        pq_path = os.path.join(temp_dir, "pq")
        # Use first 800 vectors for training codebooks (80% of dataset)
        training_data = vectors[:800]
        print(f"\nInitializing PQ DB (Training with {len(training_data)} vectors)...")
        db_pq = srvdb.SvDBPython.new_quantized(pq_path, training_data)
        
        print(f"Ingesting {n_vectors} vectors into PQ DB...")
        db_pq.add(ids, vectors, metadatas)
        # Persist to ensure data is on disk and mmapped back
        db_pq.persist()
        
        run_test(db_pq, "PQ", vectors, test_indices)

        # ---------------------------------------------------------
        # 3. Test Full Precision Mode (Baseline)
        # ---------------------------------------------------------
        fp_path = os.path.join(temp_dir, "full")
        print(f"\nInitializing Full Precision DB...")
        db_full = srvdb.SvDBPython(fp_path)
        
        print(f"Ingesting {n_vectors} vectors into Full Precision DB...")
        db_full.add(ids, vectors, metadatas)
        db_full.persist()
        
        # Test just the first index for baseline check
        run_test(db_full, "Full Precision", vectors, [0])
        
    except Exception as e:
        print(f"\n⛔ TEST FAILED WITH ERROR: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("\n" + "=" * 60)
        print("  DIAGNOSTIC COMPLETE")
        print("=" * 60)

if __name__ == "__main__":
    main()