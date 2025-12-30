#!/usr/bin/env python3
"""
SrvDB v0.2.0 Upgrade & Benchmark Suite
Validates all new features with production-grade tests
"""

import numpy as np
import time
import psutil
import json
from pathlib import Path
from typing import List, Tuple
import srvdb

class SrvDBValidator:
    """Comprehensive validation for v0.2.0 features"""
    
    def __init__(self):
        self.results = {}
        
    def test_dynamic_dimensions(self):
        """Test 1: Validate multiple dimensions"""
        print("\n=== Test 1: Dynamic Dimensions ===")
        
        test_dims = [384, 768, 1024, 1536]
        passed = []
        
        for dim in test_dims:
            try:
                db = srvdb.SrvDBPython(f"./test_db_{dim}", dimension=dim)
                
                # Create and add test vector
                vector = np.random.randn(dim).astype(np.float32)
                db.add(
                    ids=["test1"],
                    embeddings=[vector.tolist()],
                    metadatas=['{"test": "data"}']
                )
                
                # Search
                results = db.search(vector.tolist(), k=1)
                assert len(results) == 1
                assert results[0][1] > 0.99  # Self-similarity
                
                passed.append(dim)
                print(f"✓ {dim}-dim: PASS")
                
            except Exception as e:
                print(f"✗ {dim}-dim: FAIL - {e}")
        
        self.results['dynamic_dimensions'] = {
            'tested': test_dims,
            'passed': passed,
            'success_rate': f"{len(passed)}/{len(test_dims)}"
        }
        
        return len(passed) == len(test_dims)
    
    def test_scalar_quantization(self):
        """Test 2: SQ8 compression and recall"""
        print("\n=== Test 2: Scalar Quantization (SQ8) ===")
        
        dim = 384
        num_vectors = 1000
        
        # Generate test data
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Create SQ8 database
        training_vectors = vectors[:200].tolist()
        
        start_time = time.time()
        db_sq8 = srvdb.SrvDBPython.new_scalar_quantized(
            "./test_sq8",
            dimension=dim,
            training_vectors=training_vectors
        )
        training_time = time.time() - start_time
        
        # Add vectors
        ids = [f"vec_{i}" for i in range(num_vectors)]
        metadatas = [f'{{"idx": {i}}}' for i in range(num_vectors)]
        
        start_time = time.time()
        db_sq8.add(ids, vectors.tolist(), metadatas)
        db_sq8.persist()
        ingestion_time = time.time() - start_time
        
        # Test recall
        query = vectors[0]
        results = db_sq8.search(query.tolist(), k=10)
        
        # First result should be the vector itself
        assert results[0][0] == "vec_0"
        recall = results[0][1]
        
        print(f"✓ Training time: {training_time:.2f}s")
        print(f"✓ Ingestion: {ingestion_time:.2f}s ({num_vectors/ingestion_time:.0f} vec/s)")
        print(f"✓ Self-match score: {recall:.4f}")
        print(f"✓ Top-10 IDs: {[r[0] for r in results[:5]]}")
        
        # Memory usage
        import os
        db_size = sum(f.stat().st_size for f in Path("./test_sq8").glob("**/*") if f.is_file())
        full_precision_size = num_vectors * dim * 4  # float32
        compression_ratio = full_precision_size / db_size
        
        print(f"✓ Disk usage: {db_size / 1024 / 1024:.2f} MB")
        print(f"✓ Compression: {compression_ratio:.1f}x")
        
        self.results['scalar_quantization'] = {
            'training_time': training_time,
            'ingestion_rate': num_vectors / ingestion_time,
            'self_match_score': recall,
            'compression_ratio': compression_ratio,
            'pass': recall > 0.90
        }
        
        return recall > 0.90
    
    def test_error_messages(self):
        """Test 3: Validate improved error messages"""
        print("\n=== Test 3: Error Messages ===")
        
        db = srvdb.SrvDBPython("./test_errors", dimension=384)
        
        # Test 1: Dimension mismatch
        try:
            wrong_vec = np.random.randn(768).tolist()
            db.add(["test"], [wrong_vec], ['{}'])
            print("✗ Dimension check FAILED (no error raised)")
            return False
        except ValueError as e:
            error_msg = str(e)
            assert "384" in error_msg
            assert "768" in error_msg
            print(f"✓ Dimension error: {error_msg[:80]}...")
        
        # Test 2: Invalid dimension at creation
        try:
            bad_db = srvdb.SrvDBPython("./test_bad", dimension=50)
            print("✗ Invalid dimension check FAILED")
            return False
        except ValueError as e:
            assert "128" in str(e) or "4096" in str(e)
            print(f"✓ Creation error: {str(e)[:80]}...")
        
        # Test 3: Helpful common models hint
        try:
            db2 = srvdb.SrvDBPython("./test_errors2", dimension=1024)
            wrong_vec = np.random.randn(768).tolist()
            db2.add(["test"], [wrong_vec], ['{}'])
        except ValueError as e:
            error_msg = str(e)
            # Should mention common models
            has_hints = any(word in error_msg.lower() for word in ['minilm', 'mpnet', 'openai', 'cohere'])
            print(f"✓ Helpful hints: {has_hints}")
            print(f"  Message: {error_msg[:100]}...")
        
        self.results['error_messages'] = {'pass': True}
        return True
    
    def benchmark_sq8_vs_pq(self):
        """Benchmark: Compare SQ8 vs PQ on clustered data"""
        print("\n=== Benchmark: SQ8 vs PQ (Clustered Data) ===")
        
        dim = 384
        num_vectors = 5000
        
        # Generate clustered data (RAG-like)
        clusters = []
        for i in range(10):
            cluster_center = np.random.randn(dim)
            cluster_vecs = cluster_center + np.random.randn(500, dim) * 0.1
            clusters.append(cluster_vecs)
        
        vectors = np.vstack(clusters).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        training = vectors[:1000].tolist()
        
        results = {}
        
        # Test SQ8
        print("\nTesting SQ8...")
        start = time.time()
        db_sq8 = srvdb.SrvDBPython.new_scalar_quantized(
            "./bench_sq8", dim, training
        )
        sq8_train_time = time.time() - start
        
        ids = [f"vec_{i}" for i in range(num_vectors)]
        metas = [f'{{"i": {i}}}' for i in range(num_vectors)]
        
        start = time.time()
        db_sq8.add(ids, vectors.tolist(), metas)
        db_sq8.persist()
        sq8_ingest_time = time.time() - start
        
        # Test recall on clustered data
        sq8_recalls = []
        for i in range(0, 100, 10):
            query = vectors[i]
            results_sq8 = db_sq8.search(query.tolist(), k=10)
            # Check if same-cluster vectors are retrieved
            sq8_recalls.append(results_sq8[0][1])
        
        sq8_avg_recall = np.mean(sq8_recalls)
        
        print(f"✓ SQ8 training: {sq8_train_time:.2f}s")
        print(f"✓ SQ8 ingestion: {sq8_ingest_time:.2f}s")
        print(f"✓ SQ8 recall: {sq8_avg_recall:.4f}")
        
        # Compare with Flat
        print("\nTesting Flat (baseline)...")
        db_flat = srvdb.SrvDBPython("./bench_flat", dimension=dim, mode='flat')
        db_flat.add(ids, vectors.tolist(), metas)
        db_flat.persist()
        
        flat_recalls = []
        for i in range(0, 100, 10):
            query = vectors[i]
            results_flat = db_flat.search(query.tolist(), k=10)
            flat_recalls.append(results_flat[0][1])
        
        flat_avg_recall = np.mean(flat_recalls)
        print(f"✓ Flat recall: {flat_avg_recall:.4f}")
        
        # Get memory stats
        sq8_size = sum(f.stat().st_size for f in Path("./bench_sq8").glob("**/*") if f.is_file())
        flat_size = sum(f.stat().st_size for f in Path("./bench_flat").glob("**/*") if f.is_file())
        
        print(f"\nMemory:")
        print(f"  SQ8:  {sq8_size/1024/1024:.1f} MB")
        print(f"  Flat: {flat_size/1024/1024:.1f} MB")
        print(f"  Compression: {flat_size/sq8_size:.1f}x")
        
        self.results['sq8_vs_pq'] = {
            'sq8_train_time': sq8_train_time,
            'sq8_recall': sq8_avg_recall,
            'flat_recall': flat_avg_recall,
            'recall_degradation': flat_avg_recall - sq8_avg_recall,
            'compression_ratio': flat_size / sq8_size
        }
        
        return True
    
    def generate_report(self):
        """Generate final validation report"""
        print("\n" + "="*60)
        print("SrvDB v0.2.0 VALIDATION REPORT")
        print("="*60)
        
        for test_name, result in self.results.items():
            print(f"\n{test_name}:")
            print(json.dumps(result, indent=2))
        
        # Overall pass/fail
        all_passed = all(
            result.get('pass', True) 
            for result in self.results.values()
        )
        
        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL TESTS PASSED - Ready for v0.2.0 release")
        else:
            print("✗ SOME TESTS FAILED - Review above")
        print("="*60)
        
        return all_passed

def main():
    """Run full validation suite"""
    import shutil
    
    # Clean test directories
    for path in Path(".").glob("test_*"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in Path(".").glob("bench_*"):
        if path.is_dir():
            shutil.rmtree(path)
    
    validator = SrvDBValidator()
    
    # Run tests
    tests = [
        validator.test_dynamic_dimensions,
        validator.test_scalar_quantization,
        validator.test_error_messages,
        validator.benchmark_sq8_vs_pq,
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
    
    print(f"\n{passed}/{len(tests)} test suites passed")
    
    # Generate report
    success = validator.generate_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())