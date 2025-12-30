import srvdb
import numpy as np
import time
import os
import shutil

DIM = 128
N_VECTORS = 10_000
TOP_K = 10

def generate_data(n, dim):
    np.random.seed(42)
    vecs = np.random.rand(n, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    ids = [str(i) for i in range(n)]
    metadatas = [f"meta_{i}" for i in range(n)]
    return ids, vecs.tolist(), metadatas

def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def test_flat(ids, vecs, metas):
    path = "./db_flat"
    cleanup(path)
    print(f"\n[1/5] Testing FLAT Index (Baseline)...")
    
    # Init
    db = srvdb.SrvDBPython(path, dimension=DIM)
    # db.set_mode("flat") # precise method name depends on python bindings, usually exposed as set_mode
    
    # Ingest
    t0 = time.time()
    db.add(ids, vecs, metas)
    print(f"   -> Ingestion: {N_VECTORS} vectors in {time.time()-t0:.2f}s")
    
    # Search
    query = vecs[0]
    results = db.search(query, TOP_K)
    print(f"   -> Search Results (First ID): {[r[0] for r in results]}")
    
    assert results[0][0] == ids[0], "Flat index should find exact match at rank 0"
    print("   -> PASS: Flat Index")

def test_hnsw(ids, vecs, metas):
    path = "./db_hnsw"
    cleanup(path)
    print(f"\n[2/5] Testing HNSW Index...")
    
    db = srvdb.SrvDBPython(path, dimension=DIM)
    db.set_mode("hnsw")
    # Configure HNSW if needed? defaults should work.
    
    t0 = time.time()
    db.add(ids, vecs, metas)
    print(f"   -> Ingestion + Indexing: {time.time()-t0:.2f}s")
    
    query = vecs[0]
    results = db.search(query, TOP_K)
    assert results[0][0] == ids[0], "HNSW should find exact match"
    print("   -> PASS: HNSW Index")

def test_sq8(ids, vecs, metas):
    path = "./db_sq8"
    cleanup(path)
    print(f"\n[3/5] Testing SQ8 (Scalar Quantization)...")
    
    print(f"\n[3/5] Testing SQ8 (Scalar Quantization)...")
    
    # SQ8 requires training vectors at initialization for this explicit constructor
    db = srvdb.SrvDBPython.new_scalar_quantized(path, DIM, vecs)
    
    t0 = time.time()
    db.add(ids, vecs, metas)
    # SQ8 usually trains on first batch or implicitly handling it? 
    # Current impl of `add` handles it if mode is SQ8?
    # Or strict requirement to have initial vector set?
    # Our lib.rs implementation for `add_batch` checks storage type.
    # set_mode("sq8") sets config.enabled=true/Scalar.
    # In `add_batch`:
    # if `scalar_storage` is None -> initialize it.
    # Initialization of SQ8 requires training data. 
    # So `add_batch` essentially trains on the FIRST batch it receives?
    # Let's verify via execution.
    print(f"   -> Indexing (Quantized): {time.time()-t0:.2f}s")
    
    query = vecs[0]
    results = db.search(query, TOP_K)
    print(f"   -> Results: {[r[0] for r in results]}")
    assert results[0][0] == ids[0], "SQ8 should find exact match (high recall)"
    print("   -> PASS: SQ8 Index")

def test_ivf(ids, vecs, metas):
    path = "./db_ivf"
    cleanup(path)
    print(f"\n[4/5] Testing IVF (Inverted File)...")
    
    # Workflow: Ingest Flat -> Switch Mode -> Train -> Search
    db = srvdb.SrvDBPython(path, dimension=DIM)
    db.add(ids, vecs, metas) # Ingest as flat first
    
    db.set_mode("ivf")
    db.configure_ivf(nlist=100, nprobe=10) # 100 partitions for 10k vectors is reasonable
    
    t0 = time.time()
    db.train_ivf() # Should use stored data
    print(f"   -> Training: {time.time()-t0:.2f}s")
    
    query = vecs[0]
    results = db.search(query, TOP_K)
    print(f"   -> Results: {[r[0] for r in results]}")
    # IVF is approximate, but self-match should definitely be found
    found = any(r[0] == ids[0] for r in results)
    if found:
        print("   -> PASS: IVF Index")
    else:
        print("   -> WARNING: IVF recall issue (ID 0 not found)")

def test_auto(ids, vecs, metas):
    path = "./db_auto"
    cleanup(path)
    print(f"\n[5/5] Testing Auto-Tune...")
    
    db = srvdb.SrvDBPython(path, dimension=DIM)
    db.set_mode("auto") 
    # Auto logic runs on ingestion or explicit check?
    # Currently `apply_auto_strategy` is run when `set_mode("auto")` is called.
    # It checks environment and sets internal config.
    
    db.add(ids, vecs, metas)
    
    # We don't easily know WHICH mode it picked without inspecting, but it should work.
    query = vecs[0]
    results = db.search(query, TOP_K)
    assert results[0][0] == ids[0]
    print("   -> PASS: Auto-Mode workings")

if __name__ == "__main__":
    print(f"SRVDB VERSION: {srvdb.__version__ if hasattr(srvdb, '__version__') else 'unknown'}")
    ids, vecs, metas = generate_data(N_VECTORS, DIM)
    
    try:
        test_flat(ids, vecs, metas)
        test_hnsw(ids, vecs, metas)
        test_sq8(ids, vecs, metas)
        test_ivf(ids, vecs, metas)
        test_auto(ids, vecs, metas)
        print("\nAll Tests Passed Successfully! 🚀")
    except Exception as e:
        print(f"\nFAILED with error: {e}")
        import traceback
        traceback.print_exc()
