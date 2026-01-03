# Quickstart

Get up and running with SrvDB in under 5 minutes.

## 1. Setup

First, ensure you have SrvDB installed (see [Installation](install.md)).
You'll also need `numpy` to generate some sample data.

```bash
pip install srvdb numpy
```

## 2. Initialize the Database

SrvDB persists data to a directory. If the directory doesn't exist, it will be created.

```python
import srvdb
import numpy as np

# Create/Open a database at "./my_database"
# Dimension must match your embedding model (e.g., 1536 for OpenAI)
db = srvdb.SrvDBPython("./my_database", dimension=1536)

print(f"Database initialized at: {db.path}")
```

## 3. Add Vectors

Let's generate some random vectors to simulate embeddings. In a real app, these would come from OpenAI, Cohere, or Hugging Face models.

```python
# Generate 1,000 random vectors (1536 dimensions)
num_vectors = 1000
dim = 1536
embeddings = np.random.randn(num_vectors, dim).astype(np.float32)

# Create IDs and Metadata
ids = [f"doc_{i}" for i in range(num_vectors)]
metadatas = [f'{{"title": "Document {i}", "category": "random"}}' for i in range(num_vectors)]

# Batch Insert
# Returns the number of vectors added
count = db.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)
db.persist() # Flush to disk ensures durability immediately

print(f"Successfully added {count} vectors.")
```

## 4. Search

Perform a k-Nearest Neighbor (k-NN) search.

```python
# Create a random query vector
query_vector = np.random.randn(dim).astype(np.float32).tolist()

# Search for the top 5 closest vectors
results = db.search(query=query_vector, k=5)

print("\nSearch Results:")
for id, score in results:
    # Get metadata for the result
    meta = db.get(id)
    print(f"ID: {id} | Score: {score:.4f} | Metadata: {meta}")
```

## Next Steps

- **Scaling Up**: Need to store 100k+ vectors? Switch to [HNSW Mode](concepts/modes.md#hnsw-graph-index).
- **Memory Constrained?**: Use [Product Quantization](concepts/modes.md#product-quantization-pq) to reduce memory usage by 32x.
