# Python API Reference

## Class: `srvdb.SrvDBPython`

The main interface for interacting with the SrvDB database.

### Initialization

#### `__init__(path: str, dimension: int = 1536, mode: str = "flat")`
Creates or opens a standard SrvDB instance.
- **path**: Directory path to store the database.
- **dimension**: Vector dimensionality (e.g., 1536 for OpenAI).
- **mode**: Indexing mode. Options: `"flat"`, `"hnsw"`, `"ivf"`, `"auto"`.

#### `new_hnsw(path, dimension, m=16, ef_construction=200, ef_search=50)`
Creates a new database optimized for HNSW graph search.
- **m**: Max connections per node (default: 16).
- **ef_construction**: Build-time accuracy (default: 200).
- **ef_search**: Search-time accuracy (default: 50).

#### `new_product_quantized(path, dimension, training_vectors)`
Creates a memory-efficient database using Product Quantization (PQ).
- **training_vectors**: A list of vectors (List[float]) to train the quantizer. 

### Methods

#### `add(ids, embeddings, metadatas) -> int`
Adds a batch of vectors to the database.
- **ids**: List of unique string IDs.
- **embeddings**: List of vectors (each is a List[float]).
- **metadatas**: List of JSON strings corresponding to each vector.
- **Returns**: Number of vectors successfully added.

#### `search(query, k=10) -> List[Tuple[str, float]]`
Performs a nearest-neighbor search.
- **query**: A single vector (List[float]).
- **k**: Number of results to return.
- **Returns**: List of `(id, score)` tuples. Score is Cosine Similarity (0.0 to 1.0).

#### `search_batch(queries, k=10) -> List[List[Tuple[str, float]]]`
Parallel batch search for multiple queries.
- **queries**: List of query vectors.
- **Returns**: List of result lists, one for each query.

#### `get(id) -> Optional[str]`
Retrieves metadata for a specific vector ID.
- **id**: The vector ID.
- **Returns**: Metadata string or `None` if not found.

#### `count() -> int`
Returns the total number of vectors in the database.

#### `persist()`
Flushes all data to disk. Important to call after significant additions to ensure data safety.

#### `set_ef_search(ef: int)`
Updates the HNSW search depth parameter at runtime.
- **ef**: New size for the candidate list. Higher = better recall, slower search.
