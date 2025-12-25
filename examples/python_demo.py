"""
SvDB Python Example - ChromaDB-style API

Demonstrates using SvDB from Python with the familiar ChromaDB API pattern.
"""

import srvdb
import random

def main():
    print("🚀 SvDB Python Bindings Demo\n")
    print(f"SvDB Version: {srvdb.__version__}\n")

    # Initialize database
    print("Initializing database...")
    db = srvdb.SvDBPython("./python_demo_db")
    print(f"✓ Created: {db}\n")

    # Prepare data
    print("Preparing vectors...")
    ids = ["doc1", "doc2", "doc3"]
    
    # Generate random 1536-dimensional vectors
    embeddings = [
        [random.random() - 0.5 for _ in range(1536)] for _ in range(3)
    ]
    
    metadatas = [
        '{"title": "Document 1", "category": "tech"}',
        '{"title": "Document 2", "category": "science"}',
        '{"title": "Document 3", "category": "tech"}'
    ]

    # Add vectors in bulk
    print(f"Adding {len(ids)} vectors...")
    count = db.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"✓ Added {count} vectors\n")

    # Check count
    total = db.count()
    print(f"Total vectors in database: {total}\n")

    # Search for similar vectors
    print("Searching for similar vectors...")
    query = embeddings[0]  # Use first vector as query
    results = db.search(query=query, k=3)
    
    print(f"✓ Found {len(results)} results:\n")
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"  {i}. ID: {doc_id:10s} | Score: {score:.4f}")
    
    # Get metadata for a specific document
    print(f"\nRetrieving metadata for 'doc1'...")
    metadata = db.get("doc1")
    print(f"✓ Metadata: {metadata}\n")

    # Get all IDs
    all_ids = db.get_all_ids()
    print(f"All IDs in database: {all_ids}\n")

    # Delete a document
    print("Deleting 'doc2'...")
    deleted = db.delete(["doc2"])
    print(f"✓ Deleted {deleted} vector(s)\n")

    # Verify deletion
    remaining = db.count()
    print(f"Remaining vectors: {remaining}")
    remaining_ids = db.get_all_ids()
    print(f"Remaining IDs: {remaining_ids}\n")

    # Persist to disk
    print("Persisting to disk...")
    db.persist()
    print("✓ Data persisted successfully\n")

    print("🎉 Demo completed!")

if __name__ == "__main__":
    main()
