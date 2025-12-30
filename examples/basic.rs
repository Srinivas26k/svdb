//! Basic example demonstrating SrvDB v0.2.0 usage

use anyhow::Result;
use srvdb::core::types::DatabaseConfig;
use srvdb::{SrvDB, Vector, VectorEngine};

fn main() -> Result<()> {
    println!("🚀 SrvDB v0.2.0 - Zero-Gravity Vector Engine Demo\n");

    // Initialize database configuration
    // 1536 is common for OpenAI embeddings
    let dimension = 1536;
    let config = DatabaseConfig::new(dimension)?;

    println!(
        "Initializing database at ./demo_db with {} dimensions...",
        dimension
    );

    // Create database with explicit config
    let mut db = SrvDB::new_with_config("./demo_db", config)?;
    println!("✓ Database initialized\n");

    // Create some example vectors
    println!("Creating and adding vectors...");

    // Vector wrapper is still required for the Rust API
    let vec1 = Vector::new(vec![0.5; dimension]);
    let vec2 = Vector::new(vec![-0.3; dimension]);
    let vec3 = Vector::new(vec![0.8; dimension]);

    // Add vectors with metadata
    let id1 = db.add(&vec1, r#"{"title": "Document 1", "category": "tech"}"#)?;
    let id2 = db.add(&vec2, r#"{"title": "Document 2", "category": "science"}"#)?;
    let id3 = db.add(&vec3, r#"{"title": "Document 3", "category": "tech"}"#)?;

    println!("✓ Added 3 vectors with IDs: {}, {}, {}\n", id1, id2, id3);

    // Persist to ensure all data is searchable
    // (Search will also auto-persist, but explicit is good for examples)
    db.persist()?;

    // Search for similar vectors
    println!("Searching for vectors similar to vec1...");
    let query = Vector::new(vec![0.51; dimension]); // Slightly different from vec1
    let results = db.search(&query, 3)?;

    println!("✓ Found {} results:\n", results.len());
    for (i, result) in results.iter().enumerate() {
        println!(
            "  {}. ID: {} | Score: {:.4} | Metadata: {}",
            i + 1,
            result.id,
            result.score,
            result.metadata.as_ref().unwrap_or(&"None".to_string())
        );
    }

    // Retrieve metadata
    println!("\n Retrieving metadata for ID {}...", id1);
    let metadata = db.get_metadata(id1)?;
    println!(
        "✓ Metadata: {}\n",
        metadata.unwrap_or("Not found".to_string())
    );

    println!("🎉 Demo completed!");

    Ok(())
}
