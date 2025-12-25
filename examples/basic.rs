//! Basic example demonstrating SvDB usage

use srvdb::{SvDB, Vector, VectorEngine};
use anyhow::Result;

fn main() -> Result<()> {
    println!("🚀 SvDB - Zero-Gravity Vector Engine Demo\n");

    // Initialize database
    println!("Initializing database at ./demo_db...");
    let mut db = SvDB::new("./demo_db")?;
    println!("✓ Database initialized\n");

    // Create some example vectors (1536 dimensions)
    println!("Creating and adding vectors...");
    
    let vec1 = Vector::new(vec![0.5; 1536]);
    let vec2 = Vector::new(vec![-0.3; 1536]);
    let vec3 = Vector::new(vec![0.8; 1536]);
    
    // Add vectors with metadata
    let id1 = db.add(&vec1, r#"{"title": "Document 1", "category": "tech"}"#)?;
    let id2 = db.add(&vec2, r#"{"title": "Document 2", "category": "science"}"#)?;
    let id3 = db.add(&vec3, r#"{"title": "Document 3", "category": "tech"}"#)?;
    
    println!("✓ Added 3 vectors with IDs: {}, {}, {}\n", id1, id2, id3);

    // Search for similar vectors
    println!("Searching for vectors similar to vec1...");
    let query = Vector::new(vec![0.51; 1536]); // Slightly different from vec1
    let results = db.search(&query, 3)?;

    println!("✓ Found {} results:\n", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("  {}. ID: {} | Score: {:.4} | Metadata: {}", 
            i + 1, 
            result.id, 
            result.score,
            result.metadata.as_ref().unwrap_or(&"None".to_string())
        );
    }

    // Retrieve metadata
    println!("\n Retrieving metadata for ID {}...", id1);
    let metadata = db.get_metadata(id1)?;
    println!("✓ Metadata: {}\n", metadata.unwrap_or("Not found".to_string()));

    // Persist to disk
    println!("Persisting to disk...");
    db.persist()?;
    println!("✓ Data persisted successfully\n");

    println!("🎉 Demo completed!");

    Ok(())
}
