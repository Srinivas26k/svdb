//! Metadata storage module using redb
//!
//! Manages the metadata.db key-value store.

use anyhow::{Context, Result};
use redb::{Database, TableDefinition};
use std::path::Path;

const METADATA_TABLE: TableDefinition<u64, &str> = TableDefinition::new("metadata");

pub struct MetadataStore {
    db: Database,
}

impl MetadataStore {
    /// Create or open metadata store
    pub fn new(db_path: &str) -> Result<Self> {
        let db_file = Path::new(db_path).join("metadata.db");
        
        let db = Database::create(db_file)
            .context("Failed to create metadata.db")?;

        // Initialize table
        let write_txn = db.begin_write()?;
        {
            let _table = write_txn.open_table(METADATA_TABLE)?;
        }
        write_txn.commit()?;

        Ok(Self { db })
    }

    /// Set metadata for a vector ID
    pub fn set(&self, id: u64, metadata: &str) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.insert(id, metadata)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get metadata for a vector ID
    pub fn get(&self, id: u64) -> Result<Option<String>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;
        
        let result = table.get(id)?;
        Ok(result.map(|v| v.value().to_string()))
    }

    /// Delete metadata for a vector ID
    pub fn delete(&self, id: u64) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.remove(id)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Force flush to disk
    pub fn flush(&self) -> Result<()> {
        // redb handles flushing automatically with transactions
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_metadata_operations() {
        let temp_dir = TempDir::new().unwrap();
        let store = MetadataStore::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Test set and get
        store.set(0, r#"{"title": "test"}"#).unwrap();
        let result = store.get(0).unwrap();
        assert_eq!(result, Some(r#"{"title": "test"}"#.to_string()));

        // Test non-existent key
        let result = store.get(999).unwrap();
        assert_eq!(result, None);

        // Test delete
        store.delete(0).unwrap();
        let result = store.get(0).unwrap();
        assert_eq!(result, None);
    }
}
