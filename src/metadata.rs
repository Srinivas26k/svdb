//! Optimized metadata storage with batch operations
//!
//! Improvements:
//! - Batch set/get operations
//! - Write-through caching for hot metadata
//! - Reduced transaction overhead

use anyhow::{Context, Result};
use redb::{Database, ReadableTableMetadata, TableDefinition};
use std::path::Path;
use std::collections::HashMap;

const METADATA_TABLE: TableDefinition<u64, &str> = TableDefinition::new("metadata");
const CACHE_SIZE: usize = 1000; // Cache last 1000 accessed items

pub struct MetadataStore {
    db: Database,
    write_cache: HashMap<u64, String>,
    cache_dirty: bool,
}

impl MetadataStore {
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

        Ok(Self {
            db,
            write_cache: HashMap::with_capacity(CACHE_SIZE),
            cache_dirty: false,
        })
    }

    /// Set metadata with write-through cache
    pub fn set(&mut self, id: u64, metadata: &str) -> Result<()> {
        // Add to cache
        self.write_cache.insert(id, metadata.to_string());
        self.cache_dirty = true;

        // Flush if cache is full
        if self.write_cache.len() >= CACHE_SIZE {
            self.flush_cache()?;
        }

        Ok(())
    }

    /// Batch set operation (much faster than individual sets)
    pub fn set_batch(&mut self, items: &[(u64, String)]) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            for (id, metadata) in items {
                table.insert(*id, metadata.as_str())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get metadata with cache lookup
    pub fn get(&self, id: u64) -> Result<Option<String>> {
        // Check cache first
        if let Some(metadata) = self.write_cache.get(&id) {
            return Ok(Some(metadata.clone()));
        }

        // Fall back to database
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;
        
        let result = table.get(id)?;
        Ok(result.map(|v| v.value().to_string()))
    }

    /// Batch get operation
    pub fn get_batch(&self, ids: &[u64]) -> Result<Vec<Option<String>>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;
        
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            // Check cache first
            if let Some(metadata) = self.write_cache.get(id) {
                results.push(Some(metadata.clone()));
            } else {
                let result = table.get(*id)?;
                results.push(result.map(|v| v.value().to_string()));
            }
        }
        
        Ok(results)
    }

    pub fn delete(&mut self, id: u64) -> Result<()> {
        // Remove from cache
        self.write_cache.remove(&id);

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.remove(id)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Flush write cache to disk
    fn flush_cache(&mut self) -> Result<()> {
        if !self.cache_dirty || self.write_cache.is_empty() {
            return Ok(());
        }

        let items: Vec<_> = self.write_cache
            .iter()
            .map(|(id, meta)| (*id, meta.clone()))
            .collect();

        self.set_batch(&items)?;
        self.write_cache.clear();
        self.cache_dirty = false;

        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.flush_cache()?;
        Ok(())
    }

    /// Get total count of metadata entries
    pub fn count(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;
        Ok(table.len()? as usize)
    }
}

impl Drop for MetadataStore {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_write_cache() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = MetadataStore::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Add to cache
        for i in 0..500 {
            store.set(i, &format!(r#"{{"id": {}}}"#, i)).unwrap();
        }

        // Should still be in cache (not flushed)
        assert!(store.cache_dirty);
        assert_eq!(store.write_cache.len(), 500);

        // Get from cache
        let result = store.get(100).unwrap();
        assert_eq!(result, Some(r#"{"id": 100}"#.to_string()));

        // Flush
        store.flush().unwrap();
        assert!(!store.cache_dirty);
        assert_eq!(store.write_cache.len(), 0);
    }

    #[test]
    fn test_batch_operations() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = MetadataStore::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Batch set
        let items: Vec<_> = (0..100)
            .map(|i| (i, format!(r#"{{"id": {}}}"#, i)))
            .collect();

        store.set_batch(&items).unwrap();

        // Batch get
        let ids: Vec<u64> = (0..100).collect();
        let results = store.get_batch(&ids).unwrap();

        assert_eq!(results.len(), 100);
        assert!(results.iter().all(|r| r.is_some()));
    }

    #[test]
    fn test_auto_flush_on_full_cache() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = MetadataStore::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Fill cache (should auto-flush at CACHE_SIZE)
        for i in 0..(CACHE_SIZE + 100) {
            store.set(i as u64, &format!(r#"{{"id": {}}}"#, i)).unwrap();
        }

        // Cache should have been flushed
        assert!(store.write_cache.len() < CACHE_SIZE);
    }

    #[test]
    fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        {
            let mut store = MetadataStore::new(path).unwrap();
            store.set(42, r#"{"test": "data"}"#).unwrap();
            store.flush().unwrap();
        }

        // Reopen and verify
        let store = MetadataStore::new(path).unwrap();
        let result = store.get(42).unwrap();
        assert_eq!(result, Some(r#"{"test": "data"}"#.to_string()));
    }
}