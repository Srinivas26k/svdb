//! Memory-mapped vector storage module
//!
//! Manages the vectors.bin file using mmap for zero-copy access.

use anyhow::{Context, Result};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use crate::types::{QuantizedVector, VectorHeader};

pub struct VectorStorage {
    #[allow(dead_code)]
    file_path: PathBuf,
    file: File,
    mmap: Option<MmapMut>,
    count: u64,
}

impl VectorStorage {
    /// Create or open vector storage at the specified path
    pub fn new(db_path: &str) -> Result<Self> {
        let file_path = Path::new(db_path).join("vectors.bin");
        
        let exists = file_path.exists();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&file_path)
            .context("Failed to open vectors.bin")?;

        if !exists {
            // Initialize new file with header
            let header = VectorHeader::new();
            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    &header as *const VectorHeader as *const u8,
                    VectorHeader::SIZE,
                )
            };
            file.write_all(header_bytes)?;
            file.flush()?;
        }

        // Memory map the file
        let mmap = if file.metadata()?.len() > 0 {
            Some(unsafe { MmapOptions::new().map_mut(&file)? })
        } else {
            None
        };

        // Read count from header
        let count = if let Some(ref map) = mmap {
            if map.len() >= VectorHeader::SIZE {
                let header = unsafe {
                    &*(map.as_ptr() as *const VectorHeader)
                };
                header.count
            } else {
                0
            }
        } else {
            0
        };

        Ok(Self {
            file_path,
            file,
            mmap,
            count,
        })
    }

    /// Append a quantized vector to storage
    pub fn append(&mut self, vector: &QuantizedVector) -> Result<u64> {
        let id = self.count;

        // Calculate new file size
        let new_size = VectorHeader::SIZE + ((self.count as usize + 1) * 192);

        // Resize file
        self.file.set_len(new_size as u64)?;

        // Remap with new size
        drop(self.mmap.take()); // Unmap first
        let mut mmap = unsafe { MmapOptions::new().map_mut(&self.file)? };

        // Write vector data
        let offset = VectorHeader::SIZE + (self.count as usize * 192);
        mmap[offset..offset + 192].copy_from_slice(vector);

        // Update count in header
        self.count += 1;
        let header = unsafe {
            &mut *(mmap.as_mut_ptr() as *mut VectorHeader)
        };
        header.count = self.count;

        // Flush to disk
        mmap.flush()?;

        self.mmap = Some(mmap);

        Ok(id)
    }

    /// Get the number of vectors stored
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get a reference to a specific vector by index
    pub fn get(&self, index: u64) -> Option<&QuantizedVector> {
        if index >= self.count {
            return None;
        }

        let mmap = self.mmap.as_ref()?;
        let offset = VectorHeader::SIZE + (index as usize * 192);
        
        if offset + 192 <= mmap.len() {
            let slice = &mmap[offset..offset + 192];
            // Safe because we know the layout
            Some(unsafe { &*(slice.as_ptr() as *const QuantizedVector) })
        } else {
            None
        }
    }

    /// Iterate over all vectors
    pub fn iter(&self) -> VectorIterator<'_> {
        VectorIterator {
            storage: self,
            index: 0,
        }
    }

    /// Force flush to disk
    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut mmap) = self.mmap {
            mmap.flush()?;
        }
        self.file.sync_all()?;
        Ok(())
    }
}

/// Iterator over vectors in storage
pub struct VectorIterator<'a> {
    storage: &'a VectorStorage,
    index: u64,
}

impl<'a> Iterator for VectorIterator<'a> {
    type Item = (u64, &'a QuantizedVector);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.storage.count {
            return None;
        }

        let id = self.index;
        let vector = self.storage.get(id)?;
        self.index += 1;

        Some((id, vector))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = VectorStorage::new(temp_dir.path().to_str().unwrap());
        assert!(storage.is_ok());
    }

    #[test]
    fn test_append_and_get() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = VectorStorage::new(temp_dir.path().to_str().unwrap()).unwrap();

        let vector = [0xFF; 192];
        let id = storage.append(&vector).unwrap();

        assert_eq!(id, 0);
        assert_eq!(storage.count(), 1);

        let retrieved = storage.get(id).unwrap();
        assert_eq!(retrieved, &vector);
    }
}
