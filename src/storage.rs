//! Memory-mapped vector storage module
//!
//! Manages the vectors.bin file using mmap for zero-copy access.

use anyhow::{Context, Result};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use crate::types::{EmbeddedVector, VectorHeader};

const VECTOR_SIZE: usize = 6144; // 1536 floats * 4 bytes = 6KB per vector

pub struct VectorStorage {
    #[allow(dead_code)]
    file_path: PathBuf,
    writer: BufWriter<File>,
    mmap: Option<MmapMut>,
    count: u64,
}

impl VectorStorage {
    /// Create or open vector storage at the specified path
    pub fn new(db_path: &str) -> Result<Self> {
        let file_path = Path::new(db_path).join("vectors.bin");
        
        let exists = file_path.exists();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&file_path)
            .context("Failed to open vectors.bin")?;

        // Wrap file in BufWriter with 1MB buffer for maximum ingestion speed
        let mut writer = BufWriter::with_capacity(1_048_576, file);

        if !exists {
            // Initialize new file with header
            let header = VectorHeader::new();
            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    &header as *const VectorHeader as *const u8,
                    VectorHeader::SIZE,
                )
            };
            writer.write_all(header_bytes)?;
            writer.flush()?;
        }

        // Get reference to underlying file for mmap
        let file_ref = writer.get_ref();

        // Memory map the file
        let mmap = if file_ref.metadata()?.len() > 0 {
            Some(unsafe { MmapOptions::new().map_mut(file_ref)? })
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
            writer,
            mmap,
            count,
        })
    }

    /// Append a full precision f32 vector to storage
    pub fn append(&mut self, vector: &EmbeddedVector) -> Result<u64> {
        let id = self.count;

        // Calculate new file size (header + all vectors)
        let new_size = VectorHeader::SIZE + ((self.count as usize + 1) * VECTOR_SIZE);

        // Flush buffer before resizing and remapping
        self.writer.flush()?;

        // Get mutable reference to underlying file for resizing
        let file = self.writer.get_mut();

        // Resize file
        file.set_len(new_size as u64)?;

        // Remap with new size (need immutable reference for mmap)
        drop(self.mmap.take()); // Unmap first
        let mut mmap = unsafe { MmapOptions::new().map_mut(file as &File)? };

        // Write vector data as raw bytes
        let offset = VectorHeader::SIZE + (self.count as usize * VECTOR_SIZE);
        let vector_bytes = unsafe {
            std::slice::from_raw_parts(
                vector.as_ptr() as *const u8,
                VECTOR_SIZE,
            )
        };
        mmap[offset..offset + VECTOR_SIZE].copy_from_slice(vector_bytes);

        // Update count in header
        self.count += 1;
        let header = unsafe {
            &mut *(mmap.as_mut_ptr() as *mut VectorHeader)
        };
        header.count = self.count;

        // Flush mmap to disk (ensures header update is persisted)
        mmap.flush()?;

        self.mmap = Some(mmap);

        Ok(id)
    }

    /// Get the number of vectors stored
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get a reference to a specific vector by index
    pub fn get(&self, index: u64) -> Option<&EmbeddedVector> {
        if index >= self.count {
            return None;
        }

        let mmap = self.mmap.as_ref()?;
        let offset = VectorHeader::SIZE + (index as usize * VECTOR_SIZE);
        
        if offset + VECTOR_SIZE <= mmap.len() {
            let slice = &mmap[offset..offset + VECTOR_SIZE];
            // Safe because we know the layout and alignment
            Some(unsafe { &*(slice.as_ptr() as *const EmbeddedVector) })
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
        // Flush the buffer first
        self.writer.flush()?;
        
        // Sync underlying file to disk
        self.writer.get_ref().sync_all()?;
        
        // Flush mmap if present
        if let Some(ref mut mmap) = self.mmap {
            mmap.flush()?;
        }
        
        Ok(())
    }
}

// CRITICAL: Implement Drop to ensure buffer is flushed when VectorStorage is dropped
// This prevents data loss when Python objects are garbage collected
impl Drop for VectorStorage {
    fn drop(&mut self) {
        // Attempt to flush the buffer on drop
        // Ignore errors in drop() as per Rust conventions
        let _ = self.writer.flush();
        let _ = self.writer.get_ref().sync_all();
    }
}

/// Iterator over vectors in storage
pub struct VectorIterator<'a> {
    storage: &'a VectorStorage,
    index: u64,
}

impl<'a> Iterator for VectorIterator<'a> {
    type Item = (u64, &'a EmbeddedVector);

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

        // Create a test f32 vector
        let vector = [0.5f32; 1536];
        let id = storage.append(&vector).unwrap();

        assert_eq!(id, 0);
        assert_eq!(storage.count(), 1);

        let retrieved = storage.get(id).unwrap();
        assert_eq!(retrieved, &vector);
    }

    #[test]
    fn test_multiple_vectors() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = VectorStorage::new(temp_dir.path().to_str().unwrap()).unwrap();

        for i in 0..10 {
            let mut vector = [0.0f32; 1536];
            vector[0] = i as f32;
            storage.append(&vector).unwrap();
        }

        assert_eq!(storage.count(), 10);

        // Verify first vector
        let vec0 = storage.get(0).unwrap();
        assert_eq!(vec0[0], 0.0);

        // Verify last vector
        let vec9 = storage.get(9).unwrap();
        assert_eq!(vec9[0], 9.0);
    }
}
