//! Memory-mapped vector storage module
//!
//! Manages the vectors.bin file using buffered writes for speed and mmap for reads.

use anyhow::{Context, Result};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use crate::types::{EmbeddedVector, VectorHeader};

const VECTOR_SIZE: usize = std::mem::size_of::<EmbeddedVector>();

pub struct VectorStorage {
    #[allow(dead_code)]
    file_path: PathBuf,
    writer: BufWriter<File>,
    mmap: Option<MmapMut>,
    count: u64,
}

impl Drop for VectorStorage {
    fn drop(&mut self) {
        // Ensure data is persisted when object is destroyed (e.g. Python gc)
        let _ = self.flush();
    }
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

        let mut count = 0;

        if exists {
            // Read existing count from header if file is not empty
            if file.metadata()?.len() >= VectorHeader::SIZE as u64 {
                // Map briefly just to read the header safely
                let mmap = unsafe { MmapOptions::new().map(&file)? };
                let header = unsafe { &*(mmap.as_ptr() as *const VectorHeader) };
                count = header.count;
            }
            // CRITICAL: Seek to end so BufWriter appends correctly after re-opening
            file.seek(SeekFrom::End(0))?;
        } else {
            // New file: Initialize with empty header
            let header = VectorHeader::new();
            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    &header as *const VectorHeader as *const u8,
                    VectorHeader::SIZE,
                )
            };
            file.write_all(header_bytes)?;
            file.sync_all()?;
        }

        // Wrap file in BufWriter with 1MB buffer for maximum ingestion speed
        let writer = BufWriter::with_capacity(1_048_576, file);

        // Initialize mmap for reading (if file has data)
        let file_ref = writer.get_ref();
        let mmap = if file_ref.metadata()?.len() > 0 {
            Some(unsafe { MmapOptions::new().map_mut(file_ref)? })
        } else {
            None
        };

        Ok(Self {
            file_path,
            writer,
            mmap,
            count,
        })
    }

    /// Append a full precision f32 vector to storage
    /// Uses buffered I/O for 5x speedup (no immediate mmap update/flush)
    pub fn append(&mut self, vector: &EmbeddedVector) -> Result<u64> {
        let id = self.count;

        // Serialize vector to bytes
        let vector_bytes = unsafe {
            std::slice::from_raw_parts(
                vector.as_ptr() as *const u8,
                VECTOR_SIZE,
            )
        };

        // Write to buffer (RAM only, fast). Do NOT flush here.
        self.writer.write_all(vector_bytes)?;
        
        // Update in-memory count
        self.count += 1;

        Ok(id)
    }

    /// Force flush to disk and update header
    pub fn flush(&mut self) -> Result<()> {
        // 1. Flush vector data buffer to disk first
        self.writer.flush()?;
        
        // 2. Update Header with new count
        let file = self.writer.get_mut();
        
        // Seek to start to update count
        file.seek(SeekFrom::Start(0))?;
        
        let mut header = VectorHeader::new();
        header.count = self.count;
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const VectorHeader as *const u8,
                VectorHeader::SIZE,
            )
        };
        file.write_all(header_bytes)?;
        
        // 3. CRITICAL: Seek to END for future appends (not old position!)
        file.seek(SeekFrom::End(0))?;
        
        // 4. Sync to physical disk to guarantee durability
        file.sync_all()?;
        
        // 5. Refresh mmap to reflect new data for reading
        drop(self.mmap.take());
        if file.metadata()?.len() > 0 {
             self.mmap = Some(unsafe { MmapOptions::new().map_mut(file as &File)? });
        }
        
        Ok(())
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
        
        // If mmap is missing (e.g. new file before first flush), we can't read yet
        let mmap = self.mmap.as_ref()?;
        
        let offset = VectorHeader::SIZE + (index as usize * VECTOR_SIZE);
        
        // Safety check against current mmap size
        if offset + VECTOR_SIZE <= mmap.len() {
            let slice = &mmap[offset..offset + VECTOR_SIZE];
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
}

/// Iterator over vectors in storage
pub struct VectorIterator<'a> {
    storage: &'a VectorStorage,
    index: u64,
}

impl<'a> Iterator for VectorIterator<'a> {
    type Item = (u64, &'a EmbeddedVector);

    fn next(&mut self) -> Option<Self::Item> {
        // Note: This iterates only over what is currently visible in mmap
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

        let vector = [0.5f32; 1536];
        let id = storage.append(&vector).unwrap();

        assert_eq!(id, 0);
        assert_eq!(storage.count(), 1);

        // Must flush to read back via get() (which uses mmap)
        storage.flush().unwrap();

        let retrieved = storage.get(id).unwrap();
        assert_eq!(retrieved, &vector);
    }

    #[test]
    fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        
        {
            let mut storage = VectorStorage::new(path).unwrap();
            let vector = [1.0f32; 1536];
            storage.append(&vector).unwrap();
            // Drop handles flush automatically
        }
        
        let storage = VectorStorage::new(path).unwrap();
        assert_eq!(storage.count(), 1);
        let vec = storage.get(0).unwrap();
        assert_eq!(vec[0], 1.0);
    }
}