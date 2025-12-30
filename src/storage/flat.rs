//! Dynamic dimension vector storage for SrvDB v0.2.0
//! Supports 128-4096 dimensions with zero-copy mmap reads

use crate::types::VectorHeader;
use anyhow::{Context, Result};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const BUFFER_SIZE: usize = 8 * 1024 * 1024; // 8MB buffer

// Type alias for backward compatibility
pub type VectorStorage = DynamicVectorStorage;

pub struct DynamicVectorStorage {
    writer: BufWriter<File>,
    mmap: Option<MmapMut>,
    count: AtomicU64,
    dimension: usize,
    vector_size_bytes: usize,
    last_flushed_count: u64,
}

impl Drop for DynamicVectorStorage {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl DynamicVectorStorage {
    /// Create or open storage with specified dimension
    pub fn new(db_path: &str, dimension: usize) -> Result<Self> {
        let file_path = Path::new(db_path).join("vectors.bin");
        let exists = file_path.exists();

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)
            .context("Failed to open vectors.bin")?;

        let mut count = 0;
        let mut stored_dimension = dimension;

        if exists {
            // Read and validate existing header
            if file.metadata()?.len() >= VectorHeader::SIZE as u64 {
                let mmap = unsafe { MmapOptions::new().map(&file)? };
                let header = unsafe { &*(mmap.as_ptr() as *const VectorHeader) };
                header.validate()?;

                count = header.count;
                stored_dimension = header.dimension as usize;

                // Ensure dimension matches
                if stored_dimension != dimension {
                    anyhow::bail!(
                        "Database dimension mismatch: expected {}, found {}. \
                        Cannot change dimension of existing database.",
                        dimension,
                        stored_dimension
                    );
                }
            }
            file.seek(SeekFrom::End(0))?;
        } else {
            // Write new header
            let header = VectorHeader::new(dimension)?;
            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    &header as *const VectorHeader as *const u8,
                    VectorHeader::SIZE,
                )
            };
            file.write_all(header_bytes)?;
            file.sync_all()?;
        }

        let vector_size_bytes = dimension * std::mem::size_of::<f32>();
        let writer = BufWriter::with_capacity(BUFFER_SIZE, file);

        let file_ref = writer.get_ref();
        let mmap = if file_ref.metadata()?.len() > VectorHeader::SIZE as u64 {
            Some(unsafe { MmapOptions::new().map_mut(file_ref)? })
        } else {
            None
        };

        Ok(Self {
            writer,
            mmap,
            count: AtomicU64::new(count),
            dimension,
            vector_size_bytes,
            last_flushed_count: count,
        })
    }

    /// Append vector with zero-copy serialization
    #[inline]
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dimension {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            );
        }

        let id = self.count.fetch_add(1, Ordering::Relaxed);

        // Zero-copy write
        let vector_bytes = unsafe {
            std::slice::from_raw_parts(vector.as_ptr() as *const u8, self.vector_size_bytes)
        };

        self.writer.write_all(vector_bytes)?;

        // Auto-flush when buffer is 90% full
        if self.writer.buffer().len() > (BUFFER_SIZE * 9 / 10) {
            self.writer.flush()?;
        }

        Ok(id)
    }

    /// Optimized batch append
    pub fn append_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        let start_id = self.count.load(Ordering::Relaxed);
        let mut ids = Vec::with_capacity(vectors.len());

        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dimension {
                anyhow::bail!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    i,
                    self.dimension,
                    vector.len()
                );
            }

            let vector_bytes = unsafe {
                std::slice::from_raw_parts(vector.as_ptr() as *const u8, self.vector_size_bytes)
            };
            self.writer.write_all(vector_bytes)?;
            ids.push(start_id + i as u64);
        }

        self.count
            .store(start_id + vectors.len() as u64, Ordering::Relaxed);
        self.writer.flush()?;

        Ok(ids)
    }

    pub fn flush(&mut self) -> Result<()> {
        let current_count = self.count.load(Ordering::Relaxed);

        if current_count == self.last_flushed_count {
            return Ok(());
        }

        self.writer.flush()?;

        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;

        let mut header = VectorHeader::new(self.dimension)?;
        header.count = current_count;
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const VectorHeader as *const u8,
                VectorHeader::SIZE,
            )
        };
        file.write_all(header_bytes)?;
        file.seek(SeekFrom::End(0))?;
        file.sync_all()?;

        drop(self.mmap.take());
        if file.metadata()?.len() > VectorHeader::SIZE as u64 {
            self.mmap = Some(unsafe { MmapOptions::new().map_mut(file as &File)? });
        }

        self.last_flushed_count = current_count;
        Ok(())
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Zero-copy vector access
    #[inline]
    pub fn get(&self, index: u64) -> Option<&[f32]> {
        if index >= self.count.load(Ordering::Relaxed) {
            return None;
        }

        let mmap = self.mmap.as_ref()?;
        let offset = VectorHeader::SIZE + (index as usize * self.vector_size_bytes);

        if offset + self.vector_size_bytes <= mmap.len() {
            let slice = &mmap[offset..offset + self.vector_size_bytes];
            Some(unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const f32, self.dimension)
            })
        } else {
            None
        }
    }

    /// Get batch of vectors for SIMD processing
    pub fn get_batch(&self, start: u64, count: usize) -> Option<Vec<&[f32]>> {
        let end = start + count as u64;
        if end > self.count.load(Ordering::Relaxed) {
            return None;
        }

        let mut batch = Vec::with_capacity(count);
        for i in 0..count {
            if let Some(vec) = self.get(start + i as u64) {
                batch.push(vec);
            } else {
                return None;
            }
        }

        Some(batch)
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let count = self.count.load(Ordering::Relaxed) as usize;
        VectorHeader::SIZE + (count * self.vector_size_bytes)
    }
}

// ScalarQuantizedStorage moved to sq.rs
