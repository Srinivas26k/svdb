//! Quantized vector storage with Product Quantization
//!
//! Key features:
//! - 32x memory compression (6KB → 192 bytes)
//! - Memory-mapped quantized vectors
//! - Codebook persistence
//! - Atomic operations for thread safety

use anyhow::{Context, Result};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::quantization::{ProductQuantizer, QuantizedVector};
use crate::types::VectorHeader;

const QVECTOR_SIZE: usize = std::mem::size_of::<QuantizedVector>();
const BUFFER_SIZE: usize = 8 * 1024 * 1024; // 8MB buffer

pub struct QuantizedVectorStorage {
    file_path: PathBuf,
    codebook_path: PathBuf,
    writer: BufWriter<File>,
    mmap: Option<MmapMut>,
    count: AtomicU64,
    last_flushed_count: u64,
    pub quantizer: ProductQuantizer,
}

impl Drop for QuantizedVectorStorage {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl QuantizedVectorStorage {
    /// Create new quantized storage with training data
    pub fn new_with_training(db_path: &str, training_data: &[Vec<f32>]) -> Result<Self> {
        let file_path = Path::new(db_path).join("quantized_vectors.bin");
        let codebook_path = Path::new(db_path).join("codebooks.bin");
        
        // Train quantizer
        println!("Training Product Quantizer on {} vectors...", training_data.len());
        // DISABLED: ProductQuantizer needs refactoring for dynamic dimensions
        anyhow::bail!("Product Quantization is temporarily disabled in v0.2.0. Use Flat or HNSW mode.");
        
        /* DISABLED CODE - needs quantization.rs refactoring
        let quantizer = ProductQuantizer::train(training_data, 20)?;
        println!("Training complete!");
        
        // Save codebooks
        let codebook_bytes = quantizer.to_bytes()?;
        std::fs::write(&codebook_path, codebook_bytes)
            .context("Failed to save codebooks")?;
        
        let exists = file_path.exists();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&file_path)
            .context("Failed to open quantized_vectors.bin")?;

        let mut count = 0;

        if exists {
            if file.metadata()?.len() >= VectorHeader::SIZE as u64 {
                let mmap = unsafe { MmapOptions::new().map(&file)? };
                let header = unsafe { &*(mmap.as_ptr() as *const VectorHeader) };
                count = header.count;
            }
            file.seek(SeekFrom::End(0))?;
        } else {
            // Assume first vector dimension
            let dimension = training_data.get(0).map(|v| v.len()).unwrap_or(1536);
            let header = VectorHeader::new_quantized(dimension, 2)?; // mode 2 = Product Quantization
            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    &header as *const VectorHeader as *const u8,
                    VectorHeader::SIZE,
                )
            };
            file.write_all(header_bytes)?;
            file.sync_all()?;
        }

        let writer = BufWriter::with_capacity(BUFFER_SIZE, file);
        let file_ref = writer.get_ref();
        let mmap = if file_ref.metadata()?.len() > VectorHeader::SIZE as u64 {
            Some(unsafe { MmapOptions::new().map_mut(file_ref)? })
        } else {
            None
        };

        Ok(Self {
            file_path,
            codebook_path,
            writer,
            mmap,
            count: AtomicU64::new(count),
            last_flushed_count: count,
            quantizer,
        })
        */
    }

    /// Load existing quantized storage
    pub fn load(db_path: &str) -> Result<Self> {
        let file_path = Path::new(db_path).join("quantized_vectors.bin");
        let codebook_path = Path::new(db_path).join("codebooks.bin");
        
        // Load codebooks
        let codebook_bytes = std::fs::read(&codebook_path)
            .context("Failed to read codebooks")?;
        let quantizer = ProductQuantizer::from_bytes(&codebook_bytes)?;
        
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&file_path)
            .context("Failed to open quantized_vectors.bin")?;

        let mut count = 0;
        if file.metadata()?.len() >= VectorHeader::SIZE as u64 {
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let header = unsafe { &*(mmap.as_ptr() as *const VectorHeader) };
            count = header.count;
        }
        
        file.seek(SeekFrom::End(0))?;

        let writer = BufWriter::with_capacity(BUFFER_SIZE, file);
        let file_ref = writer.get_ref();
        let mmap = if file_ref.metadata()?.len() > VectorHeader::SIZE as u64 {
            Some(unsafe { MmapOptions::new().map_mut(file_ref)? })
        } else {
            None
        };

        Ok(Self {
            file_path,
            codebook_path,
            writer,
            mmap,
            count: AtomicU64::new(count),
            last_flushed_count: count,
            quantizer,
        })
    }

    /// Append vector (quantizes automatically)
    #[inline]
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        let id = self.count.fetch_add(1, Ordering::Relaxed);

        // DISABLED: PQ encode
        anyhow::bail!("Product Quantization is temporarily disabled in v0.2.0")
        
        /* DISABLED - unreachable
        // Zero-copy write
        let qvec_bytes = unsafe {
            std::slice::from_raw_parts(
                qvec.as_ptr() as *const u8,
                QVECTOR_SIZE,
            )
        };

        self.writer.write_all(qvec_bytes)?;

        // Auto-flush when buffer is 90% full
        if self.writer.buffer().len() > (BUFFER_SIZE * 9 / 10) {
            self.writer.flush()?;
        }

        Ok(id)
        */
    }

    /// Batch append (optimized)
    pub fn append_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        let start_id = self.count.load(Ordering::Relaxed);
        let mut ids = Vec::with_capacity(vectors.len());

        for (i, vector) in vectors.iter().enumerate() {
            // DISABLED: PQ encode
            return Err(anyhow::anyhow!("Product Quantization is temporarily disabled in v0.2.0"));
            
            /* DISABLED - unreachable
            let qvec_bytes = unsafe {
                std::slice::from_raw_parts(
                    qvec.as_ptr() as *const u8,
                    QVECTOR_SIZE,
                )
            };
            self.writer.write_all(qvec_bytes)?;
            ids.push(start_id + i as u64);
            */
        }

        self.count.store(start_id + vectors.len() as u64, Ordering::Relaxed);
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
        
        // PQ disabled - use hardcoded dimension for now
        let dimension = 1536;
        let mut header = VectorHeader::new_quantized(dimension, 2)?; // mode 2 = Product Quantization
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

    /// Zero-copy access to quantized vector
    #[inline]
    pub fn get(&self, index: u64) -> Option<&QuantizedVector> {
        if index >= self.count.load(Ordering::Relaxed) {
            return None;
        }
        
        let mmap = self.mmap.as_ref()?;
        let offset = VectorHeader::SIZE + (index as usize * QVECTOR_SIZE);
        
        if offset + QVECTOR_SIZE <= mmap.len() {
            let slice = &mmap[offset..offset + QVECTOR_SIZE];
            Some(unsafe { &*(slice.as_ptr() as *const QuantizedVector) })
        } else {
            None
        }
    }

    /// Get batch of quantized vectors
    pub fn get_batch(&self, start: u64, count: usize) -> Option<&[QuantizedVector]> {
        let end = start + count as u64;
        if end > self.count.load(Ordering::Relaxed) {
            return None;
        }

        let mmap = self.mmap.as_ref()?;
        let offset = VectorHeader::SIZE + (start as usize * QVECTOR_SIZE);
        let size = count * QVECTOR_SIZE;

        if offset + size <= mmap.len() {
            let slice = &mmap[offset..offset + size];
            Some(unsafe {
                std::slice::from_raw_parts(
                    slice.as_ptr() as *const QuantizedVector,
                    count
                )
            })
        } else {
            None
        }
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> StorageStats {
        let count = self.count.load(Ordering::Relaxed);
        let quantized_bytes = count * QVECTOR_SIZE as u64;
        // PQ disabled - use hardcoded dimension for now  
        let avg_dimension = 1536;
        let full_precision_bytes = count * (avg_dimension * std::mem::size_of::<f32>()) as u64;
        
        StorageStats {
            vector_count: count,
            memory_bytes: quantized_bytes,
            full_precision_bytes,
            compression_ratio: full_precision_bytes as f32 / quantized_bytes.max(1) as f32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub vector_count: u64,
    pub memory_bytes: u64,
    pub full_precision_bytes: u64,
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_quantized_storage() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create training data
        let training: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let mut v = vec![0.0f32; 1536];
                v[0] = i as f32 / 100.0;
                v
            })
            .collect();

        // Convert training data to reference slice
        // let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

        let mut storage = QuantizedVectorStorage::new_with_training(
            temp_dir.path().to_str().unwrap(),
            &training
        ).unwrap();

        // Test append
        let test_vec = vec![0.5f32; 1536];
        let id = storage.append(&test_vec).unwrap();
        assert_eq!(id, 0);

        storage.flush().unwrap();

        // Test retrieval
        let qvec = storage.get(0).unwrap();
        assert_eq!(qvec.len(), 192);

        // Test stats
        let stats = storage.get_stats();
        assert_eq!(stats.vector_count, 1);
        assert!(stats.compression_ratio > 30.0); // ~32x compression
    }
}
