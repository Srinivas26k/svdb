//! Scalar Quantized Storage (SQ8)
//! 4x compression with direct u8 storage

use crate::types::VectorHeader;
use anyhow::{Context, Result};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering}; // Will need fixing to crate::core::types

const BUFFER_SIZE: usize = 8 * 1024 * 1024; // 8MB buffer

pub struct ScalarQuantizedStorage {
    writer: BufWriter<File>,
    mmap: Option<MmapMut>,
    count: AtomicU64,
    dimension: usize,
    quantizer: crate::types::ScalarQuantizer,
    last_flushed_count: u64,
}

impl Drop for ScalarQuantizedStorage {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl ScalarQuantizedStorage {
    /// Create new SQ8 storage with training data
    pub fn new_with_training(
        db_path: &str,
        dimension: usize,
        training_data: &[Vec<f32>],
    ) -> Result<Self> {
        if training_data.is_empty() {
            anyhow::bail!("Training data required for scalar quantization");
        }

        // Train quantizer
        let quantizer = crate::types::ScalarQuantizer::train(training_data)?;

        // Save quantizer
        let quantizer_path = Path::new(db_path).join("scalar_quantizer.bin");
        let quantizer_bytes = bincode::serialize(&quantizer)?;
        std::fs::write(quantizer_path, quantizer_bytes)?;

        // Create quantized vectors file
        let file_path = Path::new(db_path).join("sq8_vectors.bin");
        let exists = file_path.exists();

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)
            .context("Failed to open sq8_vectors.bin")?;

        let mut count = 0;

        if exists {
            // Read existing header
            if file.metadata()?.len() >= VectorHeader::SIZE as u64 {
                let mmap = unsafe { MmapOptions::new().map(&file)? };
                let header = unsafe { &*(mmap.as_ptr() as *const VectorHeader) };
                header.validate()?;
                count = header.count;

                // Validate dimension
                if header.dimension as usize != dimension {
                    anyhow::bail!(
                        "Dimension mismatch: expected {}, found {}",
                        dimension,
                        header.dimension
                    );
                }
            }
            file.seek(SeekFrom::End(0))?;
        } else {
            // Write new header with SQ8 marker
            let mut header = VectorHeader::new(dimension)?;
            header.quantization_mode = 1; // SQ8 = 1
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
            writer,
            mmap,
            count: AtomicU64::new(count),
            dimension,
            quantizer,
            last_flushed_count: count,
        })
    }

    /// Load existing SQ8 storage
    pub fn load(db_path: &str, dimension: usize) -> Result<Self> {
        let quantizer_path = Path::new(db_path).join("scalar_quantizer.bin");
        let quantizer_bytes = std::fs::read(quantizer_path)?;
        let quantizer = bincode::deserialize(&quantizer_bytes)?;

        let file_path = Path::new(db_path).join("sq8_vectors.bin");
        let mut file = OpenOptions::new().read(true).write(true).open(&file_path)?;

        let mut count = 0;
        if file.metadata()?.len() >= VectorHeader::SIZE as u64 {
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let header = unsafe { &*(mmap.as_ptr() as *const VectorHeader) };
            header.validate()?;
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
            writer,
            mmap,
            count: AtomicU64::new(count),
            dimension,
            quantizer,
            last_flushed_count: count,
        })
    }

    /// Append vector (automatically quantizes to u8)
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dimension {
            anyhow::bail!(
                "Vector dimension {} doesn't match expected {}",
                vector.len(),
                self.dimension
            );
        }

        let id = self.count.fetch_add(1, Ordering::Relaxed);

        // Quantize to u8 (1 byte per dimension)
        let encoded = self.quantizer.encode(vector);

        // Write u8 bytes directly (TRUE compression!)
        self.writer.write_all(&encoded)?;

        if self.writer.buffer().len() > (BUFFER_SIZE * 9 / 10) {
            self.flush()?;
        }

        Ok(id)
    }

    /// Append batch of vectors
    pub fn append_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        let mut ids = Vec::with_capacity(vectors.len());
        for vector in vectors {
            ids.push(self.append(vector)?);
        }
        Ok(ids)
    }

    /// Get and decode vector from u8 storage
    pub fn get(&self, index: u64) -> Option<Vec<f32>> {
        let mmap = self.mmap.as_ref()?;

        if index >= self.count.load(Ordering::Relaxed) {
            return None;
        }

        // Calculate offset: header + (index * dimension bytes for u8)
        let offset = VectorHeader::SIZE + (index as usize * self.dimension);

        if offset + self.dimension <= mmap.len() {
            let encoded = &mmap[offset..offset + self.dimension];
            Some(self.quantizer.decode(encoded))
        } else {
            None
        }
    }

    /// Asymmetric search: compare full query to quantized vectors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        let count = self.count.load(Ordering::Relaxed) as usize;
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(count);
        let mmap = self.mmap.as_ref().context("No mmap available")?;

        for i in 0..count {
            let offset = VectorHeader::SIZE + (i * self.dimension);
            if offset + self.dimension <= mmap.len() {
                let encoded = &mmap[offset..offset + self.dimension];
                let score = self.quantizer.asymmetric_distance(query, encoded);
                results.push((i as u64, score));
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;

        // Update header count
        let current_count = self.count.load(Ordering::Relaxed);
        if current_count != self.last_flushed_count {
            {
                let file = self.writer.get_mut();
                file.seek(SeekFrom::Start(0))?;

                let mut header = VectorHeader::new(self.dimension)?;
                header.count = current_count;
                header.quantization_mode = 1; // SQ8 = 1

                let header_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &header as *const VectorHeader as *const u8,
                        VectorHeader::SIZE,
                    )
                };
                file.write_all(header_bytes)?;
                file.sync_all()?;
                file.seek(SeekFrom::End(0))?;
            } // Drop mutable borrow here

            // Now we can remap
            let file_ref = self.writer.get_ref();
            self.mmap = Some(unsafe { MmapOptions::new().map_mut(file_ref)? });
            self.last_flushed_count = current_count;
        }

        Ok(())
    }
}
