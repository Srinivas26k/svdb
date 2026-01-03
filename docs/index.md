# SrvDB Documentation

Welcome to the documentation for **SrvDB**.

SrvDB is a high-performance, embedded vector database designed for **offline AI**, **edge computing**, and **RAG (Retrieval-Augmented Generation)** applications. It is written in Rust for speed and safety, with zero-copy Python bindings for seamless integration into your AI stack.

## Why SrvDB?

- **Zero Dependencies**: No Docker, no external servers, no complex setup. Just `pip install srvdb`.
- **Embedded Architecture**: Runs in-process with your application. Ideal for desktop apps, CLI tools, and edge devices.
- **Modes for Every Use Case**:
    - **Flat**: 100% recall exact search.
    - **HNSW Graph**: Sub-millisecond approximate search for large datasets.
    - **Quantization (PQ/SQ8)**: 32x memory compression for running massive indexes on consumer hardware.
- **Developer Experience**: "It just works" philosophy. Intelligent defaults with deep configurability when you need it.

## Navigation

### 🚀 Getting Started
- [Installation](install.md): Get SrvDB running on your machine.
- [Quickstart](quickstart.md): Your first vector search in 5 minutes.

### 🧠 Concepts & Architecture
- [Indexing Modes](concepts/modes.md): Understand Flat, HNSW, and Quantization.
- [Architecture](concepts/modes.md): How SrvDB works under the hood.

### 📘 Guides
- [Migration Guide](guides/migration.md): Upgrading to v0.2.0.
- [Optimization & Tuning](guides/optimization_tuning.md): Tune HNSW and IVF for peak performance.
- [Benchmarking](guides/benchmarking.md): Verify performance on your hardware.

### 📊 Reports
- [GLM Benchmark Audit](reports/glm_benchmark.md): Performance on adversarial datasets.
- [Qwen Benchmark](reports/qwen_benchmark.md): Large-scale performance validation.

### ⚙️ API Reference
- [Python API](api/python.md): Comprehensive reference for the `srvdb` Python package.
