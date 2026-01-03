# Installation

SrvDB is designed to be a drop-in component for your AI application. It supports Python 3.8+ on Linux, macOS, and Windows.

## 📦 Install via Pip

The easiest way to get started is via PyPI:
```bash
pip install srvdb
```

This installs the pre-compiled binary wheel for your platform. SrvDB includes all necessary native code, so you don't need to install Rust or other system dependencies manually.

## 🛠️ Build from Source

If you want to contribute to SrvDB or need to compile for a specific architecture (e.g., optimizing for native CPU instructions), you can build from source.

**Prerequisites:**
- **Rust Toolchain**: [Install Rust](https://rustup.rs/) (1.70+)
- **Python**: 3.8+ with `pip` and `venv`

### Steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Srinivas26k/srvdb
   cd srvdb
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install build tools:**
   We use `maturin` to build the Python extension from Rust code.
   ```bash
   pip install maturin
   ```

4. **Build and Install:**
   ```bash
   # Build release version and install into current venv
   maturin develop --release
   ```

   *Note: Using `--release` is critical. Debug builds are significantly slower (up to 100x).*

## 🖥️ System Requirements

- **OS**: Linux (glibc 2.17+), macOS (10.14+), Windows (x64)
- **CPU**: x86_64 or ARM64 (Apple Silicon, AWS Graviton)
- **SIMD**: SrvDB automatically detects and uses AVX-512, AVX2, or NEON instructions for acceleration. Older CPUs without SIMD support will use a fallback implementation but will be slower.
