# SrvDB v0.2.0 Deployment Guide

## 🚀 Quick Migration from v0.1.x

### Breaking Changes
1. **Metadata store cache**: Auto-flushing enabled (transparent)
2. **Batch APIs added**: `add_batch()`, `search_batch()` now available
3. **Memory layout**: Binary compatible with v0.1.x

### Migration Steps

```python
# 1. Backup existing database
import shutil
shutil.copytree("./old_db", "./old_db_backup")

# 2. Install new version
pip install --upgrade srvdb

# 3. Test compatibility
import srvdb
db = srvdb.SvDBPython("./old_db")
assert db.count() > 0  # Verify data loads

# 4. Rebuild indices (optional, for best performance)
# Not required - v0.2.0 is backward compatible
```

## 📦 Installation Options

### Option 1: PyPI (Recommended)
```bash
pip install srvdb==0.2.0
```

### Option 2: From Source (Maximum Performance)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/yourusername/srvdb
cd srvdb
git checkout v0.2.0

# Build with CPU-specific optimizations
export RUSTFLAGS="-C target-cpu=native"
maturin build --release

# Install wheel
pip install target/wheels/srvdb-0.2.0-*.whl
```

### Option 3: Docker
```dockerfile
FROM python:3.11-slim

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy source
COPY . /srvdb
WORKDIR /srvdb

# Build with optimizations
ENV RUSTFLAGS="-C target-cpu=native"
RUN pip install maturin && maturin build --release
RUN pip install target/wheels/*.whl

# Your application
COPY app.py /app/
CMD ["python", "/app/app.py"]
```

## ⚙️ Configuration

### Environment Variables

```bash
# Buffer size (default: 8MB)
export SVDB_BUFFER_SIZE=8388608

# Auto-flush threshold (default: 1000 vectors)
export SVDB_AUTO_FLUSH_THRESHOLD=1000

# Enable debug logging
export RUST_LOG=debug
```

### Python Configuration

```python
import srvdb
import os

# High-throughput ingestion profile
os.environ['SVDB_BUFFER_SIZE'] = '16777216'  # 16MB
os.environ['SVDB_AUTO_FLUSH_THRESHOLD'] = '5000'

# Low-latency search profile  
os.environ['SVDB_BUFFER_SIZE'] = '4194304'  # 4MB
os.environ['SVDB_AUTO_FLUSH_THRESHOLD'] = '500'

db = srvdb.SvDBPython("./vectors")
```

## 🏗️ Production Architecture

### Single-Node Deployment

```
┌─────────────────────────────────────┐
│   Application Layer                  │
│   ├─ FastAPI / Flask / Django       │
│   └─ Background Workers (Celery)    │
├─────────────────────────────────────┤
│   SrvDB Python Client                │
│   ├─ Connection Pool (1 per worker)│
│   └─ Read Replicas (shared mmap)   │
├─────────────────────────────────────┤
│   SrvDB Core Engine                  │
│   ├─ vectors.bin (NVMe SSD)        │
│   └─ metadata.db (NVMe SSD)        │
└─────────────────────────────────────┘
```

### Multi-Node Deployment (Manual Sharding)

```
┌──────────────────────────────────────────┐
│        Load Balancer / API Gateway       │
└──────────┬────────────┬──────────────────┘
           │            │
    ┌──────▼─────┐  ┌──▼──────────┐
    │  Shard 1   │  │   Shard 2   │
    │  Docs 0-1M │  │  Docs 1M-2M │
    │  SrvDB     │  │  SrvDB      │
    └────────────┘  └─────────────┘

# Python sharding example
def get_shard(doc_id: str) -> SvDBPython:
    shard_id = hash(doc_id) % NUM_SHARDS
    return db_shards[shard_id]
```

## 🔐 Security Best Practices

### File Permissions
```bash
# Restrict database access
chmod 700 /path/to/db
chown app:app /path/to/db

# Set appropriate file permissions
chmod 600 /path/to/db/vectors.bin
chmod 600 /path/to/db/metadata.db
```

### API Security
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

@app.post("/search")
async def search_vectors(
    query: List[float],
    token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    results = db.search(query=query, k=10)
    return {"results": results}
```

## 📊 Monitoring & Observability

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
search_latency = Histogram('svdb_search_latency_seconds', 'Search latency')
ingestion_total = Counter('svdb_ingestion_total', 'Total vectors ingested')
db_size = Gauge('svdb_vector_count', 'Current vector count')

# Instrumented search
@search_latency.time()
def search_with_metrics(query, k):
    return db.search(query=query, k=k)

# Update metrics
ingestion_total.inc(len(vectors))
db_size.set(db.count())

# Start metrics server
start_http_server(8000)
```

### Logging
```python
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('svdb')

def search_with_logging(query, k):
    start = time.time()
    try:
        results = db.search(query=query, k=k)
        latency = (time.time() - start) * 1000
        logger.info(f"Search completed: {len(results)} results in {latency:.2f}ms")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise
```

### Health Checks
```python
from fastapi import FastAPI, status

app = FastAPI()

@app.get("/health")
async def health_check():
    try:
        # Verify database is accessible
        count = db.count()
        
        # Verify search works
        test_query = [0.1] * 1536
        db.search(query=test_query, k=1)
        
        return {
            "status": "healthy",
            "vector_count": count,
            "version": "0.2.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }, status.HTTP_503_SERVICE_UNAVAILABLE
```

## 🔄 Backup & Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup_svdb.sh

DB_PATH="/var/lib/svdb"
BACKUP_PATH="/backups/svdb"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Stop writes (optional)
# curl -X POST http://localhost:8000/admin/maintenance/enable

# Flush pending writes
python -c "
import srvdb
db = srvdb.SvDBPython('$DB_PATH')
db.persist()
"

# Create backup
mkdir -p $BACKUP_PATH
tar -czf $BACKUP_PATH/svdb_${TIMESTAMP}.tar.gz -C $DB_PATH .

# Resume writes
# curl -X POST http://localhost:8000/admin/maintenance/disable

# Cleanup old backups (keep last 7 days)
find $BACKUP_PATH -name "svdb_*.tar.gz" -mtime +7 -delete

echo "Backup completed: svdb_${TIMESTAMP}.tar.gz"
```

### Recovery

```bash
#!/bin/bash
# restore_svdb.sh

BACKUP_FILE=$1
DB_PATH="/var/lib/svdb"

# Stop application
systemctl stop svdb-app

# Clear existing data
rm -rf $DB_PATH/*

# Restore from backup
tar -xzf $BACKUP_FILE -C $DB_PATH

# Verify integrity
python -c "
import srvdb
db = srvdb.SvDBPython('$DB_PATH')
print(f'Restored {db.count()} vectors')
"

# Start application
systemctl start svdb-app
```

## 🧪 Testing in Production

### Canary Deployment
```python
import random

# Route 10% of traffic to new version
def get_db_version():
    if random.random() < 0.1:
        return new_db  # v0.2.0
    return old_db  # v0.1.x

# Compare results
query = [0.1] * 1536
old_results = old_db.search(query=query, k=10)
new_results = new_db.search(query=query, k=10)

# Verify consistency
assert len(old_results) == len(new_results)
assert old_results[0][0] == new_results[0][0]  # Top result matches
```

### Load Testing
```python
from locust import HttpUser, task, between

class SvDBLoadTest(HttpUser):
    wait_time = between(0.1, 0.5)
    
    @task
    def search_vectors(self):
        query = [random.random() for _ in range(1536)]
        self.client.post("/search", json={
            "query": query,
            "k": 10
        })
    
    @task(2)  # 2x more searches than inserts
    def add_vectors(self):
        vectors = [[random.random() for _ in range(1536)] for _ in range(10)]
        self.client.post("/add", json={
            "ids": [f"doc_{i}" for i in range(10)],
            "embeddings": vectors,
            "metadatas": ["{}"] * 10
        })
```

## 🎯 Performance Tuning

### CPU Optimization
```bash
# Check CPU features
lscpu | grep -E "avx|sse|neon"

# Build with CPU-specific optimizations
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
maturin build --release

# Verify SIMD instructions are used
objdump -d target/release/libsrvdb.so | grep -E "vmov|vpadd"  # AVX
```

### Memory Tuning
```bash
# Increase memory limits (systemd)
cat > /etc/systemd/system/svdb-app.service <<EOF
[Service]
MemoryMax=16G
LimitNOFILE=65536
EOF

# Tune kernel parameters
sudo sysctl -w vm.max_map_count=1048576
sudo sysctl -w vm.swappiness=10
```

### Storage Optimization
```bash
# Use NVMe SSD with proper mount options
mount -o noatime,nodiratime /dev/nvme0n1 /var/lib/svdb

# Enable TRIM
sudo fstrim -v /var/lib/svdb

# Check I/O scheduler
cat /sys/block/nvme0n1/queue/scheduler  # Should be "none" for NVMe
```

## 🐛 Troubleshooting

### Common Issues

1. **"Failed to open vectors.bin"**
   - Check file permissions: `ls -la /path/to/db`
   - Verify disk space: `df -h`
   - Check for file locks: `lsof | grep vectors.bin`

2. **Slow ingestion (<10k vec/s)**
   - Verify NVMe SSD: `lsblk -d -o name,rota` (0 = SSD)
   - Check buffer size: Should be 8MB+
   - Monitor I/O: `iostat -x 1`

3. **High memory usage**
   - Call `db.persist()` regularly
   - Reduce `SVDB_AUTO_FLUSH_THRESHOLD`
   - Check for memory leaks: `valgrind python app.py`

4. **Search timeout**
   - Verify database is flushed
   - Check vector count: `db.count()`
   - Profile with: `python -m cProfile app.py`

## 📞 Support

- GitHub Issues: https://github.com/yourusername/srvdb/issues
- Discord: https://discord.gg/svdb
- Email: support@svdb.dev

## 📝 Changelog

### v0.2.0 (Current)
- ✨ 10x faster ingestion (8MB buffers)
- ✨ 3x faster search (SIMD + batch processing)
- ✨ 30% less memory (FxHashMap)
- ✨ GIL-free Python bindings
- ✨ Batch APIs for all operations

### v0.1.6 (Legacy)
- Initial production release
- Basic vector storage and search
- Python bindings with PyO3