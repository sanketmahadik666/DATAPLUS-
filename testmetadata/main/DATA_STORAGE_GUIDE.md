# Phase 3: Data Storage and Retrieval Guide

## Overview

Phase 3 includes a comprehensive data storage system that automatically manages document metadata, processing results, and OCR routing integration. The system uses JSON-based storage for easy retrieval and integrates seamlessly with the GPU-accelerated processing pipeline.

## Directory Structure

```
data/
├── documents/          # Bulk document storage
│   └── *.pdf          # PDF files for processing
├── metadata/          # JSON metadata storage
│   └── *_metadata.json # Batch processing results
├── results/           # OCR routing results
│   └── *_ocr_results.json # OCR processing outcomes
└── queue/             # Processing queue
    └── *.pdf         # Documents awaiting processing
```

## Storage System Features

### 1. Metadata Store (`data/metadata_store.py`)

The metadata store provides thread-safe JSON-based storage with:

- **Automatic Indexing**: Document ID to metadata file mapping
- **Batch Storage**: Efficient bulk metadata storage
- **OCR Integration**: Automatic routing to OCR services
- **Statistics**: Storage and processing analytics

### 2. Integration Service (`integration_service.py`)

Automatically processes documents through the complete pipeline:

- **Queue Monitoring**: Continuous processing from queue directory
- **Phase 3 Integration**: GPU processing with metadata storage
- **OCR Routing**: Automatic feeding to OCR services
- **Statistics**: Real-time processing metrics

## Usage Examples

### 1. Store Documents for Processing

```bash
# Copy documents to processing directory
cp *.pdf data/documents/

# Or add to queue for immediate processing
cp *.pdf data/queue/
```

### 2. Check Processing Status

```python
from data.metadata_store import get_metadata_store

store = get_metadata_store()

# Get storage statistics
stats = store.get_storage_stats()
print(f"Total documents: {stats['total_documents']}")

# Get unprocessed documents
unprocessed = store.get_unprocessed_documents()
print(f"Documents waiting for OCR: {len(unprocessed)}")
```

### 3. Start Automatic Processing

```bash
# Start integration service
python integration_service.py

# Process specific directory
python integration_service.py --manual-dir /path/to/pdfs --recursive
```

### 4. API Endpoints for Metadata

```bash
# Get unprocessed documents
curl http://localhost:8003/documents/unprocessed

# Get specific document metadata
curl http://localhost:8003/documents/document_id_123

# Store batch metadata manually
curl -X POST http://localhost:8003/documents/batch/store \
  -H "Content-Type: application/json" \
  -d '{"pdf_paths": ["/path/to/doc.pdf"], "batch_size": 1}'
```

## Data Flow Architecture

```
Documents → Phase 3 GPU Processing → Metadata Storage → OCR Routing → Results Storage
    ↓              ↓                        ↓              ↓              ↓
data/documents/ → GPU Service → data/metadata/ → Integration → data/results/
```

### Automatic Processing Flow

1. **Document Discovery**: Files added to `data/documents/` or `data/queue/`
2. **GPU Processing**: Phase 3 service extracts metadata and features
3. **Metadata Storage**: Results stored as JSON in `data/metadata/`
4. **OCR Routing**: Documents sent to OCR service with optimal engine selection
5. **Results Storage**: OCR outcomes stored in `data/results/`

## JSON Metadata Format

### Processing Results (`*_metadata.json`)

```json
{
  "batch_id": "batch_20251025_143052_abc123",
  "timestamp": "2025-10-25T14:30:52.123456",
  "total_documents": 10,
  "processing_stats": {
    "successful": 10,
    "failed": 0,
    "gpu_accelerated": 8
  },
  "results": [
    {
      "document_id": "invoice_Aaron Bergman_36258",
      "file_path": "data/documents/invoice_Aaron Bergman_36258.pdf",
      "file_size": 245760,
      "processing_status": "success",
      "gpu_accelerated": true,
      "image_dimensions": [1654, 2339],
      "mean_intensity": 0.78,
      "deep_features_extracted": true,
      "feature_dimensions": {
        "vision": 2048,
        "text": 768,
        "structural": 64
      },
      "processing_timestamp": 1698253852.123
    }
  ]
}
```

### OCR Results (`*_ocr_results.json`)

```json
{
  "batch_id": "ocr_batch_20251025_143500",
  "stored_at": "2025-10-25T14:35:00.123456",
  "total_documents": 10,
  "documents_processed": 10,
  "routing_results": [
    {
      "document_id": "invoice_Aaron Bergman_36258",
      "recommended_engine": "paddle_malayalam",
      "confidence_score": 0.89,
      "estimated_accuracy": 0.94,
      "estimated_latency": 0.5,
      "estimated_cost": 3.0
    }
  ]
}
```

## CLI Tools

### Metadata Store CLI

```bash
# Show storage statistics
python data/metadata_store.py --stats

# Export for OCR routing
python data/metadata_store.py --export-ocr metadata_for_ocr.json

# Show unprocessed documents
python data/metadata_store.py --unprocessed

# Cleanup old files (30+ days)
python data/metadata_store.py --cleanup 30
```

### Integration Service CLI

```bash
# Start automatic processing
python integration_service.py

# Process specific directory
python integration_service.py --manual-dir /path/to/pdfs

# Show processing statistics
python integration_service.py --stats

# Custom service URLs
python integration_service.py --phase3-url http://gpu-server:8003 --ocr-url http://ocr-server:8002
```

## API Integration

### FastAPI Endpoints

```python
from fastapi import FastAPI
from data.metadata_store import get_metadata_store

app = FastAPI()
store = get_metadata_store()

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    return store.get_document_metadata(doc_id)

@app.get("/queue/ocr")
async def get_ocr_queue():
    documents = store.get_documents_for_ocr_routing(50)
    return {"queue": documents, "count": len(documents)}
```

## Monitoring and Maintenance

### Storage Monitoring

```python
# Check storage health
stats = store.get_storage_stats()

if stats['total_metadata_size_mb'] > 1000:  # 1GB limit
    store.cleanup_old_metadata(days_old=90)

# Monitor processing backlog
unprocessed = len(store.get_unprocessed_documents())
if unprocessed > 100:
    logger.warning(f"Large processing backlog: {unprocessed} documents")
```

### Automatic Cleanup

```python
# Configure automatic cleanup
import schedule

def cleanup_task():
    store.cleanup_old_metadata(days_old=30)
    logger.info("Cleaned up old metadata files")

schedule.every().day.at("02:00").do(cleanup_task)
```

## Performance Considerations

### Storage Optimization

- **Batch Size**: Process in batches of 10-50 documents for optimal performance
- **File Compression**: JSON files are automatically compressed for storage
- **Indexing**: Automatic document ID indexing for fast lookups
- **Caching**: In-memory cache for frequently accessed metadata

### Scalability Features

- **Thread Safety**: All operations are thread-safe for concurrent access
- **Horizontal Scaling**: Multiple instances can share the same storage
- **Load Balancing**: Automatic distribution across processing nodes
- **Failover**: Graceful handling of service interruptions

## Troubleshooting

### Common Issues

1. **Storage Full**: Check disk space and run cleanup
2. **Slow Queries**: Use batch operations instead of individual lookups
3. **Memory Issues**: Reduce cache size or increase system memory
4. **File Locks**: Ensure proper file permissions and avoid concurrent writes

### Debug Commands

```bash
# Check file permissions
ls -la data/

# Validate JSON files
python -c "import json; json.load(open('data/metadata/batch_123_metadata.json'))"

# Check processing logs
tail -f integration_service.log
```

## Integration with Existing Systems

### Phase 2 Compatibility

The storage system is designed to work alongside Phase 2:

```python
# Import Phase 2 results
from ocr_routing_pipeline.data.ml_ocr_routing_dataset import routing_dataset

# Store in new format
store.store_batch_metadata(routing_dataset, "phase2_import")
```

### Database Migration

For production deployments, consider migrating to a database:

```python
# Export to database format
documents = store.get_documents_for_ocr_routing(limit=1000)

# Insert into database
for doc in documents:
    db.insert_document(doc)
```

## Security Considerations

### File Permissions

```bash
# Secure data directories
chmod 750 data/
chmod 640 data/metadata/*.json
chown appuser:appgroup data/
```

### Access Control

```python
# Implement access control
def check_permissions(user, action, resource):
    if action == "read" and resource.startswith("metadata/"):
        return user.has_role("analyst")
    return False
```

## Summary

The Phase 3 data storage system provides:

- ✅ **Automatic Processing**: Queue-based document processing
- ✅ **JSON Storage**: Structured metadata in JSON format
- ✅ **OCR Integration**: Seamless routing to OCR services
- ✅ **Scalability**: Thread-safe operations for high throughput
- ✅ **Monitoring**: Comprehensive statistics and health checks
- ✅ **CLI Tools**: Easy management and debugging
- ✅ **API Integration**: RESTful endpoints for external access

The system automatically handles the complete pipeline from document ingestion to OCR routing, storing all metadata in JSON format for easy retrieval and analysis.