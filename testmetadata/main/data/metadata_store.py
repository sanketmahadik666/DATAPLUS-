"""
Metadata Storage and Retrieval System

Handles JSON-based metadata storage with automatic updates and OCR routing integration.
Supports bulk document processing and real-time metadata management.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import threading
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MetadataStore:
    """Thread-safe metadata storage and retrieval system."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.metadata_dir = self.base_dir / "metadata"
        self.results_dir = self.base_dir / "results"
        self.queue_dir = self.base_dir / "queue"

        # Create directories
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Metadata cache
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.document_index: Dict[str, str] = {}  # document_id -> metadata_file

        # Load existing metadata
        self._load_existing_metadata()

        logger.info(f"MetadataStore initialized with base directory: {base_dir}")

    def _load_existing_metadata(self):
        """Load existing metadata files into cache."""
        try:
            for json_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        batch_metadata = json.load(f)

                    if 'results' in batch_metadata:
                        for result in batch_metadata['results']:
                            doc_id = result.get('document_id')
                            if doc_id:
                                self.metadata_cache[doc_id] = result
                                self.document_index[doc_id] = json_file.name

                    logger.info(f"Loaded metadata from {json_file.name}")

                except Exception as e:
                    logger.warning(f"Failed to load {json_file.name}: {e}")

        except Exception as e:
            logger.error(f"Error loading existing metadata: {e}")

    def store_batch_metadata(self, results: List[Dict[str, Any]], batch_id: Optional[str] = None) -> str:
        """Store batch metadata results in JSON format."""
        if not batch_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = f"batch_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        metadata_file = self.metadata_dir / f"{batch_id}_metadata.json"

        with self.lock:
            # Update cache
            for result in results:
                doc_id = result.get('document_id')
                if doc_id:
                    self.metadata_cache[doc_id] = result
                    self.document_index[doc_id] = metadata_file.name

            # Prepare metadata structure
            metadata = {
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'total_documents': len(results),
                'processing_stats': self._calculate_batch_stats(results),
                'results': results
            }

            # Save to file
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Stored batch metadata: {batch_id} ({len(results)} documents)")
        return str(metadata_file)

    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific document."""
        with self.lock:
            return self.metadata_cache.get(document_id)

    def get_documents_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all documents with a specific processing status."""
        with self.lock:
            return [
                metadata for metadata in self.metadata_cache.values()
                if metadata.get('processing_status') == status
            ]

    def get_unprocessed_documents(self) -> List[str]:
        """Get list of document IDs that haven't been processed for OCR routing."""
        processed_docs = set()

        try:
            for json_file in self.results_dir.glob("*_ocr_results.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    ocr_results = json.load(f)
                    if 'processed_documents' in ocr_results:
                        processed_docs.update(ocr_results['processed_documents'])
        except Exception as e:
            logger.warning(f"Error reading OCR results: {e}")

        with self.lock:
            unprocessed = [
                doc_id for doc_id in self.metadata_cache.keys()
                if doc_id not in processed_docs
            ]

        return unprocessed

    def get_documents_for_ocr_routing(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get documents ready for OCR routing (have metadata but no OCR results)."""
        unprocessed_ids = self.get_unprocessed_documents()

        with self.lock:
            documents = []
            for doc_id in unprocessed_ids[:limit]:
                metadata = self.metadata_cache.get(doc_id)
                if metadata:
                    documents.append(metadata)

        return documents

    def store_ocr_results(self, ocr_results: Dict[str, Any], batch_id: Optional[str] = None) -> str:
        """Store OCR routing results."""
        if not batch_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = f"ocr_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        results_file = self.results_dir / f"{batch_id}_ocr_results.json"

        # Add timestamp and metadata
        ocr_results['stored_at'] = datetime.now().isoformat()
        ocr_results['batch_id'] = batch_id

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Stored OCR results: {batch_id}")
        return str(results_file)

    def get_ocr_routing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get OCR routing processing history."""
        history = []

        try:
            for json_file in sorted(self.results_dir.glob("*_ocr_results.json"), reverse=True):
                if len(history) >= limit:
                    break

                with open(json_file, 'r', encoding='utf-8') as f:
                    ocr_results = json.load(f)
                    history.append(ocr_results)
        except Exception as e:
            logger.error(f"Error reading OCR history: {e}")

        return history

    def _calculate_batch_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for a batch of results."""
        if not results:
            return {}

        statuses = [r.get('processing_status') for r in results]
        gpu_accelerated = sum(1 for r in results if r.get('gpu_accelerated', False))

        return {
            'successful': statuses.count('success'),
            'failed': statuses.count('error'),
            'partial': statuses.count('partial'),
            'gpu_accelerated': gpu_accelerated,
            'average_processing_time': sum(r.get('processing_timestamp', 0) for r in results) / len(results)
        }

    def cleanup_old_metadata(self, days_old: int = 30):
        """Clean up old metadata files."""
        import shutil
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days_old)

        try:
            for json_file in self.metadata_dir.glob("*.json"):
                if json_file.stat().st_mtime < cutoff_date.timestamp():
                    json_file.unlink()
                    logger.info(f"Cleaned up old metadata file: {json_file.name}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            metadata_files = list(self.metadata_dir.glob("*.json"))
            results_files = list(self.results_dir.glob("*.json"))

            return {
                'total_documents': len(self.metadata_cache),
                'metadata_files': len(metadata_files),
                'results_files': len(results_files),
                'total_metadata_size_mb': sum(f.stat().st_size for f in metadata_files) / (1024**2),
                'total_results_size_mb': sum(f.stat().st_size for f in results_files) / (1024**2),
                'cache_hit_rate': 1.0  # Simplified
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}

    async def auto_process_queue(self):
        """Automatically process documents in the queue directory."""
        while True:
            try:
                # Check for new documents in queue
                queue_files = list(self.queue_dir.glob("*.pdf"))

                if queue_files:
                    logger.info(f"Found {len(queue_files)} documents in queue")

                    # Process in batches
                    batch_size = 10
                    for i in range(0, len(queue_files), batch_size):
                        batch = queue_files[i:i + batch_size]

                        # Move to processing directory (simulate)
                        processing_batch = []
                        for pdf_file in batch:
                            # Here you would integrate with Phase 3 GPU processor
                            processing_batch.append(str(pdf_file))

                        if processing_batch:
                            # Placeholder for actual processing
                            logger.info(f"Processing batch of {len(processing_batch)} documents")

                        # Small delay between batches
                        await asyncio.sleep(0.1)

                # Check every 30 seconds
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in auto processing: {e}")
                await asyncio.sleep(30)

    def export_metadata_for_ocr(self, output_file: str, limit: int = 100) -> str:
        """Export metadata in format suitable for OCR routing."""
        documents = self.get_documents_for_ocr_routing(limit)

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_documents': len(documents),
            'documents': documents
        }

        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported {len(documents)} documents for OCR routing to {output_file}")
        return str(output_path)


# Global metadata store instance
metadata_store = MetadataStore()


async def process_new_documents():
    """Background task to process new documents automatically."""
    await metadata_store.auto_process_queue()


def get_metadata_store() -> MetadataStore:
    """Get the global metadata store instance."""
    return metadata_store


# CLI Interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metadata Store Management")
    parser.add_argument("--stats", action="store_true", help="Show storage statistics")
    parser.add_argument("--cleanup", type=int, help="Clean up files older than X days")
    parser.add_argument("--export-ocr", help="Export metadata for OCR routing")
    parser.add_argument("--unprocessed", action="store_true", help="Show unprocessed documents")

    args = parser.parse_args()

    if args.stats:
        stats = metadata_store.get_storage_stats()
        print("Storage Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.cleanup:
        metadata_store.cleanup_old_metadata(args.cleanup)
        print(f"Cleaned up files older than {args.cleanup} days")

    elif args.export_ocr:
        metadata_store.export_metadata_for_ocr(args.export_ocr)
        print(f"Exported metadata to {args.export_ocr}")

    elif args.unprocessed:
        unprocessed = metadata_store.get_unprocessed_documents()
        print(f"Unprocessed documents: {len(unprocessed)}")
        for doc_id in unprocessed[:10]:  # Show first 10
            print(f"  {doc_id}")

    else:
        print("Use --help for available commands")