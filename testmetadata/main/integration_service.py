"""
Phase 3 Integration Service

Automatically processes documents from the queue, stores metadata, and feeds to OCR routing.
Combines Phase 3 GPU processing with automatic metadata management.
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests

# Import Phase 3 components
try:
    from phase3_gpu_accelerated.core.gpu_metadata_processor import GPUMetadataProcessor
    from phase3_gpu_accelerated.core.deep_feature_extractor import extract_optimized_features
    from data.metadata_store import get_metadata_store
except ImportError as e:
    print(f"Import error: {e}")
    GPUMetadataProcessor = None
    extract_optimized_features = None
    get_metadata_store = None

logger = logging.getLogger(__name__)


class IntegrationService:
    """Integration service for automatic document processing pipeline."""

    def __init__(self,
                 phase3_service_url: str = "http://localhost:8003",
                 ocr_service_url: str = "http://localhost:8002",
                 documents_dir: str = "data/documents",
                 queue_dir: str = "data/queue",
                 batch_size: int = 10,
                 processing_interval: int = 30):

        self.phase3_url = phase3_service_url.rstrip('/')
        self.ocr_url = ocr_service_url.rstrip('/')
        self.documents_dir = Path(documents_dir)
        self.queue_dir = Path(queue_dir)
        self.batch_size = batch_size
        self.processing_interval = processing_interval

        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.metadata_store = get_metadata_store() if get_metadata_store else None
        self.processor = GPUMetadataProcessor(max_workers=16) if GPUMetadataProcessor else None

        # Statistics
        self.stats = {
            'documents_processed': 0,
            'ocr_routing_requests': 0,
            'errors': 0,
            'start_time': time.time()
        }

        logger.info("Integration Service initialized")

    async def start_automatic_processing(self):
        """Start automatic document processing pipeline."""
        logger.info("Starting automatic document processing pipeline")

        while True:
            try:
                await self._process_document_batch()
                await asyncio.sleep(self.processing_interval)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(self.processing_interval)

    async def _process_document_batch(self):
        """Process a batch of documents through the pipeline."""
        # Get documents from queue or documents directory
        documents_to_process = self._get_pending_documents()

        if not documents_to_process:
            logger.debug("No documents to process")
            return

        # Limit batch size
        batch = documents_to_process[:self.batch_size]
        logger.info(f"Processing batch of {len(batch)} documents")

        try:
            # Step 1: Process with Phase 3 GPU service
            phase3_results = await self._process_with_phase3(batch)

            if phase3_results and phase3_results.get('results'):
                self.stats['documents_processed'] += len(phase3_results['results'])

                # Step 2: Store metadata automatically
                if self.metadata_store:
                    batch_id = f"auto_batch_{int(time.time())}"
                    self.metadata_store.store_batch_metadata(phase3_results['results'], batch_id)

                # Step 3: Send to OCR routing service
                await self._send_to_ocr_routing(phase3_results['results'])

                # Step 4: Move processed documents (optional)
                self._mark_documents_processed(batch)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.stats['errors'] += 1

    def _get_pending_documents(self) -> List[Path]:
        """Get documents that need processing."""
        pending_docs = []

        # Check queue directory first
        if self.queue_dir.exists():
            pending_docs.extend(list(self.queue_dir.glob("*.pdf")))

        # Check documents directory for unprocessed files
        if self.documents_dir.exists():
            doc_files = list(self.documents_dir.glob("*.pdf"))

            # Filter out already processed documents (if metadata exists)
            if self.metadata_store:
                processed_ids = set(self.metadata_store.metadata_cache.keys())
                pending_docs.extend([
                    pdf for pdf in doc_files
                    if pdf.stem not in processed_ids
                ])
            else:
                pending_docs.extend(doc_files)

        return list(set(pending_docs))  # Remove duplicates

    async def _process_with_phase3(self, documents: List[Path]) -> Optional[Dict[str, Any]]:
        """Process documents using Phase 3 GPU service."""
        try:
            payload = {
                "pdf_paths": [str(doc) for doc in documents],
                "max_workers": 16,
                "extract_deep_features": True,
                "batch_size": min(self.batch_size, len(documents))
            }

            response = requests.post(
                f"{self.phase3_url}/process",
                json=payload,
                timeout=300  # 5 minutes timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Phase 3 processing failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error calling Phase 3 service: {e}")
            return None

    async def _send_to_ocr_routing(self, metadata_results: List[Dict[str, Any]]):
        """Send processed documents to OCR routing service."""
        try:
            # Get OCR routing recommendations
            ocr_payload = {
                "documents": metadata_results,
                "routing_strategy": "auto",
                "batch_size": len(metadata_results)
            }

            response = requests.post(
                f"{self.ocr_url}/route-batch",
                json=ocr_payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"OCR routing completed: {result.get('total_documents', 0)} documents")
                self.stats['ocr_routing_requests'] += 1

                # Store OCR results if metadata store available
                if self.metadata_store:
                    batch_id = f"ocr_batch_{int(time.time())}"
                    self.metadata_store.store_ocr_results(result, batch_id)

            else:
                logger.warning(f"OCR routing failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error sending to OCR routing: {e}")

    def _mark_documents_processed(self, documents: List[Path]):
        """Mark documents as processed (optional cleanup)."""
        # Could move to processed directory or add to database
        # For now, just log
        for doc in documents:
            logger.debug(f"Marked as processed: {doc.name}")

    async def manual_process_directory(self, directory: str, recursive: bool = False):
        """Manually process all PDFs in a directory."""
        target_dir = Path(directory)

        if not target_dir.exists():
            logger.error(f"Directory does not exist: {directory}")
            return

        if recursive:
            pdf_files = list(target_dir.rglob("*.pdf"))
        else:
            pdf_files = list(target_dir.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        # Process in batches
        batch_size = self.batch_size
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]
            logger.info(f"Processing manual batch {i//batch_size + 1}: {len(batch)} files")

            await self._process_document_batch()

            # Small delay between batches
            await asyncio.sleep(1)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        uptime = time.time() - self.stats['start_time']

        return {
            'uptime_seconds': round(uptime, 2),
            'documents_processed': self.stats['documents_processed'],
            'ocr_routing_requests': self.stats['ocr_routing_requests'],
            'errors': self.stats['errors'],
            'processing_rate_per_hour': round(self.stats['documents_processed'] / (uptime / 3600), 2) if uptime > 0 else 0,
            'queue_size': len(self._get_pending_documents()),
            'metadata_store_available': self.metadata_store is not None,
            'phase3_service_available': self._check_service_available(self.phase3_url),
            'ocr_service_available': self._check_service_available(self.ocr_url)
        }

    def _check_service_available(self, url: str) -> bool:
        """Check if a service is available."""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def cleanup(self):
        """Cleanup resources."""
        if self.processor:
            self.processor.cleanup()
        logger.info("Integration Service cleaned up")


# CLI Interface
async def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 Integration Service")
    parser.add_argument("--phase3-url", default="http://localhost:8003", help="Phase 3 service URL")
    parser.add_argument("--ocr-url", default="http://localhost:8002", help="OCR routing service URL")
    parser.add_argument("--documents-dir", default="data/documents", help="Documents directory")
    parser.add_argument("--queue-dir", default="data/queue", help="Queue directory")
    parser.add_argument("--batch-size", type=int, default=10, help="Processing batch size")
    parser.add_argument("--interval", type=int, default=30, help="Processing interval (seconds)")
    parser.add_argument("--manual-dir", help="Manually process directory")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories")
    parser.add_argument("--stats", action="store_true", help="Show processing statistics")

    args = parser.parse_args()

    service = IntegrationService(
        phase3_service_url=args.phase3_url,
        ocr_service_url=args.ocr_url,
        documents_dir=args.documents_dir,
        queue_dir=args.queue_dir,
        batch_size=args.batch_size,
        processing_interval=args.interval
    )

    try:
        if args.stats:
            stats = service.get_processing_stats()
            print("Integration Service Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        elif args.manual_dir:
            await service.manual_process_directory(args.manual_dir, args.recursive)
            print(f"Manual processing completed for {args.manual_dir}")

        else:
            print("Starting automatic processing (Ctrl+C to stop)...")
            await service.start_automatic_processing()

    except KeyboardInterrupt:
        print("\nStopping integration service...")
    finally:
        service.cleanup()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())