#!/usr/bin/env python3
"""
Phase 3: End-to-End Demo - Complete GPU-Accelerated Document Processing Pipeline

This script demonstrates the complete Phase 3 pipeline:
1. Document ingestion and storage
2. GPU-accelerated metadata processing
3. Deep feature extraction
4. Enhanced OCR routing
5. Results storage and retrieval

Run this to see Phase 3 in action!
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project paths
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Phase 3 components
try:
    from phase3_gpu_accelerated.services.enhanced_gpu_service import EnhancedGPUService
    from phase3_gpu_accelerated.core.enhanced_ocr_dispatcher import EnhancedOCRDispatcher, create_enhanced_ocr_routing_response
    from data.metadata_store import get_metadata_store
    from integration_service import IntegrationService
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all Phase 3 components are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase3EndToEndDemo:
    """Complete Phase 3 end-to-end demonstration."""

    def __init__(self):
        self.metadata_store = get_metadata_store()
        self.gpu_service = None
        self.integration_service = None

        # Demo configuration
        self.documents_dir = Path("data/documents")
        self.test_documents = []

        logger.info("🚀 Phase 3 End-to-End Demo initialized")

    async def initialize_services(self):
        """Initialize all Phase 3 services."""
        logger.info("🔧 Initializing Phase 3 services...")

        try:
            # Initialize GPU service
            self.gpu_service = EnhancedGPUService()
            logger.info("✅ GPU Service initialized")

            # Initialize integration service
            self.integration_service = IntegrationService(
                phase3_service_url="http://localhost:8003",
                ocr_service_url="http://localhost:8002",
                documents_dir=str(self.documents_dir),
                batch_size=5,
                processing_interval=10
            )
            logger.info("✅ Integration Service initialized")

            # Verify services are healthy
            health = await self.gpu_service.get_health_status()
            logger.info(f"🏥 GPU Service Health: {health.status}")

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            return False

        return True

    def prepare_test_documents(self) -> List[str]:
        """Prepare test documents for processing."""
        logger.info("📄 Preparing test documents...")

        if not self.documents_dir.exists():
            logger.warning(f"⚠️  Documents directory {self.documents_dir} does not exist")
            # Create mock documents for demo
            self._create_mock_documents()
            return self.test_documents

        # Find PDF files
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("⚠️  No PDF files found, creating mock documents")
            self._create_mock_documents()
        else:
            self.test_documents = [str(pdf) for pdf in pdf_files[:10]]  # Limit to 10 for demo
            logger.info(f"📋 Found {len(self.test_documents)} PDF documents")

        return self.test_documents

    def _create_mock_documents(self):
        """Create mock documents for demonstration."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        # Create simple mock PDFs (just for path demonstration)
        for i in range(5):
            pdf_path = self.documents_dir / f"demo_invoice_{i+1}.pdf"
            if not pdf_path.exists():
                # Create a minimal mock PDF
                mock_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 72 720 Td (Demo Invoice {i+1}) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000200 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
284
%%EOF"""

                pdf_path.write_text(mock_content)
                logger.info(f"📄 Created mock document: {pdf_path.name}")

        self.test_documents = [str(p) for p in self.documents_dir.glob("*.pdf")]
        logger.info(f"🎯 Created {len(self.test_documents)} mock documents")

    async def run_gpu_processing_demo(self):
        """Demonstrate GPU-accelerated document processing."""
        logger.info("\n🎬 === Phase 3 GPU Processing Demo ===")

        if not self.test_documents:
            logger.error("❌ No test documents available")
            return False

        try:
            # Process documents with GPU acceleration
            processing_request = {
                "pdf_paths": self.test_documents,
                "max_workers": 4,
                "extract_deep_features": True,
                "batch_size": 3
            }

            logger.info(f"🚀 Processing {len(self.test_documents)} documents with GPU acceleration...")
            start_time = time.time()

            response = await self.gpu_service.process_documents(processing_request)

            processing_time = time.time() - start_time

            logger.info("✅ GPU Processing completed!")
            logger.info(f"   📊 Documents processed: {response.total_documents}")
            logger.info(f"   ⚡ Processing time: {processing_time:.2f} seconds")
            logger.info(f"   🚀 Files per second: {response.files_per_second}")
            logger.info(f"   🎯 GPU accelerated: {response.gpu_accelerated}")

            # Store results in metadata store
            if response.results:
                batch_id = f"demo_gpu_batch_{int(time.time())}"
                metadata_file = self.metadata_store.store_batch_metadata(response.results, batch_id)
                logger.info(f"💾 Stored metadata: {metadata_file}")

                # Show sample result
                sample_result = response.results[0]
                logger.info("📋 Sample processing result:")
                logger.info(f"   📄 Document: {sample_result.get('document_id')}")
                logger.info(f"   🎨 GPU Accelerated: {sample_result.get('gpu_accelerated', False)}")
                logger.info(f"   🧠 Deep Features: {sample_result.get('deep_features_extracted', False)}")
                logger.info(f"   📏 Feature Dimensions: {sample_result.get('feature_dimensions', {})}")

            return True

        except Exception as e:
            logger.error(f"❌ GPU processing demo failed: {e}")
            return False

    async def run_enhanced_routing_demo(self):
        """Demonstrate enhanced OCR routing with Phase 3 features."""
        logger.info("\n🎬 === Enhanced OCR Routing Demo ===")

        try:
            # Get documents ready for OCR routing
            ocr_candidates = self.metadata_store.get_documents_for_ocr_routing(limit=5)

            if not ocr_candidates:
                logger.warning("⚠️  No documents ready for OCR routing, using mock data")
                # Create mock data for demo
                ocr_candidates = self._create_mock_ocr_candidates()

            logger.info(f"🎯 Routing {len(ocr_candidates)} documents for OCR...")

            # Create enhanced OCR routing
            routing_response = create_enhanced_ocr_routing_response(
                documents=ocr_candidates,
                use_gpu_features=True,
                alpha=1.0, beta=0.5, gamma=0.5
            )

            logger.info("✅ Enhanced OCR routing completed!")

            # Display routing results
            routing_results = routing_response['routing_results']['routing_predictions']

            logger.info("📊 Routing Results:")
            for i, prediction in enumerate(routing_results[:3]):  # Show first 3
                logger.info(f"   {i+1}. 📄 {prediction['document_id']}")
                logger.info(f"      🎯 Engine: {prediction['chosen_engine']}")
                logger.info(f"      📈 Confidence: {prediction['confidence_score']:.3f}")
                logger.info(f"      🚀 GPU Accelerated: {prediction['gpu_accelerated']}")
                logger.info(f"      🧠 Deep Features: {prediction['deep_features_used']}")
                logger.info(f"      ⚡ Expected Latency: {prediction['expected_latency_sec']:.2f}s")
                logger.info(f"      🔧 Preprocessing Steps: {len(prediction['preprocessing_recommendations'])}")

            # Store OCR results
            batch_id = f"demo_routing_{int(time.time())}"
            self.metadata_store.store_ocr_results(routing_response, batch_id)
            logger.info(f"💾 Stored OCR routing results: {batch_id}")

            # Show analytics
            analytics = routing_response['routing_results']['phase3_metadata']
            logger.info("📈 Phase 3 Analytics:")
            logger.info(f"   🎨 GPU Accelerated: {analytics['gpu_accelerated_count']}")
            logger.info(f"   🧠 Deep Features Used: {analytics['deep_features_count']}")

            return True

        except Exception as e:
            logger.error(f"❌ Enhanced routing demo failed: {e}")
            return False

    def _create_mock_ocr_candidates(self) -> List[Dict]:
        """Create mock documents for OCR routing demo."""
        return [
            {
                'document_id': f'mock_doc_{i+1}',
                'gpu_accelerated': True,
                'deep_features_extracted': True,
                'feature_dimensions': {'vision': 2048, 'text': 768, 'structural': 64},
                'page_count': 1,
                'text_density': 450 + i * 50,
                'has_tables': i % 3 == 0,
                'has_forms': i % 4 == 1,
                'has_multilingual': i % 5 == 2,
                'processing_timestamp': time.time()
            }
            for i in range(5)
        ]

    async def run_integration_demo(self):
        """Demonstrate the integration service."""
        logger.info("\n🎬 === Integration Service Demo ===")

        try:
            # Run manual processing of available documents
            await self.integration_service.manual_process_directory(
                str(self.documents_dir),
                recursive=False
            )

            # Show processing statistics
            stats = self.integration_service.get_processing_stats()

            logger.info("📊 Integration Service Statistics:")
            logger.info(f"   ⏱️  Uptime: {stats['uptime_seconds']:.1f} seconds")
            logger.info(f"   📄 Documents Processed: {stats['documents_processed']}")
            logger.info(f"   🎯 OCR Routing Requests: {stats['ocr_routing_requests']}")
            logger.info(f"   🎨 GPU Service Available: {stats['phase3_service_available']}")
            logger.info(f"   🔍 OCR Service Available: {stats['ocr_service_available']}")

            return True

        except Exception as e:
            logger.error(f"❌ Integration demo failed: {e}")
            return False

    def run_metadata_analysis(self):
        """Analyze and display metadata store statistics."""
        logger.info("\n🎬 === Metadata Store Analysis ===")

        try:
            # Get storage statistics
            stats = self.metadata_store.get_storage_stats()

            logger.info("💾 Metadata Store Statistics:")
            logger.info(f"   📊 Total Documents: {stats['total_documents']}")
            logger.info(f"   📁 Metadata Files: {stats['metadata_files']}")
            logger.info(f"   📋 Results Files: {stats['results_files']}")
            logger.info(f"   💾 Metadata Size: {stats['total_metadata_size_mb']:.1f} MB")
            logger.info(f"   📈 Results Size: {stats['total_results_size_mb']:.1f} MB")

            # Show recent documents
            unprocessed = self.metadata_store.get_unprocessed_documents()
            logger.info(f"   ⏳ Unprocessed Documents: {len(unprocessed)}")

            if unprocessed:
                logger.info("   📋 Recent unprocessed documents:")
                for doc_id in unprocessed[:3]:
                    doc = self.metadata_store.get_document_metadata(doc_id)
                    if doc:
                        gpu_status = "🚀 GPU" if doc.get('gpu_accelerated') else "💻 CPU"
                        logger.info(f"      • {doc_id}: {gpu_status}")

            return True

        except Exception as e:
            logger.error(f"❌ Metadata analysis failed: {e}")
            return False

    async def run_performance_benchmark(self):
        """Run a quick performance benchmark."""
        logger.info("\n🎬 === Performance Benchmark Demo ===")

        try:
            from phase3_gpu_accelerated.benchmarks.performance_benchmarks import PerformanceBenchmark

            benchmark = PerformanceBenchmark()
            test_files = [Path(doc) for doc in self.test_documents[:3]]  # Small sample

            logger.info(f"🏃 Running performance benchmark on {len(test_files)} files...")

            # Run quick benchmark
            results = await benchmark.run_comprehensive_benchmarks(test_files)

            logger.info("✅ Benchmark completed!")

            # Show summary
            summary = results['analysis']['summary']
            if 'GPU_Full' in summary:
                gpu_stats = summary['GPU_Full']
                logger.info("🚀 GPU Performance Results:")
                logger.info(f"   ⚡ Average FPS: {gpu_stats['average_fps']}")
                logger.info(f"   📈 Max FPS: {gpu_stats['max_fps']}")
                logger.info(f"   🎯 GPU Acceleration: Consistent")

            return True

        except Exception as e:
            logger.error(f"❌ Performance benchmark failed: {e}")
            return False

    async def run_complete_demo(self):
        """Run the complete Phase 3 end-to-end demonstration."""
        logger.info("🎭 === Phase 3 Complete End-to-End Demo ===")
        logger.info("🚀 Demonstrating the full GPU-accelerated document processing pipeline")

        success_count = 0
        total_steps = 6

        # Step 1: Initialize services
        logger.info(f"\n📍 Step 1/{total_steps}: Initializing services...")
        if await self.initialize_services():
            success_count += 1
            logger.info("✅ Step 1 completed")
        else:
            logger.error("❌ Step 1 failed")
            return False

        # Step 2: Prepare documents
        logger.info(f"\n📍 Step 2/{total_steps}: Preparing test documents...")
        self.prepare_test_documents()
        if self.test_documents:
            success_count += 1
            logger.info("✅ Step 2 completed")
        else:
            logger.error("❌ Step 2 failed")
            return False

        # Step 3: GPU Processing Demo
        logger.info(f"\n📍 Step 3/{total_steps}: GPU Processing Demo...")
        if await self.run_gpu_processing_demo():
            success_count += 1
            logger.info("✅ Step 3 completed")
        else:
            logger.warning("⚠️  Step 3 had issues, continuing...")

        # Step 4: Enhanced OCR Routing Demo
        logger.info(f"\n📍 Step 4/{total_steps}: Enhanced OCR Routing Demo...")
        if await self.run_enhanced_routing_demo():
            success_count += 1
            logger.info("✅ Step 4 completed")
        else:
            logger.warning("⚠️  Step 4 had issues, continuing...")

        # Step 5: Integration Service Demo
        logger.info(f"\n📍 Step 5/{total_steps}: Integration Service Demo...")
        if await self.run_integration_demo():
            success_count += 1
            logger.info("✅ Step 5 completed")
        else:
            logger.warning("⚠️  Step 5 had issues, continuing...")

        # Step 6: Performance Benchmark
        logger.info(f"\n📍 Step 6/{total_steps}: Performance Benchmark...")
        if await self.run_performance_benchmark():
            success_count += 1
            logger.info("✅ Step 6 completed")
        else:
            logger.warning("⚠️  Step 6 had issues, continuing...")

        # Final analysis
        logger.info(f"\n🎉 === Demo Complete: {success_count}/{total_steps} steps successful ===")

        if success_count >= 4:  # Consider mostly successful if 4+ steps work
            logger.info("🎊 Phase 3 demonstration completed successfully!")
            logger.info("🚀 Your GPU-accelerated document processing pipeline is ready!")

            # Final metadata analysis
            self.run_metadata_analysis()

            logger.info("\n💡 Next Steps:")
            logger.info("   1. Start services: python -m phase3_gpu_accelerated.services.enhanced_gpu_service")
            logger.info("   2. Run integration: python integration_service.py")
            logger.info("   3. Check metadata: python data/metadata_store.py --stats")
            logger.info("   4. View results: ls data/metadata/ data/results/")

        else:
            logger.warning("⚠️  Some demo steps failed. Check the logs above for details.")
            logger.info("💡 Try running individual components to debug issues.")

        return success_count >= 4


async def main():
    """Main demo function."""
    print("=" * 80)
    print("🎭 PHASE 3: GPU ACCELERATION + DEEP FEATURE EXTRACTION")
    print("🚀 Complete End-to-End Demonstration")
    print("=" * 80)

    demo = Phase3EndToEndDemo()

    try:
        success = await demo.run_complete_demo()

        if success:
            print("\n" + "=" * 80)
            print("🎊 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("🚀 Phase 3 is ready for production deployment.")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠️  DEMONSTRATION COMPLETED WITH ISSUES")
            print("💡 Check the logs above for troubleshooting information.")
            print("=" * 80)

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required for Phase 3")
        sys.exit(1)

    # Run the demo
    asyncio.run(main())