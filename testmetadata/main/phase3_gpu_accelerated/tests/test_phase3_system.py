"""
Phase 3: Comprehensive System Tests

End-to-end testing suite for GPU-accelerated document processing with deep features.
"""

import asyncio
import pytest
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil

# Test imports
try:
    from ..core.gpu_metadata_processor import GPUMetadataProcessor
    from ..core.deep_feature_extractor import extract_optimized_features, DeepFeatureExtractor
    from ..services.enhanced_gpu_service import EnhancedGPUService
    from ..benchmarks.performance_benchmarks import PerformanceBenchmark
except ImportError:
    # Fallback for testing without full imports
    GPUMetadataProcessor = None
    extract_optimized_features = None
    DeepFeatureExtractor = None
    EnhancedGPUService = None
    PerformanceBenchmark = None

logger = logging.getLogger(__name__)


class TestPhase3System:
    """Comprehensive test suite for Phase 3 components."""

    @pytest.fixture
    def test_files(self, tmp_path):
        """Create mock test files for testing."""
        test_files = []

        # Create mock PDF files (just for path testing)
        for i in range(5):
            pdf_path = tmp_path / f"test_doc_{i}.pdf"
            pdf_path.write_bytes(b"Mock PDF content " * 100)  # Mock content
            test_files.append(pdf_path)

        return test_files

    @pytest.fixture
    def gpu_processor(self):
        """GPU processor fixture."""
        if GPUMetadataProcessor:
            processor = GPUMetadataProcessor(max_workers=4)
            yield processor
            processor.cleanup()
        else:
            yield None

    @pytest.mark.asyncio
    async def test_gpu_metadata_processor_basic(self, test_files, gpu_processor):
        """Test basic GPU metadata processor functionality."""
        if not gpu_processor:
            pytest.skip("GPU processor not available")

        results = await gpu_processor.process_document_batch(test_files[:2])

        assert len(results) == 2
        for result in results:
            assert result['processing_status'] == 'success'
            assert 'document_id' in result
            assert 'file_path' in result
            assert 'gpu_accelerated' in result

    @pytest.mark.asyncio
    async def test_deep_feature_extractor(self):
        """Test deep feature extraction."""
        if not DeepFeatureExtractor:
            pytest.skip("Deep feature extractor not available")

        import numpy as np

        # Create mock image
        mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        extractor = DeepFeatureExtractor()
        features = extractor.extract_comprehensive_features(mock_image)

        assert 'vision_features' in features or len(features) > 0
        assert 'deep_features' in features

    def test_optimized_feature_extraction(self):
        """Test optimized feature extraction."""
        if not extract_optimized_features:
            pytest.skip("Optimized feature extraction not available")

        import numpy as np

        mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = extract_optimized_features(mock_image)

        assert 'optimized_features' in features
        assert 'semantic_features' in features

    @pytest.mark.asyncio
    async def test_enhanced_service_processing(self, test_files):
        """Test enhanced GPU service processing."""
        if not EnhancedGPUService:
            pytest.skip("Enhanced service not available")

        service = EnhancedGPUService()

        try:
            from ..services.enhanced_gpu_service import ProcessingRequest

            request = ProcessingRequest(
                pdf_paths=[str(p) for p in test_files],
                max_workers=4,
                extract_deep_features=True,
                batch_size=2
            )

            response = await service.process_documents(request)

            assert response.status == "success"
            assert response.total_documents == len(test_files)
            assert len(response.results) == len(test_files)
            assert response.gpu_accelerated in [True, False]  # May be False if no GPU

        finally:
            service.cleanup()

    @pytest.mark.asyncio
    async def test_service_health_check(self):
        """Test service health check."""
        if not EnhancedGPUService:
            pytest.skip("Enhanced service not available")

        service = EnhancedGPUService()
        try:
            health = await service.get_health_status()

            assert health.status == "healthy"
            assert 'gpu_available' in health.__dict__
            assert 'cpu_usage' in health.__dict__
            assert 'memory_usage' in health.__dict__

        finally:
            service.cleanup()

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, test_files):
        """Test performance benchmarking."""
        if not PerformanceBenchmark:
            pytest.skip("Performance benchmark not available")

        benchmark = PerformanceBenchmark()
        results = await benchmark.run_comprehensive_benchmarks(test_files)

        assert 'system_info' in results
        assert 'benchmark_results' in results
        assert 'analysis' in results
        assert len(results['benchmark_results']) > 0

    @pytest.mark.asyncio
    async def test_batch_processing_scalability(self, test_files, gpu_processor):
        """Test batch processing scalability."""
        if not gpu_processor:
            pytest.skip("GPU processor not available")

        batch_sizes = [1, 2, 5]

        for batch_size in batch_sizes:
            batch = test_files[:batch_size]
            start_time = time.time()
            results = await gpu_processor.process_document_batch(batch)
            processing_time = time.time() - start_time

            assert len(results) == len(batch)
            assert processing_time > 0

            # Check that all results are successful
            successful = sum(1 for r in results if r.get('processing_status') == 'success')
            assert successful == len(batch)

    def test_error_handling_invalid_files(self, gpu_processor):
        """Test error handling with invalid files."""
        if not gpu_processor:
            pytest.skip("GPU processor not available")

        async def run_test():
            invalid_paths = [Path("/nonexistent/file.pdf")]
            results = await gpu_processor.process_document_batch(invalid_paths)

            # Should handle gracefully
            assert len(results) == 1
            assert results[0]['processing_status'] == 'error'

        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_memory_profiling(self, test_files):
        """Test memory profiling functionality."""
        if not PerformanceBenchmark:
            pytest.skip("Performance benchmark not available")

        benchmark = PerformanceBenchmark()
        results = await benchmark.run_memory_profiling(test_files, duration_seconds=5)

        assert 'memory_stats' in results
        assert 'analysis' in results
        assert results['duration_seconds'] == 5

        memory_stats = results['memory_stats']
        assert 'timestamps' in memory_stats
        assert 'memory_usage' in memory_stats
        assert len(memory_stats['timestamps']) > 0

    def test_system_info_collection(self):
        """Test system information collection."""
        if not PerformanceBenchmark:
            pytest.skip("Performance benchmark not available")

        benchmark = PerformanceBenchmark()
        system_info = benchmark.system_info

        assert 'cpu_count' in system_info
        assert 'memory_total_gb' in system_info
        assert 'platform' in system_info
        assert 'gpu_available' in system_info

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, test_files, gpu_processor):
        """Test concurrent processing capabilities."""
        if not gpu_processor:
            pytest.skip("GPU processor not available")

        # Test multiple concurrent batches
        tasks = []
        for i in range(3):
            batch = test_files[:2]  # Small batches
            task = gpu_processor.process_document_batch(batch)
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        assert len(results) == 3
        for batch_results in results:
            assert len(batch_results) == 2
            for result in batch_results:
                assert result['processing_status'] == 'success'

        # Concurrent processing should be faster than sequential
        assert total_time < 10  # Reasonable time limit

    @pytest.mark.asyncio
    async def test_service_benchmark_endpoint(self, test_files):
        """Test service benchmark endpoint."""
        if not EnhancedGPUService:
            pytest.skip("Enhanced service not available")

        service = EnhancedGPUService()

        try:
            from ..services.enhanced_gpu_service import BenchmarkRequest

            request = BenchmarkRequest(
                test_files=[str(p) for p in test_files[:2]],
                iterations=1,
                batch_sizes=[1, 2]
            )

            response = await service.run_benchmarks(request)

            assert response.status == "success"
            assert response.total_tests > 0
            assert len(response.results) == response.total_tests
            assert 'performance_summary' in response.__dict__

        finally:
            service.cleanup()


class TestIntegrationScenarios:
    """Integration tests for complete Phase 3 workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_processing_pipeline(self, test_files):
        """Test complete end-to-end processing pipeline."""
        if not (GPUMetadataProcessor and DeepFeatureExtractor and EnhancedGPUService):
            pytest.skip("Required components not available")

        # Initialize components
        service = EnhancedGPUService()

        try:
            from ..services.enhanced_gpu_service import ProcessingRequest

            # Process documents
            request = ProcessingRequest(
                pdf_paths=[str(p) for p in test_files],
                max_workers=8,
                extract_deep_features=True,
                batch_size=3
            )

            response = await service.process_documents(request)

            # Validate complete pipeline
            assert response.status == "success"
            assert all(r.get('processing_status') == 'success' for r in response.results)
            assert response.files_per_second > 0

            # Check performance metrics
            perf = response.performance_metrics
            assert 'cpu_usage_percent' in perf
            assert 'memory_usage_percent' in perf

        finally:
            service.cleanup()

    @pytest.mark.asyncio
    async def test_scalability_under_load(self, test_files):
        """Test system scalability under load."""
        if not EnhancedGPUService:
            pytest.skip("Enhanced service not available")

        service = EnhancedGPUService()

        try:
            from ..services.enhanced_gpu_service import ProcessingRequest

            # Simulate high load with multiple concurrent requests
            requests = []
            for i in range(5):
                request = ProcessingRequest(
                    pdf_paths=[str(p) for p in test_files],
                    max_workers=16,
                    extract_deep_features=False,  # Disable for speed
                    batch_size=5
                )
                requests.append(request)

            # Process concurrently
            start_time = time.time()
            tasks = [service.process_documents(req) for req in requests]
            responses = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            # Validate all requests succeeded
            assert all(r.status == "success" for r in responses)

            # Check performance under load
            total_files_processed = sum(len(r.results) for r in responses)
            overall_fps = total_files_processed / total_time if total_time > 0 else 0

            assert overall_fps > 0
            logger.info(f"Scalability test: {overall_fps:.2f} files/sec under load")

        finally:
            service.cleanup()

    @pytest.mark.asyncio
    async def test_comprehensive_benchmark_suite(self, test_files):
        """Test comprehensive benchmark suite execution."""
        if not PerformanceBenchmark:
            pytest.skip("Performance benchmark not available")

        benchmark = PerformanceBenchmark()

        # Run full benchmark suite
        results = await benchmark.run_comprehensive_benchmarks(test_files[:3])  # Small subset

        # Validate benchmark structure
        assert 'system_info' in results
        assert 'benchmark_results' in results
        assert 'analysis' in results

        analysis = results['analysis']
        assert 'summary' in analysis
        assert 'comparisons' in analysis
        assert 'recommendations' in analysis

        # Check that multiple configurations were tested
        assert len(analysis['summary']) >= 2

        # Validate performance metrics
        for config_name, stats in analysis['summary'].items():
            assert 'average_fps' in stats
            assert stats['average_fps'] > 0


# Utility functions for test setup
def create_mock_pdf(path: Path, size_kb: int = 100):
    """Create a mock PDF file for testing."""
    # Simple mock PDF content
    content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    content += b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n"
    content += b"3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n"
    content += b"4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Mock PDF Content) Tj\nET\nendstream\nendobj\n"
    content += b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000200 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n284\n%%EOF"

    # Pad to desired size
    while len(content) < size_kb * 1024:
        content += b" " * 1000

    path.write_bytes(content[:size_kb * 1024])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])