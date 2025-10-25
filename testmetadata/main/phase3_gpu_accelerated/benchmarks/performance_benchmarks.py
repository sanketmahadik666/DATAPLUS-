"""
Phase 3: Comprehensive Performance Benchmarks

Advanced benchmarking suite for GPU-accelerated document processing with
deep feature extraction and comparative analysis.
"""

import asyncio
import logging
import time
import statistics
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json

# Import components (with fallbacks)
try:
    from ..core.gpu_metadata_processor import GPUMetadataProcessor
    from ..core.deep_feature_extractor import extract_optimized_features
except ImportError:
    GPUMetadataProcessor = None
    extract_optimized_features = None

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    name: str
    batch_sizes: List[int] = None
    iterations: int = 3
    warmup_iterations: int = 1
    max_workers: int = 32
    enable_gpu: bool = True
    enable_deep_features: bool = True
    test_files: List[Path] = None

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 5, 10, 20, 50]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    config_name: str
    batch_size: int
    iteration: int
    processing_time: float
    files_per_second: float
    cpu_usage: float
    memory_usage: float
    gpu_memory_used: Optional[float]
    gpu_accelerated: bool
    deep_features_enabled: bool
    timestamp: float


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self):
        self.results = []
        self.system_info = self._collect_system_info()
        self.executor = ThreadPoolExecutor(max_workers=32)

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmark context."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': psutil.platform,
            'python_version': __import__('sys').version,
        }

        # GPU information
        try:
            import torch
            info['gpu_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info['cuda_version'] = torch.version.cuda
        except ImportError:
            info['gpu_available'] = False

        return info

    async def run_comprehensive_benchmarks(self, test_files: List[Path]) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        logger.info(f"Starting comprehensive benchmarks with {len(test_files)} test files")

        # Define benchmark configurations
        configs = [
            BenchmarkConfig("GPU_Full", test_files=test_files),
            BenchmarkConfig("GPU_No_Deep_Features", enable_deep_features=False, test_files=test_files),
            BenchmarkConfig("CPU_Fallback", enable_gpu=False, test_files=test_files),
            BenchmarkConfig("High_Concurrency", batch_sizes=[10, 25, 50], max_workers=64, test_files=test_files),
        ]

        all_results = []

        for config in configs:
            logger.info(f"Running benchmark: {config.name}")
            config_results = await self.run_benchmark_config(config)
            all_results.extend(config_results)

        # Analyze results
        analysis = self.analyze_benchmark_results(all_results)

        return {
            'system_info': self.system_info,
            'benchmark_results': [result.__dict__ for result in all_results],
            'analysis': analysis,
            'timestamp': time.time()
        }

    async def run_benchmark_config(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run a single benchmark configuration."""
        results = []

        for batch_size in config.batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            # Warmup runs
            for _ in range(config.warmup_iterations):
                await self._run_single_test(config, batch_size, warmup=True)

            # Actual benchmark runs
            for iteration in range(config.iterations):
                result = await self._run_single_test(config, batch_size, iteration=iteration)
                results.append(result)

        return results

    async def _run_single_test(self, config: BenchmarkConfig, batch_size: int, iteration: int = 0, warmup: bool = False) -> BenchmarkResult:
        """Run a single benchmark test."""
        # Select subset of test files for this batch
        test_files = config.test_files[:min(batch_size, len(config.test_files))]

        start_time = time.time()

        # Initialize processor based on config
        if config.enable_gpu and GPUMetadataProcessor:
            processor = GPUMetadataProcessor(max_workers=config.max_workers)
        else:
            processor = None  # CPU fallback

        try:
            if processor:
                # GPU-accelerated processing
                batch_results = await processor.process_document_batch(test_files)

                # Add deep features if enabled
                if config.enable_deep_features and extract_optimized_features:
                    batch_results = await self._add_deep_features_batch(batch_results)
            else:
                # CPU fallback processing
                batch_results = await self._cpu_fallback_processing(test_files)

            processing_time = time.time() - start_time

            # Collect system metrics
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            gpu_memory = await self._get_gpu_memory_usage()

            files_per_second = len(test_files) / processing_time if processing_time > 0 else 0

            result = BenchmarkResult(
                config_name=config.name,
                batch_size=batch_size,
                iteration=iteration,
                processing_time=processing_time,
                files_per_second=files_per_second,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_memory_used=gpu_memory,
                gpu_accelerated=processor is not None,
                deep_features_enabled=config.enable_deep_features,
                timestamp=time.time()
            )

            if not warmup:
                logger.info(f"Batch size {batch_size}: {files_per_second:.2f} files/sec, {processing_time:.3f}s")

            return result

        finally:
            if processor:
                processor.cleanup()

    async def _add_deep_features_batch(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add deep features to a batch of results."""
        # Parallel processing of deep features
        tasks = []
        for result in results:
            if result.get('processing_status') == 'success':
                # This would extract from actual image data in production
                task = asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._extract_deep_features_single,
                    result
                )
                tasks.append(task)

        if tasks:
            enhanced_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, enhanced in enumerate(enhanced_results):
                if isinstance(enhanced, Exception):
                    logger.warning(f"Deep feature extraction failed: {enhanced}")
                else:
                    results[i] = enhanced

        return results

    def _extract_deep_features_single(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract deep features for a single result."""
        # Placeholder - would use actual image data
        result['deep_features_extracted'] = True
        result['feature_vector_length'] = 2048 + 768 + 64  # vision + text + structural
        return result

    async def _cpu_fallback_processing(self, test_files: List[Path]) -> List[Dict[str, Any]]:
        """CPU-based fallback processing for comparison."""
        results = []
        for pdf_path in test_files:
            result = {
                'document_id': pdf_path.stem,
                'file_path': str(pdf_path),
                'file_size': pdf_path.stat().st_size,
                'processing_status': 'success',
                'gpu_accelerated': False,
                'processing_timestamp': time.time(),
            }
            results.append(result)
        return results

    async def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            return None

    def analyze_benchmark_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark results and generate insights."""
        if not results:
            return {}

        analysis = {
            'summary': {},
            'comparisons': {},
            'recommendations': []
        }

        # Group results by configuration
        config_groups = {}
        for result in results:
            config_name = result.config_name
            if config_name not in config_groups:
                config_groups[config_name] = []
            config_groups[config_name].append(result)

        # Calculate statistics for each configuration
        for config_name, config_results in config_groups.items():
            fps_values = [r.files_per_second for r in config_results]
            time_values = [r.processing_time for r in config_results]

            analysis['summary'][config_name] = {
                'average_fps': round(statistics.mean(fps_values), 2),
                'max_fps': round(max(fps_values), 2),
                'min_fps': round(min(fps_values), 2),
                'std_fps': round(statistics.stdev(fps_values), 2) if len(fps_values) > 1 else 0,
                'average_time': round(statistics.mean(time_values), 3),
                'total_tests': len(config_results)
            }

        # Comparative analysis
        if 'GPU_Full' in analysis['summary'] and 'CPU_Fallback' in analysis['summary']:
            gpu_fps = analysis['summary']['GPU_Full']['average_fps']
            cpu_fps = analysis['summary']['CPU_Fallback']['average_fps']

            speedup = gpu_fps / cpu_fps if cpu_fps > 0 else float('inf')
            analysis['comparisons']['gpu_vs_cpu_speedup'] = round(speedup, 2)

        # Performance scaling analysis
        gpu_results = [r for r in results if r.config_name == 'GPU_Full']
        batch_scaling = {}
        for result in gpu_results:
            batch_size = result.batch_size
            if batch_size not in batch_scaling:
                batch_scaling[batch_size] = []
            batch_scaling[batch_size].append(result.files_per_second)

        analysis['scaling_analysis'] = {
            'batch_size_performance': {
                bs: round(statistics.mean(fps_list), 2)
                for bs, fps_list in batch_scaling.items()
            }
        }

        # Generate recommendations
        best_config = max(analysis['summary'].items(), key=lambda x: x[1]['average_fps'])
        analysis['recommendations'].append(f"Best configuration: {best_config[0]} with {best_config[1]['average_fps']} files/sec")

        if analysis['comparisons'].get('gpu_vs_cpu_speedup', 1) > 2:
            analysis['recommendations'].append("GPU acceleration provides significant performance improvement")

        optimal_batch = max(analysis['scaling_analysis']['batch_size_performance'].items(), key=lambda x: x[1])
        analysis['recommendations'].append(f"Optimal batch size: {optimal_batch[0]} documents")

        return analysis

    async def run_memory_profiling(self, test_files: List[Path], duration_seconds: int = 60) -> Dict[str, Any]:
        """Run memory profiling benchmark."""
        logger.info(f"Starting memory profiling for {duration_seconds} seconds")

        memory_stats = {
            'timestamps': [],
            'memory_usage': [],
            'gpu_memory_usage': [],
            'cpu_usage': []
        }

        start_time = time.time()
        processor = GPUMetadataProcessor(max_workers=16) if GPUMetadataProcessor else None

        try:
            while time.time() - start_time < duration_seconds:
                # Simulate continuous processing
                batch = test_files[:10]  # Small batch for continuous processing

                if processor:
                    await processor.process_document_batch(batch)

                # Collect memory stats
                current_time = time.time() - start_time
                memory_stats['timestamps'].append(current_time)
                memory_stats['memory_usage'].append(psutil.virtual_memory().percent)
                memory_stats['cpu_usage'].append(psutil.cpu_percent())

                gpu_mem = await self._get_gpu_memory_usage()
                memory_stats['gpu_memory_usage'].append(gpu_mem)

                await asyncio.sleep(1)  # Sample every second

        finally:
            if processor:
                processor.cleanup()

        # Calculate memory statistics
        memory_analysis = {
            'average_memory_usage': statistics.mean(memory_stats['memory_usage']),
            'max_memory_usage': max(memory_stats['memory_usage']),
            'average_cpu_usage': statistics.mean(memory_stats['cpu_usage']),
            'max_cpu_usage': max(memory_stats['cpu_usage']),
            'memory_stable': statistics.stdev(memory_stats['memory_usage']) < 5.0,  # Low variance = stable
        }

        if memory_stats['gpu_memory_usage'][0] is not None:
            gpu_usage = [m for m in memory_stats['gpu_memory_usage'] if m is not None]
            memory_analysis.update({
                'average_gpu_memory_gb': statistics.mean(gpu_usage),
                'max_gpu_memory_gb': max(gpu_usage),
                'gpu_memory_stable': statistics.stdev(gpu_usage) < 0.1,
            })

        return {
            'memory_stats': memory_stats,
            'analysis': memory_analysis,
            'duration_seconds': duration_seconds
        }

    def export_results(self, results: Dict[str, Any], output_path: Path):
        """Export benchmark results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Benchmark results exported to {output_path}")


# Convenience functions
async def run_gpu_benchmarks(test_files: List[Path], output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run comprehensive GPU benchmarks with optional export."""
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_comprehensive_benchmarks(test_files)

    if output_path:
        benchmark.export_results(results, output_path)

    return results


async def run_memory_profile(test_files: List[Path], duration: int = 60) -> Dict[str, Any]:
    """Run memory profiling benchmark."""
    benchmark = PerformanceBenchmark()
    return await benchmark.run_memory_profiling(test_files, duration)


# Main execution for standalone benchmarking
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Phase 3 Performance Benchmarks")
    parser.add_argument("--test-files", nargs="+", required=True, help="Test PDF files")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--memory-profile", action="store_true", help="Run memory profiling")
    parser.add_argument("--duration", type=int, default=60, help="Memory profiling duration")

    args = parser.parse_args()

    test_files = [Path(f) for f in args.test_files]

    async def main():
        if args.memory_profile:
            results = await run_memory_profile(test_files, args.duration)
            print(f"Memory profiling completed: {results['analysis']}")
        else:
            results = await run_gpu_benchmarks(test_files)
            print("Benchmark summary:")
            for config, stats in results['analysis']['summary'].items():
                print(f"  {config}: {stats['average_fps']} files/sec")

        if args.output:
            benchmark = PerformanceBenchmark()
            benchmark.export_results(results, Path(args.output))
            print(f"Results exported to {args.output}")

    asyncio.run(main())