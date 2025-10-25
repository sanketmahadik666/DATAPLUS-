#!/usr/bin/env python3
"""
Direct test of the optimized document analyzer without web service
"""

import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

# Add the service directory to path
sys.path.insert(0, str(Path("document_analyzer_service")))

from optimized_service import OptimizedDocumentAnalyzer, ProcessingConfig, SLOTargets

def test_optimized_analyzer():
    """Test the optimized analyzer directly"""
    
    print("Testing Optimized Document Analyzer")
    print("=" * 50)
    
    # Create analyzer with optimized config
    config = ProcessingConfig(
        max_threads=32,
        max_processes=4,
        chunk_size=16,
        cache_size_mb=256
    )
    
    analyzer = OptimizedDocumentAnalyzer(config)
    
    # Test paths
    test_paths = [
        "1000+ PDF_Invoice_Folder",
        "dataset/testing_data/images", 
        "document_analyzer_service"
    ]
    
    results = {}
    
    for path in test_paths:
        if not Path(path).exists():
            print(f"\nSkipping {path} - not found")
            continue
            
        print(f"\nTesting path: {path}")
        start_time = time.time()
        
        try:
            # Test async analysis
            async def run_analysis():
                analysis_results = []
                async for result in analyzer.analyze_path_optimized(path):
                    analysis_results.append(result)
                return analysis_results
            
            # Run the analysis
            analysis_results = asyncio.run(run_analysis())
            
            total_time = time.time() - start_time
            
            # Extract summary from last result
            summary = analysis_results[-1].get("summary", {}) if analysis_results else {}
            metrics = analyzer.metrics.get_metrics()
            
            print(f"  Analysis completed in {total_time:.2f}s")
            print(f"  Files processed: {summary.get('total_inputs', 0)}")
            print(f"  Successful: {summary.get('successful', 0)}")
            print(f"  Failed: {summary.get('failed', 0)}")
            print(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")
            print(f"  Error rate: {metrics['error_rate']:.2%}")
            print(f"  Latency P50: {metrics['latency_p50']:.2f}s")
            print(f"  Latency P95: {metrics['latency_p95']:.2f}s")
            
            # Check SLO compliance
            slo = SLOTargets()
            p50_compliant = metrics['latency_p50'] <= slo.p50_latency
            p95_compliant = metrics['latency_p95'] <= slo.p95_latency
            error_compliant = metrics['error_rate'] <= slo.max_error_rate
            cache_compliant = metrics['cache_hit_rate'] >= slo.min_cache_hit_rate
            
            print(f"  SLO Compliance:")
            print(f"    P50 Latency: {'PASS' if p50_compliant else 'FAIL'} ({metrics['latency_p50']:.2f}s <= {slo.p50_latency}s)")
            print(f"    P95 Latency: {'PASS' if p95_compliant else 'FAIL'} ({metrics['latency_p95']:.2f}s <= {slo.p95_latency}s)")
            print(f"    Error Rate: {'PASS' if error_compliant else 'FAIL'} ({metrics['error_rate']:.2%} <= {slo.max_error_rate:.2%})")
            print(f"    Cache Hit Rate: {'PASS' if cache_compliant else 'FAIL'} ({metrics['cache_hit_rate']:.2%} >= {slo.min_cache_hit_rate:.2%})")
            
            overall_compliant = p50_compliant and p95_compliant and error_compliant and cache_compliant
            print(f"    Overall SLO: {'PASS' if overall_compliant else 'FAIL'}")
            
            # Save results (simplified to avoid JSON serialization issues)
            results[path] = {
                "metrics": {
                    "total_requests": metrics.get("total_requests", 0),
                    "success_count": metrics.get("success_count", 0),
                    "error_count": metrics.get("error_count", 0),
                    "error_rate": metrics.get("error_rate", 0),
                    "cache_hit_rate": metrics.get("cache_hit_rate", 0),
                    "latency_p50": metrics.get("latency_p50", 0),
                    "latency_p95": metrics.get("latency_p95", 0),
                    "throughput_per_minute": metrics.get("throughput_per_minute", 0)
                },
                "slo_compliance": {
                    "p50_compliant": bool(p50_compliant),
                    "p95_compliant": bool(p95_compliant),
                    "error_compliant": bool(error_compliant),
                    "cache_compliant": bool(cache_compliant),
                    "overall_compliant": bool(overall_compliant)
                },
                "processing_time": float(total_time),
                "summary": summary
            }
            
        except Exception as e:
            print(f"  Analysis failed: {e}")
            results[path] = {"error": str(e)}
    
    # Save all results
    with open("optimized_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "=" * 50)
    print("Testing Complete!")
    print("Results saved to optimized_analysis_results.json")
    
    # Summary
    total_files = sum(r.get("metrics", {}).get("total_requests", 0) for r in results.values() if "metrics" in r)
    total_success = sum(r.get("metrics", {}).get("success_count", 0) for r in results.values() if "metrics" in r)
    overall_compliant = all(r.get("slo_compliance", {}).get("overall_compliant", False) for r in results.values() if "slo_compliance" in r)
    
    print(f"\nSummary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Success rate: {total_success/total_files:.2%}" if total_files > 0 else "  Success rate: N/A")
    print(f"  Overall SLO compliance: {'PASS' if overall_compliant else 'FAIL'}")

if __name__ == "__main__":
    test_optimized_analyzer()
