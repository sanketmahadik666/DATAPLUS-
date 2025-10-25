#!/usr/bin/env python3
"""
Performance testing script for the optimized document analyzer service
Tests SLO compliance and performance under load
"""

import asyncio
import aiohttp
import time
import json
from pathlib import Path
import statistics
from typing import List, Dict, Any

class PerformanceTester:
    """Test the optimized service for SLO compliance"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def test_metrics(self) -> Dict[str, Any]:
        """Test metrics endpoint"""
        async with self.session.get(f"{self.base_url}/metrics") as response:
            return await response.json()
    
    async def test_slo_status(self) -> Dict[str, Any]:
        """Test SLO compliance"""
        async with self.session.get(f"{self.base_url}/slo-status") as response:
            return await response.json()
    
    async def test_single_analysis(self, path: str) -> Dict[str, Any]:
        """Test single analysis request"""
        start_time = time.time()
        
        async with self.session.post(
            f"{self.base_url}/analyze-path",
            params={"path": path}
        ) as response:
            result = await response.json()
            latency = time.time() - start_time
            result["latency"] = latency
            return result
    
    async def test_streaming_analysis(self, path: str) -> List[Dict[str, Any]]:
        """Test streaming analysis"""
        start_time = time.time()
        results = []
        
        async with self.session.post(
            f"{self.base_url}/analyze-path-stream",
            params={"path": path}
        ) as response:
            async for line in response.content:
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:].decode())
                        results.append(data)
                    except json.JSONDecodeError:
                        continue
        
        total_latency = time.time() - start_time
        return results, total_latency
    
    async def load_test(self, path: str, concurrent_requests: int = 10) -> Dict[str, Any]:
        """Run load test with concurrent requests"""
        print(f"Running load test with {concurrent_requests} concurrent requests...")
        
        start_time = time.time()
        tasks = [self.test_single_analysis(path) for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        latencies = [r.get("latency", 0) for r in valid_results]
        
        return {
            "total_requests": len(results),
            "successful_requests": len(valid_results),
            "failed_requests": len(results) - len(valid_results),
            "total_time": total_time,
            "requests_per_second": len(valid_results) / total_time,
            "latency_stats": {
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "mean": statistics.mean(latencies) if latencies else 0,
                "median": statistics.median(latencies) if latencies else 0,
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99)
            }
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def test_slo_compliance(self, path: str) -> Dict[str, Any]:
        """Test SLO compliance"""
        print("Testing SLO compliance...")
        
        # Test single request
        single_result = await self.test_single_analysis(path)
        single_latency = single_result.get("latency", 0)
        
        # Test load
        load_result = await self.load_test(path, concurrent_requests=20)
        
        # Get SLO status
        slo_status = await self.test_slo_status()
        
        # Get metrics
        metrics = await self.test_metrics()
        
        return {
            "single_request_latency": single_latency,
            "load_test_results": load_result,
            "slo_status": slo_status,
            "current_metrics": metrics,
            "compliance_summary": {
                "latency_p50_compliant": single_latency <= 2.0,
                "latency_p95_compliant": single_latency <= 8.0,
                "error_rate_compliant": load_result["failed_requests"] / load_result["total_requests"] <= 0.001,
                "throughput_adequate": load_result["requests_per_second"] >= 1.67  # 100/min = 1.67/sec
            }
        }

async def main():
    """Run comprehensive performance tests"""
    print("Starting Document Analyzer Performance Tests")
    print("=" * 60)
    
    # Test paths
    test_paths = [
        "1000+ PDF_Invoice_Folder",
        "dataset/testing_data/images",
        "document_analyzer_service"
    ]
    
    async with PerformanceTester() as tester:
        # Test 1: Health and basic functionality
        print("\n1. Testing Health and Basic Functionality...")
        try:
            health = await tester.test_health()
            print(f"Service is healthy: {health['status']}")
            print(f"  Uptime: {health['uptime']:.2f} seconds")
        except Exception as e:
            print(f"Health check failed: {e}")
            return
        
        # Test 2: SLO Status
        print("\n2. Checking SLO Status...")
        try:
            slo_status = await tester.test_slo_status()
            print(f"SLO Status retrieved")
            print(f"  Overall compliant: {slo_status['overall_compliant']}")
            print(f"  P50 Latency: {slo_status['latency_p50']['current']:.2f}s (target: {slo_status['latency_p50']['target']}s)")
            print(f"  P95 Latency: {slo_status['latency_p95']['current']:.2f}s (target: {slo_status['latency_p95']['target']}s)")
        except Exception as e:
            print(f"SLO status check failed: {e}")
        
        # Test 3: Single document analysis
        print("\n3. Testing Single Document Analysis...")
        for path in test_paths:
            if Path(path).exists():
                print(f"\n  Testing path: {path}")
                try:
                    result = await tester.test_single_analysis(path)
                    latency = result.get("latency", 0)
                    summary = result.get("summary", {})
                    print(f"    Analysis completed in {latency:.2f}s")
                    print(f"    Files processed: {summary.get('total_inputs', 0)}")
                    print(f"    Success rate: {summary.get('successful', 0)}/{summary.get('total_inputs', 0)}")
                except Exception as e:
                    print(f"    Analysis failed: {e}")
        
        # Test 4: Load testing
        print("\n4. Running Load Tests...")
        for path in test_paths:
            if Path(path).exists():
                print(f"\n  Load testing: {path}")
                try:
                    load_result = await tester.load_test(path, concurrent_requests=5)
                    print(f"    Load test completed")
                    print(f"    Requests/sec: {load_result['requests_per_second']:.2f}")
                    print(f"    Success rate: {load_result['successful_requests']}/{load_result['total_requests']}")
                    print(f"    Latency P95: {load_result['latency_stats']['p95']:.2f}s")
                except Exception as e:
                    print(f"    Load test failed: {e}")
        
        # Test 5: Streaming analysis
        print("\n5. Testing Streaming Analysis...")
        for path in test_paths:
            if Path(path).exists():
                print(f"\n  Streaming test: {path}")
                try:
                    results, total_latency = await tester.test_streaming_analysis(path)
                    print(f"    Streaming completed in {total_latency:.2f}s")
                    print(f"    Chunks received: {len(results)}")
                except Exception as e:
                    print(f"    Streaming test failed: {e}")
        
        # Test 6: SLO Compliance Summary
        print("\n6. SLO Compliance Summary...")
        try:
            compliance = await tester.test_slo_compliance("1000+ PDF_Invoice_Folder")
            print(f"SLO Compliance Test Completed")
            
            compliance_summary = compliance["compliance_summary"]
            print(f"  Latency P50 Compliant: {compliance_summary['latency_p50_compliant']}")
            print(f"  Latency P95 Compliant: {compliance_summary['latency_p95_compliant']}")
            print(f"  Error Rate Compliant: {compliance_summary['error_rate_compliant']}")
            print(f"  Throughput Adequate: {compliance_summary['throughput_adequate']}")
            
            # Save detailed results
            with open("performance_test_results.json", "w") as f:
                json.dump(compliance, f, indent=2)
            print(f"  Detailed results saved to performance_test_results.json")
            
        except Exception as e:
            print(f"SLO compliance test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Performance Testing Complete!")
    print("Check performance_test_results.json for detailed metrics")

if __name__ == "__main__":
    asyncio.run(main())
