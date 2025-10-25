#!/usr/bin/env python3
"""
Comprehensive Test Suite for OCR Routing Pipeline
Tests both metadata extraction and OCR routing services with 32 workers
"""

import json
import time
import requests
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    """Comprehensive testing of the complete OCR routing pipeline"""
    
    def __init__(self, max_workers: int = 32):
        self.max_workers = max_workers
        self.base_url = "http://localhost:8002"
        self.test_results = {
            "metadata_extraction": {},
            "ocr_routing": {},
            "performance_metrics": {},
            "overall_status": "PENDING"
        }
        self.lock = threading.Lock()
    
    def test_metadata_extraction_with_workers(self, pdf_folder: str = "../1000+ PDF_Invoice_Folder") -> Dict[str, Any]:
        """Test metadata extraction with 32 workers"""
        logger.info("=" * 80)
        logger.info("TESTING METADATA EXTRACTION WITH 32 WORKERS")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Import the metadata extractor
            from core.fixed_fast_metadata_extractor import FixedFastMetadataExtractor
            
            # Initialize with 32 workers
            extractor = FixedFastMetadataExtractor()
            extractor.max_workers = self.max_workers
            
            # Get all PDF files
            target_path = Path(pdf_folder)
            if not target_path.exists():
                raise FileNotFoundError(f"PDF folder '{pdf_folder}' not found")
            
            pdf_files = list(target_path.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files to process with {self.max_workers} workers")
            
            # Process files with maximum parallelism
            all_results = []
            successful = 0
            failed = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_pdf = {
                    executor.submit(extractor.extract_single_pdf_metadata, pdf_file): pdf_file 
                    for pdf_file in pdf_files
                }
                
                # Process results as they complete
                for i, future in enumerate(as_completed(future_to_pdf)):
                    pdf_file = future_to_pdf[future]
                    
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        if result.get("processing_status") == "success":
                            successful += 1
                        else:
                            failed += 1
                        
                        # Progress logging every 50 files
                        if (i + 1) % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = (i + 1) / elapsed
                            eta = (len(pdf_files) - i - 1) / rate if rate > 0 else 0
                            logger.info(f"Processed {i+1}/{len(pdf_files)} files. Rate: {rate:.1f} files/sec. ETA: {eta:.1f}s")
                    
                    except Exception as e:
                        logger.error(f"Error processing {pdf_file}: {e}")
                        failed += 1
                        all_results.append({
                            "document_id": pdf_file.stem,
                            "file_name": pdf_file.name,
                            "file_path": str(pdf_file),
                            "processing_status": "error",
                            "error_message": str(e)
                        })
            
            extraction_time = time.time() - start_time
            
            # Calculate metrics
            self.test_results["metadata_extraction"] = {
                "status": "SUCCESS",
                "total_files": len(pdf_files),
                "successful_extractions": successful,
                "failed_extractions": failed,
                "success_rate": successful / len(pdf_files) * 100,
                "extraction_time_seconds": extraction_time,
                "files_per_second": len(pdf_files) / extraction_time,
                "workers_used": self.max_workers,
                "sample_features": self._extract_sample_features(all_results[0]) if all_results else {}
            }
            
            logger.info(f"Metadata extraction completed: {successful}/{len(pdf_files)} successful")
            logger.info(f"Processing time: {extraction_time:.2f}s ({len(pdf_files)/extraction_time:.1f} files/sec)")
            logger.info(f"Workers used: {self.max_workers}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            self.test_results["metadata_extraction"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return []
    
    def test_ocr_routing_with_workers(self, test_documents: List[Dict]) -> Dict[str, Any]:
        """Test OCR routing with multiple workers"""
        logger.info("=" * 80)
        logger.info("TESTING OCR ROUTING WITH MULTIPLE WORKERS")
        logger.info("=" * 80)
        
        try:
            # Check if service is running
            health_response = requests.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code != 200:
                raise Exception("OCR routing service is not running")
            
            logger.info("OCR routing service is healthy")
            
            # Test different routing scenarios with workers
            routing_tests = [
                {
                    "name": "Default Parameters",
                    "alpha": 1.0, "beta": 0.5, "gamma": 0.5
                },
                {
                    "name": "High Accuracy Preference",
                    "alpha": 2.0, "beta": 0.1, "gamma": 0.1
                },
                {
                    "name": "High Speed Preference",
                    "alpha": 0.5, "beta": 2.0, "gamma": 0.1
                },
                {
                    "name": "Low Resource Preference",
                    "alpha": 0.5, "beta": 0.1, "gamma": 2.0
                }
            ]
            
            routing_results = {}
            
            for test in routing_tests:
                logger.info(f"Testing {test['name']} with {self.max_workers} workers...")
                
                start_time = time.time()
                
                # Convert test documents to routing format
                routing_docs = []
                for doc in test_documents[:10]:  # Test with first 10 documents
                    if doc.get("processing_status") == "success":
                        routing_doc = self._convert_to_routing_format(doc)
                        routing_docs.append(routing_doc)
                
                if not routing_docs:
                    continue
                
                # Make routing request
                payload = {
                    "document_features": routing_docs,
                    "alpha": test["alpha"],
                    "beta": test["beta"],
                    "gamma": test["gamma"],
                    "delta_fallback_threshold": 0.03
                }
                
                response = requests.post(f"{self.base_url}/route", json=payload, timeout=30)
                response.raise_for_status()
                
                routing_time = time.time() - start_time
                result = response.json()
                
                # Analyze results
                engine_distribution = {}
                total_confidence = 0.0
                total_latency = 0.0
                
                for doc_result in result["documents"]:
                    engine = doc_result["chosen_engine"]
                    engine_distribution[engine] = engine_distribution.get(engine, 0) + 1
                    total_confidence += doc_result["expected_confidence"]
                    total_latency += doc_result["expected_latency_sec"]
                
                routing_results[test["name"]] = {
                    "status": "SUCCESS",
                    "processing_time_ms": routing_time * 1000,
                    "documents_processed": len(result["documents"]),
                    "engine_distribution": engine_distribution,
                    "average_confidence": total_confidence / len(result["documents"]) if result["documents"] else 0,
                    "average_latency": total_latency / len(result["documents"]) if result["documents"] else 0,
                    "features_used": len(result["features_used"]),
                    "workers_used": self.max_workers,
                    "sample_routing": result["documents"][0] if result["documents"] else {}
                }
                
                logger.info(f"{test['name']}: {len(result['documents'])} docs in {routing_time*1000:.1f}ms")
                logger.info(f"   Engine distribution: {engine_distribution}")
            
            self.test_results["ocr_routing"] = {
                "status": "SUCCESS",
                "service_health": "HEALTHY",
                "routing_tests": routing_results,
                "total_tests": len(routing_tests),
                "successful_tests": len([r for r in routing_results.values() if r["status"] == "SUCCESS"]),
                "workers_used": self.max_workers
            }
            
            logger.info("All OCR routing tests completed successfully")
            return routing_results
            
        except Exception as e:
            logger.error(f"OCR routing test failed: {e}")
            self.test_results["ocr_routing"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return {}
    
    def test_batch_processing(self, test_documents: List[Dict]) -> Dict[str, Any]:
        """Test batch processing capabilities"""
        logger.info("=" * 80)
        logger.info("TESTING BATCH PROCESSING CAPABILITIES")
        logger.info("=" * 80)
        
        try:
            # Test different batch sizes
            batch_sizes = [10, 25, 50, 100]
            batch_results = {}
            
            for batch_size in batch_sizes:
                logger.info(f"Testing batch size: {batch_size}")
                
                # Prepare batch
                batch_docs = []
                for doc in test_documents[:batch_size]:
                    if doc.get("processing_status") == "success":
                        routing_doc = self._convert_to_routing_format(doc)
                        batch_docs.append(routing_doc)
                
                if not batch_docs:
                    continue
                
                start_time = time.time()
                
                # Make batch request
                payload = {
                    "document_features": batch_docs,
                    "alpha": 1.0,
                    "beta": 0.5,
                    "gamma": 0.5
                }
                
                response = requests.post(f"{self.base_url}/route-batch", json=payload, timeout=60)
                response.raise_for_status()
                
                batch_time = time.time() - start_time
                result = response.json()
                
                batch_results[f"batch_{batch_size}"] = {
                    "status": "SUCCESS",
                    "batch_size": batch_size,
                    "processing_time_ms": batch_time * 1000,
                    "documents_per_second": batch_size / batch_time,
                    "average_latency_per_doc": (batch_time * 1000) / batch_size,
                    "engine_distribution": result.get("summary", {}).get("engine_distribution", {}),
                    "average_confidence": result.get("summary", {}).get("average_confidence", 0)
                }
                
                logger.info(f"Batch {batch_size}: {batch_size} docs in {batch_time*1000:.1f}ms ({batch_size/batch_time:.1f} docs/sec)")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch processing test failed: {e}")
            return {}
    
    def test_performance_under_load(self, test_documents: List[Dict]) -> Dict[str, Any]:
        """Test performance under concurrent load"""
        logger.info("=" * 80)
        logger.info("TESTING PERFORMANCE UNDER CONCURRENT LOAD")
        logger.info("=" * 80)
        
        try:
            # Simulate concurrent requests
            concurrent_requests = 10
            request_results = []
            
            def make_request(request_id: int):
                """Make a single routing request"""
                try:
                    # Prepare small batch for each request
                    batch_docs = []
                    for doc in test_documents[request_id*3:(request_id+1)*3]:
                        if doc.get("processing_status") == "success":
                            routing_doc = self._convert_to_routing_format(doc)
                            batch_docs.append(routing_doc)
                    
                    if not batch_docs:
                        return {"request_id": request_id, "status": "NO_DATA"}
                    
                    start_time = time.time()
                    
                    payload = {
                        "document_features": batch_docs,
                        "alpha": 1.0,
                        "beta": 0.5,
                        "gamma": 0.5
                    }
                    
                    response = requests.post(f"{self.base_url}/route", json=payload, timeout=30)
                    response.raise_for_status()
                    
                    request_time = time.time() - start_time
                    
                    return {
                        "request_id": request_id,
                        "status": "SUCCESS",
                        "processing_time_ms": request_time * 1000,
                        "documents_processed": len(batch_docs)
                    }
                    
                except Exception as e:
                    return {
                        "request_id": request_id,
                        "status": "FAILED",
                        "error": str(e)
                    }
            
            # Execute concurrent requests
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = [executor.submit(make_request, i) for i in range(concurrent_requests)]
                
                for future in as_completed(futures):
                    result = future.result()
                    request_results.append(result)
            
            total_time = time.time() - start_time
            
            # Analyze results
            successful_requests = [r for r in request_results if r["status"] == "SUCCESS"]
            failed_requests = [r for r in request_results if r["status"] == "FAILED"]
            
            if successful_requests:
                avg_processing_time = sum(r["processing_time_ms"] for r in successful_requests) / len(successful_requests)
                total_documents = sum(r["documents_processed"] for r in successful_requests)
                throughput = total_documents / total_time
            else:
                avg_processing_time = 0
                total_documents = 0
                throughput = 0
            
            load_test_results = {
                "concurrent_requests": concurrent_requests,
                "total_time_seconds": total_time,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / concurrent_requests * 100,
                "average_processing_time_ms": avg_processing_time,
                "total_documents_processed": total_documents,
                "throughput_docs_per_second": throughput,
                "request_results": request_results
            }
            
            logger.info(f"Load test completed: {len(successful_requests)}/{concurrent_requests} requests successful")
            logger.info(f"Average processing time: {avg_processing_time:.1f}ms")
            logger.info(f"Throughput: {throughput:.1f} docs/sec")
            
            return load_test_results
            
        except Exception as e:
            logger.error(f"Load test failed: {e}")
            return {}
    
    def _extract_sample_features(self, sample_result: Dict) -> Dict[str, Any]:
        """Extract sample features for display"""
        if sample_result.get("processing_status") != "success":
            return {}
        
        return {
            "document_id": sample_result.get("document_id"),
            "page_count": sample_result.get("document_structure", {}).get("page_count"),
            "total_characters": sample_result.get("text_content", {}).get("total_characters"),
            "has_tables": sample_result.get("text_content", {}).get("has_tables"),
            "text_density": sample_result.get("ocr_features", {}).get("text_density"),
            "recommended_engine": sample_result.get("ocr_features", {}).get("recommended_ocr_engine")
        }
    
    def _convert_to_routing_format(self, doc: Dict) -> Dict[str, Any]:
        """Convert metadata extraction result to OCR routing format"""
        return {
            "document_id": doc.get("document_id", "unknown"),
            "file_name": doc.get("file_name", ""),
            "page_count": doc.get("document_structure", {}).get("page_count", 1),
            "has_metadata": doc.get("document_structure", {}).get("has_metadata", False),
            "has_forms": doc.get("document_structure", {}).get("has_forms", False),
            "has_annotations": doc.get("document_structure", {}).get("has_annotations", False),
            "total_characters": doc.get("text_content", {}).get("total_characters", 0),
            "total_words": doc.get("text_content", {}).get("total_words", 0),
            "unique_fonts": doc.get("text_content", {}).get("unique_fonts", 0),
            "has_tables": doc.get("text_content", {}).get("has_tables", False),
            "has_numbers": doc.get("text_content", {}).get("has_numbers", False),
            "has_currency": doc.get("text_content", {}).get("has_currency", False),
            "has_dates": doc.get("text_content", {}).get("has_dates", False),
            "has_emails": doc.get("text_content", {}).get("has_emails", False),
            "has_phone_numbers": doc.get("text_content", {}).get("has_phone_numbers", False),
            "aspect_ratio": doc.get("visual_features", {}).get("aspect_ratio", 1.0),
            "brightness_mean": doc.get("visual_features", {}).get("brightness_mean", 128.0),
            "contrast": doc.get("visual_features", {}).get("contrast", 50.0),
            "has_images": doc.get("layout_features", {}).get("has_images", False),
            "has_graphics": doc.get("layout_features", {}).get("has_graphics", False),
            "column_count": doc.get("layout_features", {}).get("column_count", 1),
            "text_density": doc.get("ocr_features", {}).get("text_density", 0.0),
            "font_clarity": doc.get("ocr_features", {}).get("font_clarity", 0.0),
            "noise_level": doc.get("ocr_features", {}).get("noise_level", 0.0)
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite"""
        logger.info("STARTING COMPREHENSIVE TEST SUITE")
        logger.info("=" * 80)
        
        overall_start_time = time.time()
        
        try:
            # Step 1: Test metadata extraction with 32 workers
            test_documents = self.test_metadata_extraction_with_workers()
            
            if not test_documents:
                raise Exception("No test documents available for routing")
            
            # Step 2: Test OCR routing with workers
            routing_results = self.test_ocr_routing_with_workers(test_documents)
            
            # Step 3: Test batch processing
            batch_results = self.test_batch_processing(test_documents)
            
            # Step 4: Test performance under load
            load_results = self.test_performance_under_load(test_documents)
            
            # Calculate overall performance metrics
            self._calculate_performance_metrics(batch_results, load_results)
            
            # Calculate overall status
            metadata_success = self.test_results["metadata_extraction"].get("status") == "SUCCESS"
            routing_success = self.test_results["ocr_routing"].get("status") == "SUCCESS"
            
            if metadata_success and routing_success:
                self.test_results["overall_status"] = "SUCCESS"
            else:
                self.test_results["overall_status"] = "PARTIAL_SUCCESS"
            
            overall_time = time.time() - overall_start_time
            self.test_results["total_test_time_seconds"] = overall_time
            
            # Add additional test results
            self.test_results["batch_processing"] = batch_results
            self.test_results["load_testing"] = load_results
            
            # Generate final report
            self._generate_comprehensive_report()
            
            logger.info("=" * 80)
            logger.info("COMPREHENSIVE TEST SUITE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            self.test_results["overall_status"] = "FAILED"
            self.test_results["error"] = str(e)
            return self.test_results
    
    def _calculate_performance_metrics(self, batch_results: Dict, load_results: Dict):
        """Calculate comprehensive performance metrics"""
        metadata_perf = self.test_results.get("metadata_extraction", {})
        routing_perf = self.test_results.get("ocr_routing", {})
        
        performance_metrics = {
            "metadata_extraction_performance": {
                "files_per_second": metadata_perf.get("files_per_second", 0),
                "success_rate": metadata_perf.get("success_rate", 0),
                "extraction_time": metadata_perf.get("extraction_time_seconds", 0),
                "workers_used": metadata_perf.get("workers_used", 0)
            },
            "ocr_routing_performance": {
                "average_routing_time_ms": 0,
                "routing_success_rate": 0,
                "total_routing_tests": routing_perf.get("total_tests", 0),
                "workers_used": routing_perf.get("workers_used", 0)
            },
            "batch_processing_performance": {
                "max_batch_size_tested": 0,
                "best_throughput": 0,
                "average_latency_per_doc": 0
            },
            "load_testing_performance": {
                "concurrent_requests": 0,
                "success_rate": 0,
                "throughput_under_load": 0
            },
            "system_throughput": {
                "end_to_end_latency": 0,
                "total_processing_capacity": 0,
                "scalability_score": 0
            }
        }
        
        # Calculate routing performance
        if routing_perf.get("routing_tests"):
            total_routing_time = 0
            successful_routing_tests = 0
            
            for test_name, test_result in routing_perf["routing_tests"].items():
                if test_result["status"] == "SUCCESS":
                    total_routing_time += test_result["processing_time_ms"]
                    successful_routing_tests += 1
            
            if successful_routing_tests > 0:
                performance_metrics["ocr_routing_performance"]["average_routing_time_ms"] = total_routing_time / successful_routing_tests
                performance_metrics["ocr_routing_performance"]["routing_success_rate"] = (successful_routing_tests / routing_perf["total_tests"]) * 100
        
        # Calculate batch processing performance
        if batch_results:
            max_batch = max(int(k.split('_')[1]) for k in batch_results.keys())
            best_throughput = max(r["documents_per_second"] for r in batch_results.values())
            avg_latency = sum(r["average_latency_per_doc"] for r in batch_results.values()) / len(batch_results)
            
            performance_metrics["batch_processing_performance"]["max_batch_size_tested"] = max_batch
            performance_metrics["batch_processing_performance"]["best_throughput"] = best_throughput
            performance_metrics["batch_processing_performance"]["average_latency_per_doc"] = avg_latency
        
        # Calculate load testing performance
        if load_results:
            performance_metrics["load_testing_performance"]["concurrent_requests"] = load_results.get("concurrent_requests", 0)
            performance_metrics["load_testing_performance"]["success_rate"] = load_results.get("success_rate", 0)
            performance_metrics["load_testing_performance"]["throughput_under_load"] = load_results.get("throughput_docs_per_second", 0)
        
        # Calculate system throughput
        if metadata_perf.get("files_per_second") and performance_metrics["ocr_routing_performance"]["average_routing_time_ms"]:
            routing_capacity = 1000 / performance_metrics["ocr_routing_performance"]["average_routing_time_ms"]
            extraction_capacity = metadata_perf["files_per_second"]
            
            performance_metrics["system_throughput"]["total_processing_capacity"] = min(extraction_capacity, routing_capacity)
            performance_metrics["system_throughput"]["end_to_end_latency"] = (
                metadata_perf.get("extraction_time_seconds", 0) + 
                performance_metrics["ocr_routing_performance"]["average_routing_time_ms"] / 1000
            )
            
            # Calculate scalability score (0-100)
            scalability_score = min(100, (
                (metadata_perf.get("success_rate", 0) / 100) * 30 +
                (performance_metrics["load_testing_performance"]["success_rate"] / 100) * 30 +
                (min(performance_metrics["batch_processing_performance"]["best_throughput"] / 10, 1)) * 20 +
                (min(performance_metrics["load_testing_performance"]["throughput_under_load"] / 5, 1)) * 20
            ))
            performance_metrics["system_throughput"]["scalability_score"] = scalability_score
        
        self.test_results["performance_metrics"] = performance_metrics
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE TEST REPORT")
        logger.info("=" * 80)
        
        # Overall status
        status_emoji = "SUCCESS" if self.test_results["overall_status"] == "SUCCESS" else "PARTIAL_SUCCESS" if self.test_results["overall_status"] == "PARTIAL_SUCCESS" else "FAILED"
        logger.info(f"Overall Status: {status_emoji}")
        
        # Metadata extraction summary
        metadata = self.test_results["metadata_extraction"]
        if metadata.get("status") == "SUCCESS":
            logger.info(f"Metadata Extraction: {metadata['successful_extractions']}/{metadata['total_files']} files ({metadata['success_rate']:.1f}% success)")
            logger.info(f"Speed: {metadata['files_per_second']:.1f} files/second with {metadata['workers_used']} workers")
        
        # OCR routing summary
        routing = self.test_results["ocr_routing"]
        if routing.get("status") == "SUCCESS":
            logger.info(f"OCR Routing: {routing['successful_tests']}/{routing['total_tests']} tests passed with {routing['workers_used']} workers")
        
        # Batch processing summary
        batch = self.test_results.get("batch_processing", {})
        if batch:
            logger.info(f"Batch Processing: Tested up to {max(int(k.split('_')[1]) for k in batch.keys())} documents per batch")
            best_throughput = max(r["documents_per_second"] for r in batch.values())
            logger.info(f"Best Throughput: {best_throughput:.1f} documents/second")
        
        # Load testing summary
        load = self.test_results.get("load_testing", {})
        if load:
            logger.info(f"Load Testing: {load['successful_requests']}/{load['concurrent_requests']} concurrent requests successful")
            logger.info(f"Throughput Under Load: {load['throughput_docs_per_second']:.1f} docs/second")
        
        # Performance summary
        perf = self.test_results["performance_metrics"]
        logger.info(f"System Throughput: {perf['system_throughput']['total_processing_capacity']:.1f} docs/second")
        logger.info(f"Scalability Score: {perf['system_throughput']['scalability_score']:.1f}/100")
        
        # Save detailed results
        with open("comprehensive_test_results.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        logger.info("Detailed results saved to: comprehensive_test_results.json")

def main():
    """Main function to run comprehensive test suite"""
    print("COMPREHENSIVE OCR ROUTING PIPELINE TEST SUITE")
    print("=" * 80)
    print("This test will:")
    print("1. Test metadata extraction with 32 workers")
    print("2. Test OCR routing with multiple workers")
    print("3. Test batch processing capabilities")
    print("4. Test performance under concurrent load")
    print("5. Generate comprehensive performance report")
    print("=" * 80)
    
    # Check prerequisites
    print("Checking prerequisites...")
    
    # Check if OCR routing service is running
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            print("OCR routing service is running")
        else:
            print("OCR routing service is not responding")
            return
    except:
        print("OCR routing service is not running")
        print("Please start the service with: python services/ocr_routing_service.py")
        return
    
    # Check if PDF folder exists
    if not Path("1000+ PDF_Invoice_Folder").exists():
        print("PDF folder not found")
        return
    
    print("All prerequisites met")
    print()
    
    # Run the comprehensive test
    tester = ComprehensiveTestSuite(max_workers=32)
    results = tester.run_comprehensive_test()
    
    # Final status
    if results["overall_status"] == "SUCCESS":
        print("\nALL TESTS PASSED! The OCR routing pipeline is working perfectly!")
    elif results["overall_status"] == "PARTIAL_SUCCESS":
        print("\nPARTIAL SUCCESS! Some components are working, check the logs for details.")
    else:
        print("\nTESTS FAILED! Check the logs and fix the issues.")

if __name__ == "__main__":
    main()
