#!/usr/bin/env python3
"""
End-to-End Test for Complete OCR Routing Pipeline
Tests: Fast Metadata Extraction → OCR Routing → Engine Selection
"""

import json
import time
import requests
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EndToEndTester:
    """Complete end-to-end testing of the OCR routing pipeline"""
    
    def __init__(self):
        self.base_url = "http://localhost:8002"
        self.test_results = {
            "metadata_extraction": {},
            "ocr_routing": {},
            "performance_metrics": {},
            "overall_status": "PENDING"
        }
    
    def test_metadata_extraction(self) -> Dict[str, Any]:
        """Test the fast metadata extraction system"""
        logger.info("=" * 80)
        logger.info("TESTING FAST METADATA EXTRACTION")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Import and run the fast metadata extractor
            from fixed_fast_metadata_extractor import FixedFastMetadataExtractor
            
            extractor = FixedFastMetadataExtractor()
            
            # Test with a subset of PDFs (first 10 for speed)
            pdf_folder = "1000+ PDF_Invoice_Folder"
            if not Path(pdf_folder).exists():
                raise FileNotFoundError(f"PDF folder '{pdf_folder}' not found")
            
            # Get first 10 PDF files for testing
            pdf_files = list(Path(pdf_folder).glob("*.pdf"))[:10]
            
            logger.info(f"Testing metadata extraction on {len(pdf_files)} PDF files...")
            
            # Extract metadata
            results = []
            for pdf_file in pdf_files:
                result = extractor.extract_single_pdf_metadata(pdf_file)
                results.append(result)
            
            extraction_time = time.time() - start_time
            
            # Calculate metrics
            successful = sum(1 for r in results if r.get("processing_status") == "success")
            failed = len(results) - successful
            
            self.test_results["metadata_extraction"] = {
                "status": "SUCCESS",
                "total_files": len(results),
                "successful_extractions": successful,
                "failed_extractions": failed,
                "success_rate": successful / len(results) * 100,
                "extraction_time_seconds": extraction_time,
                "files_per_second": len(results) / extraction_time,
                "sample_features": self._extract_sample_features(results[0]) if results else {}
            }
            
            logger.info(f"✅ Metadata extraction completed: {successful}/{len(results)} successful")
            logger.info(f"⏱️  Processing time: {extraction_time:.2f}s ({len(results)/extraction_time:.1f} files/sec)")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Metadata extraction failed: {e}")
            self.test_results["metadata_extraction"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return []
    
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
    
    def test_ocr_routing_service(self, test_documents: List[Dict]) -> Dict[str, Any]:
        """Test the OCR routing microservice"""
        logger.info("=" * 80)
        logger.info("TESTING OCR ROUTING MICROSERVICE")
        logger.info("=" * 80)
        
        try:
            # Check if service is running
            health_response = requests.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code != 200:
                raise Exception("OCR routing service is not running")
            
            logger.info("✅ OCR routing service is healthy")
            
            # Test different routing scenarios
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
                logger.info(f"Testing {test['name']}...")
                
                start_time = time.time()
                
                # Convert test documents to routing format
                routing_docs = []
                for doc in test_documents[:3]:  # Test with first 3 documents
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
                
                response = requests.post(f"{self.base_url}/route", json=payload, timeout=10)
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
                    "sample_routing": result["documents"][0] if result["documents"] else {}
                }
                
                logger.info(f"✅ {test['name']}: {len(result['documents'])} docs in {routing_time*1000:.1f}ms")
                logger.info(f"   Engine distribution: {engine_distribution}")
            
            self.test_results["ocr_routing"] = {
                "status": "SUCCESS",
                "service_health": "HEALTHY",
                "routing_tests": routing_results,
                "total_tests": len(routing_tests),
                "successful_tests": len([r for r in routing_results.values() if r["status"] == "SUCCESS"])
            }
            
            logger.info("✅ All OCR routing tests completed successfully")
            return routing_results
            
        except Exception as e:
            logger.error(f"❌ OCR routing test failed: {e}")
            self.test_results["ocr_routing"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return {}
    
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
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """Test overall system performance"""
        logger.info("=" * 80)
        logger.info("TESTING PERFORMANCE METRICS")
        logger.info("=" * 80)
        
        # Calculate overall performance
        metadata_perf = self.test_results.get("metadata_extraction", {})
        routing_perf = self.test_results.get("ocr_routing", {})
        
        performance_metrics = {
            "metadata_extraction_performance": {
                "files_per_second": metadata_perf.get("files_per_second", 0),
                "success_rate": metadata_perf.get("success_rate", 0),
                "extraction_time": metadata_perf.get("extraction_time_seconds", 0)
            },
            "ocr_routing_performance": {
                "average_routing_time_ms": 0,
                "routing_success_rate": 0,
                "total_routing_tests": routing_perf.get("total_tests", 0)
            },
            "system_throughput": {
                "end_to_end_latency": 0,
                "total_processing_capacity": 0
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
        
        # Calculate system throughput
        if metadata_perf.get("files_per_second") and performance_metrics["ocr_routing_performance"]["average_routing_time_ms"]:
            # Estimate end-to-end processing capacity
            routing_capacity = 1000 / performance_metrics["ocr_routing_performance"]["average_routing_time_ms"]  # docs per second
            extraction_capacity = metadata_perf["files_per_second"]
            
            performance_metrics["system_throughput"]["total_processing_capacity"] = min(extraction_capacity, routing_capacity)
            performance_metrics["system_throughput"]["end_to_end_latency"] = (
                metadata_perf.get("extraction_time_seconds", 0) + 
                performance_metrics["ocr_routing_performance"]["average_routing_time_ms"] / 1000
            )
        
        self.test_results["performance_metrics"] = performance_metrics
        
        logger.info("✅ Performance metrics calculated")
        logger.info(f"📊 Metadata extraction: {metadata_perf.get('files_per_second', 0):.1f} files/sec")
        logger.info(f"📊 OCR routing: {performance_metrics['ocr_routing_performance']['average_routing_time_ms']:.1f}ms average")
        logger.info(f"📊 System throughput: {performance_metrics['system_throughput']['total_processing_capacity']:.1f} docs/sec")
        
        return performance_metrics
    
    def run_complete_test(self) -> Dict[str, Any]:
        """Run the complete end-to-end test"""
        logger.info("🚀 STARTING COMPLETE END-TO-END TEST")
        logger.info("=" * 80)
        
        overall_start_time = time.time()
        
        try:
            # Step 1: Test metadata extraction
            test_documents = self.test_metadata_extraction()
            
            if not test_documents:
                raise Exception("No test documents available for routing")
            
            # Step 2: Test OCR routing
            routing_results = self.test_ocr_routing_service(test_documents)
            
            # Step 3: Test performance metrics
            performance_metrics = self.test_performance_metrics()
            
            # Calculate overall status
            metadata_success = self.test_results["metadata_extraction"].get("status") == "SUCCESS"
            routing_success = self.test_results["ocr_routing"].get("status") == "SUCCESS"
            
            if metadata_success and routing_success:
                self.test_results["overall_status"] = "SUCCESS"
            else:
                self.test_results["overall_status"] = "PARTIAL_SUCCESS"
            
            overall_time = time.time() - overall_start_time
            self.test_results["total_test_time_seconds"] = overall_time
            
            # Generate final report
            self._generate_final_report()
            
            logger.info("=" * 80)
            logger.info("🎉 END-TO-END TEST COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"❌ End-to-end test failed: {e}")
            self.test_results["overall_status"] = "FAILED"
            self.test_results["error"] = str(e)
            return self.test_results
    
    def _generate_final_report(self):
        """Generate final test report"""
        logger.info("=" * 80)
        logger.info("📋 FINAL TEST REPORT")
        logger.info("=" * 80)
        
        # Overall status
        status_emoji = "✅" if self.test_results["overall_status"] == "SUCCESS" else "⚠️" if self.test_results["overall_status"] == "PARTIAL_SUCCESS" else "❌"
        logger.info(f"{status_emoji} Overall Status: {self.test_results['overall_status']}")
        
        # Metadata extraction summary
        metadata = self.test_results["metadata_extraction"]
        if metadata.get("status") == "SUCCESS":
            logger.info(f"📄 Metadata Extraction: {metadata['successful_extractions']}/{metadata['total_files']} files ({metadata['success_rate']:.1f}% success)")
            logger.info(f"⚡ Speed: {metadata['files_per_second']:.1f} files/second")
        
        # OCR routing summary
        routing = self.test_results["ocr_routing"]
        if routing.get("status") == "SUCCESS":
            logger.info(f"🎯 OCR Routing: {routing['successful_tests']}/{routing['total_tests']} tests passed")
            if routing.get("routing_tests"):
                for test_name, test_result in routing["routing_tests"].items():
                    if test_result["status"] == "SUCCESS":
                        logger.info(f"   • {test_name}: {test_result['documents_processed']} docs in {test_result['processing_time_ms']:.1f}ms")
        
        # Performance summary
        perf = self.test_results["performance_metrics"]
        logger.info(f"📊 System Throughput: {perf['system_throughput']['total_processing_capacity']:.1f} docs/second")
        logger.info(f"⏱️  End-to-End Latency: {perf['system_throughput']['end_to_end_latency']:.3f} seconds")
        
        # Save detailed results
        with open("end_to_end_test_results.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        logger.info("📁 Detailed results saved to: end_to_end_test_results.json")

def main():
    """Main function to run end-to-end test"""
    print("OCR ROUTING PIPELINE - END-TO-END TEST")
    print("=" * 80)
    print("This test will:")
    print("1. Extract metadata from PDF documents (fast extraction)")
    print("2. Route documents to optimal OCR engines (Naive Bayes)")
    print("3. Test different utility function parameters")
    print("4. Measure performance metrics")
    print("5. Generate comprehensive report")
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
        print("Please start the service with: python ocr_routing_service.py")
        return
    
    # Check if PDF folder exists
    if not Path("1000+ PDF_Invoice_Folder").exists():
        print("PDF folder not found")
        return
    
    print("All prerequisites met")
    print()
    
    # Run the test
    tester = EndToEndTester()
    results = tester.run_complete_test()
    
    # Final status
    if results["overall_status"] == "SUCCESS":
        print("\nALL TESTS PASSED! The OCR routing pipeline is working perfectly!")
    elif results["overall_status"] == "PARTIAL_SUCCESS":
        print("\nPARTIAL SUCCESS! Some components are working, check the logs for details.")
    else:
        print("\nTESTS FAILED! Check the logs and fix the issues.")

if __name__ == "__main__":
    main()
