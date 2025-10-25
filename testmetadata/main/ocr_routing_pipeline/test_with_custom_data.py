#!/usr/bin/env python3
"""
Test with Custom Data - No Code Changes Required
Simply place your documents in the test_data/ directory and run this script
"""

import json
import time
import requests
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomDataTester:
    """Test the OCR routing pipeline with your custom data"""
    
    def __init__(self, max_workers: int = 32):
        self.max_workers = max_workers
        self.base_url = "http://localhost:8002"
        self.test_data_dir = Path("test_data")
        self.results = {
            "metadata_extraction": {},
            "ocr_routing": {},
            "performance_metrics": {},
            "overall_status": "PENDING"
        }
    
    def discover_test_files(self) -> List[Path]:
        """Discover all supported files in test_data directory"""
        logger.info("Discovering files in test_data directory...")
        
        if not self.test_data_dir.exists():
            logger.error("test_data directory not found! Please create it and add your files.")
            return []
        
        # Supported file extensions
        supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        
        all_files = []
        for ext in supported_extensions:
            files = list(self.test_data_dir.glob(f"*{ext}"))
            all_files.extend(files)
        
        logger.info(f"Found {len(all_files)} supported files:")
        for file in all_files:
            logger.info(f"  - {file.name}")
        
        return all_files
    
    def extract_metadata_from_files(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Extract metadata from all files using the ultra-fast extractor"""
        logger.info("=" * 80)
        logger.info("EXTRACTING METADATA FROM YOUR CUSTOM FILES")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Import the ultra-fast metadata extractor
            from core.ultra_fast_metadata_extractor import UltraFastMetadataExtractor
            
            # Initialize with 32 workers
            extractor = UltraFastMetadataExtractor(max_workers=self.max_workers)
            
            # Process all files
            all_results = []
            successful = 0
            failed = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(extractor.extract_single_pdf_metadata, file): file 
                    for file in files
                }
                
                # Process results as they complete
                for i, future in enumerate(as_completed(future_to_file)):
                    file = future_to_file[future]
                    
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        if result.get("processing_status") == "success":
                            successful += 1
                        else:
                            failed += 1
                        
                        # Progress logging every 10 files
                        if (i + 1) % 10 == 0:
                            elapsed = time.time() - start_time
                            rate = (i + 1) / elapsed
                            eta = (len(files) - i - 1) / rate if rate > 0 else 0
                            logger.info(f"Processed {i+1}/{len(files)} files. Rate: {rate:.1f} files/sec. ETA: {eta:.1f}s")
                    
                    except Exception as e:
                        logger.error(f"Error processing {file}: {e}")
                        failed += 1
                        all_results.append({
                            "document_id": file.stem,
                            "file_name": file.name,
                            "file_path": str(file),
                            "processing_status": "error",
                            "error_message": str(e)
                        })
            
            extraction_time = time.time() - start_time
            
            # Calculate metrics
            self.results["metadata_extraction"] = {
                "status": "SUCCESS",
                "total_files": len(files),
                "successful_extractions": successful,
                "failed_extractions": failed,
                "success_rate": successful / len(files) * 100 if len(files) > 0 else 0,
                "extraction_time_seconds": extraction_time,
                "files_per_second": len(files) / extraction_time if extraction_time > 0 else 0,
                "workers_used": self.max_workers
            }
            
            logger.info(f"Metadata extraction completed: {successful}/{len(files)} successful")
            logger.info(f"Processing time: {extraction_time:.2f}s ({len(files)/extraction_time:.1f} files/sec)")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            self.results["metadata_extraction"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return []
    
    def test_ocr_routing(self, test_documents: List[Dict]) -> Dict[str, Any]:
        """Test OCR routing with the extracted documents"""
        logger.info("=" * 80)
        logger.info("TESTING OCR ROUTING WITH YOUR DOCUMENTS")
        logger.info("=" * 80)
        
        try:
            # Check if service is running
            health_response = requests.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code != 200:
                raise Exception("OCR routing service is not running. Please start it first.")
            
            logger.info("OCR routing service is healthy")
            
            # Convert documents to routing format
            routing_docs = []
            for doc in test_documents:
                if doc.get("processing_status") == "success":
                    routing_doc = self._convert_to_routing_format(doc)
                    routing_docs.append(routing_doc)
            
            if not routing_docs:
                logger.warning("No valid documents for OCR routing")
                return {}
            
            logger.info(f"Testing OCR routing with {len(routing_docs)} documents...")
            
            # Make routing request
            payload = {
                "document_features": routing_docs,
                "alpha": 1.0,
                "beta": 0.5,
                "gamma": 0.5,
                "delta_fallback_threshold": 0.03
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/route", json=payload, timeout=60)
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
            
            routing_results = {
                "status": "SUCCESS",
                "processing_time_ms": routing_time * 1000,
                "documents_processed": len(result["documents"]),
                "engine_distribution": engine_distribution,
                "average_confidence": total_confidence / len(result["documents"]) if result["documents"] else 0,
                "average_latency": total_latency / len(result["documents"]) if result["documents"] else 0,
                "features_used": len(result["features_used"]),
                "sample_routing": result["documents"][0] if result["documents"] else {}
            }
            
            self.results["ocr_routing"] = {
                "status": "SUCCESS",
                "service_health": "HEALTHY",
                "routing_results": routing_results,
                "total_documents": len(routing_docs),
                "successful_routing": len(result["documents"])
            }
            
            logger.info(f"OCR routing completed: {len(result['documents'])} documents processed")
            logger.info(f"Processing time: {routing_time*1000:.1f}ms")
            logger.info(f"Engine distribution: {engine_distribution}")
            
            return routing_results
            
        except Exception as e:
            logger.error(f"OCR routing test failed: {e}")
            self.results["ocr_routing"] = {
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
    
    def run_complete_test(self) -> Dict[str, Any]:
        """Run the complete test with your custom data"""
        logger.info("STARTING CUSTOM DATA TEST")
        logger.info("=" * 80)
        
        overall_start_time = time.time()
        
        try:
            # Step 1: Discover files
            test_files = self.discover_test_files()
            
            if not test_files:
                logger.error("No test files found in test_data directory!")
                self.results["overall_status"] = "FAILED"
                return self.results
            
            # Step 2: Extract metadata
            test_documents = self.extract_metadata_from_files(test_files)
            
            if not test_documents:
                logger.error("No documents could be processed!")
                self.results["overall_status"] = "FAILED"
                return self.results
            
            # Step 3: Test OCR routing
            routing_results = self.test_ocr_routing(test_documents)
            
            # Calculate overall performance metrics
            self._calculate_performance_metrics()
            
            # Calculate overall status
            metadata_success = self.results["metadata_extraction"].get("status") == "SUCCESS"
            routing_success = self.results["ocr_routing"].get("status") == "SUCCESS"
            
            if metadata_success and routing_success:
                self.results["overall_status"] = "SUCCESS"
            else:
                self.results["overall_status"] = "PARTIAL_SUCCESS"
            
            overall_time = time.time() - overall_start_time
            self.results["total_test_time_seconds"] = overall_time
            
            # Save results
            self._save_results()
            
            # Generate report
            self._generate_report()
            
            logger.info("=" * 80)
            logger.info("CUSTOM DATA TEST COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Custom data test failed: {e}")
            self.results["overall_status"] = "FAILED"
            self.results["error"] = str(e)
            return self.results
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics"""
        metadata_perf = self.results.get("metadata_extraction", {})
        routing_perf = self.results.get("ocr_routing", {})
        
        self.results["performance_metrics"] = {
            "metadata_extraction_performance": {
                "files_per_second": metadata_perf.get("files_per_second", 0),
                "success_rate": metadata_perf.get("success_rate", 0),
                "extraction_time": metadata_perf.get("extraction_time_seconds", 0),
                "workers_used": metadata_perf.get("workers_used", 0)
            },
            "ocr_routing_performance": {
                "average_routing_time_ms": routing_perf.get("routing_results", {}).get("processing_time_ms", 0),
                "routing_success_rate": 100 if routing_perf.get("status") == "SUCCESS" else 0,
                "documents_processed": routing_perf.get("total_documents", 0),
                "engine_distribution": routing_perf.get("routing_results", {}).get("engine_distribution", {})
            },
            "system_throughput": {
                "end_to_end_latency": self.results.get("total_test_time_seconds", 0),
                "total_processing_capacity": metadata_perf.get("files_per_second", 0),
                "scalability_score": 100 if self.results["overall_status"] == "SUCCESS" else 50
            }
        }
    
    def _save_results(self):
        """Save results to JSON file"""
        try:
            with open("test_data_results.json", "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info("Results saved to: test_data_results.json")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _generate_report(self):
        """Generate test report"""
        logger.info("=" * 80)
        logger.info("CUSTOM DATA TEST REPORT")
        logger.info("=" * 80)
        
        # Overall status
        status_emoji = "SUCCESS" if self.results["overall_status"] == "SUCCESS" else "PARTIAL_SUCCESS" if self.results["overall_status"] == "PARTIAL_SUCCESS" else "FAILED"
        logger.info(f"Overall Status: {status_emoji}")
        
        # Metadata extraction summary
        metadata = self.results["metadata_extraction"]
        if metadata.get("status") == "SUCCESS":
            logger.info(f"Metadata Extraction: {metadata['successful_extractions']}/{metadata['total_files']} files ({metadata['success_rate']:.1f}% success)")
            logger.info(f"Speed: {metadata['files_per_second']:.1f} files/second with {metadata['workers_used']} workers")
        
        # OCR routing summary
        routing = self.results["ocr_routing"]
        if routing.get("status") == "SUCCESS":
            logger.info(f"OCR Routing: {routing['successful_routing']}/{routing['total_documents']} documents processed")
            engine_dist = routing.get("routing_results", {}).get("engine_distribution", {})
            logger.info(f"Engine Distribution: {engine_dist}")
        
        # Performance summary
        perf = self.results["performance_metrics"]
        logger.info(f"System Throughput: {perf['system_throughput']['total_processing_capacity']:.1f} docs/second")
        logger.info(f"Scalability Score: {perf['system_throughput']['scalability_score']:.1f}/100")

def main():
    """Main function to run custom data test"""
    print("CUSTOM DATA TEST - OCR ROUTING PIPELINE")
    print("=" * 80)
    print("This will test your documents in the test_data/ directory")
    print("No code changes required - just place your files and run!")
    print("=" * 80)
    
    # Check if test_data directory exists
    if not Path("test_data").exists():
        print("Error: test_data directory not found!")
        print("Please create the test_data directory and add your files.")
        return
    
    # Check if OCR routing service is running
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            print("OCR routing service is running")
        else:
            print("OCR routing service is not responding")
            print("Please start the service with: python services/ocr_routing_service.py")
            return
    except:
        print("OCR routing service is not running")
        print("Please start the service with: python services/ocr_routing_service.py")
        return
    
    # Run the test
    tester = CustomDataTester(max_workers=32)
    results = tester.run_complete_test()
    
    # Final status
    if results["overall_status"] == "SUCCESS":
        print("\nALL TESTS PASSED! Your documents were processed successfully!")
    elif results["overall_status"] == "PARTIAL_SUCCESS":
        print("\nPARTIAL SUCCESS! Some components worked, check the logs for details.")
    else:
        print("\nTESTS FAILED! Check the logs and fix the issues.")

if __name__ == "__main__":
    main()
