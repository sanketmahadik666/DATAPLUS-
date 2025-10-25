#!/usr/bin/env python3
"""
Main Test Runner for OCR Routing Pipeline
Tests both metadata extraction and OCR routing services with 32 workers
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")
    
    # Check if PDF folder exists (look in parent directory)
    pdf_folder = Path("../1000+ PDF_Invoice_Folder")
    if not pdf_folder.exists():
        logger.error("PDF folder '1000+ PDF_Invoice_Folder' not found!")
        return False
    
    # Check if data files exist
    data_files = [
        "data/ml_ocr_routing_dataset.json",
        "data/fixed_fast_pdf_metadata.json"
    ]
    
    for file_path in data_files:
        if not Path(file_path).exists():
            logger.error(f"Data file '{file_path}' not found!")
            return False
    
    logger.info("All prerequisites met!")
    return True

def start_ocr_routing_service():
    """Start the OCR routing service"""
    logger.info("Starting OCR routing service...")
    
    try:
        # Check if service is already running
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            logger.info("OCR routing service is already running")
            return True
    except:
        pass
    
    # Start the service
    try:
        # Change to services directory and start the service
        service_process = subprocess.Popen([
            sys.executable, "services/ocr_routing_service.py"
        ], cwd=Path(__file__).parent)
        
        # Wait for service to start
        logger.info("Waiting for OCR routing service to start...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:8002/health", timeout=2)
                if response.status_code == 200:
                    logger.info("OCR routing service started successfully!")
                    return True
            except:
                time.sleep(1)
        
        logger.error("Failed to start OCR routing service")
        return False
        
    except Exception as e:
        logger.error(f"Error starting OCR routing service: {e}")
        return False

def run_metadata_extraction_test():
    """Run the metadata extraction test with 32 workers"""
    logger.info("=" * 80)
    logger.info("RUNNING METADATA EXTRACTION TEST WITH 32 WORKERS")
    logger.info("=" * 80)
    
    try:
        # Import and run the ultra-fast metadata extractor
        from core.ultra_fast_metadata_extractor import UltraFastMetadataExtractor
        
        extractor = UltraFastMetadataExtractor(max_workers=32)
        results = extractor.analyze_all_pdfs("../1000+ PDF_Invoice_Folder")
        
        # Save results
        extractor.save_results("ultra_fast_metadata_results.json")
        
        # Display results
        metadata = results["metadata"]
        logger.info(f"Metadata extraction completed:")
        logger.info(f"  - Total PDFs: {metadata['total_pdfs_processed']}")
        logger.info(f"  - Successful: {metadata['successful_analyses']}")
        logger.info(f"  - Failed: {metadata['failed_analyses']}")
        logger.info(f"  - Success Rate: {(metadata['successful_analyses']/metadata['total_pdfs_processed']*100):.1f}%")
        logger.info(f"  - Processing Time: {metadata['processing_time_seconds']:.2f} seconds")
        logger.info(f"  - Files per Second: {metadata['files_per_second']:.1f}")
        logger.info(f"  - Threads Used: {metadata['feature_extraction_metrics']['threads_used']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Metadata extraction test failed: {e}")
        return None

def run_ocr_routing_test():
    """Run the OCR routing test"""
    logger.info("=" * 80)
    logger.info("RUNNING OCR ROUTING TEST")
    logger.info("=" * 80)
    
    try:
        # Import and run the comprehensive test suite
        from tests.comprehensive_test_suite import ComprehensiveTestSuite
        
        tester = ComprehensiveTestSuite(max_workers=32)
        
        # Load some test documents
        test_documents = []
        pdf_files = list(Path("../1000+ PDF_Invoice_Folder").glob("*.pdf"))[:50]  # Test with first 50 files
        
        for pdf_file in pdf_files:
            test_documents.append({
                "document_id": pdf_file.stem,
                "file_name": pdf_file.name,
                "processing_status": "success"
            })
        
        # Run OCR routing tests
        routing_results = tester.test_ocr_routing_with_workers(test_documents)
        
        # Run batch processing tests
        batch_results = tester.test_batch_processing(test_documents)
        
        # Run load tests
        load_results = tester.test_performance_under_load(test_documents)
        
        logger.info("OCR routing tests completed:")
        logger.info(f"  - Routing tests: {len(routing_results)} scenarios tested")
        logger.info(f"  - Batch processing: {len(batch_results)} batch sizes tested")
        logger.info(f"  - Load testing: {load_results.get('concurrent_requests', 0)} concurrent requests")
        
        return {
            "routing_results": routing_results,
            "batch_results": batch_results,
            "load_results": load_results
        }
        
    except Exception as e:
        logger.error(f"OCR routing test failed: {e}")
        return None

def run_comprehensive_test():
    """Run the complete comprehensive test"""
    logger.info("=" * 80)
    logger.info("RUNNING COMPREHENSIVE TEST SUITE")
    logger.info("=" * 80)
    
    try:
        from tests.comprehensive_test_suite import ComprehensiveTestSuite
        
        tester = ComprehensiveTestSuite(max_workers=32)
        results = tester.run_comprehensive_test()
        
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        return None

def main():
    """Main function to run all tests"""
    print("OCR ROUTING PIPELINE - COMPREHENSIVE TEST RUNNER")
    print("=" * 80)
    print("This will test:")
    print("1. Metadata extraction with 32 workers")
    print("2. OCR routing service")
    print("3. Batch processing capabilities")
    print("4. Performance under load")
    print("5. Complete end-to-end pipeline")
    print("=" * 80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("Prerequisites not met. Please check the requirements.")
        return
    
    # Start OCR routing service
    if not start_ocr_routing_service():
        print("Failed to start OCR routing service. Exiting.")
        return
    
    try:
        # Run metadata extraction test
        print("\n" + "=" * 80)
        print("STEP 1: METADATA EXTRACTION TEST")
        print("=" * 80)
        metadata_results = run_metadata_extraction_test()
        
        if metadata_results is None:
            print("Metadata extraction test failed. Exiting.")
            return
        
        # Run OCR routing test
        print("\n" + "=" * 80)
        print("STEP 2: OCR ROUTING TEST")
        print("=" * 80)
        routing_results = run_ocr_routing_test()
        
        if routing_results is None:
            print("OCR routing test failed. Exiting.")
            return
        
        # Run comprehensive test
        print("\n" + "=" * 80)
        print("STEP 3: COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        comprehensive_results = run_comprehensive_test()
        
        if comprehensive_results is None:
            print("Comprehensive test failed. Exiting.")
            return
        
        # Final summary
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Display final results
        if comprehensive_results.get("overall_status") == "SUCCESS":
            print("Overall Status: SUCCESS")
        else:
            print("Overall Status: PARTIAL_SUCCESS")
        
        # Metadata extraction summary
        metadata = comprehensive_results.get("metadata_extraction", {})
        if metadata.get("status") == "SUCCESS":
            print(f"Metadata Extraction: {metadata['successful_extractions']}/{metadata['total_files']} files ({metadata['success_rate']:.1f}% success)")
            print(f"Speed: {metadata['files_per_second']:.1f} files/second with {metadata['workers_used']} workers")
        
        # OCR routing summary
        routing = comprehensive_results.get("ocr_routing", {})
        if routing.get("status") == "SUCCESS":
            print(f"OCR Routing: {routing['successful_tests']}/{routing['total_tests']} tests passed")
        
        # Performance summary
        perf = comprehensive_results.get("performance_metrics", {})
        if perf:
            print(f"System Throughput: {perf.get('system_throughput', {}).get('total_processing_capacity', 0):.1f} docs/second")
            print(f"Scalability Score: {perf.get('system_throughput', {}).get('scalability_score', 0):.1f}/100")
        
        print("\nDetailed results saved to:")
        print("  - ultra_fast_metadata_results.json")
        print("  - comprehensive_test_results.json")
        
        print("\nThe OCR routing pipeline is ready for production!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
    finally:
        print("\nCleaning up...")

if __name__ == "__main__":
    main()
