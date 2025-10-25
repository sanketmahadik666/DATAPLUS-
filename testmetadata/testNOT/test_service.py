#!/usr/bin/env python3
"""
Test the document analyzer service with different document types
"""

import sys
from pathlib import Path

# Add the service directory to path
sys.path.insert(0, str(Path("document_analyzer_service")))

from service_analyzer import analyze_path
import json

def test_document_types():
    """Test the service with different document types from the directory structure."""
    
    print("Testing Document Analyzer Service")
    print("=" * 50)
    
    # Test 1: PDF Invoices folder
    print("\n1. Testing PDF Invoices folder...")
    try:
        pdf_results = analyze_path("1000+ PDF_Invoice_Folder")
        with open("pdf_analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(pdf_results, f, indent=2, ensure_ascii=False)
        print(f"✓ PDF analysis completed: {pdf_results['summary']['total_inputs']} files processed")
        print(f"  - Successful: {pdf_results['summary']['successful']}")
        print(f"  - Failed: {pdf_results['summary']['failed']}")
    except Exception as e:
        print(f"✗ PDF analysis failed: {e}")
    
    # Test 2: Dataset images folder
    print("\n2. Testing Dataset images folder...")
    try:
        image_results = analyze_path("dataset/testing_data/images")
        with open("image_analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(image_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Image analysis completed: {image_results['summary']['total_inputs']} files processed")
        print(f"  - Successful: {image_results['summary']['successful']}")
        print(f"  - Failed: {image_results['summary']['failed']}")
    except Exception as e:
        print(f"✗ Image analysis failed: {e}")
    
    # Test 3: Single PDF file
    print("\n3. Testing single PDF file...")
    try:
        # Find a single PDF file
        pdf_files = list(Path("1000+ PDF_Invoice_Folder").glob("*.pdf"))
        if pdf_files:
            single_pdf = pdf_files[0]
            single_results = analyze_path(str(single_pdf))
            with open("single_pdf_analysis_results.json", "w", encoding="utf-8") as f:
                json.dump(single_results, f, indent=2, ensure_ascii=False)
            print(f"✓ Single PDF analysis completed: {single_pdf.name}")
            print(f"  - Status: {single_results['results'][0].get('processing_status', 'unknown')}")
        else:
            print("✗ No PDF files found")
    except Exception as e:
        print(f"✗ Single PDF analysis failed: {e}")
    
    # Test 4: Single image file
    print("\n4. Testing single image file...")
    try:
        # Find a single image file
        image_files = list(Path("dataset/testing_data/images").glob("*.png"))
        if image_files:
            single_image = image_files[0]
            single_image_results = analyze_path(str(single_image))
            with open("single_image_analysis_results.json", "w", encoding="utf-8") as f:
                json.dump(single_image_results, f, indent=2, ensure_ascii=False)
            print(f"✓ Single image analysis completed: {single_image.name}")
            print(f"  - Status: {single_image_results['results'][0].get('processing_status', 'unknown')}")
        else:
            print("✗ No image files found")
    except Exception as e:
        print(f"✗ Single image analysis failed: {e}")
    
    # Test 5: Service files (Python, Markdown, etc.)
    print("\n5. Testing service files...")
    try:
        service_results = analyze_path("document_analyzer_service")
        with open("service_files_analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(service_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Service files analysis completed: {service_results['summary']['total_inputs']} files processed")
        print(f"  - Successful: {service_results['summary']['successful']}")
        print(f"  - Failed: {service_results['summary']['failed']}")
    except Exception as e:
        print(f"✗ Service files analysis failed: {e}")
    
    print("\n" + "=" * 50)
    print("Testing completed! Check the generated JSON files for detailed results.")
    print("Generated files:")
    print("  - pdf_analysis_results.json")
    print("  - image_analysis_results.json") 
    print("  - single_pdf_analysis_results.json")
    print("  - single_image_analysis_results.json")
    print("  - service_files_analysis_results.json")

if __name__ == "__main__":
    test_document_types()
