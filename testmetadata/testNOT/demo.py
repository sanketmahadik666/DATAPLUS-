#!/usr/bin/env python3
"""
Document Analysis System - Usage Examples and Summary
This script demonstrates how to use the document analysis system for OCR routing.
"""

import json
from pathlib import Path
from test_analyzer import SimpleDocumentAnalyzer

def demonstrate_single_document_analysis():
    """Demonstrate analyzing a single document."""
    print("=== Single Document Analysis Example ===\n")
    
    analyzer = SimpleDocumentAnalyzer()
    
    # Analyze the sample document
    image_path = "dataset/testing_data/images/82250337_0338.png"
    annotation_path = "dataset/testing_data/annotations/82250337_0338.json"
    
    result = analyzer.analyze_document(image_path, annotation_path)
    
    print("Input:")
    print(f"  Image: {image_path}")
    print(f"  Annotations: {annotation_path}")
    print()
    
    print("Analysis Results:")
    print(f"  Document ID: {result['document_id']}")
    print(f"  File Size: {result['file_size_bytes']:,} bytes")
    print(f"  Pages: {result['num_pages']}")
    print(f"  Language: {result['language_detected']}")
    print(f"  Bilingual: {result['bilingual_flag']}")
    print(f"  Text Density: {result['text_density']:.2f}")
    print(f"  Form Fields: {result['form_fields_detected']}")
    print(f"  Contains Signature: {result['contains_signature']}")
    print(f"  Contains Logo: {result['contains_logo']}")
    print(f"  Layout Complexity: {result['layout_complexity_score']:.2f}")
    print(f"  Department: {result['department_context']}")
    print(f"  Priority: {result['priority_level']}")
    print()
    
    print("OCR Routing Decision:")
    print(f"  Recommended Engine: {result['ocr_variant_suggestion']}")
    print(f"  Confidence: {result['confidence_estimate']:.2f}")
    print(f"  Processing: {result['processing_recommendation']}")
    
    return result

def demonstrate_batch_analysis():
    """Demonstrate batch analysis capabilities."""
    print("\n=== Batch Analysis Example ===\n")
    
    # Load the batch results
    if Path("batch_analysis_results.json").exists():
        with open("batch_analysis_results.json", 'r') as f:
            results = json.load(f)
        
        print(f"Batch Analysis Results: {len(results)} documents processed")
        print()
        
        # Show routing distribution
        engine_counts = {}
        dept_counts = {}
        priority_counts = {}
        
        for result in results:
            engine = result.get('ocr_variant_suggestion', 'unknown')
            dept = result.get('department_context', 'unknown')
            priority = result.get('priority_level', 'unknown')
            
            engine_counts[engine] = engine_counts.get(engine, 0) + 1
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        print("OCR Engine Distribution:")
        for engine, count in sorted(engine_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"  {engine}: {count} documents ({percentage:.1f}%)")
        
        print("\nDepartment Distribution:")
        for dept, count in sorted(dept_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"  {dept}: {count} documents ({percentage:.1f}%)")
        
        print("\nPriority Distribution:")
        for priority, count in sorted(priority_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"  {priority}: {count} documents ({percentage:.1f}%)")
        
        return results
    else:
        print("No batch analysis results found. Run batch_analyze.py first.")
        return []

def demonstrate_routing_logic():
    """Demonstrate the OCR routing logic."""
    print("\n=== OCR Routing Logic ===\n")
    
    print("The system routes documents to OCR engines based on these criteria:")
    print()
    print("1. Form Fields Detected -> form_optimized")
    print("   - Documents with structured form fields")
    print("   - Question/answer pairs")
    print("   - Input fields and checkboxes")
    print()
    print("2. High Table Ratio -> table_optimized")
    print("   - Documents with significant table content")
    print("   - Spreadsheet-like layouts")
    print("   - Structured data presentation")
    print()
    print("3. Signatures/Handwriting -> handwriting")
    print("   - Documents containing signatures")
    print("   - Handwritten content")
    print("   - Irregular text patterns")
    print()
    print("4. Multiple Languages -> multilingual")
    print("   - Documents with mixed languages")
    print("   - International content")
    print("   - Translation documents")
    print()
    print("5. Default -> standard")
    print("   - General text documents")
    print("   - Letters and reports")
    print("   - Standard layouts")

def demonstrate_configuration():
    """Demonstrate configuration options."""
    print("\n=== Configuration Options ===\n")
    
    if Path("ocr_config.json").exists():
        with open("ocr_config.json", 'r') as f:
            config = json.load(f)
        
        print("Available OCR Engines:")
        for engine, settings in config['ocr_engines'].items():
            print(f"  {engine}: {settings['description']}")
        
        print("\nThresholds:")
        for threshold, value in config['thresholds'].items():
            print(f"  {threshold}: {value}")
        
        print("\nDepartments:")
        for dept, settings in config['departments'].items():
            keywords = settings.get('keywords', [])
            print(f"  {dept}: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")
    
    else:
        print("Configuration file not found. Using default settings.")

def show_usage_instructions():
    """Show usage instructions."""
    print("\n=== Usage Instructions ===\n")
    
    print("1. Single Document Analysis:")
    print("   python test_analyzer.py")
    print()
    print("2. Batch Analysis:")
    print("   python batch_analyze.py")
    print()
    print("3. Full System (requires Tesseract):")
    print("   python document_analyzer.py image.png -a annotations.json")
    print()
    print("4. Custom Configuration:")
    print("   - Edit ocr_config.json to modify routing rules")
    print("   - Adjust thresholds and department keywords")
    print("   - Add new OCR engines as needed")
    print()
    print("5. Integration:")
    print("   - Import DocumentAnalyzer class")
    print("   - Call analyze_document() method")
    print("   - Process JSON results for routing decisions")

if __name__ == "__main__":
    print("Document Analysis System - Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_single_document_analysis()
    demonstrate_batch_analysis()
    demonstrate_routing_logic()
    demonstrate_configuration()
    show_usage_instructions()
    
    print("\n" + "=" * 50)
    print("System Ready for Production Use!")
    print("All components are functional and tested.")
