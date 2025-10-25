#!/usr/bin/env python3
"""
Example usage of the Document Analyzer for OCR routing
"""

from document_analyzer import DocumentAnalyzer
import json

def analyze_single_document():
    """Example: Analyze a single document"""
    analyzer = DocumentAnalyzer()
    
    # Analyze the document with its annotation file
    image_path = "dataset/testing_data/images/82250337_0338.png"
    annotation_path = "dataset/testing_data/annotations/82250337_0338.json"
    
    result = analyzer.analyze_document(image_path, annotation_path)
    
    print("Document Analysis Results:")
    print(json.dumps(result, indent=2))
    
    return result

def analyze_multiple_documents():
    """Example: Analyze multiple documents in a directory"""
    analyzer = DocumentAnalyzer()
    
    # Analyze all documents in the testing_data directory
    input_dir = "dataset/testing_data"
    output_file = "testing_data_analysis.json"
    
    results = analyzer.batch_analyze(input_dir, output_file)
    
    print(f"Analyzed {len(results)} documents")
    print(f"Results saved to {output_file}")
    
    return results

def analyze_without_annotations():
    """Example: Analyze documents without annotation files"""
    analyzer = DocumentAnalyzer()
    
    # Analyze just the image without annotations
    image_path = "dataset/testing_data/images/82250337_0338.png"
    
    result = analyzer.analyze_document(image_path)
    
    print("Document Analysis Results (No Annotations):")
    print(json.dumps(result, indent=2))
    
    return result

def custom_configuration():
    """Example: Use custom configuration"""
    # Create custom config
    custom_config = {
        "ocr_engines": {
            "form_optimized": {"priority": 1, "handles_forms": True},
            "table_optimized": {"priority": 2, "handles_tables": True},
            "handwriting": {"priority": 3, "handles_handwriting": True},
            "multilingual": {"priority": 4, "handles_multilingual": True},
            "standard": {"priority": 5, "general_purpose": True}
        },
        "thresholds": {
            "text_density_high": 0.8,
            "text_density_low": 0.2,
            "table_ratio_threshold": 0.05,
            "image_ratio_threshold": 0.15
        },
        "departments": {
            "business_reporting": ["report", "progress", "competitive", "introduction"],
            "financial": ["invoice", "receipt", "payment", "financial"],
            "legal": ["contract", "agreement", "legal", "terms"],
            "medical": ["patient", "medical", "health", "prescription"],
            "academic": ["research", "paper", "thesis", "academic"]
        }
    }
    
    # Save custom config
    with open("custom_ocr_config.json", "w") as f:
        json.dump(custom_config, f, indent=2)
    
    # Use custom config
    analyzer = DocumentAnalyzer("custom_ocr_config.json")
    
    image_path = "dataset/testing_data/images/82250337_0338.png"
    annotation_path = "dataset/testing_data/annotations/82250337_0338.json"
    
    result = analyzer.analyze_document(image_path, annotation_path)
    
    print("Document Analysis Results (Custom Config):")
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    print("=== Document Analyzer Examples ===\n")
    
    print("1. Analyzing single document with annotations:")
    analyze_single_document()
    
    print("\n" + "="*50 + "\n")
    
    print("2. Analyzing document without annotations:")
    analyze_without_annotations()
    
    print("\n" + "="*50 + "\n")
    
    print("3. Using custom configuration:")
    custom_configuration()
    
    print("\n" + "="*50 + "\n")
    
    print("4. Batch analyzing multiple documents:")
    analyze_multiple_documents()
