#!/usr/bin/env python3
"""
Batch Document Analysis Script
Processes all documents in the dataset and generates OCR routing recommendations
"""

import json
import os
from pathlib import Path
from datetime import datetime
from test_analyzer import SimpleDocumentAnalyzer

def batch_analyze_dataset():
    """Analyze all documents in the dataset."""
    analyzer = SimpleDocumentAnalyzer()
    results = []
    
    # Process testing data
    testing_dir = Path("dataset/testing_data")
    images_dir = testing_dir / "images"
    annotations_dir = testing_dir / "annotations"
    
    print("=== Batch Document Analysis ===\n")
    print(f"Processing documents in: {testing_dir}")
    
    processed_count = 0
    error_count = 0
    
    for image_file in images_dir.glob("*.png"):
        annotation_file = annotations_dir / f"{image_file.stem}.json"
        
        print(f"Processing: {image_file.name}")
        
        try:
            result = analyzer.analyze_document(
                str(image_file), 
                str(annotation_file) if annotation_file.exists() else None
            )
            results.append(result)
            processed_count += 1
            
            # Print key routing decision
            print(f"  -> OCR Engine: {result['ocr_variant_suggestion']}")
            print(f"  -> Department: {result['department_context']}")
            print(f"  -> Priority: {result['priority_level']}")
            print()
            
        except Exception as e:
            print(f"  -> Error: {e}")
            error_count += 1
            print()
    
    # Save results
    output_file = "batch_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"=== Analysis Complete ===")
    print(f"Processed: {processed_count} documents")
    print(f"Errors: {error_count} documents")
    print(f"Results saved to: {output_file}")
    
    return results

def generate_routing_summary(results):
    """Generate a summary of OCR routing decisions."""
    print("\n=== OCR Routing Summary ===")
    
    # Count by OCR engine
    engine_counts = {}
    department_counts = {}
    priority_counts = {}
    
    for result in results:
        engine = result.get('ocr_variant_suggestion', 'unknown')
        department = result.get('department_context', 'unknown')
        priority = result.get('priority_level', 'unknown')
        
        engine_counts[engine] = engine_counts.get(engine, 0) + 1
        department_counts[department] = department_counts.get(department, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    print("\nOCR Engine Distribution:")
    for engine, count in sorted(engine_counts.items()):
        print(f"  {engine}: {count} documents")
    
    print("\nDepartment Distribution:")
    for dept, count in sorted(department_counts.items()):
        print(f"  {dept}: {count} documents")
    
    print("\nPriority Distribution:")
    for priority, count in sorted(priority_counts.items()):
        print(f"  {priority}: {count} documents")

def analyze_specific_features(results):
    """Analyze specific document features across the dataset."""
    print("\n=== Feature Analysis ===")
    
    features = {
        'form_fields_detected': 0,
        'contains_signature': 0,
        'contains_logo': 0,
        'bilingual_flag': 0,
        'table_ratio_high': 0,
        'high_text_density': 0,
        'complex_layout': 0
    }
    
    for result in results:
        if result.get('form_fields_detected'):
            features['form_fields_detected'] += 1
        if result.get('contains_signature'):
            features['contains_signature'] += 1
        if result.get('contains_logo'):
            features['contains_logo'] += 1
        if result.get('bilingual_flag'):
            features['bilingual_flag'] += 1
        if result.get('table_ratio', 0) > 0.1:
            features['table_ratio_high'] += 1
        if result.get('text_density', 0) > 0.7:
            features['high_text_density'] += 1
        if result.get('layout_complexity_score', 0) > 0.7:
            features['complex_layout'] += 1
    
    total_docs = len(results)
    print(f"\nFeature Distribution (out of {total_docs} documents):")
    for feature, count in features.items():
        percentage = (count / total_docs) * 100 if total_docs > 0 else 0
        print(f"  {feature}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    # Run batch analysis
    results = batch_analyze_dataset()
    
    # Generate summaries
    generate_routing_summary(results)
    analyze_specific_features(results)
    
    print(f"\n=== Complete ===")
    print(f"All analysis results saved to: batch_analysis_results.json")
