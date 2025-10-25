#!/usr/bin/env python3
"""
Simple Final PDF Metadata Analysis
Working version that processes all PDFs and generates comprehensive metadata
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# Add service directory to path
sys.path.insert(0, str(Path("document_analyzer_service")))

# Import working components
from service_analyzer import discover_inputs, analyze_file
from document_analyzer import DocumentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFinalAnalyzer:
    """Simple but comprehensive PDF metadata analyzer"""
    
    def __init__(self):
        self.document_analyzer = DocumentAnalyzer()
        self.results = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_pdfs_processed": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "processing_time_seconds": 0,
                "performance_metrics": {}
            },
            "pdf_metadata": []
        }
    
    def analyze_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Analyze a single PDF and extract comprehensive metadata"""
        try:
            start_time = time.time()
            
            # Use the working analyze_file function
            analysis_result = analyze_file(self.document_analyzer, pdf_path)
            
            # Extract comprehensive metadata
            metadata = {
                "document_id": pdf_path.stem,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size_bytes": pdf_path.stat().st_size,
                "file_modified_time": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),
                "processing_status": "success",
                "processing_time": time.time() - start_time,
                
                # Analysis results
                "analysis": analysis_result,
                
                # OCR routing suggestions
                "ocr_routing": {
                    "suggested_engine": analysis_result.get("ocr_variant_suggestion", "tesseract"),
                    "confidence": analysis_result.get("confidence", 0.0),
                    "processing_recommendation": analysis_result.get("processing_recommendation", "standard")
                },
                
                # Document characteristics
                "document_characteristics": {
                    "language": analysis_result.get("language", "unknown"),
                    "text_density": analysis_result.get("text_density", 0.0),
                    "table_ratio": analysis_result.get("table_ratio", 0.0),
                    "layout_complexity": analysis_result.get("layout_complexity", 0.0),
                    "font_variance": analysis_result.get("font_variance", 0.0),
                    "has_signatures": analysis_result.get("has_signatures", False),
                    "has_logos": analysis_result.get("has_logos", False),
                    "has_form_fields": analysis_result.get("has_form_fields", False)
                },
                
                # Quality metrics
                "quality_metrics": {
                    "resolution": analysis_result.get("resolution", 0),
                    "priority": analysis_result.get("priority", "normal"),
                    "department_context": analysis_result.get("department_context", "general"),
                    "word_count": analysis_result.get("word_count", 0),
                    "character_count": analysis_result.get("character_count", 0)
                }
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error analyzing {pdf_path}: {e}")
            return {
                "document_id": pdf_path.stem,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "processing_status": "error",
                "error_message": str(e),
                "metadata": {}
            }
    
    def analyze_all_pdfs(self, pdf_folder: str) -> Dict[str, Any]:
        """Analyze all PDFs in the specified folder"""
        logger.info(f"Starting comprehensive analysis of PDFs in: {pdf_folder}")
        
        start_time = time.time()
        
        # Discover all PDF files
        target_path = Path(pdf_folder)
        pdf_files = list(target_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_folder}")
            return self.results
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process files
        all_results = []
        successful = 0
        failed = 0
        
        for i, pdf_file in enumerate(pdf_files):
            if i % 100 == 0:
                logger.info(f"Processing {i+1}/{len(pdf_files)} files...")
            
            result = self.analyze_single_pdf(pdf_file)
            all_results.append(result)
            
            if result.get("processing_status") == "success":
                successful += 1
            else:
                failed += 1
        
        # Calculate final metrics
        total_time = time.time() - start_time
        
        # Update results
        self.results["metadata"].update({
            "total_pdfs_processed": len(pdf_files),
            "successful_analyses": successful,
            "failed_analyses": failed,
            "processing_time_seconds": total_time,
            "files_per_second": len(pdf_files) / total_time if total_time > 0 else 0,
            "performance_metrics": {
                "average_processing_time": total_time / len(pdf_files) if len(pdf_files) > 0 else 0,
                "success_rate": successful / len(pdf_files) if len(pdf_files) > 0 else 0
            }
        })
        
        self.results["pdf_metadata"] = all_results
        
        logger.info(f"Analysis completed: {successful}/{len(pdf_files)} successful in {total_time:.2f}s")
        
        return self.results
    
    def save_results(self, output_file: str = "simple_final_pdf_metadata.json"):
        """Save comprehensive results to JSON file"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        metadata = self.results["metadata"]
        pdf_data = self.results["pdf_metadata"]
        
        report = f"""
FINAL PDF METADATA ANALYSIS REPORT
==================================

Analysis Timestamp: {metadata['analysis_timestamp']}
Total PDFs Processed: {metadata['total_pdfs_processed']}
Successful Analyses: {metadata['successful_analyses']}
Failed Analyses: {metadata['failed_analyses']}
Success Rate: {(metadata['successful_analyses']/metadata['total_pdfs_processed']*100):.1f}%
Processing Time: {metadata['processing_time_seconds']:.2f} seconds
Files per Second: {metadata['files_per_second']:.2f}

PERFORMANCE METRICS:
- Average Processing Time: {metadata['performance_metrics']['average_processing_time']:.3f}s per file
- Success Rate: {metadata['performance_metrics']['success_rate']:.2%}

DOCUMENT CHARACTERISTICS SUMMARY:
"""
        
        # Analyze document characteristics
        if pdf_data:
            languages = {}
            ocr_engines = {}
            priorities = {}
            departments = {}
            
            for pdf in pdf_data:
                if pdf.get("processing_status") == "success":
                    # Language distribution
                    lang = pdf.get("document_characteristics", {}).get("language", "unknown")
                    languages[lang] = languages.get(lang, 0) + 1
                    
                    # OCR engine suggestions
                    engine = pdf.get("ocr_routing", {}).get("suggested_engine", "tesseract")
                    ocr_engines[engine] = ocr_engines.get(engine, 0) + 1
                    
                    # Priority distribution
                    priority = pdf.get("quality_metrics", {}).get("priority", "normal")
                    priorities[priority] = priorities.get(priority, 0) + 1
                    
                    # Department context
                    dept = pdf.get("quality_metrics", {}).get("department_context", "general")
                    departments[dept] = departments.get(dept, 0) + 1
            
            report += f"""
Language Distribution:
{chr(10).join([f"  {lang}: {count}" for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)])}

OCR Engine Recommendations:
{chr(10).join([f"  {engine}: {count}" for engine, count in sorted(ocr_engines.items(), key=lambda x: x[1], reverse=True)])}

Priority Distribution:
{chr(10).join([f"  {priority}: {count}" for priority, count in sorted(priorities.items(), key=lambda x: x[1], reverse=True)])}

Department Context:
{chr(10).join([f"  {dept}: {count}" for dept, count in sorted(departments.items(), key=lambda x: x[1], reverse=True)])}
"""
        
        return report

def main():
    """Main execution function"""
    print("=" * 80)
    print("FINAL PDF METADATA ANALYSIS - SIMPLE & EFFECTIVE")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = SimpleFinalAnalyzer()
    
    # Analyze all PDFs
    pdf_folder = "1000+ PDF_Invoice_Folder"
    
    if not Path(pdf_folder).exists():
        print(f"Error: PDF folder '{pdf_folder}' not found!")
        return
    
    print(f"Starting analysis of all PDFs in: {pdf_folder}")
    print("Processing with optimized single-threaded approach...")
    
    # Run comprehensive analysis
    results = analyzer.analyze_all_pdfs(pdf_folder)
    
    # Save results
    analyzer.save_results("simple_final_pdf_metadata.json")
    
    # Generate and display summary
    summary = analyzer.generate_summary_report()
    print(summary)
    
    # Save summary to file
    with open("simple_final_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("Files generated:")
    print("  - simple_final_pdf_metadata.json (complete metadata)")
    print("  - simple_final_analysis_summary.txt (human-readable summary)")
    print("=" * 80)

if __name__ == "__main__":
    main()
