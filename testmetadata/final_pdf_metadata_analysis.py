#!/usr/bin/env python3
"""
Final Comprehensive PDF Metadata Analysis
Assembles all components and generates complete metadata for all 2014 PDFs
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

# Import optimized components
from optimized_service import OptimizedDocumentAnalyzer, ProcessingConfig, SLOTargets
from service_analyzer import discover_inputs, convert_pdf_to_images, analyze_file
from document_analyzer import DocumentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalPDFMetadataAnalyzer:
    """Final comprehensive PDF metadata analyzer"""
    
    def __init__(self):
        # Optimized configuration for maximum performance
        self.config = ProcessingConfig(
            max_threads=64,
            max_processes=8,
            chunk_size=32,
            cache_size_mb=1024,
            temp_dir=".final_processing_tmp"
        )
        
        self.analyzer = OptimizedDocumentAnalyzer(self.config)
        self.document_analyzer = DocumentAnalyzer()
        
        # Create temp directory
        Path(self.config.temp_dir).mkdir(exist_ok=True)
        
        # Results storage
        self.results = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_pdfs_processed": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "processing_time_seconds": 0,
                "slo_compliance": {},
                "performance_metrics": {}
            },
            "pdf_metadata": []
        }
    
    async def analyze_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Analyze a single PDF and extract comprehensive metadata"""
        try:
            start_time = time.time()
            
            # Convert PDF to images
            images = convert_pdf_to_images(pdf_path, self.config.temp_dir)
            
            if not images:
                return {
                    "document_id": pdf_path.stem,
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                    "processing_status": "error",
                    "error_message": "No images generated from PDF",
                    "metadata": {}
                }
            
            # Analyze first page (most important for routing)
            first_image = images[0]
            analysis_result = analyze_file(self.document_analyzer, first_image)
            
            # Extract comprehensive metadata
            metadata = {
                "document_id": pdf_path.stem,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size_bytes": pdf_path.stat().st_size,
                "file_modified_time": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),
                "processing_status": "success",
                "processing_time": time.time() - start_time,
                
                # Document structure
                "page_count": len(images),
                "image_files": [str(img) for img in images],
                
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
    
    async def process_pdf_batch(self, pdf_files: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of PDFs in parallel"""
        tasks = [self.analyze_single_pdf(pdf) for pdf in pdf_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                continue
            valid_results.append(result)
        
        return valid_results
    
    async def analyze_all_pdfs(self, pdf_folder: str) -> Dict[str, Any]:
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
        
        # Process in optimized chunks
        all_results = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, len(pdf_files), chunk_size):
            chunk = pdf_files[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(pdf_files) + chunk_size - 1) // chunk_size
            
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} files)")
            
            chunk_results = await self.process_pdf_batch(chunk)
            all_results.extend(chunk_results)
            
            # Log progress
            processed = len(all_results)
            logger.info(f"Progress: {processed}/{len(pdf_files)} files processed")
        
        # Calculate final metrics
        total_time = time.time() - start_time
        successful = sum(1 for r in all_results if r.get("processing_status") == "success")
        failed = len(all_results) - successful
        
        # Get SLO metrics
        metrics = self.analyzer.metrics.get_metrics()
        slo = SLOTargets()
        
        # Update results
        self.results["metadata"].update({
            "total_pdfs_processed": len(pdf_files),
            "successful_analyses": successful,
            "failed_analyses": failed,
            "processing_time_seconds": total_time,
            "files_per_second": len(pdf_files) / total_time if total_time > 0 else 0,
            "slo_compliance": {
                "latency_p50_compliant": metrics.get("latency_p50", 0) <= slo.p50_latency,
                "latency_p95_compliant": metrics.get("latency_p95", 0) <= slo.p95_latency,
                "error_rate_compliant": metrics.get("error_rate", 0) <= slo.max_error_rate,
                "cache_hit_rate_compliant": metrics.get("cache_hit_rate", 0) >= slo.min_cache_hit_rate
            },
            "performance_metrics": metrics
        })
        
        self.results["pdf_metadata"] = all_results
        
        logger.info(f"Analysis completed: {successful}/{len(pdf_files)} successful in {total_time:.2f}s")
        
        return self.results
    
    def save_results(self, output_file: str = "final_pdf_metadata_analysis.json"):
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

SLO COMPLIANCE:
- P50 Latency: {'PASS' if metadata['slo_compliance']['latency_p50_compliant'] else 'FAIL'}
- P95 Latency: {'PASS' if metadata['slo_compliance']['latency_p95_compliant'] else 'FAIL'}
- Error Rate: {'PASS' if metadata['slo_compliance']['error_rate_compliant'] else 'FAIL'}
- Cache Hit Rate: {'PASS' if metadata['slo_compliance']['cache_hit_rate_compliant'] else 'FAIL'}

PERFORMANCE METRICS:
- Latency P50: {metadata['performance_metrics'].get('latency_p50', 0):.2f}s
- Latency P95: {metadata['performance_metrics'].get('latency_p95', 0):.2f}s
- Error Rate: {metadata['performance_metrics'].get('error_rate', 0):.2%}
- Cache Hit Rate: {metadata['performance_metrics'].get('cache_hit_rate', 0):.2%}

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

async def main():
    """Main execution function"""
    print("=" * 80)
    print("FINAL PDF METADATA ANALYSIS - COMPREHENSIVE PROCESSING")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = FinalPDFMetadataAnalyzer()
    
    # Analyze all PDFs
    pdf_folder = "1000+ PDF_Invoice_Folder"
    
    if not Path(pdf_folder).exists():
        print(f"Error: PDF folder '{pdf_folder}' not found!")
        return
    
    print(f"Starting analysis of all PDFs in: {pdf_folder}")
    print("This may take several minutes for optimal performance...")
    
    # Run comprehensive analysis
    results = await analyzer.analyze_all_pdfs(pdf_folder)
    
    # Save results
    analyzer.save_results("final_pdf_metadata_analysis.json")
    
    # Generate and display summary
    summary = analyzer.generate_summary_report()
    print(summary)
    
    # Save summary to file
    with open("final_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("Files generated:")
    print("  - final_pdf_metadata_analysis.json (complete metadata)")
    print("  - final_analysis_summary.txt (human-readable summary)")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
