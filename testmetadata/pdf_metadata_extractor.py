#!/usr/bin/env python3
"""
PDF Metadata Extractor for Invoice Processing
Extracts comprehensive metadata from all 1000+ PDF invoices using the document analysis service.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import concurrent.futures
from multiprocessing import cpu_count
import traceback

# Import the document analyzer
from document_analyzer import DocumentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_metadata_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PDFMetadataExtractor:
    """Extract metadata from PDF invoices using document analysis service."""
    
    def __init__(self, pdf_directory: str = "1000+ PDF_Invoice_Folder", 
                 output_file: str = "pdf_metadata_results.json",
                 max_workers: int = None):
        """
        Initialize the PDF metadata extractor.
        
        Args:
            pdf_directory: Directory containing PDF files
            output_file: Output JSON file for results
            max_workers: Maximum number of parallel workers (default: CPU count)
        """
        self.pdf_directory = Path(pdf_directory)
        self.output_file = output_file
        self.max_workers = max_workers or min(cpu_count(), 8)  # Limit to 8 workers max
        self.results = []
        self.errors = []
        
        # Initialize document analyzer
        try:
            self.analyzer = DocumentAnalyzer()
            logger.info("Document analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document analyzer: {e}")
            # Fallback to simple analyzer
            from test_analyzer import SimpleDocumentAnalyzer
            self.analyzer = SimpleDocumentAnalyzer()
            logger.info("Using simple document analyzer as fallback")
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the directory."""
        pdf_files = []
        
        if not self.pdf_directory.exists():
            logger.error(f"PDF directory does not exist: {self.pdf_directory}")
            return pdf_files
        
        # Find all PDF files
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        return pdf_files
    
    def convert_pdf_to_image(self, pdf_path: Path) -> Optional[str]:
        """
        Convert PDF to image for analysis.
        This is a placeholder - in practice, you'd use pdf2image or similar.
        """
        try:
            # For now, we'll work with PDF files directly
            # In a real implementation, you would convert PDF to image here
            # using libraries like pdf2image, PyMuPDF, or similar
            
            # Placeholder: return the PDF path as if it were an image
            # This allows the analyzer to work with PDF metadata
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path} to image: {e}")
            return None
    
    def extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            logger.info(f"Processing PDF: {pdf_path.name}")
            
            # Convert PDF to image (placeholder implementation)
            image_path = self.convert_pdf_to_image(pdf_path)
            if not image_path:
                raise ValueError("Failed to convert PDF to image")
            
            # Analyze the document
            result = self.analyzer.analyze_document(image_path)
            
            # Add PDF-specific metadata
            pdf_metadata = self._extract_pdf_specific_metadata(pdf_path)
            
            # Combine results
            combined_result = {
                **result,
                **pdf_metadata,
                "processing_status": "success",
                "processed_at": datetime.now().isoformat() + "Z"
            }
            
            logger.info(f"Successfully processed: {pdf_path.name}")
            return combined_result
            
        except Exception as e:
            error_msg = f"Error processing {pdf_path.name}: {str(e)}"
            logger.error(error_msg)
            
            # Return error result
            return {
                "document_id": pdf_path.stem,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
                "processing_status": "error",
                "error_message": str(e),
                "processed_at": datetime.now().isoformat() + "Z"
            }
    
    def _extract_pdf_specific_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract PDF-specific metadata."""
        try:
            # Basic file metadata
            stat = pdf_path.stat()
            
            # Extract information from filename
            filename_parts = pdf_path.stem.split('_')
            customer_name = filename_parts[1] if len(filename_parts) > 1 else "Unknown"
            invoice_number = filename_parts[2] if len(filename_parts) > 2 else "Unknown"
            
            return {
                "pdf_specific": {
                    "customer_name": customer_name,
                    "invoice_number": invoice_number,
                    "file_extension": pdf_path.suffix,
                    "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_pdf": True
                },
                "document_type": "invoice",
                "source_format": "pdf"
            }
            
        except Exception as e:
            logger.warning(f"Could not extract PDF-specific metadata for {pdf_path.name}: {e}")
            return {
                "pdf_specific": {
                    "customer_name": "Unknown",
                    "invoice_number": "Unknown",
                    "file_extension": pdf_path.suffix,
                    "is_pdf": True
                },
                "document_type": "invoice",
                "source_format": "pdf"
            }
    
    def process_pdf_batch(self, pdf_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Process a batch of PDF files in parallel.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            List of metadata dictionaries
        """
        results = []
        
        logger.info(f"Processing {len(pdf_files)} PDF files with {self.max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(self.extract_pdf_metadata, pdf_file): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Process completed tasks
            for i, future in enumerate(concurrent.futures.as_completed(future_to_pdf), 1):
                pdf_file = future_to_pdf[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress update
                    if i % 50 == 0 or i == len(pdf_files):
                        logger.info(f"Processed {i}/{len(pdf_files)} files")
                        
                except Exception as e:
                    error_msg = f"Unexpected error processing {pdf_file.name}: {e}"
                    logger.error(error_msg)
                    
                    # Add error result
                    error_result = {
                        "document_id": pdf_file.stem,
                        "file_name": pdf_file.name,
                        "file_path": str(pdf_file),
                        "processing_status": "error",
                        "error_message": str(e),
                        "processed_at": datetime.now().isoformat() + "Z"
                    }
                    results.append(error_result)
        
        return results
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary report of the processing results."""
        total_files = len(results)
        successful = len([r for r in results if r.get('processing_status') == 'success'])
        failed = len([r for r in results if r.get('processing_status') == 'error'])
        
        # OCR engine distribution
        ocr_engines = {}
        departments = {}
        priorities = {}
        document_types = {}
        
        for result in results:
            if result.get('processing_status') == 'success':
                # OCR engine distribution
                engine = result.get('ocr_variant_suggestion', 'unknown')
                ocr_engines[engine] = ocr_engines.get(engine, 0) + 1
                
                # Department distribution
                dept = result.get('department_context', 'unknown')
                departments[dept] = departments.get(dept, 0) + 1
                
                # Priority distribution
                priority = result.get('priority_level', 'unknown')
                priorities[priority] = priorities.get(priority, 0) + 1
                
                # Document type distribution
                doc_type = result.get('document_type', 'unknown')
                document_types[doc_type] = document_types.get(doc_type, 0) + 1
        
        summary = {
            "processing_summary": {
                "total_files": total_files,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total_files * 100) if total_files > 0 else 0,
                "processing_date": datetime.now().isoformat() + "Z"
            },
            "ocr_engine_distribution": ocr_engines,
            "department_distribution": departments,
            "priority_distribution": priorities,
            "document_type_distribution": document_types,
            "file_size_stats": self._calculate_file_size_stats(results)
        }
        
        return summary
    
    def _calculate_file_size_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate file size statistics."""
        sizes = [r.get('file_size_bytes', 0) for r in results if r.get('file_size_bytes', 0) > 0]
        
        if not sizes:
            return {"total_size_bytes": 0, "average_size_bytes": 0, "min_size_bytes": 0, "max_size_bytes": 0}
        
        return {
            "total_size_bytes": sum(sizes),
            "average_size_bytes": sum(sizes) / len(sizes),
            "min_size_bytes": min(sizes),
            "max_size_bytes": max(sizes),
            "total_size_mb": sum(sizes) / (1024 * 1024)
        }
    
    def save_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Save results to JSON file."""
        try:
            output_data = {
                "summary": summary,
                "results": results,
                "metadata": {
                    "extraction_date": datetime.now().isoformat() + "Z",
                    "total_documents": len(results),
                    "extractor_version": "1.0.0"
                }
            }
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def run_extraction(self):
        """Run the complete PDF metadata extraction process."""
        logger.info("Starting PDF metadata extraction process")
        logger.info(f"PDF Directory: {self.pdf_directory}")
        logger.info(f"Output File: {self.output_file}")
        logger.info(f"Max Workers: {self.max_workers}")
        
        try:
            # Get all PDF files
            pdf_files = self.get_pdf_files()
            
            if not pdf_files:
                logger.error("No PDF files found to process")
                return
            
            # Process all PDF files
            logger.info(f"Starting batch processing of {len(pdf_files)} PDF files")
            results = self.process_pdf_batch(pdf_files)
            
            # Generate summary report
            logger.info("Generating summary report")
            summary = self.generate_summary_report(results)
            
            # Save results
            logger.info("Saving results")
            self.save_results(results, summary)
            
            # Print final summary
            self._print_final_summary(summary)
            
            logger.info("PDF metadata extraction completed successfully")
            
        except Exception as e:
            logger.error(f"Fatal error during extraction: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print a final summary to the console."""
        print("\n" + "="*60)
        print("PDF METADATA EXTRACTION SUMMARY")
        print("="*60)
        
        ps = summary["processing_summary"]
        print(f"Total Files Processed: {ps['total_files']}")
        print(f"Successful: {ps['successful']}")
        print(f"Failed: {ps['failed']}")
        print(f"Success Rate: {ps['success_rate']:.1f}%")
        
        print(f"\nOCR Engine Distribution:")
        for engine, count in summary["ocr_engine_distribution"].items():
            percentage = (count / ps['successful'] * 100) if ps['successful'] > 0 else 0
            print(f"  {engine}: {count} ({percentage:.1f}%)")
        
        print(f"\nDepartment Distribution:")
        for dept, count in summary["department_distribution"].items():
            percentage = (count / ps['successful'] * 100) if ps['successful'] > 0 else 0
            print(f"  {dept}: {count} ({percentage:.1f}%)")
        
        print(f"\nFile Size Statistics:")
        fs = summary["file_size_stats"]
        print(f"  Total Size: {fs['total_size_mb']:.1f} MB")
        print(f"  Average Size: {fs['average_size_bytes']/1024:.1f} KB")
        print(f"  Min Size: {fs['min_size_bytes']/1024:.1f} KB")
        print(f"  Max Size: {fs['max_size_bytes']/1024:.1f} KB")
        
        print("="*60)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract metadata from PDF invoices")
    parser.add_argument("-d", "--directory", 
                       default="1000+ PDF_Invoice_Folder",
                       help="Directory containing PDF files")
    parser.add_argument("-o", "--output", 
                       default="pdf_metadata_results.json",
                       help="Output JSON file")
    parser.add_argument("-w", "--workers", 
                       type=int, 
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Create extractor and run
    extractor = PDFMetadataExtractor(
        pdf_directory=args.directory,
        output_file=args.output,
        max_workers=args.workers
    )
    
    extractor.run_extraction()


if __name__ == "__main__":
    main()
