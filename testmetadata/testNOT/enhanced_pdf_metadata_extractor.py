#!/usr/bin/env python3
"""
Enhanced PDF Metadata Extractor with PDF to Image Conversion
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
import tempfile
import shutil

# Try to import PDF processing libraries
try:
    import fitz  # PyMuPDF
    PDF_PROCESSING_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("PyMuPDF available for PDF processing")
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyMuPDF not available. PDF processing will be limited.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    logger.info("pdf2image available for PDF conversion")
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. Using PyMuPDF fallback.")

# Import the document analyzer
try:
    from document_analyzer import DocumentAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    logger.warning("DocumentAnalyzer not available, using SimpleDocumentAnalyzer")

if not ANALYZER_AVAILABLE:
    from test_analyzer import SimpleDocumentAnalyzer

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

class EnhancedPDFMetadataExtractor:
    """Enhanced PDF metadata extractor with image conversion capabilities."""
    
    def __init__(self, pdf_directory: str = "1000+ PDF_Invoice_Folder", 
                 output_file: str = "pdf_metadata_results.json",
                 max_workers: int = None,
                 temp_dir: str = None):
        """
        Initialize the enhanced PDF metadata extractor.
        
        Args:
            pdf_directory: Directory containing PDF files
            output_file: Output JSON file for results
            max_workers: Maximum number of parallel workers (default: CPU count)
            temp_dir: Temporary directory for image conversion
        """
        self.pdf_directory = Path(pdf_directory)
        self.output_file = output_file
        self.max_workers = max_workers or min(cpu_count(), 4)  # Reduced for memory efficiency
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="pdf_extraction_"))
        self.results = []
        self.errors = []
        
        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True)
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        # Initialize document analyzer
        try:
            if ANALYZER_AVAILABLE:
                self.analyzer = DocumentAnalyzer()
                logger.info("Document analyzer initialized successfully")
            else:
                self.analyzer = SimpleDocumentAnalyzer()
                logger.info("Using simple document analyzer")
        except Exception as e:
            logger.error(f"Failed to initialize document analyzer: {e}")
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
    
    def convert_pdf_to_images(self, pdf_path: Path) -> List[str]:
        """
        Convert PDF to images for analysis.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of image file paths
        """
        image_paths = []
        
        try:
            # Method 1: Try pdf2image first (better quality)
            if PDF2IMAGE_AVAILABLE:
                try:
                    images = convert_from_path(
                        pdf_path, 
                        dpi=200,  # Good balance of quality and file size
                        first_page=1,
                        last_page=3,  # Process first 3 pages only for efficiency
                        fmt='PNG'
                    )
                    
                    for i, image in enumerate(images):
                        image_path = self.temp_dir / f"{pdf_path.stem}_page_{i+1}.png"
                        image.save(image_path, 'PNG')
                        image_paths.append(str(image_path))
                    
                    logger.debug(f"Converted {pdf_path.name} to {len(image_paths)} images using pdf2image")
                    return image_paths
                    
                except Exception as e:
                    logger.warning(f"pdf2image failed for {pdf_path.name}: {e}")
            
            # Method 2: Fallback to PyMuPDF
            if PDF_PROCESSING_AVAILABLE:
                try:
                    doc = fitz.open(pdf_path)
                    
                    # Process first 3 pages only for efficiency
                    max_pages = min(3, len(doc))
                    
                    for page_num in range(max_pages):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                        
                        image_path = self.temp_dir / f"{pdf_path.stem}_page_{page_num+1}.png"
                        pix.save(str(image_path))
                        image_paths.append(str(image_path))
                    
                    doc.close()
                    logger.debug(f"Converted {pdf_path.name} to {len(image_paths)} images using PyMuPDF")
                    return image_paths
                    
                except Exception as e:
                    logger.warning(f"PyMuPDF failed for {pdf_path.name}: {e}")
            
            # Method 3: Fallback - return PDF path for basic analysis
            logger.warning(f"No PDF conversion available for {pdf_path.name}, using PDF path directly")
            return [str(pdf_path)]
            
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path.name} to images: {e}")
            return []
    
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
            
            # Convert PDF to images
            image_paths = self.convert_pdf_to_images(pdf_path)
            
            if not image_paths:
                raise ValueError("Failed to convert PDF to images")
            
            # Analyze the first image (or PDF if conversion failed)
            primary_image = image_paths[0]
            result = self.analyzer.analyze_document(primary_image)
            
            # Add PDF-specific metadata
            pdf_metadata = self._extract_pdf_specific_metadata(pdf_path)
            
            # Add conversion information
            conversion_info = {
                "conversion_method": "pdf2image" if PDF2IMAGE_AVAILABLE else "pymupdf" if PDF_PROCESSING_AVAILABLE else "none",
                "images_generated": len(image_paths),
                "image_paths": image_paths,
                "primary_image_analyzed": primary_image
            }
            
            # Combine results
            combined_result = {
                **result,
                **pdf_metadata,
                **conversion_info,
                "processing_status": "success",
                "processed_at": datetime.now().isoformat() + "Z"
            }
            
            # Clean up temporary images
            self._cleanup_temp_images(image_paths)
            
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
    
    def _cleanup_temp_images(self, image_paths: List[str]):
        """Clean up temporary image files."""
        try:
            for image_path in image_paths:
                if Path(image_path).exists() and str(image_path).startswith(str(self.temp_dir)):
                    Path(image_path).unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up temp images: {e}")
    
    def _extract_pdf_specific_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract PDF-specific metadata."""
        try:
            # Basic file metadata
            stat = pdf_path.stat()
            
            # Extract information from filename
            filename_parts = pdf_path.stem.split('_')
            customer_name = filename_parts[1] if len(filename_parts) > 1 else "Unknown"
            invoice_number = filename_parts[2] if len(filename_parts) > 2 else "Unknown"
            
            # Try to extract PDF metadata using PyMuPDF
            pdf_info = {}
            if PDF_PROCESSING_AVAILABLE:
                try:
                    doc = fitz.open(pdf_path)
                    pdf_info = {
                        "page_count": len(doc),
                        "pdf_title": doc.metadata.get("title", ""),
                        "pdf_author": doc.metadata.get("author", ""),
                        "pdf_subject": doc.metadata.get("subject", ""),
                        "pdf_creator": doc.metadata.get("creator", ""),
                        "pdf_producer": doc.metadata.get("producer", ""),
                        "pdf_creation_date": doc.metadata.get("creationDate", ""),
                        "pdf_modification_date": doc.metadata.get("modDate", "")
                    }
                    doc.close()
                except Exception as e:
                    logger.warning(f"Could not extract PDF metadata for {pdf_path.name}: {e}")
            
            return {
                "pdf_specific": {
                    "customer_name": customer_name,
                    "invoice_number": invoice_number,
                    "file_extension": pdf_path.suffix,
                    "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_pdf": True,
                    **pdf_info
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
                    if i % 10 == 0 or i == len(pdf_files):
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
        conversion_methods = {}
        
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
                
                # Conversion method distribution
                method = result.get('conversion_method', 'unknown')
                conversion_methods[method] = conversion_methods.get(method, 0) + 1
        
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
            "conversion_method_distribution": conversion_methods,
            "file_size_stats": self._calculate_file_size_stats(results),
            "pdf_processing_capabilities": {
                "pdf2image_available": PDF2IMAGE_AVAILABLE,
                "pymupdf_available": PDF_PROCESSING_AVAILABLE,
                "document_analyzer_available": ANALYZER_AVAILABLE
            }
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
                    "extractor_version": "2.0.0",
                    "temp_directory": str(self.temp_dir)
                }
            }
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def cleanup(self):
        """Clean up temporary directory."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {e}")
    
    def run_extraction(self):
        """Run the complete PDF metadata extraction process."""
        logger.info("Starting enhanced PDF metadata extraction process")
        logger.info(f"PDF Directory: {self.pdf_directory}")
        logger.info(f"Output File: {self.output_file}")
        logger.info(f"Max Workers: {self.max_workers}")
        logger.info(f"Temp Directory: {self.temp_dir}")
        
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
            
            logger.info("Enhanced PDF metadata extraction completed successfully")
            
        except Exception as e:
            logger.error(f"Fatal error during extraction: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Clean up
            self.cleanup()
    
    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print a final summary to the console."""
        print("\n" + "="*60)
        print("ENHANCED PDF METADATA EXTRACTION SUMMARY")
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
        
        print(f"\nConversion Method Distribution:")
        for method, count in summary["conversion_method_distribution"].items():
            percentage = (count / ps['successful'] * 100) if ps['successful'] > 0 else 0
            print(f"  {method}: {count} ({percentage:.1f}%)")
        
        print(f"\nFile Size Statistics:")
        fs = summary["file_size_stats"]
        print(f"  Total Size: {fs['total_size_mb']:.1f} MB")
        print(f"  Average Size: {fs['average_size_bytes']/1024:.1f} KB")
        print(f"  Min Size: {fs['min_size_bytes']/1024:.1f} KB")
        print(f"  Max Size: {fs['max_size_bytes']/1024:.1f} KB")
        
        print(f"\nProcessing Capabilities:")
        caps = summary["pdf_processing_capabilities"]
        print(f"  PDF2Image: {'✓' if caps['pdf2image_available'] else '✗'}")
        print(f"  PyMuPDF: {'✓' if caps['pymupdf_available'] else '✗'}")
        print(f"  Document Analyzer: {'✓' if caps['document_analyzer_available'] else '✗'}")
        
        print("="*60)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract metadata from PDF invoices with image conversion")
    parser.add_argument("-d", "--directory", 
                       default="1000+ PDF_Invoice_Folder",
                       help="Directory containing PDF files")
    parser.add_argument("-o", "--output", 
                       default="enhanced_pdf_metadata_results.json",
                       help="Output JSON file")
    parser.add_argument("-w", "--workers", 
                       type=int, 
                       help="Number of parallel workers")
    parser.add_argument("-t", "--temp-dir",
                       help="Temporary directory for image conversion")
    
    args = parser.parse_args()
    
    # Create extractor and run
    extractor = EnhancedPDFMetadataExtractor(
        pdf_directory=args.directory,
        output_file=args.output,
        max_workers=args.workers,
        temp_dir=args.temp_dir
    )
    
    try:
        extractor.run_extraction()
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
        extractor.cleanup()
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        extractor.cleanup()
        raise


if __name__ == "__main__":
    main()
