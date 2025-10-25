#!/usr/bin/env python3
"""
Simple runner script for PDF metadata extraction
This script provides an easy way to extract metadata from all PDF invoices.
"""

import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    # Check for PDF processing libraries
    try:
        import fitz
        logger.info("✓ PyMuPDF available")
    except ImportError:
        missing_deps.append("PyMuPDF (pip install PyMuPDF)")
    
    try:
        from pdf2image import convert_from_path
        logger.info("✓ pdf2image available")
    except ImportError:
        missing_deps.append("pdf2image (pip install pdf2image)")
    
    # Check for image processing libraries
    try:
        import cv2
        logger.info("✓ OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python (pip install opencv-python)")
    
    try:
        from PIL import Image
        logger.info("✓ Pillow available")
    except ImportError:
        missing_deps.append("Pillow (pip install Pillow)")
    
    try:
        import pytesseract
        logger.info("✓ Tesseract available")
    except ImportError:
        missing_deps.append("pytesseract (pip install pytesseract)")
    
    if missing_deps:
        logger.warning("Missing dependencies:")
        for dep in missing_deps:
            logger.warning(f"  - {dep}")
        logger.warning("Some features may not work properly.")
        return False
    
    logger.info("All dependencies available!")
    return True

def run_extraction():
    """Run the PDF metadata extraction."""
    try:
        # Import the extractor
        from enhanced_pdf_metadata_extractor import EnhancedPDFMetadataExtractor
        
        # Check if PDF directory exists
        pdf_dir = "1000+ PDF_Invoice_Folder"
        if not Path(pdf_dir).exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
            logger.info("Please ensure the PDF directory exists and contains PDF files.")
            return False
        
        # Count PDF files
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        if len(pdf_files) == 0:
            logger.error("No PDF files found in the directory")
            return False
        
        # Create extractor
        extractor = EnhancedPDFMetadataExtractor(
            pdf_directory=pdf_dir,
            output_file="pdf_metadata_results.json",
            max_workers=4  # Conservative number for stability
        )
        
        # Run extraction
        logger.info("Starting PDF metadata extraction...")
        extractor.run_extraction()
        
        logger.info("PDF metadata extraction completed successfully!")
        logger.info(f"Results saved to: pdf_metadata_results.json")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all required files are in the same directory.")
        return False
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return False

def main():
    """Main function."""
    print("PDF Metadata Extractor")
    print("=" * 50)
    
    # Check dependencies
    logger.info("Checking dependencies...")
    deps_ok = check_dependencies()
    
    if not deps_ok:
        logger.warning("Some dependencies are missing. Continuing with limited functionality...")
    
    # Run extraction
    logger.info("Starting extraction process...")
    success = run_extraction()
    
    if success:
        print("\n" + "=" * 50)
        print("EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Check the following files for results:")
        print("  - pdf_metadata_results.json (main results)")
        print("  - pdf_metadata_extraction.log (processing log)")
    else:
        print("\n" + "=" * 50)
        print("EXTRACTION FAILED!")
        print("=" * 50)
        print("Check the log file for error details:")
        print("  - pdf_metadata_extraction.log")
        sys.exit(1)

if __name__ == "__main__":
    main()
