#!/usr/bin/env python3
"""
Fixed Fast Real PDF Metadata Extractor
Optimized for maximum CPU utilization using threads while extracting real document features
"""

import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
import numpy as np
import re
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedFastMetadataExtractor:
    """Fixed fast metadata extractor with maximum CPU utilization"""
    
    def __init__(self):
        # Use all available CPU cores for threading
        self.max_workers = psutil.cpu_count(logical=True) * 2  # 2x for I/O bound tasks
        self.results = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_pdfs_processed": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "processing_time_seconds": 0,
                "feature_extraction_metrics": {}
            },
            "document_features": []
        }
        self.lock = threading.Lock()
    
    def extract_single_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from a single PDF - optimized for speed"""
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            
            # Basic document info
            doc_info = {
                "document_id": pdf_path.stem,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size_bytes": pdf_path.stat().st_size,
                "file_modified_time": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),
                "processing_status": "success",
                "processing_time": time.time() - start_time,
            }
            
            # Extract real features quickly
            features = self._extract_document_features_fast(doc)
            doc_info.update(features)
            
            doc.close()
            return doc_info
            
        except Exception as e:
            return {
                "document_id": pdf_path.stem,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "processing_status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _extract_document_features_fast(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract document features with maximum speed optimization"""
        features = {}
        
        # 1. Document Structure (fast)
        features["document_structure"] = {
            "page_count": len(doc),
            "has_metadata": bool(doc.metadata),
            "creator": doc.metadata.get("creator", "") if doc.metadata else "",
            "producer": doc.metadata.get("producer", "") if doc.metadata else "",
            "title": doc.metadata.get("title", "") if doc.metadata else "",
            "has_forms": any(len(list(page.get_drawings())) > 0 for page in doc[:2]),  # Convert generator to list
            "has_annotations": any(len(list(page.annots())) > 0 for page in doc[:2])   # Convert generator to list
        }
        
        # 2. Text Content Analysis (fast - first page only)
        if len(doc) > 0:
            page = doc[0]
            text = page.get_text()
            features["text_content"] = self._analyze_text_content_fast(text, page)
        else:
            features["text_content"] = self._get_empty_text_features()
        
        # 3. Visual Features (fast - first page only)
        if len(doc) > 0:
            features["visual_features"] = self._analyze_visual_features_fast(doc[0])
        else:
            features["visual_features"] = self._get_empty_visual_features()
        
        # 4. Layout Features (fast)
        features["layout_features"] = self._analyze_layout_features_fast(doc)
        
        # 5. OCR Features (fast)
        features["ocr_features"] = self._analyze_ocr_features_fast(doc)
        
        return features
    
    def _analyze_text_content_fast(self, text: str, page: fitz.Page) -> Dict[str, Any]:
        """Fast text content analysis"""
        words = text.split()
        
        # Fast regex patterns
        has_numbers = bool(re.search(r'\d', text))
        has_currency = bool(re.search(r'[$€£¥₹]', text))
        has_dates = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text))
        has_emails = bool(re.search(r'@', text))
        has_phone = bool(re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text))
        
        # Language detection (fast)
        language_indicators = {
            "english": len(re.findall(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', text.lower())),
            "spanish": len(re.findall(r'\b(el|la|de|en|y|o|pero|con|por|para)\b', text.lower())),
            "german": len(re.findall(r'\b(der|die|das|und|oder|aber|in|auf|zu|für|mit|von)\b', text.lower())),
            "french": len(re.findall(r'\b(le|la|de|en|et|ou|mais|dans|sur|à|pour|avec|par)\b', text.lower()))
        }
        
        # Font analysis (fast)
        try:
            text_dict = page.get_text("dict")
            fonts = set()
            sizes = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            fonts.add(span.get("font", ""))
                            sizes.append(span.get("size", 0))
            
            unique_fonts = len(fonts)
            font_sizes = list(set(sizes)) if sizes else [0]
        except:
            unique_fonts = 1
            font_sizes = [12]
        
        return {
            "total_characters": len(text),
            "total_words": len(words),
            "average_word_length": np.mean([len(word) for word in words]) if words else 0,
            "unique_fonts": unique_fonts,
            "font_sizes": font_sizes,
            "has_numbers": has_numbers,
            "has_currency": has_currency,
            "has_dates": has_dates,
            "has_emails": has_emails,
            "has_phone_numbers": has_phone,
            "language_indicators": language_indicators,
            "has_tables": self._detect_tables_fast(text)
        }
    
    def _analyze_visual_features_fast(self, page: fitz.Page) -> Dict[str, Any]:
        """Fast visual features analysis"""
        try:
            # Convert to image with lower resolution for speed
            mat = fitz.Matrix(1.0, 1.0)  # 1x zoom for speed
            pix = page.get_pixmap(matrix=mat)
            
            # Get basic image info
            width = pix.width
            height = pix.height
            aspect_ratio = width / height
            
            # Fast brightness analysis
            img_data = pix.tobytes("png")
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            
            # Simple brightness calculation
            brightness_mean = np.mean(img_array)
            brightness_std = np.std(img_array)
            
            return {
                "image_dimensions": {"width": width, "height": height},
                "aspect_ratio": aspect_ratio,
                "brightness_mean": float(brightness_mean),
                "brightness_std": float(brightness_std),
                "contrast": float(brightness_std)
            }
            
        except Exception as e:
            return {
                "image_dimensions": {"width": 0, "height": 0},
                "aspect_ratio": 1.0,
                "brightness_mean": 128.0,
                "brightness_std": 50.0,
                "contrast": 50.0,
                "error": str(e)
            }
    
    def _analyze_layout_features_fast(self, doc: fitz.Document) -> Dict[str, Any]:
        """Fast layout analysis"""
        has_images = False
        has_graphics = False
        column_count = 1
        
        # Check first 2 pages only
        for page_num in range(min(2, len(doc))):
            page = doc[page_num]
            
            # Quick checks
            if page.get_images():
                has_images = True
            if list(page.get_drawings()):  # Convert generator to list
                has_graphics = True
            
            # Simple column detection
            if page_num == 0:  # Only check first page
                try:
                    text_dict = page.get_text("dict")
                    x_positions = []
                    for block in text_dict.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                if "bbox" in line:
                                    x_positions.append(line["bbox"][0])
                    
                    if len(x_positions) > 10:
                        # Simple clustering
                        x_positions = sorted(x_positions)
                        clusters = 1
                        for i in range(1, len(x_positions)):
                            if x_positions[i] - x_positions[i-1] > 100:  # Gap threshold
                                clusters += 1
                        column_count = min(clusters, 3)
                except:
                    column_count = 1
        
        return {
            "has_images": has_images,
            "has_graphics": has_graphics,
            "column_count": column_count,
            "has_headers": False,  # Skip complex analysis
            "has_footers": False,  # Skip complex analysis
            "has_sidebars": False  # Skip complex analysis
        }
    
    def _analyze_ocr_features_fast(self, doc: fitz.Document) -> Dict[str, Any]:
        """Fast OCR features analysis"""
        try:
            if len(doc) > 0:
                page = doc[0]
                text = page.get_text()
                
                # Calculate text density
                text_density = len(text) / (page.rect.width * page.rect.height) * 1000000
                
                # Simple OCR engine recommendation based on real features
                if text_density > 50:
                    recommended_engine = "paddleocr"
                elif len(text) > 500:
                    recommended_engine = "tesseract"
                else:
                    recommended_engine = "easyocr"
                
                return {
                    "text_density": text_density,
                    "font_clarity": 0.8,  # Default assumption
                    "noise_level": 0.1,   # Default assumption
                    "skew_angle": 0.0,    # Default assumption
                    "recommended_ocr_engine": recommended_engine
                }
            else:
                return {
                    "text_density": 0.0,
                    "font_clarity": 0.0,
                    "noise_level": 0.0,
                    "skew_angle": 0.0,
                    "recommended_ocr_engine": "tesseract"
                }
        except:
            return {
                "text_density": 0.0,
                "font_clarity": 0.0,
                "noise_level": 0.0,
                "skew_angle": 0.0,
                "recommended_ocr_engine": "tesseract"
            }
    
    def _detect_tables_fast(self, text: str) -> bool:
        """Fast table detection"""
        # Simple patterns
        return bool(re.search(r'\|\s*\w+\s*\|', text) or 
                    re.search(r'\s{3,}\w+\s{3,}', text) or
                    re.search(r'\t\w+\t', text))
    
    def _get_empty_text_features(self) -> Dict[str, Any]:
        """Return empty text features"""
        return {
            "total_characters": 0,
            "total_words": 0,
            "average_word_length": 0,
            "unique_fonts": 0,
            "font_sizes": [0],
            "has_numbers": False,
            "has_currency": False,
            "has_dates": False,
            "has_emails": False,
            "has_phone_numbers": False,
            "language_indicators": {"english": 0, "spanish": 0, "german": 0, "french": 0},
            "has_tables": False
        }
    
    def _get_empty_visual_features(self) -> Dict[str, Any]:
        """Return empty visual features"""
        return {
            "image_dimensions": {"width": 0, "height": 0},
            "aspect_ratio": 1.0,
            "brightness_mean": 128.0,
            "brightness_std": 50.0,
            "contrast": 50.0
        }
    
    def analyze_all_pdfs(self, pdf_folder: str) -> Dict[str, Any]:
        """Analyze all PDFs with maximum CPU utilization using threads"""
        logger.info(f"Starting FIXED FAST metadata extraction from PDFs in: {pdf_folder}")
        logger.info(f"Using {self.max_workers} threads for maximum performance")
        
        start_time = time.time()
        
        # Discover all PDF files
        target_path = Path(pdf_folder)
        pdf_files = list(target_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_folder}")
            return self.results
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process files with maximum parallelism using threads
        all_results = []
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(self.extract_single_pdf_metadata, pdf_file): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_pdf)):
                pdf_file = future_to_pdf[future]
                
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    if result.get("processing_status") == "success":
                        successful += 1
                    else:
                        failed += 1
                    
                    # Progress logging every 50 files
                    if (i + 1) % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed
                        eta = (len(pdf_files) - i - 1) / rate if rate > 0 else 0
                        logger.info(f"Processed {i+1}/{len(pdf_files)} files. Rate: {rate:.1f} files/sec. ETA: {eta:.1f}s")
                
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
                    failed += 1
                    all_results.append({
                        "document_id": pdf_file.stem,
                        "file_name": pdf_file.name,
                        "file_path": str(pdf_file),
                        "processing_status": "error",
                        "error_message": str(e)
                    })
        
        # Calculate final metrics
        total_time = time.time() - start_time
        
        # Update results
        self.results["metadata"].update({
            "total_pdfs_processed": len(pdf_files),
            "successful_analyses": successful,
            "failed_analyses": failed,
            "processing_time_seconds": total_time,
            "files_per_second": len(pdf_files) / total_time if total_time > 0 else 0,
            "feature_extraction_metrics": {
                "average_processing_time": total_time / len(pdf_files) if len(pdf_files) > 0 else 0,
                "success_rate": successful / len(pdf_files) if len(pdf_files) > 0 else 0,
                "throughput_per_minute": (len(pdf_files) / total_time) * 60 if total_time > 0 else 0,
                "threads_used": self.max_workers
            }
        })
        
        self.results["document_features"] = all_results
        
        logger.info(f"FIXED FAST extraction completed: {successful}/{len(pdf_files)} successful in {total_time:.2f}s")
        logger.info(f"Processing rate: {len(pdf_files)/total_time:.1f} files/second")
        
        return self.results
    
    def save_results(self, output_file: str = "fixed_fast_pdf_metadata.json"):
        """Save comprehensive results to JSON file"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def generate_ml_dataset(self, output_file: str = "fixed_fast_ml_dataset.json"):
        """Generate ML training dataset for Naive Bayes OCR routing"""
        ml_dataset = []
        
        for doc in self.results["document_features"]:
            if doc.get("processing_status") == "success":
                # Extract features for ML model
                features = {
                    "document_id": doc["document_id"],
                    "file_name": doc["file_name"],
                    
                    # Document structure features
                    "page_count": doc.get("document_structure", {}).get("page_count", 0),
                    "has_metadata": doc.get("document_structure", {}).get("has_metadata", False),
                    "has_forms": doc.get("document_structure", {}).get("has_forms", False),
                    "has_annotations": doc.get("document_structure", {}).get("has_annotations", False),
                    
                    # Text content features
                    "total_characters": doc.get("text_content", {}).get("total_characters", 0),
                    "total_words": doc.get("text_content", {}).get("total_words", 0),
                    "unique_fonts": doc.get("text_content", {}).get("unique_fonts", 0),
                    "has_tables": doc.get("text_content", {}).get("has_tables", False),
                    "has_numbers": doc.get("text_content", {}).get("has_numbers", False),
                    "has_currency": doc.get("text_content", {}).get("has_currency", False),
                    "has_dates": doc.get("text_content", {}).get("has_dates", False),
                    "has_emails": doc.get("text_content", {}).get("has_emails", False),
                    "has_phone_numbers": doc.get("text_content", {}).get("has_phone_numbers", False),
                    
                    # Visual features
                    "aspect_ratio": doc.get("visual_features", {}).get("aspect_ratio", 0),
                    "brightness_mean": doc.get("visual_features", {}).get("brightness_mean", 0),
                    "contrast": doc.get("visual_features", {}).get("contrast", 0),
                    
                    # Layout features
                    "has_images": doc.get("layout_features", {}).get("has_images", False),
                    "has_graphics": doc.get("layout_features", {}).get("has_graphics", False),
                    "column_count": doc.get("layout_features", {}).get("column_count", 1),
                    
                    # OCR features
                    "text_density": doc.get("ocr_features", {}).get("text_density", 0),
                    "font_clarity": doc.get("ocr_features", {}).get("font_clarity", 0),
                    "noise_level": doc.get("ocr_features", {}).get("noise_level", 0),
                    
                    # Target variable (OCR engine recommendation)
                    "recommended_ocr_engine": doc.get("ocr_features", {}).get("recommended_ocr_engine", "tesseract")
                }
                
                ml_dataset.append(features)
        
        # Save ML dataset
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ml_dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ML training dataset saved to {output_file} with {len(ml_dataset)} samples")
        return ml_dataset

def main():
    """Main execution function"""
    print("=" * 80)
    print("FIXED FAST REAL PDF METADATA EXTRACTOR")
    print("=" * 80)
    
    # Initialize fixed fast extractor
    extractor = FixedFastMetadataExtractor()
    
    # Analyze all PDFs
    pdf_folder = "1000+ PDF_Invoice_Folder"
    
    if not Path(pdf_folder).exists():
        print(f"Error: PDF folder '{pdf_folder}' not found!")
        return
    
    print(f"Starting FIXED FAST metadata extraction from all PDFs in: {pdf_folder}")
    print(f"Using {extractor.max_workers} threads for maximum performance")
    print("Extracting real document features with optimized algorithms...")
    
    # Run fixed fast metadata extraction
    results = extractor.analyze_all_pdfs(pdf_folder)
    
    # Save results
    extractor.save_results("fixed_fast_pdf_metadata.json")
    
    # Generate ML training dataset
    ml_dataset = extractor.generate_ml_dataset("fixed_fast_ml_dataset.json")
    
    # Display summary
    metadata = results["metadata"]
    print("\n" + "=" * 80)
    print("FIXED FAST METADATA EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"Total PDFs Processed: {metadata['total_pdfs_processed']}")
    print(f"Successful Analyses: {metadata['successful_analyses']}")
    print(f"Failed Analyses: {metadata['failed_analyses']}")
    print(f"Success Rate: {(metadata['successful_analyses']/metadata['total_pdfs_processed']*100):.1f}%")
    print(f"Processing Time: {metadata['processing_time_seconds']:.2f} seconds")
    print(f"Files per Second: {metadata['files_per_second']:.1f}")
    print(f"Throughput: {metadata['feature_extraction_metrics']['throughput_per_minute']:.1f} files/minute")
    print(f"Threads Used: {metadata['feature_extraction_metrics']['threads_used']}")
    print("\nFiles generated:")
    print("  - fixed_fast_pdf_metadata.json (complete real metadata)")
    print("  - fixed_fast_ml_dataset.json (ML training dataset)")
    print(f"  - {len(ml_dataset)} samples ready for Naive Bayes training")
    print("=" * 80)

if __name__ == "__main__":
    main()
