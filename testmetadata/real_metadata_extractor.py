#!/usr/bin/env python3
"""
Real PDF Metadata Extractor for Naive Bayes OCR Routing
Extracts actual document features for machine learning-based OCR engine selection
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealMetadataExtractor:
    """Extract real metadata from PDF documents for ML-based OCR routing"""
    
    def __init__(self, max_workers: int = 16):
        self.max_workers = max_workers
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
        self.latencies = []
    
    def extract_real_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract real metadata from PDF document"""
        start_time = time.time()
        
        try:
            # Open PDF document
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
            
            # Extract real features
            features = self._extract_document_features(doc, pdf_path)
            doc_info.update(features)
            
            doc.close()
            return doc_info
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {
                "document_id": pdf_path.stem,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "processing_status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time
            }
        finally:
            latency = time.time() - start_time
            self.latencies.append(latency)
    
    def _extract_document_features(self, doc: fitz.Document, pdf_path: Path) -> Dict[str, Any]:
        """Extract real features from PDF document"""
        features = {}
        
        # 1. Document Structure Features
        features["document_structure"] = self._analyze_document_structure(doc)
        
        # 2. Text Content Features
        features["text_content"] = self._analyze_text_content(doc)
        
        # 3. Visual Features (from first page)
        if len(doc) > 0:
            page = doc[0]
            features["visual_features"] = self._analyze_visual_features(page, pdf_path)
        
        # 4. Layout Features
        features["layout_features"] = self._analyze_layout_features(doc)
        
        # 5. OCR-specific features
        features["ocr_features"] = self._analyze_ocr_features(doc, pdf_path)
        
        return features
    
    def _analyze_document_structure(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze document structure"""
        return {
            "page_count": len(doc),
            "has_metadata": bool(doc.metadata),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "title": doc.metadata.get("title", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "author": doc.metadata.get("author", ""),
            "has_outline": len(doc.get_toc()) > 0,
            "outline_levels": len(doc.get_toc()),
            "has_forms": any(page.get_drawings() for page in doc),
            "has_annotations": any(page.annots() for page in doc)
        }
    
    def _analyze_text_content(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze text content across all pages"""
        all_text = ""
        text_blocks = []
        font_info = []
        
        for page_num in range(min(3, len(doc))):  # Analyze first 3 pages
            page = doc[page_num]
            page_text = page.get_text()
            all_text += page_text + " "
            
            # Get text blocks with formatting
            blocks = page.get_text("dict")
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_blocks.append({
                                "text": span["text"],
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"]
                            })
                            font_info.append({
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"]
                            })
        
        # Analyze text characteristics
        words = all_text.split()
        sentences = re.split(r'[.!?]+', all_text)
        
        return {
            "total_characters": len(all_text),
            "total_words": len(words),
            "total_sentences": len([s for s in sentences if s.strip()]),
            "average_word_length": np.mean([len(word) for word in words]) if words else 0,
            "unique_fonts": len(set(f["font"] for f in font_info)),
            "font_sizes": list(set(f["size"] for f in font_info)),
            "text_blocks_count": len(text_blocks),
            "has_tables": self._detect_tables_in_text(all_text),
            "has_numbers": bool(re.search(r'\d+', all_text)),
            "has_currency": bool(re.search(r'[$€£¥₹]', all_text)),
            "has_dates": bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', all_text)),
            "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', all_text)),
            "has_phone_numbers": bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', all_text)),
            "language_indicators": self._detect_language_indicators(all_text)
        }
    
    def _analyze_visual_features(self, page: fitz.Page, pdf_path: Path) -> Dict[str, Any]:
        """Analyze visual features from first page"""
        try:
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to OpenCV format
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": "Could not decode image"}
            
            # Analyze image characteristics
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # Text region detection
            text_regions = self._detect_text_regions(gray)
            
            # Color analysis
            color_features = self._analyze_colors(img)
            
            return {
                "image_dimensions": {"width": width, "height": height},
                "aspect_ratio": width / height,
                "edge_density": float(edge_density),
                "text_regions_count": len(text_regions),
                "text_region_density": len(text_regions) / (height * width) * 1000000,
                "color_features": color_features,
                "brightness_mean": float(np.mean(gray)),
                "brightness_std": float(np.std(gray)),
                "contrast": float(np.std(gray))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing visual features: {e}")
            return {"error": str(e)}
    
    def _analyze_layout_features(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze layout features"""
        layout_features = {
            "has_headers": False,
            "has_footers": False,
            "has_sidebars": False,
            "column_count": 1,
            "has_tables": False,
            "has_images": False,
            "has_graphics": False
        }
        
        for page_num in range(min(2, len(doc))):
            page = doc[page_num]
            
            # Check for images
            image_list = page.get_images()
            if image_list:
                layout_features["has_images"] = True
            
            # Check for drawings/graphics
            drawings = page.get_drawings()
            if drawings:
                layout_features["has_graphics"] = True
            
            # Analyze text layout
            text_dict = page.get_text("dict")
            layout_features.update(self._analyze_text_layout(text_dict))
        
        return layout_features
    
    def _analyze_ocr_features(self, doc: fitz.Document, pdf_path: Path) -> Dict[str, Any]:
        """Analyze features specific to OCR processing"""
        ocr_features = {
            "text_extraction_quality": 0.0,
            "font_clarity": 0.0,
            "text_density": 0.0,
            "noise_level": 0.0,
            "skew_angle": 0.0,
            "recommended_ocr_engine": "tesseract"
        }
        
        try:
            if len(doc) > 0:
                page = doc[0]
                
                # Get text with confidence
                text = page.get_text()
                ocr_features["text_density"] = len(text) / (page.rect.width * page.rect.height) * 1000000
                
                # Analyze font characteristics
                font_analysis = self._analyze_font_characteristics(page)
                ocr_features.update(font_analysis)
                
                # Determine best OCR engine based on features
                ocr_features["recommended_ocr_engine"] = self._recommend_ocr_engine(ocr_features)
                
        except Exception as e:
            logger.error(f"Error analyzing OCR features: {e}")
        
        return ocr_features
    
    def _detect_tables_in_text(self, text: str) -> bool:
        """Detect if text contains table-like structures"""
        # Look for patterns that suggest tables
        table_patterns = [
            r'\|\s*\w+\s*\|',  # Pipe-separated columns
            r'\s{3,}\w+\s{3,}',  # Space-separated columns
            r'\t\w+\t',  # Tab-separated columns
        ]
        return any(re.search(pattern, text) for pattern in table_patterns)
    
    def _detect_language_indicators(self, text: str) -> Dict[str, int]:
        """Detect language indicators in text"""
        # Common words in different languages
        language_indicators = {
            "english": len(re.findall(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', text.lower())),
            "spanish": len(re.findall(r'\b(el|la|de|en|y|o|pero|con|por|para)\b', text.lower())),
            "german": len(re.findall(r'\b(der|die|das|und|oder|aber|in|auf|zu|für|mit|von)\b', text.lower())),
            "french": len(re.findall(r'\b(le|la|de|en|et|ou|mais|dans|sur|à|pour|avec|par)\b', text.lower()))
        }
        return language_indicators
    
    def _detect_text_regions(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in image"""
        try:
            # Use EAST text detector or simple contour detection
            contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_regions = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 10:  # Filter small regions
                    text_regions.append((x, y, w, h))
            
            return text_regions
        except:
            return []
    
    def _analyze_colors(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze color characteristics"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            return {
                "dominant_colors": len(np.unique(img.reshape(-1, 3), axis=0)),
                "brightness_variance": float(np.var(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))),
                "color_channels": {
                    "b_mean": float(np.mean(img[:, :, 0])),
                    "g_mean": float(np.mean(img[:, :, 1])),
                    "r_mean": float(np.mean(img[:, :, 2]))
                }
            }
        except:
            return {"error": "Color analysis failed"}
    
    def _analyze_text_layout(self, text_dict: Dict) -> Dict[str, Any]:
        """Analyze text layout patterns"""
        layout = {
            "has_headers": False,
            "has_footers": False,
            "column_count": 1
        }
        
        # Simple heuristics for layout detection
        if "blocks" in text_dict:
            blocks = text_dict["blocks"]
            y_positions = []
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "bbox" in line:
                            y_positions.append(line["bbox"][1])
            
            if y_positions:
                # Detect columns based on x-position clustering
                layout["column_count"] = self._detect_columns(text_dict)
        
        return layout
    
    def _detect_columns(self, text_dict: Dict) -> int:
        """Detect number of columns in text"""
        try:
            x_positions = []
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        if "bbox" in line:
                            x_positions.append(line["bbox"][0])
            
            if len(x_positions) < 10:
                return 1
            
            # Simple clustering to detect columns
            x_positions = sorted(x_positions)
            clusters = []
            current_cluster = [x_positions[0]]
            
            for i in range(1, len(x_positions)):
                if x_positions[i] - x_positions[i-1] > 50:  # Gap threshold
                    clusters.append(current_cluster)
                    current_cluster = [x_positions[i]]
                else:
                    current_cluster.append(x_positions[i])
            clusters.append(current_cluster)
            
            return min(len(clusters), 3)  # Cap at 3 columns
        except:
            return 1
    
    def _analyze_font_characteristics(self, page: fitz.Page) -> Dict[str, Any]:
        """Analyze font characteristics for OCR quality assessment"""
        try:
            text_dict = page.get_text("dict")
            fonts = []
            sizes = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            fonts.append(span.get("font", ""))
                            sizes.append(span.get("size", 0))
            
            if fonts and sizes:
                return {
                    "unique_fonts": len(set(fonts)),
                    "font_size_range": [min(sizes), max(sizes)],
                    "average_font_size": np.mean(sizes),
                    "font_size_variance": np.var(sizes)
                }
            else:
                return {"unique_fonts": 0, "font_size_range": [0, 0], "average_font_size": 0, "font_size_variance": 0}
        except:
            return {"error": "Font analysis failed"}
    
    def _recommend_ocr_engine(self, ocr_features: Dict[str, Any]) -> str:
        """Recommend OCR engine based on extracted features"""
        # Simple rule-based recommendation (can be replaced with ML model)
        if ocr_features.get("text_density", 0) > 100:
            return "paddleocr"  # Better for dense text
        elif ocr_features.get("font_clarity", 0) > 0.8:
            return "tesseract"  # Good for clear fonts
        else:
            return "easyocr"  # Better for complex layouts
    
    def analyze_all_pdfs(self, pdf_folder: str) -> Dict[str, Any]:
        """Analyze all PDFs and extract real metadata"""
        logger.info(f"Starting REAL metadata extraction from PDFs in: {pdf_folder}")
        
        start_time = time.time()
        
        # Discover all PDF files
        target_path = Path(pdf_folder)
        pdf_files = list(target_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_folder}")
            return self.results
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        logger.info(f"Using {self.max_workers} parallel workers for real feature extraction")
        
        # Process files in parallel
        all_results = []
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(self.extract_real_metadata, pdf_file): pdf_file 
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
                    
                    # Progress logging every 25 files
                    if (i + 1) % 25 == 0:
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
                "throughput_per_minute": (len(pdf_files) / total_time) * 60 if total_time > 0 else 0
            }
        })
        
        self.results["document_features"] = all_results
        
        logger.info(f"REAL metadata extraction completed: {successful}/{len(pdf_files)} successful in {total_time:.2f}s")
        
        return self.results
    
    def save_results(self, output_file: str = "real_pdf_metadata.json"):
        """Save comprehensive results to JSON file"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def generate_ml_dataset(self, output_file: str = "ml_training_dataset.json"):
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
                    "edge_density": doc.get("visual_features", {}).get("edge_density", 0),
                    "text_region_density": doc.get("visual_features", {}).get("text_region_density", 0),
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
    print("REAL PDF METADATA EXTRACTOR FOR NAIVE BAYES OCR ROUTING")
    print("=" * 80)
    
    # Initialize real metadata extractor
    extractor = RealMetadataExtractor(max_workers=32)
    
    # Analyze all PDFs
    pdf_folder = "1000+ PDF_Invoice_Folder"
    
    if not Path(pdf_folder).exists():
        print(f"Error: PDF folder '{pdf_folder}' not found!")
        return
    
    print(f"Starting REAL metadata extraction from all PDFs in: {pdf_folder}")
    print("Extracting actual document features for ML-based OCR routing...")
    
    # Run real metadata extraction
    results = extractor.analyze_all_pdfs(pdf_folder)
    
    # Save results
    extractor.save_results("real_pdf_metadata.json")
    
    # Generate ML training dataset
    ml_dataset = extractor.generate_ml_dataset("ml_ocr_routing_dataset.json")
    
    print("=" * 80)
    print("REAL METADATA EXTRACTION COMPLETE!")
    print("Files generated:")
    print("  - real_pdf_metadata.json (complete real metadata)")
    print("  - ml_ocr_routing_dataset.json (ML training dataset)")
    print(f"  - {len(ml_dataset)} samples ready for Naive Bayes training")
    print("=" * 80)

if __name__ == "__main__":
    main()
