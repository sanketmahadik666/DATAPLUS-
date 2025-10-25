#!/usr/bin/env python3
"""
Document Analysis System for OCR Engine Routing
Analyzes document metadata, images, and annotations to determine optimal OCR engine routing.
"""

import json
import os
import cv2
import numpy as np
from PIL import Image, ImageStat
import pytesseract
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """Main class for analyzing documents and determining OCR routing features."""
    
    def __init__(self, config_path: str = "ocr_config.json"):
        """Initialize the document analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.pdf']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file for OCR routing rules."""
        default_config = {
            "ocr_engines": {
                "form_optimized": {"priority": 1, "handles_forms": True},
                "table_optimized": {"priority": 2, "handles_tables": True},
                "handwriting": {"priority": 3, "handles_handwriting": True},
                "multilingual": {"priority": 4, "handles_multilingual": True},
                "standard": {"priority": 5, "general_purpose": True}
            },
            "thresholds": {
                "text_density_high": 0.7,
                "text_density_low": 0.3,
                "table_ratio_threshold": 0.1,
                "image_ratio_threshold": 0.2,
                "layout_complexity_high": 0.7,
                "layout_complexity_low": 0.3
            },
            "departments": {
                "business_reporting": ["report", "progress", "competitive", "introduction"],
                "financial": ["invoice", "receipt", "payment", "financial"],
                "legal": ["contract", "agreement", "legal", "terms"],
                "medical": ["patient", "medical", "health", "prescription"],
                "academic": ["research", "paper", "thesis", "academic"]
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")
        
        return default_config
    
    def analyze_document(self, image_path: str, annotation_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a document and return OCR routing features.
        
        Args:
            image_path: Path to the document image
            annotation_path: Optional path to annotation JSON file
            
        Returns:
            Dictionary containing all OCR routing features
        """
        try:
            # Basic file information
            file_info = self._get_file_info(image_path)
            
            # Load and analyze image
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Load annotations if available
            annotations = self._load_annotations(annotation_path) if annotation_path else None
            
            # Analyze document features
            analysis_results = {
                **file_info,
                "num_pages": self._detect_page_count(image, annotations),
                "language_detected": self._detect_language(image, annotations),
                "bilingual_flag": self._detect_bilingual(image, annotations),
                "text_density": self._calculate_text_density(image, annotations),
                "table_ratio": self._calculate_table_ratio(image, annotations),
                "image_ratio": self._calculate_image_ratio(image, annotations),
                "contains_signature": self._detect_signature(image, annotations),
                "contains_logo": self._detect_logo(image, annotations),
                "form_fields_detected": self._detect_form_fields(image, annotations),
                "layout_complexity_score": self._calculate_layout_complexity(image, annotations),
                "font_variance_score": self._calculate_font_variance(image, annotations),
                "resolution_dpi": self._estimate_resolution(image),
                "priority_level": self._determine_priority_level(image, annotations),
                "department_context": self._determine_department_context(image, annotations),
                "ocr_variant_suggestion": self._suggest_ocr_variant(image, annotations),
                "confidence_estimate": self._calculate_confidence(image, annotations),
                "processing_recommendation": self._get_processing_recommendation(image, annotations),
                "timestamp": datetime.now().isoformat() + "Z"
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing document {image_path}: {e}")
            return self._get_error_response(image_path, str(e))
    
    def _get_file_info(self, image_path: str) -> Dict[str, Any]:
        """Extract basic file information."""
        path_obj = Path(image_path)
        file_size = path_obj.stat().st_size if path_obj.exists() else 0
        
        return {
            "document_id": path_obj.stem,
            "file_name": path_obj.name,
            "file_size_bytes": file_size
        }
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image using OpenCV."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _load_annotations(self, annotation_path: str) -> Optional[Dict]:
        """Load annotation JSON file."""
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load annotations {annotation_path}: {e}")
            return None
    
    def _detect_page_count(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Detect number of pages in the document."""
        if annotations and 'form' in annotations:
            # Look for page indicators in annotations
            for item in annotations['form']:
                if 'text' in item and 'page' in item['text'].lower():
                    # Extract page number from text like "PAGE 1 OF 2"
                    page_match = re.search(r'page\s+(\d+)\s+of\s+(\d+)', item['text'].lower())
                    if page_match:
                        return int(page_match.group(2))
        
        # Default to 1 page if no page indicators found
        return 1
    
    def _detect_language(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Detect the primary language of the document."""
        try:
            # Use OCR to detect language
            text = pytesseract.image_to_string(image, config='--psm 6')
            
            # Simple language detection based on common patterns
            if re.search(r'[а-яё]', text.lower()):
                return 'ru'
            elif re.search(r'[一-龯]', text):
                return 'zh'
            elif re.search(r'[あ-ん]', text):
                return 'ja'
            elif re.search(r'[가-힣]', text):
                return 'ko'
            elif re.search(r'[ء-ي]', text):
                return 'ar'
            else:
                return 'en'  # Default to English
        except Exception:
            return 'en'  # Default fallback
    
    def _detect_bilingual(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect if document contains multiple languages."""
        try:
            text = pytesseract.image_to_string(image, config='--psm 6')
            
            # Check for multiple language patterns
            languages_found = set()
            
            if re.search(r'[а-яё]', text.lower()):
                languages_found.add('ru')
            if re.search(r'[一-龯]', text):
                languages_found.add('zh')
            if re.search(r'[あ-ん]', text):
                languages_found.add('ja')
            if re.search(r'[가-힣]', text):
                languages_found.add('ko')
            if re.search(r'[ء-ي]', text):
                languages_found.add('ar')
            if re.search(r'[a-zA-Z]', text):
                languages_found.add('en')
            
            return len(languages_found) > 1
        except Exception:
            return False
    
    def _calculate_text_density(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate the density of text in the document."""
        if annotations and 'form' in annotations:
            # Calculate based on annotation bounding boxes
            total_text_area = 0
            image_area = image.shape[0] * image.shape[1]
            
            for item in annotations['form']:
                if 'box' in item and 'text' in item:
                    x1, y1, x2, y2 = item['box']
                    text_area = (x2 - x1) * (y2 - y1)
                    total_text_area += text_area
            
            return min(total_text_area / image_area, 1.0)
        
        # Fallback: use OCR to estimate text density
        try:
            # Convert to grayscale and apply threshold
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate text pixels vs total pixels
            text_pixels = np.sum(binary == 0)
            total_pixels = binary.shape[0] * binary.shape[1]
            
            return min(text_pixels / total_pixels, 1.0)
        except Exception:
            return 0.5  # Default fallback
    
    def _calculate_table_ratio(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate the ratio of table content in the document."""
        try:
            # Use OpenCV to detect table structures
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Calculate table area
            table_area = np.sum(horizontal_lines > 0) + np.sum(vertical_lines > 0)
            total_area = gray.shape[0] * gray.shape[1]
            
            return min(table_area / total_area, 1.0)
        except Exception:
            return 0.0
    
    def _calculate_image_ratio(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate the ratio of image content in the document."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect non-text regions (images, graphics)
            # Use edge detection to find image regions
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours that might be images
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            image_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small contours
                    image_area += area
            
            total_area = gray.shape[0] * gray.shape[1]
            return min(image_area / total_area, 1.0)
        except Exception:
            return 0.0
    
    def _detect_signature(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect if document contains signatures."""
        try:
            # Look for signature patterns in annotations
            if annotations and 'form' in annotations:
                for item in annotations['form']:
                    if 'text' in item:
                        text_lower = item['text'].lower()
                        if any(word in text_lower for word in ['signature', 'sign', 'signed', 'authorized']):
                            return True
            
            # Use image analysis to detect signature regions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for handwritten-like patterns (more irregular than printed text)
            # This is a simplified approach - in practice, you'd use more sophisticated ML models
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:  # Signature-sized regions
                    # Check for irregularity (handwritten characteristics)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity < 0.3:  # Low circularity suggests handwriting
                            return True
            
            return False
        except Exception:
            return False
    
    def _detect_logo(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect if document contains logos."""
        try:
            # Look for logo-related text in annotations
            if annotations and 'form' in annotations:
                for item in annotations['form']:
                    if 'text' in item:
                        text_lower = item['text'].lower()
                        if any(word in text_lower for word in ['logo', 'brand', 'company', 'corporate']):
                            return True
            
            # Use image analysis to detect logo regions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for distinct regions that might be logos
            # Logos typically have high contrast and distinct shapes
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 20000:  # Logo-sized regions
                    # Check for rectangular or square shapes (common for logos)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 2.0:  # Reasonable logo aspect ratio
                        return True
            
            return False
        except Exception:
            return False
    
    def _detect_form_fields(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect if document contains form fields."""
        if annotations and 'form' in annotations:
            # Check for form field indicators in annotations
            form_indicators = ['question', 'answer', 'field', 'input', 'checkbox', 'radio']
            
            for item in annotations['form']:
                if 'label' in item and item['label'] in form_indicators:
                    return True
                
                if 'text' in item:
                    text_lower = item['text'].lower()
                    if any(indicator in text_lower for indicator in [':', '?', '___', '[]', '()']):
                        return True
            
            return True  # If we have form annotations, assume it's a form
        
        # Fallback: use image analysis to detect form fields
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for form field patterns (lines, boxes, etc.)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # If we find many lines, it might be a form
            line_count = np.sum(horizontal_lines > 0) + np.sum(vertical_lines > 0)
            return line_count > 1000  # Threshold for form detection
        except Exception:
            return False
    
    def _calculate_layout_complexity(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate the complexity of the document layout."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges and contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate complexity based on number and variety of contours
            complexity_score = 0.0
            
            if len(contours) > 0:
                # Normalize by image size
                image_area = gray.shape[0] * gray.shape[1]
                contour_density = len(contours) / (image_area / 10000)  # Normalize
                complexity_score += min(contour_density / 10, 0.5)
                
                # Calculate area variance of contours
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
                if len(areas) > 1:
                    area_variance = np.var(areas) / (np.mean(areas) + 1e-6)
                    complexity_score += min(area_variance / 1000, 0.5)
            
            return min(complexity_score, 1.0)
        except Exception:
            return 0.5  # Default moderate complexity
    
    def _calculate_font_variance(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate the variance in font sizes and styles."""
        if annotations and 'form' in annotations:
            # Calculate font variance based on bounding box sizes
            heights = []
            widths = []
            
            for item in annotations['form']:
                if 'box' in item:
                    x1, y1, x2, y2 = item['box']
                    heights.append(y2 - y1)
                    widths.append(x2 - x1)
            
            if len(heights) > 1:
                height_variance = np.var(heights) / (np.mean(heights) + 1e-6)
                width_variance = np.var(widths) / (np.mean(widths) + 1e-6)
                return min((height_variance + width_variance) / 2, 1.0)
        
        # Fallback: estimate based on image analysis
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use OCR to get text regions and estimate font variance
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            heights = [int(h) for h in data['height'] if int(h) > 0]
            if len(heights) > 1:
                height_variance = np.var(heights) / (np.mean(heights) + 1e-6)
                return min(height_variance / 100, 1.0)
            
            return 0.3  # Default moderate variance
        except Exception:
            return 0.3
    
    def _estimate_resolution(self, image: np.ndarray) -> int:
        """Estimate the DPI/resolution of the image."""
        try:
            height, width = image.shape[:2]
            
            # Estimate DPI based on image dimensions and typical document sizes
            # This is a rough estimation - in practice, you'd need metadata
            if height > 2000 or width > 2000:
                return 300  # High resolution
            elif height > 1000 or width > 1000:
                return 200  # Medium resolution
            else:
                return 150  # Low resolution
        except Exception:
            return 200  # Default medium resolution
    
    def _determine_priority_level(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Determine the priority level for processing."""
        try:
            # Extract text for priority analysis
            text = pytesseract.image_to_string(image, config='--psm 6').lower()
            
            # High priority keywords
            high_priority = ['urgent', 'asap', 'immediate', 'critical', 'emergency']
            medium_priority = ['important', 'priority', 'deadline', 'due']
            
            if any(word in text for word in high_priority):
                return 'high'
            elif any(word in text for word in medium_priority):
                return 'medium'
            else:
                return 'low'
        except Exception:
            return 'medium'  # Default medium priority
    
    def _determine_department_context(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Determine the department context based on content."""
        try:
            text = pytesseract.image_to_string(image, config='--psm 6').lower()
            
            # Check against department keywords
            for dept, keywords in self.config['departments'].items():
                if any(keyword in text for keyword in keywords):
                    return dept
            
            return 'general'
        except Exception:
            return 'general'
    
    def _suggest_ocr_variant(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Suggest the best OCR variant based on analysis."""
        # Analyze document characteristics
        has_forms = self._detect_form_fields(image, annotations)
        has_tables = self._calculate_table_ratio(image, annotations) > 0.1
        is_multilingual = self._detect_bilingual(image, annotations)
        has_handwriting = self._detect_signature(image, annotations)
        
        # Route to appropriate OCR engine
        if has_forms:
            return 'form_optimized'
        elif has_tables:
            return 'table_optimized'
        elif has_handwriting:
            return 'handwriting'
        elif is_multilingual:
            return 'multilingual'
        else:
            return 'standard'
    
    def _calculate_confidence(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate confidence estimate for the analysis."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have annotations
        if annotations and 'form' in annotations:
            confidence += 0.3
        
        # Adjust based on image quality
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Calculate image sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > 1000:  # Sharp image
                confidence += 0.1
            elif laplacian_var < 100:  # Blurry image
                confidence -= 0.1
        except Exception:
            pass
        
        return max(0.0, min(1.0, confidence))
    
    def _get_processing_recommendation(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Get processing recommendation based on analysis."""
        text_density = self._calculate_text_density(image, annotations)
        layout_complexity = self._calculate_layout_complexity(image, annotations)
        
        if text_density > 0.7 and layout_complexity < 0.3:
            return 'fast_processing'
        elif text_density < 0.3 or layout_complexity > 0.7:
            return 'careful_processing'
        else:
            return 'standard_processing'
    
    def _get_error_response(self, image_path: str, error_message: str) -> Dict[str, Any]:
        """Return error response with basic information."""
        path_obj = Path(image_path)
        return {
            "document_id": path_obj.stem,
            "file_name": path_obj.name,
            "file_size_bytes": path_obj.stat().st_size if path_obj.exists() else 0,
            "error": error_message,
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    def batch_analyze(self, input_dir: str, output_file: str = "analysis_results.json"):
        """Analyze multiple documents in a directory."""
        results = []
        
        for file_path in Path(input_dir).rglob("*"):
            if file_path.suffix.lower() in self.supported_formats:
                # Look for corresponding annotation file
                annotation_path = file_path.parent / "annotations" / f"{file_path.stem}.json"
                annotation_path = annotation_path if annotation_path.exists() else None
                
                logger.info(f"Analyzing: {file_path}")
                result = self.analyze_document(str(file_path), str(annotation_path) if annotation_path else None)
                results.append(result)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis complete. Results saved to {output_file}")
        return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Analysis for OCR Routing")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-a", "--annotations", help="Annotation JSON file (optional)")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-c", "--config", help="Configuration file", default="ocr_config.json")
    
    args = parser.parse_args()
    
    analyzer = DocumentAnalyzer(args.config)
    
    if os.path.isfile(args.input):
        # Single file analysis
        result = analyzer.analyze_document(args.input, args.annotations)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(result, indent=2))
    
    elif os.path.isdir(args.input):
        # Batch analysis
        output_file = args.output or "batch_analysis_results.json"
        analyzer.batch_analyze(args.input, output_file)
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
