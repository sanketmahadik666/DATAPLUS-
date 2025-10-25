#!/usr/bin/env python3
"""
Simple test of the Document Analyzer without requiring Tesseract installation
"""

import json
import os
from pathlib import Path
from datetime import datetime

class SimpleDocumentAnalyzer:
    """Simplified version for testing without external dependencies."""
    
    def __init__(self):
        self.config = {
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
                "image_ratio_threshold": 0.2
            },
            "departments": {
                "business_reporting": ["report", "progress", "competitive", "introduction"],
                "financial": ["invoice", "receipt", "payment", "financial"],
                "legal": ["contract", "agreement", "legal", "terms"],
                "medical": ["patient", "medical", "health", "prescription"],
                "academic": ["research", "paper", "thesis", "academic"]
            }
        }
    
    def analyze_document(self, image_path: str, annotation_path: str = None):
        """Analyze a document and return OCR routing features."""
        try:
            # Basic file information
            file_info = self._get_file_info(image_path)
            
            # Load annotations if available
            annotations = self._load_annotations(annotation_path) if annotation_path else None
            
            # Analyze document features based on annotations
            analysis_results = {
                **file_info,
                "num_pages": self._detect_page_count(annotations),
                "language_detected": self._detect_language_from_annotations(annotations),
                "bilingual_flag": self._detect_bilingual_from_annotations(annotations),
                "text_density": self._calculate_text_density_from_annotations(annotations),
                "table_ratio": self._calculate_table_ratio_from_annotations(annotations),
                "image_ratio": 0.0,  # Assume no images for this test
                "contains_signature": self._detect_signature_from_annotations(annotations),
                "contains_logo": self._detect_logo_from_annotations(annotations),
                "form_fields_detected": self._detect_form_fields_from_annotations(annotations),
                "layout_complexity_score": self._calculate_layout_complexity_from_annotations(annotations),
                "font_variance_score": self._calculate_font_variance_from_annotations(annotations),
                "resolution_dpi": 300,  # Assume high resolution
                "priority_level": self._determine_priority_from_annotations(annotations),
                "department_context": self._determine_department_from_annotations(annotations),
                "ocr_variant_suggestion": self._suggest_ocr_variant_from_annotations(annotations),
                "confidence_estimate": 0.85,  # High confidence with annotations
                "processing_recommendation": "standard_processing",
                "timestamp": datetime.now().isoformat() + "Z"
            }
            
            return analysis_results
            
        except Exception as e:
            return self._get_error_response(image_path, str(e))
    
    def _get_file_info(self, image_path: str):
        """Extract basic file information."""
        path_obj = Path(image_path)
        file_size = path_obj.stat().st_size if path_obj.exists() else 0
        
        return {
            "document_id": path_obj.stem,
            "file_name": path_obj.name,
            "file_size_bytes": file_size
        }
    
    def _load_annotations(self, annotation_path: str):
        """Load annotation JSON file."""
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Could not load annotations {annotation_path}: {e}")
            return None
    
    def _detect_page_count(self, annotations):
        """Detect number of pages from annotations."""
        if annotations and 'form' in annotations:
            for item in annotations['form']:
                if 'text' in item and 'page' in item['text'].lower():
                    import re
                    page_match = re.search(r'page\s+(\d+)\s+of\s+(\d+)', item['text'].lower())
                    if page_match:
                        return int(page_match.group(2))
        return 1
    
    def _detect_language_from_annotations(self, annotations):
        """Detect language from annotation text."""
        if not annotations or 'form' not in annotations:
            return 'en'
        
        # Collect all text
        all_text = ""
        for item in annotations['form']:
            if 'text' in item:
                all_text += item['text'] + " "
        
        # Simple language detection
        if any(char in all_text for char in 'а-яё'):
            return 'ru'
        elif any(char in all_text for char in '一-龯'):
            return 'zh'
        else:
            return 'en'
    
    def _detect_bilingual_from_annotations(self, annotations):
        """Detect if document is bilingual."""
        if not annotations or 'form' not in annotations:
            return False
        
        languages_found = set()
        all_text = ""
        for item in annotations['form']:
            if 'text' in item:
                all_text += item['text'] + " "
        
        if any(char in all_text for char in 'а-яё'):
            languages_found.add('ru')
        if any(char in all_text for char in '一-龯'):
            languages_found.add('zh')
        if any(char in all_text for char in 'a-zA-Z'):
            languages_found.add('en')
        
        return len(languages_found) > 1
    
    def _calculate_text_density_from_annotations(self, annotations):
        """Calculate text density from annotation bounding boxes."""
        if not annotations or 'form' not in annotations:
            return 0.5
        
        # Estimate based on number of text elements
        text_items = [item for item in annotations['form'] if 'text' in item and item['text'].strip()]
        return min(len(text_items) / 50, 1.0)  # Normalize by expected max items
    
    def _calculate_table_ratio_from_annotations(self, annotations):
        """Calculate table ratio from annotations."""
        if not annotations or 'form' not in annotations:
            return 0.0
        
        # Look for table-like structures in annotations
        table_indicators = 0
        for item in annotations['form']:
            if 'text' in item:
                text = item['text'].lower()
                if any(word in text for word in ['table', 'row', 'column', 'grid']):
                    table_indicators += 1
        
        return min(table_indicators / 10, 1.0)
    
    def _detect_signature_from_annotations(self, annotations):
        """Detect signatures from annotations."""
        if not annotations or 'form' not in annotations:
            return False
        
        for item in annotations['form']:
            if 'text' in item:
                text = item['text'].lower()
                if any(word in text for word in ['signature', 'sign', 'signed', 'authorized']):
                    return True
        return False
    
    def _detect_logo_from_annotations(self, annotations):
        """Detect logos from annotations."""
        if not annotations or 'form' not in annotations:
            return False
        
        for item in annotations['form']:
            if 'text' in item:
                text = item['text'].lower()
                if any(word in text for word in ['logo', 'brand', 'company', 'corporate']):
                    return True
        return False
    
    def _detect_form_fields_from_annotations(self, annotations):
        """Detect form fields from annotations."""
        if not annotations or 'form' not in annotations:
            return False
        
        form_indicators = ['question', 'answer', 'field', 'input']
        for item in annotations['form']:
            if 'label' in item and item['label'] in form_indicators:
                return True
        return True  # If we have form annotations, assume it's a form
    
    def _calculate_layout_complexity_from_annotations(self, annotations):
        """Calculate layout complexity from annotations."""
        if not annotations or 'form' not in annotations:
            return 0.5
        
        # Count different types of elements
        element_types = set()
        for item in annotations['form']:
            if 'label' in item:
                element_types.add(item['label'])
        
        return min(len(element_types) / 5, 1.0)
    
    def _calculate_font_variance_from_annotations(self, annotations):
        """Calculate font variance from annotation bounding boxes."""
        if not annotations or 'form' not in annotations:
            return 0.3
        
        heights = []
        for item in annotations['form']:
            if 'box' in item:
                x1, y1, x2, y2 = item['box']
                heights.append(y2 - y1)
        
        if len(heights) > 1:
            import numpy as np
            height_variance = np.var(heights) / (np.mean(heights) + 1e-6)
            return min(height_variance / 100, 1.0)
        
        return 0.3
    
    def _determine_priority_from_annotations(self, annotations):
        """Determine priority from annotations."""
        if not annotations or 'form' not in annotations:
            return 'medium'
        
        all_text = ""
        for item in annotations['form']:
            if 'text' in item:
                all_text += item['text'].lower() + " "
        
        if any(word in all_text for word in ['urgent', 'asap', 'immediate', 'critical']):
            return 'high'
        elif any(word in all_text for word in ['important', 'priority', 'deadline']):
            return 'medium'
        else:
            return 'low'
    
    def _determine_department_from_annotations(self, annotations):
        """Determine department from annotations."""
        if not annotations or 'form' not in annotations:
            return 'general'
        
        all_text = ""
        for item in annotations['form']:
            if 'text' in item:
                all_text += item['text'].lower() + " "
        
        for dept, keywords in self.config['departments'].items():
            if any(keyword in all_text for keyword in keywords):
                return dept
        
        return 'general'
    
    def _suggest_ocr_variant_from_annotations(self, annotations):
        """Suggest OCR variant from annotations."""
        has_forms = self._detect_form_fields_from_annotations(annotations)
        has_tables = self._calculate_table_ratio_from_annotations(annotations) > 0.1
        is_multilingual = self._detect_bilingual_from_annotations(annotations)
        has_handwriting = self._detect_signature_from_annotations(annotations)
        
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
    
    def _get_error_response(self, image_path: str, error_message: str):
        """Return error response."""
        path_obj = Path(image_path)
        return {
            "document_id": path_obj.stem,
            "file_name": path_obj.name,
            "file_size_bytes": path_obj.stat().st_size if path_obj.exists() else 0,
            "error": error_message,
            "timestamp": datetime.now().isoformat() + "Z"
        }

def test_analyzer():
    """Test the analyzer with the current document."""
    analyzer = SimpleDocumentAnalyzer()
    
    # Test with the current document
    image_path = "dataset/testing_data/images/82250337_0338.png"
    annotation_path = "dataset/testing_data/annotations/82250337_0338.json"
    
    print("=== Document Analysis Test ===\n")
    
    result = analyzer.analyze_document(image_path, annotation_path)
    
    print("Analysis Results:")
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    test_analyzer()
