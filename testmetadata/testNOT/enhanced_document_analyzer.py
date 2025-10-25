#!/usr/bin/env python3
"""
Enhanced Document Analysis System with Comprehensive OCR Routing Fields
This version includes many more fields for robust routing decisions.
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
import math

class EnhancedDocumentAnalyzer:
    """Enhanced document analyzer with comprehensive routing features."""
    
    def __init__(self, config_path: str = "enhanced_ocr_config.json"):
        """Initialize the enhanced document analyzer."""
        self.config = self._load_config(config_path)
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.pdf']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load enhanced configuration."""
        default_config = {
            "ocr_engines": {
                "form_optimized": {"priority": 1, "handles_forms": True},
                "table_optimized": {"priority": 2, "handles_tables": True},
                "handwriting": {"priority": 3, "handles_handwriting": True},
                "multilingual": {"priority": 4, "handles_multilingual": True},
                "financial": {"priority": 5, "handles_financial": True},
                "legal": {"priority": 6, "handles_legal": True},
                "medical": {"priority": 7, "handles_medical": True},
                "academic": {"priority": 8, "handles_academic": True},
                "invoice": {"priority": 9, "handles_invoices": True},
                "receipt": {"priority": 10, "handles_receipts": True},
                "contract": {"priority": 11, "handles_contracts": True},
                "standard": {"priority": 12, "general_purpose": True}
            },
            "thresholds": {
                "text_density_high": 0.7,
                "text_density_low": 0.3,
                "table_ratio_threshold": 0.1,
                "image_ratio_threshold": 0.2,
                "layout_complexity_high": 0.7,
                "layout_complexity_low": 0.3,
                "font_variance_high": 0.5,
                "font_variance_low": 0.2,
                "skew_threshold": 5.0,
                "noise_threshold": 0.1,
                "contrast_threshold": 0.3
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load config file: {e}. Using defaults.")
        
        return default_config
    
    def analyze_document_enhanced(self, image_path: str, annotation_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced document analysis with comprehensive routing features.
        
        Returns a comprehensive JSON object with all routing features.
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
            
            # Enhanced analysis results
            analysis_results = {
                # === BASIC DOCUMENT INFO ===
                **file_info,
                "num_pages": self._detect_page_count(image, annotations),
                "file_format": self._detect_file_format(image_path),
                "creation_date": self._get_file_creation_date(image_path),
                "last_modified": self._get_file_modified_date(image_path),
                
                # === LANGUAGE & CONTENT ANALYSIS ===
                "language_detected": self._detect_language(image, annotations),
                "language_confidence": self._calculate_language_confidence(image, annotations),
                "bilingual_flag": self._detect_bilingual(image, annotations),
                "multilingual_languages": self._detect_multiple_languages(image, annotations),
                "text_direction": self._detect_text_direction(image, annotations),
                "writing_system": self._detect_writing_system(image, annotations),
                
                # === TEXT ANALYSIS ===
                "text_density": self._calculate_text_density(image, annotations),
                "text_quality_score": self._calculate_text_quality(image, annotations),
                "character_count": self._count_characters(image, annotations),
                "word_count": self._count_words(image, annotations),
                "line_count": self._count_lines(image, annotations),
                "paragraph_count": self._count_paragraphs(image, annotations),
                "average_line_length": self._calculate_average_line_length(image, annotations),
                "text_coherence_score": self._calculate_text_coherence(image, annotations),
                
                # === LAYOUT & STRUCTURE ANALYSIS ===
                "layout_complexity_score": self._calculate_layout_complexity(image, annotations),
                "layout_type": self._detect_layout_type(image, annotations),
                "reading_order": self._detect_reading_order(image, annotations),
                "column_count": self._detect_columns(image, annotations),
                "header_footer_detected": self._detect_header_footer(image, annotations),
                "margins_detected": self._detect_margins(image, annotations),
                "alignment_type": self._detect_alignment(image, annotations),
                "spacing_analysis": self._analyze_spacing(image, annotations),
                
                # === TABLE & STRUCTURED DATA ===
                "table_ratio": self._calculate_table_ratio(image, annotations),
                "table_count": self._count_tables(image, annotations),
                "table_complexity": self._calculate_table_complexity(image, annotations),
                "table_structure_type": self._detect_table_structure(image, annotations),
                "grid_pattern_detected": self._detect_grid_pattern(image, annotations),
                "structured_data_ratio": self._calculate_structured_data_ratio(image, annotations),
                
                # === IMAGE & GRAPHICS ANALYSIS ===
                "image_ratio": self._calculate_image_ratio(image, annotations),
                "image_count": self._count_images(image, annotations),
                "graphics_complexity": self._calculate_graphics_complexity(image, annotations),
                "chart_detected": self._detect_charts(image, annotations),
                "diagram_detected": self._detect_diagrams(image, annotations),
                "photo_detected": self._detect_photos(image, annotations),
                "drawing_detected": self._detect_drawings(image, annotations),
                
                # === FORM & FIELD ANALYSIS ===
                "form_fields_detected": self._detect_form_fields(image, annotations),
                "form_field_count": self._count_form_fields(image, annotations),
                "form_field_types": self._detect_form_field_types(image, annotations),
                "form_complexity": self._calculate_form_complexity(image, annotations),
                "checkbox_count": self._count_checkboxes(image, annotations),
                "radio_button_count": self._count_radio_buttons(image, annotations),
                "text_input_count": self._count_text_inputs(image, annotations),
                "dropdown_count": self._count_dropdowns(image, annotations),
                
                # === SIGNATURE & HANDWRITING ===
                "contains_signature": self._detect_signature(image, annotations),
                "signature_count": self._count_signatures(image, annotations),
                "handwriting_detected": self._detect_handwriting(image, annotations),
                "handwriting_ratio": self._calculate_handwriting_ratio(image, annotations),
                "signature_quality": self._assess_signature_quality(image, annotations),
                "handwriting_legibility": self._assess_handwriting_legibility(image, annotations),
                
                # === LOGO & BRANDING ===
                "contains_logo": self._detect_logo(image, annotations),
                "logo_count": self._count_logos(image, annotations),
                "branding_elements": self._detect_branding_elements(image, annotations),
                "watermark_detected": self._detect_watermark(image, annotations),
                "letterhead_detected": self._detect_letterhead(image, annotations),
                
                # === FONT & TYPOGRAPHY ===
                "font_variance_score": self._calculate_font_variance(image, annotations),
                "font_families_detected": self._detect_font_families(image, annotations),
                "font_size_distribution": self._analyze_font_sizes(image, annotations),
                "bold_text_ratio": self._calculate_bold_text_ratio(image, annotations),
                "italic_text_ratio": self._calculate_italic_text_ratio(image, annotations),
                "underline_text_ratio": self._calculate_underline_text_ratio(image, annotations),
                "typography_complexity": self._calculate_typography_complexity(image, annotations),
                
                # === IMAGE QUALITY & TECHNICAL ===
                "resolution_dpi": self._estimate_resolution(image),
                "image_quality_score": self._calculate_image_quality(image),
                "contrast_score": self._calculate_contrast(image),
                "brightness_score": self._calculate_brightness(image),
                "sharpness_score": self._calculate_sharpness(image),
                "noise_level": self._calculate_noise_level(image),
                "skew_angle": self._calculate_skew_angle(image),
                "rotation_needed": self._detect_rotation_needed(image),
                "compression_artifacts": self._detect_compression_artifacts(image),
                
                # === DOCUMENT TYPE CLASSIFICATION ===
                "document_type": self._classify_document_type(image, annotations),
                "document_category": self._classify_document_category(image, annotations),
                "document_subtype": self._classify_document_subtype(image, annotations),
                "template_detected": self._detect_template(image, annotations),
                "standard_format": self._detect_standard_format(image, annotations),
                
                # === BUSINESS CONTEXT ===
                "priority_level": self._determine_priority_level(image, annotations),
                "department_context": self._determine_department_context(image, annotations),
                "business_function": self._determine_business_function(image, annotations),
                "compliance_requirements": self._detect_compliance_requirements(image, annotations),
                "sensitivity_level": self._determine_sensitivity_level(image, annotations),
                "retention_period": self._estimate_retention_period(image, annotations),
                
                # === PROCESSING RECOMMENDATIONS ===
                "ocr_variant_suggestion": self._suggest_ocr_variant(image, annotations),
                "preprocessing_needed": self._determine_preprocessing_needed(image, annotations),
                "postprocessing_needed": self._determine_postprocessing_needed(image, annotations),
                "processing_priority": self._determine_processing_priority(image, annotations),
                "processing_time_estimate": self._estimate_processing_time(image, annotations),
                "resource_requirements": self._estimate_resource_requirements(image, annotations),
                
                # === CONFIDENCE & RELIABILITY ===
                "confidence_estimate": self._calculate_confidence(image, annotations),
                "analysis_reliability": self._calculate_analysis_reliability(image, annotations),
                "data_quality_score": self._calculate_data_quality(image, annotations),
                "completeness_score": self._calculate_completeness(image, annotations),
                
                # === METADATA ===
                "processing_recommendation": self._get_processing_recommendation(image, annotations),
                "timestamp": datetime.now().isoformat() + "Z",
                "analyzer_version": "2.0.0",
                "analysis_duration_ms": 0  # Will be calculated
            }
            
            return analysis_results
            
        except Exception as e:
            logging.error(f"Error analyzing document {image_path}: {e}")
            return self._get_error_response(image_path, str(e))
    
    # === IMPLEMENTATION METHODS ===
    
    def _get_file_info(self, image_path: str) -> Dict[str, Any]:
        """Extract comprehensive file information."""
        path_obj = Path(image_path)
        stat = path_obj.stat() if path_obj.exists() else None
        
        return {
            "document_id": path_obj.stem,
            "file_name": path_obj.name,
            "file_size_bytes": stat.st_size if stat else 0,
            "file_extension": path_obj.suffix.lower(),
            "file_path": str(path_obj.absolute())
        }
    
    def _detect_file_format(self, image_path: str) -> str:
        """Detect the file format."""
        return Path(image_path).suffix.lower().lstrip('.')
    
    def _get_file_creation_date(self, image_path: str) -> str:
        """Get file creation date."""
        try:
            stat = Path(image_path).stat()
            return datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z"
        except:
            return datetime.now().isoformat() + "Z"
    
    def _get_file_modified_date(self, image_path: str) -> str:
        """Get file modification date."""
        try:
            stat = Path(image_path).stat()
            return datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z"
        except:
            return datetime.now().isoformat() + "Z"
    
    def _detect_language(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Enhanced language detection."""
        # Implementation would use advanced language detection
        return "en"  # Placeholder
    
    def _calculate_language_confidence(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate confidence in language detection."""
        return 0.85  # Placeholder
    
    def _detect_multiple_languages(self, image: np.ndarray, annotations: Optional[Dict]) -> List[str]:
        """Detect all languages present in the document."""
        return ["en"]  # Placeholder
    
    def _detect_text_direction(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Detect text direction (LTR, RTL, vertical)."""
        return "LTR"  # Placeholder
    
    def _detect_writing_system(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Detect writing system (Latin, Cyrillic, Arabic, etc.)."""
        return "Latin"  # Placeholder
    
    def _calculate_text_quality(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate overall text quality score."""
        return 0.8  # Placeholder
    
    def _count_characters(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count total characters in the document."""
        if annotations and 'form' in annotations:
            return sum(len(item.get('text', '')) for item in annotations['form'])
        return 0
    
    def _count_words(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count total words in the document."""
        if annotations and 'form' in annotations:
            return sum(len(item.get('text', '').split()) for item in annotations['form'])
        return 0
    
    def _count_lines(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count total lines in the document."""
        if annotations and 'form' in annotations:
            return len([item for item in annotations['form'] if 'text' in item])
        return 0
    
    def _count_paragraphs(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count paragraphs in the document."""
        return 1  # Placeholder
    
    def _calculate_average_line_length(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate average line length."""
        if annotations and 'form' in annotations:
            lines = [item.get('text', '') for item in annotations['form'] if 'text' in item]
            if lines:
                return sum(len(line) for line in lines) / len(lines)
        return 0.0
    
    def _calculate_text_coherence(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate text coherence score."""
        return 0.8  # Placeholder
    
    def _detect_layout_type(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Detect layout type (single-column, multi-column, etc.)."""
        return "single-column"  # Placeholder
    
    def _detect_reading_order(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Detect reading order."""
        return "top-to-bottom"  # Placeholder
    
    def _detect_columns(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Detect number of columns."""
        return 1  # Placeholder
    
    def _detect_header_footer(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect presence of headers/footers."""
        return False  # Placeholder
    
    def _detect_margins(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect document margins."""
        return True  # Placeholder
    
    def _detect_alignment(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Detect text alignment."""
        return "left"  # Placeholder
    
    def _analyze_spacing(self, image: np.ndarray, annotations: Optional[Dict]) -> Dict[str, float]:
        """Analyze spacing patterns."""
        return {"line_spacing": 1.2, "word_spacing": 1.0, "paragraph_spacing": 1.5}
    
    def _count_tables(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count number of tables."""
        return 0  # Placeholder
    
    def _calculate_table_complexity(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate table complexity."""
        return 0.0  # Placeholder
    
    def _detect_table_structure(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Detect table structure type."""
        return "none"  # Placeholder
    
    def _detect_grid_pattern(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect grid patterns."""
        return False  # Placeholder
    
    def _calculate_structured_data_ratio(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate ratio of structured data."""
        return 0.0  # Placeholder
    
    def _count_images(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count images in the document."""
        return 0  # Placeholder
    
    def _calculate_graphics_complexity(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate graphics complexity."""
        return 0.0  # Placeholder
    
    def _detect_charts(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect charts/graphs."""
        return False  # Placeholder
    
    def _detect_diagrams(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect diagrams."""
        return False  # Placeholder
    
    def _detect_photos(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect photographs."""
        return False  # Placeholder
    
    def _detect_drawings(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect drawings."""
        return False  # Placeholder
    
    def _count_form_fields(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count form fields."""
        if annotations and 'form' in annotations:
            return len([item for item in annotations['form'] if 'label' in item])
        return 0
    
    def _detect_form_field_types(self, image: np.ndarray, annotations: Optional[Dict]) -> List[str]:
        """Detect types of form fields."""
        if annotations and 'form' in annotations:
            return list(set(item.get('label', '') for item in annotations['form'] if 'label' in item))
        return []
    
    def _calculate_form_complexity(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate form complexity."""
        field_count = self._count_form_fields(image, annotations)
        return min(field_count / 20, 1.0)  # Normalize by expected max fields
    
    def _count_checkboxes(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count checkboxes."""
        return 0  # Placeholder
    
    def _count_radio_buttons(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count radio buttons."""
        return 0  # Placeholder
    
    def _count_text_inputs(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count text input fields."""
        return 0  # Placeholder
    
    def _count_dropdowns(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count dropdown fields."""
        return 0  # Placeholder
    
    def _count_signatures(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count signatures."""
        return 1 if self._detect_signature(image, annotations) else 0
    
    def _detect_handwriting(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect handwriting."""
        return self._detect_signature(image, annotations)  # Placeholder
    
    def _calculate_handwriting_ratio(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate handwriting ratio."""
        return 0.1 if self._detect_handwriting(image, annotations) else 0.0
    
    def _assess_signature_quality(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Assess signature quality."""
        return 0.8  # Placeholder
    
    def _assess_handwriting_legibility(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Assess handwriting legibility."""
        return 0.7  # Placeholder
    
    def _count_logos(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Count logos."""
        return 1 if self._detect_logo(image, annotations) else 0
    
    def _detect_branding_elements(self, image: np.ndarray, annotations: Optional[Dict]) -> List[str]:
        """Detect branding elements."""
        return ["logo"] if self._detect_logo(image, annotations) else []
    
    def _detect_watermark(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect watermarks."""
        return False  # Placeholder
    
    def _detect_letterhead(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect letterhead."""
        return False  # Placeholder
    
    def _detect_font_families(self, image: np.ndarray, annotations: Optional[Dict]) -> List[str]:
        """Detect font families."""
        return ["Arial", "Times New Roman"]  # Placeholder
    
    def _analyze_font_sizes(self, image: np.ndarray, annotations: Optional[Dict]) -> Dict[str, float]:
        """Analyze font size distribution."""
        return {"min": 10, "max": 16, "average": 12, "std_dev": 2}
    
    def _calculate_bold_text_ratio(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate bold text ratio."""
        return 0.2  # Placeholder
    
    def _calculate_italic_text_ratio(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate italic text ratio."""
        return 0.1  # Placeholder
    
    def _calculate_underline_text_ratio(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate underlined text ratio."""
        return 0.05  # Placeholder
    
    def _calculate_typography_complexity(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate typography complexity."""
        return 0.3  # Placeholder
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """Calculate overall image quality."""
        return 0.8  # Placeholder
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        return 0.7  # Placeholder
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate image brightness."""
        return 0.6  # Placeholder
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness."""
        return 0.8  # Placeholder
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """Calculate noise level."""
        return 0.1  # Placeholder
    
    def _calculate_skew_angle(self, image: np.ndarray) -> float:
        """Calculate skew angle."""
        return 0.0  # Placeholder
    
    def _detect_rotation_needed(self, image: np.ndarray) -> bool:
        """Detect if rotation is needed."""
        return False  # Placeholder
    
    def _detect_compression_artifacts(self, image: np.ndarray) -> bool:
        """Detect compression artifacts."""
        return False  # Placeholder
    
    def _classify_document_type(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Classify document type."""
        return "form"  # Placeholder
    
    def _classify_document_category(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Classify document category."""
        return "business"  # Placeholder
    
    def _classify_document_subtype(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Classify document subtype."""
        return "report"  # Placeholder
    
    def _detect_template(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect if document uses a template."""
        return True  # Placeholder
    
    def _detect_standard_format(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect if document follows a standard format."""
        return False  # Placeholder
    
    def _determine_business_function(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Determine business function."""
        return "reporting"  # Placeholder
    
    def _detect_compliance_requirements(self, image: np.ndarray, annotations: Optional[Dict]) -> List[str]:
        """Detect compliance requirements."""
        return []  # Placeholder
    
    def _determine_sensitivity_level(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Determine sensitivity level."""
        return "normal"  # Placeholder
    
    def _estimate_retention_period(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Estimate retention period in years."""
        return 7  # Placeholder
    
    def _determine_preprocessing_needed(self, image: np.ndarray, annotations: Optional[Dict]) -> List[str]:
        """Determine preprocessing steps needed."""
        return []  # Placeholder
    
    def _determine_postprocessing_needed(self, image: np.ndarray, annotations: Optional[Dict]) -> List[str]:
        """Determine postprocessing steps needed."""
        return []  # Placeholder
    
    def _determine_processing_priority(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Determine processing priority."""
        return "normal"  # Placeholder
    
    def _estimate_processing_time(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Estimate processing time in seconds."""
        return 30.0  # Placeholder
    
    def _estimate_resource_requirements(self, image: np.ndarray, annotations: Optional[Dict]) -> Dict[str, str]:
        """Estimate resource requirements."""
        return {"cpu": "medium", "memory": "medium", "storage": "low"}
    
    def _calculate_analysis_reliability(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate analysis reliability."""
        return 0.9  # Placeholder
    
    def _calculate_data_quality(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate data quality score."""
        return 0.85  # Placeholder
    
    def _calculate_completeness(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate completeness score."""
        return 0.9  # Placeholder
    
    # === EXISTING METHODS (simplified for brevity) ===
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image using OpenCV."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _load_annotations(self, annotation_path: str) -> Optional[Dict]:
        """Load annotation JSON file."""
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load annotations {annotation_path}: {e}")
            return None
    
    def _detect_page_count(self, image: np.ndarray, annotations: Optional[Dict]) -> int:
        """Detect number of pages."""
        if annotations and 'form' in annotations:
            for item in annotations['form']:
                if 'text' in item and 'page' in item['text'].lower():
                    page_match = re.search(r'page\s+(\d+)\s+of\s+(\d+)', item['text'].lower())
                    if page_match:
                        return int(page_match.group(2))
        return 1
    
    def _detect_bilingual(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect if document is bilingual."""
        return False  # Placeholder
    
    def _calculate_text_density(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate text density."""
        if annotations and 'form' in annotations:
            text_items = [item for item in annotations['form'] if 'text' in item and item['text'].strip()]
            return min(len(text_items) / 50, 1.0)
        return 0.5
    
    def _calculate_table_ratio(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate table ratio."""
        return 0.0  # Placeholder
    
    def _calculate_image_ratio(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate image ratio."""
        return 0.0  # Placeholder
    
    def _detect_signature(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect signatures."""
        if annotations and 'form' in annotations:
            for item in annotations['form']:
                if 'text' in item:
                    text = item['text'].lower()
                    if any(word in text for word in ['signature', 'sign', 'signed', 'authorized']):
                        return True
        return False
    
    def _detect_logo(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect logos."""
        if annotations and 'form' in annotations:
            for item in annotations['form']:
                if 'text' in item:
                    text = item['text'].lower()
                    if any(word in text for word in ['logo', 'brand', 'company', 'corporate']):
                        return True
        return False
    
    def _detect_form_fields(self, image: np.ndarray, annotations: Optional[Dict]) -> bool:
        """Detect form fields."""
        if annotations and 'form' in annotations:
            form_indicators = ['question', 'answer', 'field', 'input']
            for item in annotations['form']:
                if 'label' in item and item['label'] in form_indicators:
                    return True
        return False
    
    def _calculate_layout_complexity(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate layout complexity."""
        if annotations and 'form' in annotations:
            element_types = set()
            for item in annotations['form']:
                if 'label' in item:
                    element_types.add(item['label'])
            return min(len(element_types) / 5, 1.0)
        return 0.5
    
    def _calculate_font_variance(self, image: np.ndarray, annotations: Optional[Dict]) -> float:
        """Calculate font variance."""
        if annotations and 'form' in annotations:
            heights = []
            for item in annotations['form']:
                if 'box' in item:
                    x1, y1, x2, y2 = item['box']
                    heights.append(y2 - y1)
            if len(heights) > 1:
                height_variance = np.var(heights) / (np.mean(heights) + 1e-6)
                return min(height_variance / 100, 1.0)
        return 0.3
    
    def _estimate_resolution(self, image: np.ndarray) -> int:
        """Estimate resolution."""
        height, width = image.shape[:2]
        if height > 2000 or width > 2000:
            return 300
        elif height > 1000 or width > 1000:
            return 200
        else:
            return 150
    
    def _determine_priority_level(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Determine priority level."""
        return "medium"  # Placeholder
    
    def _determine_department_context(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Determine department context."""
        return "general"  # Placeholder
    
    def _suggest_ocr_variant(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Suggest OCR variant."""
        has_forms = self._detect_form_fields(image, annotations)
        has_tables = self._calculate_table_ratio(image, annotations) > 0.1
        is_multilingual = self._detect_bilingual(image, annotations)
        has_handwriting = self._detect_signature(image, annotations)
        
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
        """Calculate confidence."""
        confidence = 0.5
        if annotations and 'form' in annotations:
            confidence += 0.3
        return max(0.0, min(1.0, confidence))
    
    def _get_processing_recommendation(self, image: np.ndarray, annotations: Optional[Dict]) -> str:
        """Get processing recommendation."""
        return "standard_processing"
    
    def _get_error_response(self, image_path: str, error_message: str) -> Dict[str, Any]:
        """Return error response."""
        path_obj = Path(image_path)
        return {
            "document_id": path_obj.stem,
            "file_name": path_obj.name,
            "file_size_bytes": path_obj.stat().st_size if path_obj.exists() else 0,
            "error": error_message,
            "timestamp": datetime.now().isoformat() + "Z"
        }


def main():
    """Main function for testing the enhanced analyzer."""
    analyzer = EnhancedDocumentAnalyzer()
    
    # Test with sample document
    image_path = "dataset/testing_data/images/82250337_0338.png"
    annotation_path = "dataset/testing_data/annotations/82250337_0338.json"
    
    print("=== Enhanced Document Analysis ===")
    result = analyzer.analyze_document_enhanced(image_path, annotation_path)
    
    print(f"Analysis completed with {len(result)} fields")
    print(f"Document ID: {result['document_id']}")
    print(f"OCR Engine: {result['ocr_variant_suggestion']}")
    print(f"Confidence: {result['confidence_estimate']:.2f}")
    
    # Save enhanced results
    with open("enhanced_analysis_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("Enhanced analysis saved to: enhanced_analysis_result.json")


if __name__ == "__main__":
    main()
