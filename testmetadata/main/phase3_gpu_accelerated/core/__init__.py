"""
Phase 3: GPU Acceleration + Deep Feature Extraction Core Components

This package provides GPU-accelerated document processing with deep neural features
for intelligent OCR routing and analysis.
"""

from .gpu_metadata_processor import GPUMetadataProcessor, process_documents_gpu
from .deep_feature_extractor import DeepFeatureExtractor, extract_optimized_features, extract_deep_features
from .enhanced_ocr_dispatcher import EnhancedOCRDispatcher, create_enhanced_ocr_routing_response

__all__ = [
    # GPU Processing
    'GPUMetadataProcessor',
    'process_documents_gpu',

    # Deep Features
    'DeepFeatureExtractor',
    'extract_optimized_features',
    'extract_deep_features',

    # Enhanced OCR Routing
    'EnhancedOCRDispatcher',
    'create_enhanced_ocr_routing_response'
]

__version__ = "3.0.0"