# Document Analysis System for OCR Engine Routing

This system automatically analyzes documents and determines the optimal OCR engine routing based on document characteristics.

## Features

- **Automatic Document Analysis**: Analyzes images and annotations to extract OCR routing features
- **Multi-Engine Support**: Routes to appropriate OCR engines (form-optimized, table-optimized, handwriting, multilingual, standard)
- **Batch Processing**: Process multiple documents at once
- **Configurable**: Customizable routing rules and thresholds
- **Comprehensive Analysis**: Detects language, form fields, tables, signatures, logos, and more

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

## Quick Start

### Analyze a Single Document

```python
from document_analyzer import DocumentAnalyzer

analyzer = DocumentAnalyzer()
result = analyzer.analyze_document("path/to/image.png", "path/to/annotations.json")
print(result)
```

### Batch Process Multiple Documents

```python
analyzer = DocumentAnalyzer()
results = analyzer.batch_analyze("path/to/directory", "output.json")
```

### Command Line Usage

```bash
# Analyze single document
python document_analyzer.py image.png -a annotations.json -o result.json

# Batch process directory
python document_analyzer.py directory/ -o batch_results.json
```

## Output Format

The system returns a JSON object with the following structure:

```json
{
  "document_id": "string",
  "file_name": "string", 
  "file_size_bytes": "integer",
  "num_pages": "integer",
  "language_detected": "string",
  "bilingual_flag": "boolean",
  "text_density": "float",
  "table_ratio": "float",
  "image_ratio": "float",
  "contains_signature": "boolean",
  "contains_logo": "boolean",
  "form_fields_detected": "boolean",
  "layout_complexity_score": "float",
  "font_variance_score": "float",
  "resolution_dpi": "integer",
  "priority_level": "string",
  "department_context": "string",
  "ocr_variant_suggestion": "string",
  "confidence_estimate": "float",
  "processing_recommendation": "string",
  "timestamp": "ISO8601"
}
```

## Configuration

Create a `ocr_config.json` file to customize routing rules:

```json
{
  "ocr_engines": {
    "form_optimized": {"priority": 1, "handles_forms": true},
    "table_optimized": {"priority": 2, "handles_tables": true},
    "handwriting": {"priority": 3, "handles_handwriting": true},
    "multilingual": {"priority": 4, "handles_multilingual": true},
    "standard": {"priority": 5, "general_purpose": true}
  },
  "thresholds": {
    "text_density_high": 0.7,
    "text_density_low": 0.3,
    "table_ratio_threshold": 0.1,
    "image_ratio_threshold": 0.2
  },
  "departments": {
    "business_reporting": ["report", "progress", "competitive"],
    "financial": ["invoice", "receipt", "payment"],
    "legal": ["contract", "agreement", "legal"],
    "medical": ["patient", "medical", "health"],
    "academic": ["research", "paper", "thesis"]
  }
}
```

## Supported File Formats

- PNG
- JPEG/JPG
- TIFF
- PDF (basic support)

## Examples

See `example_usage.py` for comprehensive usage examples.

## OCR Engine Routing Logic

The system routes documents to OCR engines based on:

1. **Form Fields Detected** → `form_optimized`
2. **High Table Ratio** → `table_optimized` 
3. **Signatures/Handwriting** → `handwriting`
4. **Multiple Languages** → `multilingual`
5. **Default** → `standard`

## Error Handling

The system gracefully handles errors and returns partial results when possible. Check the `error` field in the output for any issues.

## Performance Notes

- Image analysis is optimized for typical document sizes
- Batch processing is recommended for large datasets
- Consider image preprocessing for better results with low-quality scans
