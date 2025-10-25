import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Ensure parent directory is on path to import existing analyzers
PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from document_analyzer import DocumentAnalyzer  # full analyzer with OCR/image
    ANALYZER_VARIANT = 'document'
except Exception:
    from test_analyzer import SimpleDocumentAnalyzer as DocumentAnalyzer  # fallback
    ANALYZER_VARIANT = 'simple'

logger = logging.getLogger(__name__)

# Optional PDF helpers
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except Exception:
    HAS_PDF2IMAGE = False

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf'}


def discover_inputs(target: Path) -> List[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_EXTS else []
    if target.is_dir():
        return [p for p in target.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
    return []


def convert_pdf_to_images(pdf_path: Path) -> List[Path]:
    images: List[Path] = []
    try:
        if HAS_PDF2IMAGE:
            imgs = convert_from_path(pdf_path, dpi=200, fmt='PNG', first_page=1, last_page=3)
            out_paths: List[Path] = []
            tmp_dir = pdf_path.parent / '.svc_tmp'
            tmp_dir.mkdir(exist_ok=True)
            for idx, pil_img in enumerate(imgs, start=1):
                out_path = tmp_dir / f"{pdf_path.stem}_p{idx}.png"
                pil_img.save(out_path, 'PNG')
                out_paths.append(out_path)
            return out_paths
        if HAS_PYMUPDF:
            doc = fitz.open(pdf_path)
            tmp_dir = pdf_path.parent / '.svc_tmp'
            tmp_dir.mkdir(exist_ok=True)
            max_pages = min(3, len(doc))
            for i in range(max_pages):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                out_path = tmp_dir / f"{pdf_path.stem}_p{i+1}.png"
                pix.save(str(out_path))
                images.append(out_path)
            doc.close()
            return images
    except Exception as e:
        logger.warning(f"PDF conversion failed for {pdf_path.name}: {e}")
    return []


def analyze_file(analyzer: DocumentAnalyzer, file_path: Path) -> Dict[str, Any]:
    try:
        suffix = file_path.suffix.lower()
        primary_input = str(file_path)
        conversion_info: Dict[str, Any] = {}

        if suffix == '.pdf':
            images = convert_pdf_to_images(file_path)
            if images:
                primary_input = str(images[0])
                conversion_info = {
                    'conversion_method': 'pdf2image' if HAS_PDF2IMAGE else ('pymupdf' if HAS_PYMUPDF else 'none'),
                    'images_generated': len(images),
                    'primary_image_analyzed': primary_input
                }
            else:
                conversion_info = {
                    'conversion_method': 'none',
                    'images_generated': 0,
                    'primary_image_analyzed': primary_input
                }

        result = analyzer.analyze_document(primary_input)

        stat = file_path.stat() if file_path.exists() else None
        generic = {
            'file_path': str(file_path),
            'file_extension': file_path.suffix,
            'created_date': datetime.fromtimestamp(stat.st_ctime).isoformat() if stat else None,
            'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else None,
            'source_format': 'pdf' if suffix == '.pdf' else 'image'
        }

        return {
            **result,
            **generic,
            **conversion_info,
            'processing_status': 'success',
            'analyzer_variant': ANALYZER_VARIANT
        }
    except Exception as e:
        logger.error(f"Error analyzing {file_path.name}: {e}")
        return {
            'document_id': file_path.stem,
            'file_name': file_path.name,
            'file_path': str(file_path),
            'processing_status': 'error',
            'error_message': str(e),
            'analyzer_variant': ANALYZER_VARIANT
        }


def analyze_path(input_path: str) -> Dict[str, Any]:
    target = Path(input_path)
    analyzer = DocumentAnalyzer()

    files = discover_inputs(target)
    results: List[Dict[str, Any]] = []
    for idx, f in enumerate(files, start=1):
        if idx % 50 == 0 or idx == len(files):
            logger.info(f"Processing {idx}/{len(files)}")
        results.append(analyze_file(analyzer, f))

    summary = {
        'total_inputs': len(files),
        'successful': sum(1 for r in results if r.get('processing_status') == 'success'),
        'failed': sum(1 for r in results if r.get('processing_status') == 'error'),
        'processed_at': datetime.now().isoformat() + 'Z'
    }

    return {
        'summary': summary,
        'results': results,
        'metadata': {
            'input_path': str(Path(input_path).resolve()),
            'analyzer_variant': ANALYZER_VARIANT,
            'pdf2image_available': HAS_PDF2IMAGE,
            'pymupdf_available': HAS_PYMUPDF
        }
    }


