Document Analyzer Service
========================

A microservice that accepts a path (file or directory) and returns designed analysis parameters as JSON. Supports PDFs (with conversion) and images.

Quick start
-----------

1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
. .venv/Scripts/activate
```

2) Install dependencies

```bash
pip install -r document_analyzer_service/requirements.txt
```

Windows PDF support (pick one):
- PyMuPDF only (simpler):
```bash
pip install PyMuPDF
```
- pdf2image + Poppler:
  - Download Poppler for Windows and add its `bin` to PATH, then:
```bash
pip install pdf2image
```

3) Run the service

```bash
uvicorn document_analyzer_service.main:app --host 0.0.0.0 --port 8000
```

4) Call the API

- Analyze any path:
```bash
curl -X POST "http://localhost:8000/analyze-path?path=1000+%20PDF_Invoice_Folder" -H "accept: application/json"
```

- Upload a single file:
```bash
curl -F "file=@path/to/file.pdf" http://localhost:8000/analyze-upload
```

Notes
-----
- The service reuses your existing `document_analyzer.py` for rich analysis and falls back to `test_analyzer.py` if dependencies are missing.
- For best PDF support, install either PyMuPDF or pdf2image with Poppler.


