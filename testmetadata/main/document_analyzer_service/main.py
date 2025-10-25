import os
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

from .service_analyzer import analyze_path

logger = logging.getLogger(__name__)

app = FastAPI(title="Document Analyzer Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze-path")
def analyze_path_endpoint(path: str, output: Optional[str] = None) -> JSONResponse:
    if not path:
        return JSONResponse(status_code=400, content={"error": "path is required"})
    result = analyze_path(path)
    if output:
        out_path = Path(output)
        out_path.write_text(JSONResponse(content=result).body.decode('utf-8'), encoding='utf-8')
    return JSONResponse(content=result)


@app.post("/analyze-upload")
async def analyze_upload(file: UploadFile = File(...)) -> JSONResponse:
    tmp_dir = Path(".upload_tmp")
    tmp_dir.mkdir(exist_ok=True)
    dest = tmp_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)
    payload = analyze_path(str(dest))
    try:
        dest.unlink(missing_ok=True)
    except Exception:
        pass
    return JSONResponse(content=payload)


def get_uvicorn_command(host: str = "0.0.0.0", port: int = 8000) -> str:
    return f"uvicorn document_analyzer_service.main:app --host {host} --port {port}"


