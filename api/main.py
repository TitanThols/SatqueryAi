"""
api/main.py
FastAPI backend for SatQuery AI.
Endpoints:
  POST /analyze   — upload images + query → ExecutionTrace JSON
  GET  /tools     — list registered tools
  GET  /health    — health check
  GET  /report    — PDF report download (after /analyze)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SatQuery AI",
    description="Agentic AI for satellite image analysis (ISRO/SAC PS-26167)",
    version="1.0.0",
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for generated masks/overlays
RESULTS_DIR = Path(os.environ.get("SATQUERY_RESULTS_DIR", "/tmp/satquery_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return {"status": "ok", "device": device, "version": "1.0.0"}


# ── Tool registry endpoint ────────────────────────────────────────────────────

@app.get("/tools")
async def list_tools():
    from models.registry.tool_registry import TOOLS
    return {"tools": TOOLS}


# ── Main analysis endpoint ────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    files:   list[UploadFile] = File(...),
    roles:   str              = Form(...),     # JSON array e.g. '["t1","t2"]'
    query:   str              = Form(...),
    session: str              = Form(default=""),
):
    """
    Upload one or more satellite images and ask a question.

    - files: GeoTIFF / PNG / JPEG uploads
    - roles: JSON array of role strings matching files in order
             Options: "single", "t1", "t2", "optical", "sar"
    - query: Natural language question
    - session: Optional session ID (for result retrieval)
    """
    # ── Parse roles ────────────────────────────────────────────────────────────
    try:
        roles_list: list[str] = json.loads(roles)
    except json.JSONDecodeError:
        raise HTTPException(400, f"'roles' must be a JSON array string, got: {roles!r}")

    if len(files) != len(roles_list):
        raise HTTPException(400, f"Number of files ({len(files)}) must match number of roles ({len(roles_list)}).")

    # ── Create session upload directory ───────────────────────────────────────
    session_id = session or str(uuid.uuid4())
    upload_dir = RESULTS_DIR / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # ── Save uploaded files ────────────────────────────────────────────────────
    file_paths: list[Path] = []
    for i, upload in enumerate(files):
        suffix = Path(upload.filename or "upload").suffix or ".tif"
        dest   = upload_dir / f"image_{i}{suffix}"
        try:
            with open(dest, "wb") as f:
                content = await upload.read()
                f.write(content)
            file_paths.append(dest)
            logger.info("Saved %s → %s", upload.filename, dest)
        except Exception as e:
            raise HTTPException(500, f"Failed to save file {upload.filename}: {e}")

    # ── Run agent ──────────────────────────────────────────────────────────────
    try:
        from agent.controller.router import AgentController
        controller = AgentController()
        trace = controller.run(
            query=query,
            file_paths=file_paths,
            roles=roles_list,
            upload_dir=upload_dir,
        )
    except Exception as e:
        logger.exception("Agent error")
        raise HTTPException(500, f"Agent error: {e}")

    # ── Generate PDF report ────────────────────────────────────────────────────
    try:
        from api.report_generator import generate_pdf_report
        pdf_path = upload_dir / "report.pdf"
        generate_pdf_report(trace, pdf_path)
        trace_dict = trace.to_dict()
        trace_dict["report_url"] = f"/results/{session_id}/report.pdf"
    except Exception as e:
        logger.warning("PDF generation failed: %s", e)
        trace_dict = trace.to_dict()

    trace_dict["session_id"] = session_id

    # Make image_outputs into relative URLs
    image_urls = []
    for p in trace.image_outputs:
        rel = Path(p).relative_to(RESULTS_DIR)
        image_urls.append(f"/results/{rel}")
    trace_dict["image_urls"] = image_urls

    return JSONResponse(content=trace_dict)


# ── PDF report download ───────────────────────────────────────────────────────

@app.get("/report/{session_id}")
async def get_report(session_id: str):
    pdf = RESULTS_DIR / session_id / "report.pdf"
    if not pdf.exists():
        raise HTTPException(404, "Report not found. Run /analyze first.")
    return FileResponse(str(pdf), media_type="application/pdf", filename="satquery_report.pdf")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
