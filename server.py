"""
AutoMLOps Genie — FastAPI backend
Serves React SPA on / and pipeline API on /api/*
"""
import io
import os
import sys
import uuid
import threading
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from genie_agent.genie_main import genie_respond
from pipelines.pipeline_builder import load_recent_mlflow_runs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoMLOps Genie API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict = {}
_lock = threading.Lock()


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


# ── Analyze ───────────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    job_id = str(uuid.uuid4())[:8]
    _set(job_id, {"status": "running", "progress": 5, "stage": "Parsing prompt with Gemini function calling…"})

    def _run():
        try:
            _set(job_id, {"progress": 10, "stage": "Gemini function calling — extracting ML config…"})

            result, parsed, model_info_path, leaderboard, model_dir = genie_respond(prompt, df)

            _set(job_id, {"progress": 88, "stage": "Computing feature importance…"})

            # ── Feature importance (correlation-based, sent as JSON for Chart.js) ──
            target_col = parsed.get("target", "")
            feature_importance = []
            if target_col and target_col in df.columns:
                from sklearn.preprocessing import LabelEncoder
                df_enc = df.copy()
                for col in df_enc.select_dtypes(include=["object"]).columns:
                    if col != target_col:
                        le = LabelEncoder()
                        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                if df_enc[target_col].dtype == object:
                    le = LabelEncoder()
                    df_enc[target_col] = le.fit_transform(df_enc[target_col].astype(str))
                corr = (
                    df_enc.corr()[target_col]
                    .abs()
                    .drop(target_col, errors="ignore")
                    .sort_values(ascending=False)
                )
                feature_importance = [
                    {"feature": k, "importance": round(float(v), 4)}
                    for k, v in corr.items()
                ]

            # ── Leaderboard ──
            lb_data = []
            if leaderboard is not None and not leaderboard.empty:
                lb_data = leaderboard.to_dict(orient="records")
                # Normalise score to 0-1 range for progress bars
                scores = [r.get("score_val", 0) for r in lb_data]
                max_s = max(abs(s) for s in scores) if scores else 1
                for r in lb_data:
                    r["score_pct"] = round(abs(r.get("score_val", 0)) / max_s * 100, 1)

            # ── Model info text ──
            model_info = ""
            abs_path = None
            if model_info_path:
                abs_path = str(Path(model_info_path).resolve())
                if os.path.exists(abs_path):
                    with open(abs_path) as f:
                        model_info = f.read()

            _set(job_id, {
                "status": "complete",
                "progress": 100,
                "stage": "Pipeline complete!",
                "result": result,
                "parsed": parsed,
                "leaderboard": lb_data,
                "feature_importance": feature_importance,
                "model_info": model_info,
                "_model_info_path": abs_path,
            })

        except Exception as exc:
            logger.exception("Pipeline failed")
            _set(job_id, {"status": "error", "progress": 0,
                          "stage": str(exc), "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    with _lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        raise HTTPException(404, "Job not found")
    job.pop("_model_info_path", None)   # internal field, not for client
    return job


@app.get("/api/download/{job_id}")
def download(job_id: str):
    with _lock:
        job = dict(_jobs.get(job_id, {}))
    if not job or job.get("status") != "complete":
        raise HTTPException(404, "Job not complete")
    path = job.get("_model_info_path")
    if path and os.path.exists(path):
        return FileResponse(path, filename="model_info.txt", media_type="text/plain")
    raise HTTPException(404, "Model file not found")


@app.get("/api/runs")
def runs():
    try:
        df = load_recent_mlflow_runs(limit=6)
        if df.empty:
            return {"runs": []}
        df = df.fillna("—")
        df["start_time"] = df["start_time"].astype(str).str[:19]
        return {"runs": df.to_dict(orient="records")}
    except Exception:
        return {"runs": []}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _set(job_id: str, patch: dict):
    with _lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(patch)


# ── Serve sample datasets ─────────────────────────────────────────────────────
data_dir = Path(__file__).parent / "data"
if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

# ── Serve React SPA ───────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="spa")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=True)
