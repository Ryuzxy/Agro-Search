"""AGRO-SEARCH FastAPI backend.

Endpoints:
    POST /api/search         - text query → TF-IDF ranked docs
    POST /api/classify       - image upload → CNN disease label
    POST /api/classify-and-search - image → CNN → auto-query → ranked docs
    GET  /api/documents      - list all corpus documents
    GET  /api/health         - health check
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from utils.search_engine import SearchEngine
from utils.classifier import PlantClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load engines
    app.state.search_engine = SearchEngine()
    app.state.classifier = PlantClassifier()
    yield


app = FastAPI(title="AGRO-SEARCH", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend build)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class SearchRequest(BaseModel):
    query: str
    top_n: int = 5


@app.get("/api/health")
def health():
    model_loaded = app.state.classifier.model is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/api/documents")
def list_documents():
    return app.state.search_engine.documents


@app.post("/api/search")
def search(req: SearchRequest):
    results = app.state.search_engine.search(req.query, req.top_n)
    return {
        "query": req.query,
        "total_results": len(results),
        "results": results,
    }


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        result = app.state.classifier.predict(tmp.name)
        return result
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    finally:
        os.unlink(tmp.name)


@app.post("/api/classify-and-search")
async def classify_and_search(file: UploadFile = File(...), top_n: int = Form(5)):
    """Hybrid flow: image → CNN classification → auto-query → TF-IDF search."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        classification = app.state.classifier.predict(tmp.name)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    finally:
        os.unlink(tmp.name)

    if classification["is_healthy"]:
        return {
            "classification": classification,
            "message": "Tanaman terdeteksi sehat. Tidak diperlukan pencarian dokumen penanganan.",
            "results": [],
        }

    search_results = app.state.search_engine.search(classification["query"], top_n)
    return {
        "classification": classification,
        "query_generated": classification["query"],
        "total_results": len(search_results),
        "results": search_results,
    }
