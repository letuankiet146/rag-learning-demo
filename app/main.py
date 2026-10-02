
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import get_document, init_db, list_documents
from .parsers import llma_parsing
from .embedding import index_document
from .chat import chat

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Document Parser MVP")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    
@app.post("/api/chat")
def chat_api(request: ChatRequest):
    result = chat(
        question=request.question,
        limit=request.top_k,
    )

    return result
@app.on_event("startup")
def startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/documents")
async def upload_document(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".text"}:
        raise HTTPException(400, "Only PDF, DOCX and TXT are supported.")

    document_id = str(uuid4())
    safe_name = Path(file.filename or "document").name
    stored_name = f"{document_id}{suffix}"
    stored_path = UPLOAD_DIR / stored_name

    with stored_path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            output.write(chunk)

    try:
        document_list = llma_parsing(stored_path)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not parse document: {exc}") from exc

    original_url = str(request.base_url).rstrip("/") + f"/api/documents/{document_id}/file"
    
    # qdrant_result = False
    qdrant_result = index_document(
        document_id=document_id,
        document_name=safe_name,
        documents=document_list,
    )

    return {
        "id": document_id,
        "filename": safe_name,
        "original_url": original_url,
        "documents": document_list,
        "qdrant": qdrant_result,
    }


@app.get("/api/documents")
def documents():
    return {"items": list_documents()}


@app.get("/api/documents/{document_id}")
def document(document_id: str):
    item = get_document(document_id)
    if not item:
        raise HTTPException(404, "Document not found.")
    return item


@app.get("/api/documents/{document_id}/file")
def original_file(document_id: str):
    item = get_document(document_id)
    if not item:
        raise HTTPException(404, "Document not found.")

    path = Path(item["original_path"])
    if not path.exists():
        raise HTTPException(404, "Original file is missing.")

    return FileResponse(
        path,
        media_type=item["content_type"],
        filename=item["filename"],
    )
