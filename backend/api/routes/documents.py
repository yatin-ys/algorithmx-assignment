import hashlib
from fastapi import APIRouter, BackgroundTasks, File, UploadFile, HTTPException
from pydantic import BaseModel
from ...repositories.documents import insert_document, find_document_by_hash
from ..ingestion import ingestion_pipeline
from ...services.qdrant import already_indexed


# 1. Define a Pydantic model for the response.
# This enables automatic validation and documentation.
class DocumentStatusResponse(BaseModel):
    id: int
    title: str
    file_hash: str
    status: str
    page_count: int | None = None


router = APIRouter(prefix="/documents", tags=["documents"])


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@router.post("/upload", response_model=DocumentStatusResponse)
async def upload_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    file_hash = compute_sha256(data)

    existing = find_document_by_hash(file_hash)
    if existing:
        doc_id, title, status, page_count, _, _ = existing

        # 2. Add resilience: re-trigger ingestion for failed jobs or if missing from vector DB.
        if status == "failed":
            background_tasks.add_task(ingestion_pipeline, doc_id, data)
        elif status == "indexed" and not already_indexed(file_hash):
            background_tasks.add_task(ingestion_pipeline, doc_id, data)

        # 3. Return a dictionary that matches the Pydantic model.
        return {
            "id": doc_id,
            "title": title,
            "file_hash": file_hash,
            "status": status,
            "page_count": page_count,
        }

    title = file.filename or f"document-{file_hash[:8]}.pdf"
    doc_id = insert_document(title, file_hash)
    background_tasks.add_task(ingestion_pipeline, doc_id, data)

    return {
        "id": doc_id,
        "title": title,
        "file_hash": file_hash,
        "status": "queued",
        "page_count": None,
    }
