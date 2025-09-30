import hashlib
import os
from datetime import datetime
from typing import Optional, List, Any, cast

import fitz  # pymupdf
import psycopg2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from .chunking import chunk_text


app = FastAPI(title="AlgorithmX RAG Backend", version="0.1.0")


def db_conn():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ragdb")
    user = os.getenv("POSTGRES_USER", "rag")
    password = os.getenv("POSTGRES_PASSWORD", "ragpass")
    return psycopg2.connect(
        f"host={host} port={port} dbname={db} user={user} password={password}"
    )


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def insert_document(title: str, file_hash: str) -> int:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
				INSERT INTO documents (title, file_hash, status)
				VALUES (%s, %s, 'queued')
				RETURNING id;
				""",
                (title, file_hash),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Insert did not return id")
            doc_id = row[0]
            conn.commit()
            return doc_id


def find_document_by_hash(file_hash: str) -> Optional[tuple]:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
				SELECT id, title, status, page_count, created_at, updated_at
				FROM documents
				WHERE file_hash = %s
				LIMIT 1;
				""",
                (file_hash,),
            )
            return cur.fetchone()


def update_status(doc_id: int, status: str, page_count: Optional[int] = None) -> None:
    with db_conn() as conn:
        with conn.cursor() as cur:
            if page_count is None:
                cur.execute(
                    "UPDATE documents SET status = %s, updated_at = now() WHERE id = %s;",
                    (status, doc_id),
                )
            else:
                cur.execute(
                    "UPDATE documents SET status = %s, page_count = %s, updated_at = now() WHERE id = %s;",
                    (status, page_count, doc_id),
                )
            conn.commit()


def parse_pdf_and_chunk(pdf_bytes: bytes) -> tuple[int, List[List[str]]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_chunks: List[List[str]] = []
    for page in doc:
        text = cast(Any, page).get_text("text")
        chunks = chunk_text(text)
        page_chunks.append(chunks)
    return doc.page_count, page_chunks


def ingestion_pipeline(doc_id: int, pdf_bytes: bytes) -> None:
    try:
        update_status(doc_id, "parsing")
        page_count, page_chunks = parse_pdf_and_chunk(pdf_bytes)

        # Phase 4 will embed/index. For Phase 3, simulate the step and move on.
        update_status(doc_id, "embedding", page_count=page_count)

        # If we wanted to validate something here, we could; for Phase 3 just succeed.
        update_status(doc_id, "indexed", page_count=page_count)
    except Exception:
        update_status(doc_id, "failed")


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    file_hash = compute_sha256(data)

    # Idempotency: if already present, return existing record
    existing = find_document_by_hash(file_hash)
    if existing:
        doc_id, title, status, page_count, created_at, updated_at = existing
        # If a previous run failed, re-run ingestion in background
        if status in ("failed",):
            background_tasks.add_task(ingestion_pipeline, doc_id, data)
        return JSONResponse(
            {
                "id": doc_id,
                "title": title,
                "file_hash": file_hash,
                "status": status,
                "page_count": page_count,
            }
        )

    title = file.filename or f"document-{file_hash[:8]}.pdf"
    doc_id = insert_document(title, file_hash)
    background_tasks.add_task(ingestion_pipeline, doc_id, data)
    return JSONResponse(
        {"id": doc_id, "title": title, "file_hash": file_hash, "status": "queued"}
    )
