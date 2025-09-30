import logging
from typing import List, Any, cast
import fitz
from ..repositories.documents import update_status, get_document
from ..services.embeddings import embed_texts
from ..services.qdrant import (
    ensure_collection,
    already_indexed,
    build_points,
    upsert_points,
)
from .chunking import chunk_text


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

        update_status(doc_id, "embedding", page_count=page_count)

        doc_row = get_document(doc_id)
        if not doc_row:
            # This would be a serious internal error
            logging.error(f"Document {doc_id} disappeared during ingestion.")
            raise RuntimeError(f"Document {doc_id} not found")
        _, doc_title, file_hash, *_ = doc_row

        ensure_collection()

        if already_indexed(file_hash):
            logging.warning(
                f"Skipping indexing for doc_id {doc_id} as file_hash {file_hash} is already indexed in Qdrant."
            )
            update_status(doc_id, "indexed", page_count=page_count)
            return

        texts_with_meta = []
        texts = []
        for page_idx, chunks in enumerate(page_chunks, start=1):
            for chunk_idx, text in enumerate(chunks):
                if text.strip():
                    texts_with_meta.append((text, page_idx, chunk_idx))
                    texts.append(text)
        if texts:
            vectors = embed_texts(texts)
            points = build_points(
                doc_id, doc_title, file_hash, texts_with_meta, vectors
            )
            upsert_points(points)

        update_status(doc_id, "indexed", page_count=page_count)
    except Exception as e:
        logging.error(
            f"Ingestion pipeline failed for doc_id {doc_id}: {e}", exc_info=True
        )
        update_status(doc_id, "failed")
