import re
import logging
import uuid
from typing import List, Tuple
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from .. import config
from .embeddings import embedding_dimension, embedding_model_name

_CLIENT: QdrantClient | None = None


def client() -> QdrantClient:
    global _CLIENT
    if _CLIENT is None:
        if config.QDRANT_URL:
            _CLIENT = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        else:
            _CLIENT = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
                api_key=config.QDRANT_API_KEY,
            )
    return _CLIENT


def collection_name() -> str:
    model = embedding_model_name()
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")
    return f"pdf_chunks__{safe}"


def ensure_collection() -> str:
    name = collection_name()
    vec_size = embedding_dimension()
    try:
        client().get_collection(name)
    except Exception:
        client().create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
        )
    return name


def already_indexed(file_hash: str) -> bool:
    try:
        res = client().count(
            collection_name=collection_name(),
            count_filter=Filter(
                must=[
                    FieldCondition(key="file_hash", match=MatchValue(value=file_hash))
                ]
            ),
            exact=True,
        )
        return (res.count or 0) > 0
    except Exception as e:
        logging.error(
            f"Failed to check Qdrant index for hash {file_hash}: {e}", exc_info=True
        )
        return False


def upsert_points(points: List[PointStruct]) -> None:
    name = ensure_collection()
    for i in range(0, len(points), config.QDRANT_UPSERT_BATCH):
        client().upsert(
            collection_name=name, points=points[i : i + config.QDRANT_UPSERT_BATCH]
        )


def build_points(
    doc_id: int,
    doc_title: str,
    file_hash: str,
    texts_with_meta: List[Tuple[str, int, int]],
    vectors: List[List[float]],
) -> List[PointStruct]:
    now = datetime.utcnow().isoformat() + "Z"
    points: List[PointStruct] = []
    namespace = uuid.UUID("a7463525-4a6c-48b8-b12e-2f5a5e334335")

    for (text, page, chunk_idx), vec in zip(texts_with_meta, vectors):
        point_name = f"{file_hash}:{page}:{chunk_idx}"
        point_id = str(uuid.uuid5(namespace, point_name))

        points.append(
            PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "page": page,
                    "chunk_id": chunk_idx,
                    "chunk_text": text,
                    "file_hash": file_hash,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        )
    return points
