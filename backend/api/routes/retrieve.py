from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from ...services.embeddings import embed_texts
from ...services.qdrant import client, collection_name
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from ... import config

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


class RetrievalRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(
        default=config.TOP_K_DEFAULT,
        ge=1,
        le=50,
        description="Number of results to return",
    )
    doc_ids: Optional[List[int]] = Field(
        default=None, description="Optional filter by document IDs"
    )


class RetrievalResult(BaseModel):
    text: str
    doc_id: int
    doc_title: str
    page: int
    score: float
    chunk_id: int


class RetrievalResponse(BaseModel):
    results: List[RetrievalResult]
    query: str
    top_k: int


@router.post("/", response_model=RetrievalResponse)
def retrieve_documents(request: RetrievalRequest):
    """
    Retrieve relevant document chunks for a given query.
    """
    try:
        # Embed the query
        query_vector = embed_texts([request.query])[0]

        # Build filter if doc_ids provided
        search_filter = None
        if request.doc_ids:
            search_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=request.doc_ids))]
            )

        # Search Qdrant
        search_results = client().search(
            collection_name=collection_name(),
            query_vector=query_vector,
            limit=request.top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in search_results:
            payload = hit.payload
            if not payload:
                continue
            results.append(
                RetrievalResult(
                    text=payload.get("chunk_text", ""),
                    doc_id=payload.get("doc_id", 0),
                    doc_title=payload.get("doc_title", ""),
                    page=payload.get("page", 0),
                    score=hit.score,
                    chunk_id=payload.get("chunk_id", 0),
                )
            )

        return RetrievalResponse(
            results=results, query=request.query, top_k=request.top_k
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
