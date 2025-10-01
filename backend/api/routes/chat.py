from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import uuid
from ...services.embeddings import embed_texts
from ...services.qdrant import client, collection_name

# Import the new LangChain-based answer generator
from ...services.llm import generate_answer_with_history
from ...repositories.sessions import (
    ensure_session,
    insert_message,
    insert_run,
    insert_retrievals,
    insert_metrics,
    get_session_messages,  # Import function to get history
)
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from ... import config

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID for tracking conversations"
    )
    message: str = Field(..., description="User's question or message")
    top_k: int = Field(
        default=config.TOP_K_DEFAULT,
        ge=1,
        le=50,
        description="Number of context chunks to retrieve",
    )
    filter_doc_ids: Optional[List[int]] = Field(
        default=None, description="Optional filter by document IDs"
    )
    model: Optional[str] = Field(
        default=None, description="Groq model name (defaults to config)"
    )
    only_if_sources: bool = Field(
        default=False,
        description="If True, abstain from answering if sources are insufficient",
    )
    temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="Model temperature"
    )


class Citation(BaseModel):
    doc_title: str
    page: int
    doc_id: Optional[int] = None


class ContextChunk(BaseModel):
    text: str
    doc_id: int
    doc_title: str
    page: int
    score: float
    chunk_id: int


class Metrics(BaseModel):
    latency_ms_total: int
    latency_ms_embed: int
    latency_ms_qdrant: int
    latency_ms_llm: int
    sources_found: bool


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    context_chunks: List[ContextChunk]
    session_id: str
    model: str
    run_id: int
    metrics: Metrics


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat endpoint with grounded RAG using Groq and LangChain.
    """
    t_start = time.perf_counter()

    session_id = request.session_id or str(uuid.uuid4())
    ensure_session(session_id)

    # Fetch conversation history from the database
    chat_history = get_session_messages(session_id)

    insert_message(session_id, "user", request.message)

    try:
        t_embed_start = time.perf_counter()
        query_vector = embed_texts([request.message])[0]
        t_embed_end = time.perf_counter()
        latency_ms_embed = int((t_embed_end - t_embed_start) * 1000)

        search_filter = None
        if request.filter_doc_ids:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id", match=MatchAny(any=request.filter_doc_ids)
                    )
                ]
            )

        t_qdrant_start = time.perf_counter()
        search_results = client().search(
            collection_name=collection_name(),
            query_vector=query_vector,
            limit=request.top_k,
            query_filter=search_filter,
            with_payload=True,
        )
        t_qdrant_end = time.perf_counter()
        latency_ms_qdrant = int((t_qdrant_end - t_qdrant_start) * 1000)

        context_chunks = []
        for hit in search_results:
            payload = hit.payload
            if not payload:
                continue
            context_chunks.append(
                {
                    "text": payload.get("chunk_text", ""),
                    "doc_id": payload.get("doc_id", 0),
                    "doc_title": payload.get("doc_title", ""),
                    "page": payload.get("page", 0),
                    "score": hit.score,
                    "chunk_id": payload.get("chunk_id", 0),
                }
            )

        # Generate answer with the new LangChain service
        t_llm_start = time.perf_counter()
        result = generate_answer_with_history(
            question=request.message,
            context_chunks=context_chunks,
            chat_history=chat_history,  # Pass the history
            model_name=request.model,
            only_if_sources=request.only_if_sources,
            temperature=request.temperature,
        )
        t_llm_end = time.perf_counter()
        latency_ms_llm = int((t_llm_end - t_llm_start) * 1000)

        t_end = time.perf_counter()
        latency_ms_total = int((t_end - t_start) * 1000)

        citations = [Citation(**cite) for cite in result["citations"]]
        chunks = [ContextChunk(**chunk) for chunk in result["context_chunks"]]
        model_used = request.model or config.GROQ_MODEL
        answer = result["answer"]

        insert_message(session_id, "assistant", answer)

        run_id = insert_run(
            session_id=session_id,
            question=request.message,
            answer=answer,
            model=model_used,
            top_k=request.top_k,
            only_if_sources=request.only_if_sources,
        )

        insert_retrievals(run_id, context_chunks)

        sources_found = len(context_chunks) > 0
        insert_metrics(
            run_id=run_id,
            latency_ms_total=latency_ms_total,
            latency_ms_embed=latency_ms_embed,
            latency_ms_qdrant=latency_ms_qdrant,
            latency_ms_llm=latency_ms_llm,
            sources_found=sources_found,
        )

        return ChatResponse(
            answer=answer,
            citations=citations,
            context_chunks=chunks,
            session_id=session_id,
            model=model_used,
            run_id=run_id,
            metrics=Metrics(
                latency_ms_total=latency_ms_total,
                latency_ms_embed=latency_ms_embed,
                latency_ms_qdrant=latency_ms_qdrant,
                latency_ms_llm=latency_ms_llm,
                sources_found=sources_found,
            ),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")
