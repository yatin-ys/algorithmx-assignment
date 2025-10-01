from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ...repositories.sessions import get_session_messages, get_session_runs

router = APIRouter(prefix="/sessions", tags=["sessions"])


class MessageResponse(BaseModel):
    id: int
    role: str
    text: str
    created_at: Optional[str]


class MetricsResponse(BaseModel):
    latency_ms_total: Optional[int]
    latency_ms_embed: Optional[int]
    latency_ms_qdrant: Optional[int]
    latency_ms_llm: Optional[int]
    sources_found: Optional[bool]


class RunResponse(BaseModel):
    id: int
    question: str
    answer: str
    model: str
    top_k: int
    only_if_sources: bool
    created_at: Optional[str]
    metrics: Optional[MetricsResponse]


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
def get_messages(session_id: str):
    """Get all messages for a session."""
    try:
        messages = get_session_messages(session_id)
        return messages
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch messages: {str(e)}"
        )


@router.get("/{session_id}/runs", response_model=List[RunResponse])
def get_runs(session_id: str):
    """Get all runs (executions) for a session with metrics."""
    try:
        runs = get_session_runs(session_id)
        return runs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch runs: {str(e)}")
