import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Iterator
from datetime import datetime
import json
from .. import config

db_pool = pool.SimpleConnectionPool(
    1,  # min connections
    20,  # max connections
    host=config.POSTGRES_HOST,
    port=config.POSTGRES_PORT,
    dbname=config.POSTGRES_DB,
    user=config.POSTGRES_USER,
    password=config.POSTGRES_PASSWORD,
)


@contextmanager
def get_db_conn() -> Iterator[psycopg2.extensions.connection]:
    """Provides a database connection from the pool."""
    conn = None
    try:
        conn = db_pool.getconn()
        yield conn
    finally:
        if conn:
            db_pool.putconn(conn)


def ensure_session(session_id: str, settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Ensure a session exists. If not, create it.
    Returns the session_id.
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Check if session exists
            cur.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
            if cur.fetchone():
                return session_id

            # Create new session
            settings_json = json.dumps(settings) if settings else None
            cur.execute(
                "INSERT INTO sessions (id, settings_json) VALUES (%s, %s)",
                (session_id, settings_json),
            )
            conn.commit()
            return session_id


def insert_message(session_id: str, role: str, text: str) -> int:
    """Insert a message and return its ID."""
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (session_id, role, text)
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (session_id, role, text),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("Failed to insert message")
            message_id = row[0]
            conn.commit()
            return message_id


def insert_run(
    session_id: str,
    question: str,
    answer: str,
    model: str,
    top_k: int,
    only_if_sources: bool,
) -> int:
    """Insert a run record and return its ID."""
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (session_id, question, answer, model, top_k, only_if_sources)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (session_id, question, answer, model, top_k, only_if_sources),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("Failed to insert run")
            run_id = row[0]
            conn.commit()
            return run_id


def insert_retrievals(run_id: int, chunks: List[Dict[str, Any]]) -> None:
    """Insert retrieval records for a run."""
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for rank, chunk in enumerate(chunks, start=1):
                cur.execute(
                    """
                    INSERT INTO retrievals (run_id, rank, score, doc_id, page, chunk_id)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (
                        run_id,
                        rank,
                        chunk.get("score", 0.0),
                        chunk.get("doc_id", 0),
                        chunk.get("page", 0),
                        chunk.get("chunk_id", 0),
                    ),
                )
            conn.commit()


def insert_metrics(
    run_id: int,
    latency_ms_total: int,
    latency_ms_embed: int,
    latency_ms_qdrant: int,
    latency_ms_llm: int,
    sources_found: bool,
) -> None:
    """Insert metrics for a run."""
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO metrics (
                    run_id, latency_ms_total, latency_ms_embed, 
                    latency_ms_qdrant, latency_ms_llm, sources_found_bool
                )
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (
                    run_id,
                    latency_ms_total,
                    latency_ms_embed,
                    latency_ms_qdrant,
                    latency_ms_llm,
                    sources_found,
                ),
            )
            conn.commit()


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Get all messages for a session, ordered by creation time."""
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, role, text, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at ASC;
                """,
                (session_id,),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "role": row[1],
                    "text": row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                }
                for row in rows
            ]


def get_session_runs(session_id: str) -> List[Dict[str, Any]]:
    """Get all runs for a session with their metrics."""
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    r.id, r.question, r.answer, r.model, r.top_k, 
                    r.only_if_sources, r.created_at,
                    m.latency_ms_total, m.latency_ms_embed,
                    m.latency_ms_qdrant, m.latency_ms_llm, m.sources_found_bool
                FROM runs r
                LEFT JOIN metrics m ON r.id = m.run_id
                WHERE r.session_id = %s
                ORDER BY r.created_at DESC;
                """,
                (session_id,),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "question": row[1],
                    "answer": row[2],
                    "model": row[3],
                    "top_k": row[4],
                    "only_if_sources": row[5],
                    "created_at": row[6].isoformat() if row[6] else None,
                    "metrics": (
                        {
                            "latency_ms_total": row[7],
                            "latency_ms_embed": row[8],
                            "latency_ms_qdrant": row[9],
                            "latency_ms_llm": row[10],
                            "sources_found": row[11],
                        }
                        if row[7] is not None
                        else None
                    ),
                }
                for row in rows
            ]
