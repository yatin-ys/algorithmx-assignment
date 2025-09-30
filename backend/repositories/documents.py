import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional, Tuple, Iterator
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


def insert_document(title: str, file_hash: str) -> int:
    with get_db_conn() as conn:
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


def find_document_by_hash(file_hash: str) -> Optional[Tuple]:
    with get_db_conn() as conn:
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


def get_document(doc_id: int) -> Optional[Tuple]:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
				SELECT id, title, file_hash, status, page_count, created_at, updated_at
				FROM documents
				WHERE id = %s
				LIMIT 1;
				""",
                (doc_id,),
            )
            return cur.fetchone()


def update_status(doc_id: int, status: str, page_count: int | None = None) -> None:
    with get_db_conn() as conn:
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
