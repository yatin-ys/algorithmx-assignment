import os
import sys
from pathlib import Path

import psycopg2


def conninfo_from_env() -> str:
    HOST = os.getenv("POSTGRES_HOST", "localhost")
    PORT = os.getenv("POSTGRES_PORT", "5432")
    DB = os.getenv("POSTGRES_DB", "ragdb")
    USER = os.getenv("POSTGRES_USER", "rag")
    PASSWORD = os.getenv("POSTGRES_PASSWORD", "ragpass")
    return f"host={HOST} port={PORT} dbname={DB} user={USER} password={PASSWORD}"


def ensure_schema_migrations(cur) -> None:
    cur.execute(
        """
		CREATE TABLE IF NOT EXISTS schema_migrations (
			filename TEXT PRIMARY KEY,
			applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
		);
		"""
    )


def migration_applied(cur, filename: str) -> bool:
    cur.execute("SELECT 1 FROM schema_migrations WHERE filename = %s;", (filename,))
    return cur.fetchone() is not None


def apply_sql(cur, sql_text: str) -> None:
    # naive splitter on ';' to avoid multi-statement issues; fine for our simple SQL
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    for stmt in statements:
        cur.execute(stmt + ";")


def main() -> int:
    migrations_dir = Path(__file__).parent / "migrations"
    if not migrations_dir.exists():
        print(f"Missing migrations directory: {migrations_dir}", file=sys.stderr)
        return 1

    conninfo = conninfo_from_env()
    with psycopg2.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE;")
            ensure_schema_migrations(cur)
            applied = 0
            for path in sorted(migrations_dir.glob("*.sql")):
                name = path.name
                if migration_applied(cur, name):
                    continue
                sql_text = path.read_text(encoding="utf-8")
                apply_sql(cur, sql_text)
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s);",
                    (name,),
                )
                applied += 1
            conn.commit()
            print(f"Applied {applied} migration(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
