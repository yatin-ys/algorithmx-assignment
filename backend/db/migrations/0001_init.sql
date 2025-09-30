-- documents table for tracking PDF ingestion lifecycle
CREATE TABLE IF NOT EXISTS documents (
	id BIGSERIAL PRIMARY KEY,
	title TEXT NOT NULL,
	file_hash CHAR(64) NOT NULL,
	status VARCHAR(32) NOT NULL DEFAULT 'queued',
	page_count INTEGER,
	created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);