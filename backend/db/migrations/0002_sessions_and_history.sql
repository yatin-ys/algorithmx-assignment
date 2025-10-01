-- sessions table: track conversation sessions
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    settings_json JSONB
);

-- messages table: store user and assistant messages
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(32) NOT NULL,  -- 'user' or 'assistant'
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- runs table: track each chat execution with parameters
CREATE TABLE IF NOT EXISTS runs (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model VARCHAR(128) NOT NULL,
    top_k INTEGER NOT NULL,
    only_if_sources BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- retrievals table: track which documents were retrieved for each run
CREATE TABLE IF NOT EXISTS retrievals (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,  -- position in results (1-based)
    score FLOAT NOT NULL,
    doc_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    chunk_id INTEGER NOT NULL
);

-- metrics table: performance metrics for each run
CREATE TABLE IF NOT EXISTS metrics (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    latency_ms_total INTEGER NOT NULL,
    latency_ms_embed INTEGER NOT NULL,
    latency_ms_qdrant INTEGER NOT NULL,
    latency_ms_llm INTEGER NOT NULL,
    sources_found_bool BOOLEAN NOT NULL
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_retrievals_run_id ON retrievals(run_id);
CREATE INDEX IF NOT EXISTS idx_retrievals_doc_id ON retrievals(doc_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id);
