from fastapi import FastAPI
from .routes.documents import router as documents_router

app = FastAPI(title="AlgorithmX RAG Backend", version="0.1.0")

app.include_router(documents_router)
