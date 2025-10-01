from fastapi import FastAPI
from .routes.documents import router as documents_router
from .routes.retrieve import router as retrieve_router
from .routes.chat import router as chat_router
from .routes.sessions import router as sessions_router

app = FastAPI(title="AlgorithmX RAG Backend", version="0.1.0")

app.include_router(documents_router)
app.include_router(retrieve_router)
app.include_router(chat_router)
app.include_router(sessions_router)
