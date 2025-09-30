from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .. import config
import numpy as np


_EMBEDDER: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _EMBEDDER


def embedding_dimension() -> int:
    m = get_model()
    method = getattr(m, "get_sentence_embedding_dimension", None)
    if callable(method):
        d = method()
        if isinstance(d, int):
            return d
    # Fallback: infer from a tiny forward pass
    arr = m.encode([""], convert_to_numpy=True)
    try:
        return int(arr.shape[1])  # type: ignore[attr-defined]
    except Exception:
        return int(len(arr[0]))


def embedding_model_name() -> str:
    m = get_model()
    method = getattr(m, "get_sentence_embedding_model_name", None)
    if callable(method):
        v = method()
        if isinstance(v, str) and v:
            return v
    return str(config.EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    arr = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    if isinstance(arr, np.ndarray):
        arr = arr.astype("float32")
    return [row.tolist() for row in arr]
