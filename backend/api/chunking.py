import os
from typing import List


def chunk_text(
    text: str, chunk_size: int | None = None, overlap: int | None = None
) -> List[str]:
    # Defaults via env or sane fallbacks
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
    if overlap is None:
        overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    chunks: List[str] = []
    i = 0
    n = len(text)
    if n == 0:
        return chunks
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = j - overlap if j - overlap > i else j
    return chunks
