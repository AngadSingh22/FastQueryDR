from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - handled at runtime
    faiss = None


def build_flat_ip_index(embeddings: np.ndarray):
    if faiss is None:
        raise ImportError("faiss is required for retrieval evaluation. Install the project dependencies first.")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index
