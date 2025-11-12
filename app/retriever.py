# app/retriever.py
from typing import List, Dict
from app.index import get_collection, Embedder
from app.config import TOP_K

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

def retrieve(query: str, k: int = TOP_K, collection_name: str = "docs") -> List[Dict]:
    col = get_collection(collection_name)
    qvec = _get_embedder().encode([query])[0]
    out = col.query(query_embeddings=[qvec], n_results=k,
                    include=["documents", "metadatas", "distances"])
    results = []
    for doc, meta, dist in zip(out["documents"][0], out["metadatas"][0], out["distances"][0]):
        results.append({"text": doc, "meta": meta, "score": float(dist)})
    return results
