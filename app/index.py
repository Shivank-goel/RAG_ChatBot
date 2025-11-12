# app/index.py
from typing import List, Dict, Callable
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_DIR, EMBEDDING_MODEL

def get_client():
    return PersistentClient(path=CHROMA_DIR)

def get_collection(name: str = "docs"):
    client = get_client()
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)

class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def build_index(docs: List[Dict], chunker: Callable[[str], List[str]], collection_name: str = "docs"):
    """
    docs: list of {id, text, source, type}
    chunker: function(text)->[chunks]; for API passages we can use identity (single chunk)
    """
    col = get_collection(collection_name)
    embedder = Embedder()

    ids, texts, metadatas = [], [], []
    for d in docs:
        # API passages are already small; chunker can return [text] directly
        chunks = chunker(d["text"])
        for i, ch in enumerate(chunks):
            ids.append(f"{d['id']}#chunk{i}")
            texts.append(ch)
            metadatas.append({
                "source": d.get("source", "alpha_vantage"),
                "chunk": i,
                "doc_id": d["id"],
                "type": d.get("type", "api/alpha_vantage"),
            })

    if not texts:
        return 0

    embeddings = embedder.encode(texts)
    try:
        col.delete(where={})
    except Exception:
        pass
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(ids)
