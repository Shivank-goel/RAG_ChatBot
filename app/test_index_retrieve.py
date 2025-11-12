# app/test_index_retrieve.py
from app.ingest_api import build_api_docs
from app.index import build_index
from app.retriever import retrieve
from app.utils import identity_chunk

if __name__ == "__main__":
    # 1) Fetch small set to keep it fast
    docs = build_api_docs(
        symbols_stocks=["AAPL"],
        symbols_crypto=["BTC"],
        market="USD",
        days=7,  # recent week
        include_overview=True,
        include_earnings=False,
        include_news=False,
    )
    print(f"Fetched {len(docs)} API passages.")

    # 2) Build vector index
    added = build_index(docs, chunker=identity_chunk, collection_name="docs")
    print(f"Indexed {added} chunks into Chroma.")

    # 3) Try a couple of queries
    queries = [
        "What was BTC/USD close yesterday?",
        "Summarize Apple Inc company overview.",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        hits = retrieve(q, k=4, collection_name="docs")
        for i, h in enumerate(hits, 1):
            meta = h["meta"]
            print(f"  {i}. {meta['doc_id']} (score={h['score']:.4f})")
            text_preview = h['text'][:140].replace('\n', ' ')
            print(f"     {text_preview} ...")

