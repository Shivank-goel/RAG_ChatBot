# app/test_rag_answer.py
from app.ingest_api import build_api_docs
from app.index import build_index
from app.utils import identity_chunk
from app.rag_chain import answer

if __name__ == "__main__":
    # 1) Fetch a small set (fast)
    docs = build_api_docs(
        symbols_stocks=["AAPL"],
        symbols_crypto=["BTC"],
        market="USD",
        days=7,
        include_overview=True,
        include_earnings=False,
        include_news=False,
    )

    # 2) Build/rebuild index
    added = build_index(docs, chunker=identity_chunk, collection_name="docs")
    print(f"Indexed {added} chunks.")

    # 3) Ask a couple questions
    queries = [
        "What was BTC/USD close yesterday?",
        "Summarize Apple Inc company overview briefly.",
    ]
    for q in queries:
        print("\nQ:", q)
        ans, chunks = answer(q, k=4, max_new_tokens=160)
        print("A:", ans)
        print("Sources:")
        for ch in chunks:
            m = ch["meta"]
            print(f" - {m['doc_id']} (chunk {m['chunk']})")
