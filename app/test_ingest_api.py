# app/test_ingest_api.py
from app.ingest_api import build_api_docs

if __name__ == "__main__":
    docs = build_api_docs(
        symbols_stocks=["AAPL"],
        symbols_crypto=["BTC"],  # BTC vs USD
        market="USD",
        days=5,
        include_overview=True,
        include_earnings=False,
        include_news=False,
    )
    print(f"Fetched {len(docs)} passages.")
    for d in docs[:6]:
        print(d["id"], "->", d["text"][:140].replace("\n", " "), "...")
