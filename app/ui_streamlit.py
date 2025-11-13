# app/ui_streamlit.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import re
from app.ingest_api import build_api_docs
from app.index import build_index, get_collection
from app.rag_chain import answer
from app.utils import identity_chunk
from app.config import CHROMA_DIR

st.set_page_config(page_title="Finance RAG Chatbot", layout="wide")

# Temporary safe CSS (you can remove once theme is stable)
st.markdown(
    """
    <style>
    .stApp { background: white; color: black; }
    .main .block-container { padding-top: 1rem; max-width: 1100px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Finance & Crypto RAG Chatbot")

# --- Sidebar: Data source + indexing ---
with st.sidebar:
    st.header("🔑 Data Source (Alpha Vantage)")
    av_key = st.text_input("API Key", type="password")
    symbols_stocks = st.text_input("Stock Symbols (comma-separated)", value="AAPL,MSFT")
    symbols_crypto = st.text_input("Crypto Symbols (comma-separated)", value="BTC,ETH")
    market = st.text_input("Market", value="USD")
    days = st.number_input("Max Trading Days", min_value=30, max_value=2000, value=365)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬇️ Fetch & Index"):
            if not av_key:
                st.error("Enter your Alpha Vantage API key.")
            else:
                syms_stocks = [s.strip().upper() for s in symbols_stocks.split(",") if s.strip()]
                syms_crypto = [s.strip().upper() for s in symbols_crypto.split(",") if s.strip()]
                with st.spinner("Fetching data from Alpha Vantage..."):
                    docs = build_api_docs(
                        symbols_stocks=syms_stocks,
                        symbols_crypto=syms_crypto,
                        market=market,
                        days=days,
                        include_overview=True,
                        include_earnings=False,
                        include_news=False,
                    )
                    count = build_index(docs, chunker=identity_chunk, collection_name="docs")
                st.success(f"Indexed {count} chunks for {', '.join(syms_stocks + syms_crypto)}")
                # compute indexed symbols and store in session
                indexed_symbols = []
                for d in docs:
                    doc_id = d.get("id", "")
                    parts = doc_id.split("/")
                    if len(parts) >= 2:
                        candidate = parts[1].split("-")[0]
                        if candidate and candidate not in indexed_symbols:
                            indexed_symbols.append(candidate)
                st.session_state["indexed_symbols"] = indexed_symbols

    with col2:
        if st.button("🧹 Clear local index"):
            try:
                col = get_collection("docs")
                col.delete(where={})
                st.success("Cleared local Chroma collection 'docs'.")
            except Exception as e:
                st.error(f"Clear failed: {e}")

    st.markdown("---")
    st.write("Indexed symbols:")
    st.write(st.session_state.get("indexed_symbols", []))

st.markdown("---")

# --- Query/filter controls ---
indexed_symbols = st.session_state.get("indexed_symbols", [])
filter_symbol = None
if indexed_symbols:
    filter_symbol = st.selectbox("Filter queries by symbol (optional)", ["(none)"] + indexed_symbols)

st.subheader("💬 Ask a Question")
user_query = st.text_input("Your question (e.g., 'What was BTC close yesterday?')")

ask_col1, ask_col2 = st.columns([1, 5])
with ask_col1:
    ask_btn = st.button("Ask")
with ask_col2:
    st.write("")  # spacer to align

# helper to highlight numeric tokens and dates
def highlight_numbers_and_dates(s: str) -> str:
    s = re.sub(r"(\d{4}-\d{2}-\d{2})", r"**\1**", s)  # dates
    s = re.sub(r"(\d{1,3}(?:[,\d]{0,})?(?:\.\d+)?)", r"**\1**", s)  # numbers
    return s

if ask_btn and user_query:
    q = user_query
    if filter_symbol and filter_symbol != "(none)":
        # bias the query with the symbol hint
        q = f"{filter_symbol} {user_query}"

    with st.spinner("Thinking..."):
        ans, chunks = answer(q, k=6)

    st.markdown("### 🧠 Answer")
    # Try to render markdown with highlighted numbers
    try:
        st.markdown(highlight_numbers_and_dates(ans))
    except Exception:
        st.write(ans)

    st.markdown("### 📄 Sources (top chunks)")
    # Show each chunk in its own expander with preview + full text
    for idx, ch in enumerate(chunks, start=1):
        m = ch["meta"]
        doc_id = m.get("doc_id", "")
        st.markdown(f"**{idx}. {doc_id}** — {m.get('type','')}, score={ch['score']:.4f}")
        preview = ch["text"][:400].replace("\n", " ")
        preview = re.sub(r"(\d{4}-\d{2}-\d{2})", r"**\1**", preview)
        preview = re.sub(r"(\d{1,3}(?:[,\d]{0,})?(?:\.\d+)?)", r"**\1**", preview)
        with st.expander("Preview / Show full text"):
            st.write(preview)
            st.write("---")
            st.write(ch["text"])
            # simple copy button (placeholder)
            if st.button(f"Copy source {idx} (placeholder)", key=f"copy_{idx}"):
                st.write("Copied (placeholder).")

st.markdown("---")
st.caption("Local dev: data is stored in the `storage/` folder (ChromaDB).")
