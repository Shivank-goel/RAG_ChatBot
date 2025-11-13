# app/rag_chain.py
from typing import List, Tuple
import re
from app.retriever import retrieve
from app.generator import LocalGenerator

gen = LocalGenerator()

SYSTEM_PROMPT = (
    "You are a concise, factual finance & crypto assistant.\n"
    "Answer FIRST (1-3 short sentences). On the NEXT LINE, print SOURCES in this exact format:\n"
    "SOURCES: [1],[2]\n"
    "Do NOT output only citation numbers. If the information is not present in the CONTEXT, reply exactly: \"I don't know.\"\n"
    "When asked for numeric values (price/volume), extract and output the numeric string exactly as present in the context.\n"
)

CHUNK_MAX_CHARS = 800

_company_keywords = {"company", "overview", "summary", "profile", "about", "headquarters", "sector", "industry"}
_numeric_keywords = {"close", "open", "volume", "high", "low", "price"}

def _shorten_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " ..."

def _format_context(chunks: List[dict]) -> str:
    lines = []
    for i, ch in enumerate(chunks, 1):
        meta = ch["meta"]
        text = _shorten_text(ch["text"])
        lines.append(f"[{i}] {meta.get('doc_id')} | {meta.get('type')}:\n{text}")
    return "\n\n".join(lines)

def _is_company_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _company_keywords)

def _contains_numeric_request(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _numeric_keywords)

def _prioritize_chunks(chunks: List[dict], query: str) -> List[dict]:
    overviews = [ch for ch in chunks if ("overview" in ch.get("meta", {}).get("doc_id", "").lower()) or ("overview" in ch.get("meta", {}).get("type", "").lower())]
    others = [ch for ch in chunks if ch not in overviews]
    overviews = sorted(overviews, key=lambda ch: ch.get("score", float("inf")))
    others = sorted(others, key=lambda ch: ch.get("score", float("inf")))
    # Put overviews first so summaries come from them
    return overviews + others

def _build_prompt(query: str, chunks: List[dict]) -> str:
    # Simplified prompt for better model performance
    if _contains_numeric_request(query):
        # For numeric queries, use a very simple template
        if chunks:
            best_chunk = chunks[0]["text"]
            prompt = f"Extract the price from this data: {best_chunk}\nQuestion: {query}\nAnswer:"
            return prompt
    
    # For company/overview queries
    if _is_company_query(query):
        if chunks:
            best_chunk = chunks[0]["text"]
            prompt = f"Summarize this company information: {best_chunk[:400]}\nQuestion: {query}\nAnswer:"
            return prompt
    
    # For general queries, use simplified context
    context_simple = ""
    for i, chunk in enumerate(chunks[:2], 1):  # Only use top 2 chunks
        context_simple += f"{chunk['text'][:200]}\n"
    
    prompt = f"Context: {context_simple}\n\nQuestion: {query}\nAnswer:"
    return prompt

# regex to detect bracket-only answers like "[3]" or "[1],[2]"
_BRACKET_ONLY_RE = re.compile(r'^\s*(\[\s*\d+\s*\]\s*(,\s*\[\s*\d+\s*\]\s*)*)\s*$')

# numeric extraction regex (captures typical decimal numbers)
_NUMERIC_RE = re.compile(r'[-+]?\d{1,3}(?:[,\d]{0,})?(?:\.\d+)?')

def _extract_numeric_from_text(text: str) -> str:
    # find all numeric tokens, return the first plausible one (prefer decimals)
    matches = _NUMERIC_RE.findall(text)
    if not matches:
        return ""
    # return first match (strip commas)
    return matches[0].replace(",", "")

def _first_sentence(text: str) -> str:
    # naive first-sentence extraction
    text = text.strip().replace("\n", " ")
    m = re.search(r'(.+?[\.!?])\s', text)
    if m:
        return m.group(1)
    # fallback: return up to 200 chars
    return text[:200].rsplit(" ", 1)[0] + ("..." if len(text) > 200 else "")

def answer(query: str, k: int = 6, max_new_tokens: int = 100) -> Tuple[str, List[dict]]:
    candidates = retrieve(query, k=k)
    prioritized = _prioritize_chunks(candidates, query)
    chunks_for_prompt = prioritized[:4]
    
    # Use simplified prompt
    prompt = _build_prompt(query, chunks_for_prompt)
    
    # Generate answer with shorter max tokens for cleaner output
    out = gen(prompt, max_new_tokens=max_new_tokens)
    
    # Clean up the output
    answer_text = out.strip()
    
    # Handle poor/empty outputs with fallback logic
    if not answer_text or len(answer_text) <= 3 or answer_text.isdigit():
        # Use fallback extraction
        if _contains_numeric_request(query) and chunks_for_prompt:
            # Extract price/numeric data directly from best chunk
            text = chunks_for_prompt[0]["text"]
            
            # Look for price patterns in the text
            if "close" in text.lower():
                import re
                close_match = re.search(r'close\s+([0-9,]+\.?[0-9]*)', text, re.IGNORECASE)
                if close_match:
                    return f"BTC close price: {close_match.group(1)}", chunks_for_prompt
            
            # General numeric extraction
            num = _extract_numeric_from_text(text)
            if num:
                return f"Price: {num}", chunks_for_prompt
        
        # Fallback for other queries
        if chunks_for_prompt:
            return _first_sentence(chunks_for_prompt[0]["text"]), chunks_for_prompt
        
        return "I don't know.", chunks_for_prompt
    
    return answer_text, chunks_for_prompt
