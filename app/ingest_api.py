# app/ingest_api.py
from typing import List, Dict
import time
import re
import requests
from datetime import datetime
from app.config import ALPHA_VANTAGE_KEY, AV_BASE_URL, AV_RATE_LIMIT_SLEEP

class AVClient:
    """Thin Alpha Vantage client with conservative backoff for free tier."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Missing Alpha Vantage API key. Set ALPHA_VANTAGE_KEY in .env.")
        self.api_key = api_key
        self.sess = requests.Session()

    def _get(self, **params) -> Dict:
        params["apikey"] = self.api_key
        # up to 5 attempts with sleep if throttled
        for _ in range(5):
            r = self.sess.get(AV_BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            # Throttle message returns a "Note" field
            if isinstance(data, dict) and any(k.lower() == "note" for k in data.keys()):
                time.sleep(AV_RATE_LIMIT_SLEEP)
                continue
            if "Error Message" in data:
                raise RuntimeError(data["Error Message"])
            return data
        raise RuntimeError("Alpha Vantage rate limit: retries exceeded")

    # ------------ Stock endpoints ------------
    def daily_adjusted(self, symbol: str, outputsize: str = "compact") -> Dict:
        return self._get(function="TIME_SERIES_DAILY_ADJUSTED", symbol=symbol, outputsize=outputsize)

    def overview(self, symbol: str) -> Dict:
        return self._get(function="OVERVIEW", symbol=symbol)

    def earnings(self, symbol: str) -> Dict:
        return self._get(function="EARNINGS", symbol=symbol)

    # ------------ Crypto endpoints ------------
    def crypto_daily(self, symbol: str, market: str = "USD") -> Dict:
        return self._get(function="DIGITAL_CURRENCY_DAILY", symbol=symbol, market=market)

    # (Optional) News
    def news(self, symbols_csv: str, limit: int = 50) -> Dict:
        return self._get(function="NEWS_SENTIMENT", tickers=symbols_csv, limit=limit)


# ----- Serialization helpers: JSON -> natural language passages -----

def _kv_to_text(title: str, kv: Dict) -> str:
    lines = [f"{title}:"]
    for k, v in kv.items():
        if isinstance(v, (dict, list)):
            continue
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)

def _ts_stock_passages(symbol: str, ts_json: Dict, max_days: int = 365) -> List[str]:
    key = next((k for k in ts_json.keys() if "Time Series" in k), None)
    if not key:
        return []
    series = ts_json[key]
    items = sorted(series.items(), key=lambda x: x[0], reverse=True)[:max_days]
    out = []
    for date_str, row in items:
        o = row.get("1. open"); h = row.get("2. high"); l = row.get("3. low")
        c = row.get("4. close") or row.get("5. adjusted close")
        v = row.get("6. volume") or row.get("5. volume") or row.get("volume")
        out.append(f"{symbol} daily bar on {date_str}: open {o}, high {h}, low {l}, close {c}, volume {v}.")
    return out

def _pick_crypto_price(row: dict, field_base: str, market: str):
    """
    Returns the value for a crypto field like 'open'|'high'|'low'|'close'.
    Preference order:
      1) exact market match (e.g., '(USD)')
      2) any currency match (fallback)
    Matches both '1a. open (USD)' and '1b. open (USD)' etc.
    """
    # 1) exact market match
    pat_exact = re.compile(rf"\b\d+[ab]\.\s*{re.escape(field_base)}\s*\(\s*{re.escape(market)}\s*\)\b", re.I)
    for k, v in row.items():
        if pat_exact.search(k):
            return v

    # 2) any currency match for that field (fallback)
    pat_any = re.compile(rf"\b\d+[ab]\.\s*{re.escape(field_base)}\s*\(\s*[A-Z]{{3,}}\s*\)\b", re.I)
    for k, v in row.items():
        if pat_any.search(k):
            return v

    # 3) last-ditch: any key containing the field_base (e.g., 'close') if above failed
    for k, v in row.items():
        if field_base.lower() in k.lower():
            return v

    return None


def _ts_crypto_passages(symbol: str, market: str, json_obj: Dict, max_days: int = 365) -> List[str]:
    # Find the time series key case-insensitively
    ts_key = next((k for k in json_obj.keys() if "time series" in k.lower()), None)
    if not ts_key:
        return []
    series = json_obj[ts_key]

    items = sorted(series.items(), key=lambda x: x[0], reverse=True)[:max_days]
    out = []
    for date_str, row in items:
        # 1) Try plain numeric keys (your output shows this format)
        o = row.get("1. open")
        h = row.get("2. high")
        l = row.get("3. low")
        c = row.get("4. close")
        v = row.get("5. volume") or row.get("6. market cap (usd)")

        # 2) Fallback to (market) variants if any are missing
        if o is None:
            o = row.get(f"1a. open ({market})") or row.get(f"1b. open ({market})")
        if h is None:
            h = row.get(f"2a. high ({market})") or row.get(f"2b. high ({market})")
        if l is None:
            l = row.get(f"3a. low ({market})") or row.get(f"3b. low ({market})")
        if c is None:
            c = row.get(f"4a. close ({market})") or row.get(f"4b. close ({market})")

        # 3) Last-ditch: any key that contains the field name
        if c is None:
            c = next((val for k, val in row.items() if "close" in k.lower()), None)

        out.append(f"{symbol}/{market} on {date_str}: open {o}, high {h}, low {l}, close {c}, volume {v}.")
    return out




def build_api_docs(
    symbols_stocks: List[str],
    symbols_crypto: List[str],
    market: str = "USD",
    days: int = 365,
    include_overview: bool = True,
    include_earnings: bool = True,
    include_news: bool = False,
) -> List[Dict]:
    """
    Returns a list of doc dicts: {id, text, source, type}
    Suitable for passing into your index builder.
    """
    client = AVClient(ALPHA_VANTAGE_KEY)
    docs: List[Dict] = []

    # Stocks
    for sym in symbols_stocks:
        if include_overview:
            ov = client.overview(sym)
            docs.append({
                "id": f"av/{sym}/overview",
                "text": _kv_to_text(f"{sym} Company Overview", ov),
                "source": "alpha_vantage:overview",
                "type": "api/alpha_vantage"
            })
        ts = client.daily_adjusted(sym, outputsize="full" if days > 100 else "compact")
        for i, passage in enumerate(_ts_stock_passages(sym, ts, max_days=days)):
            docs.append({
                "id": f"av/{sym}/daily#{i}",
                "text": passage,
                "source": "alpha_vantage:time_series_daily_adjusted",
                "type": "api/alpha_vantage"
            })
        if include_earnings:
            er = client.earnings(sym)
            docs.append({
                "id": f"av/{sym}/earnings",
                "text": _kv_to_text(f"{sym} Earnings (AV)", er),
                "source": "alpha_vantage:earnings",
                "type": "api/alpha_vantage"
            })

    # Crypto
    for sym in symbols_crypto:
        crypto_json = client.crypto_daily(sym, market=market)
        for i, passage in enumerate(_ts_crypto_passages(sym, market, crypto_json, max_days=days)):
            docs.append({
                "id": f"av/{sym}-{market}/digital_daily#{i}",
                "text": passage,
                "source": "alpha_vantage:digital_currency_daily",
                "type": "api/alpha_vantage"
            })

    # Optional news for both
    if include_news and (symbols_stocks or symbols_crypto):
        universe = ",".join(symbols_stocks) if symbols_stocks else ""
        # Alpha Vantage NEWS_SENTIMENT is mostly equity ticker oriented
        if universe:
            news_json = client.news(universe, limit=40)
            feed = news_json.get("feed", [])[:40]
            for j, item in enumerate(feed):
                dt = item.get("time_published")
                try:
                    ts = datetime.strptime(dt, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M:%S") if dt else ""
                except Exception:
                    ts = dt or ""
                title = item.get("title"); summary = item.get("summary")
                sent = item.get("overall_sentiment_label"); src = item.get("source"); url = item.get("url")
                docs.append({
                    "id": f"av/news#{j}",
                    "text": f"News on {ts}: {title} — {summary} [sentiment: {sent}] Source: {src}. URL: {url}",
                    "source": "alpha_vantage:news_sentiment",
                    "type": "api/alpha_vantage"
                })

    return docs
