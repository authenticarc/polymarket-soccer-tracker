#!/usr/bin/env python3
"""
Polymarket Soccer Tracker (UCL + EPL)
- Uses Gamma API to discover events/markets by tag_id
- Uses CLOB API to fetch orderbooks (best bid/ask, spread, etc.)
- Writes daily CSV snapshots under ./data/

Key fixes:
- Robust parsing of Gamma "clobTokenIds" which may appear as:
    * list
    * JSON-in-string (e.g. '["5860...","1234..."]')
    * malformed/truncated string (e.g. '["5860...",')
  We extract token IDs via JSON parse first, then regex fallback.

Env vars:
- GAMMA_BASE (default https://gamma-api.polymarket.com)
- CLOB_BASE  (default https://clob.polymarket.com)
- LEAGUES    (default "UCL,EPL")
- TAG_UCL    (default "13")  # you can pin
- TAG_EPL    (default "2")   # you can pin
- LIMIT_EVENTS (default 200)
- TIMEOUT_SEC (default 20)
- OUT_DIR    (default "data")
"""

import os
import json
import time
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd


# -----------------------------
# Config
# -----------------------------
GAMMA_BASE = os.getenv("GAMMA_BASE", "https://gamma-api.polymarket.com").rstrip("/")
CLOB_BASE = os.getenv("CLOB_BASE", "https://clob.polymarket.com").rstrip("/")

OUT_DIR = os.getenv("OUT_DIR", "data")
LIMIT_EVENTS = int(os.getenv("LIMIT_EVENTS", "200"))
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", "20"))

UA = "polymarket-soccer-tracker/1.1 (+github-actions)"

LEAGUE_DEFAULTS = {
    "UCL": {"tag_env": "TAG_UCL", "tag_default": "100977"},
    "EPL": {"tag_env": "TAG_EPL", "tag_default": "82"},
}

# Token id patterns:
# - huge decimal integer strings are common
# - sometimes 0x... asset ids exist in some contexts
TOKEN_EXTRACT_RE = re.compile(r"(0x[0-9a-fA-F]+|\d{10,})")


# -----------------------------
# HTTP helpers
# -----------------------------
def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    r = requests.get(url, params=params, timeout=TIMEOUT_SEC, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.json()


def _post(url: str, payload: Any) -> Any:
    r = requests.post(
        url,
        json=payload,
        timeout=TIMEOUT_SEC,
        headers={"User-Agent": UA, "Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json()


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


# -----------------------------
# Parsing: clobTokenIds (critical fix)
# -----------------------------
def parse_clob_token_ids(x: Any) -> List[str]:
    """
    Gamma clobTokenIds can be:
      - list: ["5860...", "1234..."]
      - JSON string: '["5860...","1234..."]'
      - malformed/truncated string: '["5860...",'
      - comma-separated: '5860...,1234...'
    Return a cleaned list[str] containing only plausible token ids.
    """
    if x is None:
        return []

    # Case 1: already a list
    if isinstance(x, list):
        vals = [str(i).strip() for i in x if i is not None]
        return _clean_token_list(vals)

    # Case 2: string
    if isinstance(x, str):
        s = x.strip()

        # Try JSON first (best case)
        try:
            y = json.loads(s)
            if isinstance(y, list):
                vals = [str(i).strip() for i in y if i is not None]
                return _clean_token_list(vals)
        except Exception:
            pass

        # Fallback: regex extraction (works even if truncated)
        extracted = [m.group(1) for m in TOKEN_EXTRACT_RE.finditer(s)]
        if extracted:
            return _dedup_preserve_order(_clean_token_list(extracted))

        # Fallback: comma-separated
        if "," in s:
            vals = [t.strip() for t in s.split(",") if t.strip()]
            return _dedup_preserve_order(_clean_token_list(vals))

        return _clean_token_list([s])

    # Case 3: other types
    return _clean_token_list([str(x).strip()])


def _clean_token_list(vals: List[str]) -> List[str]:
    out: List[str] = []
    for v in vals:
        if not v:
            continue
        v2 = v.strip().strip('"').strip("'")
        if not v2:
            continue
        # Reject junk artifacts like "[" or "]"
        if v2 in ("[", "]"):
            continue
        # Accept if it matches our extraction regex fully OR looks like 0x...
        if v2.startswith("0x"):
            out.append(v2)
        elif v2.isdigit() and len(v2) >= 10:
            out.append(v2)
        else:
            # sometimes we get fragments; try extracting tokens from the fragment
            frag = [m.group(1) for m in TOKEN_EXTRACT_RE.finditer(v2)]
            out.extend(frag)
    return _dedup_preserve_order(out)


def _dedup_preserve_order(vals: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# -----------------------------
# Gamma fetching
# -----------------------------
def fetch_events_by_tag(tag_id: str) -> List[Dict[str, Any]]:
    """
    Fetch active, open events for given tag_id.
    """
    events: List[Dict[str, Any]] = []
    offset = 0
    page = 100

    while len(events) < LIMIT_EVENTS:
        batch = _get(
            f"{GAMMA_BASE}/events",
            params={
                "tag_id": str(tag_id),
                "active": "true",
                "closed": "false",
                "archived": "false",
                "limit": str(min(page, LIMIT_EVENTS - len(events))),
                "offset": str(offset),
                # keep related tags on; events may have nested tags
                "related_tags": "true",
            },
        )
        if not isinstance(batch, list) or len(batch) == 0:
            break
        events.extend(batch)
        offset += len(batch)
        if len(batch) < page:
            break

    return events


def fetch_market_full(market_id: Any) -> Optional[Dict[str, Any]]:
    """
    Fetch full market object by id.
    Returns None if fetch fails.
    """
    if market_id is None:
        return None
    try:
        return _get(f"{GAMMA_BASE}/markets/{market_id}")
    except Exception:
        return None


def enable_orderbook(market: Dict[str, Any]) -> bool:
    return bool(market.get("enableOrderBook")) is True


# -----------------------------
# CLOB fetching
# -----------------------------
def fetch_books_batch(token_ids: List[str]) -> List[Dict[str, Any]]:
    """
    POST /books
    Body: [{"token_id":"..."}]
    """
    payload = [{"token_id": str(t)} for t in token_ids]
    res = _post(f"{CLOB_BASE}/books", payload)
    if isinstance(res, list):
        return res
    # Some deployments might return dict; normalize to list
    if isinstance(res, dict):
        # best-effort: values as list
        return list(res.values())
    return []


def best_from_book(book: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    bids = book.get("bids") or []
    asks = book.get("asks") or []

    best_bid = float(bids[0]["price"]) if bids else None
    best_bid_size = float(bids[0]["size"]) if bids else None
    best_ask = float(asks[0]["price"]) if asks else None
    best_ask_size = float(asks[0]["size"]) if asks else None

    mid = None
    spr = None
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
        spr = best_ask - best_bid

    return best_bid, best_bid_size, best_ask, best_ask_size, mid, spr


# -----------------------------
# Snapshot builder
# -----------------------------
def snapshot_league(league: str, tag_id: str) -> pd.DataFrame:
    ts = utc_iso_now()
    events = fetch_events_by_tag(tag_id)

    rows: List[Dict[str, Any]] = []

    # We'll gather (market_id -> token_ids, meta) then batch books across all tokens
    market_records: List[Dict[str, Any]] = []

    for ev in events:
        ev_id = ev.get("id")
        ev_slug = ev.get("slug")
        ev_title = ev.get("title") or ev.get("name") or ev.get("question")
        ev_start = ev.get("startDate") or ev.get("start_date") or ev.get("eventStartTime")

        markets = ev.get("markets") or []
        if not isinstance(markets, list):
            continue

        for m in markets:
            if not isinstance(m, dict):
                continue

            m_id = m.get("id")
            m_slug = m.get("slug")
            m_question = m.get("question") or m.get("title")
            m_liq = m.get("liquidity")
            m_vol = m.get("volume") or m.get("volumeNum") or m.get("volume_num")

            # events payload can be "light"; always try to parse clobTokenIds robustly
            token_ids = parse_clob_token_ids(m.get("clobTokenIds"))
            ob = enable_orderbook(m)

            # If missing or seems malformed, fetch full market and retry
            if (not token_ids) or (ob is False):
                full = fetch_market_full(m_id)
                if full:
                    token_ids = parse_clob_token_ids(full.get("clobTokenIds"))
                    ob = enable_orderbook(full)
                    # replace market fields with full if present
                    m_slug = full.get("slug") or m_slug
                    m_question = full.get("question") or full.get("title") or m_question
                    m_liq = full.get("liquidity", m_liq)
                    m_vol = full.get("volume") or full.get("volumeNum") or full.get("volume_num") or m_vol
                    m = full

            if not ob:
                continue
            if not token_ids:
                # record diagnostic row (no token ids)
                rows.append(
                    {
                        "ts_utc": ts,
                        "league": league,
                        "event_id": ev_id,
                        "event_slug": ev_slug,
                        "event_title": ev_title,
                        "event_start": ev_start,
                        "market_id": m_id,
                        "market_slug": m_slug,
                        "market_question": m_question,
                        "condition_id": m.get("conditionId"),
                        "token_id": None,
                        "outcome": None,
                        "best_bid": None,
                        "best_bid_size": None,
                        "best_ask": None,
                        "best_ask_size": None,
                        "mid": None,
                        "spread": None,
                        "book_hash": None,
                        "tick_size": None,
                        "min_order_size": None,
                        "last_trade_price": None,
                        "liquidity": m_liq,
                        "volume": m_vol,
                        "error": "missing_clobTokenIds",
                    }
                )
                continue

            # Store market meta for later join
            outcomes = m.get("outcomes")
            if isinstance(outcomes, str):
                # sometimes outcomes is JSON string
                try:
                    outcomes = json.loads(outcomes)
                except Exception:
                    outcomes = None

            market_records.append(
                {
                    "event_id": ev_id,
                    "event_slug": ev_slug,
                    "event_title": ev_title,
                    "event_start": ev_start,
                    "market_id": m_id,
                    "market_slug": m_slug,
                    "market_question": m_question,
                    "condition_id": m.get("conditionId"),
                    "token_ids": token_ids,
                    "outcomes": outcomes if isinstance(outcomes, list) else None,
                    "liquidity": m_liq,
                    "volume": m_vol,
                }
            )

    # Batch fetch all books for this league snapshot
    all_tokens: List[str] = []
    for rec in market_records:
        all_tokens.extend(rec["token_ids"])
    all_tokens = _dedup_preserve_order(all_tokens)

    # If nothing to fetch, return what we have
    if not all_tokens:
        return pd.DataFrame(rows)

    # Fetch in chunks (be nice to API)
    token_to_book: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    for part in chunk(all_tokens, 200):
        try:
            books = fetch_books_batch(part)
        except Exception as e:
            errors.append(f"books_batch_error: {e}")
            time.sleep(0.2)
            continue

        # Index by asset_id/token_id in response
        for b in books:
            aid = b.get("asset_id") or b.get("token_id")
            if aid is None:
                continue
            token_to_book[str(aid)] = b

        # Also index by request order as a fallback if response lacks asset_id (rare)
        if isinstance(books, list) and len(books) == len(part):
            for req, b in zip(part, books):
                if req not in token_to_book:
                    token_to_book[req] = b

        time.sleep(0.2)

    # Build rows per token outcome
    for rec in market_records:
        token_ids = rec["token_ids"]
        outcomes = rec["outcomes"] or []

        for idx, t in enumerate(token_ids):
            book = token_to_book.get(str(t), {})

            best_bid, best_bid_size, best_ask, best_ask_size, mid, spr = best_from_book(book) if book else (None, None, None, None, None, None)

            outcome = None
            if idx < len(outcomes):
                outcome = outcomes[idx]

            rows.append(
                {
                    "ts_utc": ts,
                    "league": league,
                    "event_id": rec["event_id"],
                    "event_slug": rec["event_slug"],
                    "event_title": rec["event_title"],
                    "event_start": rec["event_start"],
                    "market_id": rec["market_id"],
                    "market_slug": rec["market_slug"],
                    "market_question": rec["market_question"],
                    "condition_id": rec["condition_id"],
                    "token_id": str(t),
                    "outcome": outcome,
                    "best_bid": best_bid,
                    "best_bid_size": best_bid_size,
                    "best_ask": best_ask,
                    "best_ask_size": best_ask_size,
                    "mid": mid,
                    "spread": spr,
                    "book_hash": book.get("hash") if book else None,
                    "tick_size": book.get("tick_size") if book else None,
                    "min_order_size": book.get("min_order_size") if book else None,
                    "last_trade_price": book.get("last_trade_price") if book else None,
                    "liquidity": rec["liquidity"],
                    "volume": rec["volume"],
                    "error": None if book else "book_missing",
                }
            )

    df = pd.DataFrame(rows)

    # helpful diagnostic line (shows if token parsing is still broken)
    hit = df["book_hash"].notna().sum()
    total = df["token_id"].notna().sum()
    print(f"[{league}] books present rows: {hit}/{max(1,total)} ({hit/max(1,total):.2%}) errors={len(errors)}")
    if errors:
        print(f"[{league}] sample error: {errors[0]}")

    return df


def append_daily_csv(df: pd.DataFrame, league: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(OUT_DIR, f"{league.lower()}_{day}.csv")

    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)

    return path


def main():
    leagues_env = os.getenv("LEAGUES", "UCL,EPL")
    leagues = [x.strip().upper() for x in leagues_env.split(",") if x.strip()]

    for lg in leagues:
        if lg not in LEAGUE_DEFAULTS:
            raise RuntimeError(f"Unsupported league '{lg}'. Supported: {list(LEAGUE_DEFAULTS.keys())}")

    for lg in leagues:
        tag_env = LEAGUE_DEFAULTS[lg]["tag_env"]
        tag_id = os.getenv(tag_env, LEAGUE_DEFAULTS[lg]["tag_default"])

        df = snapshot_league(lg, tag_id)
        out = append_daily_csv(df, lg)

        print(f"saved: {out} rows={len(df)} (tag_id={tag_id})")


if __name__ == "__main__":
    main()