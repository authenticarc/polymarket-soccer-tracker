#!/usr/bin/env python3
"""
Polymarket Soccer Tracker (UCL + EPL)

Fixes:
- Robust clobTokenIds parsing (handles doubled quotes like [""a"", ""b""])
- GUARANTEE token_ids is List[str] (never a raw JSON string)
- Batch fetch via POST /books, with fallback GET /book?token_id=...
- Compute best bid/ask without assuming order sorting
- Write extra debug columns to CSV for diagnosis

Env:
- GAMMA_BASE (default https://gamma-api.polymarket.com)
- CLOB_BASE  (default https://clob.polymarket.com)
- LEAGUES    (default "UCL,EPL")
- TAG_UCL    (default "100977")
- TAG_EPL    (default "82")
- LIMIT_EVENTS (default 200)
- TIMEOUT_SEC (default 20)
- OUT_DIR    (default "data")
- DEBUG      (default "0")  -> set "1" for debug prints
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
DEBUG = os.getenv("DEBUG", "0") == "1"

UA = "polymarket-soccer-tracker/1.4 (+github-actions)"

LEAGUE_DEFAULTS = {
    "UCL": {"tag_env": "TAG_UCL", "tag_default": "100977"},
    "EPL": {"tag_env": "TAG_EPL", "tag_default": "82"},
}

# Polymarket CLOB token ids are usually huge decimal strings
TOKEN_EXTRACT_RE = re.compile(r"(0x[0-9a-fA-F]+|\d{10,})")


# -----------------------------
# HTTP helpers
# -----------------------------
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": UA, "Accept": "application/json"})


def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    r = _SESSION.get(url, params=params, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


def _post(url: str, payload: Any) -> Any:
    r = _SESSION.post(url, json=payload, timeout=TIMEOUT_SEC, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _dedup_preserve_order(vals: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# -----------------------------
# Parsing: clobTokenIds (critical)
# -----------------------------
def parse_clob_token_ids(raw: Any) -> List[str]:
    """
    Accepts:
      - list: ["5860...", "1234..."]
      - JSON string: '["5860...","1234..."]'
      - doubled-quote JSON string: '[""5860..."", ""1234...""]'
      - truncated/malformed -> regex fallback
    Returns: list[str] of plausible token ids (digits or 0x..), deduped.
    """
    if raw is None:
        return []

    # already list
    if isinstance(raw, list):
        return _clean_token_list([str(x) for x in raw if x is not None])

    # string-like
    s = str(raw).strip()
    if not s:
        return []

    # 1) normalize doubled quotes: [""a"", ""b""] -> ["a", "b"]
    s_norm = s.replace('""', '"')

    # 2) try json.loads
    try:
        y = json.loads(s_norm)
        if isinstance(y, list):
            return _clean_token_list([str(x) for x in y if x is not None])
    except Exception:
        pass

    # 3) regex fallback (works even if truncated)
    extracted = [m.group(1) for m in TOKEN_EXTRACT_RE.finditer(s_norm)]
    return _clean_token_list(extracted)


def _clean_token_list(vals: List[str]) -> List[str]:
    out: List[str] = []
    for v in vals:
        if v is None:
            continue
        v2 = str(v).strip().strip('"').strip("'")
        if not v2 or v2 in ("[", "]"):
            continue

        if v2.startswith("0x"):
            out.append(v2)
        elif v2.isdigit() and len(v2) >= 10:
            out.append(v2)
        else:
            # if it's still a fragment, extract tokens from it
            frag = [m.group(1) for m in TOKEN_EXTRACT_RE.finditer(v2)]
            out.extend(frag)

    return _dedup_preserve_order(out)


# -----------------------------
# Gamma fetching
# -----------------------------
def fetch_events_by_tag(tag_id: str) -> List[Dict[str, Any]]:
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
                "related_tags": "true",
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        events.extend(batch)
        offset += len(batch)
        if len(batch) < page:
            break

    return events


def fetch_market_full(market_id: Any) -> Optional[Dict[str, Any]]:
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
    payload = [{"token_id": str(t)} for t in token_ids]
    res = _post(f"{CLOB_BASE}/books", payload)
    return res if isinstance(res, list) else []


def fetch_book_single(token_id: str) -> Optional[Dict[str, Any]]:
    # /book?token_id=...
    try:
        return _get(f"{CLOB_BASE}/book", params={"token_id": str(token_id)})
    except Exception:
        return None


def _pick_best_level(levels: List[Dict[str, Any]], pick: str) -> Tuple[Optional[float], Optional[float]]:
    best_price = None
    best_size = None
    for lv in levels or []:
        try:
            p = float(lv.get("price"))
            s = float(lv.get("size"))
        except Exception:
            continue
        if best_price is None:
            best_price, best_size = p, s
            continue
        if pick == "max_price":
            if p > best_price:
                best_price, best_size = p, s
        else:
            if p < best_price:
                best_price, best_size = p, s
    return best_price, best_size


def best_from_book(book: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    bids = book.get("bids") or []
    asks = book.get("asks") or []

    best_bid, best_bid_size = _pick_best_level(bids, "max_price")
    best_ask, best_ask_size = _pick_best_level(asks, "min_price")

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

    if DEBUG:
        print(f"[{league}] events={len(events)} tag_id={tag_id}")

    # market_records: one per market, holds token list
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

            raw_clob = m.get("clobTokenIds")
            token_ids = parse_clob_token_ids(raw_clob)
            ob = enable_orderbook(m)

            # If missing or light payload, fetch full market and retry
            if (not token_ids) or (ob is False) or (m.get("conditionId") is None):
                full = fetch_market_full(m_id)
                if full:
                    raw_clob = full.get("clobTokenIds")
                    token_ids = parse_clob_token_ids(raw_clob)
                    ob = enable_orderbook(full)

                    m_slug = full.get("slug") or m_slug
                    m_question = full.get("question") or full.get("title") or m_question
                    m_liq = full.get("liquidity", m_liq)
                    m_vol = full.get("volume") or full.get("volumeNum") or full.get("volume_num") or m_vol
                    m = full

            if not ob:
                continue

            # HARD GUARD: token_ids must be list[str] of digits/0x; otherwise skip and log
            if not isinstance(token_ids, list) or any((not isinstance(t, str) or (not t.isdigit() and not t.startswith("0x"))) for t in token_ids):
                if DEBUG:
                    print(f"[{league}] BAD token_ids market_id={m_id} raw={raw_clob}")
                continue

            if len(token_ids) == 0:
                continue

            outcomes = m.get("outcomes")
            if isinstance(outcomes, str):
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
                    "clob_token_ids_raw": raw_clob,
                    "token_ids": token_ids,
                    "outcomes": outcomes if isinstance(outcomes, list) else None,
                    "liquidity": m_liq,
                    "volume": m_vol,
                }
            )

    # Collect unique tokens for batch books
    all_tokens: List[str] = []
    for rec in market_records:
        all_tokens.extend(rec["token_ids"])
    all_tokens = _dedup_preserve_order(all_tokens)

    if not all_tokens:
        return pd.DataFrame([])

    token_to_book: Dict[str, Dict[str, Any]] = {}

    # 1) batch fetch
    for part in chunk(all_tokens, 200):
        books = fetch_books_batch(part)
        for b in books:
            aid = b.get("asset_id") or b.get("token_id")
            if aid is not None:
                token_to_book[str(aid)] = b
        # safety: align by request order if needed
        if isinstance(books, list) and len(books) == len(part):
            for req, b in zip(part, books):
                token_to_book.setdefault(req, b)
        time.sleep(0.12)

    # 2) expand to per-token rows; fallback to single-book if still missing
    rows: List[Dict[str, Any]] = []

    for rec in market_records:
        outcomes = rec["outcomes"] or []
        for idx, t in enumerate(rec["token_ids"]):
            book = token_to_book.get(str(t))

            # fallback if missing
            if book is None:
                book = fetch_book_single(str(t))
                if book is not None:
                    token_to_book[str(t)] = book

            book_found = book is not None
            best_bid, best_bid_size, best_ask, best_ask_size, mid, spr = (
                best_from_book(book) if book_found else (None, None, None, None, None, None)
            )

            outcome = outcomes[idx] if idx < len(outcomes) else None

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
                    "clob_token_ids_raw": rec["clob_token_ids_raw"],
                    "token_ids_parsed": "|".join(rec["token_ids"]),
                    "token_id": str(t),
                    "outcome": outcome,
                    "book_found": book_found,
                    "best_bid": best_bid,
                    "best_bid_size": best_bid_size,
                    "best_ask": best_ask,
                    "best_ask_size": best_ask_size,
                    "mid": mid,
                    "spread": spr,
                    "book_hash": book.get("hash") if book_found else None,
                    "tick_size": book.get("tick_size") if book_found else None,
                    "min_order_size": book.get("min_order_size") if book_found else None,
                    "last_trade_price": book.get("last_trade_price") if book_found else None,
                    "liquidity": rec["liquidity"],
                    "volume": rec["volume"],
                }
            )

    df = pd.DataFrame(rows)
    hit = int(df["book_found"].sum()) if not df.empty else 0
    total = len(df) if not df.empty else 0
    print(f"[{league}] books found rows: {hit}/{max(1,total)} ({hit/max(1,total):.2%})")
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