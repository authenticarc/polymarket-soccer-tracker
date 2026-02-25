import os
import time
import datetime as dt
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "polymarket-ucl-epl-tracker/0.3",
        "Accept": "application/json",
    }
)

# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_get(url: str, params: Optional[dict] = None, timeout: int = 30) -> Any:
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def safe_post(url: str, payload: Any, timeout: int = 30) -> Any:
    r = SESSION.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def best_level(side: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not side:
        return {"price": None, "size": None}
    try:
        return {"price": float(side[0]["price"]), "size": float(side[0]["size"])}
    except Exception:
        return {"price": None, "size": None}


# ----------------------------
# League mapping & discovery
# ----------------------------
LEAGUE_TO_SPORT_CODE = {
    "UCL": "ucl",
    "EPL": "epl",
}

def discover_sport_id(league_code: str, sports_payload: List[Dict[str, Any]]) -> int:
    """
    Prefer env override SPORT_ID_<LEAGUE>.
    Otherwise, match by sports_payload[].sport == LEAGUE_TO_SPORT_CODE[league_code].
    """
    override = os.getenv(f"SPORT_ID_{league_code}")
    if override:
        return int(override)

    sport_code = LEAGUE_TO_SPORT_CODE.get(league_code)
    if not sport_code:
        raise ValueError(f"Unknown league_code={league_code}. Add to LEAGUE_TO_SPORT_CODE.")

    for s in sports_payload:
        if (s.get("sport") or "").lower() == sport_code:
            return int(s["id"])

    raise RuntimeError(
        f"Could not auto-detect sport_id for {league_code} (sport='{sport_code}'). "
        f"Set env SPORT_ID_{league_code} manually."
    )


# ----------------------------
# Fetch events & markets
# ----------------------------
def fetch_events_by_sport(sport_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch active, open events for a given sport_id.
    """
    events: List[Dict[str, Any]] = []
    offset = 0
    while True:
        batch = safe_get(
            f"{GAMMA}/events",
            params={
                "sport_id": sport_id,
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
            },
        )
        if not batch:
            break
        events.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return events


def flatten_markets(events: List[Dict[str, Any]], league_code: str) -> pd.DataFrame:
    """
    Expand events -> markets -> token rows.
    Record full market metadata (all markets, no filtering).
    """
    rows = []
    for ev in events:
        ev_id = ev.get("id")
        ev_slug = ev.get("slug")
        ev_title = ev.get("title") or ev.get("name")
        ev_start = ev.get("startDate") or ev.get("start_date") or ev.get("eventStartTime")

        mkts = ev.get("markets") or []
        for m in mkts:
            # Keep all orderbook-enabled markets
            if m.get("enableOrderBook") is False:
                continue

            clob_tokens = m.get("clobTokenIds") or m.get("clob_token_ids") or []
            if not clob_tokens:
                continue

            rows.append(
                {
                    "league": league_code,
                    "event_id": ev_id,
                    "event_slug": ev_slug,
                    "event_title": ev_title,
                    "event_start": ev_start,
                    "market_id": m.get("id"),
                    "market_slug": m.get("slug"),
                    "question": m.get("question") or m.get("title"),
                    "outcomes": m.get("outcomes"),
                    "outcomePrices": m.get("outcomePrices"),
                    "clobTokenIds": clob_tokens,
                    "liquidity": m.get("liquidity"),
                    "volume": m.get("volume") or m.get("volumeNum") or m.get("volume_num"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # One row per token id
    df = df.explode("clobTokenIds", ignore_index=True).rename(columns={"clobTokenIds": "token_id"})
    df["token_id"] = df["token_id"].astype(str)
    return df


# ----------------------------
# Fetch orderbooks in batch
# ----------------------------
def fetch_books(token_ids: List[str], batch_size: int = 200) -> Dict[str, Dict[str, Any]]:
    """
    Batch fetch orderbooks from CLOB.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for part in chunk(token_ids, batch_size):
        payload = [{"token_id": tid} for tid in part]
        books = safe_post(f"{CLOB}/books", payload)

        if isinstance(books, list):
            for b in books:
                tid = str(b.get("asset_id") or b.get("token_id") or "")
                if tid:
                    out[tid] = b
        elif isinstance(books, dict):
            for tid, b in books.items():
                out[str(tid)] = b

        time.sleep(0.2)
    return out


def main():
    leagues_env = os.getenv("LEAGUES", "UCL,EPL")
    league_codes = [x.strip().upper() for x in leagues_env.split(",") if x.strip()]
    if not league_codes:
        raise RuntimeError("LEAGUES is empty")

    sports = safe_get(f"{GAMMA}/sports")

    # Discover sport_ids for requested leagues
    league_sport_ids: Dict[str, int] = {}
    for code in league_codes:
        league_sport_ids[code] = discover_sport_id(code, sports)

    # Fetch events & markets for each league
    all_mkts = []
    for code, sport_id in league_sport_ids.items():
        events = fetch_events_by_sport(sport_id=sport_id, limit=100)
        mkts = flatten_markets(events, league_code=code)
        if not mkts.empty:
            all_mkts.append(mkts)

    if not all_mkts:
        print("No markets found for leagues:", league_codes)
        return

    mkts_df = pd.concat(all_mkts, ignore_index=True)
    mkts_df = mkts_df.drop_duplicates(subset=["league", "market_id", "token_id"]).reset_index(drop=True)

    token_ids = mkts_df["token_id"].dropna().unique().tolist()
    books = fetch_books(token_ids)

    ts = utc_now_iso()

    rows = []
    for _, r in mkts_df.iterrows():
        tid = str(r["token_id"])
        book = books.get(tid, {}) or {}
        bids = book.get("bids") or []
        asks = book.get("asks") or []

        bid0 = best_level(bids)
        ask0 = best_level(asks)

        mid = None
        spr = None
        if bid0["price"] is not None and ask0["price"] is not None:
            mid = (bid0["price"] + ask0["price"]) / 2.0
            spr = (ask0["price"] - bid0["price"])

        rows.append(
            {
                "ts_utc": ts,
                "league": r.get("league"),
                "event_id": r.get("event_id"),
                "event_slug": r.get("event_slug"),
                "event_title": r.get("event_title"),
                "event_start": r.get("event_start"),
                "market_id": r.get("market_id"),
                "market_slug": r.get("market_slug"),
                "question": r.get("question"),
                "token_id": tid,
                "best_bid": bid0["price"],
                "best_bid_size": bid0["size"],
                "best_ask": ask0["price"],
                "best_ask_size": ask0["size"],
                "mid": mid,
                "spread": spr,
                "book_hash": book.get("hash"),
                "tick_size": book.get("tick_size"),
                "min_order_size": book.get("min_order_size"),
                "liquidity": r.get("liquidity"),
                "volume": r.get("volume"),
            }
        )

    out_df = pd.DataFrame(rows)

    os.makedirs("data", exist_ok=True)
    day = dt.datetime.utcnow().date().isoformat()

    # Write per league
    for code in league_codes:
        part = out_df[out_df["league"] == code].copy()
        if part.empty:
            continue
        path = f"data/{code.lower()}_{day}.csv"
        write_header = not os.path.exists(path)
        part.to_csv(path, mode="a", index=False, header=write_header)
        print(f"Wrote {len(part)} rows -> {path}")

if __name__ == "__main__":
    main()