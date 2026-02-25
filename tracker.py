import os
import time
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "polymarket-ucl-epl-tracker/0.4",
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
# League config (UCL + EPL only)
# ----------------------------
LEAGUES = {
    "UCL": {
        "sport_code": "ucl",
        "tag_keywords": ["champions league", "uefa champions league", "ucl"],
    },
    "EPL": {
        "sport_code": "epl",
        "tag_keywords": ["premier league", "epl", "english premier league"],
    },
}


def parse_tags_csv(tags: Optional[str]) -> List[int]:
    if not tags:
        return []
    out = []
    for x in str(tags).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out


def get_sport_meta_by_code(sports_payload: List[Dict[str, Any]], sport_code: str) -> Dict[str, Any]:
    for s in sports_payload:
        if (s.get("sport") or "").lower() == sport_code.lower():
            return s
    raise RuntimeError(f"Could not find sport metadata for sport='{sport_code}'.")


def choose_league_tag_id(league_code: str, sports_payload: List[Dict[str, Any]]) -> int:
    """
    Pick a league-specific tag_id by:
    1) taking sport metadata object's `tags` string,
    2) resolving each tag via /tags/{id},
    3) selecting the one whose name/slug best matches league keywords.
    Docs emphasize filtering events via tag_id. :contentReference[oaicite:3]{index=3}
    """
    # Allow hard override (most stable in production)
    override = os.getenv(f"TAG_{league_code}")
    if override:
        return int(override)

    cfg = LEAGUES[league_code]
    sport_meta = get_sport_meta_by_code(sports_payload, cfg["sport_code"])
    tag_ids = parse_tags_csv(sport_meta.get("tags"))

    if not tag_ids:
        raise RuntimeError(f"No tags found in /sports metadata for {league_code} ({cfg['sport_code']}).")

    keywords = [k.lower() for k in cfg["tag_keywords"]]

    # Resolve tags and score them
    best: Tuple[int, int, str] = (-1, -10**9, "")
    for tid in tag_ids:
        try:
            t = safe_get(f"{GAMMA}/tags/{tid}")
        except Exception:
            continue
        name = (t.get("name") or "").lower()
        slug = (t.get("slug") or "").lower()
        text = f"{name} {slug}"

        score = 0
        for kw in keywords:
            if kw in text:
                score += 10
        # Prefer longer/more specific names if tie
        score += min(len(name), 50) // 10

        if score > best[1]:
            best = (tid, score, text)

        time.sleep(0.05)

    if best[0] == -1 or best[1] <= 0:
        # As fallback, pick the largest tag id (often more specific than generic '1' etc.)
        # But strongly recommend setting TAG_<LEAGUE> env if this happens.
        return max(tag_ids)

    return best[0]


# ----------------------------
# Fetch events & markets (by tag_id)
# ----------------------------
def fetch_events_by_tag(tag_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Correct filter: /events?tag_id=...
    :contentReference[oaicite:4]{index=4}
    """
    events: List[Dict[str, Any]] = []
    offset = 0
    while True:
        batch = safe_get(
            f"{GAMMA}/events",
            params={
                "tag_id": tag_id,
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
                "related_tags": "true",
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
    rows = []
    for ev in events:
        ev_id = ev.get("id")
        ev_slug = ev.get("slug")
        ev_title = ev.get("title") or ev.get("name")
        ev_start = ev.get("startDate") or ev.get("start_date") or ev.get("eventStartTime")

        mkts = ev.get("markets") or []
        for m in mkts:
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

    df = df.explode("clobTokenIds", ignore_index=True).rename(columns={"clobTokenIds": "token_id"})
    df["token_id"] = df["token_id"].astype(str)
    return df


# ----------------------------
# Fetch orderbooks in batch
# ----------------------------
def fetch_books(token_ids: List[str], batch_size: int = 200) -> Dict[str, Dict[str, Any]]:
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
    for c in league_codes:
        if c not in LEAGUES:
            raise RuntimeError(f"Unsupported league '{c}'. Supported: {list(LEAGUES.keys())}")

    sports = safe_get(f"{GAMMA}/sports")

    # Resolve tag_id per league (robust)
    league_tag_ids: Dict[str, int] = {}
    for code in league_codes:
        league_tag_ids[code] = choose_league_tag_id(code, sports)

    all_mkts = []
    for code, tag_id in league_tag_ids.items():
        events = fetch_events_by_tag(tag_id=tag_id, limit=100)
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

    for code in league_codes:
        part = out_df[out_df["league"] == code].copy()
        if part.empty:
            continue
        path = f"data/{code.lower()}_{day}.csv"
        write_header = not os.path.exists(path)
        part.to_csv(path, mode="a", index=False, header=write_header)
        print(f"Wrote {len(part)} rows -> {path}")

    # Helpful debug line (so you can pin TAG_UCL/TAG_EPL once stable)
    print("League tag_ids used:", league_tag_ids)


if __name__ == "__main__":
    main()