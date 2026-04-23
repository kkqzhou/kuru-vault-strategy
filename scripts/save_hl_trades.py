#!/usr/bin/env python3
"""
Fetch Hyperliquid open orders and fills for the HL maker strategy.

Usage:
  python scripts/fetch_hl_data.py --last-duration 24h
  python scripts/fetch_hl_data.py --last-duration 7d
  python scripts/fetch_hl_data.py --date 0323          # March 23rd of current year
  python scripts/fetch_hl_data.py --address 0xABC... --last-duration 1h
  python scripts/fetch_hl_data.py --last-duration 24h --coin MON
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import requests

DEFAULT_ADDRESS = "0xa59c4CB5C24983d1F0076a52d4F0e95cc5013Df5"
HL_API_URL = os.getenv("HL_API_URL", "https://api.hyperliquid.xyz")
INFO_URL = f"{HL_API_URL}/info"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data')
FILL_PAGE_LIMIT = 2000
MAX_RECENT_FILLS_AVAILABLE = 10000

# ── Time helpers ──────────────────────────────────────────────────────────────

def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string like 24h, 7d, 30m into a timedelta."""
    match = re.fullmatch(r"(\d+)([smhd])", duration_str.strip().lower())
    if not match:
        raise ValueError(f"Invalid duration '{duration_str}'. Use format like 30m, 24h, 7d.")
    value, unit = int(match.group(1)), match.group(2)
    return {"s": timedelta(seconds=value), "m": timedelta(minutes=value),
            "h": timedelta(hours=value),  "d": timedelta(days=value)}[unit]


def parse_date(date_str: str) -> tuple[datetime, datetime]:
    """Parse YYYYMMDD string and return (start_of_day, end_of_day) in UTC."""
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Invalid date '{date_str}'. Use YYYYMMDD format, e.g. 20260323 for March 23, 2026.")
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
    start = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# ── HL API helpers ────────────────────────────────────────────────────────────

def hl_post(payload: dict) -> dict | list:
    resp = requests.post(
        INFO_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_open_orders(address: str) -> list[dict]:
    return hl_post({"type": "openOrders", "user": address})


def fill_dedupe_key(fill: dict) -> tuple:
    return (
        fill.get("tid"),
        fill.get("hash"),
        fill.get("oid"),
        fill.get("time"),
        fill.get("coin"),
        fill.get("px"),
        fill.get("sz"),
        fill.get("side"),
    )


def fetch_fills_by_time(address: str, after: datetime, before: datetime) -> list[dict]:
    """
    Fetch fills using the userFillsByTime endpoint (ms timestamps).

    Hyperliquid caps each response at 2000 fills. In practice the endpoint can
    behave like "first 2000 fills in the requested time range", so we page
    forward using the newest fill we have already seen rather than relying on
    the response being reverse-chronological.
    """
    start_ms = to_ms(after)
    final_end_ms = to_ms(before)
    all_fills: list[dict] = []
    seen: set[tuple] = set()
    page_num = 0
    hit_recent_cap = False

    while start_ms <= final_end_ms:
        page_num += 1
        page = hl_post({
            "type": "userFillsByTime",
            "user": address,
            "startTime": start_ms,
            "endTime": final_end_ms,
        })

        if not isinstance(page, list):
            raise TypeError(f"Expected list of fills, got {type(page).__name__}")
        if not page:
            break

        page = sorted(page, key=lambda row: int(row.get("time", 0)), reverse=True)
        added = 0
        for fill in page:
            key = fill_dedupe_key(fill)
            if key in seen:
                continue
            seen.add(key)
            all_fills.append(fill)
            added += 1

        oldest_ms = min(int(fill.get("time", 0)) for fill in page)
        newest_ms = max(int(fill.get("time", 0)) for fill in page)
        oldest_dt = datetime.fromtimestamp(oldest_ms / 1000, tz=timezone.utc).isoformat()
        newest_dt = datetime.fromtimestamp(newest_ms / 1000, tz=timezone.utc).isoformat()
        print(
            f"  fills page {page_num}: received {len(page)} rows "
            f"({added} new), range {oldest_dt} -> {newest_dt}"
        )

        if len(page) < FILL_PAGE_LIMIT:
            break

        if len(all_fills) >= MAX_RECENT_FILLS_AVAILABLE:
            hit_recent_cap = True
            break

        next_start_ms = newest_ms + 1
        if next_start_ms <= start_ms:
            # Defensive guard: avoid an infinite loop if the API does not move forward.
            break
        start_ms = next_start_ms

    all_fills.sort(key=lambda row: int(row.get("time", 0)))

    if hit_recent_cap:
        earliest_dt = datetime.fromtimestamp(to_ms(after) / 1000, tz=timezone.utc).isoformat()
        print(
            "Warning: reached Hyperliquid's recent-fill ceiling while paginating. "
            f"The API only exposes the {MAX_RECENT_FILLS_AVAILABLE} most recent fills, "
            f"so older fills at or before {earliest_dt} may be unavailable."
        )

    return all_fills


# ── Display / serialise ───────────────────────────────────────────────────────

def print_open_orders(orders: list[dict], coin_filter: str | None):
    if coin_filter:
        orders = [o for o in orders if o.get("coin") == coin_filter]

    print(f"\n{'═'*64}")
    print(f"  OPEN ORDERS  ({len(orders)} total)")
    print(f"{'═'*64}")

    if not orders:
        print("  (none)")
    else:
        header = f"  {'OID':<12} {'Coin':<8} {'Side':<6} {'Price':>12} {'Sz':>12} {'Filled':>10}  {'TIF':<6}  Placed"
        print(header)
        print(f"  {'-'*95}")
        for o in orders:
            side  = "buy" if o.get("side") == "B" else "sell"
            oid   = o.get("oid", "")
            coin  = o.get("coin", "")
            price = float(o.get("limitPx", 0))
            sz    = float(o.get("sz", 0))
            orig  = float(o.get("origSz", sz))
            filled = orig - sz
            tif   = o.get("orderType", "")
            ts_ms = o.get("timestamp", 0)
            ts_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if ts_ms else ""
            print(f"  {oid!s:<12} {coin:<8} {side:<6} {price:>12.5g} {sz:>12.4f} {filled:>10.4f}  {tif:<6}  {ts_str}")

    print()


def flatten_fill(fill: dict) -> dict:
    ts_ms = fill.get("time", 0)
    ts_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat() if ts_ms else ""
    return {
        "time":       ts_str,
        "coin":       fill.get("coin"),
        "side":       "buy" if fill.get("side") == "B" else "sell",
        "price":      fill.get("px"),
        "size":       fill.get("sz"),
        "fee":        fill.get("fee"),
        "fee_token":  fill.get("feeToken"),
        "closed_pnl": fill.get("closedPnl"),
        "hash":       fill.get("hash"),
        "oid":        fill.get("oid"),
        "crossed":    fill.get("crossed"),   # True = taker, False = maker
        "dir":        fill.get("dir"),       # e.g. "Open Long"
        "start_pos":  fill.get("startPosition"),
    }


def write_fills_csv(fills: list[dict], coin_filter: str | None, output_path: str):
    if coin_filter:
        fills = [f for f in fills if f.get("coin") == coin_filter]

    if not fills:
        print("No fills found for the specified window.")
        return

    rows = [flatten_fill(f) for f in fills]
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} fills to {output_path}")


def print_fills_summary(fills: list[dict], coin_filter: str | None):
    if coin_filter:
        fills = [f for f in fills if f.get("coin") == coin_filter]

    if not fills:
        return

    buys  = [f for f in fills if f.get("side") == "B"]
    sells = [f for f in fills if f.get("side") == "S"]

    def notional(rows):
        return sum(float(r.get("px", 0)) * float(r.get("sz", 0)) for r in rows)

    def vol(rows):
        return sum(float(r.get("sz", 0)) for r in rows)

    total_notional = notional(fills)
    total_fee = sum(float(f.get("fee", 0)) for f in fills)
    total_pnl = sum(float(f.get("closedPnl", 0)) for f in fills)
    maker_fills = [f for f in fills if not f.get("crossed")]
    taker_fills = [f for f in fills if f.get("crossed")]

    print(f"{'─'*64}")
    print(f"  FILLS SUMMARY  ({len(fills)} fills)")
    print(f"{'─'*64}")
    print(f"  Buys              {len(buys)} trades  |  {vol(buys):.4f} tokens  |  ${notional(buys):,.2f}")
    print(f"  Sells             {len(sells)} trades  |  {vol(sells):.4f} tokens  |  ${notional(sells):,.2f}")
    print(f"  Total notional    ${total_notional:,.2f}")
    print(f"  Maker / Taker     {len(maker_fills)} / {len(taker_fills)} fills")
    print(f"  Total fees        ${total_fee:.4f}")
    print(f"  Closed PnL        ${total_pnl:.4f}")
    print(f"{'─'*64}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Hyperliquid open orders and fills to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS,
                        help="HL wallet address (or set HL_ACCOUNT_ADDRESS env var)")
    parser.add_argument("--date", metavar="YYYYMMDD", default=(datetime.now() - timedelta(days=1)).strftime('%Y%m%d'),
                        help="Fetch fills for a specific day, e.g. 20260323 for March 23, 2026")
    parser.add_argument("--coin", default=None,
                        help="Filter by coin, e.g. MON (default: all coins)")
    parser.add_argument("--no-open-orders", action="store_true",
                        help="Skip fetching open orders")
    args = parser.parse_args()
    start, end = parse_date(args.date)
    label = args.date

    coin_label = f"_{args.coin}" if args.coin else ""
    output = os.path.join(OUTPUT_DIR, f"hl_fills_{label}.csv")

    print(f"Address : {args.address}")
    print(f"Window  : {start.isoformat()} → {end.isoformat()}")
    if args.coin:
        print(f"Coin    : {args.coin}")

    # ── Open orders ───────────────────────────────────────────────────────────
    if not args.no_open_orders:
        print("\nFetching open orders...")
        try:
            open_orders = fetch_open_orders(args.address)
            print_open_orders(open_orders, args.coin)
        except Exception as e:
            print(f"Error fetching open orders: {e}", file=sys.stderr)

    # ── Fills ─────────────────────────────────────────────────────────────────
    print("Fetching fills...")
    try:
        fills = fetch_fills_by_time(args.address, start, end)
    except Exception as e:
        print(f"Error fetching fills: {e}", file=sys.stderr)
        sys.exit(1)

    print_fills_summary(fills, args.coin)
    write_fills_csv(fills, args.coin, output)


if __name__ == "__main__":
    main()
