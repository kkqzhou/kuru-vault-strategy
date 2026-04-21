#!/usr/bin/env python3
"""
Fetch Hyperliquid historical orders (the full lifecycle of every order
placed: placed -> filled / partially filled / cancelled / rejected).

Unlike fills, this includes orders that never filled, which is the signal
needed to distinguish "hedge intent is wrong" from "hedge intent is right
but orders aren't filling".

Usage:
  python scripts/save_hl_orders.py --date 20260421
  python scripts/save_hl_orders.py --date 20260421 --coin MON
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import requests

DEFAULT_ADDRESS = "0xa59c4CB5C24983d1F0076a52d4F0e95cc5013Df5"
HL_API_URL = os.getenv("HL_API_URL", "https://api.hyperliquid.xyz")
INFO_URL = f"{HL_API_URL}/info"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data')


def parse_date(date_str: str) -> tuple[datetime, datetime]:
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Invalid date '{date_str}'. Use YYYYMMDD format.")
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
    start = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def hl_post(payload: dict):
    resp = requests.post(INFO_URL, json=payload,
                         headers={"Content-Type": "application/json"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_historical_orders(address: str) -> list[dict]:
    return hl_post({"type": "historicalOrders", "user": address})


def flatten_order(record: dict) -> dict:
    """HL returns [{order: {...}, status: str, statusTimestamp: int}, ...]."""
    order = record.get("order", {})
    status = record.get("status", "")
    status_ts = record.get("statusTimestamp", 0)
    placed_ts = order.get("timestamp", 0)

    def iso(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat() if ms else ""

    orig_sz = float(order.get("origSz", 0))
    sz_remaining = float(order.get("sz", 0))
    filled_sz = orig_sz - sz_remaining

    return {
        "time_placed":   iso(placed_ts),
        "time_updated":  iso(status_ts),
        "oid":           order.get("oid"),
        "coin":          order.get("coin"),
        "side":          "buy" if order.get("side") == "B" else "sell",
        "price":         order.get("limitPx"),
        "orig_size":     orig_sz,
        "size_remaining": sz_remaining,
        "size_filled":   filled_sz,
        "status":        status,
        "tif":           order.get("tif", ""),
        "reduce_only":   order.get("reduceOnly", False),
        "trigger_px":    order.get("triggerPx"),
        "is_trigger":    order.get("isTrigger", False),
    }


def filter_window(records: list[dict], after: datetime, before: datetime,
                  coin: str | None) -> list[dict]:
    """Keep records where the order was placed OR last updated inside [after, before)."""
    after_ms = int(after.timestamp() * 1000)
    before_ms = int(before.timestamp() * 1000)
    out = []
    for r in records:
        order = r.get("order", {})
        placed = order.get("timestamp", 0)
        updated = r.get("statusTimestamp", 0)
        overlaps = (after_ms <= placed < before_ms) or (after_ms <= updated < before_ms)
        if not overlaps:
            continue
        if coin and order.get("coin") != coin:
            continue
        out.append(r)
    return out


def print_summary(records: list[dict]):
    """
    HL's historicalOrders returns one record per status transition, so each
    oid typically appears twice (placed as 'open', then a terminal status).
    We dedupe by (oid, status) and count each oid's terminal status only.
    """
    if not records:
        print("No orders found in window.")
        return

    # Keep the latest status per oid (priority: non-'open' wins over 'open')
    latest: dict = {}
    for r in records:
        oid = r.get("order", {}).get("oid")
        status = r.get("status", "")
        if oid not in latest:
            latest[oid] = r
        else:
            # prefer a terminal status over 'open'
            prev = latest[oid].get("status", "")
            if prev == "open" and status != "open":
                latest[oid] = r
            elif r.get("statusTimestamp", 0) > latest[oid].get("statusTimestamp", 0):
                latest[oid] = r

    unique = list(latest.values())
    by_status: dict[str, int] = {}
    by_side: dict[str, int] = {}
    total_notional = 0.0
    filled_notional = 0.0

    for r in unique:
        status = r.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

        o = r.get("order", {})
        side = "buy" if o.get("side") == "B" else "sell"
        by_side[side] = by_side.get(side, 0) + 1

        orig = float(o.get("origSz", 0))
        rem = float(o.get("sz", 0))
        px = float(o.get("limitPx", 0))
        total_notional += orig * px
        filled_notional += (orig - rem) * px

    print(f"\n{'─'*64}")
    print(f"  HISTORICAL ORDERS SUMMARY  ({len(unique)} unique orders, {len(records)} raw records)")
    print(f"{'─'*64}")
    print(f"  Buys / Sells        {by_side.get('buy',0)} / {by_side.get('sell',0)}")
    print(f"  Total orig notional ${total_notional:>14,.2f}")
    fill_rate = filled_notional/total_notional*100 if total_notional else 0
    print(f"  Filled notional     ${filled_notional:>14,.2f}   ({fill_rate:.1f}%)")
    print(f"  Status breakdown:")
    for status, n in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"    {status:<24}{n:>6}")
    print(f"{'─'*64}\n")


def write_csv(records: list[dict], output_path: str):
    if not records:
        return
    rows = [flatten_order(r) for r in records]
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} orders to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Hyperliquid historical orders to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS,
                        help="HL wallet address")
    parser.add_argument("--date", metavar="YYYYMMDD", required=True,
                        help="Fetch orders for a specific UTC day, e.g. 20260421")
    parser.add_argument("--coin", default=None,
                        help="Filter by coin, e.g. MON")
    args = parser.parse_args()

    after, before = parse_date(args.date)
    print(f"Address : {args.address}")
    print(f"Window  : {after.isoformat()} → {before.isoformat()}")
    if args.coin:
        print(f"Coin    : {args.coin}")

    print("\nFetching historical orders...")
    try:
        all_orders = fetch_historical_orders(args.address)
    except Exception as e:
        print(f"Error fetching historical orders: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  API returned {len(all_orders)} orders (across entire account history)")
    records = filter_window(all_orders, after, before, args.coin)
    print_summary(records)

    output = os.path.join(OUTPUT_DIR, f"hl_orders_{args.date}.csv")
    write_csv(records, output)


if __name__ == "__main__":
    main()
