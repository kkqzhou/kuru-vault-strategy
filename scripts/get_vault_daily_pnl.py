#!/usr/bin/env python3
"""
Fetch vault performance snapshots from the Kuru API and compute PnL for a day.

Examples:
  ./venv/bin/python scripts/get_vault_daily_pnl.py --date 20260324
  ./venv/bin/python scripts/get_vault_daily_pnl.py --date 20260324 --time-period 3M --write-csv
  ./venv/bin/python scripts/get_vault_daily_pnl.py --vault 0xanothervault --date 20260324
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import requests

API_BASE_URL = os.getenv("KURU_API_BASE_URL", "https://api.kuru.io/api/v3")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data")
DEFAULT_VAULT_ADDRESS = "0xd0f8a6422ccdd812f29d8fb75cf5fcd41483badc"


def parse_date(date_str: str) -> tuple[datetime, datetime]:
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Invalid date '{date_str}'. Use YYYYMMDD format, e.g. 20260324.")
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
    start = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def fetch_performance(vault: str, time_period: str, metric: str) -> list[dict]:
    url = f"{API_BASE_URL}/vaults/{vault}/performance"
    resp = requests.get(
        url,
        params={"timePeriod": time_period, "metric": metric},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    if not payload.get("success"):
        raise ValueError(f"API returned success=false: {payload}")

    rows = payload.get("data", {}).get("data", [])
    if not isinstance(rows, list):
        raise TypeError(f"Expected payload.data.data to be a list, got {type(rows).__name__}")
    return rows


def normalize_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        ts = datetime.fromisoformat(row["snapshotTimestamp"].replace("Z", "+00:00"))
        normalized.append({
            "snapshotTimestamp": ts,
            "tvl": Decimal(str(row["tvl"])),
            "totalPnl": Decimal(str(row["totalPnl"])),
        })
    normalized.sort(key=lambda row: row["snapshotTimestamp"])
    return normalized


def latest_snapshot_at_or_before(rows: list[dict], boundary: datetime) -> dict | None:
    candidate = None
    for row in rows:
        if row["snapshotTimestamp"] <= boundary:
            candidate = row
        else:
            break
    return candidate


def earliest_snapshot_at_or_after(rows: list[dict], boundary: datetime) -> dict | None:
    for row in rows:
        if row["snapshotTimestamp"] >= boundary:
            return row
    return None


def write_snapshots_csv(vault: str, date_str: str, rows: list[dict]) -> str:
    output_path = os.path.join(OUTPUT_DIR, f"vault_performance_{vault}_{date_str}.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snapshotTimestamp", "tvl", "totalPnl"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "snapshotTimestamp": row["snapshotTimestamp"].isoformat(),
                "tvl": str(row["tvl"]),
                "totalPnl": str(row["totalPnl"]),
            })
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compute a vault's PnL for a specific UTC day.")
    parser.add_argument("--vault", default=DEFAULT_VAULT_ADDRESS,
                        help=f"Vault address (default: {DEFAULT_VAULT_ADDRESS}).")
    parser.add_argument("--date", required=True, metavar="YYYYMMDD",
                        help="UTC day to measure, e.g. 20260324.")
    parser.add_argument("--time-period", default="1M",
                        help="Performance API timePeriod to request (default: 1M).")
    parser.add_argument("--metric", default="pnl",
                        help="Performance API metric to request (default: pnl).")
    parser.add_argument("--write-csv", action="store_true",
                        help="Write the returned performance snapshots to data/.")
    args = parser.parse_args()

    day_start, day_end = parse_date(args.date)

    print(f"Vault      : {args.vault}")
    print(f"UTC window : {day_start.isoformat()} -> {day_end.isoformat()}")
    print(f"API query  : timePeriod={args.time_period}, metric={args.metric}")

    try:
        rows = normalize_rows(fetch_performance(args.vault, args.time_period, args.metric))
    except Exception as exc:
        print(f"Error fetching vault performance: {exc}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("No performance snapshots returned.", file=sys.stderr)
        sys.exit(1)

    start_snapshot = latest_snapshot_at_or_before(rows, day_start)
    end_snapshot = latest_snapshot_at_or_before(rows, day_end)
    first_in_day = earliest_snapshot_at_or_after(rows, day_start)
    last_in_day = latest_snapshot_at_or_before(rows, day_end - timedelta(microseconds=1))

    if start_snapshot is None:
        print(
            "No snapshot exists at or before the start of the requested day. "
            "Use a larger --time-period so the boundary can be anchored.",
            file=sys.stderr,
        )
        sys.exit(1)
    if end_snapshot is None:
        print(
            "No snapshot exists at or before the end of the requested day. "
            "The API response does not yet cover that boundary.",
            file=sys.stderr,
        )
        sys.exit(1)

    daily_pnl = end_snapshot["totalPnl"] - start_snapshot["totalPnl"]

    print()
    print(f"Start anchor : {start_snapshot['snapshotTimestamp'].isoformat()}  totalPnl={start_snapshot['totalPnl']}")
    print(f"End anchor   : {end_snapshot['snapshotTimestamp'].isoformat()}  totalPnl={end_snapshot['totalPnl']}")
    if first_in_day is not None:
        print(f"First in day : {first_in_day['snapshotTimestamp'].isoformat()}  totalPnl={first_in_day['totalPnl']}")
    if last_in_day is not None:
        print(f"Last in day  : {last_in_day['snapshotTimestamp'].isoformat()}  totalPnl={last_in_day['totalPnl']}")
    print(f"Daily PnL    : {daily_pnl:+f}")

    if args.write_csv:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = write_snapshots_csv(args.vault.lower(), args.date, rows)
        print(f"Wrote {len(rows)} snapshots to {output_path}")


if __name__ == "__main__":
    main()
