"""
Plot cumulative hedge PnL (target vs actual) alongside unhedged position,
to localize where the hedge miss bleeds money.

Uses data/hedge_tracking_YYYYMMDD.csv produced by analyze_hedge_intent.py.

Usage:
  python scripts/plot_hedge_pnl_breakdown.py --date 20260421
  python scripts/plot_hedge_pnl_breakdown.py --date 20260421 --last 1h
  python scripts/plot_hedge_pnl_breakdown.py --date 20260421 --from 10:00 --to 11:00
"""
import argparse
import os
import re
import sys
from datetime import timedelta

import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data')


def parse_duration(s: str) -> timedelta:
    m = re.fullmatch(r"(\d+)([smhd])", s.strip().lower())
    if not m:
        raise ValueError(f"Invalid duration '{s}'. Use e.g. 30m, 1h, 2d.")
    n, u = int(m.group(1)), m.group(2)
    return {"s": timedelta(seconds=n), "m": timedelta(minutes=n),
            "h": timedelta(hours=n),  "d": timedelta(days=n)}[u]


def parse_hhmm(date_str: str, hhmm: str) -> pd.Timestamp:
    day = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", hhmm.strip())
    if not m:
        raise ValueError(f"Invalid time '{hhmm}'. Use HH:MM or HH:MM:SS.")
    h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return day + timedelta(hours=h, minutes=mm, seconds=ss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, metavar='YYYYMMDD')
    parser.add_argument('--last', metavar='DURATION',
                        help='Only plot the trailing N units (e.g. 30m, 1h, 6h).')
    parser.add_argument('--from', dest='from_', metavar='HH:MM',
                        help='Start of window on the given date.')
    parser.add_argument('--to', metavar='HH:MM',
                        help='End of window on the given date.')
    args = parser.parse_args()

    if args.last and (args.from_ or args.to):
        parser.error("Use either --last or --from/--to, not both.")
    if bool(args.from_) ^ bool(args.to):
        parser.error("--from and --to must be provided together.")

    csv_path = os.path.join(OUTPUT_DIR, f'hedge_tracking_{args.date}.csv')
    if not os.path.exists(csv_path):
        print(f"Missing input: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'], format='mixed').dt.tz_localize(None)
    df = df.sort_values('time').drop_duplicates('time', keep='last').set_index('time')

    # Apply window filter before computing PnL so cumulative resets at window start.
    window_label = ''
    if args.last:
        delta = parse_duration(args.last)
        end = df.index.max()
        start = end - delta
        df = df[df.index >= start]
        window_label = f" [last {args.last}]"
    elif args.from_ and args.to:
        start = parse_hhmm(args.date, args.from_)
        end = parse_hhmm(args.date, args.to)
        df = df[(df.index >= start) & (df.index < end)]
        window_label = f" [{args.from_}–{args.to}]"

    if df.empty:
        print("No data in requested window.", file=sys.stderr)
        sys.exit(1)

    dP = df['fair_value'].diff().fillna(0.0)
    df['cum_target_pnl'] = (df['expected_pos'].shift().fillna(0.0) * dP).cumsum()
    df['cum_actual_pnl'] = (df['actual_pos'].shift().fillna(0.0)   * dP).cumsum()
    df['unhedged'] = df['actual_pos'] - df['expected_pos']

    df['target_pnl_incr'] = df['expected_pos'].shift().fillna(0.0) * dP
    df['actual_pnl_incr'] = df['actual_pos'].shift().fillna(0.0)   * dP

    # Bucket size scales with window length: ≤2h → 5-min buckets, ≤24h → 1h, else 1h
    span = df.index.max() - df.index.min()
    if span <= timedelta(hours=2):
        rule, bucket_fmt = '5min', '%H:%M'
    else:
        rule, bucket_fmt = '1h', '%H:00'

    buckets = df.resample(rule).agg(
        fv_start=('fair_value', 'first'),
        fv_end=('fair_value', 'last'),
        avg_unhedged=('unhedged', 'mean'),
        target_pnl=('target_pnl_incr', 'sum'),
        actual_pnl=('actual_pnl_incr', 'sum'),
    )
    buckets['miss'] = buckets['actual_pnl'] - buckets['target_pnl']

    total_target = df['cum_target_pnl'].iloc[-1]
    total_actual = df['cum_actual_pnl'].iloc[-1]
    total_miss = total_actual - total_target

    w = 96
    header_label = f" PnL ATTRIBUTION ({rule} buckets){window_label} "
    print()
    print(f"  {header_label:═^{w}}")
    print(f"  {'bucket':<8}{'fv start':>12}{'fv end':>12}"
          f"{'avg unhedged':>16}{'target $':>12}{'actual $':>12}{'miss $':>12}")
    print(f"  {'-'*(w-2)}")
    for ts, row in buckets.iterrows():
        if pd.isna(row['fv_start']):
            continue
        print(f"  {ts.strftime(bucket_fmt):<8}"
              f"{row['fv_start']:>12.6f}{row['fv_end']:>12.6f}"
              f"{row['avg_unhedged']:>16,.0f}"
              f"{row['target_pnl']:>12,.2f}{row['actual_pnl']:>12,.2f}"
              f"{row['miss']:>12,.2f}")
    print(f"  {'-'*(w-2)}")
    print(f"  {'TOTAL':<8}{'':>12}{'':>12}{'':>16}"
          f"{total_target:>12,.2f}{total_actual:>12,.2f}{total_miss:>12,.2f}")
    print(f"  {'═'*(w-2)}")
    print()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                           gridspec_kw={'height_ratios': [3, 2, 1]})

    ax[0].plot(df.index, df['cum_target_pnl'], label='target (perfect hedge)',
               color='tab:green', lw=1.5)
    ax[0].plot(df.index, df['cum_actual_pnl'], label='actual',
               color='tab:red', lw=1.5)
    ax[0].fill_between(df.index, df['cum_target_pnl'], df['cum_actual_pnl'],
                       where=(df['cum_actual_pnl'] < df['cum_target_pnl']),
                       color='tab:red', alpha=0.15, interpolate=True, label='miss (actual < target)')
    ax[0].axhline(0, color='gray', lw=0.5)
    ax[0].set_ylabel('Cumulative hedge PnL ($)')
    ax[0].set_title(f'Hedge PnL breakdown — {args.date}{window_label}  '
                    f'(target ${total_target:+,.0f}  /  actual ${total_actual:+,.0f}  '
                    f'/  miss ${total_miss:+,.0f})')
    ax[0].legend(loc='best')
    ax[0].grid(alpha=0.3)

    ax[1].plot(df.index, df['unhedged'], color='black', lw=0.8)
    ax[1].fill_between(df.index, df['unhedged'], 0,
                       where=(df['unhedged'] < 0),
                       color='tab:red', alpha=0.25, interpolate=True, label='over-short')
    ax[1].fill_between(df.index, df['unhedged'], 0,
                       where=(df['unhedged'] > 0),
                       color='tab:green', alpha=0.25, interpolate=True, label='over-long')
    ax[1].axhline(0, color='gray', lw=0.5)
    ax[1].set_ylabel('Unhedged = actual − expected (tokens)')
    ax[1].legend(loc='best')
    ax[1].grid(alpha=0.3)

    ax[2].plot(df.index, df['fair_value'], color='tab:blue', lw=1.0)
    ax[2].set_ylabel('Fair value')
    ax[2].set_xlabel('Time (UTC)')
    ax[2].grid(alpha=0.3)

    suffix = ''
    if args.last:
        suffix = f'_last{args.last}'
    elif args.from_ and args.to:
        suffix = f"_{args.from_.replace(':', '')}-{args.to.replace(':', '')}"
    plot_path = os.path.join(OUTPUT_DIR, f'hedge_pnl_breakdown_{args.date}{suffix}.png')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print(f"Wrote plot to {plot_path}")


if __name__ == '__main__':
    main()
