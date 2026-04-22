"""
Plot cumulative hedge PnL (target vs actual) alongside unhedged position,
to localize where the hedge miss bleeds money.

Reads strategy_state and hl_fills CSVs directly — no orders data needed.

Inputs:
  data/{market}_vault_strategy_state_YYYYMMDD.csv
  data/hl_fills_YYYYMMDD.csv

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

import numpy as np
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
    parser.add_argument('--market', default='MONUSDC')
    parser.add_argument('--hedge-ratio', type=float, default=1.0,
                        help='Fraction of vault non-rebalanced exposure being hedged.')
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

    ss_path    = os.path.join(OUTPUT_DIR, f'{args.market}_vault_strategy_state_{args.date}.csv')
    fills_path = os.path.join(OUTPUT_DIR, f'hl_fills_{args.date}.csv')
    for p in (ss_path, fills_path):
        if not os.path.exists(p):
            print(f"Missing input: {p}", file=sys.stderr)
            sys.exit(1)

    ss = pd.read_csv(ss_path)
    ss['time'] = pd.to_datetime(ss['time'], format='mixed').dt.tz_localize(None)
    ss = (ss.sort_values('time')
            .drop_duplicates('time', keep='last')
            .set_index('time')
            .dropna(subset=['base_balance', 'quote_balance', 'fair_value']))

    non_rebal_base = (ss['base_balance'] * ss['fair_value'] - ss['quote_balance']) \
                     / (2 * ss['fair_value'])
    expected_pos = -non_rebal_base * args.hedge_ratio

    f = pd.read_csv(fills_path)
    f['time'] = pd.to_datetime(f['time'], format='mixed').dt.tz_localize(None)
    f = f.sort_values('time').reset_index(drop=True)
    f['signed'] = np.where(f['side'] == 'buy', f['size'], -f['size']).astype(float)

    # HL's start_pos on the first fill is the position BEFORE that fill, i.e.
    # the position the bot opened the day with. Initializing with it fixes the
    # mark-to-market PnL and the unhedged diagnostic (otherwise a carried-over
    # short or long is invisible to cumsum-from-zero).
    start_pos = float(f['start_pos'].iloc[0]) if len(f) else 0.0

    actual_cum = f.set_index('time')['signed'].cumsum()
    actual_cum = actual_cum[~actual_cum.index.duplicated(keep='last')].sort_index()
    actual_pos = actual_cum.reindex(ss.index, method='ffill').fillna(0) + start_pos

    fee_cum = f.set_index('time')['fee'].cumsum()
    fee_cum = fee_cum[~fee_cum.index.duplicated(keep='last')].sort_index()
    fee_cum = fee_cum.reindex(ss.index, method='ffill').fillna(0)

    df = pd.DataFrame({
        'fair_value':     ss['fair_value'],
        'expected_pos':   expected_pos,
        'actual_pos':     actual_pos,
        'non_rebal_base': non_rebal_base,
        'cum_fees':       fee_cum,
    })

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

    # Rebase fees to 0 at window start so --last / --from show only
    # fees accrued inside the window.
    df['cum_fees'] = df['cum_fees'] - df['cum_fees'].iloc[0]
    df['fee_incr'] = df['cum_fees'].diff().fillna(0.0)

    # For gross PnL we want the position *going into* each interval; at t=0
    # that's the position at window start (= actual_pos.iloc[0]), not 0.
    dP = df['fair_value'].diff().fillna(0.0)
    pos_prev_expected = df['expected_pos'].shift().fillna(df['expected_pos'].iloc[0])
    pos_prev_actual   = df['actual_pos'].shift().fillna(df['actual_pos'].iloc[0])

    df['target_pnl_incr'] = pos_prev_expected * dP
    df['gross_pnl_incr']  = pos_prev_actual   * dP
    df['net_pnl_incr']    = df['gross_pnl_incr'] - df['fee_incr']

    df['cum_target_pnl'] = df['target_pnl_incr'].cumsum()
    df['cum_gross_pnl']  = df['gross_pnl_incr'].cumsum()
    df['cum_net_pnl']    = df['net_pnl_incr'].cumsum()

    df['unhedged'] = df['actual_pos'] - df['expected_pos']

    span = df.index.max() - df.index.min()
    if span <= timedelta(hours=2):
        rule, bucket_fmt = '5min', '%H:%M'
    else:
        rule, bucket_fmt = '1h', '%H:00'

    buckets = df.resample(rule).agg(
        fv_start=('fair_value', 'first'),
        avg_unhedged=('unhedged', 'mean'),
        target_pnl=('target_pnl_incr', 'sum'),
        gross_pnl=('gross_pnl_incr', 'sum'),
        fees=('fee_incr', 'sum'),
    )
    buckets['net_pnl'] = buckets['gross_pnl'] - buckets['fees']
    buckets['miss'] = buckets['net_pnl'] - buckets['target_pnl']

    total_target = df['cum_target_pnl'].iloc[-1]
    total_gross  = df['cum_gross_pnl'].iloc[-1]
    total_net    = df['cum_net_pnl'].iloc[-1]
    total_fees   = df['cum_fees'].iloc[-1]
    total_miss   = total_net - total_target

    w = 108
    header_label = f" PnL ATTRIBUTION ({rule} buckets){window_label} "
    print()
    print(f"  {header_label:═^{w}}")
    print(f"  {'bucket':<8}{'fv':>10}{'avg unhedged':>16}"
          f"{'target $':>12}{'gross $':>12}{'fees $':>10}{'net $':>12}{'miss $':>12}")
    print(f"  {'-'*(w-2)}")
    for ts, row in buckets.iterrows():
        if pd.isna(row['fv_start']):
            continue
        print(f"  {ts.strftime(bucket_fmt):<8}"
              f"{row['fv_start']:>10.6f}"
              f"{row['avg_unhedged']:>16,.0f}"
              f"{row['target_pnl']:>12,.2f}{row['gross_pnl']:>12,.2f}"
              f"{row['fees']:>10,.2f}{row['net_pnl']:>12,.2f}"
              f"{row['miss']:>12,.2f}")
    print(f"  {'-'*(w-2)}")
    print(f"  {'TOTAL':<8}{'':>10}{'':>16}"
          f"{total_target:>12,.2f}{total_gross:>12,.2f}"
          f"{total_fees:>10,.2f}{total_net:>12,.2f}"
          f"{total_miss:>12,.2f}")
    print(f"  {'═'*(w-2)}")
    print(f"  opening position: {start_pos:+,.0f} tokens @ {df['fair_value'].iloc[0]:.6f}")
    print()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                           gridspec_kw={'height_ratios': [3, 2, 1]})

    ax[0].plot(df.index, df['cum_target_pnl'], label='target (perfect hedge)',
               color='tab:green', lw=1.5)
    ax[0].plot(df.index, df['cum_gross_pnl'], label='gross (∫ pos · dP)',
               color='tab:orange', lw=1.2, alpha=0.9)
    ax[0].plot(df.index, df['cum_net_pnl'], label='net (gross − fees)',
               color='tab:red', lw=1.5)
    ax[0].fill_between(df.index, df['cum_target_pnl'], df['cum_net_pnl'],
                       where=(df['cum_net_pnl'] < df['cum_target_pnl']),
                       color='tab:red', alpha=0.15, interpolate=True, label='miss (net < target)')
    ax[0].axhline(0, color='gray', lw=0.5)
    ax[0].set_ylabel('Cumulative hedge PnL ($)')
    ax[0].set_title(f'Hedge PnL breakdown — {args.date}{window_label}  '
                    f'(target ${total_target:+,.0f}  /  gross ${total_gross:+,.0f}  '
                    f'/  net ${total_net:+,.0f}  /  miss ${total_miss:+,.0f})')
    ax[0].legend(loc='best')
    ax[0].grid(alpha=0.3)

    ax[1].plot(df.index, df['unhedged'], color='black', lw=0.8, label='unhedged (actual − expected)')
    ax[1].plot(df.index, df['non_rebal_base'], color='tab:blue', lw=1.0, alpha=0.9,
               label='vault non-rebal exposure (b·P − q)/(2P)')
    ax[1].plot(df.index, df['actual_pos'], color='tab:orange', lw=1.0, alpha=0.9,
               label='hedger open position (HL)')
    ax[1].fill_between(df.index, df['unhedged'], 0,
                       where=(df['unhedged'] < 0),
                       color='tab:red', alpha=0.25, interpolate=True, label='over-short')
    ax[1].fill_between(df.index, df['unhedged'], 0,
                       where=(df['unhedged'] > 0),
                       color='tab:green', alpha=0.25, interpolate=True, label='over-long')
    ax[1].axhline(0, color='gray', lw=0.5)
    ax[1].set_ylabel('Position (tokens)')
    ax[1].legend(loc='best', fontsize=8)
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
