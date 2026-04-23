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
  python scripts/plot_hedge_pnl_breakdown.py --date 20260421 --show
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


def choose_slope_window(span: timedelta) -> str:
    if span <= timedelta(hours=2):
        return '5min'
    if span <= timedelta(hours=12):
        return '15min'
    if span <= timedelta(days=2):
        return '1h'
    return '4h'


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
    parser.add_argument('--show', action='store_true',
                        help='Open an interactive Matplotlib window with hover tooltips.')
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
    df['gross_minus_target'] = df['cum_gross_pnl'] - df['cum_target_pnl']
    df['net_minus_target'] = df['cum_net_pnl'] - df['cum_target_pnl']
    df['target_minus_gross'] = df['cum_target_pnl'] - df['cum_gross_pnl']
    df['target_minus_net'] = df['cum_target_pnl'] - df['cum_net_pnl']
    df['abs_target_minus_net'] = df['target_minus_net'].abs()

    dt_hours = df.index.to_series().diff().dt.total_seconds().div(3600.0)
    dt_hours = dt_hours.replace(0, np.nan)
    df['target_minus_gross_slope_per_hour'] = df['target_minus_gross'].diff().div(dt_hours)
    df['target_minus_net_slope_per_hour'] = df['target_minus_net'].diff().div(dt_hours)

    slope_window = choose_slope_window(df.index.max() - df.index.min())
    df['target_minus_gross_slope_per_hour_smooth'] = (
        df['target_minus_gross_slope_per_hour']
        .rolling(slope_window, min_periods=2)
        .mean()
    )
    df['target_minus_net_slope_per_hour_smooth'] = (
        df['target_minus_net_slope_per_hour']
        .rolling(slope_window, min_periods=2)
        .mean()
    )

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

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(5, 1, figsize=(15, 14), sharex=True,
                           gridspec_kw={'height_ratios': [3.2, 2.1, 1.8, 1.8, 1.0]})
    axes = list(ax)

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

    ax[2].plot(df.index, df['target_minus_gross'], color='tab:orange', lw=1.1,
               label='target - gross')
    ax[2].plot(df.index, df['target_minus_net'], color='tab:red', lw=1.4,
               label='target - net')
    ax[2].plot(df.index, df['abs_target_minus_net'], color='tab:blue', lw=1.0,
               linestyle='--', label='|target - net|')
    ax[2].fill_between(df.index, df['target_minus_net'], 0,
                       where=(df['target_minus_net'] > 0),
                       color='tab:red', alpha=0.16, interpolate=True, label='underperforming target')
    ax[2].axhline(0, color='gray', lw=0.5)
    ax[2].set_ylabel('Divergence ($)')
    ax[2].legend(loc='best', fontsize=8)
    ax[2].grid(alpha=0.3)

    ax[3].plot(df.index, df['target_minus_gross_slope_per_hour_smooth'],
               color='tab:orange', lw=1.0, alpha=0.9,
               label=f'slope(target - gross) [{slope_window} mean]')
    ax[3].plot(df.index, df['target_minus_net_slope_per_hour_smooth'],
               color='tab:red', lw=1.3,
               label=f'slope(target - net) [{slope_window} mean]')
    ax[3].axhline(0, color='gray', lw=0.5)
    ax[3].set_ylabel('Slope ($/hr)')
    ax[3].legend(loc='best', fontsize=8)
    ax[3].grid(alpha=0.3)

    ax[4].plot(df.index, df['fair_value'], color='tab:blue', lw=1.0)
    ax[4].set_ylabel('Fair value')
    ax[4].set_xlabel('Time (UTC)')
    ax[4].grid(alpha=0.3)

    if args.show:
        x_values = mdates.date2num(df.index.to_pydatetime())
        hover_line_by_ax = {
            axis: axis.axvline(df.index[0], color='0.35', lw=0.8, ls='--', alpha=0.7, visible=False)
            for axis in axes
        }
        hover_note = ax[0].annotate(
            '',
            xy=(0, 0),
            xytext=(12, 12),
            textcoords='offset points',
            bbox={'boxstyle': 'round', 'fc': 'white', 'ec': '0.5', 'alpha': 0.95},
            fontsize=8.5,
        )
        hover_note.set_visible(False)

        def fmt_hover_value(value: float, decimals: int = 2, signed: bool = True, suffix: str = '') -> str:
            if pd.isna(value):
                return 'n/a'
            fmt = f"{{:{'+' if signed else ''},.{decimals}f}}"
            return f"{fmt.format(float(value))}{suffix}"

        def format_hover(idx: int) -> str:
            ts = df.index[idx]
            return '\n'.join([
                ts.strftime('%Y-%m-%d %H:%M:%S'),
                f"target: {fmt_hover_value(df['cum_target_pnl'].iloc[idx])}",
                f"gross:  {fmt_hover_value(df['cum_gross_pnl'].iloc[idx])}",
                f"net:    {fmt_hover_value(df['cum_net_pnl'].iloc[idx])}",
                f"target-gross: {fmt_hover_value(df['target_minus_gross'].iloc[idx])}",
                f"target-net:   {fmt_hover_value(df['target_minus_net'].iloc[idx])}",
                f"|target-net|: {fmt_hover_value(df['abs_target_minus_net'].iloc[idx], signed=False)}",
                f"slope(target-net): {fmt_hover_value(df['target_minus_net_slope_per_hour_smooth'].iloc[idx], suffix=' $/hr')}",
                f"unhedged: {fmt_hover_value(df['unhedged'].iloc[idx], decimals=0, suffix=' tokens')}",
                f"fair value: {df['fair_value'].iloc[idx]:.6f}",
            ])

        def on_move(event):
            if event.inaxes not in axes or event.xdata is None:
                if hover_note.get_visible():
                    hover_note.set_visible(False)
                    for line in hover_line_by_ax.values():
                        line.set_visible(False)
                    fig.canvas.draw_idle()
                return

            idx = np.searchsorted(x_values, event.xdata)
            if idx <= 0:
                nearest = 0
            elif idx >= len(x_values):
                nearest = len(x_values) - 1
            else:
                prev_idx = idx - 1
                nearest = idx if abs(x_values[idx] - event.xdata) < abs(x_values[prev_idx] - event.xdata) else prev_idx

            x = df.index[nearest]
            y = df['cum_net_pnl'].iloc[nearest]
            for line in hover_line_by_ax.values():
                line.set_xdata([x, x])
                line.set_visible(True)

            hover_note.xy = (x, y)
            hover_note.set_text(format_hover(nearest))
            hover_note.set_visible(True)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_move)

    suffix = ''
    if args.last:
        suffix = f'_last{args.last}'
    elif args.from_ and args.to:
        suffix = f"_{args.from_.replace(':', '')}-{args.to.replace(':', '')}"
    plot_path = os.path.join(OUTPUT_DIR, f'hedge_pnl_breakdown_{args.date}{suffix}.png')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print(f"Wrote plot to {plot_path}")
    if args.show:
        print("Opening interactive plot window with hover tooltips...")
        plt.show()
    else:
        print("Run with --show to inspect exact values by hovering in the interactive plot window.")


if __name__ == '__main__':
    main()
