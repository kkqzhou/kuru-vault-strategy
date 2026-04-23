import pandas as pd
import sys

import dotenv
dotenv.load_dotenv()

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.trading import compute_markouts, print_trading_report, print_hl_hedge_report

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data')

def get_vault_markouts(date, market='MONUSDC', report=False, start=None, end=None, hedge_ratio=None):
    date_str = date.strftime('%Y%m%d')
    strategy_state = pd.read_csv(f'{OUTPUT_DIR}/{market}_vault_strategy_state_{date_str}.csv')
    hl_fills = pd.read_csv(f'{OUTPUT_DIR}/hl_fills_{date_str}.csv')
    hl_fills['is_bid'] = hl_fills['side'] == 'buy'
    hl_fills['size_usd'] = hl_fills['price'] * hl_fills['size']
    strategy_state['time'] = pd.to_datetime(strategy_state['time'], format='mixed')
    hl_fills['time'] = pd.to_datetime(hl_fills['time'], format='mixed').dt.tz_localize(None)

    if start is not None and end is not None:
        hl_fills = hl_fills[(hl_fills['time'] >= start) & (hl_fills['time'] < end)]
        strategy_state = strategy_state[(strategy_state['time'] >= start) & (strategy_state['time'] < end)]

    strategy_state_indexed = strategy_state.set_index('time')
    hl_markouts = compute_markouts(hl_fills, strategy_state_indexed['fair_value'])
    if report:
        pnl_df = print_trading_report(
            hl_markouts,
            markout_col='mo_fair_value_5s',
            aux_print_cols=['time', 'fair_value_0s', 'is_bid', 'price', 'size', 'size_usd', 'mo_fair_value_0s', 'mo_fair_value_5s'],
            time_index=pd.to_datetime(hl_fills['time'], format='mixed').dt.tz_localize(None)
        )
        print_hl_hedge_report(hl_markouts, pnl_df, strategy_state_indexed, hedge_ratio=hedge_ratio)
    return hl_markouts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute HL fill markout analysis.')
    parser.add_argument('--date', metavar='YYYYMMDD', help='Run analysis for a full day, e.g. 20250411')
    parser.add_argument('--from', dest='from_ts', metavar='TIMESTAMP', help='Start timestamp, e.g. "2025-04-11 08:00:00"')
    parser.add_argument('--to', dest='to_ts', metavar='TIMESTAMP', help='End timestamp, e.g. "2025-04-11 16:00:00"')
    parser.add_argument('--market', default='MONUSDC', help='Market (default: MONUSDC)')
    parser.add_argument('--hedge-ratio', type=float, default=1.0, metavar='RATIO',
                        help='Fraction of non-rebalanced position being hedged, e.g. 0.05 for 5%% (default: 1.0)')
    args = parser.parse_args()

    if args.date and (args.from_ts or args.to_ts):
        parser.error('Provide either --date or --from/--to, not both.')
    if not args.date and not (args.from_ts and args.to_ts):
        parser.error('Provide either --date or both --from and --to.')

    if args.date:
        start = pd.Timestamp(args.date)
        filter_start, filter_end = None, None
    else:
        start = pd.Timestamp(args.from_ts)
        end = pd.Timestamp(args.to_ts)
        filter_start, filter_end = start, end

    date_str = start.strftime('%Y%m%d')
    mos_fair_value = get_vault_markouts(start, market=args.market, report=True, start=filter_start, end=filter_end, hedge_ratio=args.hedge_ratio)
    print(mos_fair_value['mo_fair_value_5s'].describe())
    mos_fair_value.to_csv(f'{OUTPUT_DIR}/hl_markouts_{date_str}.csv', index=False)
