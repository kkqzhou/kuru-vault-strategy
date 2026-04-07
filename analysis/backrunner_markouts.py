import pandas as pd

import sys
sys.path.append('..')

from lib.trading import compute_markouts, print_trading_report

def backrunner_markouts(date_str, filename, timestamp_col, market='MONUSDC'):
    all_trades = pd.read_csv(f'../data/{filename}', index_col=0)
    all_trades[timestamp_col] = pd.to_datetime(all_trades[timestamp_col], format='mixed').dt.tz_localize(None)
    trades = all_trades[all_trades[timestamp_col].dt.floor('1D') == pd.Timestamp(date_str)].rename(columns={'is_buy': 'is_bid', timestamp_col: 'time'})
    trades['price'] = trades['quote_amount'] / trades['base_amount']
    trades['size_usd'] = trades['quote_amount']
    print((trades.size_usd * (trades.is_bid * 2 - 1)).sum())

    strategy_state = pd.read_csv(f'../data/{market}_vault_strategy_state_{date_str}.csv')
    strategy_state['time'] = pd.to_datetime(strategy_state['time'], format='mixed')
    strategy_state = strategy_state.set_index('time')
    markouts = compute_markouts(trades, strategy_state['fair_value'])
    print_trading_report(
        markouts, markout_col='mo_fair_value_5s',
        aux_print_cols=['time', 'is_bid', 'price', 'base_amount', 'size_usd', 'mo_fair_value_0s', 'mo_fair_value_5s'],
        time_index=trades['time'],
        size_col_name='base_amount',
        fee_col_name=None
    )

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python backrunner_markouts.py <yyyymmdd>')
        sys.exit(1)
    else:
        date_str = sys.argv[1]
    filename = 'backrunner_swaps_last_30_days.csv'
    timestamp_col = 'blocktimestamp'
    backrunner_markouts(date_str, filename, timestamp_col)
