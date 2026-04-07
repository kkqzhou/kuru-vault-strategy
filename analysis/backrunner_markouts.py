import pandas as pd

import sys
sys.path.append('..')

from lib.trading import compute_markouts, print_trading_report

def backrunner_markouts(date_str, filename, market='MONUSDC'):
    all_trades = pd.read_csv(f'../data/{filename}', index_col=0)
    # columns: ['orderid', 'marketaddress', 'makeraddress', 'isbuy', 'price', 'updatedsize', 'takeraddress', 'filledsize', 'blocknumber', 'txindex', 'logindex', 'transactionhash', 'triggertime', 'monusdprice']
    all_trades['triggertime'] = pd.to_datetime(all_trades['triggertime'], format='mixed').dt.tz_localize(None)
    trades = all_trades[all_trades['triggertime'].dt.floor('1D') == pd.Timestamp(date_str)].rename(columns={'isbuy': 'is_bid', 'triggertime': 'time'})
    trades = trades.drop(columns=['makeraddress', 'marketaddress', 'transactionhash', 'txindex', 'logindex'])
    trades['size_usd'] = trades['price'] * trades['filledsize']

    strategy_state = pd.read_csv(f'../data/{market}_vault_strategy_state_{date_str}.csv')
    strategy_state['time'] = pd.to_datetime(strategy_state['time'], format='mixed')
    strategy_state = strategy_state.set_index('time')
    markouts = compute_markouts(trades, strategy_state['fair_value'])
    print_trading_report(
        markouts, markout_col='mo_fair_value_5s',
        aux_print_cols=['time', 'is_bid', 'price', 'filledsize', 'size_usd', 'monusdprice', 'mo_fair_value_0s', 'mo_fair_value_5s'],
        time_index=trades['time'],
        size_col_name='filledsize',
        fee_col_name=None
    )

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python backrunner_markouts.py <yyyymmdd>')
        sys.exit(1)
    else:
        date_str = sys.argv[1]
    filename = 'backrunner_trades_last_60_days.csv'
    backrunner_markouts(date_str, filename)
