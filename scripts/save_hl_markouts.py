import pandas as pd
import sys

import dotenv
dotenv.load_dotenv()

import os
sys.path.append('..')
from lib.trading import compute_markouts, print_trading_report

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data')

def get_vault_markouts(date, market='MONUSDC', report=False):
    date_str = date.strftime('%Y%m%d')
    strategy_state = pd.read_csv(f'{OUTPUT_DIR}/{market}_vault_strategy_state_{date_str}.csv')
    hl_fills = pd.read_csv(f'{OUTPUT_DIR}/hl_fills_{date_str}.csv')
    hl_fills['is_bid'] = hl_fills['side'] == 'buy'
    hl_fills['size_usd'] = hl_fills['price'] * hl_fills['size']
    strategy_state['time'] = pd.to_datetime(strategy_state['time'])
    hl_markouts = compute_markouts(hl_fills, strategy_state.set_index('time')['fair_value'])
    if report:
        print_trading_report(
            hl_markouts,
            markout_col='mo_fair_value_5s',
            aux_print_cols=['time', 'fair_value_0s', 'is_bid', 'price', 'size', 'size_usd', 'mo_fair_value_0s', 'mo_fair_value_5s'],
            time_index=pd.to_datetime(hl_fills['time']).dt.tz_localize(None)
        )
    return hl_markouts


if __name__  == '__main__':
    if len(sys.argv) == 1:
        print('Usage: python vault_trading_summary.py <yyyymmdd> [market]')
        print('If market is not provided, it will default to MONUSDC')
        sys.exit(1)
    elif len(sys.argv) == 2:
        date = pd.Timestamp(sys.argv[1])
        date_str = date.strftime('%Y%m%d')
        mos_fair_value = get_vault_markouts(date, market='MONUSDC', report=True)
        mos_fair_value.to_csv(f'{OUTPUT_DIR}/hl_markouts_{date_str}.csv', index=False)
    elif len(sys.argv) == 3:
        date = pd.Timestamp(sys.argv[1])
        date_str = date.strftime('%Y%m%d')
        market = sys.argv[2]
        mos_fair_value = get_vault_markouts(date, market=market)
        print(mos_fair_value['mo_fair_value_5s'].describe())
        mos_fair_value.to_csv(f'{OUTPUT_DIR}/hl_markouts_{date_str}.csv', index=False)
